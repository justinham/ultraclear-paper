"""ASTA localization for industry1 with 23 anchors (TWR datasets).

Trains ASTA encoder (from scratch) + per-timestep location head end-to-end
on the pivoted_superframe_distances CSVs we generated. No SSL, no
MAML/meta-learning, no tier1/tier2, no multi-header — just ASTA encoder
→ per-timestep (x, y) regression.

Key design points:
  - n_anchors = 23, n_channels = 1 (distance-only, no AoA)
  - Window: T = 32 superframes
  - NaN distances → 0 (per user spec)
  - Per-feature z-score normalization computed from training set only
  - Train: industry1 _rh pivoted CSV
  - Val:   industry1 _sl pivoted CSV (held-out capture)
  - Same encoder/head shapes as scripts/train_asta_persample_loc.py
    but with M=23, F=1, no AoA channel.

Output:
  - ckpt:      checkpoints/_tmp_asta_industry1/head.pt (and encoder.pt)
  - eval CSV:  figures/asta_industry1_eval.csv
  - figures:   per-window MAE bar, per-step err hist, GT-vs-pred scatter
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.asta import ASTAEncoder
from ultraclear.config import Config


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_pivoted_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read a pivoted superframe CSV.

    Returns:
        X:  (N, M=23) float32 — distances in METERS, NaN filled with 0
        Y:  (N, 2)    float32 — (tag_x, tag_y) in meters
        anchor_hex: list of the 23 anchor hex ids, in column order

    Note: distances in the source CSVs are in mm.  We convert to
    meters here so the model and the GT are in the same units.
    Input is ToA only — there is no AoA channel in the industry1
    pivoted CSVs, so n_channels in the model is fixed to 1.
    """
    with open(csv_path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = list(rdr)

    anchor_cols = [i for i, h in enumerate(header) if h.startswith("dist_")]
    gx = header.index("gt_x")
    gy = header.index("gt_y")
    anchor_hex = [header[i].removeprefix("dist_") for i in anchor_cols]

    X = np.zeros((len(rows), len(anchor_cols)), dtype=np.float32)
    Y = np.zeros((len(rows), 2), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(anchor_cols):
            v = r[c].strip()
            # mm → m; NaN → 0
            X[i, j] = 0.0 if v == "" else float(v) / 1000.0
        Y[i, 0] = float(r[gx])
        Y[i, 1] = float(r[gy])
    return X, Y, anchor_hex


def make_windows(X: np.ndarray, Y: np.ndarray, T: int, stride: int = 1):
    """Strided windows. With stride=1 (default) we use a *rotation
    window* — every (t, t+1, ..., t+T-1) slice of the sequence, so
    a length-N series gives N-T+1 windows. Drops any window where
    all 23*T distances are 0 (i.e. completely missing data)."""
    N = len(X)
    if N < T:
        return np.zeros((0, T, X.shape[1]), dtype=np.float32), \
               np.zeros((0, T, 2), dtype=np.float32)
    starts = list(range(0, N - T + 1, stride))
    wx = np.stack([X[s:s + T] for s in starts])   # (W, T, M)
    wy = np.stack([Y[s:s + T] for s in starts])   # (W, T, 2)
    # Drop windows with no data
    keep = (wx.sum(axis=(1, 2)) > 0)
    return wx[keep], wy[keep]


# --------------------------------------------------------------------------- #
# Model: ASTA encoder (from scratch, trainable) + per-timestep regression head
# --------------------------------------------------------------------------- #

class PerTimestepLocHead(nn.Module):
    """Map (B, T, M, d) → (B, T, 2) by flattening M anchor features per t
    and running a small MLP."""
    def __init__(self, d_model: int, n_anchors: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model * n_anchors, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, M, d) → (B, T, M*d) → (B, T, 2)
        B, T, M, D = h.shape
        return self.head(h.reshape(B, T, M * D))


class ASTAIndustry1(nn.Module):
    """ASTA encoder (trainable from scratch) + per-timestep head."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = ASTAEncoder(cfg)
        self.head = PerTimestepLocHead(cfg.d_model, cfg.n_anchors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)              # (B, T, M, d)
        return self.head(h)              # (B, T, 2)


# --------------------------------------------------------------------------- #
# Train + eval
# --------------------------------------------------------------------------- #

def train_and_eval(*,
                   csv_path: Path,
                   T: int = 32, stride: int = 8, n_epochs: int = 200,
                   lr: float = 1e-3, val_frac: float = 0.2,
                   device: str = "cpu",
                   out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    X_all, Y_all, anchor_hex = load_pivoted_csv(csv_path)
    print(f"[data] {csv_path.name}: X={X_all.shape}  Y={Y_all.shape}  "
          f"anchors={len(anchor_hex)}")

    # ---- Build windows on the full data, THEN split by window
    #      (so train/val don't overlap).  With stride=1 (default)
    #      we get a rotation/sliding window: every (t, t+1, ...,
    #      t+T-1) slice of the N-row sequence, giving N-T+1
    #      overlapping windows. ----
    wx_all, wy_all = make_windows(X_all, Y_all, T, stride)
    print(f"[data] windows: {wx_all.shape} → {wy_all.shape}  "
          f"(T={T}, stride={stride} — "
          f"{'rotation' if stride == 1 else f'strided by {stride}'})")

    # Chronological 80/20 split by window index.  With rotation
    # windows the train/val windows can overlap by at most T-1
    # superframes at the split boundary — that's the
    # standard sliding-window CV caveat.
    n_total = len(wx_all)
    n_val = max(1, int(round(val_frac * n_total)))
    n_tr = n_total - n_val
    wx_tr, wy_tr = wx_all[:n_tr], wy_all[:n_tr]
    wx_va, wy_va = wx_all[n_tr:], wy_all[n_tr:]
    print(f"[split] {n_tr} train windows  |  {n_val} val windows  "
          f"(val_frac={val_frac})")

    # ---- z-score normalization (computed on TRAIN distances only,
    #      ignoring the 0-fill cells that came from NaN) ----
    train_vals = wx_tr[wx_tr > 0]   # ignore the 0-fill cells
    if train_vals.size == 0:
        raise RuntimeError("no positive distance values in train — check data")
    norm_mean = float(train_vals.mean())
    norm_std = float(train_vals.std() + 1e-6)
    print(f"[norm]  mean={norm_mean:.3f} m   std={norm_std:.3f} m   "
          f"(ToA only, n_channels=1)")

    def norm(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        m = out > 0
        out[m] = (out[m] - norm_mean) / norm_std
        return out

    wx_tr_n = norm(wx_tr)
    wx_va_n = norm(wx_va)

    # ---- To tensors: input needs F=1 channel → expand last dim ----
    wx_tr_t = torch.from_numpy(wx_tr_n).to(device).unsqueeze(-1)  # (W, T, M, 1)
    wy_tr_t = torch.from_numpy(wy_tr).to(device)                   # (W, T, 2)
    wx_va_t = torch.from_numpy(wx_va_n).to(device).unsqueeze(-1)  # (W, T, M, 1)
    wy_va_t = torch.from_numpy(wy_va).to(device)

    # ---- Build model ----
    cfg = Config(n_anchors=23, n_channels=1, d_model=64, n_asta_blocks=2)
    print(f"[model] n_anchors=23  n_channels=1  d_model=64  "
          f"n_asta_blocks=2  seq_len={T}")
    model = ASTAIndustry1(cfg).to(device)
    n_params_enc = sum(p.numel() for p in model.encoder.parameters())
    n_params_head = sum(p.numel() for p in model.head.parameters())
    print(f"[model] encoder params: {n_params_enc:,}   "
          f"head params: {n_params_head:,}   "
          f"total: {n_params_enc + n_params_head:,}")

    # ---- Train (mini-batch) ----
    batch_size = 16
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    n_tr, n_va = len(wx_tr_t), len(wx_va_t)
    rng = np.random.default_rng(0)
    history = []
    for ep in range(1, n_epochs + 1):
        model.train()
        # Shuffle train windows each epoch
        perm = rng.permutation(n_tr)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            xb = wx_tr_t[idx]
            yb = wy_tr_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        ep_loss /= max(n_batches, 1)

        if ep == 1 or ep % 20 == 0 or ep == n_epochs:
            model.eval()
            with torch.no_grad():
                # Train MAE: averaged across mini-batches
                tr_mae = 0.0
                for i in range(0, n_tr, batch_size):
                    xb = wx_tr_t[i:i + batch_size]
                    yb = wy_tr_t[i:i + batch_size]
                    tr_mae += (model(xb) - yb).norm(dim=-1).sum().item()
                tr_mae /= (n_tr * T)
                # Val MAE: batched too
                pred_va_list = []
                for i in range(0, n_va, batch_size):
                    pred_va_list.append(model(wx_va_t[i:i + batch_size]))
                pred_va = torch.cat(pred_va_list, dim=0)
                va_mae = (pred_va - wy_va_t).norm(dim=-1).mean().item()
            history.append((ep, ep_loss, tr_mae, va_mae))
            print(f"  ep {ep:3d}  train_mse={ep_loss:.4f}  "
                  f"train_mae={tr_mae:.3f}m  val_mae={va_mae:.3f}m")

    # ---- Final eval ----
    model.eval()
    with torch.no_grad():
        pred_va = model(wx_va_t).cpu().numpy()          # (n_va, T, 2)
        pred_tr_full = model(wx_tr_t).cpu().numpy()     # for train side too
    true_va = wy_va                                      # (n_va, T, 2)
    true_tr_full = wy_tr

    err_va = np.linalg.norm(pred_va - true_va, axis=-1)             # (n_va, T)
    err_tr = np.linalg.norm(pred_tr_full - true_tr_full, axis=-1)   # (n_tr, T)

    per_win_va = err_va.mean(axis=1)        # (n_va,)
    per_win_tr = err_tr.mean(axis=1)
    overall_va_mae = float(err_va.mean())
    overall_tr_mae = float(err_tr.mean())
    print()
    print(f"[result] train per-step MAE: {overall_tr_mae:.3f} m  "
          f"(n={n_tr} windows)")
    print(f"[result] val   per-step MAE: {overall_va_mae:.3f} m  "
          f"(n={n_va} windows)")

    # ---- Save ckpts ----
    enc_path = out_dir / "encoder.pt"
    head_path = out_dir / "head.pt"
    torch.save({
        "model_state": model.state_dict(),
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "anchor_hex": anchor_hex,
        "config": {"n_anchors": 23, "n_channels": 1, "d_model": 64,
                   "n_asta_blocks": 2, "T": T, "stride": stride},
    }, enc_path)
    torch.save(model.head.state_dict(), head_path)
    print(f"[save]  {enc_path}   {head_path}")

    # ---- Save eval CSV (per-step) ----
    eval_csv = out_dir / "eval_per_step.csv"
    with open(eval_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "window_idx", "t", "x_pred", "y_pred",
                    "x_true", "y_true", "err_m"])
        for wi in range(pred_va.shape[0]):
            for t in range(pred_va.shape[1]):
                w.writerow(["val", wi, t,
                            f"{pred_va[wi, t, 0]:.4f}",
                            f"{pred_va[wi, t, 1]:.4f}",
                            f"{true_va[wi, t, 0]:.4f}",
                            f"{true_va[wi, t, 1]:.4f}",
                            f"{err_va[wi, t]:.4f}"])
        for wi in range(pred_tr_full.shape[0]):
            for t in range(pred_tr_full.shape[1]):
                w.writerow(["train", wi, t,
                            f"{pred_tr_full[wi, t, 0]:.4f}",
                            f"{pred_tr_full[wi, t, 1]:.4f}",
                            f"{true_tr_full[wi, t, 0]:.4f}",
                            f"{true_tr_full[wi, t, 1]:.4f}",
                            f"{err_tr[wi, t]:.4f}"])
    print(f"[save]  {eval_csv}")

    # ---- Save training history ----
    hist_csv = out_dir / "history.csv"
    with open(hist_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "train_mae_m", "val_mae_m"])
        for ep, mse, tr, va in history:
            w.writerow([ep, f"{mse:.6f}", f"{tr:.4f}", f"{va:.4f}"])
    print(f"[save]  {hist_csv}")

    # ---- Figure 1: GT vs predicted scatter (val set, dots only) ----
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    # All per-step points (no connecting lines)
    ax.scatter(true_va[..., 0].flatten(), true_va[..., 1].flatten(),
               c='green', s=8, alpha=0.5, label='GT')
    ax.scatter(pred_va[..., 0].flatten(), pred_va[..., 1].flatten(),
               c='red', s=8, alpha=0.5, label='Pred')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'ASTA 23-anchor — val (chronological 80/20)\n'
                 f'val MAE = {overall_va_mae:.3f} m, '
                 f'n_val_windows = {n_va}, T = {T}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    scatter_path = out_dir / "val_gt_vs_pred.png"
    fig.savefig(scatter_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {scatter_path}")

    # ---- Figure 2: per-window MAE bar chart (val) ----
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar(np.arange(len(per_win_va)), per_win_va, color="tab:blue", alpha=0.7)
    ax.axhline(per_win_va.mean(), color='black', linestyle='--',
               label=f"val mean = {per_win_va.mean():.3f} m")
    ax.set_xlabel("validation window index (chronological)")
    ax.set_ylabel("per-window MAE (m)")
    ax.set_title(f"ASTA 23-anchor per-window MAE (val = industry1 _sl)\n"
                 f"T={T}, stride={stride}, n_train={n_tr}, n_val={n_va}")
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    bar_path = out_dir / "val_per_window_mae.png"
    fig.savefig(bar_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {bar_path}")

    # ---- Figure 3: per-step error histogram (val) ----
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.hist(err_va.flatten(), bins=40, color="tab:blue", edgecolor="black",
            alpha=0.75)
    ax.axvline(overall_va_mae, color='black', linestyle='--',
               label=f"mean = {overall_va_mae:.3f} m")
    p50 = float(np.median(err_va))
    p95 = float(np.percentile(err_va, 95))
    ax.axvline(p50, color='orange', linestyle='--',
               label=f"p50 = {p50:.3f} m")
    ax.axvline(p95, color='red', linestyle='--',
               label=f"p95 = {p95:.3f} m")
    ax.set_xlabel("per-step Euclidean error (m)")
    ax.set_ylabel("count")
    ax.set_title("Per-step error distribution (val, industry1 _sl)")
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    hist_path = out_dir / "val_err_hist.png"
    fig.savefig(hist_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {hist_path}")

    # ---- Figure 4: training curves ----
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    eps = [h[0] for h in history]
    tr_mae = [h[2] for h in history]
    va_mae = [h[3] for h in history]
    ax.plot(eps, tr_mae, '-', color='tab:blue', label='train MAE (m)')
    ax.plot(eps, va_mae, '-', color='tab:orange', label='val MAE (m)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('per-step MAE (m)')
    ax.set_title('Training curves (ASTA 23-anchor, industry1)')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    curves_path = out_dir / "training_curves.png"
    fig.savefig(curves_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {curves_path}")

    return {
        "n_train": int(n_tr),
        "n_val": int(n_va),
        "T": int(T),
        "stride": int(stride),
        "n_epochs": int(n_epochs),
        "lr": float(lr),
        "input_units": "meters (ToA only, n_channels=1)",
        "norm_mean_m": float(norm_mean),
        "norm_std_m": float(norm_std),
        "train_mae_m": float(overall_tr_mae),
        "val_mae_m": float(overall_va_mae),
        "val_mae_p50_m": float(np.median(err_va)),
        "val_mae_p95_m": float(np.percentile(err_va, 95)),
        "val_mae_max_m": float(err_va.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default="/Users/justin/ultraclear-paper/figure/anc20/"
                           "pivoted_superframe_distances.csv",
                   help="single pivoted CSV — all data; "
                        "chronological 80/20 train/val split is applied")
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--stride", type=int, default=1,
                   help="window stride (default 1 = rotation/sliding window, "
                        "every consecutive slice)")
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="fraction of windows held out for val "
                        "(chronological split)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out",
                   default="/Users/justin/ultraclear-paper/code/"
                           "checkpoints/_tmp_asta_industry1_23anchor")
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    print("=" * 70)
    print("ASTA 23-anchor localization on industry1 (TWR)")
    print("  no SSL, no MAML/meta, no tier1/tier2, no multi-header")
    print("  input: ToA distances in meters only (n_channels=1)")
    print(f"  data:  {csv_path}  (single file, chronological 80/20 split)")
    print(f"  T={args.T}  stride={args.stride}  "
          f"epochs={args.n_epochs}  lr={args.lr}  val_frac={args.val_frac}")
    print(f"  out:   {out_dir}")
    print("=" * 70)
    print()

    summary = train_and_eval(
        csv_path=csv_path,
        T=args.T, stride=args.stride,
        n_epochs=args.n_epochs, lr=args.lr,
        val_frac=args.val_frac,
        device=args.device, out_dir=out_dir,
    )
    print()
    print("summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
