"""ASTA localization for the office-rack sionna track datasets.

Trains ASTA encoder (from scratch) + per-timestep (x, y) regression
head end-to-end on the per-anchor ToA + AoA (azimuth) channels.
No SSL, no MAML/meta, no tier1/tier2 — just ASTA encoder →
per-timestep regression. Same architecture shape as
scripts/train_asta_industry1_23anchor.py, but with:

  * 23 anchors (A1..A23) instead of 0x-hex addresses
  * 2 input channels: ToA (measured_dist_m) + AoA (aoa_azimuth_deg)
  * Ground truth from tag_x, tag_y (already in meters, 2D — z
    is constant at 1.0 across all three scenes and is dropped)
  * Three scenes: 'big' / 'middle' / 'right'. Default train = big,
    val = right. Middle is unused by default but selectable.
  * Per-channel z-score normalization (ToA and AoA normalized
    independently, each on training-only stats)

This script generalizes the industry1 23-anchor trainer to a
different schema. Diffs vs that script:
  - column names (no hex, A1..A23)
  - mm→m removed (already in meters)
  - extra AoA channel (F=2 instead of F=1)
  - 3-scene selection via --train_scene / --val_scene instead of
    a single CSV path

Output:
  - ckpt:     checkpoints/_tmp_asta_office_rack_<train>_to_<val>/{encoder,head}.pt
  - CSV:      .../eval_per_step.csv
  - Figures:  .../{val,train}_gt_vs_pred.png, val_per_window_mae.png,
              val_err_hist.png, training_curves.png
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

N_ANCHORS = 23           # A1..A23
N_CHANNELS = 2           # (ToA, AoA_azimuth)
CHANNEL_NAMES = ("toa", "aoa_az")
TOA_CAP_M = 50.0         # clip ToA to [0, 50] m (matches the rest of the repo)
AOA_CAP_DEG = 180.0      # clip AoA to [-180, 180] deg (already in that range)

DEFAULT_DATA_ROOT = Path(
    "/Users/justin/ultraclear-paper/code/data/office_rack_track_sionna"
)
SCENES = ("big", "middle", "right")


def load_office_rack_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a sionna_uwb_track_office_rack.csv.

    Returns:
        X:  (N, M=23, F=2) float32 — (ToA_m, AoA_azimuth_deg)
                                     NaN cells filled with 0.
        Y:  (N, 2)         float32 — (tag_x, tag_y) in meters

    The input is NaN-safe: any cell that is missing (rare in this
    dataset) is zero-filled. Normalization is the caller's job.
    """
    # Read just the columns we need to keep the load cheap
    want = ["sample_id",
            "tag_x", "tag_y"] + \
           [f"A{i}_{c}" for i in range(1, N_ANCHORS + 1)
            for c in ("measured_dist_m", "aoa_azimuth_deg")]
    df = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None,
                       encoding="utf-8")

    n_rows = len(df)
    X = np.zeros((n_rows, N_ANCHORS, N_CHANNELS), dtype=np.float32)
    Y = np.zeros((n_rows, 2), dtype=np.float32)

    Y[:, 0] = df["tag_x"].astype(np.float32)
    Y[:, 1] = df["tag_y"].astype(np.float32)

    for m in range(N_ANCHORS):
        a = m + 1
        toa = df[f"A{a}_measured_dist_m"].astype(np.float32)
        aoa = df[f"A{a}_aoa_azimuth_deg"].astype(np.float32)
        # NaN → 0 (matching the industry1 trainer's convention)
        toa = np.where(np.isnan(toa), 0.0, toa)
        aoa = np.where(np.isnan(aoa), 0.0, aoa)
        # Clip to reasonable physical ranges
        toa = np.clip(toa, 0.0, TOA_CAP_M)
        aoa = np.clip(aoa, -AOA_CAP_DEG, AOA_CAP_DEG)
        X[:, m, 0] = toa
        X[:, m, 1] = aoa

    return X, Y


def make_windows(X: np.ndarray, Y: np.ndarray, T: int, stride: int = 1):
    """Strided sliding windows over the sequence.

    With stride=1 (default) we get a *rotation window* —
    every (t, t+1, ..., t+T-1) slice, so a length-N series gives
    N-T+1 windows. Drops any window whose ToA channel is
    all-zero (i.e. completely missing data).
    """
    N = len(X)
    if N < T:
        return np.zeros((0, T, X.shape[1], X.shape[2]), dtype=np.float32), \
               np.zeros((0, T, 2), dtype=np.float32)
    starts = list(range(0, N - T + 1, stride))
    wx = np.stack([X[s:s + T] for s in starts])   # (W, T, M, F)
    wy = np.stack([Y[s:s + T] for s in starts])   # (W, T, 2)
    # Drop windows whose ToA channel is all zero (no signal at all)
    keep = (wx[..., 0].sum(axis=(1, 2)) > 0)
    return wx[keep], wy[keep]


# --------------------------------------------------------------------------- #
# Model: ASTA encoder (from scratch, trainable) + per-timestep regression head
# --------------------------------------------------------------------------- #

class PerTimestepLocHead(nn.Module):
    """Map (B, T, M, d) → (B, T, 2) by flattening M anchor features per t
    and running a small MLP. Mirrors scripts/train_asta_industry1_23anchor.py."""
    def __init__(self, d_model: int, n_anchors: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model * n_anchors, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, M, D) → (B, T, M*D) → (B, T, 2)
        B, T, M, D = h.shape
        return self.head(h.reshape(B, T, M * D))


class ASTAOfficeRack(nn.Module):
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
                   train_csv: Path,
                   val_csv: Path,
                   train_scene: str,
                   val_scene: str,
                   T: int = 32, stride: int = 1, n_epochs: int = 200,
                   lr: float = 1e-3,
                   device: str = "cpu",
                   out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    X_tr_raw, Y_tr = load_office_rack_csv(train_csv)
    X_va_raw, Y_va = load_office_rack_csv(val_csv)
    print(f"[data] train ({train_scene}): X={X_tr_raw.shape}  Y={Y_tr.shape}")
    print(f"[data] val   ({val_scene}):   X={X_va_raw.shape}  Y={Y_va.shape}")

    # ---- Build windows (per-scene, then concat) ----
    wx_tr, wy_tr = make_windows(X_tr_raw, Y_tr, T, stride)
    wx_va, wy_va = make_windows(X_va_raw, Y_va, T, stride)
    print(f"[data] windows: train={wx_tr.shape} → {wy_tr.shape},  "
          f"val={wx_va.shape} → {wy_va.shape}  "
          f"(T={T}, stride={stride} — "
          f"{'rotation' if stride == 1 else f'strided by {stride}'})")
    n_tr, n_va = len(wx_tr), len(wx_va)

    # ---- Per-channel z-score normalization (computed on TRAIN only,
    #      ignoring the 0-fill cells that came from NaN) ----
    norm_mean = np.zeros(N_CHANNELS, dtype=np.float32)
    norm_std = np.ones(N_CHANNELS, dtype=np.float32)
    for c in range(N_CHANNELS):
        vals = wx_tr[..., c]
        nonzero = vals[vals > 0]   # ignore the 0-fill cells
        if nonzero.size == 0:
            continue
        norm_mean[c] = float(nonzero.mean())
        norm_std[c] = float(nonzero.std() + 1e-6)
    print(f"[norm]  per-channel mean = {norm_mean}")
    print(f"[norm]  per-channel std  = {norm_std}")

    def norm(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for c in range(N_CHANNELS):
            ch = out[..., c]
            m = ch > 0
            ch[m] = (ch[m] - norm_mean[c]) / norm_std[c]
            out[..., c] = ch
        return out

    wx_tr_n = norm(wx_tr)
    wx_va_n = norm(wx_va)

    # ---- To tensors ----
    wx_tr_t = torch.from_numpy(wx_tr_n).to(device)   # (n_tr, T, M, F)
    wy_tr_t = torch.from_numpy(wy_tr).to(device)
    wx_va_t = torch.from_numpy(wx_va_n).to(device)
    wy_va_t = torch.from_numpy(wy_va).to(device)

    # ---- Build model ----
    cfg = Config(n_anchors=N_ANCHORS, n_channels=N_CHANNELS,
                 d_model=64, n_asta_blocks=2)
    print(f"[model] n_anchors={N_ANCHORS}  n_channels={N_CHANNELS}  "
          f"d_model=64  n_asta_blocks=2  seq_len={T}")
    model = ASTAOfficeRack(cfg).to(device)
    n_params_enc = sum(p.numel() for p in model.encoder.parameters())
    n_params_head = sum(p.numel() for p in model.head.parameters())
    print(f"[model] encoder params: {n_params_enc:,}   "
          f"head params: {n_params_head:,}   "
          f"total: {n_params_enc + n_params_head:,}")

    # ---- Train (mini-batch) ----
    batch_size = 16
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    rng = np.random.default_rng(0)
    history = []
    for ep in range(1, n_epochs + 1):
        model.train()
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
                tr_mae = 0.0
                for i in range(0, n_tr, batch_size):
                    xb = wx_tr_t[i:i + batch_size]
                    yb = wy_tr_t[i:i + batch_size]
                    tr_mae += (model(xb) - yb).norm(dim=-1).sum().item()
                tr_mae /= (n_tr * T)

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
        pred_va = model(wx_va_t).cpu().numpy()
        pred_tr_full = model(wx_tr_t).cpu().numpy()
    true_va = wy_va
    true_tr_full = wy_tr

    err_va = np.linalg.norm(pred_va - true_va, axis=-1)
    err_tr = np.linalg.norm(pred_tr_full - true_tr_full, axis=-1)

    per_win_va = err_va.mean(axis=1)
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
        "config": {
            "n_anchors": N_ANCHORS, "n_channels": N_CHANNELS,
            "d_model": 64, "n_asta_blocks": 2, "T": T, "stride": stride,
            "toa_cap_m": TOA_CAP_M, "aoa_cap_deg": AOA_CAP_DEG,
            "channel_names": list(CHANNEL_NAMES),
            "anchor_labels": [f"A{i}" for i in range(1, N_ANCHORS + 1)],
        },
        "train_scene": train_scene,
        "val_scene": val_scene,
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

    # ---- Figure 1: GT vs predicted scatter (val) ----
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.scatter(true_va[..., 0].flatten(), true_va[..., 1].flatten(),
               c='green', s=8, alpha=0.5, label='GT')
    ax.scatter(pred_va[..., 0].flatten(), pred_va[..., 1].flatten(),
               c='red', s=8, alpha=0.5, label='Pred')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'ASTA 23-anchor ToA+AoA — val ({val_scene})\n'
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
    ax.set_title(f"ASTA 23-anchor per-window MAE "
                 f"(train={train_scene}, val={val_scene})\n"
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
    ax.set_title(f"Per-step error distribution (val, {val_scene})")
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
    ax.set_title(f'Training curves (ASTA 23-anchor ToA+AoA, '
                 f'train={train_scene}, val={val_scene})')
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
        "input_channels": list(CHANNEL_NAMES),
        "norm_mean": norm_mean.tolist(),
        "norm_std": norm_std.tolist(),
        "train_mae_m": float(overall_tr_mae),
        "val_mae_m": float(overall_va_mae),
        "val_mae_p50_m": float(np.median(err_va)),
        "val_mae_p95_m": float(np.percentile(err_va, 95)),
        "val_mae_max_m": float(err_va.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT,
                   help="parent dir containing office_rack_track_<scene>/")
    p.add_argument("--train_scene", choices=SCENES, default="big",
                   help="scene to train on (default: big)")
    p.add_argument("--val_scene", choices=SCENES, default="right",
                   help="scene to validate on (default: right)")
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--stride", type=int, default=1,
                   help="window stride (default 1 = rotation/sliding window)")
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path,
                   default=None,
                   help="output dir (default: "
                        "code/checkpoints/_tmp_asta_office_rack_<train>_to_<val>)")
    args = p.parse_args()

    train_csv = args.data_root / f"office_rack_track_{args.train_scene}" / \
                "sionna_uwb_track_office_rack.csv"
    val_csv = args.data_root / f"office_rack_track_{args.val_scene}" / \
              "sionna_uwb_track_office_rack.csv"
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    if not val_csv.exists():
        raise FileNotFoundError(val_csv)
    if args.out is None:
        args.out = (Path("/Users/justin/ultraclear-paper/code/checkpoints")
                    / f"_tmp_asta_office_rack_{args.train_scene}_to_{args.val_scene}")
    out_dir = args.out

    print("=" * 70)
    print("ASTA 23-anchor ToA+AoA on office_rack sionna track datasets")
    print("  no SSL, no MAML/meta, no tier1/tier2")
    print(f"  train scene: {args.train_scene}  ({train_csv.name})")
    print(f"  val scene:   {args.val_scene}  ({val_csv.name})")
    print(f"  T={args.T}  stride={args.stride}  "
          f"epochs={args.n_epochs}  lr={args.lr}")
    print(f"  out:         {out_dir}")
    print("=" * 70)
    print()

    summary = train_and_eval(
        train_csv=train_csv,
        val_csv=val_csv,
        train_scene=args.train_scene,
        val_scene=args.val_scene,
        T=args.T, stride=args.stride,
        n_epochs=args.n_epochs, lr=args.lr,
        device=args.device, out_dir=out_dir,
    )
    print()
    print("summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
