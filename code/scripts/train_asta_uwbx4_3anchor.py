"""Train a stripped-down ASTA model on uwbx4_aligned_2hz.csv.

This is a clean baseline experiment:
  - Model: ASTAEncoder (1 channel ToA-only, 3 anchors) + 2-layer MLP
    location head that predicts (tx_car, ty_car). No SSL, no FOMAML,
    no FiLM, no adversary.
  - Data: uwbx4_aligned_2hz.csv (45 valid points at 2 Hz).
  - Labels: (px_car, py_car) — 2D horizontal location in car frame.

Saved weights go to a tmp directory so we never touch the
canonical checkpoints (last_model_toa_aoa.pth or anything in
code/checkpoints/).

Usage:
    python3.11 -m scripts.train_asta_uwbx4_3anchor
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.config import Config
from ultraclear.asta import ASTAEncoder


# ---- Data ------------------------------------------------------------------

def load_uwbx4_aligned(csv_path: Path) -> dict:
    """Read the aligned CSV; return per-row arrays.

    Uses 3 working anchors (d1, d2, d3) and labels
    (px_car, py_car) — phone position in car frame.
    Drops rows where any of those 5 fields is NaN.
    """
    d_cols = ["d1", "d2", "d3"]
    y_cols = ["px_car", "py_car"]
    times, toa, y = [], [], []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                ds = [float(r[c]) for c in d_cols]
                ys = [float(r[c]) for c in y_cols]
            except (ValueError, KeyError):
                continue
            if any(np.isnan(v) for v in ds + ys):
                continue
            times.append(r["t_grid"])
            toa.append(ds)
            y.append(ys)
    return {
        "times": times,
        "toa": np.asarray(toa, dtype=np.float32),     # (N, 3)
        "y": np.asarray(y, dtype=np.float32),         # (N, 2)
    }


def make_windows(toa: np.ndarray, y: np.ndarray, T: int, stride: int):
    """Non-overlapping (or strided) windows. Each window is (T, 3) ToA;
    the label is the y at the LAST row of the window (matches the eval
    script's convention of using the last timestep's GT for a window)."""
    N = len(toa)
    idx = list(range(0, N - T + 1, stride))
    X = np.stack([toa[i:i + T] for i in idx])        # (W, T, 3)
    Y = np.stack([y[i + T - 1] for i in idx])         # (W, 2)
    return X, Y, idx


def normalize_toa(toa: np.ndarray, toa_cap: float = 10.0) -> np.ndarray:
    """Clip to [0,1] using the same toa_cap=10 convention as the eval
    script. No train/val split here — the cap is a fixed prior, not a
    learned statistic."""
    return np.clip(toa / toa_cap, 0.0, 1.0).astype(np.float32)


def time_ordered_split(n: int, val_frac: float = 0.2):
    """Time-ordered 80/20 split. With small n this returns 0 or 1
    val samples; the caller should handle that."""
    if val_frac <= 0.0:
        # No split: everything is training data. Val is the same
        # windows, so the reported "val MAE" is just the in-sample
        # residual error. Caller is responsible for knowing this.
        return list(range(n)), []
    n_val = max(1, int(round(n * val_frac)))
    n_val = min(n_val, max(1, n // 2))   # always keep at least 1 train
    n_train = max(1, n - n_val)          # always keep at least 1 train
    n_val = n - n_train
    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n))
    return train_idx, val_idx


# ---- Model -----------------------------------------------------------------

class ASTALocator(nn.Module):
    """ASTAEncoder + 2-layer MLP location head. No SSL, no MAML."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = ASTAEncoder(cfg)               # outputs (B, T, M, d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model * cfg.n_anchors, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, toa: torch.Tensor) -> torch.Tensor:
        """toa: (B, T, M, F=1) ToA only."""
        h = self.encoder(toa)                         # (B, T, M, d)
        # Mean-pool over time, flatten over anchors
        h_pool = h.mean(dim=1)                        # (B, M, d)
        h_flat = h_pool.reshape(h.size(0), -1)        # (B, M*d)
        return self.head(h_flat)                      # (B, 2)


# ---- LSE baseline (closed-form 3-anchor trilateration) ---------------------

def lse3(d: np.ndarray, A: np.ndarray) -> np.ndarray:
    """3-anchor closed-form. d: (3,), A: (3,3) [x,y,z] anchors.
    Returns (x, y, z) — but we only use (x, y) for the 2D comparison.
    """
    p0, p1, p2 = A[0], A[1], A[2]
    A_mat = 2 * (A[1:, :2] - p0[:2])
    b = np.array([
        d[0] ** 2 - d[1] ** 2 + (A[1] ** 2).sum() - (p0 ** 2).sum(),
        d[0] ** 2 - d[2] ** 2 + (A[2] ** 2).sum() - (p0 ** 2).sum(),
    ], dtype=np.float64)
    sol, *_ = np.linalg.lstsq(A_mat, b, rcond=None)
    # z from the first range
    z = d[0] - np.linalg.norm(A[0, :2] - sol)
    return np.array([sol[0], sol[1], z])


# ---- Training --------------------------------------------------------------

def train(model, Xtr, Ytr, Xva, Yva, n_epochs, lr, device):
    """Train on the windows. Reports train+val MAE per epoch."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_t = torch.from_numpy(Xtr).to(device)
    Ytr_t = torch.from_numpy(Ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device) if len(Xva) > 0 else None
    Yva_t = torch.from_numpy(Yva).to(device) if len(Yva) > 0 else None
    history = {"epoch": [], "train_mae": [], "val_mae": []}
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        pred = model(Xtr_t)
        loss = TF.mse_loss(pred, Ytr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_mae = float((pred - Ytr_t).abs().mean().item())
        val_mae = float("nan")
        if Xva_t is not None:
            model.eval()
            with torch.no_grad():
                pred_v = model(Xva_t)
                val_mae = float((pred_v - Yva_t).abs().mean().item())
        history["epoch"].append(ep + 1)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        if (ep + 1) % max(1, n_epochs // 10) == 0 or ep == 0:
            v = f"{val_mae:.3f}" if not np.isnan(val_mae) else "  n/a"
            print(f"  ep {ep+1:3d}/{n_epochs}  "
                  f"train_mae={train_mae:.3f}  val_mae={v}  "
                  f"({time.time()-t0:.1f}s)")
    return history


# ---- Eval + plot -----------------------------------------------------------

def evaluate_and_plot(model, X, Y, anchor_pos_world, toa_cap,
                       history, out_dir: Path, T: int, stride: int,
                       csv_path: Path):
    """Evaluate on every window, run LSE baseline, plot predicted track
    over ARKit GT, save MAE table and history CSV."""
    device = next(model.parameters()).device
    X_t = torch.from_numpy(X).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()                 # (W, 2)
    # LSE baseline using the LAST timestep's ToA of each window
    lse_preds = []
    for w in range(len(X)):
        last_d = X[w, -1, :, 0] * toa_cap                # (3,) in meters
        try:
            p = lse3(last_d, anchor_pos_world)
            lse_preds.append(p[:2])                      # (x, y) only
        except Exception:
            lse_preds.append([np.nan, np.nan])
    lse_preds = np.asarray(lse_preds, dtype=np.float32)

    # ---- MAE table ----
    def mae_per_axis(P, Y):
        diff = P - Y
        return float(np.nanmean(np.abs(diff[:, 0]))), \
               float(np.nanmean(np.abs(diff[:, 1]))), \
               float(np.nanmean(np.linalg.norm(diff, axis=1)))
    asta_x, asta_y, asta_norm = mae_per_axis(preds, Y)
    lse_x, lse_y, lse_norm = mae_per_axis(lse_preds, Y)
    print(f"\nMAE on all {len(X)} windows (car frame, m):")
    print(f"  ASTA : x={asta_x:.3f}  y={asta_y:.3f}  ||p-p*||={asta_norm:.3f}")
    print(f"  LSE  : x={lse_x:.3f}  y={lse_y:.3f}  ||p-p*||={lse_norm:.3f}")

    # Save MAE table
    mae_path = out_dir / "mae_table.csv"
    with open(mae_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mae_x_m", "mae_y_m", "mae_norm_m", "n_windows"])
        w.writerow(["ASTA", f"{asta_x:.4f}", f"{asta_y:.4f}",
                     f"{asta_norm:.4f}", len(X)])
        w.writerow(["LSE3", f"{lse_x:.4f}", f"{lse_y:.4f}",
                     f"{lse_norm:.4f}", len(X)])
    print(f"  MAE table → {mae_path}")

    # Save history CSV
    hist_path = out_dir / "train_history.csv"
    with open(hist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mae", "val_mae"])
        for ep, t, v in zip(history["epoch"], history["train_mae"],
                            history["val_mae"]):
            w.writerow([ep, f"{t:.6f}", f"{v:.6f}"])

    # ---- Plot 1: Predicted track vs ARKit GT ----
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.plot(Y[:, 1], Y[:, 0], "ko-", lw=1.5, ms=6, label="ARKit GT (ty_car, tx_car)",
            zorder=3)
    ax.plot(preds[:, 1], preds[:, 0], "b^-", lw=1.5, ms=8,
            label="ASTA predicted", zorder=4)
    ax.plot(lse_preds[:, 1], lse_preds[:, 0], "rs--", lw=1.0, ms=6,
            alpha=0.7, label="LSE3 baseline", zorder=2)
    ax.set_xlabel("py_car (m, lateral)")
    ax.set_ylabel("px_car (m, forward)")
    ax.set_title(
        f"uwbx4_aligned_2hz: ASTA vs LSE3 (car frame)\n"
        f"{len(X)} windows, T={T}, stride={stride}, "
        f"ASTA norm MAE = {asta_norm:.3f} m, LSE = {lse_norm:.3f} m"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(out_dir / "pred_track.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  pred_track.png → {out_dir / 'pred_track.png'}")

    # ---- Plot 2: Train history ----
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ep = history["epoch"]
    ax.plot(ep, history["train_mae"], "b-", lw=1.5, label="train MAE (m)")
    val_mae = np.array(history["val_mae"])
    if not np.isnan(val_mae).all():
        ax.plot(ep, val_mae, "r--", lw=1.5, label="val MAE (m)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE (m)")
    ax.set_title("ASTA training history (car frame location MAE)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.savefig(out_dir / "train_history.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  train_history.png → {out_dir / 'train_history.png'}")


# ---- Main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default="/Users/justin/ultraclear-paper/code/data/"
                           "real_data_arkit/uwbx4_aligned_2hz.csv")
    p.add_argument("--T", type=int, default=10,
                   help="window length (rows)")
    p.add_argument("--stride", type=int, default=10,
                   help="window stride (rows); set to 1 for overlapping")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="time-ordered val fraction. 0 = no split, train "
                        "and report MAE on the same windows (in-sample).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--toa_cap", type=float, default=10.0)
    p.add_argument("--out_dir",
                   default="/Users/justin/ultraclear-paper/code/"
                           "checkpoints/_tmp_asta_uwbx4_3anchor")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: cpu  seed: {args.seed}  T={args.T}  "
          f"stride={args.stride}  epochs={args.epochs}  "
          f"val_frac={args.val_frac}  toa_cap={args.toa_cap}")
    print(f"input CSV: {csv_path}")
    print(f"output dir: {out_dir}")
    print()

    # ---- Load data ----
    d = load_uwbx4_aligned(csv_path)
    n = len(d["y"])
    print(f"loaded {n} valid (d1, d2, d3, px_car, py_car) rows")
    if n < args.T:
        raise RuntimeError(
            f"only {n} rows but T={args.T} — need at least T+1 rows")

    # ---- Anchor positions in WORLD frame (constant per row) ----
    # These come from anc{i}_wx, anc{i}_wy, anc{i}_wz in the CSV.
    anc_pos = []
    with open(csv_path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        first = next(rdr)
    for i in [1, 2, 3]:
        anc_pos.append([float(first[header.index(f"anc{i}_wx")]),
                        float(first[header.index(f"anc{i}_wy")]),
                        float(first[header.index(f"anc{i}_wz")])])
    ANCHOR_POS = np.asarray(anc_pos, dtype=np.float64)
    print(f"anchor positions (world, 3 anchors):\n  {ANCHOR_POS}")

    # ---- Build windows ----
    toa_norm = normalize_toa(d["toa"], toa_cap=args.toa_cap)
    # make_windows returns X_win of shape (W, T, 3); we add the channel
    # dim to get (W, T, 3, 1) for the model.
    X_win, Y_win, idx = make_windows(
        toa_norm, d["y"], T=args.T, stride=args.stride)
    X_win = X_win[..., np.newaxis]
    print(f"built {len(X_win)} windows of T={args.T} (stride={args.stride})")
    print(f"  X_win shape: {X_win.shape}, Y_win shape: {Y_win.shape}")
    if len(X_win) < 2:
        raise RuntimeError(
            f"only {len(X_win)} windows — need at least 2 for train/val "
            f"split. Try a smaller --stride (e.g. --stride 1) or smaller --T."
        )

    # ---- Time-ordered split ----
    train_idx, val_idx = time_ordered_split(len(X_win), val_frac=args.val_frac)
    Xtr, Ytr = X_win[train_idx], Y_win[train_idx]
    Xva, Yva = X_win[val_idx], Y_win[val_idx]
    print(f"split: {len(train_idx)} train, {len(val_idx)} val")

    # ---- Build model ----
    cfg = Config()
    cfg.n_anchors = 3
    cfg.n_channels = 1
    cfg.n_asta_blocks = 2
    cfg.d_model = 64
    cfg.n_attention_heads = 4
    cfg.tcn_kernel = 3
    cfg.device = "cpu"
    print(f"\nmodel config: anchors=3, channels=1, "
          f"d_model={cfg.d_model}, n_blocks={cfg.n_asta_blocks}, "
          f"heads={cfg.n_attention_heads}")
    model = ASTALocator(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params:,}")

    # ---- Train ----
    print(f"\n=== train ({args.epochs} epochs) ===")
    history = train(model, Xtr, Ytr, Xva, Yva,
                     n_epochs=args.epochs, lr=args.lr, device="cpu")

    # ---- Save ckpt ----
    ckpt_path = out_dir / "asta_uwbx4_3anchor.pt"
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
        "history": history,
        "anchor_pos_world": ANCHOR_POS,
        "toa_cap": args.toa_cap,
        "T": args.T,
        "stride": args.stride,
        "train_idx": train_idx,
        "val_idx": val_idx,
        # Also save the windows themselves so the viz script can
        # re-evaluate on the full sequence (not just the val split).
        "X_win": X_win,
        "Y_win": Y_win,
    }, ckpt_path)
    print(f"\nsaved → {ckpt_path}")

    # ---- Evaluate + plot ----
    print(f"\n=== evaluate (all {len(X_win)} windows) ===")
    evaluate_and_plot(model, X_win, Y_win, ANCHOR_POS, args.toa_cap,
                       history, out_dir,
                       T=args.T, stride=args.stride, csv_path=csv_path)


if __name__ == "__main__":
    main()
