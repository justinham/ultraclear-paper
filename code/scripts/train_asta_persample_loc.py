"""ASTA per-sample localization (input 32-step window → output 32 (x, y)
predictions, one per timestep), with NO NLOS branch and NO signal
restoration. Frozen pretrained ASTAEncoder + fresh per-timestep
regression head.

Architecture:
  - Input:  (B, T=32, M=4, F=2)  raw ToA+AoA windows
  - ASTAEncoder (frozen):  (B, T, M, d_model=64)  features per (t, anchor)
  - Per-timestep pooling: mean over the 4 anchors →  (B, T, d_model)
  - Per-timestep regression head (2-layer MLP):  (B, T, 2)  (x, y)

Head is small (~10K params) and trained end-to-end on top of the
frozen encoder. Eval is per-sample localization MAE on test 32-step
windows of the sionna track2 data (e.g. indoor_garage_parking_05).

Output:
  - ckpt:       checkpoints/_tmp_asta_persample/head.pt
  - eval CSV:   figures/asta_persample_eval_<scene>.csv
  - scatter:    figures/asta_persample_track_<scene>.png
  - err hist:   figures/asta_persample_errhist_<scene>.png
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
# Data: 32-step windows, label = (x, y) per step (the dwell/transition point)
# --------------------------------------------------------------------------- #

def load_track2_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a sionna track2 swapped CSV; return (X, Y) where
        X: (N, M, F) = per-row ToA+azimuth-AoA from 4 anchors
             (M=4, F=2: [distance_m, azimuth_deg])
        Y: (N, 2)    = per-row (tag_x, tag_y)

    Matches the convention used by scripts/eval_tier1_on_track2_*
    so that the frozen ASTA encoder (trained on the same input
    layout) gets the right shape. Skips rows with any NaN/empty
    in those columns.
    """
    X, Y = [], []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                ds = [float(r[f"A{i}_measured_dist_m"]) for i in (1, 2, 3, 4)]
                aoas = [float(r[f"A{i}_aoa_azimuth_deg"]) for i in (1, 2, 3, 4)]
                tag = [float(r["tag_x"]), float(r["tag_y"])]
                if any(np.isnan(v) for v in ds + aoas + tag):
                    continue
                X.append(np.array([ds, aoas], dtype=np.float32).T)  # (M=4, F=2)
                Y.append(tag)
            except (KeyError, ValueError):
                continue
    if not X:
        return np.zeros((0, 4, 2), dtype=np.float32), \
               np.zeros((0, 2), dtype=np.float32)
    return np.stack(X), np.array(Y, dtype=np.float32)


def make_windows(X: np.ndarray, Y: np.ndarray, T: int, stride: int):
    """Strided windows; each window has T timesteps and T labels
    (one (x, y) per timestep). Returns windows_X (W, T, M, F) and
    windows_Y (W, T, 2). Drops the trailing partial window."""
    N = len(X)
    if N < T:
        return np.zeros((0, T, *X.shape[1:]), dtype=np.float32), \
               np.zeros((0, T, 2), dtype=np.float32)
    starts = list(range(0, N - T + 1, stride))
    wx = np.stack([X[s:s + T] for s in starts])   # (W, T, M, F)
    wy = np.stack([Y[s:s + T] for s in starts])   # (W, T, 2)
    return wx, wy


# --------------------------------------------------------------------------- #
# Model: frozen encoder + small per-timestep regression head
# --------------------------------------------------------------------------- #

class PerSampleLocHead(nn.Module):
    """Maps encoder output (B, T, M, d) to per-timestep (B, T, 2).

    For each timestep, flatten the M anchor features and pass through
    a 2-layer MLP. Same shape as the original Tier1Localizer head,
    but applied independently at every t.
    """

    def __init__(self, d_model: int, n_anchors: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model * n_anchors, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, M, d) → (B, T, M*d) → head → (B, T, 2)
        B, T, M, D = h.shape
        flat = h.reshape(B, T, M * D)
        return self.head(flat)             # (B, T, 2)


class ASTAPerSampleLoc(nn.Module):
    """Frozen ASTA encoder + fresh per-sample localization head."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = ASTAEncoder(cfg)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.head = PerSampleLocHead(cfg.d_model, cfg.n_anchors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, M, F) — encoder expects raw ToA+AoA windows
        with torch.no_grad():
            h = self.encoder(x)               # (B, T, M, d)
        return self.head(h)                   # (B, T, 2)


# --------------------------------------------------------------------------- #
# Training + eval
# --------------------------------------------------------------------------- #

def to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(x).to(device)


def train_and_eval(csv_path: Path, encoder_ckpt: Path, *,
                   T: int = 32, stride: int = 8, n_epochs: int = 200,
                   lr: float = 1e-2, val_frac: float = 0.2,
                   device: str = "cpu", out_dir: Path) -> dict:
    """Train the per-sample head on the frozen encoder features.
    Returns a dict with train/val MAE and per-window MAE on the test split."""
    cfg = Config(n_anchors=4, n_channels=2, d_model=64, n_asta_blocks=2)
    X_np, Y_np = load_track2_csv(csv_path)
    print(f"  loaded {len(X_np)} valid rows from {csv_path.name}")
    print(f"  X shape: {X_np.shape}   Y shape: {Y_np.shape}")
    wx, wy = make_windows(X_np, Y_np, T, stride)
    print(f"  windows: {wx.shape} → {wy.shape}   (T={T}, stride={stride})")
    if len(wx) == 0:
        raise RuntimeError("no full windows — need more data or smaller T")

    # 80/20 split: per-row, with a per-class limit so every waypoint
    # is represented in both train and val.  This mirrors the
    # strategy in scripts/eval_tier1_on_track2_localization.py —
    # it lets the head learn to map features to (x, y) for every
    # waypoint instead of overfitting to a contiguous time chunk.
    n = len(wx)
    val_per_class = max(1, int(0.2 * (n // 12)))  # ~4 windows per wp for val
    # Group windows by which waypoint they END on (last-step label)
    last_wp = wy[:, -1]                              # (W, 2)
    wp_ids = (last_wp[:, 0] * 100 + last_wp[:, 1] * 7).astype(int)  # crude hash
    train_idx, val_idx = [], []
    for u in np.unique(wp_ids):
        idxs = np.where(wp_ids == u)[0]
        n_val_u = min(val_per_class, len(idxs) // 2)
        val_idx.extend(idxs[:n_val_u].tolist())
        train_idx.extend(idxs[n_val_u:].tolist())
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    wx_tr, wy_tr = wx[train_idx], wy[train_idx]
    wx_va, wy_va = wx[val_idx], wy[val_idx]
    n_tr, n_val = len(wx_tr), len(wx_va)
    print(f"  per-class split: {n_tr} train, {n_val} val "
          f"({val_per_class} val per waypoint)")

    # Load ckpt first to get config and normalization stats
    ckpt_full = torch.load(encoder_ckpt, map_location=device, weights_only=False)
    norm_mean = ckpt_full.get("norm_mean", None)
    norm_std = ckpt_full.get("norm_std", None)
    if norm_mean is None or norm_std is None:
        raise RuntimeError("ckpt has no norm_mean/norm_std — needed for input scaling")
    norm_mean = torch.as_tensor(norm_mean, device=device, dtype=torch.float32)  # (F=2,)
    norm_std = torch.as_tensor(norm_std, device=device, dtype=torch.float32)
    print(f"  norm mean: {norm_mean.cpu().tolist()}   std: {norm_std.cpu().tolist()}")

    # X is already (W, T, M, F) — no reshape needed
    wx_tr_t = to_tensor(wx_tr, device)
    wy_tr_t = to_tensor(wy_tr, device)
    wx_va_t = to_tensor(wx_va, device)
    wy_va_t = to_tensor(wy_va, device)
    # Apply the per-channel normalization that the tier1 ckpt was
    # trained on.  X is (W, T, M, F=2) and norm_mean/norm_std are
    # of shape (F=2,).
    wx_tr_t = (wx_tr_t - norm_mean) / norm_std
    wx_va_t = (wx_va_t - norm_mean) / norm_std

    # Build model and load the frozen encoder
    model = ASTAPerSampleLoc(cfg).to(device)
    state = ckpt_full
    # The tier1 ckpt has the encoder weights under 'model_state',
    # and the keys are prefixed with 'encoder.'.
    if isinstance(state, dict):
        for k in ("model_state", "model", "state_dict"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    if isinstance(state, dict):
        # strip 'encoder.' prefix so it matches self.encoder's keys
        state = {k.removeprefix("encoder."): v
                 for k, v in state.items()}
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  encoder load: missing={len(missing)}  unexpected={len(unexpected)}")
        if missing:
            print(f"  missing sample: {missing[:3]}")
        if unexpected:
            print(f"  unexpected sample: {unexpected[:3]}")
    model.encoder.eval()

    # Train the head
    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.head.train()
    for ep in range(1, n_epochs + 1):
        opt.zero_grad()
        pred = model(wx_tr_t)                              # (B, T, 2)
        loss = crit(pred, wy_tr_t)
        loss.backward()
        opt.step()
        if ep in (1, 20, 50, 100, 200) or ep % 50 == 0:
            with torch.no_grad():
                model.head.eval()
                tr_mae = (pred - wy_tr_t).norm(dim=-1).mean().item()
                va_pred = model(wx_va_t)
                va_mae = (va_pred - wy_va_t).norm(dim=-1).mean().item()
                model.head.train()
            print(f"  ep {ep:3d}  train_mse={loss.item():.4f}  "
                  f"train_mae={tr_mae:.3f}m  val_mae={va_mae:.3f}m")

    # Final eval
    model.head.eval()
    with torch.no_grad():
        va_pred = model(wx_va_t).cpu().numpy()            # (n_val, T, 2)
        va_true = wy_va                                   # (n_val, T, 2)
    per_step_err = np.linalg.norm(va_pred - va_true, axis=-1)  # (n_val, T)
    per_window_mae = per_step_err.mean(axis=1)            # (n_val,)
    overall_mae = per_step_err.mean()
    print(f"  FINAL val MAE per step: {overall_mae:.3f} m   "
          f"per window mean {per_window_mae.mean():.3f} m")

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    head_path = out_dir / "head.pt"
    torch.save(model.head.state_dict(), head_path)
    print(f"  saved head ckpt → {head_path}")

    csv_out = out_dir / "eval.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_idx", "t", "x_pred", "y_pred", "x_true", "y_true",
                    "err_m"])
        for wi in range(va_pred.shape[0]):
            for t in range(va_pred.shape[1]):
                w.writerow([wi, t,
                            f"{va_pred[wi, t, 0]:.4f}",
                            f"{va_pred[wi, t, 1]:.4f}",
                            f"{va_true[wi, t, 0]:.4f}",
                            f"{va_true[wi, t, 1]:.4f}",
                            f"{per_step_err[wi, t]:.4f}"])
    print(f"  saved eval CSV → {csv_out}")

    # Plot: per-window MAE bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(np.arange(len(per_window_mae)), per_window_mae,
           color="tab:blue", alpha=0.7)
    ax.axhline(per_window_mae.mean(), color="black", linestyle="--",
               label=f"mean = {per_window_mae.mean():.3f} m")
    ax.set_xlabel("validation window index (chronological)")
    ax.set_ylabel("per-window MAE (m)")
    ax.set_title(f"ASTA per-sample localization (no NLOS branch, no signal "
                 f"restoration)\n{csv_path.name} — T={T}, stride={stride}, "
                 f"n_train={n_tr}, n_val={n_val}")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    bar_path = out_dir / "per_window_mae.png"
    fig.savefig(bar_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved per-window MAE figure → {bar_path}")

    # Plot: histogram of per-step errors
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(per_step_err.flatten(), bins=40, color="tab:blue",
            edgecolor="black", alpha=0.75)
    ax.axvline(overall_mae, color="black", linestyle="--",
               label=f"mean = {overall_mae:.3f} m")
    ax.set_xlabel("per-step Euclidean error (m)")
    ax.set_ylabel("count")
    ax.set_title(f"Per-step error distribution — {csv_path.name}")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    hist_path = out_dir / "per_step_err_hist.png"
    fig.savefig(hist_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved per-step err hist → {hist_path}")

    return {
        "csv": str(csv_path),
        "n_rows": int(len(X_np)),
        "n_windows": int(len(wx)),
        "n_train": int(n_tr),
        "n_val": int(n_val),
        "T": int(T),
        "stride": int(stride),
        "overall_mae_m": float(overall_mae),
        "per_window_mae_mean_m": float(per_window_mae.mean()),
        "per_window_mae_std_m": float(per_window_mae.std()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default="/Users/justin/ultraclear-paper/code/data/"
                           "sionna_track2_swapped/"
                           "sionna_uwb_track_indoor_garage_parking_05_"
                           "A1A4swapped.csv")
    p.add_argument("--encoder_ckpt",
                   default="/Users/justin/ultraclear-paper/code/"
                           "checkpoints/tier1_mixed.pt/tier1_sionna.pt")
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--stride", type=int, default=32,
                   help="stride between windows (default == T, no overlap)")
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out",
                   default="/Users/justin/ultraclear-paper/code/"
                           "checkpoints/_tmp_asta_persample_garage_05")
    args = p.parse_args()

    csv_path = Path(args.csv)
    encoder_ckpt = Path(args.encoder_ckpt)
    out_dir = Path(args.out)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not encoder_ckpt.exists():
        raise FileNotFoundError(encoder_ckpt)

    print(f"ASTA per-sample localization (no NLOS, no signal restoration)")
    print(f"  csv:        {csv_path}")
    print(f"  encoder:    {encoder_ckpt}")
    print(f"  T={args.T}  stride={args.stride}  epochs={args.n_epochs}  "
          f"lr={args.lr}")
    print(f"  out_dir:    {out_dir}")
    print()
    summary = train_and_eval(csv_path, encoder_ckpt,
                              T=args.T, stride=args.stride,
                              n_epochs=args.n_epochs, lr=args.lr,
                              val_frac=args.val_frac, device=args.device,
                              out_dir=out_dir)
    print()
    print("summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
