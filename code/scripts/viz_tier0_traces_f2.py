"""Option B (F=2): per-anchor time-series traces for both ToA and AoA
channels, showing SSL refill quality.

Layout per window: 4 rows (one per anchor) × 2 columns (ToA | AoA).
Each subplot shows the noisy measured trace, mask positions, and
SSL predictions colored by |pred-true| error.

Usage:
    python3.11 -m scripts.viz_tier0_traces_f2 --ckpt <ckpt.pt>
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer, random_mask
from scripts.train_tier0_sionna_f2 import SionnaGridWindowF2Dataset, CHANNEL_NAMES

ERR_GOOD = 0.01
ERR_BAD = [0.10, 10.0]   # per-channel: ToA 10cm, AoA 10 deg
ERR_CAP = [0.50, 30.0]   # for colorbar
MASK_X_COLOR = "magenta"
UNITS = ["m", "deg"]


def _err_cmap():
    cdict = {
        "red":   [(0.0, 1.0, 1.0), (0.5, 0.85, 0.85), (1.0, 0.85, 0.85)],
        "green": [(0.0, 1.0, 1.0), (0.5, 0.95, 0.95), (1.0, 0.20, 0.20)],
        "blue":  [(0.0, 1.0, 1.0), (0.5, 1.00, 1.00), (1.0, 0.20, 0.20)],
    }
    return cm.colors.LinearSegmentedColormap("err_white_to_red", cdict, N=256)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_windows", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt["cfg"].items() if k in Config.__dataclass_fields__})
    csv = args.csv or ckpt["csv"]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.n_channels != 2:
        raise ValueError(f"this script requires n_channels=2, got {cfg.n_channels}")

    norm_mean = ckpt["norm_mean"]
    norm_std = ckpt["norm_std"]

    full_ds = SionnaGridWindowF2Dataset(csv, seq_len=cfg.seq_len, stride=cfg.seq_len)
    n_grid = len(full_ds.grid_ids)
    n_train_grid = max(1, int(0.8 * n_grid))
    train_grid_ids = set(full_ds.grid_ids[:n_train_grid])
    full_ds.compute_norm_stats(train_grid_ids)
    full_ds.norm_mean = norm_mean
    full_ds.norm_std = norm_std

    model = SSLPretrainer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(full_ds), size=min(args.n_windows, len(full_ds)), replace=False)

    cmap = _err_cmap()
    norm = [Normalize(vmin=0.0, vmax=ERR_CAP[f], clip=True) for f in range(2)]

    for idx in indices:
        X, _, _, _, _ = full_ds[int(idx)]
        Xb = X.unsqueeze(0)
        torch.manual_seed(args.seed + int(idx))
        with torch.no_grad():
            X_masked, mask = random_mask(Xb, cfg.mask_ratio)
            x_pred, _ = model(X_masked)
        Xnp = X.squeeze(-1).numpy()                # (T, M, 2)
        x_pred_np = x_pred.squeeze(0).squeeze(-1).numpy()
        mask_np = mask.squeeze(0).numpy()          # (T, M)
        # denormalize
        X_den = Xnp * norm_std + norm_mean
        x_pred_den = x_pred_np * norm_std + norm_mean
        T, M, F_ = X_den.shape
        n_masked = int(mask_np.sum())
        win_rmse = [float(np.sqrt(np.mean(
            (x_pred_den[..., f][mask_np] - X_den[..., f][mask_np]) ** 2)))
            for f in range(F_)]

        fig, axes = plt.subplots(M, F_, figsize=(12, 1.7 * M), sharex=True,
                                  constrained_layout=True)
        if M == 1:
            axes = axes[None, :]

        for m in range(M):
            for f in range(F_):
                ax = axes[m, f]
                t = np.arange(T)
                # full measured trace (gray)
                ax.plot(t, X_den[:, m, f], "-", color="0.55", alpha=0.55,
                        linewidth=1.2, label="measured" if (m == 0 and f == 0) else None)
                # mask positions
                t_masked = np.where(mask_np[:, m])[0]
                if len(t_masked) > 0:
                    ax.plot(t_masked, np.zeros_like(t_masked, dtype=float),
                            "x", color=MASK_X_COLOR, markersize=9,
                            markeredgewidth=1.6,
                            label="masked" if (m == 0 and f == 0) else None,
                            zorder=3)
                    # color-coded prediction dots
                    errs_m = np.abs(x_pred_den[t_masked, m, f] - X_den[t_masked, m, f])
                    colors = cmap(norm[f](errs_m))
                    ax.scatter(t_masked, x_pred_den[t_masked, m, f],
                               c=colors, s=55, edgecolors="black",
                               linewidths=0.7, zorder=4,
                               label="SSL pred (color=|err|)" if (m == 0 and f == 0) else None)
                    ax.scatter(t_masked, X_den[t_masked, m, f],
                               marker="|", s=30, color="black", zorder=5,
                               label="true" if (m == 0 and f == 0) else None)
                anchor_rmse_f = float(np.sqrt(np.mean(
                    (x_pred_den[mask_np[:, m], m, f] - X_den[mask_np[:, m], m, f]) ** 2
                ))) if mask_np[:, m].any() else float("nan")
                if f == 0:
                    ax.set_ylabel(f"A{m+1}  (ToA)\nRMSE = {anchor_rmse_f:.3f} {UNITS[f]}",
                                   fontsize=8)
                else:
                    ax.set_ylabel(f"A{m+1}  (AoA az)\nRMSE = {anchor_rmse_f:.2f} {UNITS[f]}",
                                   fontsize=8)
                ax.grid(True, alpha=0.3)
                if m == 0:
                    ax.set_title(f"channel {f}: {CHANNEL_NAMES[f]}", fontsize=9)
                if m == M - 1:
                    ax.set_xlabel("trial index", fontsize=8)
                if m == 0 and f == 0:
                    ax.legend(loc="upper right", fontsize=7, ncol=4, framealpha=0.9)

        # one colorbar per channel
        cax1 = fig.add_axes([0.92, 0.55, 0.012, 0.32])
        cb1 = fig.colorbar(cm.ScalarMappable(norm=norm[0], cmap=cmap), cax=cax1)
        cb1.set_label("|err| (m)", fontsize=8)
        cb1.ax.tick_params(labelsize=7)
        cax2 = fig.add_axes([0.92, 0.13, 0.012, 0.32])
        cb2 = fig.colorbar(cm.ScalarMappable(norm=norm[1], cmap=cmap), cax=cax2)
        cb2.set_label("|err| (deg)", fontsize=8)
        cb2.ax.tick_params(labelsize=7)

        fig.suptitle(
            f"Window {int(idx)} — per-anchor / per-channel refill  "
            f"(window RMSE: ToA={win_rmse[0]:.3f} m, AoA={win_rmse[1]:.2f} deg, "
            f"mask ratio = {cfg.mask_ratio*100:.0f}%)",
            fontsize=10,
        )
        out = out_dir / f"viz_B_traces_f2_win{int(idx):04d}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
