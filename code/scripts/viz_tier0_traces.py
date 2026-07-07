"""Option B (v2): per-anchor time-series traces showing SSL refill quality.

Per-window, per-anchor trace:
  - Background: measured ToA over T trials (gray line, low alpha).
  - Markers at masked cells, color-coded by |pred - true| error:
      * white → perfect (|err| < 1 cm)
      * light blue → small error (< 10 cm)
      * red → large error (>= 10 cm)
  - Masked-cell input positions get magenta X markers so the reader
    can see where the model is being asked to refill.
  - Per-anchor RMSE printed in the y-axis label.

This is the most direct "see the refill" visualization, and the
color-coded markers make it obvious where the model is succeeding
and where it is failing.
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
from scripts.train_tier0_sionna import SionnaGridWindowDataset


# Color thresholds (meters)
ERR_GOOD = 0.01    # below this: white "perfect" marker
ERR_BAD = 0.10     # above this: full-saturation red
ERR_CAP = 0.50     # beyond this is clamped (for colorbar normalization)
MASK_X_COLOR = "magenta"


def _err_color(err, cmap_name="RdBu_r"):
    """Map an |error| to a (r, g, b, a) color.

    White at 0, light blue near ERR_GOOD, full red at >= ERR_BAD.
    Uses a custom colormap so the white-to-red transition is monotonic
    in error magnitude (not diverging, since |err| is always >= 0).
    """
    # custom colormap: white -> light blue -> red
    cdict = {
        "red":   [(0.0, 1.0, 1.0), (0.5, 0.85, 0.85), (1.0, 0.85, 0.85)],
        "green": [(0.0, 1.0, 1.0), (0.5, 0.95, 0.95), (1.0, 0.20, 0.20)],
        "blue":  [(0.0, 1.0, 1.0), (0.5, 1.00, 1.00), (1.0, 0.20, 0.20)],
    }
    cmap = cm.colors.LinearSegmentedColormap("err_white_to_red", cdict, N=256)
    norm = Normalize(vmin=0.0, vmax=ERR_CAP, clip=True)
    return cmap(norm(err)), cmap, norm


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

    ds = SionnaGridWindowDataset(
        csv, seq_len=cfg.seq_len, n_channels=cfg.n_channels, stride=cfg.seq_len)
    model = SSLPretrainer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.n_windows, len(ds)), replace=False)

    # for the per-window colorbar we need the cmap + norm
    _, cmap, norm = _err_color(ERR_BAD)  # values don't matter, just get cmap+norm

    for idx in indices:
        X, _, _, _, _ = ds[int(idx)]
        Xb = X.unsqueeze(0)
        torch.manual_seed(args.seed + int(idx))
        with torch.no_grad():
            X_masked, mask = random_mask(Xb, cfg.mask_ratio)
            x_pred, _ = model(X_masked)

        Xnp = X.squeeze(-1).numpy()
        X_masked_np = X_masked.squeeze(0).squeeze(-1).numpy()
        x_pred_np = x_pred.squeeze(0).squeeze(-1).numpy()
        mask_np = mask.squeeze(0).numpy()
        T, M = Xnp.shape
        n_masked = int(mask_np.sum())
        win_rmse = float(np.sqrt(np.mean((x_pred_np[mask_np] - Xnp[mask_np]) ** 2)))

        fig, axes = plt.subplots(M, 1, figsize=(11, 1.9 * M), sharex=True,
                                  constrained_layout=True)
        if M == 1:
            axes = [axes]

        for m in range(M):
            ax = axes[m]
            t = np.arange(T)
            # full measured trace (gray)
            ax.plot(t, Xnp[:, m], "-", color="0.55", alpha=0.55,
                    linewidth=1.4, label="measured ToA")
            # mask positions as magenta X (these cells were zeroed in input)
            t_masked = np.where(mask_np[:, m])[0]
            if len(t_masked) > 0:
                ax.plot(t_masked, np.zeros_like(t_masked, dtype=float),
                        "x", color=MASK_X_COLOR, markersize=10,
                        markeredgewidth=1.8, label="masked (input=0)",
                        zorder=3)
                # color-coded prediction dots
                errs_m = np.abs(x_pred_np[t_masked, m] - Xnp[t_masked, m])
                colors, _, _ = _err_color(errs_m)
                ax.scatter(t_masked, x_pred_np[t_masked, m],
                           c=colors, s=70, edgecolors="black",
                           linewidths=0.8, zorder=4,
                           label="SSL prediction (color = |error|)")
                # ground-truth dots (small, just for reference)
                ax.scatter(t_masked, Xnp[t_masked, m],
                           marker="|", s=40, color="black", zorder=5,
                           label="true ToA")
            anchor_rmse = float(np.sqrt(np.mean(
                (x_pred_np[mask_np[:, m], m] - Xnp[mask_np[:, m], m]) ** 2
            ))) if mask_np[:, m].any() else float("nan")
            ax.set_ylabel(f"A{m+1}\nToA (m)\nRMSE = {anchor_rmse:.3f}",
                           fontsize=9)
            ax.grid(True, alpha=0.3)
            if m == 0:
                ax.legend(loc="upper right", fontsize=7, ncol=4,
                           framealpha=0.9)
            if m == M - 1:
                ax.set_xlabel("trial index")

        # shared error colorbar on the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label("|pred − true|  (m)", fontsize=9)
        cb.ax.tick_params(labelsize=8)

        fig.suptitle(
            f"Window {int(idx)} — per-anchor refill (color = prediction error, "
            f"magenta X = masked input, window RMSE = {win_rmse:.3f} m, "
            f"mask ratio = {cfg.mask_ratio*100:.0f}%)",
            fontsize=10,
        )
        out = out_dir / f"viz_B_traces_win{int(idx):04d}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
