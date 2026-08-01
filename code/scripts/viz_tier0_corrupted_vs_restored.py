"""Tier-0 SSL corrupted-vs-restored heatmap visualization.

For one picked val window, shows for each of the 2 channels
(ToA, AoA azimuth) a side-by-side comparison of:
  - LEFT:  the *corrupted* input. Visible cells show their actual
           value; masked cells are highlighted (magenta outline)
           to show where data is missing.
  - RIGHT: the *restored* output. Visible cells show the same value
           (passthrough from the skip-connection); masked cells show
           the model's prediction at the same position, also
           highlighted so the viewer can see what the model filled in.

Both panels use a shared per-channel colormap range so the
restored values are directly comparable to the surrounding
visible signal. The goal is a pure visual sanity check: "does
the model's prediction at a missing cell look like a plausible
value for that anchor in that trial?"

No statistics are reported on the figure; the goal is
visualization, not measurement.

Usage:
    python3.11 -m scripts.viz_tier0_corrupted_vs_restored --ckpt <ckpt.pt>
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer
from scripts.train_tier0_sionna_f2 import (
    SionnaGridWindowF2Dataset, CHANNEL_NAMES,
)

CMAP_VALUE = "viridis"
CORRUPT_EDGE = "black"        # outline on the masked cells in corrupted
RESTORE_EDGE = "white"        # outline on the masked cells in restored
CORRUPT_FILL = 0.15           # gray value used to render masked cells
                              #   in the corrupted panel (so the
                              #   viewer can see they are *missing*,
                              #   not just outlined)
EDGE_LW = 1.4
# Short display labels (CHANNEL_NAMES from train_tier0_sionna_f2
# uses the full "AoA azimuth (deg)" form; we use the short
# "AoA (deg)" for figure titles only).
CHANNEL_DISPLAY = ["ToA (m)", "AoA (deg)"]


def _percentile_vmin_vmax(arr, low=2.0, high=98.0):
    return float(np.percentile(arr, low)), float(np.percentile(arr, high))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--window_idx", type=int, default=0,
                    help="which val window to visualize (after the "
                         "80/20 grid-id split)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--toa_range", type=float, default=2.0,
                    help="Total ToA colormap range in meters "
                         "(centered on the data mean, ±range/2)")
    p.add_argument("--aoa_range", type=float, default=20.0,
                    help="Total AoA colormap range in degrees "
                         "(centered on the data mean, ±range/2)")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt["cfg"].items()
                     if k in Config.__dataclass_fields__})
    csv = args.csv or ckpt["csv"]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.n_channels != 2:
        raise ValueError(f"this script requires n_channels=2, "
                          f"got {cfg.n_channels}")

    norm_mean = ckpt["norm_mean"]
    norm_std = ckpt["norm_std"]
    print(f"norm stats: ToA  mean={norm_mean[0]:.3f} std={norm_std[0]:.3f}  "
          f"AoA  mean={norm_mean[1]:.3f} std={norm_std[1]:.3f}")

    full_ds = SionnaGridWindowF2Dataset(csv, seq_len=cfg.seq_len,
                                          stride=cfg.seq_len)
    n_grid = len(full_ds.grid_ids)
    n_train_grid = max(1, int(0.8 * n_grid))
    train_grid_ids = set(full_ds.grid_ids[:n_train_grid])
    val_grid_ids = set(full_ds.grid_ids[n_train_grid:])
    full_ds.compute_norm_stats(train_grid_ids)
    full_ds.norm_mean = norm_mean
    full_ds.norm_std = norm_std
    val_indices = [i for i, (gi, _) in enumerate(full_ds.windows)
                    if full_ds.grid_ids[gi] in val_grid_ids]
    print(f"val windows: {len(val_indices)}")

    model = SSLPretrainer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if args.window_idx >= len(val_indices):
        raise ValueError(f"window_idx {args.window_idx} out of range "
                          f"(have {len(val_indices)} val windows)")
    i = val_indices[args.window_idx]
    X, *_ = full_ds[int(i)]
    Xb = X.unsqueeze(0)
    torch.manual_seed(args.seed + int(i))
    with torch.no_grad():
        # pass un-masked X; the model masks internally so the
        # skip-connection passthrough uses the true values at
        # visible cells (see bug-fix note in viz_tier0_metrics_f2.py)
        x_pred, model_mask, _x_in = model(Xb)

    mask_np = model_mask.squeeze(0).numpy()        # (T, M)
    Xnp = X.squeeze(-1).numpy()                    # (T, M, 2) normalized
    x_pred_np = x_pred.squeeze(0).squeeze(-1).numpy()
    # denormalize for display so the colormaps show real units
    X_den = Xnp * norm_std + norm_mean
    x_pred_den = x_pred_np * norm_std + norm_mean
    # the *corrupted* input: replace the masked cells with a neutral
    # gray fill so the viewer can see they are *missing* (not
    # highlighted zeros that look like real data), then highlight
    # with a black outline.
    X_corr_den = X_den.copy()
    X_corr_den[mask_np] = float("nan")  # let the renderer skip them

    T = cfg.seq_len       # 32
    M = cfg.n_anchors     # 4
    anchor_labels = [f"A{m+1}" for m in range(M)]

    # Layout: 2 rows (channels) × 2 cols (corrupted, restored)
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4),
                              gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=0.06, right=0.88, top=0.90, bottom=0.10,
                         hspace=0.30, wspace=0.15)

    for f in range(2):  # 0=ToA, 1=AoA
        chan_name = CHANNEL_DISPLAY[f]
        # shared vmin/vmax between corrupted and restored so the two
        # panels are directly comparable; based on the percentile range
        # of the *visible* signal so masked zeros don't pull the
        # scale toward 0.
        # FIXED per-channel color range (centered on the WINDOW mean)
        # so the colormap always captures the within-window variation
        # regardless of which window is selected. The window mean
        # varies by 5+ m / 100+ deg across the dataset, so centering
        # on the dataset mean (norm_mean) would clip most windows.
        # The half-range is fixed (default ±1 m, ±10 deg) so the
        # visual scale is comparable across windows.
        center = float(X_den[..., f].mean())
        half_range = args.toa_range / 2 if f == 0 else args.aoa_range / 2
        vmin, vmax = center - half_range, center + half_range
        norm = Normalize(vmin=vmin, vmax=vmax)

        # ---- LEFT: corrupted input ----
        ax = axes[f, 0]
        # build a masked RGBA array so missing cells render as
        # the neutral gray fill
        rgba = cm.get_cmap(CMAP_VALUE)(norm(X_corr_den[..., f]))
        rgba[mask_np] = [CORRUPT_FILL, CORRUPT_FILL, CORRUPT_FILL, 1.0]
        ax.imshow(rgba.transpose(1, 0, 2), aspect="auto",
                   origin="lower", interpolation="nearest")
        # outline the masked cells in black (where data is missing)
        for t in range(T):
            for m in range(M):
                if mask_np[t, m]:
                    ax.add_patch(plt.Rectangle(
                        (t - 0.5, m - 0.5), 1, 1,
                        fill=False, edgecolor=CORRUPT_EDGE,
                        linewidth=EDGE_LW, alpha=0.95))
        ax.set_yticks(range(M))
        ax.set_yticklabels(anchor_labels, fontsize=14)
        ax.set_xticks(range(0, T, 4))
        if f == 0:
            ax.set_xticklabels([])  # hide top-row tick labels (shared x-axis)
        else:
            ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(f"{chan_name} — corrupted input",
                      fontsize=15, loc="left")
        if f == 1:
            ax.set_xlabel("trial", fontsize=14)
        if f == 0:
            ax.set_ylabel("anchor", fontsize=14)

        # ---- RIGHT: restored output ----
        ax = axes[f, 1]
        rgba = cm.get_cmap(CMAP_VALUE)(norm(x_pred_den[..., f]))
        # highlight the masked cells with a white outline so the
        # viewer's eye finds the model's predictions; the underlying
        # value is still shown (the prediction itself)
        for t in range(T):
            for m in range(M):
                if mask_np[t, m]:
                    ax.add_patch(plt.Rectangle(
                        (t - 0.5, m - 0.5), 1, 1,
                        fill=False, edgecolor=RESTORE_EDGE,
                        linewidth=EDGE_LW, alpha=0.95))
        ax.imshow(rgba.transpose(1, 0, 2), aspect="auto",
                   origin="lower", interpolation="nearest")
        ax.set_yticks(range(M))
        ax.set_yticklabels(anchor_labels, fontsize=14)
        ax.set_xticks(range(0, T, 4))
        if f == 0:
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(f"{chan_name} — restored output",
                      fontsize=15, loc="left")
        if f == 1:
            ax.set_xlabel("trial", fontsize=14)
        if f == 0:
            ax.set_ylabel("anchor", fontsize=14)

        # shared colorbar for this row, placed to the right of the
        # right subplot
        cax = fig.add_axes([0.93, 0.55 - 0.5 * f, 0.012, 0.35])
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=CMAP_VALUE),
                           cax=cax)
        cb.set_label(chan_name, fontsize=13)
        cb.ax.tick_params(labelsize=12)

    fig.suptitle(f"Tier 0 SSL — corrupted input vs. restored output, "
                  f"val window {int(i)}",
                  fontsize=16, y=1.02)

    out = out_dir / "viz_tier0_corrupted_vs_restored.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
