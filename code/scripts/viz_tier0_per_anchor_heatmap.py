"""Per-anchor Tier-0 SSL error heatmap (single-window, single
subplot per channel).

For one picked val window, shows the |recon − truth| error at
masked cells as a 2-subplot figure:
  - top:    ToA (m) — y-axis = anchors A1..A4, x-axis = 32 trials
  - bottom: AoA azimuth (deg) — same shape

Each cell is colored by its per-cell |error| when masked,
and shown transparent at visible cells. Magenta outlines
mark the masked cells, drawn behind the colormap so the
error color stays the dominant signal.

This is the natural "heatmap" form of the per-anchor error
story: the 2D shape (anchors × trials) makes the per-anchor
error contribution directly visible. A reviewer can scan
vertically to compare anchors and horizontally to see
where in the window the errors cluster.

Usage:
    python3.11 -m scripts.viz_tier0_per_anchor_heatmap --ckpt <ckpt.pt>
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

MASK_EDGE = "magenta"
MASK_LW = 1.0

# Per-channel error caps (in original units)
ERR_CAP = [0.5, 30.0]  # ToA: 0.5 m, AoA: 30 deg
# Short display labels (CHANNEL_NAMES from train_tier0_sionna_f2
# uses the full "AoA azimuth (deg)" form; we use the short
# "AoA (deg)" for figure titles only).
CHANNEL_DISPLAY = ["ToA (m)", "AoA (deg)"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--window_idx", type=int, default=0,
                    help="which val window to visualize (after the "
                         "80/20 grid-id split)")
    p.add_argument("--seed", type=int, default=0)
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
        x_pred, model_mask, _x_in = model(Xb)
    mask_np = model_mask.squeeze(0).numpy()        # (T, M)
    Xnp = X.squeeze(-1).numpy()                    # (T, M, 2) normalized
    x_pred_np = x_pred.squeeze(0).squeeze(-1).numpy()
    X_den = Xnp * norm_std + norm_mean
    x_pred_den = x_pred_np * norm_std + norm_mean

    T = cfg.seq_len       # 32
    M = cfg.n_anchors     # 4
    anchor_labels = [f"A{m+1}" for m in range(M)]
    cmap = matplotlib.colormaps["viridis_r"]  # dark = high error

    # Layout: 2 subplots (one per channel), each is a 4×32 heatmap
    # (anchors on y, trials on x). y-axis is 0..M-1 (we use imshow
    # with origin="lower" so A1 is at the bottom).
    fig, axes = plt.subplots(2, 1, figsize=(14, 4.6),
                              constrained_layout=False,
                              gridspec_kw={"height_ratios": [1, 1]})

    for f in range(2):
        ax = axes[f]
        err = np.abs(x_pred_den[..., f] - X_den[..., f])   # (T, M)
        # set error to NaN at visible cells so they show as transparent
        err_visible = np.where(mask_np, err, np.nan)
        # tighter vmax: use 95th percentile of masked errors, capped
        # at ERR_CAP[f], so the colormap uses more of its dynamic range
        flat_err = err[mask_np]
        if len(flat_err):
            vmax = min(ERR_CAP[f], float(np.percentile(flat_err, 95)) * 1.2)
        else:
            vmax = ERR_CAP[f]
        rgba = cmap(Normalize(vmin=0, vmax=vmax)(
                        np.nan_to_num(err_visible, nan=0.0)))
        rgba[np.isnan(err_visible), 3] = 0.0
        # imshow expects (rows, cols); transpose so y=anchor, x=trial
        ax.imshow(rgba.transpose(1, 0, 2), aspect="auto",
                   origin="lower", interpolation="nearest")
        # magenta outlines at masked cells, drawn behind the data
        for t in range(T):
            for m in range(M):
                if mask_np[t, m]:
                    ax.add_patch(plt.Rectangle(
                        (t - 0.5, m - 0.5), 1, 1,
                        fill=False, edgecolor=MASK_EDGE,
                        linewidth=MASK_LW, alpha=0.6))
        # faint horizontal separators between anchor rows so the
        # eye can see where A1 ends and A2 begins
        for m in range(1, M):
            ax.axhline(m - 0.5, color="black", linewidth=0.4,
                        alpha=0.25)
        # per-anchor RMSE annotation on the right side of each row
        for m in range(M):
            merr = err[mask_np[:, m], m]
            if len(merr):
                rmse = float(np.sqrt(np.mean(merr ** 2)))
                mae = float(np.mean(merr))
            else:
                rmse = mae = float("nan")
            n_m = int(mask_np[:, m].sum())
            ax.text(T + 0.5, m,
                     f"  RMSE={rmse:.3f}  MAE={mae:.3f}  n={n_m}",
                     fontsize=13, va="center", ha="left",
                     family="monospace")
        ax.set_yticks(range(M))
        ax.set_yticklabels(anchor_labels, fontsize=14)
        ax.set_xticks(range(0, T, 4))
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xlim(-0.5, T + 15)  # extra room for per-anchor text annotations
        ax.set_ylim(-0.5, M - 0.5)
        chan_name = CHANNEL_DISPLAY[f]
        units = "m" if f == 0 else "deg"
        ax.set_title(chan_name, fontsize=15, loc="left")
        if f == 1:
            ax.set_xlabel("trial", fontsize=14)
        if f == 0:
            ax.set_ylabel("anchor", fontsize=14)

    fig.subplots_adjust(left=0.04, right=0.84, top=0.92, bottom=0.10,
                         hspace=0.35)

    # one colorbar per subplot on the right
    cax0 = fig.add_axes([0.86, 0.55, 0.012, 0.35])
    cb0 = fig.colorbar(cm.ScalarMappable(
        norm=Normalize(vmin=0, vmax=ERR_CAP[0]), cmap=cmap), cax=cax0)
    cb0.set_label("|error| (m)", fontsize=13)
    cb0.ax.tick_params(labelsize=12)
    cax1 = fig.add_axes([0.86, 0.08, 0.012, 0.35])
    cb1 = fig.colorbar(cm.ScalarMappable(
        norm=Normalize(vmin=0, vmax=ERR_CAP[1]), cmap=cmap), cax=cax1)
    cb1.set_label("|error| (deg)", fontsize=13)
    cb1.ax.tick_params(labelsize=12)

    fig.suptitle(f"Tier 0 SSL (F=2, 3:1 weighting) — per-anchor "
                  f"|reconstruction error| heatmap, val window {int(i)}",
                  fontsize=16, y=1.03)
    out = out_dir / "viz_tier0_per_anchor_heatmap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
