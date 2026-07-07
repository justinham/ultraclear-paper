"""Option A (F=2): per-channel heatmaps of (masked input, GT, model
output, |recon−truth| error) for a small set of example windows.

Layout per window: 2 rows (one per channel) × 4 columns (masked / GT /
model output / error). Channels are ToA (m) and horizontal AoA (deg).

Design choices (v3 of Option A):
  - Joint mask is preserved: if (t, m) is masked, both ToA and AoA
    are zeroed at that cell. The mask outlines are drawn on both
    rows.
  - Per-channel colormap & vmin/vmax: each row uses the percentile
    range of that channel's ground truth.
  - Row 1 (input) masked cells: solid black for ToA, solid black for
    AoA (uniform treatment).
  - Row 3 (model output): blend of GT at visible cells, prediction at
    masked cells, with yellow outlines on the predicted cells.
  - Row 4 (error): diverging RdBu_r centered at 0, with cap chosen
    per channel (different for ToA and AoA).

Usage:
    python3.11 -m scripts.viz_tier0_heatmaps_f2 --ckpt <ckpt.pt>
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
from matplotlib.colors import Normalize, TwoSlopeNorm

from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer, random_mask
from scripts.train_tier0_sionna_f2 import (
    SionnaGridWindowF2Dataset, CHANNEL_NAMES,
)

CMAP_VALUE = "viridis"
CMAP_ERROR = "RdBu_r"
MASK_EDGE = "magenta"
MASK_LW = 1.2
PRED_EDGE = "yellow"
PRED_LW = 1.5

# Per-channel error caps (in original units)
ERR_CAP = [0.5, 30.0]  # ToA: 0.5 m, AoA: 30 deg


def _percentile_vmin_vmax(arr, low=2.0, high=98.0):
    return float(np.percentile(arr, low)), float(np.percentile(arr, high))


def _draw_outlines(ax, mask, T, M, color, lw):
    for t in range(T):
        for m in range(M):
            if mask[t, m]:
                ax.add_patch(plt.Rectangle((t - 0.5, m - 0.5), 1, 1,
                                            fill=False, edgecolor=color,
                                            linewidth=lw, alpha=0.95))


def _window_rmse(X, X_pred, mask):
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((X_pred[mask] - X[mask]) ** 2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_windows", type=int, default=3)
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
    print(f"norm stats: ToA  mean={norm_mean[0]:.3f} std={norm_std[0]:.3f}  "
          f"AoA  mean={norm_mean[1]:.3f} std={norm_std[1]:.3f}")

    # build dataset and apply saved norm stats
    full_ds = SionnaGridWindowF2Dataset(csv, seq_len=cfg.seq_len, stride=cfg.seq_len)
    n_grid = len(full_ds.grid_ids)
    n_train_grid = max(1, int(0.8 * n_grid))
    train_grid_ids = set(full_ds.grid_ids[:n_train_grid])
    full_ds.compute_norm_stats(train_grid_ids)
    # overwrite with the ckpt's norm stats (they should match, but be safe)
    full_ds.norm_mean = norm_mean
    full_ds.norm_std = norm_std

    model = SSLPretrainer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(full_ds), size=min(args.n_windows, len(full_ds)), replace=False)

    # Layout per window: 2 channels × 4 columns
    # 4 columns: 0=masked input, 1=GT, 2=model output, 3=|recon-truth|
    # 2 rows:    0=ToA, 1=AoA
    # Plus a 5th column for the shared colorbar
    n_rows = len(indices) * 2
    n_cols = 4
    fig = plt.figure(figsize=(15.5, 1.9 * n_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols + 1,
                           width_ratios=[1, 1, 1, 1, 0.06],
                           wspace=0.25, hspace=0.4)
    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])
    cax_value = fig.add_subplot(gs[0, n_cols])  # colorbar for value rows
    cax_err = fig.add_subplot(gs[1, n_cols])    # colorbar for error rows

    anchor_labels = [f"A{m+1}" for m in range(cfg.n_anchors)]
    T = cfg.seq_len
    M = cfg.n_anchors

    # cache for second pass: per-channel per-row vmin/vmax for value scale,
    # and per-channel per-window error data for shared colorbars.
    val_scales = np.zeros((n_rows, 2))  # (vmin, vmax) per row
    err_max = np.zeros(2)              # per-channel
    err_data_per_row = []              # list of (row, channel, err_data)

    for w_i, i in enumerate(indices):
        X, _, _, _, _ = full_ds[int(i)]
        Xb = X.unsqueeze(0)
        torch.manual_seed(args.seed + int(i))
        with torch.no_grad():
            X_masked, mask = random_mask(Xb, cfg.mask_ratio)
            x_pred, _ = model(X_masked)
        # all in normalized units
        Xnp = X.squeeze(-1).numpy()              # (T, M, F=2)
        X_masked_np = X_masked.squeeze(0).squeeze(-1).numpy()
        x_pred_np = x_pred.squeeze(0).squeeze(-1).numpy()
        mask_np = mask.squeeze(0).numpy()        # (T, M)

        # denormalize for display so the colormaps show real units
        def denorm(x):
            return x * norm_std + norm_mean
        X_den = denorm(Xnp)
        X_masked_den = denorm(X_masked_np)
        x_pred_den = denorm(x_pred_np)

        for f in range(2):  # 0=ToA, 1=AoA
            r = w_i * 2 + f
            chan_name = CHANNEL_NAMES[f]
            err_cap = ERR_CAP[f]
            vmin, vmax = _percentile_vmin_vmax(X_den[..., f])
            val_scales[r] = (vmin, vmax)

            # ---- column 1: masked input (blacked at masked cells) ----
            ax = axes[r, 0]
            rgba = cm.viridis(Normalize(vmin=vmin, vmax=vmax)(X_masked_den[..., f]))
            rgba[mask_np] = [0.0, 0.0, 0.0, 1.0]
            ax.imshow(rgba.transpose(1, 0, 2), aspect="auto", origin="lower", interpolation="nearest")
            _draw_outlines(ax, mask_np, T, M, MASK_EDGE, MASK_LW)
            ax.set_yticks(range(M))
            ax.set_yticklabels(anchor_labels, fontsize=7)
            ax.set_xticks([])
            if f == 0 and w_i == 0:
                title = (f"masked input ({cfg.mask_ratio*100:.0f}% blacked)\n"
                          f"channel: {chan_name}")
            elif f == 0:
                title = f"channel: {chan_name}\n({int(mask_np.sum())}/{T*M} cells masked)"
            else:
                title = f"channel: {chan_name}"
            if w_i == 0:
                title = (f"window {int(i)} — {title}\n"
                          f"window RMSE = {_window_rmse(X_den[..., f], x_pred_den[..., f], mask_np):.3f} {('m' if f==0 else 'deg')}")
            ax.set_title(title, fontsize=8, loc="left")

            # ---- column 2: ground truth ----
            ax = axes[r, 1]
            ax.imshow(X_den[..., f].T, aspect="auto", cmap=CMAP_VALUE,
                       vmin=vmin, vmax=vmax, origin="lower",
                       interpolation="nearest")
            _draw_outlines(ax, mask_np, T, M, MASK_EDGE, MASK_LW)
            ax.set_yticks(range(M))
            ax.set_yticklabels(anchor_labels, fontsize=7)
            ax.set_xticks([])
            ax.set_title(f"ground truth\n{chan_name}", fontsize=8, loc="left")

            # ---- column 3: model output (blend + yellow outline) ----
            ax = axes[r, 2]
            blend = np.where(mask_np, x_pred_den[..., f], X_den[..., f])
            ax.imshow(blend.T, aspect="auto", cmap=CMAP_VALUE,
                       vmin=vmin, vmax=vmax, origin="lower",
                       interpolation="nearest")
            _draw_outlines(ax, mask_np, T, M, PRED_EDGE, PRED_LW)
            ax.set_yticks(range(M))
            ax.set_yticklabels(anchor_labels, fontsize=7)
            ax.set_xticks([])
            ax.set_title(f"model output\n{chan_name}", fontsize=8, loc="left")

            # ---- column 4: |recon - truth| at masked cells ----
            ax = axes[r, 3]
            err = np.where(mask_np, np.abs(x_pred_den[..., f] - X_den[..., f]), np.nan)
            err_max[f] = max(err_max[f], float(np.nanmax(err)) if np.any(~np.isnan(err)) else 0.0)
            err_data_per_row.append((r, f, err))
            # placeholder imshow now, colorbar set in second pass
            ax.set_yticks(range(M))
            ax.set_yticklabels(anchor_labels, fontsize=7)
            ax.set_xticks([])

    # ---- second pass: error rows with shared per-channel colorbars ----
    err_norm = []
    for f in range(2):
        cap = min(err_max[f], ERR_CAP[f])
        err_norm.append(TwoSlopeNorm(vmin=-cap, vcenter=0.0, vmax=cap))
    for (r, f, err) in err_data_per_row:
        ax = axes[r, 3]
        im = ax.imshow(err.T, aspect="auto", cmap=CMAP_ERROR,
                        norm=err_norm[f], origin="lower",
                        interpolation="nearest")
        _draw_outlines(ax, mask_np, T, M, MASK_EDGE, MASK_LW)
        # worst-cell star
        if np.any(~np.isnan(err)):
            t_worst, m_worst = np.unravel_index(np.nanargmax(err), err.shape)
            ax.plot([t_worst], [m_worst], marker="*", markersize=12,
                    markeredgecolor="black", markerfacecolor="yellow",
                    markeredgewidth=1.0, zorder=5)
        chan_name = CHANNEL_NAMES[f]
        ax.set_title(f"|recon − truth|\n{chan_name}", fontsize=8, loc="left")
        if r == n_rows - 1:
            ax.set_xlabel("trial")

    # ---- shared value colorbar (one per channel pair) ----
    # We have n_windows windows; for the cb use the LAST window's scales.
    val_vmin = float(np.min(val_scales[:, 0]))
    val_vmax = float(np.max(val_scales[:, 1]))
    cb_val = fig.colorbar(cm.ScalarMappable(
        norm=Normalize(vmin=val_vmin, vmax=val_vmax), cmap=CMAP_VALUE),
        cax=cax_value)
    cb_val.set_label("value\n(m or deg)", fontsize=8)
    cb_val.ax.tick_params(labelsize=7)

    # ---- shared error colorbar (per-channel averages) ----
    cb_err = fig.colorbar(cm.ScalarMappable(norm=err_norm[0], cmap=CMAP_ERROR),
                           cax=cax_err)
    cb_err.set_label("|error|\n(m or deg)", fontsize=8)
    cb_err.ax.tick_params(labelsize=7)

    fig.suptitle(f"Tier 0 SSL (F=2) — masked input / GT / model output / error  "
                  f"for each channel  (mask ratio = {cfg.mask_ratio*100:.0f}%)",
                  fontsize=11, y=1.02)
    out = out_dir / "viz_A_heatmaps_f2.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
