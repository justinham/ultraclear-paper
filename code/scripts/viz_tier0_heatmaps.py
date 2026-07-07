"""Option A (v2): publication-friendly heatmaps of (masked input,
ground truth, reconstruction, |recon−truth| error) for a small set of
example windows.

Design choices for paper-friendliness:
  - Rows 1-3 (masked / GT / reconstruction): viridis with vmin/vmax
    at the 2nd/98th percentile of *ground truth* so the colormap
    spans the typical ToA range, not the rare extremes.
  - Row 4 (error): diverging RdBu_r colormap centered at 0, capped
    at ±0.5 m so small errors are subtle (near-white) and large
    errors pop in saturated red/blue.
  - Mask outlines: high-contrast magenta so they are visible against
    both dark and light cells of the viridis colormap.
  - Per-window RMSE is printed in the row-1 title so the figure is
    self-quantifying.
  - Shared colorbar per column (one for value rows, one for error).
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
from scripts.train_tier0_sionna import SionnaGridWindowDataset


# Color scheme — chosen for print-friendliness (high contrast, colorblind-OK)
CMAP_VALUE = "viridis"        # perceptually uniform; good for continuous magnitude
CMAP_ERROR = "RdBu_r"         # diverging, white at 0
ERR_CAP_M = 0.5               # |error| >= this is full saturation
MASK_EDGE = "magenta"         # high-contrast against viridis
MASK_LW = 1.2                 # outline thickness


def _percentile_vmin_vmax(arr: np.ndarray, low=2.0, high=98.0):
    """Percentile-based vmin/vmax so rare extremes don't squash the colormap."""
    return float(np.percentile(arr, low)), float(np.percentile(arr, high))


def plot_window(ax_row, X, X_masked, mask, X_pred, anchor_labels, cfg, win_id):
    """Plot a 4-row stack of heatmaps for one window.

    X, X_masked, mask, X_pred: (T, M) arrays (already F-squeezed).
    ax_row: 4 axes (one per row).
    """
    T, M = X.shape
    vmin, vmax = _percentile_vmin_vmax(X)

    # ----- value rows (1, 2, 3): viridis -----
    value_data = [X_masked, X, X_pred]
    value_titles = [
        f"masked input ({cfg.mask_ratio*100:.0f}% zeroed)",
        "ground truth",
        "SSL reconstruction",
    ]
    for i, (data, title) in enumerate(zip(value_data, value_titles)):
        ax = ax_row[i]
        im = ax.imshow(data.T, aspect="auto", cmap=CMAP_VALUE,
                        vmin=vmin, vmax=vmax, origin="lower",
                        interpolation="nearest")
        _draw_mask_outlines(ax, mask, T, M)
        ax.set_yticks(range(M))
        ax.set_yticklabels(anchor_labels, fontsize=8)
        ax.set_xticks([])
        # title: include per-window RMSE on the recon row, masked-count on row 1
        if i == 0:
            ax.set_title(f"{title}\n"
                          f"({int(mask.sum())} cells masked out of {T*M}, "
                          f"RMSE on masked = {_window_rmse(X, X_pred, mask):.3f} m)",
                          fontsize=9, loc="left")
        else:
            ax.set_title(title, fontsize=9, loc="left")
    # save one value colorbar handle for later
    return im


def _window_rmse(X, X_pred, mask):
    """RMSE on masked cells for this one window."""
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((X_pred[mask] - X[mask]) ** 2)))


def _draw_mask_outlines(ax, mask, T, M):
    """Draw a high-contrast outline around each masked cell."""
    for t in range(T):
        for m in range(M):
            if mask[t, m]:
                ax.add_patch(plt.Rectangle((t - 0.5, m - 0.5), 1, 1,
                                            fill=False, edgecolor=MASK_EDGE,
                                            linewidth=MASK_LW, alpha=0.95))


def _draw_predicted_cell_outlines(ax, mask, T, M):
    """Yellow outlines around cells where the model's prediction was used
    (the masked cells in column 3 of the figure)."""
    for t in range(T):
        for m in range(M):
            if mask[t, m]:
                ax.add_patch(plt.Rectangle((t - 0.5, m - 0.5), 1, 1,
                                            fill=False, edgecolor="yellow",
                                            linewidth=1.5, alpha=1.0))


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

    ds = SionnaGridWindowDataset(
        csv, seq_len=cfg.seq_len, n_channels=cfg.n_channels, stride=cfg.seq_len)
    model = SSLPretrainer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.n_windows, len(ds)), replace=False)

    # Layout: n_windows rows x 4 columns; plus a 5th column for the error
    # row's diverging colormap.
    fig = plt.figure(figsize=(14.5, 2.4 * args.n_windows),
                      constrained_layout=True)
    gs = fig.add_gridspec(args.n_windows, 5,
                           width_ratios=[1, 1, 1, 1, 0.06],
                           wspace=0.25, hspace=0.35)
    axes = np.empty((args.n_windows, 4), dtype=object)
    for r in range(args.n_windows):
        for c in range(4):
            axes[r, c] = fig.add_subplot(gs[r, c])
    cax_value = fig.add_subplot(gs[0, 4])  # colorbar for value rows
    cax_err = fig.add_subplot(gs[1, 4])    # colorbar for error row

    anchor_labels = [f"A{m+1}" for m in range(cfg.n_anchors)]

    # First pass: plot all windows; track per-row error data so we can
    # build a single shared error colormap.
    err_vmin, err_vmax = None, None
    err_ims = []  # cache to set colorbar after vmin/vmax known

    for row, i in enumerate(indices):
        X, _, _, _, _ = ds[int(i)]
        Xb = X.unsqueeze(0)
        torch.manual_seed(args.seed + int(i))
        with torch.no_grad():
            X_masked, mask = random_mask(Xb, cfg.mask_ratio)
            x_pred, _ = model(X_masked)
        Xnp = X.squeeze(-1).numpy()
        X_masked_np = X_masked.squeeze(0).squeeze(-1).numpy()
        x_pred_np = x_pred.squeeze(0).squeeze(-1).numpy()
        mask_np = mask.squeeze(0).numpy()
        T = cfg.seq_len
        M = cfg.n_anchors

        # rows 1-3: value heatmaps
        vmin, vmax = _percentile_vmin_vmax(Xnp)

        # ---- column 1: masked input. Black out the masked cells so
        # the eye reads them as "missing" (not as 0 m measurements). ----
        ax = axes[row, 0]
        # Build an RGBA array: visible cells show X_masked_np in viridis,
        # masked cells are solid black with full alpha.
        rgba = cm.viridis(Normalize(vmin=vmin, vmax=vmax)(X_masked_np))
        rgba[mask_np] = [0.0, 0.0, 0.0, 1.0]   # pure black for masked cells
        ax.imshow(rgba.transpose(1, 0, 2), aspect="auto", origin="lower",
                   interpolation="nearest")
        # draw magenta outlines on top of the black cells for contrast
        _draw_mask_outlines(ax, mask_np, T, M)
        ax.set_yticks(range(M))
        ax.set_yticklabels(anchor_labels, fontsize=8)
        ax.set_xticks([])
        ax.set_title(
            f"masked input ({cfg.mask_ratio*100:.0f}% blacked out)\n"
            f"({int(mask_np.sum())}/{T * M} cells masked, "
            f"window RMSE = {_window_rmse(Xnp, x_pred_np, mask_np):.3f} m)",
            fontsize=9, loc="left")

        # ---- column 2: ground truth (current behavior, unchanged) ----
        ax = axes[row, 1]
        ax.imshow(Xnp.T, aspect="auto", cmap=CMAP_VALUE,
                   vmin=vmin, vmax=vmax, origin="lower",
                   interpolation="nearest")
        _draw_mask_outlines(ax, mask_np, T, M)
        ax.set_yticks(range(M))
        ax.set_yticklabels(anchor_labels, fontsize=8)
        ax.set_xticks([])
        ax.set_title("ground truth", fontsize=9, loc="left")

        # ---- column 3: hybrid — GT at visible cells, prediction at masked.
        # The non-masked cells display the true value; the masked cells
        # display the model's reconstruction. This makes the figure show
        # exactly what the model is being asked to predict, without
        # hiding the prediction behind a copy of the ground truth. ----
        ax = axes[row, 2]
        recon_blend = np.where(mask_np, x_pred_np, Xnp)
        # show the blend on the standard viridis colormap, then draw a
        # yellow outline around each predicted cell so the reader can
        # see at a glance which cells came from the model.
        ax.imshow(recon_blend.T, aspect="auto", cmap=CMAP_VALUE,
                   vmin=vmin, vmax=vmax, origin="lower",
                   interpolation="nearest")
        _draw_predicted_cell_outlines(ax, mask_np, T, M)
        ax.set_yticks(range(M))
        ax.set_yticklabels(anchor_labels, fontsize=8)
        ax.set_xticks([])
        ax.set_title("model output\n(GT at visible, prediction at masked)",
                      fontsize=9, loc="left")

        # row 4: error heatmap (diverging, NaN for non-masked)
        ax = axes[row, 3]
        err_data = np.where(mask_np, np.abs(x_pred_np - Xnp), np.nan)
        # track global err_vmax across all windows for shared colorbar
        cur_max = float(np.nanmax(err_data)) if np.any(~np.isnan(err_data)) else 0.0
        err_vmax = cur_max if err_vmax is None else max(err_vmax, cur_max)
        err_ims.append((ax, err_data, row))

    # shared diverging norm for error: 0 in the middle, ±ERR_CAP_M at edges
    err_vmax = min(err_vmax, ERR_CAP_M)  # don't blow out the colormap
    err_norm = TwoSlopeNorm(vmin=-err_vmax, vcenter=0.0, vmax=err_vmax)
    for ax, err_data, row in err_ims:
        im = ax.imshow(err_data.T, aspect="auto", cmap=CMAP_ERROR,
                        norm=err_norm, origin="lower",
                        interpolation="nearest")
        _draw_mask_outlines(ax, mask_np, cfg.seq_len, cfg.n_anchors)
        ax.set_yticks(range(cfg.n_anchors))
        ax.set_yticklabels(anchor_labels, fontsize=8)
        ax.set_xticks([])
        if row == 0:
            ax.set_title(f"|recon − truth|\nat masked cells (capped ±{ERR_CAP_M} m)",
                          fontsize=9, loc="left")
        else:
            ax.set_title("|recon − truth|", fontsize=9, loc="left")
        # annotate the worst masked-cell error
        if np.any(~np.isnan(err_data)):
            t_worst, m_worst = np.unravel_index(np.nanargmax(err_data), err_data.shape)
            ax.plot([t_worst], [m_worst], marker="*", markersize=14,
                    markeredgecolor="black", markerfacecolor="yellow",
                    markeredgewidth=1.0, zorder=5)
    # single shared value colorbar (top-right) and error colorbar (below)
    cb_val = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                                              cmap=CMAP_VALUE),
                            cax=cax_value)
    cb_val.set_label("ToA (m)", fontsize=9)
    cb_val.ax.tick_params(labelsize=8)
    cb_err = fig.colorbar(cm.ScalarMappable(norm=err_norm, cmap=CMAP_ERROR),
                            cax=cax_err)
    cb_err.set_label("|error| (m)", fontsize=9)
    cb_err.ax.tick_params(labelsize=8)

    # row labels on the left of the first column
    for row, i in enumerate(indices):
        axes[row, 0].set_ylabel(f"window {int(i)}\n\nanchor", fontsize=8)
        axes[row, -1].set_xlabel("trial")

    fig.suptitle(f"Tier 0 SSL — masked input (blacked) / GT / "
                  f"model output (GT at visible, prediction at masked) / error  "
                  f"(mask ratio = {cfg.mask_ratio*100:.0f}%, F=1)",
                  fontsize=11, y=1.02)
    out = out_dir / "viz_A_heatmaps.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
