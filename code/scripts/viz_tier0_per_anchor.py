"""Per-anchor Tier-0 SSL error visualization (aggregated across val windows).

For each of the 4 anchors on each of the 2 channels, shows the
distribution of |reconstruction error| at masked cells across the
full validation split (~240 windows × 4 anchors × ~7 masked
cells/window ≈ 7,000 cells per anchor). Each subplot is a
histogram + KDE of the pooled per-cell |error| values, with a
vertical line at the per-anchor RMSE.

The contrast between anchors on the AoA row visually demonstrates
the "trivial-anchor" phenomenon: A2's azimuth is near-constant
in the dataset, so its |error| distribution is sharply peaked at
a low value; A1/A3/A4 carry the actual geometric information and
have visibly broader/heavier-tailed distributions.

Why aggregated: a single 32-trial window has only ~5-9 masked
cells per anchor — the resulting bar plot is statistically
unreliable. Pooling across the full val split gives hundreds of
samples per anchor and makes the per-anchor asymmetry
quantitatively trustworthy.

Usage:
    python3.11 -m scripts.viz_tier0_per_anchor --ckpt <ckpt.pt>
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer
from scripts.train_tier0_sionna_f2 import (
    SionnaGridWindowF2Dataset, CHANNEL_NAMES,
)
from scripts.viz_tier0_metrics_f2 import evaluate

ERR_CAP = [0.5, 30.0]  # ToA: 0.5 m, AoA: 30 deg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_windows", type=int, default=None,
                    help="how many val windows to pool over "
                         "(default: all)")
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
    n_windows = (args.n_windows if args.n_windows is not None
                  else len(val_indices))
    print(f"val windows: {len(val_indices)}; pooling over {n_windows}")

    model = SSLPretrainer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # `evaluate` from viz_tier0_metrics_f2.py already runs the model
    # on the val split and returns denormalized pred/true/anchor
    # arrays. Reuse that — keeps the eval pipeline consistent.
    res = evaluate(model, full_ds, cfg, norm_mean, norm_std,
                    n_windows=n_windows, seed=args.seed)
    M = cfg.n_anchors
    anchor_labels = [f"A{m+1}" for m in range(M)]
    anchor_colors = ["C0", "C1", "C2", "C3"]

    # Layout: 2 rows (channels) × 4 cols (anchors)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6),
                              constrained_layout=True)

    for f in range(2):  # 0=ToA, 1=AoA
        err = np.abs(res["pred"][f] - res["true"][f])  # (N_masked,)
        anchor_id = res["anchor"][f]                    # (N_masked,)
        chan_name = CHANNEL_NAMES[f]
        units = "m" if f == 0 else "deg"
        # shared y-axis per row so per-anchor densities are visually
        # comparable within a row
        shared_ymax = 0.0
        for m in range(M):
            ax = axes[f, m]
            merr = err[anchor_id == m]
            n = len(merr)
            if n == 0:
                ax.text(0.5, 0.5, "no masked cells",
                         ha="center", va="center",
                         transform=ax.transAxes, fontsize=10)
                continue
            # histogram + KDE
            bins = np.linspace(0, ERR_CAP[f], 41)
            ax.hist(merr, bins=bins, color=anchor_colors[m],
                     alpha=0.5, edgecolor="black", linewidth=0.4,
                     density=True, label=f"n={n}")
            try:
                kde = gaussian_kde(merr)
                xs = np.linspace(0, ERR_CAP[f], 200)
                shared_ymax = max(shared_ymax, float(kde(xs).max()))
            except Exception:
                pass  # KDE can fail on degenerate data
            # vertical lines for RMSE, MAE
            rmse = float(np.sqrt(np.mean(merr ** 2)))
            mae = float(np.mean(merr))
            ax.axvline(rmse, color="black", linestyle="-", lw=1.5)
            ax.axvline(mae, color="black", linestyle="--", lw=1.0)
            ax.set_xlim(0, ERR_CAP[f])
            ax.set_title(f"{chan_name} — {anchor_labels[m]}\n"
                          f"RMSE={rmse:.3f} {units}  "
                          f"MAE={mae:.3f} {units}  (n={n})",
                          fontsize=8, loc="left")
            ax.grid(True, alpha=0.3)
        # shared y-axis for this row, set after the loop
        for m in range(M):
            axes[f, m].set_ylim(0, shared_ymax * 1.15)
            if f == 0:
                axes[f, m].set_xlabel(f"|error| ({units})", fontsize=8)
            if f == 1:
                axes[f, m].set_xlabel(f"|error| ({units})", fontsize=8)
            if m == 0:
                axes[f, m].set_ylabel("density", fontsize=8)

    fig.suptitle(f"Tier 0 SSL (F=2, 3:1 weighting) — per-anchor "
                  f"|reconstruction error| distribution, pooled over "
                  f"{n_windows} val windows (mask ratio = "
                  f"{cfg.mask_ratio*100:.0f}%)  "
                  f"(solid line: RMSE, dashed line: MAE)",
                  fontsize=11, y=1.02)
    out = out_dir / "viz_tier0_per_anchor.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
