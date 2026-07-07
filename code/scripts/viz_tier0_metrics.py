"""Option C: quantitative metrics for SSL refill quality.

For a trained SSL encoder, evaluate on the validation set and report:

  - Overall RMSE, MAE, R^2 of masked-cell predictions.
  - Per-anchor breakdown of RMSE.
  - Bias: signed mean of (pred - true) at masked cells. A denoiser
    should be unbiased; a shrinkage estimator will have negative
    bias on extremes.
  - Per-mask-distance RMSE: bin masked cells by |true ToA - per-(grid,anchor)
    mean| and report RMSE in each bin. A model that just predicts the
    mean will have low RMSE in the low-deviation bin and high RMSE in
    the high-deviation bin; a model that follows extremes will be flat.
  - Correlation between predicted and true *deviation* from the
    per-(grid,anchor) mean. This is the "did the model pick up the
    fluctuations, not just the average?" test.
  - Residual std at masked cells vs. at visible cells. If reconstruction
    is good, the residual at masked cells should be close to the noise
    std at visible cells (~0.05 m for the LOS Sionna data).
  - Shrinkage factor: regression slope of (pred - mean) onto (true - mean).
    A perfect model has slope 1.0; a "predict the mean" model has slope 0.

Outputs:
  - prints a Markdown-formatted table to stdout
  - writes viz_C_metrics.png (4-panel diagnostic figure)
  - writes viz_C_metrics.csv (per-bin numbers for paper tables)

Usage:
    python3.11 -m scripts.viz_tier0_metrics --ckpt <ckpt.pt>
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer, random_mask
from scripts.train_tier0_sionna import SionnaGridWindowDataset


def evaluate(model, ds, cfg, n_windows=None, seed=0):
    """Run the model on the dataset and collect (pred, true, mask, anchor_id)
    at all masked cells.

    ds is either a SionnaGridWindowDataset or a torch.utils.data.Subset of one.
    Returns a dict of numpy arrays.
    """
    model.eval()
    # Unwrap Subset if needed so we can read .windows / ._per_grid
    base_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    if isinstance(ds, torch.utils.data.Subset):
        indices = ds.indices
    else:
        indices = list(range(len(ds)))

    rng = np.random.default_rng(seed)
    if n_windows is not None:
        indices = [indices[i] for i in rng.choice(len(indices), size=min(n_windows, len(indices)), replace=False)]

    preds, gts, anchor_ids, win_ids = [], [], [], []
    # also collect per-window per-anchor "ground-truth center" (the
    # per-(grid, anchor) mean) so we can compute deviations.
    centers = []   # for each masked cell, the per-(grid, anchor) mean
    t_ids = []     # trial index within the window (for residual analysis)

    for w_i, idx in enumerate(indices):
        X, _, _, _, _ = base_ds[int(idx)]
        # X is (T, M, F). We need per-(grid, anchor) mean over the 100
        # trials. The Dataset already loaded the full grid_id's data into
        # base_ds._per_grid[gi][0]; pull the per-(grid, anchor) mean from there.
        gi, s = base_ds.windows[int(idx)]
        feat, _, _ = base_ds._per_grid[gi]            # feat: (n_trials, M, F)
        # mean over trials of the F=0 (ToA) channel
        center = feat[:, :, 0].mean(axis=0)      # (M,)
        Xb = X.unsqueeze(0)
        torch.manual_seed(seed + int(idx))
        with torch.no_grad():
            X_masked, mask = random_mask(Xb, cfg.mask_ratio)
            x_pred, _ = model(X_masked)
        Xnp = X.squeeze(-1).numpy()              # (T, M)
        x_pred_np = x_pred.squeeze(0).squeeze(-1).numpy()
        mask_np = mask.squeeze(0).numpy()        # (T, M)
        T, M = Xnp.shape
        if mask_np.sum() == 0:
            continue
        masked_pred = x_pred_np[mask_np]
        masked_gt = Xnp[mask_np]
        ti, ai = np.where(mask_np)               # time_idx, anchor_idx (np.where returns (row, col))
        preds.append(masked_pred)
        gts.append(masked_gt)
        anchor_ids.append(ai)
        win_ids.append(np.full(ai.shape, w_i, dtype=int))
        centers.append(center[ai])
        t_ids.append(ti)

    return {
        "pred": np.concatenate(preds),
        "true": np.concatenate(gts),
        "anchor": np.concatenate(anchor_ids),
        "win": np.concatenate(win_ids),
        "center": np.concatenate(centers),
        "t": np.concatenate(t_ids),
    }


def shrink_slope(pred, true, center):
    """Slope of (pred - center) regressed on (true - center).

    1.0 = perfect tracking of fluctuations.
    0.0 = predict the mean.
    """
    dev_pred = pred - center
    dev_true = true - center
    if dev_true.std() < 1e-9:
        return float("nan")
    slope, _, _, _, _ = sp_stats.linregress(dev_true, dev_pred)
    return float(slope)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_windows", type=int, default=None,
                   help="cap on number of val windows to evaluate "
                        "(default: all)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt["cfg"].items() if k in Config.__dataclass_fields__})
    csv = args.csv or ckpt["csv"]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = SionnaGridWindowDataset(
        csv, seq_len=cfg.seq_len, n_channels=cfg.n_channels, stride=cfg.seq_len)

    # split by grid_id exactly the way train_tier0_sionna does
    n_grid = len(ds.grid_ids)
    n_train_grid = max(1, int(0.8 * n_grid))
    val_grid_ids = set(ds.grid_ids[n_train_grid:])
    val_indices = [i for i, (gi, _) in enumerate(ds.windows)
                    if ds.grid_ids[gi] in val_grid_ids]
    val_ds = torch.utils.data.Subset(ds, val_indices)
    print(f"val windows: {len(val_ds)}")

    model = SSLPretrainer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_windows = args.n_windows if args.n_windows is not None else len(val_ds)
    print(f"evaluating on {n_windows} windows...")
    out = evaluate(model, val_ds, cfg, n_windows=n_windows, seed=args.seed)
    pred, true, anchor, center = out["pred"], out["true"], out["anchor"], out["center"]
    err = pred - true

    overall_rmse = float(np.sqrt(np.mean(err ** 2)))
    overall_mae = float(np.mean(np.abs(err)))
    overall_bias = float(err.mean())
    # R^2 = 1 - SS_res / SS_tot
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    slope = shrink_slope(pred, true, center)
    # residual std at visible cells (approximation: noise std of ToA per
    # (grid, anchor) over 100 trials)
    base_ds = ds  # the full SionnaGridWindowDataset (not a Subset)
    visible_noise_std = float("nan")
    # we have the per-grid per-anchor ToA in base_ds._per_grid
    all_resid = []
    for gi in range(len(base_ds._per_grid)):
        item = base_ds._per_grid[gi]
        if item is None:
            continue
        feat, _, _ = item
        center_ga = feat[:, :, 0].mean(axis=0, keepdims=True)
        resid = feat[:, :, 0] - center_ga
        all_resid.append(resid.ravel())
    if all_resid:
        visible_noise_std = float(np.std(np.concatenate(all_resid)))
    # per-anchor RMSE
    per_anchor_rmse = [float(np.sqrt(np.mean(err[anchor == m] ** 2)))
                        for m in range(cfg.n_anchors)]
    per_anchor_bias = [float(err[anchor == m].mean())
                        for m in range(cfg.n_anchors)]
    # per-mask-distance bin RMSE
    dev = np.abs(true - center)
    bins = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0])
    bin_rmse = []
    bin_counts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (dev >= lo) & (dev < hi)
        if sel.sum() > 0:
            bin_rmse.append(float(np.sqrt(np.mean(err[sel] ** 2))))
            bin_counts.append(int(sel.sum()))
        else:
            bin_rmse.append(float("nan"))
            bin_counts.append(0)
    # correlation between predicted and true deviation
    dev_pred = pred - center
    dev_true = true - center
    corr_dev = float(np.corrcoef(dev_pred, dev_true)[0, 1]) if dev_true.std() > 0 else float("nan")

    # ---- print Markdown table ----
    print()
    print("=" * 64)
    print("  Tier 0 SSL — quantitative refill-quality metrics (val)")
    print("=" * 64)
    print(f"  windows evaluated:       {n_windows}")
    print(f"  total masked cells:      {len(pred)}")
    print(f"  overall RMSE:            {overall_rmse:.4f} m")
    print(f"  overall MAE:             {overall_mae:.4f} m")
    print(f"  overall bias:            {overall_bias:+.4f} m  (unbiased denoiser ≈ 0)")
    print(f"  R^2 vs. constant-mean:   {r2:.4f}")
    print(f"  shrink slope:            {slope:.4f}  (1.0 = perfect tracking, 0.0 = predict mean)")
    print(f"  corr(pred_dev, true_dev):{corr_dev:.4f}")
    print(f"  visible-cell noise std:  {visible_noise_std:.4f} m  (lower bound on residual std)")
    print()
    print("  per-anchor breakdown:")
    print(f"    {'anchor':<8} {'RMSE (m)':>10} {'bias (m)':>12}")
    for m in range(cfg.n_anchors):
        print(f"    A{m+1:<7} {per_anchor_rmse[m]:>10.4f} {per_anchor_bias[m]:>+12.4f}")
    print()
    print("  per-|true-mean-deviation| bin:")
    print(f"    {'|x - x_bar| bin (m)':<24} {'RMSE (m)':>10} {'count':>8}")
    for (lo, hi), r, c in zip(zip(bins[:-1], bins[1:]), bin_rmse, bin_counts):
        rstr = f"{r:.4f}" if not np.isnan(r) else "  n/a"
        print(f"    [{lo:.2f}, {hi:.2f}){' '*(24-14)}{rstr:>10} {c:>8}")

    # ---- save per-bin CSV ----
    csv_out = out_dir / "viz_C_metrics.csv"
    with open(csv_out, "w") as f:
        f.write("metric,value\n")
        f.write(f"overall_rmse_m,{overall_rmse}\n")
        f.write(f"overall_mae_m,{overall_mae}\n")
        f.write(f"overall_bias_m,{overall_bias}\n")
        f.write(f"r2_vs_mean,{r2}\n")
        f.write(f"shrink_slope,{slope}\n")
        f.write(f"corr_pred_dev_true_dev,{corr_dev}\n")
        f.write(f"visible_noise_std_m,{visible_noise_std}\n")
        f.write(f"n_masked_cells,{len(pred)}\n")
        for m in range(cfg.n_anchors):
            f.write(f"rmse_A{m+1}_m,{per_anchor_rmse[m]}\n")
            f.write(f"bias_A{m+1}_m,{per_anchor_bias[m]}\n")
        f.write("bin_lo_m,bin_hi_m,bin_rmse_m,bin_count\n")
        for (lo, hi), r, c in zip(zip(bins[:-1], bins[1:]), bin_rmse, bin_counts):
            rstr = "" if np.isnan(r) else str(r)
            f.write(f"{lo},{hi},{rstr},{c}\n")
    print()
    print(f"saved {csv_out}")

    # ---- 4-panel diagnostic figure ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    # (a) pred vs true scatter
    ax = axes[0, 0]
    ax.scatter(true, pred, s=3, alpha=0.3)
    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.plot([lo, hi], [center.mean() + 0, center.mean() + 0], "w", alpha=0)  # spacer
    ax.set_xlabel("true ToA (m)")
    ax.set_ylabel("predicted ToA (m)")
    ax.set_title(f"(a) pred vs true (RMSE = {overall_rmse:.3f} m)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (b) per-anchor RMSE
    ax = axes[0, 1]
    ax.bar(range(cfg.n_anchors), per_anchor_rmse, color="C0", alpha=0.8)
    ax.axhline(visible_noise_std, color="green", linestyle="--", alpha=0.6,
                label=f"visible noise std ({visible_noise_std:.3f} m)")
    ax.set_xticks(range(cfg.n_anchors))
    ax.set_xticklabels([f"A{m+1}" for m in range(cfg.n_anchors)])
    ax.set_ylabel("RMSE (m)")
    ax.set_title("(b) per-anchor RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) per-bin RMSE vs. |true - mean|
    ax = axes[1, 0]
    centers_bin = (bins[:-1] + bins[1:]) / 2
    ax.semilogx(centers_bin, bin_rmse, "-o", color="C1")
    ax.axhline(visible_noise_std, color="green", linestyle="--", alpha=0.6,
                label=f"visible noise std ({visible_noise_std:.3f} m)")
    ax.set_xlabel("|true ToA − per-(grid,anchor) mean|  (m)")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("(c) RMSE vs. deviation-from-mean bin")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    # (d) shrinkage plot: pred_dev vs true_dev
    ax = axes[1, 1]
    ax.scatter(dev_true, dev_pred, s=3, alpha=0.3)
    lim = max(np.abs(dev_true).max(), np.abs(dev_pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, label="y = x (slope=1)")
    # regression line
    if dev_true.std() > 0:
        s, b, *_ = sp_stats.linregress(dev_true, dev_pred)
        xx = np.linspace(-lim, lim, 100)
        ax.plot(xx, s * xx + b, "C1-", alpha=0.8,
                label=f"fit slope = {s:.3f}")
    ax.set_xlabel("true dev from per-(grid,anchor) mean (m)")
    ax.set_ylabel("predicted dev from per-(grid,anchor) mean (m)")
    ax.set_title(f"(d) shrinkage: pred_dev vs true_dev  (corr={corr_dev:.3f})")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Tier 0 SSL — quantitative refill diagnostics (val)", fontsize=11)
    fig.tight_layout()
    out_png = out_dir / "viz_C_metrics.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
