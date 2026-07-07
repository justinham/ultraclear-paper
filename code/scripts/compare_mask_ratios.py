"""Side-by-side comparison: 25% vs 15% random-mask SSL runs.

Loads both checkpoints and produces:
  - compare_loss_curves.png  : train/val MSE per epoch, both runs on the
                               same axes (linear + log y)
  - compare_metrics.csv      : one row per metric per mask-ratio

Usage:
    python3.11 -m scripts.compare_mask_ratios \
        --ckpt_25 /tmp/.../tier0_sionna.pt \
        --ckpt_15 /tmp/.../tier0_sionna.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer, random_mask
from scripts.train_tier0_sionna import SionnaGridWindowDataset
from scripts.viz_tier0_metrics import evaluate, shrink_slope


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_25", required=True, help="25% mask checkpoint")
    p.add_argument("--ckpt_15", required=True, help="15% mask checkpoint")
    p.add_argument("--csv", default=None,
                   help="CSV to evaluate on (default: ckpt_25's CSV)")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt_25).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- load both checkpoints -----
    runs = []
    for tag, path in [("25pct", args.ckpt_25), ("15pct", args.ckpt_15)]:
        ckpt = torch.load(path, weights_only=False)
        cfg = Config(**{k: v for k, v in ckpt["cfg"].items()
                        if k in Config.__dataclass_fields__})
        runs.append({"tag": tag, "ckpt": ckpt, "cfg": cfg})
    csv = args.csv or runs[0]["ckpt"]["csv"]

    # ----- side-by-side loss curves -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = {"25pct": "C0", "15pct": "C1"}
    for r in runs:
        tag = r["tag"]
        c = colors[tag]
        ep = np.arange(1, len(r["ckpt"]["train_losses"]) + 1)
        tr = r["ckpt"]["train_losses"]
        va = np.array(r["ckpt"]["val_losses"])
        ax1.plot(ep, tr, "-", color=c, alpha=0.85, label=f"train ({tag})")
        ax1.plot(ep, np.where(np.isnan(va), np.nan, va),
                 "--", color=c, alpha=0.85, label=f"val ({tag})")
        ax2.semilogy(ep, tr, "-", color=c, alpha=0.85, label=f"train ({tag})")
        ax2.semilogy(ep, np.where(np.isnan(va), np.nan, va),
                     "--", color=c, alpha=0.85, label=f"val ({tag})")
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    ax1.set_ylabel("MSE on masked cells (m²)")
    ax1.set_title("25% vs 15% mask — linear")
    ax2.set_title("25% vs 15% mask — log y")
    ax2.set_ylabel("MSE (log)")
    fig.suptitle("Tier 0 SSL on Sionna empty_open: mask-ratio comparison",
                 fontsize=11)
    fig.tight_layout()
    out1 = out_dir / "compare_loss_curves.png"
    fig.savefig(out1, dpi=140)
    plt.close(fig)
    print(f"saved {out1}")

    # ----- quantitative comparison on val -----
    ds = SionnaGridWindowDataset(
        csv, seq_len=runs[0]["cfg"].seq_len,
        n_channels=runs[0]["cfg"].n_channels,
        stride=runs[0]["cfg"].seq_len)
    n_grid = len(ds.grid_ids)
    n_train_grid = max(1, int(0.8 * n_grid))
    val_grid_ids = set(ds.grid_ids[n_train_grid:])
    val_indices = [i for i, (gi, _) in enumerate(ds.windows)
                    if ds.grid_ids[gi] in val_grid_ids]
    val_ds = torch.utils.data.Subset(ds, val_indices)

    rows = []
    for r in runs:
        cfg = r["cfg"]
        model = SSLPretrainer(cfg)
        model.load_state_dict(r["ckpt"]["model_state"])
        model.eval()
        out = evaluate(model, val_ds, cfg, n_windows=len(val_ds), seed=0)
        pred, true, anchor, center = out["pred"], out["true"], out["anchor"], out["center"]
        err = pred - true
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        bias = float(err.mean())
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        slope = shrink_slope(pred, true, center)
        dev = pred - center
        dev_true = true - center
        corr = float(np.corrcoef(dev, dev_true)[0, 1]) if dev_true.std() > 0 else float("nan")
        rows.append({
            "mask_ratio": cfg.mask_ratio,
            "n_masked_cells": len(pred),
            "rmse_m": rmse,
            "mae_m": mae,
            "bias_m": bias,
            "r2": r2,
            "shrink_slope": slope,
            "corr_pred_dev_true_dev": corr,
            "train_mse_final": r["ckpt"]["train_losses"][-1],
            "val_mse_final": r["ckpt"]["val_losses"][-1],
        })

    csv_out = out_dir / "compare_metrics.csv"
    cols = list(rows[0].keys())
    with open(csv_out, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")
    print(f"saved {csv_out}")

    # ----- print summary table -----
    print()
    print("=" * 78)
    print(f"  {'metric':<28} {'25% mask':>14} {'15% mask':>14}")
    print("=" * 78)
    cols_show = [
        ("mask_ratio",                "{:.2f}",      "mask_ratio"),
        ("n_masked_cells",            "{:>14d}",     "n_masked_cells"),
        ("RMSE on masked cells (m)",  "{:.4f}",      "rmse_m"),
        ("MAE on masked cells (m)",   "{:.4f}",      "mae_m"),
        ("Bias (m)",                  "{:+.4f}",     "bias_m"),
        ("R^2 vs. constant-mean",     "{:.4f}",      "r2"),
        ("Shrink slope",              "{:.4f}",      "shrink_slope"),
        ("Corr(pred_dev, true_dev)",  "{:.4f}",      "corr_pred_dev_true_dev"),
        ("Final train MSE",           "{:.4f}",      "train_mse_final"),
        ("Final val MSE",             "{:.4f}",      "val_mse_final"),
    ]
    r25 = rows[1] if rows[0]["mask_ratio"] == 0.15 else rows[0]
    r15 = rows[0] if rows[0]["mask_ratio"] == 0.15 else rows[1]
    for label, fmt, key in cols_show:
        v25 = fmt.format(r25[key]) if r25[key] is not None else "n/a"
        v15 = fmt.format(r15[key]) if r15[key] is not None else "n/a"
        print(f"  {label:<28} {v25:>14} {v15:>14}")
    print("=" * 78)
    print(f"  baseline (per-(window, anchor)-mean of visible cells): RMSE ≈ 0.051 m")
    print(f"  oracle (per-(grid, anchor)-mean over 100 trials):    RMSE ≈ 0.050 m")


if __name__ == "__main__":
    main()
