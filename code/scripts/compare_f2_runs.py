"""Compare two F=2 SSL runs with different channel_weights.

Loads both checkpoints and produces:
  - compare_loss_curves.png  : train/val loss per epoch, both runs on
                               the same axes (linear + log y).
                               NB: the loss is *weighted*, so absolute
                               numbers aren't directly comparable, but
                               the *trend* (convergence speed, val/train
                               gap) is.
  - compare_metrics.csv      : per-channel per-run metrics
  - print a Markdown table to stdout

Usage:
    python3.11 -m scripts.compare_f2_runs \
        --ckpt_a <50-50 ckpt> --name_a "1.0 1.0" \
        --ckpt_b <3-1 ckpt>   --name_b "3.0 1.0" \
        --csv <csv>
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
from scripts.train_tier0_sionna_f2 import SionnaGridWindowF2Dataset, CHANNEL_NAMES
from scripts.viz_tier0_metrics_f2 import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_a", required=True)
    p.add_argument("--name_a", required=True, help="display name for run A")
    p.add_argument("--ckpt_b", required=True)
    p.add_argument("--name_b", required=True, help="display name for run B")
    p.add_argument("--csv", default=None)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt_a).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load both ----
    runs = []
    for tag, name, path in [("A", args.name_a, args.ckpt_a),
                             ("B", args.name_b, args.ckpt_b)]:
        ckpt = torch.load(path, weights_only=False)
        cfg = Config(**{k: v for k, v in ckpt["cfg"].items()
                        if k in Config.__dataclass_fields__})
        runs.append({"tag": tag, "name": name, "ckpt": ckpt, "cfg": cfg})
    csv = args.csv or runs[0]["ckpt"]["csv"]

    # ---- loss curves side by side ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = {"A": "C0", "B": "C1"}
    for r in runs:
        tag = r["tag"]
        c = colors[tag]
        ep = np.arange(1, len(r["ckpt"]["train_losses"]) + 1)
        tr = r["ckpt"]["train_losses"]
        va = np.array(r["ckpt"]["val_losses"])
        ax1.plot(ep, tr, "-", color=c, alpha=0.85, label=f"train ({r['name']})")
        ax1.plot(ep, np.where(np.isnan(va), np.nan, va),
                 "--", color=c, alpha=0.85, label=f"val ({r['name']})")
        ax2.semilogy(ep, tr, "-", color=c, alpha=0.85, label=f"train ({r['name']})")
        ax2.semilogy(ep, np.where(np.isnan(va), np.nan, va),
                     "--", color=c, alpha=0.85, label=f"val ({r['name']})")
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    ax1.set_ylabel("Weighted MSE on masked cells")
    ax1.set_title("Linear y")
    ax2.set_title("Log y (loss values not directly comparable across weights)")
    ax2.set_ylabel("Weighted MSE (log)")
    fig.suptitle("F=2 SSL — channel-weights comparison", fontsize=11)
    fig.tight_layout()
    out1 = out_dir / "compare_loss_curves_f2.png"
    fig.savefig(out1, dpi=140)
    plt.close(fig)
    print(f"saved {out1}")

    # ---- metrics on val ----
    full_ds = SionnaGridWindowF2Dataset(
        csv, seq_len=runs[0]["cfg"].seq_len, stride=runs[0]["cfg"].seq_len)
    n_grid = len(full_ds.grid_ids)
    n_train_grid = max(1, int(0.8 * n_grid))
    train_grid_ids = set(full_ds.grid_ids[:n_train_grid])
    full_ds.compute_norm_stats(train_grid_ids)

    rows = []
    for r in runs:
        cfg = r["cfg"]
        # use the ckpt's own norm stats
        full_ds.norm_mean = r["ckpt"]["norm_mean"]
        full_ds.norm_std = r["ckpt"]["norm_std"]
        model = SSLPretrainer(cfg)
        model.load_state_dict(r["ckpt"]["model_state"])
        model.eval()
        res = evaluate(model, full_ds, cfg, full_ds.norm_mean, full_ds.norm_std,
                        n_windows=None, seed=0)
        for f in range(2):
            pred = res["pred"][f]
            true = res["true"][f]
            anchor = res["anchor"][f]
            center = res["center"][f]
            err = pred - true
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            bias = float(err.mean())
            ss_res = np.sum(err ** 2)
            ss_tot = np.sum((true - true.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            per_a_rmse = [float(np.sqrt(np.mean(err[anchor == m] ** 2)))
                           for m in range(cfg.n_anchors)]
            rows.append({
                "run_tag": r["tag"],
                "run_name": r["name"],
                "channel": f,
                "channel_name": CHANNEL_NAMES[f],
                "units": ["m", "deg"][f],
                "n_masked": len(pred),
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "r2": r2,
                "rmse_A1": per_a_rmse[0],
                "rmse_A2": per_a_rmse[1],
                "rmse_A3": per_a_rmse[2],
                "rmse_A4": per_a_rmse[3],
            })

    csv_out = out_dir / "compare_metrics_f2.csv"
    cols = list(rows[0].keys())
    with open(csv_out, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")
    print(f"saved {csv_out}")

    # ---- print summary table ----
    print()
    print("=" * 78)
    print(f"  F=2 SSL — channel-weights comparison (val metrics)")
    print("=" * 78)
    print(f"  {'channel':<14} {'metric':<22} {args.name_a:>20} {args.name_b:>20}")
    print("-" * 78)
    for f in range(2):
        ra = next(r for r in rows if r["run_tag"] == "A" and r["channel"] == f)
        rb = next(r for r in rows if r["run_tag"] == "B" and r["channel"] == f)
        print(f"  {CHANNEL_NAMES[f]:<14} {'RMSE':<22} "
              f"{ra['rmse']:>17.4f} {ra['units']:<3} "
              f"{rb['rmse']:>17.4f} {rb['units']:<3}")
        print(f"  {'':<14} {'MAE':<22} "
              f"{ra['mae']:>17.4f} {ra['units']:<3} "
              f"{rb['mae']:>17.4f} {rb['units']:<3}")
        print(f"  {'':<14} {'Bias':<22} "
              f"{ra['bias']:>+17.4f} {ra['units']:<3} "
              f"{rb['bias']:>+17.4f} {rb['units']:<3}")
        print(f"  {'':<14} {'R^2':<22} {ra['r2']:>20.4f} {rb['r2']:>20.4f}")
        print("-" * 78)


if __name__ == "__main__":
    main()
