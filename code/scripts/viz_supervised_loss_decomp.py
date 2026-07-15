"""Plot loss decomposition for the supervised baseline trainer.

Reads the per-epoch history CSV produced by
`train_supervised_baseline.py --out_csv_history` and produces a
4-panel figure:

  1. Raw training losses (loc / rest / nlos) on log scale, one curve
     per component. Shows each task's own convergence behavior.
  2. Weighted training losses (component * weight). Shows what the
     optimizer actually minimizes — useful for diagnosing whether one
     term is dominating the gradient (the original 2026-07-15
     ablation: with restore_weight=1.0 the rest term outweighed loc).
  3. Per-cond validation loc error (low / mid / high NLOS bucket)
     + overall average, in meters.
  4. Stacked bar of the weighted-loss share at each epoch, so you
     can read off "epoch X: loc was 70%, rest was 20%, nlos was 10%".

Usage:
    python3.11 -m scripts.viz_supervised_loss_decomp \\
        --history_csv path/to/history.csv \\
        --out figures/baseline_loss_decomp.png
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_history(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history_csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Supervised baseline loss decomposition")
    args = p.parse_args()

    rows = load_history(args.history_csv)
    if not rows:
        sys.exit(f"empty history: {args.history_csv}")
    ep = np.array([r["epoch"] for r in rows])
    loc = np.array([r["train_loc"] for r in rows])
    rest = np.array([r["train_rest"] for r in rows])
    nlos = np.array([r["train_nlos"] for r in rows])
    loc_w = np.array([r.get("train_loc_w", r["train_loc"]) for r in rows])
    rest_w = np.array([r.get("train_rest_w", r["train_rest"]) for r in rows])
    nlos_w = np.array([r.get("train_nlos_w", r["train_nlos"]) for r in rows])
    val_avg = np.array([r["val_loc_avg"] for r in rows])
    val_low = np.array([r["val_loc_cond0"] for r in rows])
    val_mid = np.array([r["val_loc_cond1"] for r in rows])
    val_high = np.array([r["val_loc_cond2"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(args.title, fontsize=13)

    # (1) Raw losses, log scale
    ax = axes[0, 0]
    ax.plot(ep, loc, "o-", label="loc (MSE, norm units)", color="C0")
    ax.plot(ep, rest, "s-", label="rest (MSE, norm units)", color="C1")
    ax.plot(ep, nlos, "^-", label="nlos (BCE)", color="C2")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log)")
    ax.set_title("Raw training losses")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2) Weighted losses (what the optimizer sees)
    ax = axes[0, 1]
    ax.plot(ep, loc_w, "o-", label=f"loc × w_loc", color="C0")
    ax.plot(ep, rest_w, "s-", label=f"rest × w_rest", color="C1")
    ax.plot(ep, nlos_w, "^-", label=f"nlos × w_nlos", color="C2")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted loss (symlog)")
    ax.set_title("Weighted training losses (optimizer objective)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3) Per-cond val loc err
    ax = axes[1, 0]
    ax.plot(ep, val_low, "o-", label="low NLOS (<30%)", color="C2")
    ax.plot(ep, val_mid, "s-", label="mid NLOS (30-70%)", color="C1")
    ax.plot(ep, val_high, "^-", label="high NLOS (>70%)", color="C0")
    ax.plot(ep, val_avg, "k--", lw=1.5, label="average", alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loc error (m)")
    ax.set_title("Per-NLOS-bucket validation loc error")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (4) Stacked bar of weighted-loss share
    ax = axes[1, 1]
    total = loc_w + rest_w + nlos_w
    p_loc = loc_w / total
    p_rest = rest_w / total
    p_nlos = nlos_w / total
    ax.bar(ep, p_loc, color="C0", label="loc share")
    ax.bar(ep, p_rest, bottom=p_loc, color="C1", label="rest share")
    ax.bar(ep, p_nlos, bottom=p_loc + p_rest, color="C2", label="nlos share")
    ax.set_xlabel("epoch")
    ax.set_ylabel("share of total weighted loss")
    ax.set_title("Loss composition (weighted share)")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
