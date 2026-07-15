"""Side-by-side loss-decomposition comparison: w_rest=1.0 vs w_rest=0.1.

Reads two history CSVs (one per restore_weight setting) and produces
a single figure that overlays them so the loss-curve story is
visible at a glance. This is the figure that goes into the paper's
"ablation of restoration weight" subsection.

Usage:
    python3.11 -m scripts.viz_baseline_loss_compare \\
        --hist_a /tmp/hist_w1.csv --label_a "w_rest=1.0" \\
        --hist_b /tmp/hist_w01.csv --label_b "w_rest=0.1" \\
        --out figures/baseline_loss_compare.png
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: str) -> dict:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows.append(row)
    if not rows:
        sys.exit(f"empty history: {path}")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hist_a", required=True)
    p.add_argument("--label_a", default="w_rest=1.0")
    p.add_argument("--hist_b", required=True)
    p.add_argument("--label_b", default="w_rest=0.1")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Supervised baseline: restore_weight ablation")
    args = p.parse_args()

    a = load(args.hist_a)
    b = load(args.hist_b)
    ep_a, ep_b = a["epoch"], b["epoch"]
    final_a, final_b = a["val_loc_avg"][-1], b["val_loc_avg"][-1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"{args.title}\n"
        f"val avg loc err: {args.label_a} = {final_a:.2f} m, "
        f"{args.label_b} = {final_b:.2f} m "
        f"({(final_a - final_b):+.2f} m improvement)",
        fontsize=12,
    )

    # Common style: solid line for A, dashed for B
    sa, sb = "-", "--"
    ca, cb = "C0", "C1"

    # (1) Weighted loc loss (the one we care about)
    ax = axes[0, 0]
    ax.plot(ep_a, a["train_loc_w"], f"o{sa}", color=ca, label=args.label_a)
    ax.plot(ep_b, b["train_loc_w"], f"s{sb}", color=cb, label=args.label_b)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted loc loss (log)")
    ax.set_title("(1) Weighted loc loss \u2014 should be lower for the better run")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2) Weighted rest loss (the term we varied)
    ax = axes[0, 1]
    ax.plot(ep_a, a["train_rest_w"], f"o{sa}", color=ca, label=args.label_a)
    ax.plot(ep_b, b["train_rest_w"], f"s{sb}", color=cb, label=args.label_b)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted rest loss (log)")
    ax.set_title("(2) Weighted rest loss \u2014 this is what we changed")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (3) Per-cond val loc err
    ax = axes[1, 0]
    for d, sty, lab, col in [(a, sa, args.label_a, ca), (b, sb, args.label_b, cb)]:
        ax.plot(d["epoch"], d["val_loc_avg"], f"k{sty}", lw=2, label=f"{lab} (avg)")
        ax.plot(d["epoch"], d["val_loc_cond0"], f"g{sty}", alpha=0.6, label=f"{lab} low-NLOS")
        ax.plot(d["epoch"], d["val_loc_cond1"], f"y{sty}", alpha=0.6, label=f"{lab} mid-NLOS")
        ax.plot(d["epoch"], d["val_loc_cond2"], f"r{sty}", alpha=0.6, label=f"{lab} high-NLOS")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loc err (m)")
    ax.set_title("(3) Per-cond val loc err")
    # Combine legend entries (skip the 6 redundant ones)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    keep_h, keep_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            keep_h.append(h); keep_l.append(l); seen.add(l)
    ax.legend(keep_h, keep_l, loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (4) Loss share stacked bars (one set per run)
    ax = axes[1, 1]
    width = 0.35
    x_a = ep_a - 0.2
    x_b = ep_b + 0.2
    for x, d, sty, lab, col in [(x_a, a, sa, args.label_a, ca), (x_b, b, sb, args.label_b, cb)]:
        tot = d["train_loc_w"] + d["train_rest_w"] + d["train_nlos_w"]
        p_loc = d["train_loc_w"] / tot
        p_rest = d["train_rest_w"] / tot
        p_nlos = d["train_nlos_w"] / tot
        ax.bar(x, p_loc, width, color="steelblue", edgecolor="white")
        ax.bar(x, p_rest, width, bottom=p_loc, color="darkorange", edgecolor="white")
        ax.bar(x, p_nlos, width, bottom=p_loc + p_rest, color="seagreen", edgecolor="white")
    # Custom legend
    ax.bar([], [], color="steelblue", label="loc share")
    ax.bar([], [], color="darkorange", label="rest share")
    ax.bar([], [], color="seagreen", label="nlos share")
    ax.set_xlabel("epoch")
    ax.set_ylabel("share of weighted loss")
    ax.set_title("(4) Loss composition by run")
    ax.set_ylim(0, 1)
    # x-tick at integer epochs
    all_eps = sorted(set(list(ep_a) + list(ep_b)))
    ax.set_xticks(all_eps)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
