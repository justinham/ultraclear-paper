"""Tier 2 — Headline numbers: location error comparison.

The paper's main Tier 2 result. Compares per-cond location error on
the val set (unseen envs) for:
  - baseline:   no Tier 2 (ASTA encoder + location head, supervised)
  - tier2:      full DisentangledModel + FiLM + location head

Reads `tier2_per_cond.csv` (produced by `train_tier2_synthetic.py`
when `--run_baseline` is passed).

Usage:
  python -m scripts.viz_tier2_loc_comparison
  python -m scripts.viz_tier2_loc_comparison --per_cond PATH/to/tier2_per_cond.csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per_cond", type=str, default="checkpoints/tier2_synthetic/tier2_per_cond.csv")
    p.add_argument("--out", type=str, default="figures/tier2_synthetic/loc_error_comparison.png")
    args = p.parse_args()

    rows = []
    with open(args.per_cond) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    conds = [int(r["cond_id"]) for r in rows]
    baseline = [float(r["baseline_loc_err"]) if r["baseline_loc_err"] != "NA" else float("nan")
                for r in rows]
    tier2 = [float(r["tier2_loc_err"]) for r in rows]

    # Micro-condition labels (matches data.py _micro_params)
    cond_labels = {0: "cond 0\nmostly LOS", 1: "cond 1\nmoderate NLOS", 2: "cond 2\nheavy NLOS"}

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(conds))
    width = 0.35
    ax.bar(x - width/2, baseline, width, label="baseline (no Tier 2)", color="grey", alpha=0.8)
    ax.bar(x + width/2, tier2, width, label="Tier 2 (DisentangledModel)", color="C0", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([cond_labels[c] for c in conds])
    ax.set_ylabel("mean Euclidean location error\n(val envs, lower = better)")
    ax.set_title("Tier 2 — location error on unseen environments")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate values + relative improvement
    for xi, b, t in zip(x, baseline, tier2):
        if not np.isnan(b) and b > 0:
            delta = (b - t) / b * 100
            label = f"{b:.2f}\n{t:.2f}\n({delta:+.1f}%)"
        else:
            label = f"NA\n{t:.2f}"
        ax.text(xi, max(b, t) + 0.05, label, ha="center", fontsize=8)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")

    # Also print a plain-text summary to stdout
    print("\nPer-cond location error summary (val envs, mean Euclidean):")
    print(f"  {'cond':<6}  {'baseline':>10}  {'tier2':>10}  {'Δ':>8}")
    for c, b, t in zip(conds, baseline, tier2):
        if not np.isnan(b):
            print(f"  {c:<6}  {b:>10.3f}  {t:>10.3f}  {(t-b)/b*100:>+7.1f}%")
        else:
            print(f"  {c:<6}  {'NA':>10}  {t:>10.3f}  {'NA':>8}")


if __name__ == "__main__":
    main()
