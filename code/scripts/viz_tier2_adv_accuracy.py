"""Tier 2 — Disentanglement diagnostic: linear-probe env accuracy over training.

The headline Tier 2 figure. The probe is a 1-layer linear classifier on
c trained to predict env_id. If c is uninformative about env (the
disentanglement goal), the probe should be at chance.

This script plots probe_env_acc vs epoch. It overlays the chance line
(1/n_envs) and a "perfectly entangled" reference (1.0).

If the curve trends DOWN toward chance, Tier 2 is working.

Usage:
  python -m scripts.viz_tier2_adv_accuracy
  python -m scripts.viz_tier2_adv_accuracy --history PATH/to/tier2_history.csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=str, default="checkpoints/tier2_synthetic/tier2_history.csv")
    p.add_argument("--out", type=str, default="figures/tier2_synthetic/adv_accuracy.png")
    args = p.parse_args()

    rows = []
    with open(args.history) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    eps = [int(r["epoch"]) for r in rows]
    acc = [r["probe_env_acc"] for r in rows]
    chance = rows[0]["probe_chance"]    # per-row, but should be constant

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(eps, acc, "o-", color="C2", label="linear probe (c → env_id)")
    ax.axhline(chance, color="grey", linestyle="--",
               label=f"chance = 1/{int(1/chance)} (unseen envs)")
    ax.axhline(1.0, color="red", linestyle=":", alpha=0.5, label="perfectly entangled (c = env)")
    ax.fill_between(eps, chance, 1.0, alpha=0.05, color="green", label="disentanglement region")
    ax.set_xlabel("epoch")
    ax.set_ylabel("env_id classification accuracy (val envs only)")
    ax.set_title("Tier 2 — disentanglement diagnostic\n"
                 "(probe on UNSEEN val envs; acc → chance = c is env-uninformative)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
