"""Tier 2 — Loss curves over training.

Plots 4 subplots from `tier2_history.csv`:
  - train_task_mse + val_task_mse
  - train_adv_ce (with chance baseline)
  - train_nlos_ce
  - lambda schedule (sanity check that warmup → ramp → 1.0 worked)

Usage:
  python -m scripts.viz_tier2_loss_curves
  python -m scripts.viz_tier2_loss_curves --history PATH/to/tier2_history.csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=str, default="checkpoints/tier2_synthetic/tier2_history.csv")
    p.add_argument("--out", type=str, default="figures/tier2_synthetic/loss_curves.png")
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
    eps = [r["epoch"] for r in rows]
    train_task = [r["train_task_mse"] for r in rows]
    val_task = [r["val_task_mse"] for r in rows]
    train_adv = [r["train_adv_ce"] for r in rows]
    train_nlos = [r["train_nlos_ce"] for r in rows]
    train_cond = [r.get("train_cond_ce", float("nan")) for r in rows]
    lam = [r["lambda"] for r in rows]
    chance_adv = np.log(8)   # ln(8) = 2.08
    chance_cond = np.log(3)  # ln(3) = 1.10

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    ax = axes[0, 0]
    ax.plot(eps, train_task, "o-", label="train task MSE", color="C0")
    ax.plot(eps, val_task, "s-", label="val task MSE (OOD envs)", color="C1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("location MSE")
    ax.set_title("Task loss (location)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(eps, train_adv, "o-", label="train adv CE", color="C2")
    ax.axhline(chance_adv, color="grey", linestyle="--", label=f"chance = ln(8) = {chance_adv:.2f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("env_id cross-entropy")
    ax.set_title("Adversary CE (down = env-info leaked)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(eps, train_nlos, "o-", color="C3")
    ax.set_xlabel("epoch")
    ax.set_ylabel("per-cell BCE")
    ax.set_title("NLOS head loss (down = micro-signal learned)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(eps, train_cond, "o-", color="C5")
    ax.axhline(chance_cond, color="grey", linestyle="--", label=f"chance = ln(3) = {chance_cond:.2f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cond_id cross-entropy")
    ax.set_title("Cond-classifier CE on c (down = c is cond-informative)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 2]
    ax.plot(eps, lam, "o-", color="C4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("adv_lambda")
    ax.set_title("Adversarial schedule")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Tier 2 — adversarial training dynamics", fontsize=14)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
