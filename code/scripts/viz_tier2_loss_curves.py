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
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scripts.viz_tier2_helpers import load_history_csv, TIER2_FIG, JOINT_FIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=str, default="checkpoints/tier2_synthetic/tier2_history.csv")
    p.add_argument("--joint_history", type=str, default="checkpoints/joint/phase_C_history.csv")
    p.add_argument("--out", type=str, default=f"{TIER2_FIG}/loss_curves.png")
    p.add_argument("--joint", action="store_true",
                   help="use joint phase C history instead of standalone Tier 2")
    args = p.parse_args()

    if args.joint:
        path = args.joint_history
        if "tier2_synthetic" in args.out:
            args.out = args.out.replace("tier2_synthetic", "joint")
    else:
        path = args.history

    rows = load_history_csv(path, kind="joint_C" if args.joint else "tier2")
    eps = [r["ep"] for r in rows]
    train_task = [r["task_mse"] for r in rows]
    val_task = [r.get("val_task_mse", float("nan")) for r in rows]
    train_adv = [r["adv_ce"] for r in rows]
    train_nlos = [r["nlos_ce"] for r in rows]
    train_cond = [r["cond_ce"] for r in rows]
    task_disent = [r.get("task_disent", float("nan")) for r in rows]
    feat_loss = [r.get("feat_loss", float("nan")) for r in rows]
    probe_acc = [r["probe_env_acc"] for r in rows]
    lam = [r["lam"] for r in rows]
    chance_adv = np.log(8)   # ln(8) = 2.08
    chance_cond = np.log(3)  # ln(3) = 1.10

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    ax = axes[0, 0]
    ax.plot(eps, train_task, "o-", label="task MSE" + (" (MAML query)" if args.joint else ""), color="C0")
    if not args.joint:
        ax.plot(eps, val_task, "s-", label="val task MSE (OOD envs)", color="C1")
    else:
        ax.plot(eps, task_disent, "^-", label="disent task MSE (on query)", color="C1", alpha=0.7)
    ax.set_xlabel("epoch" if not args.joint else "episode")
    ax.set_ylabel("location MSE")
    ax.set_title("Task loss (location)" + (" -- joint" if args.joint else ""))
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(eps, train_adv, "o-", label="adv CE", color="C2")
    ax.axhline(chance_adv, color="grey", linestyle="--", label=f"chance = ln(8) = {chance_adv:.2f}")
    ax.set_xlabel("epoch" if not args.joint else "episode")
    ax.set_ylabel("env_id cross-entropy")
    ax.set_title("Adversary CE (down = env-info leaked)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 2]
    if args.joint:
        ax.plot(eps, feat_loss, "o-", color="C4", label="feat_loss (sum)")
    ax.plot(eps, train_cond, "o-", color="C5", label="cond CE")
    ax.axhline(chance_cond, color="grey", linestyle="--", label=f"chance = ln(3) = {chance_cond:.2f}")
    ax.set_xlabel("episode" if args.joint else "epoch")
    ax.set_ylabel("loss")
    ax.set_title("Feat-side losses" + (" (joint)" if args.joint else ""))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(eps, train_nlos, "o-", color="C3")
    ax.set_xlabel("epoch" if not args.joint else "episode")
    ax.set_ylabel("per-cell BCE")
    ax.set_title("NLOS head loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(eps, probe_acc, "o-", color="C2", label="probe c -> env acc")
    if args.joint:
        ax.axhline(0.5, color="grey", linestyle="--", label="chance (2 val envs)")
    ax.set_xlabel("episode" if args.joint else "epoch")
    ax.set_ylabel("linear probe env acc")
    ax.set_title("Disentanglement diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(eps, lam, "o-", color="C4")
    ax.set_xlabel("epoch" if not args.joint else "episode")
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
