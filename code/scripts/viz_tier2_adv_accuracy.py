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
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scripts.viz_tier2_helpers import load_history_csv, TIER2_CKPT, JOINT_C_CKPT, TIER2_FIG, JOINT_FIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=str, default=f"{TIER2_CKPT.rsplit('/', 1)[0]}/tier2_history.csv",
                   help="standalone Tier 2 history CSV (default: tier2_synthetic/tier2_history.csv)")
    p.add_argument("--joint_history", type=str, default=f"{JOINT_C_CKPT.rsplit('/', 1)[0]}/phase_C_history.csv",
                   help="joint phase C history CSV (default: joint/phase_C_history.csv)")
    p.add_argument("--out", type=str, default=f"{TIER2_FIG}/adv_accuracy.png")
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
    eps = [int(r["ep"]) for r in rows]
    acc = [r["probe_env_acc"] for r in rows]
    # Chance: tier2 stores it per-row; joint_C doesn't compute it inline.
    # For joint, probe is on the query set (n_envs=2 val envs in our setup)
    # so chance is 0.5. For tier2, use the stored chance.
    if args.joint:
        chance = 0.5
    else:
        # tier2 stores probe_chance per row
        import csv as _csv
        with open(path) as f:
            r0 = next(_csv.DictReader(f))
        chance = float(r0["probe_chance"]) if r0.get("probe_chance") not in (None, "", "nan") else 0.5

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
