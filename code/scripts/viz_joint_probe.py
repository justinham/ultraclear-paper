"""Joint Tier 1+2 — Disentanglement diagnostic with curriculum overlay.

The Tier 1+2 counterpart to `viz_tier2_adv_accuracy.py`. Shows the
linear-probe env accuracy on the query set's c vectors, with the
lambda curriculum overlaid so you can see the relationship between
adversarial pressure and disentanglement quality.

Two side-by-side panels:
  Left:  probe_env_acc over episodes, with chance line and disentangle
         region shaded
  Right: lambda schedule over episodes (twin axis would be cramped,
         so we just plot lambda separately on the right)

Usage:
  python -m scripts.viz_joint_probe
  python -m scripts.viz_joint_probe --joint_dir checkpoints/joint
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scripts.viz_tier2_helpers import load_history_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--joint_dir", type=str, default="checkpoints/joint")
    p.add_argument("--out", type=str, default="figures/joint/probe_with_curriculum.png")
    args = p.parse_args()

    joint = Path(args.joint_dir)
    rows_C = load_history_csv(str(joint / "phase_C_history.csv"), kind="joint_C")
    if not rows_C:
        raise RuntimeError(f"no Phase C history at {joint / 'phase_C_history.csv'}")

    eps = [r["ep"] for r in rows_C]
    acc = [r["probe_env_acc"] for r in rows_C]
    lam = [r["lam"] for r in rows_C]
    chance = 0.5  # 2 unseen val envs

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: probe accuracy
    ax = axes[0]
    ax.plot(eps, acc, "o-", color="C2", label="linear probe (c → env_id)")
    ax.axhline(chance, color="grey", linestyle="--",
               label=f"chance = 1/2 (unseen val envs)")
    ax.axhline(1.0, color="red", linestyle=":", alpha=0.5, label="perfectly entangled")
    ax.fill_between(eps, chance, 1.0, alpha=0.05, color="green",
                    label="disentanglement region")
    # Mark curriculum transitions
    lam_arr = np.array(lam)
    transitions = np.where(np.diff(lam_arr) > 0)[0]
    for t in transitions:
        ax.axvline(eps[t + 1], color="C4", linestyle=":", alpha=0.4)
    ax.set_xlabel("episode (Phase C)")
    ax.set_ylabel("env_id classification accuracy (val envs only)")
    ax.set_title("Joint Tier 1+2 — disentanglement diagnostic\n"
                 "(probe on UNSEEN val envs; acc → chance = c is env-uninformative)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Right: lambda schedule (curriculum)
    ax = axes[1]
    ax.plot(eps, lam, "o-", color="C4", markersize=6)
    ax.fill_between(eps, 0, lam, alpha=0.15, color="C4")
    ax.set_xlabel("episode (Phase C)")
    ax.set_ylabel("λ (adversarial weight)")
    ax.set_title("Adversarial curriculum\n"
                 "warmup_episodes at λ=0, then linear ramp to λ=1.0")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    # Mark warmup end
    warmup_end_eps = [e for e, l in zip(eps, lam) if l > 0]
    if warmup_end_eps:
        ax.axvline(warmup_end_eps[0], color="C2", linestyle="--", alpha=0.5,
                   label=f"warmup end (ep {warmup_end_eps[0]})")
        ax.legend(fontsize=8)

    fig.suptitle("Joint Tier 1+2 — disentanglement vs curriculum", fontsize=14)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")

    # Summary stats
    valid_acc = [a for a in acc if not np.isnan(a)]
    if valid_acc:
        print(f"\nProbe accuracy summary over {len(valid_acc)} valid episodes:")
        print(f"  min  = {min(valid_acc):.3f}")
        print(f"  max  = {max(valid_acc):.3f}")
        print(f"  mean = {np.mean(valid_acc):.3f}  (chance = {chance})")
        print(f"  fraction at-or-below chance: "
              f"{sum(a <= chance + 0.05 for a in valid_acc) / len(valid_acc):.0%}")


if __name__ == "__main__":
    main()
