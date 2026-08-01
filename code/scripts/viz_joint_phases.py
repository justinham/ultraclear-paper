"""Joint Tier 1+2 — All-phase loss curve overlay.

The headline figure for the joint training pipeline. Shows the loss
trajectory of the entire SSL -> MAML -> disentangle -> joint finetune
narrative in one figure, with phase boundaries and curriculum markers
so you can see which phase does what.

Six panels:
  - Top: SSL pretraining loss (Phase A pre-MAML) and MAML query loss
         (Phase A's MAML episodes, encoder frozen)
  - Middle: Phase B disentanglement losses (task MSE, adv CE, cond CE,
            nlos CE) with lambda schedule overlay
  - Bottom: Phase C joint training losses (meta_loss, task_disent,
            adv CE, cond CE, feat_loss) with lambda schedule overlay

The point of this figure is the narrative: SSL sets up the encoder,
MAML calibrates the location head, Phase B disentangles c with
encoder frozen, Phase C ties it all together with the encoder
unfrozen. Each phase's lambda schedule is overlaid to make the
curriculum explicit.

Usage:
  python -m scripts.viz_joint_phases
  python -m scripts.viz_joint_phases --joint_dir checkpoints/joint
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scripts.viz_tier2_helpers import load_history_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--joint_dir", type=str, default="checkpoints/joint")
    p.add_argument("--out", type=str, default="figures/joint/all_phases.png")
    args = p.parse_args()

    joint = Path(args.joint_dir)

    # Phase B (epoch-based) and Phase C (episode-based) have different
    # CSV formats. Load both via the helper.
    rows_B = []
    if (joint / "phase_B_history.csv").exists():
        rows_B = load_history_csv(str(joint / "phase_B_history.csv"), kind="joint_B")
    rows_C = []
    if (joint / "phase_C_history.csv").exists():
        rows_C = load_history_csv(str(joint / "phase_C_history.csv"), kind="joint_C")

    # We don't have a Phase A history CSV in the current implementation
    # (the MAML losses are in the .pt checkpoint). Show a placeholder
    # noting where Phase A would be.
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 1, hspace=0.35, height_ratios=[0.7, 1.5, 1.8])

    # ---- Top: Phase A placeholder ----
    ax = fig.add_subplot(gs[0, 0])
    ax.text(0.5, 0.5,
            "Phase A: SSL pretraining + MAML warmup (encoder frozen)\n"
            "(per-episode query loss is logged to phase_A.pt, not a CSV;\n"
            "see `python -m scripts.train_tier12_joint` for the run log)",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="grey"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Joint Tier 1+2 training — full pipeline (all phases)",
                 fontsize=14, pad=10)

    # ---- Middle: Phase B ----
    ax = fig.add_subplot(gs[1, 0])
    if rows_B:
        eps = [r["ep"] for r in rows_B]
        ax.plot(eps, [r["task_mse"] for r in rows_B], "o-", color="C0",
                label="task MSE", markersize=4)
        ax.plot(eps, [r["adv_ce"] for r in rows_B], "s-", color="C2",
                label="adv CE", markersize=4)
        ax.plot(eps, [r["cond_ce"] for r in rows_B], "^-", color="C5",
                label="cond CE", markersize=4)
        ax.axhline(np.log(8), color="C2", linestyle="--", alpha=0.4, label="chance ln(8)")
        ax.axhline(np.log(3), color="C5", linestyle="--", alpha=0.4, label="chance ln(3)")
        # Lambda schedule on twin axis
        ax2 = ax.twinx()
        ax2.plot(eps, [r["lam"] for r in rows_B], "D-", color="C4", alpha=0.6,
                 label="λ (curriculum)", markersize=4)
        ax2.set_ylabel("λ (adversarial weight)", color="C4")
        ax2.set_ylim(-0.05, 1.1)
        ax2.tick_params(axis="y", labelcolor="C4")
        ax.set_xlabel("epoch (Phase B)")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        ax.set_title("Phase B: disentangle with frozen encoder (curriculum: 0 → ramp → 1.0)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    else:
        ax.text(0.5, 0.5, "no Phase B history found",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

    # ---- Bottom: Phase C ----
    ax = fig.add_subplot(gs[2, 0])
    if rows_C:
        eps = [r["ep"] for r in rows_C]
        # The helper maps meta_loss -> task_mse and task_loss_disent -> task_disent.
        ax.plot(eps, [r["task_mse"] for r in rows_C], "o-", color="C0",
                label="MAML meta_loss", markersize=4)
        ax.plot(eps, [r["task_disent"] for r in rows_C], "s-", color="C1",
                label="disent task MSE", markersize=4)
        ax.plot(eps, [r["feat_loss"] for r in rows_C], "^-", color="C4",
                label="feat_loss (sum)", markersize=4)
        ax.set_xlabel("episode (Phase C)")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        # Twin axis for probe_env_acc
        ax2 = ax.twinx()
        ax2.plot(eps, [r["probe_env_acc"] for r in rows_C], "D-", color="C2",
                 label="probe c→env acc", markersize=4)
        ax2.axhline(0.5, color="C2", linestyle="--", alpha=0.4, label="chance (2 val envs)")
        ax2.set_ylabel("linear probe acc", color="C2")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor="C2")
        ax.set_title("Phase C: joint FOMAML + disentangle (encoder unfrozen, "
                     "MAML query loss + disentangle losses + curriculum)")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no Phase C history found",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
