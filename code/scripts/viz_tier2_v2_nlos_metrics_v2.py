"""Per-env NLOS classification viz v2 — uses class-imbalance-aware
metrics (AUROC, MCC, Brier) instead of raw accuracy.

Why this exists: half the envs in the Sionna dataset have 0% NLOS
(rooms with no occlusions), so raw accuracy is misleading — a
"always predict LOS" classifier gets 100% accuracy on those envs
and looks like the best model, when in fact it's the degenerate
classifier. AUROC and MCC are undefined on those envs (only one
class present), so we plot them as "N/A — no NLOS cells" and only
show the 12 envs that have NLOS.

Three panels:
  - Top: bar plot of AUROC per env (12 envs with NLOS), colored
    by role. 0.5 reference line = random; 1.0 = perfect ranking.
    p_actual_nlos overlaid as a line on a secondary y-axis.
  - Middle-left: bar plot of MCC per env. +1 = perfect, 0 = random,
    -1 = perfectly wrong. p_actual_nlos overlaid.
  - Middle-right: bar plot of Brier score per env. Lower = better
    calibration. 0 = perfect calibration; 0.25 = uniform 0.5
    predictions on balanced classes.
  - Bottom: per-anchor AUROC heatmap (rows = envs, cols = anchors)
    so we can see if certain anchors are systematically harder.

Usage:
    python3.11 -m scripts.viz_tier2_v2_nlos_metrics_v2 \
      --metrics_json checkpoints/tier2_sionna_v2_500/nlos_metrics_per_env.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_json", required=True,
                    help="output of scripts.eval_tier2_v2_nlos_metrics.py")
    p.add_argument("--out_fig", default=None)
    args = p.parse_args()

    with open(args.metrics_json) as f:
        data = json.load(f)
    per_env = {int(e): m for e, m in data["per_env"].items()}
    per_anchor = {int(e): m for e, m in data["per_anchor_per_env"].items()}
    train_scenes = set(data["train_scenes"])
    val_scenes = set(data["val_scenes"])

    # Envs with NLOS cells (where AUROC/MCC/etc are defined)
    nlos_envs = sorted([e for e, m in per_env.items() if m["auroc"] is not None])
    nlos_envs_train = [e for e in nlos_envs if e in train_scenes]
    nlos_envs_val = [e for e in nlos_envs if e in val_scenes]
    # All envs (including 0% NLOS) — used in the per-anchor heatmap
    all_envs = sorted(per_env.keys())

    role_color = {"train": "tab:blue", "val": "tab:orange"}
    env_role = {}
    for e in all_envs:
        if e in train_scenes:
            env_role[e] = "train"
        elif e in val_scenes:
            env_role[e] = "val"
        else:
            env_role[e] = "excl"

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.30, wspace=0.20)

    # --- Panel 1 (top, spans both columns): AUROC bar chart ---
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(nlos_envs))
    aurocs = [per_env[e]["auroc"] for e in nlos_envs]
    colors = [role_color[env_role[e]] for e in nlos_envs]
    bars = ax1.bar(x, aurocs, color=colors, alpha=0.75,
                    edgecolor="black", linewidth=0.5)
    # Mark the boundary between TRAIN and VAL within the nlos_envs list
    if nlos_envs_train and nlos_envs_val:
        n_train = len(nlos_envs_train)
        ax1.axvline(n_train - 0.5, color="black", linestyle="--",
                     alpha=0.5, linewidth=1.0)
    # Per-bar AUROC value annotation
    for i, (e, a) in enumerate(zip(nlos_envs, aurocs)):
        ax1.text(i, a + 0.01, f"{a:.3f}", ha="center", va="bottom",
                 fontsize=8)
    # p_actual_nlos overlay
    ax1b = ax1.twinx()
    p_actual = [per_env[e]["p_actual_nlos"] for e in nlos_envs]
    ax1b.plot(x, p_actual, color="purple", marker="^", markersize=8,
              alpha=0.7, linewidth=1.5, label="p_actual_nlos (right axis)")
    ax1b.set_ylim(0, max(0.45, max(p_actual) * 1.1))
    ax1b.set_ylabel("p_actual_nlos (NLOS fraction)", color="purple", fontsize=10)
    ax1b.tick_params(axis="y", labelcolor="purple", labelsize=9)
    # Reference lines
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.6,
                 label="AUROC=0.5 (random)")
    ax1.axhline(1.0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(nlos_envs, fontsize=9)
    ax1.set_xlabel("env_id (only envs with NLOS cells shown; "
                    f"n={len(nlos_envs)})")
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(0.5, 1.02)
    n_train = len(nlos_envs_train)
    n_val = len(nlos_envs_val)
    ax1.set_title(
        f"Per-env AUROC — Tier 2 v2 NLOS classification (unfrozen encoder)\n"
        f"only {len(nlos_envs)} envs have NLOS cells "
        f"({n_train} TRAIN, {n_val} VAL); 9 envs omitted (0% NLOS, AUROC undefined)",
        fontsize=11
    )
    legend_elements = [
        Patch(facecolor="tab:blue", alpha=0.75, label=f"TRAIN env (n={n_train})"),
        Patch(facecolor="tab:orange", alpha=0.75, label=f"VAL env (n={n_val})"),
        plt.Line2D([0], [0], color="purple", marker="^", linestyle="-",
                   label="p_actual_nlos (right axis)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    # --- Panel 2 (bottom-left): MCC ---
    ax2 = fig.add_subplot(gs[1, 0])
    mccs = [per_env[e]["mcc"] for e in nlos_envs]
    bars2 = ax2.bar(x, mccs, color=colors, alpha=0.75,
                     edgecolor="black", linewidth=0.5)
    if nlos_envs_train and nlos_envs_val:
        ax2.axvline(n_train - 0.5, color="black", linestyle="--",
                     alpha=0.5, linewidth=1.0)
    for i, (e, m) in enumerate(zip(nlos_envs, mccs)):
        ax2.text(i, m + 0.02, f"{m:+.2f}", ha="center", va="bottom",
                 fontsize=7)
    ax2.axhline(0.0, color="gray", linestyle=":", alpha=0.5,
                 label="MCC=0 (random)")
    ax2.axhline(1.0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(nlos_envs, fontsize=9)
    ax2.set_xlabel("env_id (envs with NLOS cells)")
    ax2.set_ylabel("Matthews correlation coefficient")
    ax2.set_ylim(-0.1, 1.05)
    mean_mcc_train = np.mean([per_env[e]["mcc"] for e in nlos_envs_train])
    mean_mcc_val = np.mean([per_env[e]["mcc"] for e in nlos_envs_val])
    ax2.set_title(
        f"Per-env MCC\n"
        f"mean TRAIN = {mean_mcc_train:+.3f} (n={n_train}), "
        f"mean VAL = {mean_mcc_val:+.3f} (n={n_val})",
        fontsize=10
    )
    # MCC subplot needs its own TRAIN/VAL legend since the AUROC
    # legend is only on the top panel
    mcc_legend = [
        Patch(facecolor="tab:blue", alpha=0.75, label=f"TRAIN (n={n_train})"),
        Patch(facecolor="tab:orange", alpha=0.75, label=f"VAL (n={n_val})"),
    ]
    ax2.legend(handles=mcc_legend, loc="upper left", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    # --- Panel 3 (bottom-right): Brier score (calibration) ---
    ax3 = fig.add_subplot(gs[1, 1])
    briers = [per_env[e]["brier"] for e in nlos_envs]
    bars3 = ax3.bar(x, briers, color=colors, alpha=0.75,
                     edgecolor="black", linewidth=0.5)
    if nlos_envs_train and nlos_envs_val:
        ax3.axvline(n_train - 0.5, color="black", linestyle="--",
                     alpha=0.5, linewidth=1.0)
    for i, (e, b) in enumerate(zip(nlos_envs, briers)):
        ax3.text(i, b + 0.003, f"{b:.3f}", ha="center", va="bottom",
                 fontsize=7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(nlos_envs, fontsize=9)
    ax3.set_xlabel("env_id (envs with NLOS cells)")
    ax3.set_ylabel("Brier score (lower = better calibration)")
    # Clip y-axis at 0.07 so env 4 (Brier=0.104) goes off-scale with
    # an explicit annotation, instead of compressing the other 11 envs
    # to invisibility
    ax3.set_ylim(0, 0.07)
    # Brier subplot also needs a TRAIN/VAL legend
    brier_legend = [
        Patch(facecolor="tab:blue", alpha=0.75, label=f"TRAIN (n={n_train})"),
        Patch(facecolor="tab:orange", alpha=0.75, label=f"VAL (n={n_val})"),
    ]
    ax3.legend(handles=brier_legend, loc="upper left", fontsize=8)
    # Show env 4's actual value as text inside the panel, with an
    # arrow pointing up to indicate the bar is off-scale
    if nlos_envs_val:
        env4_brier = per_env.get(4, {}).get("brier")
        if env4_brier is not None and env4_brier > 0.07:
            env4_idx = nlos_envs.index(4) if 4 in nlos_envs else None
            if env4_idx is not None:
                # Place annotation inside the panel near the top,
                # with an arrow pointing to the clipped env-4 bar
                ax3.annotate(f"env 4 Brier = {env4_brier:.3f}",
                              xy=(env4_idx, 0.065),
                              xytext=(env4_idx - 2.5, 0.055),
                              fontsize=8, color="red", ha="left",
                              arrowprops=dict(arrowstyle="->",
                                              color="red", lw=1.0))
    mean_brier_train = np.mean([per_env[e]["brier"] for e in nlos_envs_train])
    mean_brier_val = np.mean([per_env[e]["brier"] for e in nlos_envs_val])
    ax3.set_title(
        f"Per-env Brier score\n"
        f"mean TRAIN = {mean_brier_train:.3f}, mean VAL = {mean_brier_val:.3f}",
        fontsize=10
    )
    ax3.grid(True, axis="y", alpha=0.3)

    out_fig = args.out_fig or str(Path(args.metrics_json).parent / "tier2_v2_nlos_metrics.png")
    if not out_fig.startswith("/"):
        out_fig = str(Path("/Users/justin/ultraclear-paper") / out_fig)
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_fig}")


if __name__ == "__main__":
    main()
