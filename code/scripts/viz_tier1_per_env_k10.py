"""Per-env Tier 1 K=10 loc RMSE — TRAIN (envs 0-9) vs VAL (envs 10-20).

Reads `figures/tier1_per_env_k10.csv` and produces a 2-subplot figure
with the per-env loc RMSE for the 10 train envs and 10 val envs
(env 14 excluded — it's the missing-data env).

Usage:
    python -m scripts.viz_tier1_per_env_k10
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="figures/tier1_per_env_k10.csv")
    p.add_argument("--tier2_csv", default="figures/tier2_per_env.csv",
                    help="per-env Tier 2 zero-shot CSV (optional)")
    p.add_argument("--out_fig", default="figures/tier1_per_env_k10.png")
    args = p.parse_args()

    # Read Tier 1 CSV
    rows = {}
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            rows[int(r["env_id"])] = {
                "role": r["role"],
                "rmse": float(r["tier1_k10_rmse_m"]),
            }
    # Read Tier 2 CSV if exists
    t2_rows = {}
    if Path(args.tier2_csv).exists():
        with open(args.tier2_csv) as f:
            for r in csv.DictReader(f):
                t2_rows[int(r["env_id"])] = float(r["tier2_zero_shot_rmse_m"])

    # Train: envs 0-9, Val: envs 10-20 (skip 14)
    train_envs = [e for e in range(10) if e in rows]
    val_envs = [e for e in range(10, 21) if e != 14 and e in rows]

    train_vals_t1 = [rows[e]["rmse"] for e in train_envs]
    val_vals_t1 = [rows[e]["rmse"] for e in val_envs]
    train_vals_t2 = [t2_rows.get(e, float("nan")) for e in train_envs]
    val_vals_t2 = [t2_rows.get(e, float("nan")) for e in val_envs]

    # Load env coverage for NLOS annotation
    cov = {}
    cov_csv = Path("figures/sionna_env_coverage.csv")
    if cov_csv.exists():
        with open(cov_csv) as f:
            for r in csv.DictReader(f):
                cov[int(r["env_id"])] = r

    # Build 2-subplot figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # ---- Subplot 1: TRAIN envs ----
    ax = axes[0]
    nlos_train = [float(cov.get(e, {}).get("nlos_rate", 0)) * 100 for e in train_envs]
    colors_train = ["C2" if nl == 0 else ("C3" if nl < 5 else "C4") for nl in nlos_train]
    width = 0.4
    x_pos = np.arange(len(train_envs))
    bars1 = ax.bar(x_pos - width/2, train_vals_t1, width, color="C0",
                    edgecolor="white", label="Tier 1 K=10")
    if t2_rows:
        bars2 = ax.bar(x_pos + width/2, train_vals_t2, width, color="C1",
                        edgecolor="white", label="Tier 2 zero-shot", alpha=0.85)
        for b, v in zip(bars2, train_vals_t2):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                         ha="center", va="bottom", fontsize=7)
    for b, v in zip(bars1, train_vals_t1):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"e{e}" for e in train_envs], rotation=45, fontsize=9)
    ax.set_ylabel("loc RMSE (m)", fontsize=11)
    ax.set_title("TRAIN envs — Tier 1 K=10 vs Tier 2 zero-shot\n"
                  "(model has seen these during MAML training)",
                  fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ymax = max(max(train_vals_t1), max(train_vals_t2) if t2_rows else 0,
                max(val_vals_t1), max(val_vals_t2) if t2_rows else 0) * 1.15
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=9, loc="upper left")
    # Annotate area on top of each pair
    for xi, e in zip(x_pos, train_envs):
        if e in cov:
            area = float(cov[e]["area_m2"])
            nl = float(cov[e]["nlos_rate"]) * 100
            ax.text(xi, ymax * 0.95, f"{area:.0f}m² {nl:.0f}%NLOS",
                     ha="center", va="top", fontsize=6.5, color="gray",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    # ---- Subplot 2: VAL envs ----
    ax = axes[1]
    nlos_val = [float(cov.get(e, {}).get("nlos_rate", 0)) * 100 for e in val_envs]
    colors_val = ["C2" if nl == 0 else ("C3" if nl < 5 else "C4") for nl in nlos_val]
    x_pos = np.arange(len(val_envs))
    bars1 = ax.bar(x_pos - width/2, val_vals_t1, width, color="C0",
                    edgecolor="white", label="Tier 1 K=10")
    if t2_rows:
        bars2 = ax.bar(x_pos + width/2, val_vals_t2, width, color="C1",
                        edgecolor="white", label="Tier 2 zero-shot", alpha=0.85)
        for b, v in zip(bars2, val_vals_t2):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                         ha="center", va="bottom", fontsize=7)
    for b, v in zip(bars1, val_vals_t1):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"e{e}" for e in val_envs], rotation=45, fontsize=9)
    ax.set_ylabel("loc RMSE (m)", fontsize=11)
    ax.set_title("VAL envs — Tier 1 K=10 vs Tier 2 zero-shot\n"
                  "(model has NOT seen these — true OOD generalization)",
                  fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=9, loc="upper left")
    for xi, e in zip(x_pos, val_envs):
        if e in cov:
            area = float(cov[e]["area_m2"])
            nl = float(cov[e]["nlos_rate"]) * 100
            ax.text(xi, ymax * 0.95, f"{area:.0f}m² {nl:.0f}%NLOS",
                     ha="center", va="top", fontsize=6.5, color="gray",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    fig.suptitle("Per-env loc RMSE — Tier 1 K=10 (MAML) vs Tier 2 zero-shot (FiLM+adv) on 20 Sionna scenes",
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure → {args.out_fig}")
    print()
    print("=== Per-env summary ===")
    print(f"TRAIN: T1 mean={np.mean(train_vals_t1):.3f}m, max={max(train_vals_t1):.3f}m (env{train_envs[np.argmax(train_vals_t1)]}), min={min(train_vals_t1):.3f}m")
    print(f"VAL:   T1 mean={np.mean(val_vals_t1):.3f}m, max={max(val_vals_t1):.3f}m (env{val_envs[np.argmax(val_vals_t1)]}), min={min(val_vals_t1):.3f}m")
    if t2_rows:
        print(f"TRAIN: T2 mean={np.nanmean(train_vals_t2):.3f}m, max={np.nanmax(train_vals_t2):.3f}m")
        print(f"VAL:   T2 mean={np.nanmean(val_vals_t2):.3f}m, max={np.nanmax(val_vals_t2):.3f}m")


if __name__ == "__main__":
    main()
