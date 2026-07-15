"""Plot FOMAML training + val trajectory for the tier 1 MAML run.

Reads the history CSV produced by `train_tier1_sionna.py
--out_history` and produces a 3-panel figure:
  (1) train loc err (RMSE in z-scored units) per episode
  (2) val loc err (meters) per episode, one curve per held-out env
  (3) per-env val loc err at the last checkpoint, as a bar chart
      (so the headline numbers per env are easy to read)

Usage:
    python3.11 -m scripts.viz_tier1_history \\
        --history_csv figures/tier1_maml_1000ep_history.csv \\
        --out figures/tier1_maml_1000ep.png \\
        --title "Tier 1 FOMAML on real Sionna, 1000 episodes"
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_history(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                row["episode"] = int(row["episode"])
                row["value"] = float(row["value"])
                if row.get("env_id"):
                    row["env_id"] = int(row["env_id"])
                if row.get("k"):
                    row["k"] = int(row["k"])
            except (TypeError, ValueError):
                pass
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history_csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="Tier 1 FOMAML training trajectory")
    args = p.parse_args()

    rows = load_history(args.history_csv)
    if not rows:
        sys.exit(f"empty history: {args.history_csv}")

    # Group rows by phase
    train_eps = []
    train_vals = []
    val_eps_to_env: dict[int, dict[int, float]] = defaultdict(dict)
    for r in rows:
        if r["phase"] == "train_loc_err_z_rmse":
            train_eps.append(r["episode"])
            train_vals.append(r["value"])
        elif r["phase"] == "val_loc_err_m":
            val_eps_to_env[r["episode"]][r["env_id"]] = r["value"]

    # Final per-env val (last checkpoint)
    last_ep = max(val_eps_to_env.keys())
    final_per_env = val_eps_to_env[last_ep]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(args.title, fontsize=13)

    # (1) Train trajectory
    ax = axes[0]
    ax.plot(train_eps, train_vals, color="C0", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("episode")
    ax.set_ylabel("train loc err (RMSE, z-scored)")
    ax.set_title(f"(1) FOMAML training loss ({len(train_eps)} episodes)")
    ax.grid(True, alpha=0.3)

    # (2) Val per env
    ax = axes[1]
    sorted_env_ids = sorted(final_per_env.keys())
    for env_id in sorted_env_ids:
        eps, vals = [], []
        for ep, env_dict in val_eps_to_env.items():
            if env_id in env_dict:
                eps.append(ep)
                vals.append(env_dict[env_id])
        ax.plot(eps, vals, marker="o", markersize=4, lw=1.2,
                label=f"env{env_id} (final {vals[-1]:.2f}m)")
    ax.set_xlabel("episode")
    ax.set_ylabel("val loc err (m)")
    ax.set_title(f"(2) Per-env val loc err at K={next(iter(rows)).get('k', '?')}")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (3) Final per-env bar
    ax = axes[2]
    env_labels = [f"env{e}" for e in sorted_env_ids]
    env_vals = [final_per_env[e] for e in sorted_env_ids]
    colors = ["C2" if e in (10, 11, 13, 20) else "C1" for e in sorted_env_ids]
    ax.bar(env_labels, env_vals, color=colors, edgecolor="white")
    ax.set_ylabel(f"val loc err at ep{last_ep} (m)")
    ax.set_title(f"(3) Final per-env val loc err")
    ax.grid(True, alpha=0.3, axis="y")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    # Annotate the mean
    mean_val = float(np.mean(env_vals))
    ax.axhline(mean_val, color="k", linestyle="--", lw=1, alpha=0.5)
    ax.text(0.02, 0.95, f"mean = {mean_val:.2f} m",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")
    print(f"final per-env val loc err (m): {final_per_env}")
    print(f"mean across val envs: {mean_val:.3f} m")


if __name__ == "__main__":
    main()
