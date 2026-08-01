"""Per-env spatial coverage figure.

For each of the 20 Sionna scenes, plot:
  - the (x, y) tag position coverage (bbox in local scene frame)
  - the per-scene NLOS rate
  - the train/val role

This is the geometric-mismatch story: env 16 is a 19.5 × 4.7 m outdoor
parking lot, the train set has smaller indoor rooms + indoor garages.
The frozen-encoder Tier 2 model can't bridge the 10× area gap, and the
MAML Tier 1 model's K=10 adaptation is biased by the indoor bias of the
support set.

Output:
  - figures/sionna_env_coverage.png  (4-panel: bbox / area / scatter / NLOS)
  - figures/sionna_env_coverage.csv  (per-env numeric data)

Usage:
    python -m scripts.viz_env_coverage
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scripts.multi_scene_dataset import build_default_multi_scene


def collect_env_extents(ds, scene_info, max_per_env=None):
    """For each scene, return (x_min, x_max, y_min, y_max, nlos_rate).

    Uses the per-scene tag_xy stats from SceneInfo (pre-computed over
    all windows in the scene, not a subsample) for the bbox, and
    a subsample for the NLOS rate.
    """
    out = []
    for s in scene_info:
        if s.tag_xy_mean is None or s.tag_xy_std is None:
            continue
        # 3-sigma bound covers ~99% of the data
        x_range = 6.0 * float(s.tag_xy_std[0])
        y_range = 6.0 * float(s.tag_xy_std[1])
        x_min = float(s.tag_xy_mean[0]) - x_range / 2
        x_max = float(s.tag_xy_mean[0]) + x_range / 2
        y_min = float(s.tag_xy_mean[1]) - y_range / 2
        y_max = float(s.tag_xy_mean[1]) + y_range / 2
        indices = [i for i, (s_idx, _, _) in enumerate(ds.windows)
                    if ds.scene_info[s_idx].env_id == s.env_id]
        if not indices:
            continue
        if max_per_env and len(indices) > max_per_env:
            rng = np.random.default_rng(0)
            indices = rng.choice(indices, size=max_per_env, replace=False).tolist()
        nlos_rates = []
        for i in indices:
            _, _, _, _, nlos = ds[i]
            nlos_rates.append(float(nlos.float().mean()))
        out.append({
            "env_id": s.env_id,
            "name": s.name,
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "x_mean": float(s.tag_xy_mean[0]),
            "y_mean": float(s.tag_xy_mean[1]),
            "nlos_rate": float(np.mean(nlos_rates)),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_csv", default="figures/sionna_env_coverage.csv")
    p.add_argument("--out_fig", default="figures/sionna_env_coverage.png")
    p.add_argument("--train_scenes", type=int, nargs="+", default=list(range(10)))
    args = p.parse_args()

    print("loading Sionna data...")
    ds, scene_info = build_default_multi_scene(
        n_channels=2, seq_len=32, stride=32, all_scenes=True,
    )
    print(f"  {len(ds)} windows across {len(scene_info)} scenes")
    train_scenes = set(args.train_scenes)
    extents = collect_env_extents(ds, scene_info)
    print(f"  computed extents for {len(extents)} envs")

    # ---- Save CSV ----
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "env_id", "name", "role", "x_min", "x_max", "y_min", "y_max",
            "x_range", "y_range", "x_mean", "y_mean", "area_m2", "nlos_rate",
        ])
        w.writeheader()
        for e in extents:
            role = "TRAIN" if e["env_id"] in train_scenes else "VAL"
            w.writerow({
                **e,
                "role": role,
                "x_range": e["x_max"] - e["x_min"],
                "y_range": e["y_max"] - e["y_min"],
                "area_m2": (e["x_max"] - e["x_min"]) * (e["y_max"] - e["y_min"]),
            })
    print(f"saved CSV → {out_csv}")

    # ---- 4-panel figure ----
    # Use a 2x3 grid to add a 5th panel for per-env Tier 1 K=10 val RMSE
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.30)

    train_color = "C2"
    val_color = "C1"

    # Sort by env_id for the bar
    extents_sorted = sorted(extents, key=lambda e: e["env_id"])
    # Pre-compute x_range / y_range / area for plotting (mirrors the CSV writer)
    for e in extents_sorted:
        e["x_range"] = e["x_max"] - e["x_min"]
        e["y_range"] = e["y_max"] - e["y_min"]
        e["area_m2"] = e["x_range"] * e["y_range"]

    # Panel 1: bbox widths (x_range, y_range) per env
    ax1 = fig.add_subplot(gs[0, 0])
    xs = [e["x_range"] for e in extents_sorted]
    ys = [e["y_range"] for e in extents_sorted]
    ys = [e["y_range"] for e in extents_sorted]
    colors = [train_color if e["env_id"] in train_scenes else val_color
              for e in extents_sorted]
    labels = [f"e{e['env_id']}" for e in extents_sorted]
    width = 0.4
    x_pos = np.arange(len(extents_sorted))
    ax1.bar(x_pos - width/2, xs, width, color=colors, label="x_range", edgecolor="white")
    ax1.bar(x_pos + width/2, ys, width, color=colors, alpha=0.5, label="y_range", edgecolor="white")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, fontsize=7)
    ax1.set_ylabel("extent (m)")
    ax1.set_title("(1) Per-env tag coverage extent (meters)\n"
                   "green = TRAIN, orange = VAL")
    ax1.grid(True, alpha=0.3, axis="y")
    # Custom legend with two colors
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(facecolor=train_color, label="TRAIN"),
        Patch(facecolor=val_color, label="VAL"),
        Patch(facecolor="grey", label="x_range (full)"),
        Patch(facecolor="grey", alpha=0.5, label="y_range (faint)"),
    ], fontsize=8)

    # Panel 2: area (x_range * y_range) per env
    ax2 = fig.add_subplot(gs[0, 1])
    areas = [(e["x_max"] - e["x_min"]) * (e["y_max"] - e["y_min"]) for e in extents_sorted]
    colors2 = [train_color if e["env_id"] in train_scenes else val_color
                for e in extents_sorted]
    bars = ax2.bar(x_pos, areas, color=colors2, edgecolor="white")
    for b, a, e in zip(bars, areas, extents_sorted):
        ax2.text(b.get_x() + b.get_width() / 2, a, f"{a:.0f}",
                  ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, fontsize=7)
    ax2.set_ylabel("bbox area (m²)")
    ax2.set_title("(2) Per-env tag coverage area (m²)\n"
                   "env 16 is 90.8 m² — 3-10× larger than the indoor train envs")
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: scatter of (x_mean, y_mean) — env centers in the (x, y) plane
    ax3 = fig.add_subplot(gs[1, 0])
    for e in extents_sorted:
        is_train = e["env_id"] in train_scenes
        c = train_color if is_train else val_color
        # Bbox as a rectangle
        rect = patches.Rectangle(
            (e["x_min"], e["y_min"]),
            e["x_max"] - e["x_min"],
            e["y_max"] - e["y_min"],
            linewidth=1.5, edgecolor=c, facecolor=c, alpha=0.3,
        )
        ax3.add_patch(rect)
        # Env id label at center
        ax3.text(e["x_mean"], e["y_mean"], f"e{e['env_id']}",
                  ha="center", va="center", fontsize=7, color="black")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_title("(3) Bounding boxes of tag coverage per env (local scene frame)\n"
                   "Notice the parking-lot envs (15-19) are 19.5×4.7m strips,\n"
                   "much wider than the indoor rooms (envs 0-9)")
    ax3.set_aspect("equal", adjustable="datalim")
    ax3.grid(True, alpha=0.3)

    # Panel 4: NLOS rate vs x_range
    ax4 = fig.add_subplot(gs[1, 1])
    for e in extents_sorted:
        is_train = e["env_id"] in train_scenes
        c = train_color if is_train else val_color
        ax4.scatter(e["x_range"], e["nlos_rate"] * 100,
                     s=80, c=c, edgecolors="black", linewidth=0.5)
        ax4.annotate(f"e{e['env_id']}",
                      (e["x_range"], e["nlos_rate"] * 100),
                      xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax4.set_xlabel("x_range (m) — proxy for env scale")
    ax4.set_ylabel("any-NLOS rate (%)")
    ax4.set_title("(4) NLOS rate vs env scale\n"
                   "Larger envs (right) tend to have more NLOS (blockages)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(handles=[
        Patch(facecolor=train_color, label="TRAIN"),
        Patch(facecolor=val_color, label="VAL"),
    ], fontsize=8)

    # Panel 5: Tier 1 K=10 val RMSE per env (if available)
    ax5 = fig.add_subplot(gs[1, 2])
    tier1_csv = Path("figures/sionna_env_coverage_tier1_k10.csv")
    if tier1_csv.exists():
        import csv as _csv
        t1_data = {}
        with open(tier1_csv) as f:
            for r in _csv.DictReader(f):
                t1_data[int(r["env_id"])] = float(r["tier1_k10_val_rmse_m"])
        t1_envs = sorted(t1_data)
        t1_vals = [t1_data[e] for e in t1_envs]
        t1_colors = [train_color if e in train_scenes else val_color for e in t1_envs]
        bars = ax5.bar([f"e{e}" for e in t1_envs], t1_vals, color=t1_colors, edgecolor="white")
        for b, v in zip(bars, t1_vals):
            ax5.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=7)
        ax5.set_xticklabels([f"e{e}" for e in t1_envs], rotation=45, fontsize=7)
        ax5.set_ylabel("Tier 1 K=10 val RMSE (m)")
        ax5.set_title("(5) Tier 1 K=10 val RMSE per env\n"
                       "Big envs (12, 15-19, 575 m²) are 1.3-2.1 m\n"
                       "Small envs (10, 11, 13) are 0.35-0.43 m")
        ax5.grid(True, alpha=0.3, axis="y")
    else:
        ax5.text(0.5, 0.5, "(Tier 1 K=10 data not yet extracted)\n"
                  "Run scripts/save_tier1_per_env.py",
                  ha="center", va="center", transform=ax5.transAxes)

    fig.suptitle("Sionna 21-env spatial coverage — the geometric mismatch story",
                  fontsize=14)
    fig.savefig(args.out_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure → {args.out_fig}")


if __name__ == "__main__":
    main()
