"""Draw the ground-truth track from one sionna_uwb_track_*_A1A4swapped.csv file.

Shows:
  - The 4 anchor positions (red triangles)
  - The 12 waypoints (gold stars, labeled wp0..wp11)
  - The actual tag path (tag_x, tag_y), color = seg_type
    (dwell = blue, transition = orange)
  - Anchor-to-waypoint connector lines so the path geometry
    is obvious

Default input: sionna_uwb_track_empty_open_A1A4swapped.csv
(pristine LOS, cleanest geometry for a first look).

Usage:
    python3.11 -m scripts.draw_track2_gt
    python3.11 -m scripts.draw_track2_gt --csv data/sionna_track2_swapped/sionna_uwb_track_indoor_blockages_pillars_A1A4swapped.csv
"""
from __future__ import annotations
import argparse
import csv
from collections import OrderedDict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Anchor positions in training order (z=2.2, matches the
# run_asta_ou_best.py script and the ASTA models trained on
# the track2 data). A1=anc4, A2=anc2, A3=anc3, A4=anc1 in
# the original metadata.
ANCHOR_POS = np.array([
    [-0.52, -0.90, 2.2],
    [-0.54,  0.35, 2.2],
    [ 0.52, -0.90, 2.2],
    [ 0.54,  0.35, 2.2],
])


def load_track(csv_path: Path) -> dict:
    """Read a sionna_uwb_track_*_A1A4swapped.csv file.

    Returns dict with arrays: tag_x, tag_y, seg_type, seg_index,
    seg_progress, wp_x, wp_y, and the list of 12 waypoint positions.
    """
    rows = []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                rows.append({
                    "tag_x": float(r["tag_x"]),
                    "tag_y": float(r["tag_y"]),
                    "seg_type": r["seg_type"],
                    "seg_index": int(r["seg_index"]),
                    "seg_progress": float(r["seg_progress"]),
                    "wp_x": float(r["wp_x"]),
                    "wp_y": float(r["wp_y"]),
                })
            except (ValueError, KeyError):
                continue
    out = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            out[k].append(v)
    for k in out:
        if k in ("seg_type",):
            continue
        out[k] = np.asarray(out[k])

    # Canonical 12 waypoints (dedup)
    wps = OrderedDict()
    for wx, wy in zip(out["wp_x"], out["wp_y"]):
        wps.setdefault((float(wx), float(wy)), None)
    out["waypoints"] = np.asarray(list(wps.keys()), dtype=float)
    out["n_rows"] = len(out["tag_x"])
    return out


def plot_track(d: dict, csv_path: Path, out_path: Path,
               dwell_only: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    tag_x, tag_y = d["tag_x"], d["tag_y"]
    seg_type = np.array(d["seg_type"])
    if dwell_only:
        mask = seg_type == "dwell"
        tag_x, tag_y = tag_x[mask], tag_y[mask]
        seg_type = seg_type[mask]
        d["n_rows"] = int(mask.sum())
        print(f"dwell_only: kept {d['n_rows']} dwell rows "
              f"(dropped {(~mask).sum()} transition rows)")
    wp_x, wp_y = d["wp_x"], d["wp_y"]
    waypoints = d["waypoints"]

    # --- Tag path: color by seg_type ---
    # Build an integer "phase" so we can color the two segment types
    phase = np.where(seg_type == "dwell", 0, 1)
    # Draw the full path as a thin gray line for context
    ax.plot(tag_x, tag_y, color="lightgray", lw=0.8, zorder=1)
    # Then overlay the colored points (dwell = blue, transition = orange)
    is_dwell = seg_type == "dwell"
    is_trans = seg_type == "transition"
    # Dwell points are at the EXACT waypoint position, so they sit
    # on top of the gold star. Make them visible by drawing them BIG
    # and on top (zorder > waypoint stars at 4).
    if is_dwell.any():
        ax.scatter(tag_x[is_dwell], tag_y[is_dwell],
                    c="tab:blue", s=120, alpha=0.45,
                    edgecolor="navy", lw=0.8,
                    label="dwell (100 samples per waypoint)", zorder=7)
    if (not dwell_only) and is_trans.any():
        ax.scatter(tag_x[is_trans], tag_y[is_trans],
                    c="tab:orange", s=8, alpha=0.6,
                    label="transition", zorder=2)

    # --- 12 waypoint gold stars, labeled wp0..wp11 ---
    # Order them by their first-occurrence index in the file
    seen = OrderedDict()
    for wx, wy in zip(wp_x, wp_y):
        seen.setdefault((float(wx), float(wy)), len(seen))
    for (wx, wy), idx in seen.items():
        ax.scatter(wx, wy, s=380, marker="*", c="gold",
                    edgecolor="black", linewidth=1.2, zorder=4)
        ax.annotate(f"wp{idx}", (wx, wy),
                     textcoords="offset points", xytext=(8, 6),
                     fontsize=11, fontweight="bold", zorder=5)

    # --- 4 anchor red triangles ---
    for i, (ax_x, ax_y, _) in enumerate(ANCHOR_POS):
        ax.scatter(ax_x, ax_y, s=300, marker="^", c="red",
                    edgecolor="black", linewidths=1.5, zorder=5)
        ax.annotate(f"A{i+1}", (ax_x, ax_y),
                     textcoords="offset points", xytext=(10, 8),
                     fontsize=11, fontweight="bold", zorder=6)

    # --- The wp_index of each row's CURRENT waypoint target ---
    # (the wp it is dwelling at OR transitioning to)
    # wp_x/wp_y are the TARGET waypoint for that row. Color the
    # target waypoint arrows so the transition endpoints are obvious.
    # Just draw thin arrows from each waypoint to its neighbors in
    # seg_index order. (cheaper than overplotting 1800 arrows.)
    seen_target = OrderedDict()
    for wx, wy, st, si in zip(wp_x, wp_y, seg_type, d["seg_index"]):
        seen_target.setdefault((float(wx), float(wy), int(si)), None)
    # draw a faint guide rectangle through the 4 waypoint x-positions
    for y in np.unique(waypoints[:, 1]):
        xs = waypoints[waypoints[:, 1] == y, 0]
        ax.plot([xs.min(), xs.max()], [y, y], color="gold",
                 lw=0.6, alpha=0.4, zorder=1)
    for x in np.unique(waypoints[:, 0]):
        ys = waypoints[waypoints[:, 0] == 1] if False else waypoints[:, 1]
        ax.plot([x, x], [ys.min(), ys.max()], color="gold",
                 lw=0.6, alpha=0.4, zorder=1)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_title(
        f"GT track from {csv_path.name}\n"
        f"{d['n_rows']} samples"
        f"{' (dwell only)' if dwell_only else ', 12 waypoints (4×3 grid), '}"
        f"\n4 anchors (red ▲), "
        f"{int(is_dwell.sum())} dwell"
        f"{'' if dwell_only else f' + {int(is_trans.sum())} transition'}"
    )
    ax.legend(loc="upper left", fontsize=10)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default="/Users/justin/ultraclear-paper/code/data/"
                           "sionna_track2_swapped/"
                           "sionna_uwb_track_empty_open_A1A4swapped.csv")
    p.add_argument("--dwell_only", action="store_true",
                   help="only plot the dwell rows (drop transitions)")
    p.add_argument("--out",
                   default="/Users/justin/ultraclear-paper/code/figures/"
                           "track2_gt_empty_open.png")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = load_track(csv_path)
    print(f"loaded {d['n_rows']} rows from {csv_path.name}")
    print(f"  12 waypoints at: {d['waypoints'].tolist()}")
    print(f"  seg_type counts: dwell={int(sum(s=='dwell' for s in d['seg_type']))}, "
           f"transition={int(sum(s=='transition' for s in d['seg_type']))}")

    plot_track(d, csv_path, out_path, dwell_only=args.dwell_only)


if __name__ == "__main__":
    main()
