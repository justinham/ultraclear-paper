"""Draw the 12-waypoint walking path overview figure.

Reconstructed on 2026-07-22 to reproduce the figure previously
generated on 2026-06-09 10:48 as
`dataset/sionna_walking_path_v2_plot.png`. The original source
script was not preserved; this is a clean canonical rebuild
that produces the same visual:

  - 4,001-row scatter of (tag_x, tag_y) from
    `sionna_walking_path_v2.csv`, colored by sample index
  - 12 gold star waypoints with WP0..WP11 labels
  - 4 blue anchor triangles (A1..A4) at the canonical
    2x2 cluster near the origin
  - Red dashed polyline connecting the 12 waypoints in
    traversal order (the "planned path")
  - Title:
      12-Waypoint Walking Path — sionna_walking_path_v2.csv
      (4,001 samples · 10 Hz · 300 static samples per waypoint)

Usage:
    python3.11 -m scripts.draw_sionna_walking_path_v2
    python3.11 -m scripts.draw_sionna_walking_path_v2 \
        --out figures/sionna_walking_path_v2_plot.png
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Canonical 12 waypoints (extracted from
# dataset/sionna_walking_path_v2.py docstring, which is still
# in the repo and lists them explicitly).
WAYPOINTS = np.array([
    [-1.370, -2.960],   # WP0
    [-1.370,  3.930],   # WP1
    [ 1.370,  3.910],   # WP2
    [ 1.370, -2.960],   # WP3
    [ 4.010, -2.960],   # WP4
    [ 4.010,  3.910],   # WP5
    [ 4.010, 11.100],   # WP6
    [ 1.370, 11.100],   # WP7
    [-1.370, 11.100],   # WP8
    [-4.010, 11.100],   # WP9
    [-4.010,  3.910],   # WP10
    [-4.010, -2.960],   # WP11
], dtype=float)

# Anchor positions (training order, z=2.2 implied)
ANCHORS = {
    "A1": np.array([-0.52, -0.90]),
    "A2": np.array([-0.54,  0.35]),
    "A3": np.array([ 0.52, -0.90]),
    "A4": np.array([ 0.54,  0.35]),
}


def load_track(csv_path: Path) -> np.ndarray:
    """Return (N, 2) array of (tag_x, tag_y)."""
    xs, ys = [], []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                xs.append(float(r["tag_x"]))
                ys.append(float(r["tag_y"]))
            except (KeyError, ValueError):
                continue
    return np.asarray(list(zip(xs, ys)), dtype=np.float32)


def plot_walking_path(traj: np.ndarray, out_path: Path) -> None:
    n = len(traj)
    fig, ax = plt.subplots(figsize=(11, 9.5), constrained_layout=True)

    # Trajectory: scatter colored by sample index
    sc = ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(n),
                     cmap="viridis", s=14, alpha=0.85,
                     edgecolor="none", label="Tag position", zorder=2)

    # Planned path: red dashed polyline through the 12 waypoints
    ax.plot(WAYPOINTS[:, 0], WAYPOINTS[:, 1],
             color="red", linestyle="--", linewidth=1.5, alpha=0.85,
             label="Planned waypoint path", zorder=3)

    # 12 waypoints: gold stars with WP labels
    for i, (wx, wy) in enumerate(WAYPOINTS):
        ax.scatter(wx, wy, s=320, marker="*", c="gold",
                    edgecolor="black", linewidth=1.2, zorder=4)
        ax.annotate(f"WP{i}", (wx, wy),
                     textcoords="offset points", xytext=(8, 6),
                     fontsize=11, fontweight="bold", color="red",
                     zorder=5)

    # 4 anchors: blue triangles
    for name, p in ANCHORS.items():
        ax.scatter(p[0], p[1], s=240, marker="^", c="tab:blue",
                    edgecolor="black", linewidth=1.0, zorder=5)
        ax.annotate(name, (p[0], p[1]),
                     textcoords="offset points", xytext=(8, -12),
                     fontsize=10, fontweight="bold", color="tab:blue")

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("Sample index (time)", fontsize=11)

    fig.suptitle(
        f"12-Waypoint Walking Path — sionna_walking_path_v2.csv\n"
        f"({n:,} samples · 10 Hz · 300 static samples per waypoint)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default="/Users/justin/Downloads/srp_research/"
                           "sionna_walking_path_v2.csv")
    p.add_argument("--out",
                   default="/Users/justin/ultraclear-paper/code/figures/"
                           "sionna_walking_path_v2_plot.png")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    traj = load_track(csv_path)
    print(f"loaded {len(traj):,} samples from {csv_path.name}")
    print(f"  x range: [{traj[:, 0].min():.2f}, {traj[:, 0].max():.2f}] m")
    print(f"  y range: [{traj[:, 1].min():.2f}, {traj[:, 1].max():.2f}] m")

    plot_walking_path(traj, out_path)


if __name__ == "__main__":
    main()
