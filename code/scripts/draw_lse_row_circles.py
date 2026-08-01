"""Redraw lse_row{ROW}_circles.png — no title, bigger fonts.

The original (figures/lse_row0_circles.png) showed 23 projected 2D ToA
distance circles from a single row of the office_rack sionna track dataset.
The circles, LSE solve, and error are all derived from the *measured* ToA
(never the GT distance) so this script is fully self-contained and doesn't
rely on whatever `lse_est_x/y` was stored in the CSV by an earlier run.

Conventions (matching the original figure):
  - Circles are projected 2D horizontal radius: r = sqrt(d^2 - (z_tag - z_anchor)^2)
    where d = A_m_measured_dist_m, z_tag = 1.0.
  - A1 is the only LOS anchor in row 0 (light blue), the other 22 are NLOS (light red).
  - Ceiling anchors (z=2.85) are blue triangles; floor anchors (z=0.30) are red squares.
  - Green star = GT tag, magenta X = LSE prediction.

Run:
  python3.11 -m scripts.draw_lse_row_circles               # row 0, scene "big"
  python3.11 -m scripts.draw_lse_row_circles --row 100 --scene middle
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.lines import Line2D

DATA = Path("/Users/justin/ultraclear-paper/code/data/office_rack_track_sionna")
OUT = Path("/Users/justin/ultraclear-paper/figures")
SCENE_DEFAULT = "big"
ROW_DEFAULT = 0
TAG_Z = 1.0
N_ANCHORS = 23


# ----------------------------------------------------------------------- #
# LSE (2D) — identical math to scripts/lse_office_rack.py:lse_2d, but
# inlined here so the figure is self-contained and we don't import a
# different script. Uses MEASURED ToA, never gt_dist.
# ----------------------------------------------------------------------- #
def lse_2d(anchor_xyz: np.ndarray, distances: np.ndarray, ref_idx: int = 0):
    """Linearized 2D LSE. anchor_xyz: (N,3). distances: (N,) — projected 2D."""
    other = [i for i in range(anchor_xyz.shape[0]) if i != ref_idx]
    ai = anchor_xyz[other]
    ar = anchor_xyz[ref_idx]
    di = distances[other]
    dr = distances[ref_idx]
    A = 2.0 * (ar[None, :2] - ai[:, :2])                # (N-1, 2)
    b = (di ** 2 - dr ** 2
         - (ai[:, 0] ** 2 - ar[0] ** 2)
         - (ai[:, 1] ** 2 - ar[1] ** 2))                 # (N-1,)
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return p


# ----------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=SCENE_DEFAULT, choices=["big", "middle", "right"])
    parser.add_argument("--row", type=int, default=ROW_DEFAULT)
    args = parser.parse_args()
    scene, row_i = args.scene, args.row

    # ----- load data -----
    anchors = pd.read_csv(DATA / "anchor_positions.csv")
    df = pd.read_csv(DATA / f"office_rack_track_{scene}/sionna_uwb_track_office_rack.csv",
                     nrows=row_i + 1)
    row = df.iloc[row_i]
    print(f"scene={scene!r}  row={row_i}: tag at "
          f"x={row['tag_x']}, y={row['tag_y']}, z={row['tag_z']}")

    ax_xy = anchors[["X_m", "Y_m"]].values[:N_ANCHORS]
    ax_z = anchors["Z_m"].values[:N_ANCHORS]
    notes = anchors["notes"].values[:N_ANCHORS]
    anchor_ids = [f"An{i+1}" for i in range(N_ANCHORS)]

    # ----- measured ToA (NOT gt) -----
    meas = np.array([row[f"A{i+1}_measured_dist_m"] for i in range(N_ANCHORS)])
    los_flag = np.array([row[f"A{i+1}_los_nlos"] for i in range(N_ANCHORS)])
    is_los = (los_flag == 0)  # matches original figure's color convention

    # ----- 3D -> 2D projected horizontal radius -----
    dz = TAG_Z - ax_z
    r2 = meas ** 2 - dz ** 2
    r2 = np.where(r2 < 0, 0.0, r2)   # NLOS can over-shoot in z; clip
    r_2d = np.sqrt(r2)
    n_los = int(is_los.sum())
    n_nlos = N_ANCHORS - n_los

    # ----- LSE solve on measured ToA -----
    anchor_xyz = np.column_stack([ax_xy, ax_z])
    pred = lse_2d(anchor_xyz, r_2d, ref_idx=0)
    lse_x, lse_y = float(pred[0]), float(pred[1])
    gt_x, gt_y = float(row['tag_x']), float(row['tag_y'])
    lse_err = float(np.hypot(lse_x - gt_x, lse_y - gt_y))

    # ----- plot -----
    FS_LEGEND = 16
    FS_LABEL = 18
    FS_TICK = 14
    FS_ANNOT = 11

    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)

    los_color, nlos_color = "#7FB3D5", "#F1948A"
    los_alpha, nlos_alpha = 0.65, 0.50

    for i in range(N_ANCHORS):
        if is_los[i]:
            c = Circle(ax_xy[i], r_2d[i], fill=False,
                       edgecolor=los_color, linewidth=1.2, alpha=los_alpha, zorder=1)
        else:
            c = Circle(ax_xy[i], r_2d[i], fill=False,
                       edgecolor=nlos_color, linewidth=1.0, alpha=nlos_alpha, zorder=1)
        ax.add_patch(c)

    # anchor markers
    for i in range(N_ANCHORS):
        if notes[i] == "ceiling":
            m = RegularPolygon(ax_xy[i], numVertices=3, radius=0.45, orientation=0,
                               facecolor="#1F77B4", edgecolor="black",
                               linewidth=0.7, zorder=3)
        else:
            m = RegularPolygon(ax_xy[i], numVertices=4, radius=0.45, orientation=0,
                               facecolor="#D62728", edgecolor="black",
                               linewidth=0.7, zorder=3)
        ax.add_patch(m)
        ax.annotate(anchor_ids[i], ax_xy[i], xytext=(6, 6),
                    textcoords="offset points", fontsize=FS_ANNOT,
                    color="black", zorder=4)

    # GT tag (green star), LSE prediction (magenta X)
    ax.scatter([gt_x], [gt_y], marker="*", s=380, c="#2CA02C",
               edgecolor="black", linewidth=0.8, zorder=5)
    ax.scatter([lse_x], [lse_y], marker="X", s=240, c="#E377C2",
               edgecolor="black", linewidth=0.8, zorder=5)

    ax.set_xlabel("X (m)", fontsize=FS_LABEL)
    ax.set_ylabel("Y (m)", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-12, 12)

    legend_handles = [
        Line2D([0], [0], color=los_color, lw=2.4, alpha=los_alpha,
               label="LOS circle (projected 2D ToA)"),
        Line2D([0], [0], color=nlos_color, lw=2.4, alpha=nlos_alpha,
               label="NLOS circle (projected 2D ToA)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#1F77B4",
               markeredgecolor="black", markersize=13, label="ceiling anchor (z=2.85)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#D62728",
               markeredgecolor="black", markersize=12, label="floor anchor (z=0.30)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#2CA02C",
               markeredgecolor="black", markersize=18, label="GT tag (ground truth)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#E377C2",
               markeredgecolor="black", markersize=14, label="LSE prediction"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=FS_LEGEND,
              framealpha=0.92, edgecolor="black")

    # NO TITLE per user request
    out_path = OUT / f"lse_row{row_i}_circles.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    print(f"  meas-based LSE pred: ({lse_x:.2f}, {lse_y:.2f})")
    print(f"  GT:                  ({gt_x:.2f}, {gt_y:.2f})")
    print(f"  LSE error: {lse_err:.2f} m  ({n_nlos}/{N_ANCHORS} NLOS, {n_los} LOS)")


if __name__ == "__main__":
    main()
