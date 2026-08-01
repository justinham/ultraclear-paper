"""Single-panel plot of the first 30 samples of uwbx4_aligned_2hz.csv.

Uses the car-frame horizontal plane (px_car, pz_car) and drops the
elevation (py_car). Filters to the first 30 valid (non-empty
px_car/py_car/pz_car) rows.

Output: code/figures/uwbx4_arkit_path_2d_first30.png
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_first_n_valid(csv_path: Path, n: int = 30):
    """Read the aligned CSV; return the first N rows where px_car,
    py_car, and pz_car are all non-empty (in CSV order, not
    sorted by time). Also returns the 4 anchor positions in the
    car frame (anc{1..4}_x_car, anc{1..4}_y_car)."""
    rows = []
    anchors = None
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if anchors is None:
                # Anchors are constant per row; grab from the first
                # row that has them. Z is excluded (top-down view).
                try:
                    anchors = np.array([
                        [float(r["anc1_x_car"]), float(r["anc1_y_car"])],
                        [float(r["anc2_x_car"]), float(r["anc2_y_car"])],
                        [float(r["anc3_x_car"]), float(r["anc3_y_car"])],
                        [float(r["anc4_x_car"]), float(r["anc4_y_car"])],
                    ])
                except (KeyError, ValueError):
                    anchors = None
            if not (r.get("px_car") and r.get("py_car") and r.get("pz_car")):
                continue
            try:
                rows.append({
                    "t": r["t_grid"],
                    "px": float(r["px_car"]),
                    "py": float(r["py_car"]),
                    "pz": float(r["pz_car"]),
                })
            except (ValueError, KeyError):
                continue
            if len(rows) >= n:
                break
    return rows, anchors


def plot_first_n(rows: list, anchors: np.ndarray, csv_path: Path,
                out_path: Path) -> None:
    """Plot (px_car, py_car) — the car-frame horizontal plane.

    Convention (from eval_uwbx4_vs_arkit.py):
        px_car = forward (m)
        py_car = lateral (m)
        pz_car = elevation (m)  -- excluded from this top-down view
    """
    n = len(rows)
    px = np.array([r["px"] for r in rows])
    py = np.array([r["py"] for r in rows])
    pz = np.array([r["pz"] for r in rows])
    t_secs = np.arange(n) * 0.5   # 2 Hz, relative to first sample

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    sc = ax.scatter(px, py, c=t_secs, cmap="viridis", s=80,
                     edgecolor="k", lw=0.6, zorder=3)
    # Connecting line for path visualization
    ax.plot(px, py, color="lightgray", lw=1, alpha=0.6, zorder=1)
    # Start / end markers
    ax.plot(px[0], py[0], "go", ms=14, label="start", zorder=4)
    ax.plot(px[-1], py[-1], "r*", ms=20, label="end", zorder=4)
    # 4 anchor triangles (z=0, all at car frame floor)
    if anchors is not None:
        for i, (ax_x, ax_y) in enumerate(anchors):
            ax.scatter(ax_x, ax_y, s=120, marker="^", c="red",
                        edgecolor="black", linewidth=1.0, zorder=5)
            ax.annotate(f"A{i+1}", (ax_x, ax_y),
                         textcoords="offset points", xytext=(8, 6),
                         fontsize=11, fontweight="bold")

    ax.set_xlabel("px_car (m, forward)", fontsize=12)
    ax.set_ylabel("py_car (m, lateral)", fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=11)
    ax.set_title(
        f"uwbx4 ARKit-labeled path (top-down, horizontal plane)\n"
        f"first {n} valid samples from {csv_path.name} "
        f"(2 Hz, pz_car (elevation) excluded)"
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("time (s, 2 Hz)", fontsize=11)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default="/Users/justin/ultraclear-paper/code/data/"
                           "real_data_arkit/uwbx4_aligned_2hz.csv")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--out",
                   default="/Users/justin/ultraclear-paper/code/figures/"
                           "uwbx4_arkit_path_2d_first30.png")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, anchors = load_first_n_valid(csv_path, args.n)
    print(f"loaded first {len(rows)} valid samples from {csv_path.name}")
    if anchors is not None:
        print(f"  anchor positions (car frame, m):")
        for i, (ax, ay) in enumerate(anchors):
            print(f"    A{i+1}: ({ax:+.2f}, {ay:+.2f})")
    if not rows:
        raise RuntimeError("no valid samples found")
    print(f"  px range: [{min(r['px'] for r in rows):.2f}, "
          f"{max(r['px'] for r in rows):.2f}] m")
    print(f"  py range: [{min(r['py'] for r in rows):.2f}, "
          f"{max(r['py'] for r in rows):.2f}] m")
    print(f"  pz range: [{min(r['pz'] for r in rows):.2f}, "
          f"{max(r['pz'] for r in rows):.2f}] m")

    plot_first_n(rows, anchors, csv_path, out_path)


if __name__ == "__main__":
    main()
