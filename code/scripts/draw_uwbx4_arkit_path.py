"""Draw the ARKit-labeled path from uwbx4_aligned_2hz.csv.

Uses `px, py, pz` columns as the tag location in world frame.
The CSV is time-aligned at 2 Hz; rows where `px` is NaN are skipped.

Output:
  - figures/uwbx4_arkit_path_2d.png  (top-down x/y, colored by time)
  - figures/uwbx4_arkit_path_3d.png  (3D, with z color-mapped)
  - figures/uwbx4_arkit_path.csv     (cleaned path data for re-use)
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_path(csv_path: Path) -> dict:
    """Read the aligned CSV, extract t_grid + px/py/pz, drop NaN rows."""
    rows: list[dict] = []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                px = float(r["px"]); py = float(r["py"]); pz = float(r["pz"])
            except (ValueError, KeyError):
                continue
            if any(np.isnan(v) for v in (px, py, pz)):
                continue
            rows.append({"t": r["t_grid"], "px": px, "py": py, "pz": pz,
                         "n": int(rdr.line_num - 1)})
    return {
        "t": [r["t"] for r in rows],
        "px": np.array([r["px"] for r in rows]),
        "py": np.array([r["py"] for r in rows]),
        "pz": np.array([r["pz"] for r in rows]),
        "n": np.array([r["n"] for r in rows]),
    }


def plot_2d(d: dict, out_path: Path) -> None:
    # In ARKit world frame, y is up/elevation; x and z are horizontal.
    # For 2D localization the horizontal plane is (px, pz).
    px, pz = d["px"], d["pz"]
    py = d["py"]
    t_secs = np.arange(len(px)) * 0.5  # 2 Hz
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sc = ax.scatter(px, pz, c=t_secs, cmap="viridis", s=60, edgecolor="k",
                     lw=0.6, zorder=3)
    ax.plot(px, pz, color="gray", lw=1, alpha=0.4, zorder=2)
    ax.plot(px[0], pz[0], "go", ms=12, label="start", zorder=4)
    ax.plot(px[-1], pz[-1], "r*", ms=18, label="end", zorder=4)
    ax.set_xlabel("px (m, world frame, horizontal)")
    ax.set_ylabel("pz (m, world frame, horizontal)")
    ax.set_title(
        f"uwbx4 ARKit-labeled path (top-down, horizontal plane)\n"
        f"{len(px)} valid points, 2 Hz, "
        f"x range [{px.min():.2f}, {px.max():.2f}] m, "
        f"z range [{pz.min():.2f}, {pz.max():.2f}] m\n"
        f"elevation py range [{py.min():.2f}, {py.max():.2f}] m (color in 3D plot)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_aspect("equal", adjustable="datalim")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("time (s, 2 Hz)")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_3d(d: dict, out_path: Path) -> None:
    # 3D path in ARKit world frame: x and z are horizontal, y is up.
    px, py, pz = d["px"], d["py"], d["pz"]
    t_secs = np.arange(len(px)) * 0.5
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(px, pz, py, c=t_secs, cmap="viridis", s=60,
                     edgecolor="k", lw=0.4)
    ax.plot(px, pz, py, color="gray", lw=1, alpha=0.4)
    ax.plot([px[0]], [pz[0]], [py[0]], "go", ms=12, label="start")
    ax.plot([px[-1]], [pz[-1]], [py[-1]], "r*", ms=18, label="end")
    ax.set_xlabel("px (m, horizontal)")
    ax.set_ylabel("pz (m, horizontal)")
    ax.set_zlabel("py (m, up)")
    ax.set_title(
        f"uwbx4 ARKit-labeled path (3D, ARKit world frame)\n"
        f"{len(px)} valid points, py (elevation) range "
        f"[{py.min():.2f}, {py.max():.2f}] m"
    )
    ax.legend(loc="best")
    cb = fig.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label("time (s, 2 Hz)")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_both_2d(paths: list, csv_paths: list, out_path: Path) -> None:
    """Side-by-side 2D top-down plot of both uwbx4 CSVs.

    Uses the (px, pz) horizontal plane from ARKit world frame.
    Each subplot is its own coordinate system (the two CSVs are
    from different sessions in different parts of the room).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                               constrained_layout=True)
    for ax, d, cp in zip(axes, paths, csv_paths):
        if len(d["px"]) == 0:
            ax.text(0.5, 0.5, "no valid points",
                     ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cp.name)
            continue
        px, pz = d["px"], d["pz"]
        t_secs = np.arange(len(px)) * 0.5
        sc = ax.scatter(px, pz, c=t_secs, cmap="viridis", s=60,
                         edgecolor="k", lw=0.6, zorder=3)
        ax.plot(px, pz, color="gray", lw=1, alpha=0.4, zorder=2)
        ax.plot(px[0], pz[0], "go", ms=12, label="start", zorder=4)
        ax.plot(px[-1], pz[-1], "r*", ms=18, label="end", zorder=4)
        ax.set_xlabel("px (m, horizontal)")
        ax.set_ylabel("pz (m, horizontal)")
        ax.set_title(
            f"{cp.name}\n"
            f"{len(px)} pts, 2 Hz, "
            f"x [{px.min():.2f}, {px.max():.2f}], "
            f"z [{pz.min():.2f}, {pz.max():.2f}] m"
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_aspect("equal", adjustable="datalim")
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label("time (s, 2 Hz)")
    fig.suptitle(
        "uwbx4 ARKit-labeled paths (top-down, horizontal plane)\n"
        "two separate collection sessions; py (elevation) excluded",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def save_csv(d: dict, out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_grid", "row_in_source", "px_m", "py_m", "pz_m"])
        for t, n, px, py, pz in zip(d["t"], d["n"], d["px"], d["py"], d["pz"]):
            w.writerow([t, int(n), f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}"])
    print(f"cleaned path → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default=None,
                   help="single CSV to plot. If omitted, plots BOTH "
                        "files in data/real_data_arkit/ side by side.")
    p.add_argument("--out_dir",
                   default="/Users/justin/ultraclear-paper/code/figures")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv is not None:
        csv_paths = [Path(args.csv)]
    else:
        csv_paths = [
            Path("/Users/justin/ultraclear-paper/code/data/real_data_arkit/"
                 "uwbx4_aligned_2hz.csv"),
            Path("/Users/justin/ultraclear-paper/code/data/real_data_arkit/"
                 "uwbx4_2_real_test_aligned_2hz.csv"),
        ]
    for cp in csv_paths:
        if not cp.exists():
            raise FileNotFoundError(cp)

    if len(csv_paths) == 1:
        d = load_path(csv_paths[0])
        print(f"loaded {len(d['px'])} valid (px,py,pz) points from "
              f"{csv_paths[0].name}")
        if len(d["px"]) == 0:
            raise RuntimeError("no valid points found")
        stem = csv_paths[0].stem
        save_csv(d, out_dir / f"{stem}_path.csv")
        plot_2d(d, out_dir / f"{stem}_path_2d.png")
        plot_3d(d, out_dir / f"{stem}_path_3d.png")
    else:
        # Side-by-side plot of both CSVs (horizontal plane only,
        # which is what you actually want for 2D localization).
        paths = [load_path(cp) for cp in csv_paths]
        for cp, d in zip(csv_paths, paths):
            print(f"loaded {len(d['px'])} valid (px,py,pz) points from "
                  f"{cp.name}")
            if len(d["px"]) > 0:
                save_csv(d, out_dir / f"{cp.stem}_path.csv")
        plot_both_2d(paths, csv_paths, out_dir / "uwbx4_both_path_2d.png")


if __name__ == "__main__":
    main()
