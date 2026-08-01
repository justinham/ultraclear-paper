"""Interpolate the ARKit-labeled path to 2x the sample count with
random Gaussian noise, then redraw the top-down 2D figure.

Reads the same CSV as draw_uwbx4_arkit_path_2d_first30.py
(`uwbx4_aligned_2hz.csv`), takes the first N valid samples (default
30), cubic-spline-interpolates to 2N points at 0.5s spacing, adds
small Gaussian noise (default sigma = 0.05 m, ~5 cm) to each
interpolated point, then plots the noisy dense path the same way
the original script does (top-down, horizontal plane, time-colored,
anchors marked).

The interpolation is on the time axis (0, 0.5, 1.0, ... seconds);
`px_car` and `py_car` are interpolated separately with
scipy.interpolate.CubicSpline. Noise is added to both axes.

Output: code/figures/uwbx4_arkit_path_2d_interp2x.png

CLI:
  --csv        input aligned CSV (default: uwbx4_aligned_2hz.csv)
  --n          number of original valid samples to use (default 30)
  --factor     interpolation factor (default 2 → 2x points)
  --noise_std  Gaussian noise sigma in meters (default 0.05)
  --seed       RNG seed for reproducibility (default 0)
  --out        output PNG path
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def load_first_n_valid(csv_path: Path, n: int):
    """Same loader as draw_uwbx4_arkit_path_2d_first30.py — first N
    valid rows + 4 anchor positions in the car frame."""
    rows = []
    anchors = None
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if anchors is None:
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


def interpolate_2x_with_noise(rows, factor: int, noise_std: float,
                                seed: int):
    """Cubic-spline interpolate (px_car, py_car) to factor * len(rows)
    points at the original 0.5s spacing, then add iid Gaussian noise
    to each interpolated point.

    Returns (px_noisy, py_noisy, t_secs) where t_secs has length
    factor * len(rows), spanning the same time window as the input.
    """
    n = len(rows)
    if n < 4:
        # CubicSpline needs >= 4 points (k=3); fall back to linear
        # for very short sequences.
        kind = "linear"
    else:
        kind = "cubic"

    t_input = np.arange(n, dtype=float) * 0.5  # 2 Hz
    px = np.array([r["px"] for r in rows])
    py = np.array([r["py"] for r in rows])

    n_out = n * factor
    # Re-sample at the *original* 0.5s spacing but with `factor`
    # points packed between each pair of original samples.
    # i.e. between input[i] and input[i+1] we place (factor-1) new
    # points; we also keep the original samples at their exact
    # positions, so the first and last points of the output are the
    # noise-free original measurements.
    # Total: (factor-1)*(n-1) + n = factor*n - factor + 1... hmm
    # that doesn't give exactly factor*n. Instead just generate
    # n*factor points spanning [0, t_input[-1]] inclusive.
    t_out = np.linspace(0.0, t_input[-1], n_out)

    if kind == "cubic":
        cs_x = CubicSpline(t_input, px)
        cs_y = CubicSpline(t_input, py)
    else:
        cs_x = lambda t: np.interp(t, t_input, px)  # noqa: E731
        cs_y = lambda t: np.interp(t, t_input, py)  # noqa: E731

    px_interp = cs_x(t_out)
    py_interp = cs_y(t_out)

    rng = np.random.default_rng(seed)
    px_noisy = px_interp + rng.normal(0.0, noise_std, size=n_out)
    py_noisy = py_interp + rng.normal(0.0, noise_std, size=n_out)

    return px_noisy, py_noisy, t_out


def plot_interp(px, py, t_secs, anchors, csv_path: Path, out_path: Path,
                n_orig: int, factor: int, noise_std: float,
                show_orig: bool = False) -> None:
    """Same look as draw_uwbx4_arkit_path_2d_first30.py but with the
    dense + noisy interpolated path."""
    n_out = len(px)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    sc = ax.scatter(px, py, c=t_secs, cmap="viridis", s=60,
                     edgecolor="k", lw=0.4, alpha=0.85, zorder=3)
    ax.plot(px, py, color="lightgray", lw=1, alpha=0.6, zorder=1)
    # By default we don't overlay the original samples — the
    # interpolated path is the whole story. Set show_orig=True
    # (or pass --show_orig) to mark them with a small white dot.
    px_orig = px[::factor]
    py_orig = py[::factor]
    if show_orig:
        ax.scatter(px_orig, py_orig, s=20, c="white", edgecolor="black",
                    lw=0.8, zorder=4,
                    label=f"original {n_orig} samples (2 Hz)")
    ax.plot(px[0], py[0], "go", ms=14, label="start", zorder=5)
    ax.plot(px[-1], py[-1], "r*", ms=20, label="end", zorder=5)
    if anchors is not None:
        for i, (ax_x, ax_y) in enumerate(anchors):
            ax.scatter(ax_x, ax_y, s=120, marker="^", c="red",
                        edgecolor="black", linewidth=1.0, zorder=6)
            ax.annotate(f"A{i+1}", (ax_x, ax_y),
                         textcoords="offset points", xytext=(8, 6),
                         fontsize=11, fontweight="bold")

    ax.set_xlabel("px_car (m, forward)", fontsize=12)
    ax.set_ylabel("py_car (m, lateral)", fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title(
        f"uwbx4 ARKit-labeled path (top-down, horizontal plane)\n"
        f"interpolated {factor}x to {n_out} samples "
        f"(cubic spline, sigma={noise_std:.3f} m Gaussian noise)\n"
        f"source: {csv_path.name}"
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("time (s, 2 Hz → %.1f Hz effective)" % (2 * factor),
                 fontsize=11)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default="/Users/justin/ultraclear-paper/code/data/"
                           "real_data_arkit/uwbx4_aligned_2hz.csv")
    p.add_argument("--n", type=int, default=30,
                   help="number of original valid samples to use")
    p.add_argument("--factor", type=int, default=2,
                   help="interpolation factor (default 2 → 2x points)")
    p.add_argument("--noise_std", type=float, default=0.05,
                   help="Gaussian noise sigma in meters (default 0.05)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show_orig", action="store_true",
                   help="overlay the original 2 Hz samples on the path")
    p.add_argument("--out",
                   default="/Users/justin/ultraclear-paper/code/figures/"
                           "uwbx4_arkit_path_2d_interp2x.png")
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

    px, py, t_secs = interpolate_2x_with_noise(
        rows, args.factor, args.noise_std, args.seed
    )
    print(f"interpolated {len(rows)} → {len(px)} samples "
          f"(factor={args.factor}, sigma={args.noise_std} m, seed={args.seed})")
    print(f"  px range: [{px.min():.3f}, {px.max():.3f}] m")
    print(f"  py range: [{py.min():.3f}, {py.max():.3f}] m")

    plot_interp(px, py, t_secs, anchors, csv_path, out_path,
                n_orig=len(rows), factor=args.factor,
                noise_std=args.noise_std, show_orig=args.show_orig)


if __name__ == "__main__":
    main()
