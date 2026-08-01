"""Overlay fake model predictions on the interpolated 3x ARKit path.

Treats the 3x interpolated ARKit samples as ground truth, then
generates a fake prediction at each timestep by adding isotropic
2D Gaussian noise calibrated to give a target *mean Euclidean*
error. For 2D isotropic Gaussian noise, E[||n||] = sigma * sqrt(pi/2),
so for target mean error E in meters, sigma = E * sqrt(2/pi) per axis.

Default target: 0.35 m mean Euclidean error (matches the
localization model's reported accuracy).

Output: code/figures/uwbx4_arkit_path_2d_interp3x_with_pred.png
        (also writes a CSV with GT + pred for downstream use)

CLI:
  --csv        input aligned CSV (default: uwbx4_aligned_2hz.csv)
  --n          number of original valid samples to use (default 30)
  --factor     interpolation factor (default 3 → 90 points)
  --noise_std_interp  noise sigma for the interp step (default 0.05)
  --sigma_per_axis    per-axis Gaussian sigma for the fake pred
                      noise, in meters (default 0.279 — gives mean
                      Euclidean err ~0.35 m)
  --target_mean_err   DEPRECATED alias; if set, overrides sigma
  --seed_interp       RNG seed for interp noise (default 0)
  --seed_pred         RNG seed for pred noise (default 100)
  --out               output PNG path
  --out_csv           output CSV path
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Re-use the loader + interpolator from the previous script.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from draw_uwbx4_arkit_path_2d_interp2x import (
    load_first_n_valid, interpolate_2x_with_noise,
)


def add_fake_predictions(px_gt, py_gt, sigma_per_axis: float,
                          seed: int) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray]:
    """Add isotropic 2D Gaussian noise to (px_gt, py_gt) with the
    given per-axis sigma (in meters). The empirical mean
    Euclidean error is then ~sigma_per_axis * sqrt(pi/2).

    Returns (px_pred, py_pred, euclidean_err_per_point).
    """
    sigma = sigma_per_axis
    rng = np.random.default_rng(seed)
    n = len(px_gt)
    dx = rng.normal(0.0, sigma, size=n)
    dy = rng.normal(0.0, sigma, size=n)
    px_pred = px_gt + dx
    py_pred = py_gt + dy
    err = np.sqrt(dx * dx + dy * dy)
    return px_pred, py_pred, err


def plot_gt_plus_pred(px_gt, py_gt, px_pred, py_pred, t_secs,
                       err, anchors, csv_path: Path, out_path: Path,
                       n_orig: int, factor: int, noise_std_interp: float,
                       target_mean_err: float) -> None:
    n_out = len(px_gt)
    mean_err = float(err.mean())
    p50 = float(np.percentile(err, 50))
    p90 = float(np.percentile(err, 90))
    p99 = float(np.percentile(err, 99))
    max_err = float(err.max())

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    # GT path (noisy interp) — light gray dots + line
    ax.plot(px_gt, py_gt, color="lightgray", lw=1, alpha=0.5, zorder=1)
    ax.scatter(px_gt, py_gt, c=t_secs, cmap="viridis", s=35,
                edgecolor="k", lw=0.3, alpha=0.55, zorder=2,
                label="interp GT (synthetic 'truth')")
    # Fake predictions — red dots, larger, on top
    ax.scatter(px_pred, py_pred, c="red", s=55, edgecolor="darkred",
                lw=0.5, zorder=4, alpha=0.95,
                label="fake model prediction")
    # Connecting lines GT→pred, faint, to show error vectors
    for x1, y1, x2, y2 in zip(px_gt, py_gt, px_pred, py_pred):
        ax.plot([x1, x2], [y1, y2], color="red", lw=0.4, alpha=0.18,
                 zorder=3)
    # Start / end markers (on the GT path)
    ax.plot(px_gt[0], py_gt[0], "go", ms=14, label="start (GT)",
             zorder=6)
    ax.plot(px_gt[-1], py_gt[-1], "g*", ms=18, label="end (GT)",
             zorder=6)
    # Anchors
    if anchors is not None:
        for i, (ax_x, ax_y) in enumerate(anchors):
            ax.scatter(ax_x, ax_y, s=120, marker="^", c="red",
                        edgecolor="black", linewidth=1.0, zorder=7)
            ax.annotate(f"A{i+1}", (ax_x, ax_y),
                         textcoords="offset points", xytext=(8, 6),
                         fontsize=11, fontweight="bold")

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title(
        f"uwbx4 ARKit-labeled path + fake model prediction\n"
        f"interp {factor}x to {n_out} pts (sigma={noise_std_interp:.3f} m)  |  "
        f"pred noise calibrated for mean err {target_mean_err:.2f} m\n"
        f"empirical: mean={mean_err:.3f} m  p50={p50:.3f}  "
        f"p90={p90:.3f}  p99={p99:.3f}  max={max_err:.3f}"
    )

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"  empirical err: mean={mean_err:.4f} m  p50={p50:.4f}  "
          f"p90={p90:.4f}  p99={p99:.4f}  max={max_err:.4f}")


def save_csv(px_gt, py_gt, px_pred, py_pred, t_secs, err,
              out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "gt_px_m", "gt_py_m",
                     "pred_px_m", "pred_py_m", "err_m"])
        for t, gx, gy, px, py, e in zip(t_secs, px_gt, py_gt,
                                          px_pred, py_pred, err):
            w.writerow([f"{t:.4f}", f"{gx:.6f}", f"{gy:.6f}",
                         f"{px:.6f}", f"{py:.6f}", f"{e:.6f}"])
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default="/Users/justin/ultraclear-paper/code/data/"
                           "real_data_arkit/uwbx4_aligned_2hz.csv")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--noise_std_interp", type=float, default=0.05)
    p.add_argument("--sigma_per_axis", type=float, default=0.279,
                   help="per-axis Gaussian sigma for fake pred noise (m); "
                        "default 0.279 gives mean Euclidean err "
                        "~0.35 m (matches the localization model's "
                        "reported accuracy)")
    p.add_argument("--target_mean_err", type=float, default=None,
                   help="DEPRECATED alias; if set, treated as sigma_per_axis")
    p.add_argument("--seed_interp", type=int, default=0)
    p.add_argument("--seed_pred", type=int, default=100)
    p.add_argument("--out",
                   default="/Users/justin/ultraclear-paper/code/figures/"
                           "uwbx4_arkit_path_2d_interp3x_with_pred.png")
    p.add_argument("--out_csv",
                   default="/Users/justin/ultraclear-paper/code/figures/"
                           "uwbx4_arkit_path_2d_interp3x_with_pred.csv")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, anchors = load_first_n_valid(csv_path, args.n)
    print(f"loaded first {len(rows)} valid samples from {csv_path.name}")
    if not rows:
        raise RuntimeError("no valid samples found")

    px_gt, py_gt, t_secs = interpolate_2x_with_noise(
        rows, args.factor, args.noise_std_interp, args.seed_interp
    )
    sigma = args.sigma_per_axis
    if args.target_mean_err is not None:
        sigma = args.target_mean_err
    px_pred, py_pred, err = add_fake_predictions(
        px_gt, py_gt, sigma, args.seed_pred
    )
    print(f"interpolated {len(rows)} → {len(px_gt)} pts (factor={args.factor}, "
          f"sigma={args.noise_std_interp} m)")
    target_mean = sigma * math.sqrt(math.pi / 2.0)
    print(f"fake pred: per-axis sigma = {sigma:.4f} m  "
          f"(expected mean Euclidean err = {target_mean:.3f} m)")

    plot_gt_plus_pred(px_gt, py_gt, px_pred, py_pred, t_secs,
                       err, anchors, csv_path, out_path,
                       n_orig=len(rows), factor=args.factor,
                       noise_std_interp=args.noise_std_interp,
                       target_mean_err=target_mean)
    save_csv(px_gt, py_gt, px_pred, py_pred, t_secs, err,
              Path(args.out_csv))


if __name__ == "__main__":
    main()
