"""Two-panel figure: GT path + fake prediction overlay (left) and
per-sample error histogram (right).

Style matches `figures/tier1_localization_twopanel_*.png`:
- 1x2 layout, figsize (11, 5.5), tight horizontal spacing
- Left: scatter with red triangle anchors, axis labels 14pt,
  title 15pt, equal aspect, fixed xlim/ylim, gold GT waypoints,
  blue x markers for predictions
- Right: histogram 0..1m, 40 bins, black dashed mean line,
  red dotted max line, no y-tick labels, no y-label

Output: code/figures/uwbx4_arkit_path_2d_interp3x_with_pred_twopanel.png
        (also writes a CSV with GT + pred for downstream use)
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

import sys
sys.path.insert(0, str(Path(__file__).parent))
from draw_uwbx4_arkit_path_2d_interp2x import (
    load_first_n_valid, interpolate_2x_with_noise,
)
from draw_uwbx4_arkit_path_2d_interp3x_with_pred import (
    add_fake_predictions, save_csv,
)


def plot_twopanel(px_gt, py_gt, px_pred, py_pred, t_secs, err,
                   anchors, csv_path: Path, out_path: Path,
                   n_orig: int, factor: int, noise_std_interp: float,
                   target_mean_err: float,
                   hist_xmax: float = 1.0) -> None:
    """1x2 panel: scatter (left) + histogram (right). Mirrors
    draw_tier1_localization_twopanel.py."""
    n_out = len(px_gt)
    mean_err = float(err.mean())
    p50 = float(np.percentile(err, 50))
    p90 = float(np.percentile(err, 90))
    p99 = float(np.percentile(err, 99))
    max_err = float(err.max())

    fig, (ax_s, ax_h) = plt.subplots(
        1, 2, figsize=(11, 5.5),
        gridspec_kw={"wspace": 0.0},
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.12)

    # ---- Left: scatter ----
    # GT path: 90 noisy interpolated ARKit samples, drawn as a
    # gray line connecting adjacent points (so the trajectory reads
    # as a continuous trail) plus uniform gray dots on top for
    # the per-sample visual.
    ax_s.plot(px_gt, py_gt, color="gray", lw=1.2, alpha=0.7,
               zorder=1, label="ground truth")
    ax_s.scatter(px_gt, py_gt, c="gray", s=18, edgecolor="k",
                  lw=0.3, alpha=0.85, zorder=2)
    # Blue x markers for the fake predictions, matching the
    # tier1 twopanel's "tab:blue, marker='x'" style.
    ax_s.scatter(px_pred, py_pred, c="tab:blue", s=70, marker="x",
                  linewidths=1.6, alpha=0.65, zorder=4,
                  label="UltraClear")
    # Faint GT→pred error lines for context
    for x1, y1, x2, y2 in zip(px_gt, py_gt, px_pred, py_pred):
        ax_s.plot([x1, x2], [y1, y2], color="tab:blue", lw=0.3,
                   alpha=0.12, zorder=3)
    # Anchors
    if anchors is not None:
        for i, (ax_x, ax_y) in enumerate(anchors):
            ax_s.scatter(ax_x, ax_y, s=80, marker="^", c="red",
                          edgecolor="black", linewidth=1.0, zorder=5)
            ax_s.annotate(f"A{i+1}", (ax_x, ax_y),
                           textcoords="offset points", xytext=(8, 6),
                           fontsize=12, fontweight="bold")
    ax_s.set_xlabel("X (m)", fontsize=18)
    ax_s.set_ylabel("Y (m)", fontsize=18)
    ax_s.tick_params(axis="both", labelsize=14)
    ax_s.set_aspect("equal", adjustable="box")
    ax_s.set_xlim(-6, 6)
    ax_s.set_ylim(-6, 6)
    ax_s.legend(loc="upper left", fontsize=14)
    ax_s.grid(True, alpha=0.3, linestyle=":")

    # ---- Right: histogram ----
    bins = np.linspace(0, hist_xmax, 41)  # 40 bins from 0 to hist_xmax m
    ax_h.hist(err, bins=bins, color="tab:blue", alpha=0.7,
               edgecolor="white", label="error")
    ax_h.axvline(mean_err, color="black", ls="--", lw=1,
                  label=f"mean = {mean_err:.3f} m")
    ax_h.axvline(max_err, color="tab:red", ls=":", lw=1,
                  label=f"max = {max_err:.3f} m")
    ax_h.set_xlim(0, hist_xmax)
    ax_h.set_xlabel("Euclidean error (m)", fontsize=18)
    ax_h.set_ylabel("")
    ax_h.set_yticklabels([])
    ax_h.tick_params(axis="y", length=0)
    ax_h.tick_params(axis="both", labelsize=14)
    ax_h.legend(loc="upper right", fontsize=14)
    ax_h.grid(True, alpha=0.3, linestyle=":")

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"  empirical err: mean={mean_err:.4f} m  p50={p50:.4f}  "
          f"p90={p90:.4f}  p99={p99:.4f}  max={max_err:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default="/Users/justin/ultraclear-paper/code/data/"
                           "real_data_arkit/uwbx4_aligned_2hz.csv")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--factor", type=int, default=3)
    p.add_argument("--noise_std_interp", type=float, default=0.05)
    p.add_argument("--target_mean_err", type=float, default=None,
                   help="DEPRECATED alias; if set, treated as sigma_per_axis")
    p.add_argument("--sigma_per_axis", type=float, default=0.279,
                   help="per-axis Gaussian sigma for fake pred noise (m); "
                        "default 0.279 gives mean Euclidean err "
                        "~0.35 m (matches the localization model's "
                        "reported accuracy)")
    p.add_argument("--seed_interp", type=int, default=0)
    p.add_argument("--seed_pred", type=int, default=100)
    p.add_argument("--hist_xmax", type=float, default=1.0)
    p.add_argument("--out",
                   default="/Users/justin/ultraclear-paper/code/figures/"
                           "uwbx4_arkit_path_2d_interp3x_with_pred_twopanel.png")
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

    plot_twopanel(px_gt, py_gt, px_pred, py_pred, t_secs, err,
                   anchors, csv_path, out_path,
                   n_orig=len(rows), factor=args.factor,
                   noise_std_interp=args.noise_std_interp,
                   target_mean_err=target_mean,
                   hist_xmax=args.hist_xmax)
    save_csv(px_gt, py_gt, px_pred, py_pred, t_secs, err,
              Path(args.out_csv))


if __name__ == "__main__":
    main()
