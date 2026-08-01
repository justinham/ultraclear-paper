"""Two-panel figure for the ASTA per-sample localization result,
matching the visual style of
`figures/tier1_localization_twopanel_indoor_garage_parking_05.png`:
  - Left:  scatter of GT (gold dots) vs predicted (blue x) per step,
           with 4 red anchor triangles + A1-A4 labels.
  - Right: histogram of per-step Euclidean errors.

Output: code/figures/asta_persample_twopanel_<scene>.png

Reads the eval CSV written by `infer_asta_persample_loc.py`
(checkpoints/_tmp_asta_persample_<scene>/inference_all.csv).
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

# Reuse the data + ckpt helpers from the existing scripts.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_asta_persample_loc import (
    ASTAPerSampleLoc, load_track2_csv, make_windows, to_tensor,
)
from ultraclear.config import Config


def load_inference_csv(csv_path: Path):
    """Return arrays (x_pred, y_pred, x_true, y_true, err) from a
    `inference_<mode>.csv` file written by infer_asta_persample_loc.py."""
    xs_p, ys_p, xs_t, ys_t, errs = [], [], [], [], []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            xs_p.append(float(r["x_pred"]))
            ys_p.append(float(r["y_pred"]))
            xs_t.append(float(r["x_true"]))
            ys_t.append(float(r["y_true"]))
            errs.append(float(r["err_m"]))
    return (np.array(xs_t), np.array(ys_t),
            np.array(xs_p), np.array(ys_p),
            np.array(errs))


def make_figure(csv_eval_path: Path, encoder_ckpt: Path,
                scene_name: str, out_path: Path) -> dict:
    """Build the 2-panel figure; return the headline metrics."""
    # 1. GT + prediction arrays
    xt, yt, xp, yp, errs = load_inference_csv(csv_eval_path)
    # For each window, only the LAST step is a unique GT waypoint
    # when stride==T (non-overlapping windows of T=32 dwell rows).
    # Since `csv_eval_path` is `inference_all.csv` with 56 windows
    # of 32 steps each, we take the last step of every window to
    # get the 56 distinct GT positions (12 waypoints + 44
    # transition-line positions).
    n_steps_per_window = 32
    n_windows = len(xt) // n_steps_per_window
    last_idx = (np.arange(n_windows) + 1) * n_steps_per_window - 1
    wp_xs = xt[last_idx]
    wp_ys = yt[last_idx]

    mean_err = errs.mean()
    max_err = errs.max()

    # 2. Anchor positions (load from the same CSV we used at train time)
    # We don't have the CSV here, so re-derive from the encoder ckpt's
    # training scenario.  For sionna track2 (the 12-waypoint setup), the
    # 4 anchors are: A1=(-0.52,-0.90), A2=(-0.54,0.35),
    # A3=(0.52,-0.90), A4=(0.54,0.35) at z=2.2.  We use the (x, y)
    # 2D positions on the horizontal plane.
    anchor_pos = np.array([
        [-0.52, -0.90],
        [-0.54, +0.35],
        [+0.52, -0.90],
        [+0.54, +0.35],
    ])

    # 3. Build the figure (same layout as the tier1 twopanel)
    fig, (ax_s, ax_h) = plt.subplots(
        1, 2, figsize=(11, 5.5),
        gridspec_kw={"wspace": 0.0},
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.12)

    # ---- Left: scatter ----
    # Set the data extents so the data fits cleanly with a little
    # margin.  Match the tier1 figure's manual xlim/ylim.
    ax_s.set_xlim(-7, 7)
    ax_s.set_ylim(-7, 17)
    # Predicted x (blue), alpha 0.55 so the GT dots show through
    ax_s.scatter(xp, yp, s=70, marker="x", c="tab:blue",
                  linewidths=1.6, alpha=0.55, zorder=4,
                  label="prediction")
    # GT dwell waypoints (gold dots)
    ax_s.scatter(wp_xs, wp_ys, s=70, marker="o", c="gold",
                  edgecolor="black", linewidth=0.6, zorder=3,
                  label="ground truth")
    # 4 anchor triangles
    for i, (ax_x, ax_y) in enumerate(anchor_pos):
        ax_s.scatter(ax_x, ax_y, s=80, marker="^", c="red",
                      edgecolor="black", linewidth=1.0, zorder=5)
        ax_s.annotate(f"A{i+1}", (ax_x, ax_y),
                       textcoords="offset points", xytext=(8, 6),
                       fontsize=9, fontweight="bold")
    ax_s.set_xlabel("x (m)", fontsize=14)
    ax_s.set_ylabel("y (m)", fontsize=14)
    ax_s.tick_params(axis="both", labelsize=12)
    ax_s.set_aspect("equal", adjustable="box")
    ax_s.set_title("Location GT vs Prediction", fontsize=15)
    ax_s.legend(loc="upper left", fontsize=11)

    # ---- Right: histogram ----
    bins = np.linspace(0, 1.0, 41)
    ax_h.hist(errs, bins=bins, color="tab:blue", edgecolor="black",
               alpha=0.75, label="error")
    ax_h.axvline(mean_err, color="black", linestyle="--",
                  label=f"mean = {mean_err:.3f} m")
    ax_h.axvline(max_err, color="red", linestyle=":",
                  label=f"max = {max_err:.3f} m")
    ax_h.set_xlim(0, 1.0)
    ax_h.set_xlabel("Euclidean error (m)", fontsize=14)
    ax_h.set_ylabel("")
    ax_h.set_yticklabels([])
    ax_h.tick_params(axis="y", length=0)
    ax_h.tick_params(axis="both", labelsize=12)
    ax_h.set_title("Location error distribution", fontsize=15)
    ax_h.legend(loc="upper right", fontsize=10)
    ax_h.grid(True, axis="y", linestyle=":", alpha=0.4)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    return {
        "n_steps": int(len(errs)),
        "mean_err_m": float(mean_err),
        "max_err_m": float(max_err),
        "n_waypoints_drawn": int(len(wp_xs)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_eval",
                   default="/Users/justin/ultraclear-paper/code/checkpoints/"
                           "_tmp_asta_persample_garage_05/inference_all.csv")
    p.add_argument("--encoder_ckpt",
                   default="/Users/justin/ultraclear-paper/code/"
                           "checkpoints/tier1_mixed.pt/tier1_sionna.pt")
    p.add_argument("--scene", default="indoor_garage_parking_05")
    p.add_argument("--out",
                   default="/Users/justin/ultraclear-paper/code/figures/"
                           "asta_persample_twopanel_indoor_garage_parking_05.png")
    args = p.parse_args()

    csv_eval = Path(args.csv_eval)
    if not csv_eval.exists():
        raise FileNotFoundError(csv_eval)
    out_path = Path(args.out)

    print(f"ASTA per-sample localization two-panel figure")
    print(f"  csv_eval: {csv_eval}")
    print(f"  scene:    {args.scene}")
    print(f"  out:      {out_path}")
    print()
    summary = make_figure(csv_eval, Path(args.encoder_ckpt),
                           args.scene, out_path)
    print()
    print("summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
