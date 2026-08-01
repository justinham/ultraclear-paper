"""Inference-time evaluation of a trained per-sample localization head.

Loads the frozen ASTA encoder from the tier1 ckpt, the trained
per-timestep regression head from a per-scene dir, and runs
prediction on either:
  - the val split (matches the training-time split, for sanity
    check / reproduction of the headline number)
  - the entire data range (all 1800 rows, all 12 waypoints + 12
    transitions) — gives a clean "what the model thinks the full
    trajectory looks like" plot

Outputs:
  - <out_dir>/inference_<mode>.csv  (window_idx, t, x_pred, y_pred, x_true, y_true, err_m)
  - <out_dir>/inference_<mode>_track.png  (GT vs predicted scatter)
  - <out_dir>/inference_<mode>_errhist.png  (per-step error histogram)
  - <out_dir>/inference_<mode>_summary.txt  (overall MAE / per-waypoint breakdown)
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

from ultraclear.asta import ASTAEncoder
from ultraclear.config import Config

# Reuse the data + model definitions from the trainer script.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_asta_persample_loc import (
    load_track2_csv, make_windows, ASTAPerSampleLoc,
    to_tensor,
)


def infer(csv_path: Path, encoder_ckpt: Path, head_ckpt: Path,
          out_dir: Path, mode: str = "val",
          T: int = 32, stride: int = 32,
          device: str = "cpu") -> dict:
    """Run inference; return a summary dict."""
    cfg = Config(n_anchors=4, n_channels=2, d_model=64, n_asta_blocks=2)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ckpt to get the normalization stats
    ckpt_full = torch.load(encoder_ckpt, map_location=device, weights_only=False)
    norm_mean = torch.as_tensor(ckpt_full["norm_mean"], device=device,
                                  dtype=torch.float32)
    norm_std = torch.as_tensor(ckpt_full["norm_std"], device=device,
                                 dtype=torch.float32)
    print(f"  norm mean: {norm_mean.cpu().tolist()}   std: {norm_std.cpu().tolist()}")

    # Build model + load encoder
    model = ASTAPerSampleLoc(cfg).to(device)
    state = ckpt_full
    for k in ("model_state", "model", "state_dict"):
        if k in state and isinstance(state[k], dict):
            state = state[k]
            break
    if isinstance(state, dict):
        state = {k.removeprefix("encoder."): v for k, v in state.items()}
    model.encoder.load_state_dict(state, strict=False)
    model.encoder.eval()

    # Load trained head
    head_state = torch.load(head_ckpt, map_location=device, weights_only=False)
    model.head.load_state_dict(head_state)
    model.head.eval()
    print(f"  loaded head from {head_ckpt}")

    # Build windows
    X_np, Y_np = load_track2_csv(csv_path)
    print(f"  loaded {len(X_np)} valid rows from {csv_path.name}")
    wx, wy = make_windows(X_np, Y_np, T, stride)
    print(f"  windows: {wx.shape}  (T={T}, stride={stride})")

    if mode == "val":
        # Mirror the per-class split from training
        n = len(wx)
        val_per_class = max(1, int(0.2 * (n // 12)))
        last_wp = wy[:, -1]
        wp_ids = (last_wp[:, 0] * 100 + last_wp[:, 1] * 7).astype(int)
        val_idx = []
        for u in np.unique(wp_ids):
            idxs = np.where(wp_ids == u)[0]
            n_val_u = min(val_per_class, len(idxs) // 2)
            val_idx.extend(idxs[:n_val_u].tolist())
        val_idx = np.asarray(val_idx, dtype=int)
        wx_infer, wy_infer = wx[val_idx], wy[val_idx]
        print(f"  mode=val: {len(val_idx)} val windows "
              f"({val_per_class} per waypoint)")
    elif mode == "all":
        wx_infer, wy_infer = wx, wy
        print(f"  mode=all: {len(wx_infer)} windows (all of them)")
    else:
        raise ValueError(f"unknown mode: {mode!r} (use 'val' or 'all')")

    # Normalize and run
    X_t = to_tensor(wx_infer, device)
    X_t = (X_t - norm_mean) / norm_std
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()                  # (W, T, 2)
    true = wy_infer                                        # (W, T, 2)

    # Per-step error
    per_step_err = np.linalg.norm(pred - true, axis=-1)  # (W, T)
    overall_mae = per_step_err.mean()
    per_window_mae = per_step_err.mean(axis=1)            # (W,)
    print(f"  overall per-step MAE: {overall_mae:.3f} m   "
          f"per-window mean: {per_window_mae.mean():.3f} m   "
          f"max: {per_step_err.max():.3f} m")

    # Per-waypoint breakdown
    wp_xs, wp_ys = true[:, -1, 0], true[:, -1, 1]        # (W,) waypoint x, y
    wp_keys = np.unique(np.stack([wp_xs, wp_ys], axis=1), axis=0)
    per_wp_mae = []
    for wpx, wpy in wp_keys:
        mask = (wp_xs == wpx) & (wp_ys == wpy)
        mae = per_step_err[mask].mean()
        per_wp_mae.append((float(wpx), float(wpy), float(mae), int(mask.sum())))
    per_wp_mae.sort()

    # Save CSV (per-step)
    csv_path_out = out_dir / f"inference_{mode}.csv"
    with open(csv_path_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_idx", "t", "x_pred", "y_pred",
                    "x_true", "y_true", "err_m"])
        for wi in range(pred.shape[0]):
            for t in range(pred.shape[1]):
                w.writerow([wi, t,
                            f"{pred[wi, t, 0]:.4f}",
                            f"{pred[wi, t, 1]:.4f}",
                            f"{true[wi, t, 0]:.4f}",
                            f"{true[wi, t, 1]:.4f}",
                            f"{per_step_err[wi, t]:.4f}"])
    print(f"  saved CSV → {csv_path_out}")

    # Save summary
    summary_path = out_dir / f"inference_{mode}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"ASTA per-sample localization inference report\n")
        f.write(f"  csv:        {csv_path}\n")
        f.write(f"  encoder:    {encoder_ckpt}\n")
        f.write(f"  head:       {head_ckpt}\n")
        f.write(f"  mode:       {mode}\n")
        f.write(f"  T={T}  stride={stride}\n")
        f.write(f"  n_windows:  {len(wx_infer)}\n")
        f.write(f"\noverall per-step MAE: {overall_mae:.4f} m\n")
        f.write(f"per-window MAE mean:  {per_window_mae.mean():.4f} m\n")
        f.write(f"per-window MAE std:   {per_window_mae.std():.4f} m\n")
        f.write(f"per-step MAE max:     {per_step_err.max():.4f} m\n")
        f.write(f"\nper-waypoint breakdown:\n")
        for wpx, wpy, mae, n in per_wp_mae:
            f.write(f"  wp=({wpx:+.2f}, {wpy:+.2f})  MAE={mae:.4f} m  "
                    f"n={n}\n")
    print(f"  saved summary → {summary_path}")

    # Plot 1: GT vs predicted scatter (use last step of each window,
    # since the windows are non-overlapping when stride==T)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    # Plot the trajectory of GT (last step of every window)
    if mode == "all":
        ax.plot(true[:, -1, 0], true[:, -1, 1],
                color="black", lw=1.5, alpha=0.7, label="GT (per window)", zorder=2)
        ax.plot(pred[:, -1, 0], pred[:, -1, 1],
                color="tab:red", lw=1.5, alpha=0.7, ls="--",
                label="predicted (per window)", zorder=2)
    # Also plot every per-step pred vs true as a soft cloud
    ax.scatter(true[:, :, 0].flatten(), true[:, :, 1].flatten(),
                s=6, c="tab:blue", alpha=0.25, label="GT (per step)", zorder=3)
    ax.scatter(pred[:, :, 0].flatten(), pred[:, :, 1].flatten(),
                s=6, c="tab:red", alpha=0.25, label="predicted (per step)", zorder=4)
    ax.set_xlabel("tag_x (m)")
    ax.set_ylabel("tag_y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(f"ASTA per-sample localization inference ({mode})\n"
                  f"{csv_path.name}\n"
                  f"overall MAE = {overall_mae:.3f} m   "
                  f"per-window mean = {per_window_mae.mean():.3f} m")
    ax.legend(loc="upper left", fontsize=10)
    fig_path = out_dir / f"inference_{mode}_track.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved track figure → {fig_path}")

    # Plot 2: per-step error histogram
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.hist(per_step_err.flatten(), bins=40, color="tab:red",
            edgecolor="black", alpha=0.75)
    ax.axvline(overall_mae, color="black", linestyle="--",
                label=f"mean = {overall_mae:.3f} m")
    ax.set_xlabel("per-step Euclidean error (m)")
    ax.set_ylabel("count")
    ax.set_title(f"Per-step error distribution ({mode}) — {csv_path.name}")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    hist_path = out_dir / f"inference_{mode}_errhist.png"
    fig.savefig(hist_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved err hist → {hist_path}")

    return {
        "mode": mode,
        "n_windows": int(len(wx_infer)),
        "overall_mae_m": float(overall_mae),
        "per_window_mae_mean_m": float(per_window_mae.mean()),
        "per_window_mae_std_m": float(per_window_mae.std()),
        "per_step_max_m": float(per_step_err.max()),
        "per_wp_mae": per_wp_mae,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default="/Users/justin/ultraclear-paper/code/data/"
                           "sionna_track2_swapped/"
                           "sionna_uwb_track_indoor_garage_parking_05_"
                           "A1A4swapped.csv")
    p.add_argument("--encoder_ckpt",
                   default="/Users/justin/ultraclear-paper/code/"
                           "checkpoints/tier1_mixed.pt/tier1_sionna.pt")
    p.add_argument("--head_ckpt",
                   default="/Users/justin/ultraclear-paper/code/"
                           "checkpoints/_tmp_asta_persample_garage_05/head.pt")
    p.add_argument("--out_dir", default=None,
                   help="if unset, defaults to head_ckpt's directory")
    p.add_argument("--mode", default="all", choices=["val", "all"])
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    head_ckpt = Path(args.head_ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else head_ckpt.parent
    csv_path = Path(args.csv)
    encoder_ckpt = Path(args.encoder_ckpt)

    print(f"ASTA per-sample inference ({args.mode} mode)")
    print(f"  csv:        {csv_path}")
    print(f"  encoder:    {encoder_ckpt}")
    print(f"  head:       {head_ckpt}")
    print(f"  out_dir:    {out_dir}")
    print(f"  T={args.T}  stride={args.stride}")
    print()
    summary = infer(csv_path, encoder_ckpt, head_ckpt, out_dir,
                     mode=args.mode, T=args.T, stride=args.stride,
                     device=args.device)
    print()
    print("summary:")
    for k, v in summary.items():
        if k == "per_wp_mae":
            print(f"  per-waypoint MAE (worst 5):")
            for wpx, wpy, mae, n in sorted(v, key=lambda r: -r[2])[:5]:
                print(f"    wp=({wpx:+.2f}, {wpy:+.2f})  MAE={mae:.4f} m  n={n}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
