"""Visualize the trained UWB corrector on the office_rack
training scene (default: `big`).

Loads the saved encoder.pt, applies the corrector to every
row of the training scene, and produces a multi-panel figure
showing GT (green), LSE baseline (red), and corrector (blue)
trajectories.

This is intentionally simple — just a sanity-check figure,
not a full eval. For a proper multi-scene eval, see
`scripts/eval_office_rack_all_scenes.py` (which is the
ASTA-equivalent; we could write a corrector-equivalent later
if useful).

Run:
  python3.11 -m scripts.visualize_corrector_training
  python3.11 -m scripts.visualize_corrector_training \\
      --ckpt checkpoints/_tmp_uwb_corrector_big_to_right/encoder.pt \\
      --scene big
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ultraclear.uwb_transformer_correction import (
    UWBTransformerCorrection,
    load_office_rack_anchor_positions,
)
from scripts.train_uwb_corrector import (
    UWBCorrectorWindowed,
    load_scene,
    parse_cir_row,
    N_ANCHORS, CIR_LENGTH, TAG_Z_FIXED,
)

DEFAULT_DATA_ROOT = Path(
    "/Users/justin/ultraclear-paper/code/data/office_rack_track_sionna"
)
DEFAULT_CKPT = Path(
    "/Users/justin/ultraclear-paper/code/checkpoints/"
    "_tmp_uwb_corrector_big_to_right/encoder.pt"
)
DEFAULT_OUT = Path(
    "/Users/justin/ultraclear-paper/code/checkpoints/"
    "_tmp_uwb_corrector_big_to_right/_training_viz"
)
SCENES = ("big", "middle", "right")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--scene", choices=SCENES, default="big",
                   help="scene to visualize (default: training scene 'big')")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.out is None:
        args.out = args.ckpt.parent / "_training_viz"
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)
    if not (args.data_root / f"office_rack_track_{args.scene}").exists():
        raise FileNotFoundError(args.data_root / f"office_rack_track_{args.scene}")

    # ---- Load ckpt ----
    bundle = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = bundle["config"]
    norm_mean = np.asarray(bundle["norm_mean"], dtype=np.float32)
    norm_std = np.asarray(bundle["norm_std"], dtype=np.float32)
    train_scene_ckpt = bundle.get("train_scene", "?")
    print(f"[ckpt]  {args.ckpt}")
    print(f"        trained on '{train_scene_ckpt}', T={cfg['T']} stride={cfg['stride']}  "
          f"predict_z={cfg['predict_z']}")
    print(f"        norm: mean={norm_mean.tolist()}  std={norm_std.tolist()}")

    # ---- Rebuild model + load weights ----
    base = UWBTransformerCorrection.for_office_rack_cir(
        cir_length=cfg.get("cir_length", CIR_LENGTH),
        patch_size=cfg.get("patch_size", 6),
        embed_dim=cfg.get("embed_dim", 64),
        num_layers=cfg.get("n_asta_blocks", cfg.get("num_layers", 2)),
    )
    model = UWBCorrectorWindowed(base, predict_z=cfg.get("predict_z", False))
    # The training script saved model.state_dict() (the wrapped
    # model), so its keys are prefixed with `base.`. We can load
    # straight into the wrapper.
    missing, unexpected = model.load_state_dict(bundle["model_state"], strict=False)
    if missing or unexpected:
        print(f"[load]  missing={len(missing)}  unexpected={len(unexpected)}")
    model.eval()

    # ---- Load scene data ----
    scene = load_scene(args.data_root / f"office_rack_track_{args.scene}"
                              / "sionna_uwb_track_office_rack.csv")
    cir = scene["cir"]            # (N, 23, 60)
    tag_xy = scene["tag_xy"]      # (N, 2)
    lse_xy = scene["lse_xy"]      # (N, 2)
    print(f"\n[data]  scene '{args.scene}': {len(cir)} rows")
    print(f"        CIR shape: {cir.shape}  "
          f"(max |CIR|^2 = {float(cir.max()):.3e} — should be ~1e-5 if noise)")
    print(f"        LSE MAE: {float(np.linalg.norm(lse_xy - tag_xy, axis=-1).mean()):.3f} m")

    # ---- Per-row prediction ----
    # Apply the model per-row (no windowing — we want per-step
    # predictions for the trajectory figure).
    coords_t = load_office_rack_anchor_positions().to(args.device)  # (23, 3)
    coords_b = coords_t.unsqueeze(0)  # (1, 23, 3) — will be expanded inside UWBCorrectorWindowed.forward

    N = len(cir)
    pred_resid = np.zeros((N, 2), dtype=np.float32)   # (N, 2) in normalized space
    with torch.no_grad():
        for i in range(N):
            # Model expects (B, T, N_anchors, CIR_length). Single
            # row → T=1, B=1.
            cir_t = torch.from_numpy(cir[i:i+1]).to(args.device).unsqueeze(1)  # (1, 1, 23, 60)
            lse_3d = torch.tensor([[[lse_xy[i, 0], lse_xy[i, 1], TAG_Z_FIXED]]],
                                 dtype=torch.float32, device=args.device)  # (1, 1, 3)
            out = model(cir_t, coords_t, lse_3d)  # (1, 1, 2)
            pred_resid[i] = out[0, 0].cpu().numpy()

    # Un-normalize
    pred_resid_m = pred_resid * norm_std + norm_mean
    pred_corr = lse_xy + pred_resid_m

    # Per-step errors
    err_lse = np.linalg.norm(lse_xy - tag_xy, axis=-1)
    err_corr = np.linalg.norm(pred_corr - tag_xy, axis=-1)
    lse_mae = float(err_lse.mean())
    corr_mae = float(err_corr.mean())
    print(f"\n[result] LSE baseline MAE:    {lse_mae:.3f} m  "
          f"(p50={float(np.median(err_lse)):.3f}  p95={float(np.percentile(err_lse, 95)):.3f}  "
          f"max={float(err_lse.max()):.3f})")
    print(f"[result] Corrector MAE:        {corr_mae:.3f} m  "
          f"(p50={float(np.median(err_corr)):.3f}  p95={float(np.percentile(err_corr, 95)):.3f}  "
          f"max={float(err_corr.max()):.3f})")
    print(f"[result] Delta:               {corr_mae - lse_mae:+.3f} m  "
          f"({'+' if corr_mae < lse_mae else '-'}improvement)")

    # Per-axis errors
    lse_axis = np.abs(lse_xy - tag_xy).mean(axis=0)
    corr_axis = np.abs(pred_corr - tag_xy).mean(axis=0)
    print(f"[axis]   LSE   per-axis MAE:  x = {lse_axis[0]:.3f} m   y = {lse_axis[1]:.3f} m")
    print(f"[axis]   Corr. per-axis MAE:  x = {corr_axis[0]:.3f} m   y = {corr_axis[1]:.3f} m")

    # ---- Figure 1: trajectory (GT vs LSE vs corrector) ----
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    # GT trajectory (green line + dots)
    ax.plot(tag_xy[:, 0], tag_xy[:, 1], '-', color='tab:green',
            alpha=0.7, lw=1.4, label='GT', zorder=2)
    ax.scatter(tag_xy[:, 0], tag_xy[:, 1], c='tab:green', s=6, alpha=0.6, zorder=3)
    # GT start/end
    ax.scatter([tag_xy[0, 0]], [tag_xy[0, 1]], c='tab:green', s=120,
               marker='o', edgecolor='black', linewidth=1.0, zorder=5)
    ax.scatter([tag_xy[-1, 0]], [tag_xy[-1, 1]], c='tab:green', s=120,
               marker='X', edgecolor='black', linewidth=1.0, zorder=5)

    # LSE baseline (red dashed line)
    ax.plot(lse_xy[:, 0], lse_xy[:, 1], '--', color='tab:red',
            alpha=0.6, lw=1.0, label=f'LSE baseline (MAE = {lse_mae:.2f} m)', zorder=2)
    ax.scatter(lse_xy[:, 0], lse_xy[:, 1], c='tab:red', s=4, alpha=0.4, zorder=3)

    # Corrector (blue line + dots)
    ax.plot(pred_corr[:, 0], pred_corr[:, 1], '-', color='tab:blue',
            alpha=0.7, lw=1.0, label=f'Corrector (MAE = {corr_mae:.2f} m)', zorder=2)
    ax.scatter(pred_corr[:, 0], pred_corr[:, 1], c='tab:blue', s=4, alpha=0.5, zorder=3)

    # Anchor geometry in the background
    anchor_xyz = coords_t.cpu().numpy()
    ceiling = anchor_xyz[anchor_xyz[:, 2] > 1.0]
    floor = anchor_xyz[anchor_xyz[:, 2] <= 1.0]
    ax.scatter(ceiling[:, 0], ceiling[:, 1], marker='^', s=70,
               c='lightblue', edgecolor='gray', linewidth=0.5, zorder=1, alpha=0.6)
    ax.scatter(floor[:, 0], floor[:, 1], marker='s', s=70,
               c='lightcoral', edgecolor='gray', linewidth=0.5, zorder=1, alpha=0.6)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"UWB corrector — scene '{args.scene}' (training set)\n"
        f"LSE baseline MAE = {lse_mae:.3f} m  →  corrector MAE = {corr_mae:.3f} m  "
        f"({corr_mae - lse_mae:+.3f} m delta)\n"
        f"n = {N} samples, T_per_window = {cfg['T']}, predict_z = {cfg['predict_z']}"
    )
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim(-17, 17)
    ax.set_ylim(-7, 7)
    # Manual legend
    legend_elements = [
        Line2D([0], [0], color='tab:green', lw=1.4, label='GT'),
        Line2D([0], [0], color='tab:red', lw=1.0, linestyle='--',
               label=f'LSE baseline (MAE {lse_mae:.2f} m)'),
        Line2D([0], [0], color='tab:blue', lw=1.0,
               label=f'Corrector (MAE {corr_mae:.2f} m)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

    traj_path = args.out / f"{args.scene}_corrector_vs_lse_vs_gt.png"
    fig.savefig(traj_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[save]  {traj_path}")

    # ---- Figure 2: per-step error time series (LSE vs corrector) ----
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(N)
    ax.plot(x, err_lse, '-', color='tab:red', alpha=0.5, lw=0.7,
            label=f'LSE baseline (mean = {lse_mae:.3f} m)')
    ax.plot(x, err_corr, '-', color='tab:blue', alpha=0.7, lw=0.9,
            label=f'Corrector (mean = {corr_mae:.3f} m)')
    ax.axhline(lse_mae, color='tab:red', linestyle=':', lw=0.8, alpha=0.6)
    ax.axhline(corr_mae, color='tab:blue', linestyle=':', lw=0.8, alpha=0.6)
    ax.set_xlabel('sample index (chronological)')
    ax.set_ylabel('per-step Euclidean error (m)')
    ax.set_title(
        f"Per-step error: LSE vs corrector — scene '{args.scene}'\n"
        f"delta MAE = {corr_mae - lse_mae:+.3f} m  "
        f"({(corr_mae - lse_mae) / lse_mae * 100:+.1f}% relative)"
    )
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle=':')
    err_path = args.out / f"{args.scene}_err_vs_t.png"
    fig.savefig(err_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {err_path}")

    # ---- Figure 3: per-step scatter of LSE err vs corrector err ----
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.scatter(err_lse, err_corr, s=4, alpha=0.5, c='tab:blue')
    # y=x line
    lim_max = max(float(err_lse.max()), float(err_corr.max()))
    ax.plot([0, lim_max], [0, lim_max], '--', color='gray', lw=1, alpha=0.6,
            label='y = x (no change)')
    # counts above/below the line
    above = int((err_corr > err_lse).sum())
    below = int((err_corr < err_lse).sum())
    ax.set_xlabel("LSE baseline error (m)")
    ax.set_ylabel("Corrector error (m)")
    ax.set_title(
        f"Per-step error: corrector vs LSE — scene '{args.scene}'\n"
        f"Below diagonal = corrector better ({below} / {N}, {100*below/N:.1f}%)  |  "
        f"above = corrector worse ({above} / {N}, {100*above/N:.1f}%)"
    )
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper left')
    scatter_path = args.out / f"{args.scene}_err_scatter_lse_vs_corr.png"
    fig.savefig(scatter_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {scatter_path}")

    # ---- Figure 4: per-axis bar chart ----
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    methods = ['LSE baseline', 'Corrector']
    x_mae = [lse_axis[0], corr_axis[0]]
    y_mae = [lse_axis[1], corr_axis[1]]
    x_pos = np.arange(len(methods))
    w = 0.35
    ax.bar(x_pos - w/2, x_mae, w, color='tab:blue', alpha=0.7, label='x MAE (m)')
    ax.bar(x_pos + w/2, y_mae, w, color='tab:orange', alpha=0.7, label='y MAE (m)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_ylabel('per-axis MAE (m)')
    ax.set_title(
        f"Per-axis MAE — scene '{args.scene}'\n"
        f"LSE: x={lse_axis[0]:.3f} y={lse_axis[1]:.3f}  |  "
        f"Corrector: x={corr_axis[0]:.3f} y={corr_axis[1]:.3f}"
    )
    ax.legend()
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    bar_path = args.out / f"{args.scene}_per_axis_mae.png"
    fig.savefig(bar_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {bar_path}")

    # ---- Save summary CSV ----
    summary_csv = args.out / f"{args.scene}_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "method", "mae_m", "p50_m", "p95_m", "max_m",
                    "mae_x_m", "mae_y_m", "delta_vs_lse_m"])
        w.writerow([args.scene, "LSE baseline",
                    f"{lse_mae:.4f}",
                    f"{float(np.median(err_lse)):.4f}",
                    f"{float(np.percentile(err_lse, 95)):.4f}",
                    f"{float(err_lse.max()):.4f}",
                    f"{lse_axis[0]:.4f}", f"{lse_axis[1]:.4f}",
                    "0.000"])
        w.writerow([args.scene, "Corrector",
                    f"{corr_mae:.4f}",
                    f"{float(np.median(err_corr)):.4f}",
                    f"{float(np.percentile(err_corr, 95)):.4f}",
                    f"{float(err_corr.max()):.4f}",
                    f"{corr_axis[0]:.4f}", f"{corr_axis[1]:.4f}",
                    f"{corr_mae - lse_mae:+.4f}"])
    print(f"[save]  {summary_csv}")

    print()
    print(f"all figures written to {args.out}/")


if __name__ == "__main__":
    main()
