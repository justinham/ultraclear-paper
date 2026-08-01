"""Train the UWB Transformer Distance Correction model on the
office_rack sionna track datasets.

This is the corrector model from
`ultraclear.uwb_transformer_correction.UWBTransformerDistanceCorrection`.

It takes
  - raw CIR (per-anchor, length L=60)
  - anchor (X, Y, Z) positions (An1..An23)
and outputs a per-anchor distance residual (Δd_1, ..., Δd_23)
that, when added to the measured ToA, gives corrected distances.
The position is then recovered by running LSE on the
corrected distances.

Training target
----------------
For each (row, anchor), the target is
    target[i, a] = gt_dist[a] − measured_dist[a]
i.e. the NLOS bias the model should subtract from the
measured distance. We normalize this per-anchor (across the
training set) so the loss is balanced across anchors with
different noise scales.

Eval
----
For each row:
  corrected_dist = measured_dist + model(cir, anchor_xyz)
  pos = LSE(corrected_dist, anchor_xyz)
And compare `pos` to the GT (tag_x, tag_y) in meters.

Run
---
  # Train on big, validate on right:
  python3.11 -m scripts.train_uwb_distance_corrector

  # Quick smoke test:
  python3.11 -m scripts.train_uwb_distance_corrector --n_epochs 3 --T 8 \\
      --out /tmp/_smoke
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.uwb_transformer_correction import (
    UWBTransformerDistanceCorrection,
    load_office_rack_anchor_positions,
)
from scripts.train_uwb_corrector import (
    load_scene,
    make_windows,
    N_ANCHORS, CIR_LENGTH, TAG_Z_FIXED,
)
from scripts.lse_office_rack import lse_2d, project_to_2d


# --------------------------------------------------------------------------- #
# Data loading — same as the position-corrector
# --------------------------------------------------------------------------- #

DEFAULT_DATA_ROOT = Path(
    "/Users/justin/ultraclear-paper/code/data/office_rack_track_sionna"
)
SCENES = ("big", "middle", "right")


# --------------------------------------------------------------------------- #
# Model — a thin wrapper that applies the per-anchor
# UWBTransformerDistanceCorrection to each timestep in a
# T-length window independently.
# --------------------------------------------------------------------------- #

class UWBDistCorrectorWindowed(nn.Module):
    """Applies UWBTransformerDistanceCorrection to each timestep
    in a T-length window independently.

    Inputs:
        cir:    (B, T, N_anchors, CIR_length) float32
        coords: (N_anchors, 3) float32 — shared across the window

    Output:
        delta_d: (B, T, N_anchors) float32 — per-anchor distance
                 correction in meters for each timestep.
    """
    def __init__(self, base: UWBTransformerDistanceCorrection):
        super().__init__()
        self.base = base

    def forward(self, cir: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, T, N, L = cir.shape
        # Flatten the window into the batch dim.
        cir_flat = cir.view(B * T, N, L)                       # (B*T, N, L)
        coords_b = coords.unsqueeze(0).expand(B * T, -1, -1)   # (B*T, N, 3)
        delta_flat = self.base(cir_flat, coords_b)             # (B*T, N)
        return delta_flat.view(B, T, N)


# --------------------------------------------------------------------------- #
# Train + eval
# --------------------------------------------------------------------------- #

def train_and_eval(*,
                   train_csv: Path,
                   val_csv: Path,
                   train_scene: str,
                   val_scene: str,
                   T: int = 32, stride: int = 1, n_epochs: int = 200,
                   lr: float = 1e-3,
                   device: str = "mps",
                   use_anchor: int = 1,
                   out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load anchor positions ----
    anchor_xyz_t = load_office_rack_anchor_positions()  # (23, 3) An1..An23
    anchor_xyz = anchor_xyz_t.cpu().numpy()
    print(f"[anchors] 23 loaded; X range: [{anchor_xyz[:, 0].min()}, "
          f"{anchor_xyz[:, 0].max()}]  Z values: {sorted(set(anchor_xyz[:, 2].tolist()))}")

    # ---- Load train + val ----
    print(f"\n[load] train scene '{train_scene}': {train_csv.name}")
    train = load_scene(train_csv)
    print(f"        X={train['cir'].shape}  Y={train['tag_xy'].shape}  "
          f"(max |CIR|^2 = {float(train['cir'].max()):.3e} — should be ~1e-5 if noise)")
    print(f"[load] val scene   '{val_scene}':   {val_csv.name}")
    val = load_scene(val_csv)
    print(f"        X={val['cir'].shape}  Y={val['tag_xy'].shape}")

    # ---- Mask CIR to keep only the selected anchor's signal ----
    # All other anchors' CIR is set to zero. The model still
    # outputs Δd for all N_ANCHORS, but only the selected
    # anchor's input carries signal — a sanity check that the
    # input modality has any predictive content at all.
    if not (1 <= use_anchor <= N_ANCHORS):
        raise ValueError(f"use_anchor must be in 1..{N_ANCHORS}, got {use_anchor}")
    keep = np.zeros(N_ANCHORS, dtype=bool)
    keep[use_anchor - 1] = True
    n_zeroed = N_ANCHORS - 1
    print(f"[mask] keeping only anchor {use_anchor} CIR; "
          f"zeroing {n_zeroed} other anchors")
    train["cir"][:, ~keep, :] = 0.0
    val["cir"][:, ~keep, :] = 0.0
    kept_power = float(np.abs(train["cir"][:, use_anchor - 1, :]).max())
    print(f"[mask] anchor {use_anchor} |CIR| max on train = {kept_power:.3e}")

    # ---- Build windows ----
    wx_tr_cir, wy_tr_tag, wy_tr_lse = make_windows(
        train["cir"], train["tag_xy"], train["lse_xy"], T, stride,
    )
    wx_va_cir, wy_va_tag, wy_va_lse = make_windows(
        val["cir"], val["tag_xy"], val["lse_xy"], T, stride,
    )
    n_tr = wx_tr_cir.shape[0]
    n_va = wx_va_cir.shape[0]
    print(f"\n[windows] T={T} stride={stride} → train: {n_tr}  val: {n_va}")
    if n_tr == 0 or n_va == 0:
        raise RuntimeError(
            f"Empty window set: train={n_tr}, val={n_va}. "
            f"Check T (current {T}) vs sequence length."
        )

    # ---- Per-anchor target: gt_dist - measured_dist (in meters) ----
    # We need to recompute measured_dist (3D, in meters) and
    # gt_dist (3D, in meters) per row. The CSVs already have
    # these as columns A{i}_measured_dist_m and A{i}_gt_dist_m.
    import pandas as pd
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    meas_cols = [f"A{i}_measured_dist_m" for i in range(1, N_ANCHORS + 1)]
    gt_cols   = [f"A{i}_gt_dist_m"      for i in range(1, N_ANCHORS + 1)]
    meas_train = df_train[meas_cols].to_numpy().astype(np.float32)  # (N, 23)
    meas_val   = df_val[meas_cols].to_numpy().astype(np.float32)
    gt_train   = df_train[gt_cols].to_numpy().astype(np.float32)
    gt_val     = df_val[gt_cols].to_numpy().astype(np.float32)
    bias_train = gt_train - meas_train     # (N, 23)
    bias_val   = gt_val   - meas_val       # (N, 23)

    # Build windowed target
    wy_tr_bias = np.stack([bias_train[s:s + T] for s in range(0, len(bias_train) - T + 1, stride)])
    wy_va_bias = np.stack([bias_val[s:s + T]   for s in range(0, len(bias_val)   - T + 1, stride)])
    # We don't filter on the bias being zero because the bias is
    # never exactly zero in this data (NLOS bias is always
    # present somewhere). The keep_* in make_windows() already
    # dropped any degenerate windows.
    n_tr = wx_tr_cir.shape[0]
    n_va = wx_va_cir.shape[0]

    # ---- Per-anchor z-score normalization (computed on TRAIN only) ----
    # The bias has very different scales per anchor (A20 has 9m
    # mean bias, A23 has 0m). Per-anchor norm gives the loss
    # balanced gradients across anchors.
    flat = wy_tr_bias.reshape(-1, N_ANCHORS)              # (W*T, 23)
    bias_mean = flat.mean(axis=0).astype(np.float32)        # (23,)
    bias_std  = (flat.std(axis=0) + 1e-6).astype(np.float32)  # (23,)
    print(f"[norm]  per-anchor bias mean (m): {bias_mean.tolist()}")
    print(f"[norm]  per-anchor bias std  (m): {bias_std.tolist()}")

    def norm_bias(b):
        return (b - bias_mean) / bias_std

    wy_tr_bias_n = norm_bias(wy_tr_bias)
    wy_va_bias_n = norm_bias(wy_va_bias)

    # ---- To tensors ----
    coords_t = anchor_xyz_t.to(device)                  # (23, 3)
    wx_tr_t   = torch.from_numpy(wx_tr_cir).to(device)  # (W, T, 23, 60)
    wy_tr_bias_t = torch.from_numpy(wy_tr_bias_n).to(device)
    wx_va_t   = torch.from_numpy(wx_va_cir).to(device)
    wy_va_bias_t = torch.from_numpy(wy_va_bias_n).to(device)
    wy_va_lse_t  = torch.from_numpy(wy_va_lse).to(device)
    wy_va_tag_t  = torch.from_numpy(wy_va_tag).to(device)
    wy_tr_lse_t  = torch.from_numpy(wy_tr_lse).to(device)
    wy_tr_tag_t  = torch.from_numpy(wy_tr_tag).to(device)

    # ---- Build model ----
    base = UWBTransformerDistanceCorrection.for_office_rack_cir(
        cir_length=CIR_LENGTH, patch_size=6, embed_dim=64, num_layers=2,
    )
    model = UWBDistCorrectorWindowed(base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[model] n_anchors={N_ANCHORS}  CIR_length={CIR_LENGTH}  "
          f"total params: {n_params:,}")
    print(f"[model] per-window: cir {tuple(wx_tr_t.shape[1:])}  "
          f"bias_target {tuple(wy_tr_bias_t.shape[1:])} (per-anchor Δd in normalized space)")

    # ---- Train ----
    batch_size = 16
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    rng = np.random.default_rng(0)
    history = []

    for ep in range(1, n_epochs + 1):
        model.train()
        perm = rng.permutation(n_tr)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            cir_b = wx_tr_t[idx]                    # (B, T, 23, 60)
            tgt_b = wy_tr_bias_t[idx]               # (B, T, 23) (normalized bias)
            opt.zero_grad()
            pred = model(cir_b, coords_t)            # (B, T, 23)
            loss = crit(pred, tgt_b)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        ep_loss /= max(n_batches, 1)

        if ep == 1 or ep % 20 == 0 or ep == n_epochs:
            model.eval()
            with torch.no_grad():
                tr_n_mae = 0.0
                for i in range(0, n_tr, batch_size):
                    cir_b = wx_tr_t[i:i + batch_size]
                    tgt_b = wy_tr_bias_t[i:i + batch_size]
                    pred_b = model(cir_b, coords_t)
                    tr_n_mae += (pred_b - tgt_b).abs().sum().item()
                tr_n_mae /= (n_tr * T * N_ANCHORS)

                va_n_mae = 0.0
                for i in range(0, n_va, batch_size):
                    cir_b = wx_va_t[i:i + batch_size]
                    tgt_b = wy_va_bias_t[i:i + batch_size]
                    pred_b = model(cir_b, coords_t)
                    va_n_mae += (pred_b - tgt_b).abs().sum().item()
                va_n_mae /= (n_va * T * N_ANCHORS)
            history.append((ep, ep_loss, tr_n_mae, va_n_mae))
            print(f"  ep {ep:3d}  train_mse={ep_loss:.4f}  "
                  f"train_mae_norm={tr_n_mae:.4f}  val_mae_norm={va_n_mae:.4f}")

    # ---- Final eval: un-normalize and run LSE to get position error ----
    # For each (window, t), the model predicts a normalized
    # (Δd_1, ..., Δd_23). Un-normalize, add to measured 3D distance,
    # project to 2D, run LSE → position. Compare to GT.
    model.eval()
    with torch.no_grad():
        # Re-load raw 3D measured distances for the eval.
        meas_train_3d = meas_train        # (N_train, 23)
        meas_val_3d   = meas_val          # (N_val, 23)
        # Build the windowed measured distance (for the LSE step).
        # We use the same stride/T as training.
        def windowed(arr, T, stride):
            return np.stack([arr[s:s + T] for s in range(0, len(arr) - T + 1, stride)])
        wd_meas_tr = windowed(meas_train_3d, T, stride)
        wd_meas_va = windowed(meas_val_3d,   T, stride)
        wd_tag_tr  = wy_tr_tag                 # already (n_tr, T, 2)
        wd_tag_va  = wy_va_tag
        wd_lse_tr  = wy_tr_lse                 # (n_tr, T, 2)
        wd_lse_va  = wy_va_lse

        pred_tr_bias_n_list = []
        pred_va_bias_n_list = []
        for i in range(0, n_tr, batch_size):
            cir_b = wx_tr_t[i:i + batch_size]
            pred_tr_bias_n_list.append(model(cir_b, coords_t).cpu().numpy())
        for i in range(0, n_va, batch_size):
            cir_b = wx_va_t[i:i + batch_size]
            pred_va_bias_n_list.append(model(cir_b, coords_t).cpu().numpy())
        pred_tr_bias_n = np.concatenate(pred_tr_bias_n_list, axis=0)   # (n_tr, T, 23)
        pred_va_bias_n = np.concatenate(pred_va_bias_n_list, axis=0)   # (n_va, T, 23)

    # Un-normalize
    pred_tr_bias = pred_tr_bias_n * bias_std + bias_mean
    pred_va_bias = pred_va_bias_n * bias_std + bias_mean
    # Corrected 3D distances
    wd_corr_tr = wd_meas_tr + pred_tr_bias
    wd_corr_va = wd_meas_va + pred_va_bias

    # Now run LSE per (window, t) on the corrected 3D distances
    # (with the 3D->2D projection). We do this in a Python loop
    # because LSE per-row is not vectorized easily.
    pred_tr_pos = np.zeros((n_tr, T, 2), dtype=np.float64)
    pred_va_pos = np.zeros((n_va, T, 2), dtype=np.float64)
    anchor_z = anchor_xyz[:, 2]
    for w in range(n_tr):
        for t in range(T):
            d_2d = project_to_2d(wd_corr_tr[w, t], anchor_z, TAG_Z_FIXED)
            p, _ = lse_2d(anchor_xyz, d_2d, ref_idx=0)
            pred_tr_pos[w, t] = p
    for w in range(n_va):
        for t in range(T):
            d_2d = project_to_2d(wd_corr_va[w, t], anchor_z, TAG_Z_FIXED)
            p, _ = lse_2d(anchor_xyz, d_2d, ref_idx=0)
            pred_va_pos[w, t] = p

    err_lse_tr = np.linalg.norm(wd_lse_tr - wd_tag_tr, axis=-1)   # (n_tr, T)
    err_corr_tr = np.linalg.norm(pred_tr_pos - wd_tag_tr, axis=-1)
    err_lse_va = np.linalg.norm(wd_lse_va - wd_tag_va, axis=-1)
    err_corr_va = np.linalg.norm(pred_va_pos - wd_tag_va, axis=-1)

    overall_lse_tr  = float(err_lse_tr.mean())
    overall_corr_tr = float(err_corr_tr.mean())
    overall_lse_va  = float(err_lse_va.mean())
    overall_corr_va = float(err_corr_va.mean())

    print()
    print(f"[result] TRAIN per-step MAE:  LSE baseline = {overall_lse_tr:.3f} m  |  "
          f"distance-corrector = {overall_corr_tr:.3f} m  "
          f"(delta = {overall_corr_tr - overall_lse_tr:+.3f} m)")
    print(f"[result] VAL   per-step MAE:  LSE baseline = {overall_lse_va:.3f} m  |  "
          f"distance-corrector = {overall_corr_va:.3f} m  "
          f"(delta = {overall_corr_va - overall_lse_va:+.3f} m)")
    print(f"[result] VAL p50 = {float(np.median(err_corr_va)):.3f} m   "
          f"p95 = {float(np.percentile(err_corr_va, 95)):.3f} m   "
          f"max = {float(err_corr_va.max()):.3f} m")

    # ---- Per-anchor bias prediction quality (un-normalized) ----
    # How well does the model predict the per-anchor bias?
    # bias_pred - bias_true, in meters
    err_bias_tr = pred_tr_bias - wy_tr_bias
    err_bias_va = pred_va_bias - wy_va_bias
    bias_mae_tr = float(np.abs(err_bias_tr).mean())
    bias_mae_va = float(np.abs(err_bias_va).mean())
    print(f"\n[bias]  per-anchor bias prediction MAE:  train = {bias_mae_tr:.3f} m   "
          f"val = {bias_mae_va:.3f} m")

    # ---- Save ckpt ----
    enc_path = out_dir / "encoder.pt"
    head_path = out_dir / "head.pt"
    torch.save({
        "model_state": model.state_dict(),
        "bias_mean": bias_mean,
        "bias_std": bias_std,
        "config": {
            "model_class": "UWBTransformerDistanceCorrection",
            "n_anchors": N_ANCHORS, "cir_length": CIR_LENGTH,
            "T": T, "stride": stride,
            "patch_size": 6, "embed_dim": 64, "n_asta_blocks": 2,
            "use_anchor": use_anchor,
        },
        "anchor_xyz": anchor_xyz,
        "train_scene": train_scene,
        "val_scene": val_scene,
    }, enc_path)
    torch.save(model.base.state_dict(), head_path)
    print(f"[save]  {enc_path}   {head_path}")

    # ---- Save per-step CSV ----
    eval_csv = out_dir / "eval_per_step.csv"
    with open(eval_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "window_idx", "t", "x_lse", "y_lse",
                    "x_corr", "y_corr", "x_true", "y_true",
                    "err_lse_m", "err_corr_m"])
        for wi in range(n_va):
            for t in range(T):
                w.writerow(["val", wi, t,
                            f"{wd_lse_va[wi, t, 0]:.4f}",
                            f"{wd_lse_va[wi, t, 1]:.4f}",
                            f"{pred_va_pos[wi, t, 0]:.4f}",
                            f"{pred_va_pos[wi, t, 1]:.4f}",
                            f"{wd_tag_va[wi, t, 0]:.4f}",
                            f"{wd_tag_va[wi, t, 1]:.4f}",
                            f"{err_lse_va[wi, t]:.4f}",
                            f"{err_corr_va[wi, t]:.4f}"])
        for wi in range(n_tr):
            for t in range(T):
                w.writerow(["train", wi, t,
                            f"{wd_lse_tr[wi, t, 0]:.4f}",
                            f"{wd_lse_tr[wi, t, 1]:.4f}",
                            f"{pred_tr_pos[wi, t, 0]:.4f}",
                            f"{pred_tr_pos[wi, t, 1]:.4f}",
                            f"{wd_tag_tr[wi, t, 0]:.4f}",
                            f"{wd_tag_tr[wi, t, 1]:.4f}",
                            f"{err_lse_tr[wi, t]:.4f}",
                            f"{err_corr_tr[wi, t]:.4f}"])
    print(f"[save]  {eval_csv}")

    # ---- Save training history ----
    hist_csv = out_dir / "history.csv"
    with open(hist_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse_norm", "train_mae_norm", "val_mae_norm"])
        for ep, mse, tr, va in history:
            w.writerow([ep, f"{mse:.6f}", f"{tr:.6f}", f"{va:.6f}"])
    print(f"[save]  {hist_csv}")

    # ---- Figures ----
    # Figure 1: training curves
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    eps = [h[0] for h in history]
    ax.plot(eps, [h[2] for h in history], '-', color='tab:blue',
            label='train MAE (normalized bias)')
    ax.plot(eps, [h[3] for h in history], '-', color='tab:orange',
            label='val MAE (normalized bias)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('per-step MAE (normalized bias)')
    ax.set_title(f"Distance-corrector training curves "
                 f"(train={train_scene}, val={val_scene})")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    curves = out_dir / "training_curves.png"
    fig.savefig(curves, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {curves}")

    # Figure 2: per-anchor bias prediction quality
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    # Train
    ax = axes[0]
    bias_mae_per_anchor_tr = np.abs(err_bias_tr).mean(axis=(0, 1))
    ax.bar(np.arange(N_ANCHORS), bias_mae_per_anchor_tr, color='tab:blue', alpha=0.7)
    ax.set_xlabel("Anchor index (An)")
    ax.set_ylabel("|predicted bias - true bias| (m)")
    ax.set_title(f"TRAIN: per-anchor bias prediction MAE "
                 f"(overall = {bias_mae_tr:.3f} m)")
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    # Val
    ax = axes[1]
    bias_mae_per_anchor_va = np.abs(err_bias_va).mean(axis=(0, 1))
    ax.bar(np.arange(N_ANCHORS), bias_mae_per_anchor_va, color='tab:orange', alpha=0.7)
    ax.set_xlabel("Anchor index (An)")
    ax.set_ylabel("|predicted bias - true bias| (m)")
    ax.set_title(f"VAL: per-anchor bias prediction MAE "
                 f"(overall = {bias_mae_va:.3f} m)")
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    bias_path = out_dir / "per_anchor_bias_mae.png"
    fig.savefig(bias_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {bias_path}")

    # Figure 3: val scatter (GT vs LSE vs corrector)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.scatter(wd_tag_va[..., 0].flatten(), wd_tag_va[..., 1].flatten(),
               c='lime', s=8, alpha=0.5, label='GT')
    ax.scatter(wd_lse_va[..., 0].flatten(), wd_lse_va[..., 1].flatten(),
               c='tab:red', s=8, alpha=0.4,
               label=f'LSE baseline (MAE = {overall_lse_va:.2f} m)')
    ax.scatter(pred_va_pos[..., 0].flatten(), pred_va_pos[..., 1].flatten(),
               c='tab:blue', s=8, alpha=0.5,
               label=f'Distance-corrector (MAE = {overall_corr_va:.2f} m)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"Distance-corrector — VAL ({val_scene})\n"
                 f"LSE {overall_lse_va:.3f} m  |  "
                 f"corrector {overall_corr_va:.3f} m  |  "
                 f"delta {overall_corr_va - overall_lse_va:+.3f} m")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    scat = out_dir / "val_scatter.png"
    fig.savefig(scat, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {scat}")

    return {
        "n_train": int(n_tr),
        "n_val": int(n_va),
        "T": int(T),
        "stride": int(stride),
        "lse_baseline_train_mae_m": float(overall_lse_tr),
        "lse_baseline_val_mae_m": float(overall_lse_va),
        "corrector_train_mae_m": float(overall_corr_tr),
        "corrector_val_mae_m": float(overall_corr_va),
        "delta_train_m": float(overall_corr_tr - overall_lse_tr),
        "delta_val_m": float(overall_corr_va - overall_lse_va),
        "bias_pred_train_mae_m": float(bias_mae_tr),
        "bias_pred_val_mae_m": float(bias_mae_va),
        "val_p50_m": float(np.median(err_corr_va)),
        "val_p95_m": float(np.percentile(err_corr_va, 95)),
        "val_max_m": float(err_corr_va.max()),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--train_scene", choices=SCENES, default="big")
    p.add_argument("--val_scene", choices=SCENES, default="right")
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None,
                   help="torch device (mps/cuda/cpu). Default: auto-pick "
                        "mps if available, else cpu.")
    p.add_argument("--use_anchor", type=int, default=1,
                   help="1-indexed anchor number whose CIR is kept; "
                        "all other anchors' CIR is zeroed. Default 1. "
                        "Range 1..N_ANCHORS.")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: code/checkpoints/"
                        "_tmp_uwb_dist_corrector_<train>_to_<val>)")
    args = p.parse_args()

    train_csv = args.data_root / f"office_rack_track_{args.train_scene}" / \
               "sionna_uwb_track_office_rack.csv"
    val_csv   = args.data_root / f"office_rack_track_{args.val_scene}" / \
               "sionna_uwb_track_office_rack.csv"
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    if not val_csv.exists():
        raise FileNotFoundError(val_csv)
    if args.out is None:
        args.out = (Path("/Users/justin/ultraclear-paper/code/checkpoints")
                    / f"_tmp_uwb_dist_corrector_{args.train_scene}_to_{args.val_scene}")

    print("=" * 70)
    print("UWB Transformer Distance Correction \u2014 training on office_rack sionna track")
    print(f"  train scene: {args.train_scene}  ({train_csv.name})")
    print(f"  val scene:   {args.val_scene}  ({val_csv.name})")
    print(f"  T={args.T}  stride={args.stride}  epochs={args.n_epochs}  lr={args.lr}")
    print(f"  out:         {args.out}")
    print("=" * 70)
    print()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    print(f"  device:       {args.device}")
    print(f"  use_anchor:   {args.use_anchor} (1-indexed; CIR for other "
          f"anchors is zeroed)")

    summary = train_and_eval(
        train_csv=train_csv, val_csv=val_csv,
        train_scene=args.train_scene, val_scene=args.val_scene,
        T=args.T, stride=args.stride,
        n_epochs=args.n_epochs, lr=args.lr,
        device=args.device, use_anchor=args.use_anchor,
        out_dir=args.out,
    )
    print()
    print("summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
