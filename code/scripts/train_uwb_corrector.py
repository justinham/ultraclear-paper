"""Train the UWB Transformer Correction model on the
office_rack sionna track datasets.

This is the corrector model from
`ultraclear/uwb_transformer_correction.py`. It takes
  - raw CIR (per-anchor, length L=60)
  - anchor (X, Y, Z) positions (An1..An23)
  - an initial LSE position estimate (x, y, z)
and outputs a residual (Δx, Δy, Δz) to add to the LSE
estimate. The training target is the residual between
GT and LSE on the training scene.

Pipeline
--------
1. Load the office_rack CSV for the training scene (default
   `big`). It already has `lse_est_x, lse_est_y` columns from
   the LSE baseline (see `scripts/lse_office_rack.py`).
2. For each row, parse the 23 CIRs (`A{i}_cir_I` + `A{i}_cir_Q`
   → |CIR|^2) into a length-60 complex-magnitude vector per
   anchor. (NB: as of 2026-07-23 the office_rack CIRs are
   essentially zero — max |CIR|^2 < 1e-5 — so this just gives
   the model zeros to chew on. See memory 2026-07-23.)
3. Build a sliding-window dataset (T superframes, stride
   configurable). Each window has
     X = (T, 23, 60)  — CIR magnitudes
     anchor_xyz (23, 3)  — shared across the window
     lse_pred (T, 2)  — per-timestep LSE estimates
   Target Y = (T, 2)  — per-timestep GT (tag_x, tag_y).
   (Optional: predict z too, see `--predict_z`.)
4. Train ASTA-style: encoder + per-timestep residual head,
   with MSE on (Δx, Δy) between predicted residual and
   (gt - lse_pred).
5. Evaluate on a held-out scene (default `right`).

Run
---
  # Train on big, validate on right (the ASTA default pair):
  python3.11 -m scripts.train_uwb_corrector

  # Quick smoke test:
  python3.11 -m scripts.train_uwb_corrector --n_epochs 3 --T 8 --out /tmp/_smoke
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
    UWBTransformerCorrection,
    load_office_rack_anchor_positions,
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

N_ANCHORS = 23
CIR_LENGTH = 60   # confirmed 2026-07-23 13:48 EDT: each row has
                  # 23 anchors × 60 I-taps + 60 Q-taps.
TAG_Z_FIXED = 1.0

DEFAULT_DATA_ROOT = Path(
    "/Users/justin/ultraclear-paper/code/data/office_rack_track_sionna"
)
SCENES = ("big", "middle", "right")


def parse_cir_row(df_row, anchor_index: int) -> np.ndarray:
    """Parse one row's CIR for one anchor into a length-60
    real-valued vector: |CIR|^2 = I^2 + Q^2 (per-tap power).

    Returns (60,) float32. If the I/Q strings are empty/NaN,
    the whole vector is zeros. (Matches the ASTA trainer's
    convention of zero-filling missing data.)
    """
    i_str = df_row[f"A{anchor_index}_cir_I"]
    q_str = df_row[f"A{anchor_index}_cir_Q"]
    try:
        cir_i = np.array(str(i_str).split(';'), dtype=np.float32)
        cir_q = np.array(str(q_str).split(';'), dtype=np.float32)
    except (ValueError, AttributeError):
        return np.zeros(CIR_LENGTH, dtype=np.float32)
    if cir_i.size != CIR_LENGTH or cir_q.size != CIR_LENGTH:
        # Wrong length: pad or truncate to CIR_LENGTH.
        out_i = np.zeros(CIR_LENGTH, dtype=np.float32)
        out_q = np.zeros(CIR_LENGTH, dtype=np.float32)
        n = min(cir_i.size, CIR_LENGTH)
        out_i[:n] = cir_i[:n]
        out_q[:n] = cir_q[:n]
        cir_i, cir_q = out_i, out_q
    return cir_i**2 + cir_q**2   # per-tap |CIR|^2


def load_scene(csv_path: Path) -> dict:
    """Load a full scene CSV.

    Returns a dict with:
      - 'cir':      (N, 23, 60) float32 — |CIR|^2 per (row, anchor, tap)
      - 'tag_xy':   (N, 2)     float32 — (tag_x, tag_y)
      - 'lse_xy':   (N, 2)     float32 — (lse_est_x, lse_est_y)
                                (zeros if columns missing)
      - 'nlos':     (N, 23)    int8     — 1 = NLOS, 0 = LOS
      - 'df':       the raw pandas DataFrame (for debugging)
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    n = len(df)

    cir = np.zeros((n, N_ANCHORS, CIR_LENGTH), dtype=np.float32)
    for i in range(n):
        row = df.iloc[i]
        for a in range(1, N_ANCHORS + 1):
            cir[i, a - 1] = parse_cir_row(row, a)

    tag_xy = np.stack([df["tag_x"].to_numpy(),
                       df["tag_y"].to_numpy()], axis=1).astype(np.float32)

    if "lse_est_x" in df.columns and "lse_est_y" in df.columns:
        lse_xy = np.stack([df["lse_est_x"].to_numpy(),
                           df["lse_est_y"].to_numpy()], axis=1).astype(np.float32)
    else:
        print(f"[warn] {csv_path.name} has no lse_est_x/lse_est_y columns; "
              f"using (0, 0) as the LSE estimate")
        lse_xy = np.zeros((n, 2), dtype=np.float32)

    nlos_cols = [f"A{i}_los_nlos" for i in range(1, N_ANCHORS + 1)]
    nlos = df[nlos_cols].to_numpy().astype(np.int8)

    return {"cir": cir, "tag_xy": tag_xy, "lse_xy": lse_xy,
            "nlos": nlos, "df": df}


def make_windows(cir: np.ndarray, tag_xy: np.ndarray, lse_xy: np.ndarray,
                 T: int, stride: int = 1):
    """Strided sliding windows. With stride=1 (default) we get a
    rotation window — every (t, t+1, ..., t+T-1) slice of the
    sequence, so a length-N series gives N-T+1 windows.

    Drops any window where ALL T*23*CIR_LENGTH values are
    zero (the CIR is noise = zero in the current data, so
    this only catches truly degenerate rows; for real CIR data
    this check might be too strict — we may want to revise).
    """
    N = len(cir)
    if N < T:
        empty_cir = np.zeros((0, T, N_ANCHORS, CIR_LENGTH), dtype=np.float32)
        empty_xy  = np.zeros((0, T, 2), dtype=np.float32)
        return empty_cir, empty_xy
    starts = list(range(0, N - T + 1, stride))
    wx_cir = np.stack([cir[s:s + T] for s in starts])    # (W, T, 23, 60)
    wy_tag = np.stack([tag_xy[s:s + T] for s in starts])  # (W, T, 2)
    wy_lse = np.stack([lse_xy[s:s + T] for s in starts])  # (W, T, 2)
    # Drop windows where the per-timestep residual is also zero
    # (would mean the LSE perfectly matches GT, so the model
    # has nothing to learn from — but in our data the LSE has
    # a 5m MAE so this should never trigger).
    keep = np.any(np.linalg.norm(wy_tag - wy_lse, axis=-1) > 0, axis=1)
    return wx_cir[keep], wy_tag[keep], wy_lse[keep]


# --------------------------------------------------------------------------- #
# Model: a thin wrapper that adapts UWBTransformerCorrection (which
# was designed for single-timestep inputs) to a sliding window
# of T timesteps. We unroll the per-timestep loop in the
# forward pass.
# --------------------------------------------------------------------------- #

class UWBCorrectorWindowed(nn.Module):
    """Applies UWBTransformerCorrection to each timestep in a
    T-length window independently, using the same shared
    transformer weights. The window is just a way to give
    the model temporal context (we could add it later via
    a temporal transformer, but for now per-timestep is the
    simplest).

    Inputs:
        cir:      (B, T, N_anchors, CIR_length) float32
        coords:   (N_anchors, 3) float32 — shared
        lse_pred: (B, T, 3)        float32
                  (we only use [:, :, :2] for 2D; pass z=1.0
                   in the third column for 3D mode)

    Output:
        delta_pos: (B, T, 2) float32  — residual to add to lse_pred[:, :, :2]
                                     (or (B, T, 3) if --predict_z)
    """
    def __init__(self, base: UWBTransformerCorrection, predict_z: bool = False):
        super().__init__()
        self.base = base
        self.predict_z = predict_z

    def forward(self, cir: torch.Tensor, coords: torch.Tensor,
                lse_pred: torch.Tensor) -> torch.Tensor:
        B, T, N, L = cir.shape
        # Flatten the window into the batch dim
        cir_flat = cir.view(B * T, N, L)                       # (B*T, N, L)
        coords_b = coords.unsqueeze(0).expand(B * T, -1, -1)   # (B*T, N, 3)
        lse_flat = lse_pred.view(B * T, -1)                    # (B*T, 3)
        if not self.predict_z:
            # Set the z column to 1.0 (known tag elevation) and
            # only predict Δx, Δy. The base model still outputs
            # 3 deltas, but we'll only use the first 2.
            lse_flat = lse_flat.clone()
            lse_flat[:, 2] = TAG_Z_FIXED
        delta_flat = self.base(cir_flat, coords_b, lse_flat)   # (B*T, 3)
        if not self.predict_z:
            delta = delta_flat[:, :2]                          # (B*T, 2)
        else:
            delta = delta_flat                                  # (B*T, 3)
        return delta.view(B, T, -1)


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
                   predict_z: bool = False,
                   device: str = "cpu",
                   out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load anchor positions (always 23) ----
    anchor_xyz_t = load_office_rack_anchor_positions()  # (23, 3) An1..An23
    anchor_xyz = anchor_xyz_t.cpu().numpy()
    print(f"[anchors] 23 loaded; X range: [{anchor_xyz[:, 0].min()}, "
          f"{anchor_xyz[:, 0].max()}]  Z values: {sorted(set(anchor_xyz[:, 2].tolist()))}")

    # ---- Load train + val ----
    print(f"\n[load] train scene '{train_scene}': {train_csv.name}")
    train = load_scene(train_csv)
    print(f"        X={train['cir'].shape}  Y={train['tag_xy'].shape}  "
          f"LSE={train['lse_xy'].shape}  ({(~np.isnan(train['cir'])).sum()} non-zero CIR values)")
    print(f"[load] val scene   '{val_scene}':   {val_csv.name}")
    val = load_scene(val_csv)
    print(f"        X={val['cir'].shape}  Y={val['tag_xy'].shape}  "
          f"LSE={val['lse_xy'].shape}  ({(~np.isnan(val['cir'])).sum()} non-zero CIR values)")

    # ---- Build windows ----
    wx_tr_cir, wy_tr_tag, wy_tr_lse = make_windows(
        train["cir"], train["tag_xy"], train["lse_xy"], T, stride,
    )
    wx_va_cir, wy_va_tag, wy_va_lse = make_windows(
        val["cir"], val["tag_xy"], val["lse_xy"], T, stride,
    )
    # Per-timestep target: the residual (gt - lse) that the
    # corrector should output. This is what the model learns
    # to predict.
    wy_tr_target = wy_tr_tag - wy_tr_lse              # (W, T, 2)
    wy_va_target = wy_va_tag - wy_va_lse              # (W, T, 2)

    n_tr = wx_tr_cir.shape[0]
    n_va = wx_va_cir.shape[0]
    print(f"\n[windows] T={T} stride={stride} → train: {n_tr}  val: {n_va}")
    if n_tr == 0 or n_va == 0:
        raise RuntimeError(
            f"Empty window set: train={n_tr}, val={n_va}. "
            f"Check T (current {T}) vs sequence length."
        )

    # Per-axis z-score on the residual (train only)
    flat = wy_tr_target.reshape(-1, 2)               # (W*T, 2)
    norm_mean = flat.mean(axis=0).astype(np.float32)  # (2,)
    norm_std  = (flat.std(axis=0) + 1e-6).astype(np.float32)
    print(f"[norm]  residual mean = {norm_mean}  std = {norm_std}")

    def norm_resid(r):
        return (r - norm_mean) / norm_std

    wy_tr_target_n = norm_resid(wy_tr_target)
    wy_va_target_n = norm_resid(wy_va_target)

    # ---- To tensors ----
    coords_t = anchor_xyz_t.to(device)                # (23, 3)
    wx_tr_t = torch.from_numpy(wx_tr_cir).to(device)  # (W, T, 23, 60)
    wy_tr_lse_t = torch.from_numpy(wy_tr_lse).to(device)
    wy_tr_tgt_t = torch.from_numpy(wy_tr_target_n).to(device)
    wx_va_t = torch.from_numpy(wx_va_cir).to(device)
    wy_va_lse_t = torch.from_numpy(wy_va_lse).to(device)
    wy_va_tgt_t = torch.from_numpy(wy_va_target_n).to(device)

    # ---- Build model ----
    base = UWBTransformerCorrection.for_office_rack_cir(
        cir_length=CIR_LENGTH, patch_size=6, embed_dim=64, num_layers=2,
    )
    model = UWBCorrectorWindowed(base, predict_z=predict_z).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[model] n_anchors={N_ANCHORS}  n_channels=CIR (length {CIR_LENGTH})  "
          f"predict_z={predict_z}  total params: {n_params:,}")
    print(f"[model] per-window shape: cir {tuple(wx_tr_t.shape[1:])}  "
          f"lse_pred {tuple(wy_tr_lse_t.shape[1:])}  target {tuple(wy_tr_tgt_t.shape[1:])}")

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
            cir_b  = wx_tr_t[idx]               # (B, T, 23, 60)
            lse_b  = wy_tr_lse_t[idx]            # (B, T, 2)
            tgt_b  = wy_tr_tgt_t[idx]           # (B, T, 2)  (normalized residual)
            lse_3d = torch.cat([lse_b, torch.full_like(lse_b[:, :, :1], TAG_Z_FIXED)],
                               dim=-1)            # (B, T, 3)
            opt.zero_grad()
            pred = model(cir_b, coords_t, lse_3d)  # (B, T, 2)  (normalized residual)
            loss = crit(pred, tgt_b)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        ep_loss /= max(n_batches, 1)

        if ep == 1 or ep % 20 == 0 or ep == n_epochs:
            model.eval()
            with torch.no_grad():
                # Train MAE in normalized space
                tr_n_mae = 0.0
                for i in range(0, n_tr, batch_size):
                    cir_b  = wx_tr_t[i:i + batch_size]
                    lse_b  = wy_tr_lse_t[i:i + batch_size]
                    tgt_b  = wy_tr_tgt_t[i:i + batch_size]
                    lse_3d = torch.cat([lse_b, torch.full_like(lse_b[:, :, :1], TAG_Z_FIXED)], dim=-1)
                    pred = model(cir_b, coords_t, lse_3d)
                    tr_n_mae += (pred - tgt_b).abs().sum().item()
                tr_n_mae /= (n_tr * T * 2)

                # Val MAE in normalized space
                va_n_mae = 0.0
                pred_va_list = []
                for i in range(0, n_va, batch_size):
                    cir_b  = wx_va_t[i:i + batch_size]
                    lse_b  = wy_va_lse_t[i:i + batch_size]
                    tgt_b  = wy_va_tgt_t[i:i + batch_size]
                    lse_3d = torch.cat([lse_b, torch.full_like(lse_b[:, :, :1], TAG_Z_FIXED)], dim=-1)
                    pred_b = model(cir_b, coords_t, lse_3d)
                    pred_va_list.append(pred_b.cpu().numpy())
                    va_n_mae += (pred_b - tgt_b).abs().sum().item()
                va_n_mae /= (n_va * T * 2)
                pred_va_n = np.concatenate(pred_va_list, axis=0)
            history.append((ep, ep_loss, tr_n_mae, va_n_mae))
            print(f"  ep {ep:3d}  train_mse={ep_loss:.4f}  "
                  f"train_mae_norm={tr_n_mae:.4f}  val_mae_norm={va_n_mae:.4f}")

    # ---- Final eval (un-normalize back to meters) ----
    model.eval()
    with torch.no_grad():
        # Train predictions (in normalized residual space)
        pred_tr_list = []
        for i in range(0, n_tr, batch_size):
            cir_b  = wx_tr_t[i:i + batch_size]
            lse_b  = wy_tr_lse_t[i:i + batch_size]
            lse_3d = torch.cat([lse_b, torch.full_like(lse_b[:, :, :1], TAG_Z_FIXED)], dim=-1)
            pred_tr_list.append(model(cir_b, coords_t, lse_3d).cpu().numpy())
        pred_tr_n = np.concatenate(pred_tr_list, axis=0)   # (n_tr, T, 2)
        pred_va_n_local = pred_va_n                          # (n_va, T, 2)

    # Un-normalize: pred_resid_m = pred_resid_norm * norm_std + norm_mean
    pred_tr_resid = pred_tr_n * norm_std + norm_mean       # (n_tr, T, 2) meters
    pred_va_resid = pred_va_n_local * norm_std + norm_mean # (n_va, T, 2) meters
    # Corrected LSE = lse_pred + residual
    pred_tr_corr = wy_tr_lse + pred_tr_resid              # (n_tr, T, 2)
    pred_va_corr = wy_va_lse + pred_va_resid              # (n_va, T, 2)

    err_tr = np.linalg.norm(pred_tr_corr - wy_tr_tag, axis=-1)  # (n_tr, T)
    err_va = np.linalg.norm(pred_va_corr - wy_va_tag, axis=-1)  # (n_va, T)
    err_lse_tr = np.linalg.norm(wy_tr_lse - wy_tr_tag, axis=-1)
    err_lse_va = np.linalg.norm(wy_va_lse - wy_va_tag, axis=-1)

    # Per-window MAE
    per_win_tr = err_tr.mean(axis=1)
    per_win_va = err_va.mean(axis=1)
    overall_tr = float(err_tr.mean())
    overall_va = float(err_va.mean())
    overall_lse_tr = float(err_lse_tr.mean())
    overall_lse_va = float(err_lse_va.mean())
    print()
    print(f"[result] TRAIN per-step MAE:  LSE baseline = {overall_lse_tr:.3f} m  |  "
          f"corrector = {overall_tr:.3f} m  "
          f"(delta = {overall_tr - overall_lse_tr:+.3f} m)")
    print(f"[result] VAL   per-step MAE:  LSE baseline = {overall_lse_va:.3f} m  |  "
          f"corrector = {overall_va:.3f} m  "
          f"(delta = {overall_va - overall_lse_va:+.3f} m)")
    print(f"[result] VAL p50 = {float(np.median(err_va)):.3f} m   "
          f"p95 = {float(np.percentile(err_va, 95)):.3f} m   "
          f"max = {float(err_va.max()):.3f} m")

    # ---- Save ckpt ----
    enc_path = out_dir / "encoder.pt"
    head_path = out_dir / "head.pt"
    torch.save({
        "model_state": model.state_dict(),
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "config": {
            "n_anchors": N_ANCHORS, "cir_length": CIR_LENGTH,
            "predict_z": predict_z, "T": T, "stride": stride,
            "patch_size": 6, "embed_dim": 64, "n_asta_blocks": 2,
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
        for wi in range(pred_va_corr.shape[0]):
            for t in range(pred_va_corr.shape[1]):
                w.writerow(["val", wi, t,
                            f"{wy_va_lse[wi, t, 0]:.4f}",
                            f"{wy_va_lse[wi, t, 1]:.4f}",
                            f"{pred_va_corr[wi, t, 0]:.4f}",
                            f"{pred_va_corr[wi, t, 1]:.4f}",
                            f"{wy_va_tag[wi, t, 0]:.4f}",
                            f"{wy_va_tag[wi, t, 1]:.4f}",
                            f"{err_lse_va[wi, t]:.4f}",
                            f"{err_va[wi, t]:.4f}"])
        for wi in range(pred_tr_corr.shape[0]):
            for t in range(pred_tr_corr.shape[1]):
                w.writerow(["train", wi, t,
                            f"{wy_tr_lse[wi, t, 0]:.4f}",
                            f"{wy_tr_lse[wi, t, 1]:.4f}",
                            f"{pred_tr_corr[wi, t, 0]:.4f}",
                            f"{pred_tr_corr[wi, t, 1]:.4f}",
                            f"{wy_tr_tag[wi, t, 0]:.4f}",
                            f"{wy_tr_tag[wi, t, 1]:.4f}",
                            f"{err_lse_tr[wi, t]:.4f}",
                            f"{err_tr[wi, t]:.4f}"])
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
    tr_n_mae_hist = [h[2] for h in history]
    va_n_mae_hist = [h[3] for h in history]
    ax.plot(eps, tr_n_mae_hist, '-', color='tab:blue', label='train MAE (normalized)')
    ax.plot(eps, va_n_mae_hist, '-', color='tab:orange', label='val MAE (normalized)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('per-step MAE (normalized residual)')
    ax.set_title(f"UWB corrector training curves (train={train_scene}, val={val_scene})")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    curves = out_dir / "training_curves.png"
    fig.savefig(curves, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {curves}")

    # Figure 2: val scatter (GT, LSE, corrected)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.scatter(wy_va_tag[..., 0].flatten(), wy_va_tag[..., 1].flatten(),
               c='lime', s=8, alpha=0.5, label='GT')
    ax.scatter(wy_va_lse[..., 0].flatten(), wy_va_lse[..., 1].flatten(),
               c='tab:red', s=8, alpha=0.4, label='LSE baseline')
    ax.scatter(pred_va_corr[..., 0].flatten(), pred_va_corr[..., 1].flatten(),
               c='tab:blue', s=8, alpha=0.5, label='Corrector (LSE + residual)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"UWB corrector — VAL ({val_scene})\n"
                 f"LSE baseline MAE = {overall_lse_va:.3f} m  |  "
                 f"Corrector MAE = {overall_va:.3f} m  |  "
                 f"delta = {overall_va - overall_lse_va:+.3f} m")
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
        "corrector_train_mae_m": float(overall_tr),
        "corrector_val_mae_m": float(overall_va),
        "delta_train_m": float(overall_tr - overall_lse_tr),
        "delta_val_m": float(overall_va - overall_lse_va),
        "val_p50_m": float(np.median(err_va)),
        "val_p95_m": float(np.percentile(err_va, 95)),
        "val_max_m": float(err_va.max()),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--train_scene", choices=SCENES, default="big",
                   help="scene to train on (default 'big')")
    p.add_argument("--val_scene", choices=SCENES, default="right",
                   help="scene to validate on (default 'right')")
    p.add_argument("--T", type=int, default=32,
                   help="sliding-window length in superframes")
    p.add_argument("--stride", type=int, default=1,
                   help="window stride (1 = rotation/sliding)")
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--predict_z", action="store_true",
                   help="predict Δz too (default off: tag_z=1.0 fixed)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: code/checkpoints/"
                        "_tmp_uwb_corrector_<train>_to_<val>)")
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
                    / f"_tmp_uwb_corrector_{args.train_scene}_to_{args.val_scene}")

    print("=" * 70)
    print("UWB Transformer Correction \u2014 training on office_rack sionna track")
    print(f"  train scene: {args.train_scene}  ({train_csv.name})")
    print(f"  val scene:   {args.val_scene}  ({val_csv.name})")
    print(f"  T={args.T}  stride={args.stride}  epochs={args.n_epochs}  "
          f"lr={args.lr}  predict_z={args.predict_z}")
    print(f"  out:         {args.out}")
    print("=" * 70)
    print()

    summary = train_and_eval(
        train_csv=train_csv, val_csv=val_csv,
        train_scene=args.train_scene, val_scene=args.val_scene,
        T=args.T, stride=args.stride,
        n_epochs=args.n_epochs, lr=args.lr,
        predict_z=args.predict_z,
        device=args.device, out_dir=args.out,
    )
    print()
    print("summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
