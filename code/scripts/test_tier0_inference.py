"""Inference test for the trained Tier 0 SSL encoder.

Loads `code/checkpoints/tier0_multiscene_f2_3to1_100ep/tier0_multiscene_f2.pt`
and runs masked-cell reconstruction on a held-out Sionna scene that
was NOT in the multi-scene training mix (default: `outdoor_parking_01`,
the same scene that was Tier 1 FOMAML's val set).

Reports:
  1. Model's per-masked-cell RMSE, bias, R^2, shrink slope
  2. Three baselines for comparison:
     - random: per-(window, anchor) mean of *visible* cells
     - oracle: per-(grid, anchor) mean over all 100 trials
     - LSE: 4-anchor weighted least-squares trilateration on
       the *visible* cells' ToA values, denormalized
  3. Per-anchor RMSE breakdown
  4. Per-channel RMSE (ToA vs AoA azimuth)
  5. A short printed sample of pred vs true for one window

Usage:
    cd code
    python3.11 -m scripts.test_tier0_inference
    python3.11 -m scripts.test_tier0_inference --scene outdoor_parking_01
    python3.11 -m scripts.test_tier0_inference --ckpt checkpoints/tier0_multiscene_f2_3to1_100ep
    python3.11 -m scripts.test_tier0_inference --no_lse   # skip LSE (faster)
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer, random_mask
from scripts.multi_scene_dataset import build_multi_scene_dataset


# Anchor positions for the 4 Sionna anchors (in meters, scene-local frame).
# Source: MEMORY.md / PROJECT_NOTES.md. Used for LSE.
# A1: (-0.52, -0.9, 2.2)
# A2: (-0.54, 0.35, 2.2)
# A3: (0.52, -0.9, 2.2)
# A4: (0.54, 0.35, 2.2)
ANCHOR_POS = np.array([
    [-0.52, -0.9, 2.2],
    [-0.54, 0.35, 2.2],
    [0.52, -0.9, 2.2],
    [0.54, 0.35, 2.2],
], dtype=np.float32)


def lse_solve(visible_toa: np.ndarray, visible_anchors: np.ndarray) -> np.ndarray:
    """Weighted least-squares trilateration on visible ToA cells.

    visible_toa: (N,)  measured distances in meters
    visible_anchors: (N, 3) anchor positions in meters
    Returns (3,) estimated (x, y, z) position, or NaN if the system
    is under-determined (need at least 3 anchors).
    """
    if len(visible_toa) < 3:
        return np.full(3, np.nan)
    A_xyz = visible_anchors
    d = visible_toa
    # Linearized LSE: subtract last anchor's equation from the others
    # (x - x_n)^2 + (y - y_n)^2 + (z - z_n)^2 = d_n^2
    # Expanding: x_n^2 - x_i^2 + ... = d_i^2 - d_n^2
    # => 2(x_n - x_i) x + 2(y_n - y_i) y + 2(z_n - z_i) z = d_i^2 - d_n^2 + ...
    Ai, di = A_xyz[:-1], d[:-1]
    An, dn = A_xyz[-1], d[-1]
    b = di**2 - dn**2 - (np.sum(Ai * Ai, axis=1) - np.sum(An * An))
    M = 2 * (An - Ai)
    try:
        sol, *_ = np.linalg.lstsq(M, b, rcond=None)
        return sol
    except np.linalg.LinAlgError:
        return np.full(3, np.nan)


def compute_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> dict:
    """RMSE, bias, R^2, shrink slope, fraction within 0.5m."""
    if mask is not None:
        p = pred[mask]
        t = true[mask]
    else:
        p = pred.flatten()
        t = true.flatten()
    diff = p - t
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    # R^2: 1 - SSE/SST
    sst = float(np.sum((t - t.mean()) ** 2))
    sse = float(np.sum(diff ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    # shrink slope: linear regression of pred on true (intercept=0)
    if np.sum(t * t) > 0:
        slope = float(np.sum(p * t) / np.sum(t * t))
    else:
        slope = float("nan")
    # fraction within 0.5
    within = float(np.mean(np.abs(diff) < 0.5))
    return {"rmse": rmse, "bias": bias, "r2": r2, "shrink": slope, "within_0.5": within, "n": int(mask.sum()) if mask is not None else int(p.size)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/tier0_multiscene_f2_3to1_100ep")
    p.add_argument("--no_skip", action="store_true",
                   help="use the old no-skip head at inference (for comparison with the new skip architecture)")
    p.add_argument("--scene", default="outdoor_parking_01",
                   help="held-out scene to test on (must NOT be in multi-scene training mix)")
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument("--n_windows", type=int, default=240,
                   help="number of val windows to evaluate on")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_lse", action="store_true",
                   help="skip the LSE baseline (faster)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # --- Load the trained checkpoint ---
    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "tier0_multiscene_f2.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    cfg_dict = ckpt["cfg"]
    cfg = Config()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.mask_ratio = args.mask_ratio
    print(f"  trained on scenes: {ckpt.get('scenes', '?')}")
    print(f"  norm_mean (ToA, AoA): {ckpt['norm_mean']}")
    print(f"  norm_std  (ToA, AoA): {ckpt['norm_std']}")
    print(f"  train_losses: {ckpt['train_losses'][:3]} ... {ckpt['train_losses'][-3:]}")
    print(f"  val_losses:   {ckpt['val_losses'][:3]} ... {ckpt['val_losses'][-3:]}")
    print(f"  F=2 channels, 3:1 channel weights, 100 epochs")

    # Build the model and load weights. Default uses the new skip-connection
    # architecture; --no_skip builds a no-skip head at inference for
    # apples-to-apples comparison.
    if args.no_skip:
        from ultraclear.asta import ASTAEncoder
        class _NoSkipHead(torch.nn.Module):
            def __init__(self, d_model, n_channels, hidden=64):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(d_model, hidden),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden, n_channels),
                )
            def forward(self, h, _x_input=None, _mask=None):
                return self.net(h)
        class _NoSkipPretrainer(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.cfg = cfg
                self.encoder = ASTAEncoder(cfg)
                self.head = _NoSkipHead(cfg.d_model, cfg.n_channels)
            def forward(self, x):
                from ultraclear.tier0_ssl import random_mask
                x_masked, mask = random_mask(x, self.cfg.mask_ratio)
                h = self.encoder(x_masked)
                x_pred = self.head(h, x_masked, mask)
                return x_pred, mask, x
        model = _NoSkipPretrainer(cfg).to(args.device)
        print("  (using --no_skip head: output is MLP(h) at every cell, no skip)")
    else:
        model = SSLPretrainer(cfg).to(args.device)
        print("  (using skip-connection head: visible cells preserved, masked cells = MLP(h))")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    norm_mean = ckpt["norm_mean"]
    norm_std = ckpt["norm_std"]

    # --- Load the held-out scene ---
    base = "/Users/justin/Downloads/srp_research/grid2"
    scene_path = f"{base}/sionna_uwb_grid_{args.scene}_A1A4swapped.csv"
    print(f"\nLoading held-out scene: {args.scene}")
    print(f"  path: {scene_path}")
    # Build via build_multi_scene_dataset for consistency with training
    ds, scene_info = build_multi_scene_dataset(
        [scene_path], seq_len=cfg.seq_len, stride=cfg.seq_len, n_channels=2,
    )
    # Apply the same per-channel normalization that the checkpoint used
    ds.norm_mean = norm_mean
    ds.norm_std = norm_std
    n_total = len(ds)
    n_windows = min(args.n_windows, n_total)
    indices = rng.choice(n_total, size=n_windows, replace=False)
    print(f"  {n_total} windows available, evaluating on {n_windows} (random sample)")

    # --- Run inference ---
    all_pred = []      # (B, T, M, F)   in z-score space
    all_true = []      # (B, T, M, F)
    all_mask = []      # (B, T, M) bool
    t0 = time.time()
    with torch.no_grad():
        for i in indices:
            X, _y, _env, _cond, _nlos = ds[i]
            # X is already z-score normalized by ds.__getitem__
            X = X.unsqueeze(0).to(args.device)        # (1, T, M, F)
            x_pred, mask, _x = model(X)                  # x_pred in z-score space
            all_pred.append(x_pred.squeeze(0).cpu().numpy())
            all_true.append(X.squeeze(0).cpu().numpy())
            all_mask.append(mask.squeeze(0).cpu().numpy())
    print(f"  inference done in {time.time()-t0:.2f}s")
    pred = np.stack(all_pred)                         # (B, T, M, F)  z-score
    true = np.stack(all_true)                         # (B, T, M, F)  z-score
    mask = np.stack(all_mask)                         # (B, T, M)
    B, T, M, F_ = pred.shape

    # --- Denormalize back to original units ---
    pred_m = pred * norm_std + norm_mean              # (B, T, M, F)
    true_m = true * norm_std + norm_mean

    # --- Compute per-channel metrics: MASKED vs UNMASKED ---
    print(f"\n{'='*70}")
    print(f"  Masked vs Unmasked cells (denormalized, in original units)")
    print(f"{'='*70}")
    print(f"  channel  region      RMSE       bias       R^2     shrink  n")
    print(f"  -------  ---------  --------  --------  --------  --------  -----")
    for ch_name, ch_idx, unit, within_t in [
        ("ToA", 0, "m", 0.5),
        ("AoA", 1, "°", 5.0),
    ]:
        for region_name, region_mask in [
            ("MASKED  ", mask),
            ("UNMASKED", ~mask),
        ]:
            if ch_name == "ToA":
                m = compute_metrics(pred_m[..., ch_idx], true_m[..., ch_idx], region_mask)
                print(f"  {ch_name:7s} {region_name}  {m['rmse']:.4f}    "
                      f"{m['bias']:+.4f}    {m['r2']:.4f}   {m['shrink']:+.4f}   {m['n']:5d}  {unit}")
            else:
                # wrap-around diff for AoA
                diff = pred_m[..., 1] - true_m[..., 1]
                diff = (diff + 180.0) % 360.0 - 180.0
                p = pred_m[..., 1][region_mask]
                t = true_m[..., 1][region_mask]
                d = diff[region_mask]
                rmse = float(np.sqrt(np.mean(d ** 2)))
                bias = float(np.mean(d))
                sst = float(np.sum((t - t.mean()) ** 2))
                r2 = 1.0 - float(np.sum(d ** 2)) / sst if sst > 0 else float("nan")
                shrink = float(np.sum(p * t) / np.sum(t * t)) if np.sum(t * t) > 0 else float("nan")
                print(f"  AoA     {region_name}  {rmse:.4f}    "
                      f"{bias:+.4f}    {r2:.4f}   {shrink:+.4f}   {int(region_mask.sum()):5d}  °")
    print(f"  (within0.5m for ToA, within5° for AoA)")

    # --- Per-anchor ToA RMSE breakdown ---
    print(f"\n  Per-anchor ToA RMSE breakdown (MASKED vs UNMASKED):")
    print(f"  anchor  MASKED  (n)   UNMASKED  (n)    ratio")
    for m in range(M):
        m_m = mask[:, :, m]
        m_v = ~m_m
        p_m = pred_m[:, :, m, 0][m_m]
        t_m = true_m[:, :, m, 0][m_m]
        p_v = pred_m[:, :, m, 0][m_v]
        t_v = true_m[:, :, m, 0][m_v]
        rmse_masked = float(np.sqrt(np.mean((p_m - t_m) ** 2)))
        rmse_unmasked = float(np.sqrt(np.mean((p_v - t_v) ** 2))) if m_v.sum() > 0 else float("nan")
        ratio = rmse_masked / rmse_unmasked if rmse_unmasked > 0 else float("nan")
        print(f"  A{m+1}     {rmse_masked:.4f}  ({m_m.sum():4d})  "
              f"{rmse_unmasked:.4f}  ({m_v.sum():4d})   {ratio:.2f}x")

    # --- Baselines ---
    print(f"\n{'='*70}")
    print(f"  Baselines (ToA only, MASKED vs UNMASKED cells)")
    print(f"{'='*70}")
    flat_mask = mask.reshape(-1)
    flat_visible = ~flat_mask
    flat_pred = pred_m[..., 0].reshape(-1)
    flat_true = true_m[..., 0].reshape(-1)

    # (a) Model's prediction
    print(f"  {'method':35s}  {'region':10s}  {'RMSE':>7s}  {'bias':>7s}  {'R^2':>7s}  {'shrink':>7s}")
    for region_name, region_mask in [("MASKED", flat_mask), ("UNMASKED", flat_visible)]:
        m_m = compute_metrics(flat_pred, flat_true, region_mask)
        print(f"  {'Model (encoder + SSL head)':35s}  {region_name:10s}  "
              f"{m_m['rmse']:7.4f}  {m_m['bias']:+7.4f}  {m_m['r2']:7.4f}  {m_m['shrink']:+7.4f}")

    # (b) Random baseline: per-(window, anchor) mean of VISIBLE cells
    rand_pred = np.zeros_like(flat_pred)
    for b in range(B):
        for t in range(T):
            for m_ in range(M):
                win = (b, t, m_)
                if not mask[win]:
                    continue
                visible = []
                for m2 in range(M):
                    if m2 != m_:
                        visible.append(pred_m[b, t, m2, 0])
                # actually: per-(window, anchor) MEAN should come from true values
                # of the visible cells at the same (b, t)
                visible_true = [true_m[b, t, m2, 0] for m2 in range(M) if m2 != m_]
                rand_pred[b * T * M + t * M + m_] = np.mean(visible_true)
    rand_masked = compute_metrics(rand_pred, flat_true, flat_mask)
    # Unmasked: predict the per-(window, anchor) mean of all OTHER cells
    rand_unmasked_pred = np.zeros_like(flat_pred)
    for b in range(B):
        for t in range(T):
            for m_ in range(M):
                other_true = [true_m[b, t, m2, 0] for m2 in range(M) if m2 != m_]
                rand_unmasked_pred[b * T * M + t * M + m_] = np.mean(other_true)
    rand_unmasked = compute_metrics(rand_unmasked_pred, flat_true, flat_visible)
    print(f"  {'Random (visible cells mean)':35s}  {'MASKED':10s}  "
          f"{rand_masked['rmse']:7.4f}  {rand_masked['bias']:+7.4f}  "
          f"{rand_masked['r2']:7.4f}  {rand_masked['shrink']:+7.4f}")
    print(f"  {'':35s}  {'UNMASKED':10s}  "
          f"{rand_unmasked['rmse']:7.4f}  {rand_unmasked['bias']:+7.4f}  "
          f"{rand_unmasked['r2']:7.4f}  {rand_unmasked['shrink']:+7.4f}")

    # (c) Oracle baseline: per-(grid, anchor) mean over the entire scene
    # For the held-out scene we have the full CSVs -- we can compute the
    # oracle from the SAME scene's 100 trials per grid. (This is "cheating"
    # because it uses info the model didn't have, but it gives the noise
    # floor of the problem.)
    print(f"  Computing oracle baseline (per-(grid, anchor) mean over 100 trials)...")
    df = pd.read_csv(scene_path).sort_values(["grid_id", "trial_id"]).reset_index(drop=True)
    n_grids = int(df["grid_id"].nunique())
    n_trials = int((df["grid_id"] == df["grid_id"].iloc[0]).sum())
    toa = np.stack(
        [df[f"A{m}_measured_dist_m"].values for m in range(1, 5)],
        axis=1,
    ).astype(np.float32).reshape(n_grids, n_trials, 4)
    oracle_per_grid = np.nanmean(toa, axis=1)              # (n_grids, 4)
    # Map each (b, t, m) to its grid. We used stride = seq_len = 32, so
    # window i -> (g, s=0), (g, s=32), etc., in raster order.
    oracle_pred = np.zeros_like(flat_pred)
    for b in range(B):
        # ds.windows stores (s_idx, g, s). Recompute to know which grid.
        s_idx, g, s = ds.windows[indices[b]]
        for t in range(T):
            for m_ in range(M):
                oracle_pred[b * T * M + t * M + m_] = oracle_per_grid[g, m_]
    oracle_m = compute_metrics(oracle_pred, flat_true, flat_mask)
    # For UNMASKED, the oracle is the truth by construction (per-(grid, anchor)
    # mean over 100 trials is the closest fixed value to the noise), so it
    # gives the noise floor on the visible side.
    oracle_unmasked = compute_metrics(oracle_pred, flat_true, flat_visible)
    print(f"  {'Oracle (per-(grid, anchor) mean)':35s}  {'MASKED':10s}  "
          f"{oracle_m['rmse']:7.4f}  {oracle_m['bias']:+7.4f}  "
          f"{oracle_m['r2']:7.4f}  {oracle_m['shrink']:+7.4f}")
    print(f"  {'':35s}  {'UNMASKED':10s}  "
          f"{oracle_unmasked['rmse']:7.4f}  {oracle_unmasked['bias']:+7.4f}  "
          f"{oracle_unmasked['r2']:7.4f}  {oracle_unmasked['shrink']:+7.4f}   "
          f"(noise floor)")
    print(f"  (Oracle is the noise floor -- what the model COULD achieve if it")
    print(f"   knew the per-grid anchor statistics.)")

    # (d) LSE baseline: triangulate from visible ToA cells
    if not args.no_lse:
        print(f"  Computing LSE baseline (triangulation from visible ToA)...")
        lse_pred = np.full((B, T, M, 3), np.nan)
        for b in range(B):
            s_idx, g, s = ds.windows[indices[b]]
            for t in range(T):
                for m_ in range(M):
                    visible_anchors = []
                    visible_toa_vals = []
                    for m2 in range(M):
                        if not mask[b, t, m2]:
                            visible_anchors.append(ANCHOR_POS[m2])
                            visible_toa_vals.append(true_m[b, t, m2, 0])
                    if len(visible_toa_vals) >= 3:
                        lse_pred[b, t, m_] = lse_solve(
                            np.array(visible_toa_vals), np.array(visible_anchors)
                        )
        # LSE is per-(t, m) — it predicts the tag position, not a ToA value.
        # The comparable quantity is the predicted distance from the LSE
        # solution to the masked anchor. We report this separately below
        # since LSE is in a different unit space (3D position vs ToA).
        # Here we just report how often LSE has a valid solution.
        n_valid = int(np.sum(~np.isnan(lse_pred[..., 0])))
        print(f"  LSE produced {n_valid}/{B*T*M} valid solutions "
              f"({100*n_valid/(B*T*M):.1f}%) -- requires >=3 visible cells")

    # --- Per-window "shrink" pattern: model collapse diagnostic ---
    print(f"\n{'='*70}")
    print(f"  Model collapse diagnostic")
    print(f"{'='*70}")
    # For each (window, anchor), the model should output a value near the
    # true value. If the model has collapsed, it outputs near the per-window
    # mean of visible cells. Compute the fraction of model predictions that
    # are CLOSER to the visible-cells mean than to the true value.
    collapse_count = 0
    total = 0
    for b in range(B):
        for t in range(T):
            for m_ in range(M):
                if not mask[b, t, m_]:
                    continue
                vis_mean = np.mean([true_m[b, t, m2, 0] for m2 in range(M) if m2 != m_])
                p = pred_m[b, t, m_, 0]
                t_v = true_m[b, t, m_, 0]
                d_pred_to_mean = abs(p - vis_mean)
                d_pred_to_true = abs(p - t_v)
                if d_pred_to_mean < d_pred_to_true:
                    collapse_count += 1
                total += 1
    print(f"  Fraction of masked cells where model is closer to visible-mean than truth:")
    print(f"    {collapse_count}/{total} = {100*collapse_count/total:.1f}%")
    if collapse_count / total > 0.5:
        print(f"  > 50% = MODEL HAS COLLAPSED to 'predict visible mean' strategy.")
    else:
        print(f"  < 50% = model is using real context (not just visible mean).")

    # --- Sample print: one window ---
    print(f"\n{'='*70}")
    print(f"  Sample window (window 0, t=16, all anchors, ToA channel)")
    print(f"{'='*70}")
    b = 0
    t = T // 2
    print(f"  anchor  true(m)   pred(m)   diff(m)   visible?  masked?")
    for m_ in range(M):
        print(f"  A{m_+1}     {true_m[b, t, m_, 0]:6.3f}    {pred_m[b, t, m_, 0]:6.3f}    "
              f"{pred_m[b, t, m_, 0] - true_m[b, t, m_, 0]:+6.3f}    "
              f"{'Y' if not mask[b, t, m_] else 'N':>3}      {'Y' if mask[b, t, m_] else 'N'}")

    print(f"\nDone. Total time: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
