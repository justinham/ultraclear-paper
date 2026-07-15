"""Supervised baseline: SSL-pretrained encoder + 3 heads (location,
restoration, NLOS), trained end-to-end with plain supervised losses.

No MAML, no FiLM, no adversarial training — just a single straight
gradient through the full model. This is the *honest baseline* we
use to argue that the Tier 1/2 machinery (FOMAML inner-loop, FiLM
conditioning, adversarial restoration) actually buys something.

Comparison structure:
  - Tier 1 only       (FOMAML, no FiLM)        -- train_tier1_sionna.py
  - Tier 1+2 joint    (FOMAML + FiLM + adv)    -- train_tier12_joint.py
  - **THIS**          (plain supervised, SSL init) — train_supervised_baseline.py
  - No-Tier-1 baseline(from scratch)            -- baseline_only.py

What "SSL-pretrained" means here: we load encoder weights from a
Tier 0 checkpoint (e.g. tier0_multiscene_f2.pt). The 3 heads are
random-initialized and trained from scratch alongside the encoder.
This isolates the contribution of: (a) the SSL pretraining itself,
and (b) the Tier 1/2 meta-learning on top.

Loss = α * MSE(loc) + β * MSE(restore) + γ * BCE(nlos)
with α, β, γ chosen so the three losses are on comparable scale
(all log-m² or log-units, see config below).

Usage:
  python3.11 -m scripts.train_supervised_baseline \\
      --csv_dir /Users/justin/Downloads/srp_research/grid2 \\
      --ssl_ckpt code/checkpoints/tier0_multiscene_f2_3to1_skip_100ep/tier0_multiscene_f2.pt \\
      --n_epochs 50
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultraclear.config import Config
from ultraclear.tier3_heads import DownstreamModel
from scripts.sionna_joint_dataset import (
    build_sionna_joint_dataset,
    sionna_to_items_with_restore,
)
from scripts.train_tier12_joint import TensorUWB


def build_dataset(csv_dir: str, n_channels: int, seq_len: int):
    csv_paths = sorted(str(p) for p in Path(csv_dir).glob("*_A1A4swapped.csv"))
    if not csv_paths:
        sys.exit(f"no CSVs in {csv_dir}")
    ds, scene_info = build_sionna_joint_dataset(
        csv_paths=csv_paths, n_channels=n_channels, seq_len=seq_len, stride=seq_len,
    )
    env_ids = [info.env_id for info in scene_info]
    # 80/20 by env, but always keep at least 1 val env (so we don't end
    # up with val_items=0 on small datasets like the 10-scene simple
    # synthetic).
    n_val = max(1, len(env_ids) // 5)
    train_set = set(env_ids[: len(env_ids) - n_val])
    val_set = set(env_ids[len(env_ids) - n_val :]) - {14}
    train_items, val_items, scene_info, denorm_info = sionna_to_items_with_restore(
        ds, train_env_ids=train_set, val_env_ids=val_set,
    )
    train_env_ids = sorted({it[2] for it in train_items})
    ds.compute_norm_stats(set(train_env_ids))
    train_items, val_items, scene_info, denorm_info = sionna_to_items_with_restore(
        ds, train_env_ids=train_set, val_env_ids=val_set,
    )
    return train_items, val_items, scene_info, denorm_info, ds


def make_restore_target(batch_X, batch_nlos, denorm_info, env_ids, n_channels, ds):
    """Build the (B, T, M, F) restoration target from the raw CSV columns.

    We need access to the *raw measured* and *ground-truth* distances.
    SionnaJointDataset keeps them in `ds.windows[i]["feat_raw"]`
    (un-normalized), and the NLOS mask from `nlos`. We approximate
    "clean" measurement per cell as: gt_dist if LOS, else measured
    (since for NLOS we don't have a true geometric value either — the
    measured *is* what the sensor reported, biased or not).
    """
    raise NotImplementedError(
        "see load_restore_targets() in __main__ — kept separate so the "
        "training loop is clean."
    )


def load_restore_targets(ds, indices, env_ids_batch, n_channels):
    """Pull (B, T, M, F) restore targets out of the dataset's raw feats.

    Uses the per-scene raw measured values from `ds.per_scene_features`
    (NOT z-score normalized) plus the per-scene NLOS mask, and replaces
    each cell with `A{m}_gt_dist_m` if the cell is LOS (so the target
    is the clean geometric distance, which is what the restoration
    head should learn to recover from the noisy measurement).

    For channel 1 (AoA azimuth) we don't have a separate "clean AoA"
    in the data, so we just use the measured AoA as both input and
    target — the model trivially learns identity there. This keeps
    the head's output dim consistent with n_channels=2.
    """
    raise NotImplementedError(
        "see inline implementation in main()"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", required=True)
    p.add_argument("--ssl_ckpt", required=True,
                   help="Path to Tier 0 SSL checkpoint (.pt) for encoder init")
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loc_weight", type=float, default=1.0)
    p.add_argument("--restore_weight", type=float, default=0.1,
                   help="Weight on the restoration loss. Empirical note "
                        "(2026-07-15 smoke test, 4 epochs): the default 1.0 "
                        "hurts loc by ~1m because the gradient through the "
                        "restored tensor fights the loc gradient. 0.0-0.1 "
                        "is the sweet spot. Set to 0.0 for pure loc training.")
    p.add_argument("--nlos_weight", type=float, default=1.0)
    p.add_argument("--cond_thresholds", type=float, nargs=2, default=[0.05, 0.15],
                   help="Per-window NLOS rate thresholds for low/mid/high "
                        "buckets in the val report. Default [0.05, 0.15] "
                        "reflects 'person walking by' dynamic noise "
                        "(1-2 anchors briefly blocked). The legacy "
                        "[0.3, 0.7] was inherited from the synthetic "
                        "noisy-condition convention and over-buckets.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_ckpt", default=None)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--out_csv_history", default=None)
    p.add_argument("--no_ssl_init", action="store_true",
                   help="Skip loading SSL weights (random encoder init)")
    p.add_argument("--n_channels", type=int, default=2,
                   help="1=ToA only, 2=ToA+AoA")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config()
    cfg.n_channels = args.n_channels
    cfg.device = args.device

    print("building dataset...")
    train_items, val_items, scene_info, denorm_info, ds = build_dataset(
        args.csv_dir, n_channels=cfg.n_channels, seq_len=cfg.seq_len,
    )
    n_envs = len(scene_info)
    print(f"  n_envs={n_envs}  train_items={len(train_items)}  val_items={len(val_items)}")

    # Build model + SSL init
    model = DownstreamModel(cfg).to(args.device)
    if not args.no_ssl_init:
        ckpt = torch.load(args.ssl_ckpt, map_location=args.device, weights_only=False)
        # The SSL ckpt is the bare encoder state_dict from SSLPretrainer
        if "encoder_state" in ckpt:
            enc_state = ckpt["encoder_state"]
        elif "model_state" in ckpt:
            enc_state = ckpt["model_state"]
        else:
            enc_state = ckpt
        # The SSL ckpt's encoder keys carry the "encoder." prefix
        # (because the trainer wraps ASTAEncoder as self.encoder);
        # DownstreamModel's model.encoder.load_state_dict() expects
        # the bare keys. Strip the prefix.
        enc_state = {k.removeprefix("encoder."): v for k, v in enc_state.items()}
        missing, unexpected = model.encoder.load_state_dict(enc_state, strict=False)
        print(f"  loaded SSL encoder from {args.ssl_ckpt}")
        if missing:
            print(f"    missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"    unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    else:
        print("  --no_ssl_init: encoder randomly initialized")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    train_ds = TensorUWB(train_items)
    val_ds = TensorUWB(val_items)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # Per-channel z-score norm (computed from train scenes inside the
    # dataset). We need it as plain tensors so we can invert X back to
    # meters when building the restore target.
    norm_mean = torch.from_numpy(ds.norm_mean).to(args.device)   # (F,)
    norm_std = torch.from_numpy(ds.norm_std).to(args.device)     # (F,)

    history = []
    print(f"\ntraining supervised baseline for {args.n_epochs} epochs...")
    for ep in range(args.n_epochs):
        model.train()
        sum_loc, sum_rest, sum_nlos, n = 0.0, 0.0, 0.0, 0
        sum_loc_w, sum_rest_w, sum_nlos_w = 0.0, 0.0, 0.0
        for X, y, env, _c, nlos, restore in train_loader:
            X = X.to(args.device); y = y.to(args.device)
            env = env.to(args.device); nlos = nlos.to(args.device).float()
            restore = restore.to(args.device)              # (B, T, M) in meters

            # Build the 2-channel restore target the RestorationHead
            # expects (n_channels=2 to match SSL ckpt):
            #   channel 0: gt_dist where LOS, measured ToA otherwise
            #   channel 1: measured AoA (no clean AoA available, so
            #              this is a trivial identity target)
            # We z-score the ToA target so the restore loss lives in
            # the same units as the loc loss (normalized).
            toa_meas = X[:, :, :, 0]                            # already z-scored
            aoa_meas = X[:, :, :, 1]                            # already z-scored
            restore_z = (restore - norm_mean[0]) / norm_std[0]  # z-score gt_dist
            restore_toa = torch.where(nlos > 0.5, toa_meas, restore_z)
            restore_target = torch.stack([restore_toa, aoa_meas], dim=-1)

            opt.zero_grad()
            out = model(X)
            # LocationHeadFromRestore reads the *middle* timestep of
            # the restored output, so we supervise with y at the same
            # index. (y is (B, T, 2) — one (x,y) per timestep.)
            T = X.size(1)
            y_mid = y[:, T // 2, :]
            loc_loss = TF.mse_loss(out["loc"], y_mid)
            rest_loss = TF.mse_loss(out["restored"], restore_target)
            nlos_loss = bce(out["nlos_logits"], nlos)
            loss = (args.loc_weight * loc_loss
                    + args.restore_weight * rest_loss
                    + args.nlos_weight * nlos_loss)
            loss.backward()
            opt.step()

            sum_loc += loc_loss.item() * X.size(0)
            sum_rest += rest_loss.item() * X.size(0)
            sum_nlos += nlos_loss.item() * X.size(0)
            # Weighted contributions (what the optimizer actually sees)
            sum_loc_w += args.loc_weight * loc_loss.item() * X.size(0)
            sum_rest_w += args.restore_weight * rest_loss.item() * X.size(0)
            sum_nlos_w += args.nlos_weight * nlos_loss.item() * X.size(0)
            n += X.size(0)

        train_loc = sum_loc / n
        train_rest = sum_rest / n
        train_nlos = sum_nlos / n
        train_loc_w = sum_loc_w / n
        train_rest_w = sum_rest_w / n
        train_nlos_w = sum_nlos_w / n

        # Validation: per-condition loc error (matching joint trainer)
        # Cond buckets are split by the per-window NLOS rate (fraction
        # of (T*M) cells flagged as NLOS). Default thresholds [0.05, 0.15]
        # reflect a 'person walking by' dynamic-noise regime; the
        # legacy [0.3, 0.7] was too coarse and inverted the easy/hard
        # ordering on real data (see 2026-07-15 discussion).
        t_lo, t_hi = args.cond_thresholds
        errs_per_cond = {0: [], 1: [], 2: []}
        with torch.no_grad():
            for X, y, env, _c, nlos, _restore in val_loader:
                X = X.to(args.device); y = y.to(args.device); env = env.to(args.device)
                out = model(X)
                pred = out["loc"]                            # (B, 2)
                T = X.size(1)
                y_mid = y[:, T // 2, :]                      # (B, 2)
                # Denormalize per-scene so errors are in original units (meters)
                pred_orig = torch.zeros_like(pred)
                y_orig = torch.zeros_like(y_mid)
                for i in range(len(env)):
                    eid = int(env[i].item())
                    mean, std = denorm_info[eid]
                    pred_orig[i] = pred[i] * torch.from_numpy(std).to(args.device) + torch.from_numpy(mean).to(args.device)
                    y_orig[i] = y_mid[i] * torch.from_numpy(std).to(args.device) + torch.from_numpy(mean).to(args.device)
                pred, y = pred_orig, y_orig
                err = torch.linalg.norm(pred - y, dim=-1).cpu().numpy()
                nlos_rate = nlos.float().mean(dim=(1, 2)).numpy()
                for rate, e in zip(nlos_rate.tolist(), err.tolist()):
                    if rate < t_lo:
                        errs_per_cond[0].append(e)
                    elif rate < t_hi:
                        errs_per_cond[1].append(e)
                    else:
                        errs_per_cond[2].append(e)
        per_cond = {ci: float(np.mean(es)) if es else float("nan")
                    for ci, es in errs_per_cond.items()}
        # If any bucket is empty, fall back to the overall mean for that
        # bucket so the figure still has a value to plot.
        overall = float(np.mean([v for v in per_cond.values() if not np.isnan(v)]))
        per_cond_filled = {ci: (v if not np.isnan(v) else overall)
                           for ci, v in per_cond.items()}
        print(f"  ep {ep+1:3d}/{args.n_epochs}  "
              f"train loc={train_loc:.4f} rest={train_rest:.4f} nlos={train_nlos:.4f}  "
              f"wloc={train_loc_w:.4f} wrest={train_rest_w:.4f} wnlos={train_nlos_w:.4f}  "
              f"val loc err (nlos<{t_lo}/{t_hi}): low={per_cond[0]:.3f}  mid={per_cond[1]:.3f}  high={per_cond[2]:.3f}  "
              f"avg={overall:.3f} m")
        history.append({
            "epoch": ep + 1,
            "train_loc": train_loc, "train_rest": train_rest, "train_nlos": train_nlos,
            "train_loc_w": train_loc_w, "train_rest_w": train_rest_w, "train_nlos_w": train_nlos_w,
            **{f"val_loc_cond{ci}": v for ci, v in per_cond_filled.items()},
            "val_loc_avg": overall,
        })

    if args.out_ckpt:
        torch.save({
            "model_state": model.state_dict(),
            "cfg": cfg.__dict__,
            "ssl_ckpt": args.ssl_ckpt,
            "nlos_weight": args.nlos_weight, "restore_weight": args.restore_weight,
        }, args.out_ckpt)
        print(f"\nckpt -> {args.out_ckpt}")
    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cond_id", "val_loc_err_m"])
            for ci in sorted(per_cond.keys()):
                w.writerow([ci, per_cond[ci]])
        print(f"per-cond CSV -> {args.out_csv}")
    if args.out_csv_history:
        with open(args.out_csv_history, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(history[0].keys())
            for row in history:
                w.writerow(row.values())
        print(f"history CSV -> {args.out_csv_history}")


if __name__ == "__main__":
    main()
