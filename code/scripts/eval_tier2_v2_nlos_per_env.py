"""Per-env, per-cond NLOS classification eval for the v2 unfrozen-encoder
Tier 2 ckpt.

For each env, build a per-env DataLoader over both train and val
subsets of that env, run the model, threshold `nlos_logits` at 0, and
compare to the per-(T, M) ground-truth nlos mask. Report:
  - per-env NLOS accuracy on train windows of that env
  - per-env NLOS accuracy on val windows of that env
  - per-env NLOS accuracy on any-NLOS cells vs all-LOS cells
  - overall summary table

Usage:
    python3.11 -m scripts.eval_tier2_v2_nlos_per_env \
      --ckpt checkpoints/tier2_sionna_v2_500/tier2_sionna_best.pt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ultraclear.config import Config, make_default
from ultraclear.data import TensorUWB
from scripts.multi_scene_dataset import build_default_multi_scene
from scripts.train_tier2_synthetic import DisentangledTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ck["cfg"]
    cfg = make_default()
    # Re-apply arch params from the saved cfg (same pattern as the
    # trainers — see train_tier2_sionna.py main())
    arch_keys = {"n_channels", "d_model", "seq_len", "n_anchors",
                 "n_synthetic_envs", "n_micro_conditions",
                 "samples_per_env", "ssl_hidden", "adv_hidden",
                 "micro_dim", "k_shot"}
    for k, v in cfg_dict.items():
        if k in arch_keys and hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    cfg.device = args.device
    n_envs = ck["n_envs"]
    train_scenes = set(ck["train_scenes"])
    val_scenes = set(ck["val_scenes"])
    print(f"Loaded ckpt: {args.ckpt}")
    print(f"  best_epoch (per ckpt): {ck.get('best_epoch', 'N/A')}")
    print(f"  best_val_rmse_m (per ckpt): {ck.get('best_val_rmse_m', 'N/A')}")
    print(f"  n_envs (adversary class set): {n_envs}")
    print(f"  train_scenes: {sorted(train_scenes)}")
    print(f"  val_scenes: {sorted(val_scenes)}")
    print(f"  freeze_encoder: {ck.get('freeze_encoder', 'N/A')}")

    # Load data with the same builder as the trainer
    print(f"\nloading Sionna multi-scene data...")
    ds, scene_info = build_default_multi_scene(
        n_channels=cfg.n_channels, seq_len=cfg.seq_len, stride=32,
        all_scenes=True,
    )
    ds.compute_norm_stats(train_scenes)
    # Don't overwrite ds.norm_mean/std — the trainer already saved them
    # via the arch-params path. The eval just uses them.

    # Build the model and load weights
    model = DisentangledTrainer(cfg, n_envs=n_envs, n_conds=2).to(args.device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    # Group items by env_id
    print(f"\nindexing items by env_id (n={len(ds)})...")
    items_by_env: dict[int, list] = {e: [] for e in range(21)}
    for i in range(len(ds)):
        item = ds[i]
        eid = int(item[2])  # 3rd field is env_id
        items_by_env[eid].append(item)
    for eid in range(21):
        print(f"  env {eid:2d}: {len(items_by_env[eid]):5d} items "
              f"({'TRAIN' if eid in train_scenes else 'val' if eid in val_scenes else '?'})")

    # Per-env eval
    print(f"\n--- per-env NLOS classification (threshold = 0 on nlos_logits) ---")
    print(f"{'env':>3} {'role':>6} {'role_':>5} {'n_items':>7} {'n_NLOS':>7} "
          f"{'acc_all':>8} {'acc_LOS':>8} {'acc_NLOS':>9} {'p_actual':>9} {'p_pred':>8}")
    results = {}
    for eid in range(21):
        items = items_by_env[eid]
        if len(items) == 0:
            continue
        if eid in train_scenes:
            role, role_ = "TRN", "train"
        elif eid in val_scenes:
            role, role_ = "VAL", "val"
        else:
            role, role_ = "?", "?"
        loader = DataLoader(TensorUWB(items), batch_size=64)
        n_total = 0
        n_correct = 0
        n_correct_los = 0
        n_total_los = 0
        n_correct_nlos = 0
        n_total_nlos = 0
        sum_pred_prob = 0.0  # mean predicted P(NLOS) across all cells
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 6:
                    X, y, env, cond, nlos, _ = batch
                else:
                    X, y, env, cond, nlos = batch
                X = X.to(args.device)
                # nlos is (B, T, M) bool ground truth
                nlos = nlos.to(args.device)
                # forward — model returns h_prime, c, adv_logits, loc, nlos_logits, cond_logits
                _h, _c, _adv, _loc, nlos_logits, _cond = model(X)
                # nlos_logits: (B, T, M) raw logits
                pred = (nlos_logits > 0).long()  # 1 if predicted NLOS
                gt = nlos.long()
                n_total += pred.numel()
                n_correct += (pred == gt).sum().item()
                # Per-class
                los_mask = (gt == 0)
                nlos_mask = (gt == 1)
                n_total_los += los_mask.sum().item()
                n_total_nlos += nlos_mask.sum().item()
                n_correct_los += (pred[los_mask] == 0).sum().item()
                n_correct_nlos += (pred[nlos_mask] == 1).sum().item()
                # Mean predicted P(NLOS) = sigmoid(nlos_logits)
                sum_pred_prob += torch.sigmoid(nlos_logits).sum().item()
        acc_all = n_correct / n_total
        acc_los = n_correct_los / n_total_los if n_total_los > 0 else float("nan")
        acc_nlos = n_correct_nlos / n_total_nlos if n_total_nlos > 0 else float("nan")
        p_actual = n_total_nlos / n_total
        p_pred = sum_pred_prob / n_total
        results[eid] = {
            "role": role_,
            "n_items": len(items),
            "n_cells": n_total,
            "n_nlos_cells": n_total_nlos,
            "acc_all": acc_all,
            "acc_los": acc_los,
            "acc_nlos": acc_nlos,
            "p_actual_nlos": p_actual,
            "p_pred_nlos": p_pred,
        }
        print(f"{eid:>3} {role:>6} {role_:>5} {len(items):>7d} {n_total_nlos:>7d} "
              f"{acc_all:>8.3f} {acc_los:>8.3f} {acc_nlos:>9.3f} "
              f"{p_actual:>9.3f} {p_pred:>8.3f}")

    # Summary
    print(f"\n--- summary (per role, averaged over envs) ---")
    print(f"{'role':>6} {'mean_acc_all':>13} {'mean_acc_los':>13} "
          f"{'mean_acc_nlos':>14} {'n_envs':>7}")
    for role in ("train", "val"):
        envs = [e for e, r in results.items() if r["role"] == role]
        if not envs:
            continue
        mean_acc_all = np.mean([results[e]["acc_all"] for e in envs])
        mean_acc_los = np.mean([results[e]["acc_los"] for e in envs])
        mean_acc_nlos = np.mean([results[e]["acc_nlos"] for e in envs])
        print(f"{role:>6} {mean_acc_all:>13.3f} {mean_acc_los:>13.3f} "
              f"{mean_acc_nlos:>14.3f} {len(envs):>7d}")

    # Save json
    out_json = args.out_json or str(Path(args.ckpt).parent / "nlos_per_env.json")
    with open(out_json, "w") as f:
        json.dump({
            "ckpt": str(args.ckpt),
            "best_epoch": ck.get("best_epoch"),
            "best_val_rmse_m": ck.get("best_val_rmse_m"),
            "per_env": {str(e): r for e, r in results.items()},
        }, f, indent=2)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
