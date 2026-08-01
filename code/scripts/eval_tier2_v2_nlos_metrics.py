"""Per-env, per-cond, per-anchor classification metrics for the v2
unfrozen-encoder Tier 2 ckpt.

Accuracy is misleading on envs with 0% NLOS (the degenerate
"always predict LOS" classifier gets 100% accuracy on them, so
they appear to be the best-performing envs when they're actually
uninformative). This script computes classification metrics that
handle class imbalance properly:

  - AUROC: area under the ROC curve. Uses predicted P(NLOS)
    (sigmoid of nlos_logits), so the threshold choice doesn't
    matter. Undefined when only one class is present.
  - MCC: Matthews correlation coefficient. Single-number summary
    that handles class imbalance. Undefined when one class is
    entirely absent.
  - Balanced accuracy: (TPR + TNR) / 2. Defined even when one
    class is rare, but undefined when one class is absent.
  - Brier score: mean squared error of P(NLOS) vs binary target.
    Lower is better; measures calibration.
  - ECE (expected calibration error): weighted average of
    |bin_accuracy - bin_confidence| across probability bins.
  - n_NLOS_cells: count of actual NLOS cells in the env (so the
    reader knows whether the metrics are computed on 0 cells
    vs 50,000 cells).

For each env, also reports per-anchor AUROC, so a per-anchor
breakdown is available for the figure.

Outputs:
  - checkpoints/.../nlos_metrics_per_env.json
  - stdout: per-env table

Usage:
    python3.11 -m scripts.eval_tier2_v2_nlos_metrics \
      --ckpt checkpoints/tier2_sionna_v2_500/tier2_sionna_best.pt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, brier_score_loss,
    balanced_accuracy_score,
)

from ultraclear.config import Config, make_default
from ultraclear.data import TensorUWB
from scripts.multi_scene_dataset import build_default_multi_scene
from scripts.train_tier2_synthetic import DisentangledTrainer


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                 n_bins: int = 10) -> float:
    """ECE: weighted mean |bin_accuracy - bin_confidence|."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (probs > lo) & (probs <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = labels[in_bin].mean()
        bin_conf = probs[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def per_env_metrics(probs_flat: np.ndarray,
                    labels_flat: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Compute all classification metrics from per-cell probs+labels.

    Returns a dict with all metrics, plus 'n_cells', 'n_nlos_cells',
    'p_actual_nlos'. Some metrics may be None when undefined
    (e.g. AUROC on an all-LOS env).
    """
    n = len(probs_flat)
    n_nlos = int(labels_flat.sum())
    p_actual = n_nlos / n if n > 0 else 0.0
    out = {
        "n_cells": n,
        "n_nlos_cells": n_nlos,
        "p_actual_nlos": p_actual,
    }
    if n_nlos == 0 or n_nlos == n:
        # Only one class present — most metrics are undefined
        out["auroc"] = None
        out["mcc"] = None
        out["balanced_acc"] = None
        out["brier"] = None
        out["ece"] = None
        # Hard-prediction accuracy is still well-defined (always
        # predict majority class)
        preds = (probs_flat > threshold).astype(int)
        out["acc_at_0.5"] = float((preds == labels_flat).mean())
        out["tpr_at_0.5"] = None
        out["tnr_at_0.5"] = float((preds[labels_flat == 0] == 0).mean())
        out["fpr_at_0.5"] = 1.0 - out["tnr_at_0.5"]
        return out
    # Both classes present — all metrics defined
    preds = (probs_flat > threshold).astype(int)
    out["auroc"] = float(roc_auc_score(labels_flat, probs_flat))
    out["mcc"] = float(matthews_corrcoef(labels_flat, preds))
    out["balanced_acc"] = float(balanced_accuracy_score(labels_flat, preds))
    out["brier"] = float(brier_score_loss(labels_flat, probs_flat))
    out["ece"] = expected_calibration_error(probs_flat, labels_flat)
    out["acc_at_0.5"] = float((preds == labels_flat).mean())
    out["tpr_at_0.5"] = float((preds[labels_flat == 1] == 1).mean())
    out["tnr_at_0.5"] = float((preds[labels_flat == 0] == 0).mean())
    out["fpr_at_0.5"] = 1.0 - out["tnr_at_0.5"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ck["cfg"]
    cfg = make_default()
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
    print(f"  train_scenes: {sorted(train_scenes)}")
    print(f"  val_scenes: {sorted(val_scenes)}")

    print(f"\nloading Sionna multi-scene data...")
    ds, scene_info = build_default_multi_scene(
        n_channels=cfg.n_channels, seq_len=cfg.seq_len, stride=32,
        all_scenes=True,
    )
    ds.compute_norm_stats(train_scenes)

    model = DisentangledTrainer(cfg, n_envs=n_envs, n_conds=2).to(args.device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    # Index items by env_id
    print(f"\nindexing items by env_id...")
    items_by_env: dict[int, list] = {e: [] for e in range(21)}
    for i in range(len(ds)):
        item = ds[i]
        eid = int(item[2])
        items_by_env[eid].append(item)

    # Per-env per-anchor metric collection
    print(f"\nrunning inference and computing per-cell metrics...")
    per_env: dict[int, dict] = {}
    per_anchor_per_env: dict[int, list[dict | None]] = {}
    for eid in range(21):
        items = items_by_env[eid]
        if not items:
            continue
        loader = DataLoader(TensorUWB(items), batch_size=64)
        all_probs = []  # (N, T, M) predicted P(NLOS)
        all_labels = []  # (N, T, M) ground truth NLOS mask
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 6:
                    X, y, env, cond, nlos, _ = batch
                else:
                    X, y, env, cond, nlos = batch
                X = X.to(args.device)
                nlos = nlos.to(args.device)
                _h, _c, _adv, _loc, nlos_logits, _cond = model(X)
                probs = torch.sigmoid(nlos_logits).cpu().numpy()  # (B, T, M)
                all_probs.append(probs)
                all_labels.append(nlos.cpu().numpy().astype(np.int32))
        probs = np.concatenate(all_probs, axis=0)  # (N, T, M)
        labels = np.concatenate(all_labels, axis=0)  # (N, T, M)
        # Overall per-env metrics (cells flattened)
        probs_flat = probs.ravel()
        labels_flat = labels.ravel()
        m = per_env_metrics(probs_flat, labels_flat)
        m["role"] = ("train" if eid in train_scenes
                      else "val" if eid in val_scenes
                      else "excl")
        # Per-anchor metrics
        per_anchor = []
        for a in range(cfg.n_anchors):
            ma = per_env_metrics(probs[:, :, a].ravel(),
                                  labels[:, :, a].ravel())
            per_anchor.append(ma)
        per_env[eid] = m
        per_anchor_per_env[eid] = per_anchor

    # Print table
    print()
    print(f"{'env':>3} {'role':>5} {'n_cells':>8} {'n_NLOS':>7} "
          f"{'p_NLOS':>7} {'AUROC':>7} {'MCC':>7} {'bal_acc':>8} "
          f"{'Brier':>7} {'ECE':>6} {'acc@0.5':>8}")
    for eid in range(21):
        m = per_env.get(eid)
        if m is None:
            continue
        auroc = f"{m['auroc']:.3f}" if m['auroc'] is not None else "  n/a "
        mcc = f"{m['mcc']:+.3f}" if m['mcc'] is not None else "  n/a "
        bacc = f"{m['balanced_acc']:.3f}" if m['balanced_acc'] is not None else "  n/a "
        brier = f"{m['brier']:.3f}" if m['brier'] is not None else "  n/a "
        ece = f"{m['ece']:.3f}" if m['ece'] is not None else "  n/a "
        print(f"{eid:>3} {m['role']:>5} {m['n_cells']:>8d} {m['n_nlos_cells']:>7d} "
              f"{m['p_actual_nlos']:>7.3f} {auroc:>7} {mcc:>7} {bacc:>8} "
              f"{brier:>7} {ece:>6} {m['acc_at_0.5']:>8.3f}")

    # Summary
    print()
    print(f"--- summary (per role, averaged over envs that have NLOS cells) ---")
    print(f"{'role':>5} {'n_envs':>7} {'mean_AUROC':>11} {'mean_MCC':>10} "
          f"{'mean_bal_acc':>13} {'mean_Brier':>11} {'mean_ECE':>10}")
    for role in ("train", "val", "excl"):
        envs_with_nlos = [e for e, m in per_env.items()
                           if m["role"] == role and m["auroc"] is not None]
        if not envs_with_nlos:
            continue
        mean_auroc = np.mean([per_env[e]["auroc"] for e in envs_with_nlos])
        mean_mcc = np.mean([per_env[e]["mcc"] for e in envs_with_nlos])
        mean_bacc = np.mean([per_env[e]["balanced_acc"] for e in envs_with_nlos])
        mean_brier = np.mean([per_env[e]["brier"] for e in envs_with_nlos])
        mean_ece = np.mean([per_env[e]["ece"] for e in envs_with_nlos])
        print(f"{role:>5} {len(envs_with_nlos):>7d} {mean_auroc:>11.3f} "
              f"{mean_mcc:>+10.3f} {mean_bacc:>13.3f} "
              f"{mean_brier:>11.3f} {mean_ece:>10.3f}")

    # Save json
    out_json = args.out_json or str(Path(args.ckpt).parent / "nlos_metrics_per_env.json")
    with open(out_json, "w") as f:
        json.dump({
            "ckpt": str(args.ckpt),
            "train_scenes": sorted(train_scenes),
            "val_scenes": sorted(val_scenes),
            "per_env": {str(e): m for e, m in per_env.items()},
            "per_anchor_per_env": {
                str(e): [m for m in per_anchor_per_env[e]]
                for e in range(21)
            },
        }, f, indent=2)
    print(f"\nMetrics saved to {out_json}")


if __name__ == "__main__":
    main()
