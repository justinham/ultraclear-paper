"""Per-env confusion-matrix metrics for the v2 unfrozen-encoder
Tier 2 ckpt.

For each env, computes the 4 entries of the 2x2 confusion matrix at
threshold 0.5 (nlos_logits > 0) and the derived per-class metrics
that reviewers expect:

  - TP, FN, FP, TN (raw counts)
  - Precision, Recall (TPR), Specificity (TNR), F1
  - False Negative Rate (FNR), False Positive Rate (FPR)
  - Per-class support (n_P = TP + FN = actual NLOS cells,
                       n_N = FP + TN = actual LOS cells)

For the 9 envs with 0% NLOS, only TN/FP are defined (TP=FN=0);
we report them separately and mark the precision/recall/F1 as N/A.

Outputs:
  - checkpoints/.../nlos_confusion_per_env.json
  - stdout: per-env table
  - figures/.../tier2_v2_nlos_confusion.png (4-panel figure)

Usage:
    python3.11 -m scripts.eval_tier2_v2_nlos_confusion \
      --ckpt checkpoints/tier2_sionna_v2_500/tier2_sionna_best.pt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ultraclear.config import Config, make_default
from ultraclear.data import TensorUWB
from scripts.multi_scene_dataset import build_default_multi_scene
from scripts.train_tier2_synthetic import DisentangledTrainer


def confusion_metrics(probs_flat: np.ndarray,
                       labels_flat: np.ndarray,
                       threshold: float = 0.5) -> dict:
    """Compute 2x2 confusion matrix entries and derived metrics.

    Positive class = NLOS (label=1), negative class = LOS (label=0).
    Returns a dict with all entries; some will be None when
    the corresponding class is absent.
    """
    n = len(probs_flat)
    n_P = int(labels_flat.sum())  # actual NLOS count
    n_N = n - n_P                 # actual LOS count
    preds = (probs_flat > threshold).astype(int)
    tp = int(((preds == 1) & (labels_flat == 1)).sum())
    fn = int(((preds == 0) & (labels_flat == 1)).sum())
    fp = int(((preds == 1) & (labels_flat == 0)).sum())
    tn = int(((preds == 0) & (labels_flat == 0)).sum())
    out = {
        "n_cells": n,
        "n_P_actual": n_P,
        "n_N_actual": n_N,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }
    # Precision / Recall / F1 / Specificity / FNR / FPR
    if n_P == 0:
        # No NLOS cells in this env — only specificity is defined
        out["precision"] = None
        out["recall"] = None
        out["specificity"] = tn / n_N if n_N > 0 else None
        out["f1"] = None
        out["fnr"] = None
        out["fpr"] = fp / n_N if n_N > 0 else None
    elif n_N == 0:
        # No LOS cells (shouldn't happen in our envs, but be safe)
        out["precision"] = tp / (tp + fp) if (tp + fp) > 0 else None
        out["recall"] = tp / n_P
        out["specificity"] = None
        out["f1"] = None
        out["fnr"] = fn / n_P
        out["fpr"] = None
    else:
        # Both classes present — all metrics defined
        prec_denom = tp + fp
        out["precision"] = tp / prec_denom if prec_denom > 0 else 0.0
        out["recall"] = tp / n_P
        out["specificity"] = tn / n_N
        out["fpr"] = fp / n_N
        out["fnr"] = fn / n_P
        p, r = out["precision"], out["recall"]
        out["f1"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_json", default=None)
    p.add_argument("--out_fig", default=None)
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

    # Per-env confusion matrix computation
    print(f"\nrunning inference and computing confusion matrices...")
    per_env: dict[int, dict] = {}
    for eid in range(21):
        items = items_by_env[eid]
        if not items:
            continue
        loader = DataLoader(TensorUWB(items), batch_size=64)
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 6:
                    X, y, env, cond, nlos, _ = batch
                else:
                    X, y, env, cond, nlos = batch
                X = X.to(args.device)
                nlos = nlos.to(args.device)
                _h, _c, _adv, _loc, nlos_logits, _cond = model(X)
                all_probs.append(torch.sigmoid(nlos_logits).cpu().numpy())
                all_labels.append(nlos.cpu().numpy().astype(np.int32))
        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        m = confusion_metrics(probs.ravel(), labels.ravel())
        m["role"] = ("train" if eid in train_scenes
                      else "val" if eid in val_scenes
                      else "excl")
        per_env[eid] = m

    # Print table
    print()
    print(f"{'env':>3} {'role':>5} {'n_P':>6} {'n_N':>7} "
          f"{'TP':>6} {'FN':>6} {'FP':>6} {'TN':>7} "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'FNR':>6}")
    for eid in range(21):
        m = per_env.get(eid)
        if m is None:
            continue
        prec = f"{m['precision']:.3f}" if m['precision'] is not None else "  n/a "
        rec = f"{m['recall']:.3f}" if m['recall'] is not None else "  n/a "
        f1 = f"{m['f1']:.3f}" if m['f1'] is not None else "  n/a "
        fnr = f"{m['fnr']:.3f}" if m['fnr'] is not None else "  n/a "
        print(f"{eid:>3} {m['role']:>5} {m['n_P_actual']:>6d} {m['n_N_actual']:>7d} "
              f"{m['TP']:>6d} {m['FN']:>6d} {m['FP']:>6d} {m['TN']:>7d} "
              f"{prec:>6} {rec:>6} {f1:>6} {fnr:>6}")

    # Summary
    print()
    print(f"--- summary (per role, averaged over envs that have NLOS cells) ---")
    print(f"{'role':>5} {'n_envs':>7} {'mean_P':>7} {'mean_R':>7} "
          f"{'mean_F1':>8} {'mean_FNR':>9} {'mean_specificity':>17}")
    for role in ("train", "val", "excl"):
        envs_with_nlos = [e for e, m in per_env.items()
                           if m["role"] == role and m["precision"] is not None]
        if not envs_with_nlos:
            continue
        mean_prec = np.mean([per_env[e]["precision"] for e in envs_with_nlos])
        mean_rec = np.mean([per_env[e]["recall"] for e in envs_with_nlos])
        mean_f1 = np.mean([per_env[e]["f1"] for e in envs_with_nlos])
        mean_fnr = np.mean([per_env[e]["fnr"] for e in envs_with_nlos])
        # Specificity is defined for both all-LOS and mixed envs
        spec_envs = [e for e, m in per_env.items()
                      if m["role"] == role and m["specificity"] is not None]
        mean_spec = np.mean([per_env[e]["specificity"] for e in spec_envs])
        print(f"{role:>5} {len(envs_with_nlos):>7d} {mean_prec:>7.3f} "
              f"{mean_rec:>7.3f} {mean_f1:>8.3f} {mean_fnr:>9.3f} "
              f"{mean_spec:>17.3f}")

    # Save json
    out_json = args.out_json or str(Path(args.ckpt).parent / "nlos_confusion_per_env.json")
    with open(out_json, "w") as f:
        json.dump({
            "ckpt": str(args.ckpt),
            "train_scenes": sorted(train_scenes),
            "val_scenes": sorted(val_scenes),
            "per_env": {str(e): m for e, m in per_env.items()},
        }, f, indent=2)
    print(f"\nMetrics saved to {out_json}")

    # ---- Figure ----
    out_fig = args.out_fig or "figures/tier2_v2_nlos_confusion.png"
    if not out_fig.startswith("/"):
        out_fig = str(Path("/Users/justin/ultraclear-paper") / out_fig)
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)

    role_color = {"train": "tab:blue", "val": "tab:orange"}
    nlos_envs = sorted([e for e, m in per_env.items() if m["n_P_actual"] > 0])
    nlos_envs_train = [e for e in nlos_envs if e in train_scenes]
    nlos_envs_val = [e for e in nlos_envs if e in val_scenes]
    n_train = len(nlos_envs_train)
    n_val = len(nlos_envs_val)

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.30, wspace=0.20)
    x = np.arange(len(nlos_envs))
    colors = [role_color[per_env[e]["role"]] for e in nlos_envs]

    # --- Panel 1 (top, spans both): stacked bar of TP/FN/FP/TN ---
    ax1 = fig.add_subplot(gs[0, :])
    tps = np.array([per_env[e]["TP"] for e in nlos_envs])
    fns = np.array([per_env[e]["FN"] for e in nlos_envs])
    fps = np.array([per_env[e]["FP"] for e in nlos_envs])
    tns = np.array([per_env[e]["TN"] for e in nlos_envs])
    # Stack: FN (bottom, danger) | TP (correct NLOS) | FP (wasted) | TN (correct LOS)
    # Color choices: red=FN (missed NLOS, the dangerous one),
    #                green=TP (caught NLOS), yellow=FP (false alarm),
    #                gray=TN (correct LOS, vast majority)
    ax1.bar(x, fns, color="tab:red", alpha=0.85, label="FN (missed NLOS)",
             edgecolor="black", linewidth=0.4)
    ax1.bar(x, tps, bottom=fns, color="tab:green", alpha=0.85,
             label="TP (caught NLOS)", edgecolor="black", linewidth=0.4)
    ax1.bar(x, fps, bottom=fns + tps, color="goldenrod", alpha=0.85,
             label="FP (false alarm: LOS → NLOS)", edgecolor="black",
             linewidth=0.4)
    ax1.bar(x, tns, bottom=fns + tps + fps, color="lightgray", alpha=0.85,
             label="TN (correct LOS)", edgecolor="black", linewidth=0.4)
    if n_train and n_val:
        ax1.axvline(n_train - 0.5, color="black", linestyle="--",
                     alpha=0.5, linewidth=1.0)
    # Total cells per env (for reference)
    totals = fns + tps + fps + tns
    for i, e in enumerate(nlos_envs):
        ax1.text(i, totals[i] + 2000, f"{totals[i]/1000:.0f}k",
                  ha="center", va="bottom", fontsize=7, color="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nlos_envs, fontsize=9)
    ax1.set_xlabel("env_id (envs with NLOS cells)")
    ax1.set_ylabel("# cells (stacked TP/FN/FP/TN)")
    ax1.set_title(
        f"Per-env confusion matrix (cell counts) — Tier 2 v2 NLOS classification\n"
        f"FN (red) is the danger: NLOS cells passed through unmarked. "
        f"FP (yellow) is the waste: good LOS cells down-weighted. "
        f"Each bar = 1,200 windows × 128 cells = 153,600 cells",
        fontsize=11
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    # --- Panel 2 (bottom-left): Precision/Recall/F1 per env ---
    ax2 = fig.add_subplot(gs[1, 0])
    precs = [per_env[e]["precision"] for e in nlos_envs]
    recs = [per_env[e]["recall"] for e in nlos_envs]
    f1s = [per_env[e]["f1"] for e in nlos_envs]
    width = 0.27
    ax2.bar(x - width, precs, width=width, color="tab:purple",
             alpha=0.75, label="Precision", edgecolor="black",
             linewidth=0.4)
    ax2.bar(x, recs, width=width, color="tab:cyan",
             alpha=0.75, label="Recall (TPR)", edgecolor="black",
             linewidth=0.4)
    ax2.bar(x + width, f1s, width=width, color="tab:olive",
             alpha=0.75, label="F1", edgecolor="black", linewidth=0.4)
    if n_train and n_val:
        ax2.axvline(n_train - 0.5, color="black", linestyle="--",
                     alpha=0.5, linewidth=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(nlos_envs, fontsize=9)
    ax2.set_xlabel("env_id (envs with NLOS cells)")
    ax2.set_ylabel("score")
    ax2.set_ylim(0, 1.05)
    mean_p = np.mean(precs); mean_r = np.mean(recs); mean_f1 = np.mean(f1s)
    ax2.set_title(
        f"Per-env Precision / Recall / F1\n"
        f"mean: P={mean_p:.3f}, R={mean_r:.3f}, F1={mean_f1:.3f}",
        fontsize=10
    )
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    # --- Panel 3 (bottom-right): FNR (miss rate) per env ---
    ax3 = fig.add_subplot(gs[1, 1])
    fnrs = [per_env[e]["fnr"] for e in nlos_envs]
    ax3.bar(x, fnrs, color="tab:red", alpha=0.85,
             label="FNR (NLOS cells missed)",
             edgecolor="black", linewidth=0.4)
    if n_train and n_val:
        ax3.axvline(n_train - 0.5, color="black", linestyle="--",
                     alpha=0.5, linewidth=1.0)
    for i, (e, f) in enumerate(zip(nlos_envs, fnrs)):
        ax3.text(i, f + 0.015, f"{f:.2f}", ha="center", va="bottom",
                 fontsize=7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(nlos_envs, fontsize=9)
    ax3.set_xlabel("env_id (envs with NLOS cells)")
    ax3.set_ylabel("False Negative Rate (miss rate)")
    ax3.set_ylim(0, max(0.30, max(fnrs) * 1.15))
    # How many actual NLOS cells did env 4 miss?
    n_missed_env4 = per_env[4]["FN"]
    n_actual_env4 = per_env[4]["n_P_actual"]
    ax3.set_title(
        f"Per-env False Negative Rate (miss rate)\n"
        f"env 4 missed {n_missed_env4:,} of {n_actual_env4:,} NLOS cells "
        f"({per_env[4]['fnr']*100:.1f}%) — the danger for downstream loc",
        fontsize=10
    )
    ax3.grid(True, axis="y", alpha=0.3)

    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_fig}")


if __name__ == "__main__":
    main()
