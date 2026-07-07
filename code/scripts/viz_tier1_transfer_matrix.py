"""Tier 1 (E) — Transfer matrix: train env -> val env.

For each (train_env, val_env) pair, do K-shot adaptation using K
windows from train_env, then evaluate query MSE on val_env. The
resulting 2D matrix shows which train environments transfer
best to which held-out environments.

This complements the few-shot curve: the few-shot curve shows
*how many* shots are needed; the transfer matrix shows *which
scene* the shots should come from for best cross-scene transfer.

A "good" FOMAML initialization produces a transfer matrix with
no catastrophically bad cells --- any train env's K shots
should be useful, and the matrix should be roughly uniform.
A "brittle" init produces a sparse matrix with some cells much
worse than others (the model has memorized a particular
geometry).

Usage:
    python3.11 -m scripts.viz_tier1_transfer_matrix \\
        --ckpt_random /tmp/tier1_random/tier1_sionna.pt \\
        --ckpt_tier0  /tmp/tier1_t0init/tier1_sionna.pt \\
        --out_dir /path/to/save/figures
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as TF
from torch.func import functional_call
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.config import Config
from ultraclear.tier1_maml import MAMLCore, adapt_and_evaluate
from scripts.train_tier1_sionna import sample_task_multi
from scripts.multi_scene_dataset import build_default_multi_scene


def transfer_eval(
    core_state, cfg, train_env, val_env, ds, K=20, n_trials=15
):
    """Adapt using K shots from train_env, evaluate on val_env.
    Returns (mean, se) of query MSE over n_trials independent
    K-shot draws.
    """
    errs = []
    for trial in range(n_trials):
        try:
            sx, sy = sample_task_multi(ds, train_env, K, cfg.seed + 11000 + trial)
            qx, qy = sample_task_multi(ds, val_env, 20, cfg.seed + 12000 + trial)
        except RuntimeError:
            continue
        fresh = MAMLCore(cfg)
        fresh.load_state_dict(core_state)
        q_loss, _ = adapt_and_evaluate(
            fresh, sx, sy, qx, qy,
            inner_lr=cfg.maml_inner_lr,
            inner_steps=cfg.maml_inner_steps,
        )
        errs.append(float(q_loss.item()))
    if not errs:
        return float("nan"), float("nan")
    return float(np.mean(errs)), float(np.std(errs) / np.sqrt(len(errs)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_random", required=True)
    p.add_argument("--ckpt_tier0", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--train_scenes", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--val_scenes", type=int, nargs="+", default=[4, 5])
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--n_trials", type=int, default=15)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt_random).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading multi-scene dataset...")
    ds, scene_info = build_default_multi_scene(n_channels=2, seq_len=32, stride=32)
    train_set = set(args.train_scenes)
    norm_mean, norm_std = ds.compute_norm_stats(train_set)
    ds.norm_mean = norm_mean
    ds.norm_std = norm_std

    ckpt_r = torch.load(args.ckpt_random, weights_only=False)
    ckpt_t = torch.load(args.ckpt_tier0, weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt_r["cfg"].items()
                    if k in Config.__dataclass_fields__})

    # compute transfer matrices for both variants
    n_train = len(args.train_scenes)
    n_val = len(args.val_scenes)
    mat_r = np.zeros((n_val, n_train))
    mat_t = np.zeros((n_val, n_train))
    mat_r_se = np.zeros((n_val, n_train))
    mat_t_se = np.zeros((n_val, n_train))

    print(f"K = {args.K}, n_trials = {args.n_trials}")
    print(f"  train scenes (columns): {[scene_info[i].name for i in args.train_scenes]}")
    print(f"  val scenes (rows):      {[scene_info[i].name for i in args.val_scenes]}")
    print()
    for vi, ve in enumerate(args.val_scenes):
        for ti, te in enumerate(args.train_scenes):
            print(f"  ({scene_info[ve].name} | {scene_info[te].name})... ", end="", flush=True)
            mr, ser = transfer_eval(ckpt_r["model_state"], cfg, te, ve, ds,
                                     K=args.K, n_trials=args.n_trials)
            mt, set_ = transfer_eval(ckpt_t["model_state"], cfg, te, ve, ds,
                                     K=args.K, n_trials=args.n_trials)
            mat_r[vi, ti] = mr
            mat_t[vi, ti] = mt
            mat_r_se[vi, ti] = ser
            mat_t_se[vi, ti] = set_
            print(f"random={mr:.2f}  tier0={mt:.2f}")

    # ---- plot: 2-panel side-by-side heatmap ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    vmax = max(np.nanmax(mat_r), np.nanmax(mat_t))
    vmin = min(np.nanmin(mat_r), np.nanmin(mat_t))
    train_labels = [scene_info[i].name for i in args.train_scenes]
    val_labels = [scene_info[i].name for i in args.val_scenes]

    for ax, mat, title in [(axes[0], mat_r, "random init"),
                            (axes[1], mat_t, "Tier-0 pretrained")]:
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                        vmin=vmin, vmax=vmax)
        # annotate
        for vi in range(n_val):
            for ti in range(n_train):
                ax.text(ti, vi, f"{mat[vi, ti]:.1f}",
                        ha="center", va="center",
                        color="white" if mat[vi, ti] > (vmax + vmin) / 2 else "black",
                        fontsize=10)
        ax.set_xticks(range(n_train))
        ax.set_xticklabels(train_labels, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(n_val))
        ax.set_yticklabels(val_labels, fontsize=9)
        ax.set_xlabel("train scene (where K shots come from)")
        ax.set_ylabel("held-out scene (where we evaluate)")
        ax.set_title(f"{title}\n"
                      f"mean MSE = {np.nanmean(mat):.1f} m²")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MSE (m²)")

    fig.suptitle(f"Transfer matrix: train→val MSE at K={args.K}\n"
                  "(rows = held-out scene, columns = scene the K shots come from)",
                  fontsize=11)
    fig.tight_layout()
    out = out_dir / "tier1_transfer_matrix.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nsaved {out}")

    # save CSV
    csv_out = out_dir / "tier1_transfer_matrix.csv"
    with open(csv_out, "w") as f:
        f.write("variant,val_env,train_env,mse,se\n")
        for vi, ve in enumerate(args.val_scenes):
            for ti, te in enumerate(args.train_scenes):
                f.write(f"random,{ve},{te},{mat_r[vi, ti]:.4f},{mat_r_se[vi, ti]:.4f}\n")
                f.write(f"tier0,{ve},{te},{mat_t[vi, ti]:.4f},{mat_t_se[vi, ti]:.4f}\n")
    print(f"saved {csv_out}")

    # print summary
    print()
    print("=" * 70)
    print(f"  Transfer matrix (MSE at K={args.K})")
    print("=" * 70)
    for vi, ve in enumerate(args.val_scenes):
        print(f"\n  val scene: {scene_info[ve].name}")
        for ti, te in enumerate(args.train_scenes):
            red = (mat_r[vi, ti] - mat_t[vi, ti]) / mat_r[vi, ti] * 100
            print(f"    from {scene_info[te].name:<32s}  "
                  f"random={mat_r[vi, ti]:6.2f}  tier0={mat_t[vi, ti]:6.2f}  "
                  f"red={red:+5.1f}%")


if __name__ == "__main__":
    main()
