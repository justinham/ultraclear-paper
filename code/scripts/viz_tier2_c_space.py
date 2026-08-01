"""Tier 2 — t-SNE of c, colored by env and by cond.

The visual disentanglement figure. Two side-by-side panels:
  - Left:  c projected to 2D with t-SNE, colored by env_id (8 colors)
           → should look like one undifferentiated blob (entangled envs)
  - Right: same c, colored by cond_id (3 colors)
           → should show 3 clean clusters (micro signal preserved)

If the left looks blob-y and the right looks cluster-y, Tier 2 worked.

Usage:
  python -m scripts.viz_tier2_c_space
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from ultraclear.config import Config
from ultraclear.data import build_synthetic_dataset, TensorUWB
from scripts.train_tier2_synthetic import DisentangledTrainer
from scripts.viz_tier2_helpers import add_joint_flag, resolve_paths, load_model_from_ckpt


def main():
    p = argparse.ArgumentParser()
    add_joint_flag(p)
    p.add_argument("--joint_ckpt", type=str, default="checkpoints/joint/phase_C.pt")
    p.add_argument("--max_points", type=int, default=600)
    p.add_argument("--perplexity", type=int, default=30)
    args = p.parse_args()

    args.ckpt, args.out = resolve_paths(args)

    print(f"loading {args.ckpt}...")
    model, cfg, ck = load_model_from_ckpt(args.ckpt, args.device)

    print("building data...")
    all_items = build_synthetic_dataset(cfg, seed=cfg.seed)
    if len(all_items) > args.max_points:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(all_items))[:args.max_points]
        items = [all_items[i] for i in idx]
    else:
        items = all_items
    loader = DataLoader(TensorUWB(items), batch_size=64)

    c_buf, env_buf, cond_buf = [], [], []
    with torch.no_grad():
        for X, _y, env, cond, _nlos in loader:
            X = X.to(args.device)
            _h, c, _adv, _loc, _nlos, _cond = model(X)
            c_buf.append(c.cpu().numpy())
            env_buf.append(env.numpy())
            cond_buf.append(cond.numpy())
    c_all = np.concatenate(c_buf, axis=0)
    env_all = np.concatenate(env_buf, axis=0)
    cond_all = np.concatenate(cond_buf, axis=0)
    print(f"c shape: {c_all.shape}, n_envs in sample: {len(np.unique(env_all))}, "
          f"n_conds: {len(np.unique(cond_all))}")

    print("t-SNE...")
    # n_components=2, perplexity in [5, 50], init=pca for stability
    perp = min(args.perplexity, max(5, (len(c_all) - 1) // 3))
    z = TSNE(n_components=2, perplexity=perp, init="pca",
             random_state=0).fit_transform(c_all)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap_env = plt.get_cmap("tab10", max(n_envs, 10))
    cmap_cond = plt.get_cmap("viridis", 3)

    for ei in range(n_envs):
        mask = env_all == ei
        axes[0].scatter(z[mask, 0], z[mask, 1], s=10, alpha=0.6,
                        color=cmap_env(ei), label=f"env {ei}")
    axes[0].set_title("c colored by env_id\n"
                      "(blob = envs entangled in c ✓)")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.8)
    axes[0].grid(True, alpha=0.3)

    for ci in range(3):
        mask = cond_all == ci
        axes[1].scatter(z[mask, 0], z[mask, 1], s=10, alpha=0.6,
                        color=cmap_cond(ci), label=f"cond {ci}")
    axes[1].set_title("c colored by micro condition\n"
                      "(clusters = micro signal preserved ✓)")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Tier 2 — disentangled condition space (n={len(c_all)})", fontsize=14)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
