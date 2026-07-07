"""Tier 2 — FiLM modulation: visualize gamma(c) and beta(c).

Sanity check that the FiLM is *doing* something. For a few example
windows, plot the per-channel gamma and beta values produced by the
FiLM layer given the c for that window.

Expected: gamma is near 1.0 and beta is near 0.0 at the start of
training (the init), and diverges from those defaults over training.
If gamma is all 1.0 and beta is all 0.0 in the trained model, the
FiLM collapsed and Tier 2 is broken.

Two plots:
  - Histogram of gamma values across all windows and channels (should
    have spread, not be a delta at 1.0).
  - Heatmap of gamma and beta for a few specific windows, channel-wise.

Usage:
  python -m scripts.viz_tier2_film
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ultraclear.config import Config
from ultraclear.data import build_synthetic_dataset, TensorUWB
from scripts.train_tier2_synthetic import DisentangledTrainer


@torch.no_grad()
def collect_film_params(model, loader, device):
    """Collect (gamma, beta, env, cond) for every window."""
    gammas, betas, envs, conds = [], [], [], []
    for X, _y, env, cond, _nlos in loader:
        X = X.to(device)
        h = model.disent.encoder(X)
        c = model.disent.h_phi(h)
        # FiLM forward: gamma = 1 + gamma_net(c), beta = beta_net(c)
        gamma = 1.0 + model.disent.film.gamma_net(c)
        beta = model.disent.film.beta_net(c)
        gammas.append(gamma.cpu().numpy())
        betas.append(beta.cpu().numpy())
        envs.append(env.numpy())
        conds.append(cond.numpy())
    return (np.concatenate(gammas, 0), np.concatenate(betas, 0),
            np.concatenate(envs, 0), np.concatenate(conds, 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/tier2_synthetic/tier2_synthetic.pt")
    p.add_argument("--out", type=str, default="figures/tier2_synthetic/film_modulation.png")
    p.add_argument("--max_points", type=int, default=400)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    print(f"loading {args.ckpt}...")
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    cfg.device = args.device
    n_envs = ck["n_envs"]
    model = DisentangledTrainer(cfg, n_envs=n_envs).to(args.device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    print("building data...")
    all_items = build_synthetic_dataset(cfg, seed=cfg.seed)
    if len(all_items) > args.max_points:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(all_items))[:args.max_points]
        items = [all_items[i] for i in idx]
    else:
        items = all_items
    loader = DataLoader(TensorUWB(items), batch_size=64)
    gammas, betas, envs, conds = collect_film_params(model, loader, args.device)
    print(f"gamma shape={gammas.shape}  mean={gammas.mean():.3f}  std={gammas.std():.3f}  "
          f"min={gammas.min():.3f}  max={gammas.max():.3f}")
    print(f"beta  shape={betas.shape}  mean={betas.mean():.3f}  std={betas.std():.3f}  "
          f"min={betas.min():.3f}  max={betas.max():.3f}")

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # Top: histograms of gamma and beta values
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(gammas.flatten(), bins=60, color="C0", alpha=0.7, edgecolor="white")
    ax.axvline(1.0, color="red", linestyle="--", label="γ=1 (init)")
    ax.set_xlabel("γ value")
    ax.set_ylabel("count")
    ax.set_title("FiLM gamma distribution\n(should NOT be δ at 1.0 if FiLM is active)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(betas.flatten(), bins=60, color="C1", alpha=0.7, edgecolor="white")
    ax.axvline(0.0, color="red", linestyle="--", label="β=0 (init)")
    ax.set_xlabel("β value")
    ax.set_ylabel("count")
    ax.set_title("FiLM beta distribution\n(should NOT be δ at 0.0 if FiLM is active)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Middle: gamma by micro condition (should differ across conds)
    ax = fig.add_subplot(gs[1, 0])
    for ci in range(3):
        mask = conds == ci
        if mask.sum() == 0:
            continue
        per_chan = gammas[mask].mean(axis=0)               # (d_model,)
        ax.plot(per_chan, label=f"cond {ci} (n={mask.sum()})", alpha=0.8)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="γ=1 (init)")
    ax.set_xlabel("channel index")
    ax.set_ylabel("mean γ")
    ax.set_title("FiLM γ per channel, by micro condition\n"
                 "(different curves = FiLM responds to c)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    for ci in range(3):
        mask = conds == ci
        if mask.sum() == 0:
            continue
        per_chan = betas[mask].mean(axis=0)
        ax.plot(per_chan, label=f"cond {ci} (n={mask.sum()})", alpha=0.8)
    ax.axhline(0.0, color="red", linestyle="--", alpha=0.5, label="β=0 (init)")
    ax.set_xlabel("channel index")
    ax.set_ylabel("mean β")
    ax.set_title("FiLM β per channel, by micro condition")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom: heatmaps of 6 specific windows (2 per cond)
    ax = fig.add_subplot(gs[2, :])
    # pick first 2 of each cond
    chosen = []
    for ci in range(3):
        idxs = np.where(conds == ci)[0][:2]
        chosen.extend(list(idxs))
    n_show = len(chosen)
    im = ax.imshow(gammas[chosen], aspect="auto", cmap="RdBu_r",
                   vmin=0.5, vmax=1.5, interpolation="nearest")
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f"env {envs[i]} cond {conds[i]}" for i in chosen], fontsize=8)
    ax.set_xlabel("channel index")
    ax.set_title("FiLM γ heatmap for 6 example windows (2 per micro condition)\n"
                 "clamped to [0.5, 1.5] for visibility")
    plt.colorbar(im, ax=ax, label="γ")

    fig.suptitle("Tier 2 — FiLM modulation analysis (γ, β vs micro condition)", fontsize=14)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
