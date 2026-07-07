"""Tier 1 (A) — Per-scene localization error heatmap.

For each held-out val scene, take the FOMAML-adapted model
(using K=20 shots) and evaluate on all available windows.
Plot the tag (x, y) positions on the floor plan, colored by
per-position query MSE. Side-by-side: random init vs
Tier-0 pretrained.

This shows *where in the room* the model fails. If the
random init fails mostly at the edges (where geometry is
harder) while the Tier-0 init is uniformly good, that's
the qualitative story we want for the paper.

Usage:
    python3.11 -m scripts.viz_tier1_error_heatmap \\
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
from matplotlib.colors import LogNorm

from ultraclear.config import Config
from ultraclear.tier1_maml import MAMLCore, adapt_and_evaluate
from scripts.train_tier1_sionna import sample_task_multi
from scripts.multi_scene_dataset import build_default_multi_scene


def evaluate_on_all_windows(
    core_state, cfg, eid, ds, K=20, n_shot_reps=3, n_query_per_pos=1
):
    """Adapt the model to scene eid using K shots, then evaluate
    query MSE on every available window in that scene. Returns
    (positions, per_position_mse).
    """
    # adapt on K shots (n_shot_reps independent K-shots to reduce
    # support-sample variance)
    n_trials = n_shot_reps
    fresh = MAMLCore(cfg)
    fresh.load_state_dict(core_state)
    fresh.train()
    # a single adaptation from a single K-shot set
    sx, sy = sample_task_multi(ds, eid, K, cfg.seed + 9000)
    qx, qy = sample_task_multi(ds, eid, 20, cfg.seed + 9100)
    adapt_and_evaluate(fresh, sx, sy, qx, qy,
                        inner_lr=cfg.maml_inner_lr,
                        inner_steps=cfg.maml_inner_steps)
    # now evaluate on all windows in this scene
    fresh.eval()
    indices = [i for i, (s_idx, _, _) in enumerate(ds.windows)
                if ds.scene_info[s_idx].env_id == eid]
    positions = []
    mses = []
    with torch.no_grad():
        for i in indices:
            x, y, _, _, _ = ds[i]
            xb = x.unsqueeze(0)
            yb = y.unsqueeze(0)
            pred = functional_call(fresh, dict(fresh.named_parameters()), (xb,))
            err = float(((pred - yb) ** 2).sum().item())  # squared L2 in 2D = MSE
            pos = (float(y[0]), float(y[1]))
            positions.append(pos)
            mses.append(err)
    return np.array(positions), np.array(mses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_random", required=True)
    p.add_argument("--ckpt_tier0", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--val_scenes", type=int, nargs="+", default=[4, 5])
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--downsample", type=int, default=5,
                   help="evaluate every Nth window to keep the plot "
                        "readable; set to 1 for every window")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt_random).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading multi-scene dataset...")
    ds, scene_info = build_default_multi_scene(n_channels=2, seq_len=32, stride=32)
    train_scenes = set([0, 1, 2, 3])
    norm_mean, norm_std = ds.compute_norm_stats(train_scenes)
    ds.norm_mean = norm_mean
    ds.norm_std = norm_std

    # load both checkpoints
    ckpt_r = torch.load(args.ckpt_random, weights_only=False)
    ckpt_t = torch.load(args.ckpt_tier0, weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt_r["cfg"].items()
                    if k in Config.__dataclass_fields__})

    # one figure per val scene, with 2 subplots (random, tier0)
    for eid in args.val_scenes:
        info = scene_info[eid]
        print(f"\nscene {eid} ({info.name}): {info.n_grids} grids, "
              f"NLOS rate {info.nlos_rate*100:.1f}%")

        print(f"  evaluating random init (K={args.K})...")
        pos_r, mse_r = evaluate_on_all_windows(ckpt_r["model_state"], cfg, eid, ds,
                                                  K=args.K)
        print(f"  evaluating tier0 init (K={args.K})...")
        pos_t, mse_t = evaluate_on_all_windows(ckpt_t["model_state"], cfg, eid, ds,
                                                  K=args.K)

        # downsample for plotting
        if args.downsample > 1:
            pos_r = pos_r[::args.downsample]
            mse_r = mse_r[::args.downsample]
            pos_t = pos_t[::args.downsample]
            mse_t = mse_t[::args.downsample]

        # shared color scale (log, m²)
        vmin = max(0.5, min(mse_r.min(), mse_t.min()))
        vmax = max(mse_r.max(), mse_t.max())
        norm = LogNorm(vmin=vmin, vmax=vmax)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, (pos, mse, name) in zip(axes,
                                          [(pos_r, mse_r, "random init"),
                                           (pos_t, mse_t, "Tier-0 pretrained")]):
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=mse, s=20, cmap="viridis",
                             norm=norm, edgecolors="none")
            ax.set_aspect("equal")
            ax.set_xlabel("tag x (m)")
            ax.set_ylabel("tag y (m)")
            ax.set_title(f"{name}\n"
                          f"median MSE = {np.median(mse):.2f} m², "
                          f"max = {mse.max():.1f} m²")
            ax.grid(True, alpha=0.3)
            # add a small colorbar
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="MSE (m²)")
        fig.suptitle(f"Held-out scene: {info.name}  (K={args.K}, "
                      f"vmin={vmin:.2f} m² log-scale)",
                      fontsize=11)
        fig.tight_layout()
        out = out_dir / f"tier1_error_heatmap_env{eid}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  saved {out}")

        # also save CSV with per-position MSE
        csv_out = out_dir / f"tier1_error_heatmap_env{eid}.csv"
        with open(csv_out, "w") as f:
            f.write("tag_x,tag_y,random_mse,tier0_mse\n")
            # use the non-downsampled arrays for the CSV
            pos_r_full, mse_r_full = evaluate_on_all_windows(
                ckpt_r["model_state"], cfg, eid, ds, K=args.K)
            pos_t_full, mse_t_full = evaluate_on_all_windows(
                ckpt_t["model_state"], cfg, eid, ds, K=args.K)
            # match by position
            for i, (x, y) in enumerate(pos_r_full):
                # find matching position in tier0
                idx = np.where((np.abs(pos_t_full - (x, y)).sum(axis=1)) < 1e-3)[0]
                if len(idx) > 0:
                    t_mse = mse_t_full[idx[0]]
                else:
                    t_mse = float("nan")
                f.write(f"{x},{y},{mse_r_full[i]:.4f},{t_mse:.4f}\n")
        print(f"  saved {csv_out}")


if __name__ == "__main__":
    main()
