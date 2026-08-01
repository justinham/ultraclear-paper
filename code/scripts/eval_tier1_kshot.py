"""Tier 1 K-shot adaptation curve per env (K=0, 1, 5, 10, 20).

For each Sionna scene, run few-shot adaptation at K=0 (raw meta-init),
K=1, K=5, K=10, K=20 and report the per-window loc RMSE in meters
(per-scene denormalization).

Output:
  - figures/tier1_kshot_per_env.csv   per-env, per-K RMSE
  - figures/tier1_kshot_per_env.png   2-subplot figure (TRAIN vs VAL)

Usage:
    python -m scripts.eval_tier1_kshot
    python -m scripts.eval_tier1_kshot --ks 0 1 5 10 20 --n_trials 30
    python -m scripts.eval_tier1_kshot --train_only
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.config import Config
from ultraclear.tier1_maml import MAMLCore, adapt_and_evaluate
from scripts.train_tier1_sionna import sample_task_multi
from scripts.multi_scene_dataset import build_default_multi_scene


def denorm_z_to_meters(z_xy, sinfo):
    """Convert z-scored (x, y) back to meters using per-scene stats."""
    if sinfo.tag_xy_mean is None or sinfo.tag_xy_std is None:
        return z_xy
    return z_xy * sinfo.tag_xy_std + sinfo.tag_xy_mean


def eval_K_on_env(env_id, K, n_trials, ds, cfg, core_state, sinfo, device="cpu"):
    """K-shot adaptation on env_id. Returns RMSE in meters over n_trials.

    K=0: no support, no inner loop; raw meta-init. We get pred
    via `MAMLCore.forward(X)` directly (no adapt_and_evaluate).
    K>0: standard MAML K-shot adapt on K support, predict on 20 query.
    """
    loc_errs_m = []
    for trial in range(n_trials):
        if K == 0:
            # Zero-shot: just forward the query through the meta-init model
            # Sample 20 random windows for query, no support used
            qx, qy = sample_task_multi(ds, env_id, 20, cfg.seed + 6000 + trial)
            qx, qy = qx.to(device), qy.to(device)
            fresh = MAMLCore(cfg)
            fresh.load_state_dict(core_state)
            with torch.no_grad():
                pred_z = fresh(qx).cpu().numpy()
            T = qx.size(1)
            y_mid_z = qy[:, T // 2, :]
            pred_m = denorm_z_to_meters(pred_z, sinfo)
            y_mid_m = denorm_z_to_meters(y_mid_z.cpu().numpy(), sinfo)
            err = np.linalg.norm(pred_m - y_mid_m, axis=-1)
            loc_errs_m.extend(err.tolist())
        else:
            sx, sy = sample_task_multi(ds, env_id, K, cfg.seed + 5000 + trial)
            qx, qy = sample_task_multi(ds, env_id, 20, cfg.seed + 6000 + trial)
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)
            fresh = MAMLCore(cfg).to(device)
            fresh.load_state_dict(core_state)
            q_loss, info = adapt_and_evaluate(
                fresh, sx, sy, qx, qy,
                inner_lr=cfg.maml_inner_lr,
                inner_steps=cfg.maml_inner_steps,
            )
            pred_z = info["q_pred"].cpu().numpy()
            T = qx.size(1)
            y_mid_z = qy[:, T // 2, :].cpu().numpy()
            pred_m = denorm_z_to_meters(pred_z, sinfo)
            y_mid_m = denorm_z_to_meters(y_mid_z, sinfo)
            err = np.linalg.norm(pred_m - y_mid_m, axis=-1)
            loc_errs_m.extend(err.tolist())
    rmse = float(np.sqrt(np.mean(np.array(loc_errs_m) ** 2)))
    return rmse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/tier1_sionna_best.pt")
    p.add_argument("--ks", type=int, nargs="+", default=[0, 1, 5, 10, 20])
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--out_csv", default="figures/tier1_kshot_per_env.csv")
    p.add_argument("--out_fig", default="figures/tier1_kshot_per_env.png")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    print(f"loading ckpt: {args.ckpt}")
    ckpt_path = Path(args.ckpt)
    # Footgun guard: the trainer writes the ckpt to
    # `<out>/tier1_sionna.pt` where `<out>` is a directory. If the
    # user passes the directory itself, fix it up before loading.
    if ckpt_path.is_dir():
        nested = ckpt_path / "tier1_sionna.pt"
        if nested.exists():
            print(f"  (ckpt path is a directory; using {nested} instead)")
            args.ckpt = str(nested)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    core_state = ck["model_state"]
    print(f"  K values: {args.ks}, n_trials: {args.n_trials}")
    print(f"  train_scenes: {ck.get('train_scenes')}")
    print(f"  val_scenes: {ck.get('val_scenes')}")

    print("\nloading Sionna data...")
    ds, scene_info = build_default_multi_scene(
        n_channels=cfg.n_channels, seq_len=cfg.seq_len, stride=32, all_scenes=True,
    )
    train_scenes = set(ck.get("train_scenes") or list(range(10)))
    norm_mean, norm_std = ds.compute_norm_stats(train_scenes)
    ds.norm_mean = norm_mean
    ds.norm_std = norm_std
    print(f"  {len(ds)} windows across {len(scene_info)} scenes")

    # Per-env, per-K eval
    env_ids = [e for e in range(21) if e != 14]  # skip env 14 (missing-data)
    results = {}  # env_id -> {K -> rmse}
    for eid in env_ids:
        sinfo = next(s for s in scene_info if s.env_id == eid)
        n_w = sum(1 for s_idx, _, _ in ds.windows if ds.scene_info[s_idx].env_id == eid)
        if n_w == 0:
            continue
        results[eid] = {}
        for K in args.ks:
            rmse = eval_K_on_env(eid, K, args.n_trials, ds, cfg, core_state, sinfo, device=args.device)
            results[eid][K] = rmse
            print(f"  env {eid:>2} K={K:>2}: {rmse:.3f} m")

    # Save CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_id", "role"] + [f"K={K}" for K in args.ks])
        for eid in sorted(results):
            role = "TRAIN" if eid in train_scenes else "VAL"
            w.writerow([eid, role] + [f"{results[eid][K]:.4f}" for K in args.ks])
    print(f"\nsaved CSV → {out_csv}")

    # Plot: 2 subplots (TRAIN + VAL), each with grouped bars
    # Use the ckpt's actual train/val split (mixed splits put some
    # high-index envs in TRAIN and some low-index envs in VAL).
    train_envs = [e for e in sorted(train_scenes) if e in results]
    val_envs = [e for e in sorted(set(ck.get("val_scenes") or []) - train_scenes)
                if e in results and e != 14]
    Ks = args.ks
    n_K = len(Ks)

    # Per-env trn/val tag so a reader can tell at a glance which envs
    # are in the ckpt's train set vs val set. No indoor/outdoor tag —
    # env 2 ("empty_open") is neither and the env type is secondary
    # to the train/val distinction.
    def _xtick(eid):
        if eid in train_scenes:
            tag = "trn"
        else:
            tag = "val"
        return f"e{eid}\n{tag}"

    # K-shot curve colors (K=0 grey, K=1 blue, K=5 green, K=10 orange, K=20 red)
    K_colors = ["C7", "C0", "C2", "C1", "C3"]
    K_labels = [f"K={K}" for K in Ks]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # ---- TRAIN ----
    ax = axes[0]
    x_pos = np.arange(len(train_envs))
    width = 0.8 / n_K
    for ki, K in enumerate(Ks):
        vals = [results[e][K] for e in train_envs]
        offset = (ki - (n_K - 1) / 2) * width
        bars = ax.bar(x_pos + offset, vals, width, color=K_colors[ki % len(K_colors)],
                       label=K_labels[ki], edgecolor="white", linewidth=0.3)
        # only label the last bar (K=10) for clarity
        if K == 10:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                         ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([_xtick(e) for e in train_envs], rotation=0, fontsize=8)
    ax.set_xlabel("(trn = ckpt train, in/out = env type)", fontsize=9, style="italic")
    ax.set_ylabel("loc RMSE (m)", fontsize=11)
    ax.set_title(f"TRAIN — Tier 1 K-shot adaptation curve", fontsize=11)
    ax.legend(fontsize=8, ncol=n_K, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ymax = max(max(results[e][K] for K in Ks) for e in results) * 1.15
    ax.set_ylim(0, ymax)

    # ---- VAL ----
    ax = axes[1]
    x_pos = np.arange(len(val_envs))
    for ki, K in enumerate(Ks):
        vals = [results[e][K] for e in val_envs]
        offset = (ki - (n_K - 1) / 2) * width
        bars = ax.bar(x_pos + offset, vals, width, color=K_colors[ki % len(K_colors)],
                       label=K_labels[ki], edgecolor="white", linewidth=0.3)
        if K == 10:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                         ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([_xtick(e) for e in val_envs], rotation=0, fontsize=8)
    ax.set_xlabel("(trn = ckpt train, in/out = env type)", fontsize=9, style="italic")
    ax.set_ylabel("loc RMSE (m)", fontsize=11)
    ax.set_title(f"VAL — Tier 1 K-shot adaptation curve", fontsize=11)
    ax.legend(fontsize=8, ncol=n_K, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, ymax)

    ckpt_name = Path(args.ckpt).parent.name + "/" + Path(args.ckpt).name
    fig.suptitle(
        f"Per-env Tier 1 K-shot adaptation curve (K={','.join(map(str, Ks))}) on 20 Sionna scenes\n"
        f"ckpt: {ckpt_name}  |  TRAIN={train_scenes}  VAL={sorted(set(ck.get('val_scenes') or []) - train_scenes)}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure → {args.out_fig}")

    # Summary
    print()
    print("=== Per-K summary across all envs ===")
    for K in Ks:
        all_vals = [results[e][K] for e in results]
        train_vals = [results[e][K] for e in train_envs]
        val_vals = [results[e][K] for e in val_envs]
        print(f"  K={K:>2}: TRAIN mean={np.mean(train_vals):.3f}m  VAL mean={np.mean(val_vals):.3f}m  "
              f"OVERALL mean={np.mean(all_vals):.3f}m  max={max(all_vals):.3f}m")


if __name__ == "__main__":
    main()
