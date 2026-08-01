"""Tier 2 reference eval (Sionna) — single model, env 0 + env 11.

Companion to `eval_tier1_reference.py`. Runs the Tier 2 DisentangledTrainer
(trained on Sionna with the 10/2 split from `train_tier2_sionna.py`) on:
  (a) env 0 = custom_room (TRAINED)
  (b) env 11 = outdoor_blockages_narrow (UNTRAINED)

Reports loc RMSE in meters (per-scene denormalization), at the BEST val
epoch + the FINAL epoch. Tier 2 is zero-shot (no K-shot needed).

Usage:
    python -m scripts.eval_tier2_sionna_reference
    python -m scripts.eval_tier2_sionna_reference --ckpt checkpoints/tier2_sionna_frozen_tier1/tier2_sionna.pt
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.config import Config
from scripts.multi_scene_dataset import build_default_multi_scene
from scripts.train_tier2_sionna import DisentangledTrainer


def denormalize_xy_per_scene(z_xy, sinfo):
    """Convert z-scored (x, y) back to meters using per-scene stats.
    Accepts both torch.Tensor and np.ndarray."""
    if sinfo.tag_xy_mean is None or sinfo.tag_xy_std is None:
        return z_xy
    if isinstance(z_xy, torch.Tensor):
        return z_xy * torch.from_numpy(sinfo.tag_xy_std).float().to(z_xy.device) \
                  + torch.from_numpy(sinfo.tag_xy_mean).float().to(z_xy.device)
    return z_xy * sinfo.tag_xy_std + sinfo.tag_xy_mean


def load_ckpt(ckpt_path: str, device: str = "cpu"):
    print(f"loading ckpt: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  saved cfg n_channels = {ck['cfg'].get('n_channels')}")
    print(f"  saved cfg n_envs (adversary) = {ck.get('n_envs')}")
    print(f"  saved train_scenes: {ck.get('train_scenes')}")
    print(f"  saved val_scenes: {ck.get('val_scenes')}")

    cfg = Config()
    from dataclasses import fields
    valid_keys = {f.name for f in fields(Config)}
    for k, v in ck["cfg"].items():
        if k in valid_keys:
            setattr(cfg, k, v)
    n_envs = ck.get("n_envs", len(set(ck.get("train_scenes", []))))
    # n_conds: read from cond_head shape in state_dict (most reliable),
    # fall back to 3 (synthetic default) or 2 (Sionna binary).
    n_conds = 3
    if "model_state" in ck:
        ms = ck["model_state"]
        for k, v in ms.items():
            if k.endswith("cond_head.weight"):
                n_conds = v.shape[0]
                break
    print(f"  using n_conds = {n_conds} (from cond_head shape)")
    model = DisentangledTrainer(cfg, n_envs=n_envs, n_conds=n_conds).to(device)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    if missing:
        print(f"  WARNING missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  WARNING unexpected keys (first 5): {unexpected[:5]}")
    model.eval()
    return model, cfg, ck


def run_eval(
    env_id: int,
    model: DisentangledTrainer,
    ds,
    sinfo,
    device: str = "cpu",
) -> dict:
    """Eval the model on env_id. Per-scene denormalization to meters."""
    indices = [i for i, (s_idx, _, _) in enumerate(ds.windows)
                if ds.scene_info[s_idx].env_id == env_id]
    if not indices:
        raise RuntimeError(f"no windows for env_id={env_id}")
    n_windows = len(indices)
    Xs, ys = [], []
    for i in indices:
        X, y, _, _, _ = ds[i]
        Xs.append(X); ys.append(y)
    X = torch.stack(Xs).to(device)
    y = torch.stack(ys).to(device)
    with torch.no_grad():
        _h, _c, _adv, loc, _nlos, _cond = model(X)
    # loc: (B, T, 2); y: (B, T, 2). Sionna y is constant across T, so
    # we denormalize at the middle timestep and use that.
    T = X.size(1)
    y_mid = y[:, T // 2, :]  # (B, 2) z-scored
    if loc.dim() == 3:
        loc_z = loc[:, T // 2, :]
    else:
        loc_z = loc
    # Per-sample denormalization
    loc_m = torch.zeros_like(loc_z)
    y_m = torch.zeros_like(y_mid)
    for i in range(X.size(0)):
        loc_m[i] = denormalize_xy_per_scene(loc_z[i], sinfo)
        y_m[i] = denormalize_xy_per_scene(y_mid[i], sinfo)
    errs = torch.linalg.norm(loc_m - y_m, dim=-1).cpu().numpy()  # (B,)
    loc_rmse = float(np.sqrt(np.mean(errs ** 2)))
    loc_mae = float(np.mean(errs))
    loc_median = float(np.median(errs))
    loc_p90 = float(np.percentile(errs, 90))
    return {
        "env_id": env_id,
        "n_windows": n_windows,
        "loc_rmse_m": loc_rmse,
        "loc_mae_m": loc_mae,
        "loc_median_m": loc_median,
        "loc_p90_m": loc_p90,
        "loc_errs": errs.tolist(),
        "pred_xy_m": loc_m.cpu().numpy().tolist(),
        "gt_xy_m": y_m.cpu().numpy().tolist(),
    }


def make_figure_one_env(env_id: int, r: dict, train_scenes: list, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    is_trained = env_id in train_scenes

    # Panel 1: bar of summary stats
    ax = axes[0]
    errs = np.asarray(r["loc_errs"])
    metrics = [("RMSE", r["loc_rmse_m"], "C0"),
               ("MAE", r["loc_mae_m"], "C1"),
               ("median", r["loc_median_m"], "C2"),
               ("p90", r["loc_p90_m"], "C3")]
    names = [m[0] for m in metrics]
    vals = [m[1] for m in metrics]
    colors = [m[2] for m in metrics]
    bars = ax.bar(names, vals, color=colors, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}m",
                 ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("loc error (m)")
    role = "TRAINED" if is_trained else "untrained"
    ax.set_title(f"env {env_id} ({role}, zero-shot, n_windows={r['n_windows']})\n"
                  f"loc error summary")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: CDF
    ax = axes[1]
    errs_sorted = np.sort(errs)
    cdf = np.arange(1, len(errs_sorted) + 1) / len(errs_sorted)
    ax.plot(errs_sorted, cdf, color="C0", lw=2, label=f"env {env_id} (n={len(errs_sorted)})")
    for thresh in [0.5, 1.0, 2.0, 5.0]:
        frac = float((errs_sorted <= thresh).mean())
        ax.axvline(thresh, color="grey", linestyle=":", alpha=0.5)
        ax.text(thresh, 0.02, f"{thresh:.1f}m\n{frac*100:.0f}%",
                 ha="center", va="bottom", fontsize=7, color="grey")
    ax.set_xlabel("loc error (m)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Per-window loc error CDF\n"
                  f"mean = {r['loc_mae_m']:.2f}m, median = {r['loc_median_m']:.2f}m, "
                  f"p90 = {r['loc_p90_m']:.2f}m")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Panel 3: scatter
    ax = axes[2]
    gt = np.asarray(r["gt_xy_m"])
    pr = np.asarray(r["pred_xy_m"])
    idx = np.random.default_rng(0).choice(len(gt), size=min(800, len(gt)), replace=False)
    ax.scatter(gt[idx, 0], gt[idx, 1], s=12, alpha=0.5, color="C0", label="GT", edgecolors="none")
    ax.scatter(pr[idx, 0], pr[idx, 1], s=12, alpha=0.5, color="C3", marker="x", label="pred")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Pred vs GT (● GT, × pred)\nn = {len(idx)} (subsampled from {len(gt)})")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right")

    fig.suptitle(f"Tier 2 reference eval (Sionna, zero-shot) — env {env_id} ({role})",
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure → {out_path}")


def print_summary(results: dict, ckpt_path: str, train_scenes: list):
    print(f"\n{'='*72}")
    print(f"Tier 2 reference eval (Sionna) — {ckpt_path}")
    print(f"{'='*72}")
    print(f"{'env':<6} {'role':<10} {'n_q':>6} {'RMSE':>8} {'MAE':>8} "
          f"{'median':>8} {'p90':>8}")
    print("-" * 72)
    for env_id, r in results.items():
        role = "TRAINED" if env_id in train_scenes else "untrained"
        print(f"{env_id:<6} {role:<10} {r['n_windows']:>6} "
              f"{r['loc_rmse_m']:>7.3f}m {r['loc_mae_m']:>7.3f}m "
              f"{r['loc_median_m']:>7.3f}m {r['loc_p90_m']:>7.3f}m")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/tier2_sionna_frozen_tier1/tier2_sionna.pt")
    p.add_argument("--train_env", type=int, default=0)
    p.add_argument("--val_env", type=int, nargs="+", default=[11])
    p.add_argument("--out_csv", default=None)
    p.add_argument("--out_fig", default=None,
                    help="output figure template with {env} placeholder")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model, cfg, ck = load_ckpt(args.ckpt, device=args.device)
    train_scenes = ck.get("train_scenes") or list(range(10))

    # Build Sionna dataset
    print(f"\nloading Sionna data...")
    ds, scene_info = build_default_multi_scene(
        n_channels=cfg.n_channels, seq_len=cfg.seq_len, stride=32,
        all_scenes=True,
    )
    norm_mean, norm_std = ds.compute_norm_stats(set(train_scenes))
    ds.norm_mean = norm_mean
    ds.norm_std = norm_std
    print(f"  {len(ds)} windows across {len(scene_info)} scenes")

    env_ids = [args.train_env] + list(args.val_env)
    results = {}
    for env_id in env_ids:
        is_train = env_id in train_scenes
        sinfo = next(s for s in scene_info if s.env_id == env_id)
        print(f"\n--- env {env_id} ({'TRAINED' if is_train else 'UNTRAINED'}) "
              f"n_windows={sum(1 for s_idx, _, _ in ds.windows if ds.scene_info[s_idx].env_id == env_id)} ---")
        r = run_eval(env_id, model, ds, sinfo, device=args.device)
        results[env_id] = r
        print(f"  loc RMSE = {r['loc_rmse_m']:.3f} m, MAE = {r['loc_mae_m']:.3f} m, "
              f"median = {r['loc_median_m']:.3f} m, p90 = {r['loc_p90_m']:.3f} m "
              f"(n={r['n_windows']})")

    print_summary(results, args.ckpt, train_scenes)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w") as f:
            f.write("env_id,is_trained,n_windows,loc_rmse_m,loc_mae_m,"
                    "loc_median_m,loc_p90_m\n")
            for env_id, r in results.items():
                is_train = env_id in train_scenes
                f.write(f"{env_id},{int(is_train)},{r['n_windows']},"
                        f"{r['loc_rmse_m']:.4f},{r['loc_mae_m']:.4f},"
                        f"{r['loc_median_m']:.4f},{r['loc_p90_m']:.4f}\n")
        print(f"\nsaved CSV → {out_csv}")

    if args.out_fig:
        out_fig = Path(args.out_fig)
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        if "{env}" in args.out_fig:
            for env_id, r in results.items():
                per_env_path = str(out_fig).format(env=env_id)
                make_figure_one_env(env_id, r, train_scenes, per_env_path)
        else:
            raise NotImplementedError("combined figure for Tier 2 Sionna not implemented; "
                                       "use --out_fig with {env} placeholder")


if __name__ == "__main__":
    main()
