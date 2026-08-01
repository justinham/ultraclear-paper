"""Tier 1 reference eval — single model, loc RMSE on trained + untrained envs.

This is the Tier 1 component of the "reference test on 3 models" plan.
Runs the Tier 1 MAML model with K-shot adaptation on:
  (a) one specific TRAINED environment (from ckpt['train_scenes'])
  (b) one or more UNTRAINED environments (from ckpt['val_scenes'])

For each env, reports:
  - Loc RMSE in meters (denormalized from z-scored xy)
  - Loc MAE in meters
  - Per-window loc error distribution (saved to CSV)

NLOS detection is NOT done here. The Tier 1 model has no NLOS head —
the 3-part loss (NLOS + loc + restore) only exists in the
downstream tier2 / supervised-baseline trainers. Adding the NLOS
head back to the FOMAML pipeline is a parked follow-up. When that's
done, this script will get a `--nlos` flag that extracts the per-
anchor NLOS logits and computes accuracy / F1.

Usage:
    python -m scripts.eval_tier1_reference
    python -m scripts.eval_tier1_reference --train_env 0 --val_env 16
    python -m scripts.eval_tier1_reference --val_env 11 --val_env 16 --K 10 --n_trials 50
"""
from __future__ import annotations
import argparse
import json
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


def denormalize_xy(z_xy: np.ndarray, scene_info) -> np.ndarray:
    """Convert z-scored (x, y) back to meters using the per-scene stats."""
    if scene_info.tag_xy_mean is None or scene_info.tag_xy_std is None:
        return z_xy
    return z_xy * scene_info.tag_xy_std + scene_info.tag_xy_mean


def run_eval(
    env_id: int,
    ds,
    core_state: dict,
    cfg: Config,
    device: str = "cpu",
    K: int = 10,
    n_trials: int = 30,
) -> dict:
    """Run K-shot adaptation on env_id, n_trials times, return aggregate metrics.

    Each trial:
      1. Sample K support + q_query windows from env_id
      2. Build a fresh MAMLCore from the saved weights
      3. Run MAML inner loop (K inner steps on support)
      4. Evaluate query loss (loc MSE in z-scored units) and per-window
         predicted (x, y) in z-scored units
      5. Denormalize predicted (x, y) and query label (x, y) to meters
         using the per-scene tag_xy_mean / std
      6. Record per-window loc error in meters
    """
    info_idx = next(
        (i for i, s in enumerate(ds.scene_info) if s.env_id == env_id), None
    )
    if info_idx is None:
        raise RuntimeError(f"env_id {env_id} not in dataset")
    sinfo = ds.scene_info[info_idx]

    loc_errs = []
    pred_xy_per_trial = []  # for the scatter panel
    gt_xy_per_trial = []
    for trial in range(n_trials):
        try:
            sx, sy = sample_task_multi(ds, env_id, K,
                                        cfg.seed + 5000 + trial)
            qx, qy = sample_task_multi(ds, env_id, cfg.q_query,
                                        cfg.seed + 6000 + trial)
        except RuntimeError as e:
            print(f"  [env {env_id} trial {trial}] skip: {e}")
            continue
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)
        fresh = MAMLCore(cfg).to(device)
        fresh.load_state_dict(core_state)
        q_loss, info = adapt_and_evaluate(
            fresh, sx, sy, qx, qy,
            inner_lr=cfg.maml_inner_lr,
            inner_steps=cfg.maml_inner_steps,
        )
        q_pred_z = info["q_pred"].cpu().numpy()      # (B, 2) z-scored
        T_q = qx.size(1)
        q_y_mid_z = qy[:, T_q // 2, :].cpu().numpy()  # (B, 2) z-scored
        q_pred_m = denormalize_xy(q_pred_z, sinfo)
        q_y_mid_m = denormalize_xy(q_y_mid_z, sinfo)
        loc_errs.append(np.linalg.norm(q_pred_m - q_y_mid_m, axis=-1))
        pred_xy_per_trial.append(q_pred_m)
        gt_xy_per_trial.append(q_y_mid_m)

    loc_errs_all = np.concatenate(loc_errs) if loc_errs else np.array([])
    if len(loc_errs_all) == 0:
        raise RuntimeError(f"no query windows for env {env_id}")
    loc_rmse = float(np.sqrt(np.mean(loc_errs_all ** 2)))
    loc_mae = float(np.mean(loc_errs_all))
    loc_median = float(np.median(loc_errs_all))
    loc_p90 = float(np.percentile(loc_errs_all, 90))
    return {
        "env_id": env_id,
        "K": K,
        "n_trials": n_trials,
        "n_query_total": int(len(loc_errs_all)),
        "loc_rmse_m": loc_rmse,
        "loc_mae_m": loc_mae,
        "loc_median_m": loc_median,
        "loc_p90_m": loc_p90,
        "loc_errs": loc_errs_all.tolist(),
        "pred_xy_m": np.concatenate(pred_xy_per_trial, axis=0).tolist(),
        "gt_xy_m": np.concatenate(gt_xy_per_trial, axis=0).tolist(),
    }


def make_figure_one_env(
    env_id: int,
    r: dict,
    train_scenes: set,
    out_path: str,
):
    """3-panel figure for a single env: bar summary, error CDF, scatter.

    For the bar panel, when only one env is shown, we still want to
    compare against a reference: so we always show 0.40 m and 0.33 m
    reference markers if the env is env 0 or env 11 (the two we ran).
    For now, just show the env's own RMSE with a per-cond breakdown if
    possible.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    is_trained = env_id in train_scenes

    # Panel 1: bar of loc RMSE for this env, with per-stat breakdown
    ax = axes[0]
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
    ax.set_title(f"env {env_id} ({role}, K={r['K']}, n_q={r['n_query_total']})\n"
                  f"loc error summary statistics")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: per-window loc error CDF
    ax = axes[1]
    errs = np.sort(np.asarray(r["loc_errs"]))
    cdf = np.arange(1, len(errs) + 1) / len(errs)
    ax.plot(errs, cdf, color="C0", lw=2,
             label=f"env {env_id} (n={len(errs)})")
    # Reference markers at common thresholds
    for thresh in [0.5, 1.0, 2.0, 5.0]:
        frac = float((errs <= thresh).mean())
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

    # Panel 3: scatter pred vs GT
    ax = axes[2]
    gt = np.asarray(r["gt_xy_m"])
    pr = np.asarray(r["pred_xy_m"])
    idx = np.random.default_rng(0).choice(len(gt),
                                             size=min(800, len(gt)),
                                             replace=False)
    ax.scatter(gt[idx, 0], gt[idx, 1], s=12, alpha=0.5, color="C0",
                label="GT", edgecolors="none")
    ax.scatter(pr[idx, 0], pr[idx, 1], s=12, alpha=0.5, color="C3",
                marker="x", label="pred")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Predicted vs GT tag positions (● GT, × pred)\n"
                  f"n = {len(idx)} (subsampled from {len(gt)})")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right")

    fig.suptitle(f"Tier 1 reference eval — env {env_id} ({role})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure → {out_path}")


def make_figure(results: dict, out_path: str, train_scenes: set):
    """Legacy combined 3-env figure. Kept for backward compat. Use
    make_figure_one_env() for the per-env figures.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    env_ids = list(results.keys())
    # Panel 1: bar of loc RMSE per env, color-coded trained vs untrained
    ax = axes[0]
    rmses = [results[e]["loc_rmse_m"] for e in env_ids]
    colors = ["C2" if e in train_scenes else "C1" for e in env_ids]
    bars = ax.bar([f"env{e}" for e in env_ids], rmses, color=colors, edgecolor="white")
    for b, r in zip(bars, rmses):
        ax.text(b.get_x() + b.get_width() / 2, r, f"{r:.2f}m",
                 ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("loc RMSE (m)")
    ax.set_title(f"Loc RMSE per env (K={results[env_ids[0]]['K']})\n"
                  f"green = trained, orange = untrained")
    ax.grid(True, alpha=0.3, axis="y")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    # Panel 2: per-window loc error distribution (CDF)
    ax = axes[1]
    for e in env_ids:
        errs = np.sort(np.asarray(results[e]["loc_errs"]))
        cdf = np.arange(1, len(errs) + 1) / len(errs)
        ax.plot(errs, cdf, label=f"env{e} (μ={results[e]['loc_mae_m']:.2f}m)")
    ax.set_xlabel("loc error (m)")
    ax.set_ylabel("CDF")
    ax.set_title("Per-window loc error CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 3: scatter pred vs GT
    ax = axes[2]
    cmap_colors = ["C2", "C1", "C3", "C0", "C4"]
    for i, e in enumerate(env_ids):
        gt = np.asarray(results[e]["gt_xy_m"])
        pr = np.asarray(results[e]["pred_xy_m"])
        idx = np.random.default_rng(0).choice(len(gt),
                                                 size=min(400, len(gt)),
                                                 replace=False)
        ax.scatter(gt[idx, 0], gt[idx, 1], s=8, alpha=0.4,
                    color=cmap_colors[i % 5], label=f"env{e} GT")
        ax.scatter(pr[idx, 0], pr[idx, 1], s=8, alpha=0.4,
                    color=cmap_colors[i % 5], marker="x", label=f"env{e} pred")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Predicted vs GT (● GT, × pred)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, ncol=2, loc="upper right")

    fig.suptitle("Tier 1 reference eval — K-shot localization on trained + untrained envs",
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure → {out_path}")


def print_summary(results: dict, ckpt_path: str, train_env_id: int, train_scenes: set):
    print(f"\n{'='*72}")
    print(f"Tier 1 reference eval — {ckpt_path}")
    print(f"{'='*72}")
    print(f"{'env':<6} {'role':<10} {'K':>3} {'n_q':>6} {'RMSE':>8} {'MAE':>8} "
          f"{'median':>8} {'p90':>8}")
    print("-" * 72)
    for env_id, r in results.items():
        # Role reflects whether the env was actually in the MAML train_scenes,
        # not just whether the user passed --train_env. This way --val_env 1
        # still shows as TRAINED if env 1 was in train_scenes.
        role = "TRAINED" if env_id in train_scenes else "untrained"
        if env_id == train_env_id:
            role = f"{role}*"  # starred = explicitly requested as train_env
        print(f"{env_id:<6} {role:<10} {r['K']:>3} {r['n_query_total']:>6} "
              f"{r['loc_rmse_m']:>7.3f}m {r['loc_mae_m']:>7.3f}m "
              f"{r['loc_median_m']:>7.3f}m {r['loc_p90_m']:>7.3f}m")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/tier1_sionna_best.pt")
    p.add_argument("--train_env", type=int, default=0,
                    help="one trained env (default: 0 = empty_open, 0% NLOS)")
    p.add_argument("--val_env", type=int, nargs="+", default=[11, 16],
                    help="one or more untrained envs (default: 11 best + 16 worst)")
    p.add_argument("--K", type=int, default=10, help="few-shot K")
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--all_scenes", action="store_true",
                    help="load all 21 Sionna scenes (default: only env 0-5)")
    p.add_argument("--out_csv", default=None)
    p.add_argument("--out_fig", default=None,
                    help="output figure template. If the path contains '{env}', "
                         "one figure per env is written (e.g. figures/tier1_env{env}.png). "
                         "Otherwise, one combined figure is written (legacy).")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    # Load ckpt + cfg
    print(f"loading ckpt: {args.ckpt}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items()
                    if k in Config.__dataclass_fields__})
    print(f"  cfg: d_model={cfg.d_model}, n_channels={cfg.n_channels}, "
           f"seq_len={cfg.seq_len}, n_anchors={cfg.n_anchors}")
    print(f"  train_scenes: {ck.get('train_scenes')}")
    print(f"  val_scenes:   {ck.get('val_scenes')}")

    # Build dataset
    print("\nloading multi-scene dataset...")
    ds, scene_info = build_default_multi_scene(n_channels=cfg.n_channels,
                                                 seq_len=cfg.seq_len,
                                                 stride=32,
                                                 all_scenes=args.all_scenes)
    train_scenes = set(ck.get("train_scenes") or [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    norm_mean, norm_std = ds.compute_norm_stats(train_scenes)
    ds.norm_mean = norm_mean
    ds.norm_std = norm_std
    print(f"  {len(ds)} windows across {len(scene_info)} scenes")

    env_ids = [args.train_env] + list(args.val_env)
    print(f"\n  Per-env details:")
    for s in scene_info:
        if s.env_id in env_ids:
            nlos_str = f"{s.nlos_rate*100:.0f}% any-NLOS" if s.nlos_rate else "0% NLOS"
            n_w = sum(1 for s_idx, _, _ in ds.windows
                      if ds.scene_info[s_idx].env_id == s.env_id)
            print(f"    env {s.env_id:2d} ({s.name}): {nlos_str}, n_windows={n_w}")

    # Run per-env eval
    results = {}
    for env_id in env_ids:
        is_train = env_id in train_scenes
        print(f"\n--- env {env_id} ({'TRAINED' if is_train else 'UNTRAINED'}) "
              f"K={args.K} n_trials={args.n_trials} ---")
        r = run_eval(env_id, ds, ck["model_state"], cfg,
                      device=args.device,
                      K=args.K,
                      n_trials=args.n_trials)
        results[env_id] = r
        print(f"  loc RMSE = {r['loc_rmse_m']:.3f} m, MAE = {r['loc_mae_m']:.3f} m, "
              f"median = {r['loc_median_m']:.3f} m, p90 = {r['loc_p90_m']:.3f} m "
              f"(n_query={r['n_query_total']})")

    print_summary(results, args.ckpt, args.train_env, train_scenes)

    # Save CSV
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w") as f:
            f.write("env_id,is_trained,K,n_query,loc_rmse_m,loc_mae_m,"
                    "loc_median_m,loc_p90_m\n")
            for env_id, r in results.items():
                is_train = env_id in train_scenes
                f.write(f"{env_id},{int(is_train)},{r['K']},{r['n_query_total']},"
                        f"{r['loc_rmse_m']:.4f},{r['loc_mae_m']:.4f},"
                        f"{r['loc_median_m']:.4f},{r['loc_p90_m']:.4f}\n")
        print(f"\nsaved CSV → {out_csv}")

    if args.out_fig:
        out_fig = Path(args.out_fig)
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        if "{env}" in args.out_fig:
            # one figure per env
            for env_id, r in results.items():
                per_env_path = str(out_fig).format(env=env_id)
                make_figure_one_env(env_id, r, train_scenes, per_env_path)
        else:
            # combined (legacy)
            make_figure(results, str(out_fig), train_scenes)


if __name__ == "__main__":
    main()
