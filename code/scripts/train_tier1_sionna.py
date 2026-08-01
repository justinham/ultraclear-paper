"""Tier 1 FOMAML on multi-scene Sionna data, with optional
encoder transfer from a Tier 0 SSL checkpoint.

This is the experiment that tests whether the SSL-pretrained
encoder (Tier 0) is actually a better initialization for FOMAML
than a random init.

Two modes:
  --tier0_ckpt PATH    : transfer encoder weights from this Tier 0
                         checkpoint before FOMAML
  (no flag)            : random init baseline (Tier 0 ablated)

Scenes are split:
  - train scenes: 4 (used for FOMAML outer-loop sampling)
  - val scenes: 2  (held out for few-shot evaluation)

The val evaluation runs FOMAML-style: for each held-out scene,
sample K support windows, do `inner_steps` of gradient descent,
then measure query MSE on Q windows. We sweep K = 1, 2, 5, 10,
20 to produce a few-shot curve.

Usage:
    # random init baseline
    python3.11 -m scripts.train_tier1_sionna --n_episodes 100

    # with Tier 0 transfer
    python3.11 -m scripts.train_tier1_sionna \\
        --tier0_ckpt checkpoints/tier0_empty_open_f2_25pct_3to1_100ep.pt \\
        --n_episodes 100
"""
from __future__ import annotations
import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as TF
from torch.func import functional_call

from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer
from ultraclear.tier1_maml import MAMLCore, adapt_and_evaluate
from scripts.multi_scene_dataset import (
    build_default_multi_scene, MultiSceneSionnaDataset,
    SceneInfo, ANCHOR_COLS,
)


def make_core(
    cfg: Config, tier0_ckpt: str | None, device: str
) -> tuple[MAMLCore, dict]:
    """Construct MAMLCore, optionally with encoder weights transferred
    from a Tier 0 SSL checkpoint.
    """
    core = MAMLCore(cfg).to(device)
    info = {"encoder_init": "random", "tier0_ckpt": None}
    if tier0_ckpt is not None:
        print(f"loading Tier 0 ckpt: {tier0_ckpt}")
        ckpt = torch.load(tier0_ckpt, weights_only=False)
        ckpt_cfg = Config(**{k: v for k, v in ckpt["cfg"].items()
                            if k in Config.__dataclass_fields__})
        # build a fresh SSLPretrainer with the ckpt's config (so the
        # encoder is shape-compatible), load weights, then pull the
        # encoder out.
        pretrainer = SSLPretrainer(ckpt_cfg)
        pretrainer.load_state_dict(ckpt["model_state"])
        # strict copy of encoder weights
        core.encoder.load_state_dict(pretrainer.encoder.state_dict())
        info["encoder_init"] = "tier0_pretrained"
        info["tier0_ckpt"] = tier0_ckpt
        print(f"  transferred encoder weights from {Path(tier0_ckpt).name}")
        # norm stats from Tier 0 (for re-using on the multi-scene data)
        info["t0_norm_mean"] = ckpt.get("norm_mean")
        info["t0_norm_std"] = ckpt.get("norm_std")
    return core, info


def sample_task_multi(
    ds: MultiSceneSionnaDataset, env_id: int, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick `n` random windows from one environment (env_id).

    Returns (X, y) where X has shape (n, T, M, F) and y has shape (n, 2).
    """
    indices = [i for i, (s_idx, _, _) in enumerate(ds.windows)
                if ds.scene_info[s_idx].env_id == env_id]
    if not indices:
        raise RuntimeError(f"no windows for env_id={env_id}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=min(n, len(indices)), replace=False)
    xs, ys = [], []
    for i in chosen:
        x, y, _, _, _ = ds[int(i)]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def fomaml_train(
    core: MAMLCore,
    train_scene_env_ids: list[int],
    val_scene_env_ids: list[int] | None,
    ds: MultiSceneSionnaDataset,
    cfg: Config,
    device: str,
    n_episodes: int,
    log_every: int = 5,
    val_every: int = 0,
    val_k: int = 10,
    val_trials: int = 10,
) -> list[float]:
    """First-order MAML outer loop over multiple scenes.

    Each episode: sample N envs, build K-shot support and Q-query per
    env, run inner-loop adaptation, accumulate query loss, backprop
    into core's initialization.

    If val_scene_env_ids is provided and val_every > 0, also runs a
    few-shot eval on a small held-out set every `val_every` episodes
    and returns (train_losses, val_checkpoints) where val_checkpoints
    is a list of (ep, k, env_id, loc_err_m) tuples for the per-scene
    loc error after adaptation.
    """
    opt = torch.optim.Adam(core.parameters(), lr=cfg.maml_outer_lr)
    rng = np.random.default_rng(cfg.seed)
    losses: list[float] = []
    val_checkpoints: list[dict] = []
    t0 = time.time()
    for ep in range(n_episodes):
        opt.zero_grad()
        env_sample = list(rng.choice(train_scene_env_ids,
                                      size=min(cfg.n_way, len(train_scene_env_ids)),
                                      replace=False))
        total_q = 0.0
        for eid in env_sample:
            sx, sy = sample_task_multi(ds, int(eid), cfg.k_shot, cfg.seed + ep)
            qx, qy = sample_task_multi(ds, int(eid), cfg.q_query, cfg.seed + ep + 1)
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)
            q_loss, _ = adapt_and_evaluate(
                core, sx, sy, qx, qy,
                inner_lr=cfg.maml_inner_lr,
                inner_steps=cfg.maml_inner_steps,
            )
            total_q = total_q + q_loss
        total_q = total_q / max(1, len(env_sample))
        total_q.backward()
        opt.step()
        losses.append(float(total_q.item()))
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"  [maml] ep {ep+1:3d}/{n_episodes}  "
                  f"query_loss = {total_q.item():.4f}  "
                  f"({time.time()-t0:.1f}s)")
        # Periodic val eval (on a fresh copy of core)
        if val_scene_env_ids and val_every > 0 and (ep + 1) % val_every == 0:
            core.eval()
            for eid in val_scene_env_ids:
                # Find the SceneInfo for this env_id so we can
                # denormalize pred/y back to meters for the
                # per-env loc err in meters.
                info_eid = next(
                    (i for i, s in enumerate(ds.scene_info) if s.env_id == int(eid)),
                    None,
                )
                if info_eid is None:
                    continue
                sinfo = ds.scene_info[info_eid]
                tag_xy_mean = torch.from_numpy(sinfo.tag_xy_mean).to(device) if sinfo.tag_xy_mean is not None else None
                tag_xy_std = torch.from_numpy(sinfo.tag_xy_std).to(device) if sinfo.tag_xy_std is not None else None
                trial_errs_m = []
                for trial in range(val_trials):
                    sx, sy = sample_task_multi(ds, int(eid), val_k,
                                                cfg.seed + 5000 + trial)
                    qx, qy = sample_task_multi(ds, int(eid), cfg.q_query,
                                                cfg.seed + 6000 + trial)
                    sx, sy = sx.to(device), sy.to(device)
                    qx, qy = qx.to(device), qy.to(device)
                    adapted = MAMLCore(cfg).to(device)
                    adapted.load_state_dict(core.state_dict())
                    q_loss, info = adapt_and_evaluate(
                        adapted, sx, sy, qx, qy,
                        inner_lr=cfg.maml_inner_lr,
                        inner_steps=cfg.maml_inner_steps,
                    )
                    # q_pred and q_y are both in z-scored xy units
                    # (MultiSceneSionnaDataset now z-scores tag_xy
                    # per scene). Denormalize to meters for the
                    # loc err in meters, matching the supervised
                    # baseline's per-cond loc err convention.
                    q_pred = info.get("q_pred")
                    if q_pred is not None and tag_xy_mean is not None:
                        T_q = qx.size(1)
                        q_y_mid = qy[:, T_q // 2, :]
                        q_pred_m = q_pred * tag_xy_std + tag_xy_mean
                        q_y_mid_m = q_y_mid * tag_xy_std + tag_xy_mean
                        per_sample_err = torch.linalg.norm(
                            q_pred_m - q_y_mid_m, dim=-1
                        ).cpu().numpy()
                        trial_errs_m.append(float(per_sample_err.mean()))
                if trial_errs_m:
                    mean_err_m = float(np.mean(trial_errs_m))
                else:
                    mean_err_m = float("nan")
                val_checkpoints.append({
                    "episode": ep + 1,
                    "k": val_k,
                    "env_id": int(eid),
                    "loc_err_m": mean_err_m,
                })
            print(f"    [val] ep {ep+1:3d}  K={val_k}  per-env loc err (m): " +
                  ", ".join(f"env{v['env_id']}={v['loc_err_m']:.3f}"
                            for v in val_checkpoints if v['episode'] == ep + 1))
            core.train()
    return losses, val_checkpoints


def few_shot_eval(
    core: MAMLCore,
    val_scene_env_ids: list[int],
    ds: MultiSceneSionnaDataset,
    cfg: Config,
    device: str,
    ks: list[int] = (1, 2, 5, 10, 20),
    n_trials: int = 30,
) -> dict:
    """Few-shot evaluation on held-out scenes.

    For each (k_shot, val_env_id) pair:
      - sample n_trials independent K-shot supports, adapt the model,
        measure query MSE on a fresh Q=20 query set
    Returns: dict[(k, env_id)] -> mean query MSE
    """
    results: dict = {}
    core.eval()
    for eid in val_scene_env_ids:
        for k in ks:
            errs = []
            for trial in range(n_trials):
                sx, sy = sample_task_multi(ds, int(eid), k, cfg.seed + 1000 + trial)
                qx, qy = sample_task_multi(ds, int(eid), cfg.q_query,
                                            cfg.seed + 2000 + trial)
                sx, sy = sx.to(device), sy.to(device)
                qx, qy = qx.to(device), qy.to(device)
                # make a fresh copy of the model for adaptation
                adapted = MAMLCore(cfg).to(device)
                adapted.load_state_dict(core.state_dict())
                q_loss, _ = adapt_and_evaluate(
                    adapted, sx, sy, qx, qy,
                    inner_lr=cfg.maml_inner_lr,
                    inner_steps=cfg.maml_inner_steps,
                )
                errs.append(float(q_loss.item()))
            results[(k, eid)] = float(np.mean(errs))
    core.train()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier0_ckpt", default=None,
                   help="path to Tier 0 SSL checkpoint to transfer from "
                        "(default: random init baseline)")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k_shots", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    p.add_argument("--out", default=None,
                   help="if set, save the trained model + losses + "
                        "few-shot results here")
    p.add_argument("--train_scenes", type=int, nargs="+", default=[0, 1, 2, 3],
                   help="env_ids for FOMAML training (default: first 4)")
    p.add_argument("--val_scenes", type=int, nargs="+", default=[4, 5],
                   help="env_ids for few-shot evaluation (default: last 2)")
    p.add_argument("--no_train", action="store_true",
                   help="skip FOMAML training; just run few-shot eval "
                        "(useful for debugging the eval pipeline)")
    p.add_argument("--all_scenes", action="store_true",
                   help="load all 20 grid2 scenes (default loads 6) so "
                        "FOMAML can sample from a diverse env set. "
                        "Required if you want to use train_scenes beyond 0-5.")
    p.add_argument("--val_every", type=int, default=10,
                   help="run few-shot val eval every N episodes during "
                        "FOMAML training (0 = disabled, only at end).")
    p.add_argument("--val_k", type=int, default=10,
                   help="K-shot for the in-training val eval.")
    p.add_argument("--val_trials", type=int, default=10,
                   help="number of K-shot trials per val scene during "
                        "in-training eval (kept small to limit overhead).")
    p.add_argument("--out_history", default=None,
                   help="if set, save per-episode FOMAML losses and "
                        "per-checkpoint val loc errors to this CSV.")
    p.add_argument("--n_eval_trials", type=int, default=30)
    p.add_argument("--n_way", type=int, default=None,
                   help="override cfg.n_way (envs sampled per FOMAML "
                        "episode). If unset, uses Config default.")
    args = p.parse_args()

    cfg = Config()
    cfg.n_channels = 2  # F=2
    cfg.device = args.device
    cfg.seed = args.seed
    if args.n_way is not None:
        cfg.n_way = args.n_way

    print(f"device: {cfg.device}  seed: {cfg.seed}  n_episodes: {args.n_episodes}  n_way: {cfg.n_way}")
    print(f"Tier 0 ckpt: {args.tier0_ckpt or '(none — random init)'}")
    print(f"train scenes: {args.train_scenes}  val scenes: {args.val_scenes}")
    print()
    print("building multi-scene dataset...")
    ds, scene_info = build_default_multi_scene(
        n_channels=2, seq_len=32, stride=32,
        all_scenes=args.all_scenes,
    )
    print(f"\ntotal windows: {len(ds)}")
    # compute per-channel norm stats from train scenes only
    train_env_ids = set(args.train_scenes)
    norm_mean, norm_std = ds.compute_norm_stats(train_env_ids)
    print(f"per-channel norm (from train scenes):")
    for f in range(2):
        print(f"  ch {f}: mean={norm_mean[f]:.3f}  std={norm_std[f]:.3f}")
    print()

    # build the model (with optional Tier 0 transfer)
    core, info = make_core(cfg, args.tier0_ckpt, args.device)
    print(f"encoder init: {info['encoder_init']}")
    print(f"  total params: {sum(p.numel() for p in core.parameters()):,}")
    print(f"  encoder params: {sum(p.numel() for p in core.encoder.parameters()):,}")
    print(f"  head params: {sum(p.numel() for p in core.head.parameters()):,}")
    print()

    # FOMAML training
    train_losses = []
    val_checkpoints = []
    if not args.no_train:
        print("=" * 60)
        print("  Tier 1 FOMAML training")
        print("=" * 60)
        train_losses, val_checkpoints = fomaml_train(
            core, args.train_scenes, args.val_scenes, ds, cfg, args.device,
            n_episodes=args.n_episodes,
            log_every=max(1, args.n_episodes // 10),
            val_every=args.val_every,
            val_k=args.val_k,
            val_trials=args.val_trials,
        )

    # few-shot eval
    print()
    print("=" * 60)
    print("  Few-shot evaluation on held-out scenes")
    print("=" * 60)
    fs_results = few_shot_eval(
        core, args.val_scenes, ds, cfg, args.device,
        ks=args.k_shots, n_trials=args.n_eval_trials,
    )

    # pretty-print
    print()
    print(f"  {'K':>4} | " + " | ".join(f"env {e}" for e in args.val_scenes)
          + f"  ({scene_info[args.val_scenes[0]].name.split('_')[0]})")
    print("  " + "-" * 60)
    for k in args.k_shots:
        row = f"  {k:>4} |"
        for eid in args.val_scenes:
            row += f" {fs_results[(k, eid)]:>10.4f} |"
        print(row)

    # save
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        # serialize few-shot results with string keys
        fs_serial = {f"k={k}_env={e}": v for (k, e), v in fs_results.items()}
        torch.save({
            "model_state": core.state_dict(),
            "train_losses": train_losses,
            "val_checkpoints": val_checkpoints,
            "fs_results": fs_serial,
            "cfg": cfg.__dict__,
            "info": info,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "train_scenes": args.train_scenes,
            "val_scenes": args.val_scenes,
            "n_episodes": args.n_episodes,
        }, out / "tier1_sionna.pt")
        print(f"\nsaved → {out / 'tier1_sionna.pt'}")

    # save per-episode history as a flat CSV (long format, easy to plot)
    if args.out_history:
        import csv as _csv
        with open(args.out_history, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["episode", "phase", "k", "env_id", "value"])
            for i, v in enumerate(train_losses):
                # q_loss is MSE in z-scored xy units (the
                # MultiSceneSionnaDataset z-scores tag_xy per scene
                # so the FOMAML sees consistent normalized xy
                # regardless of scene scale). sqrt(MSE) = RMSE in
                # z-scored units. For a per-episode loc err in
                # meters, multiply by the mean per-scene xy_std
                # across the envs in the meta-batch (approximate,
                # since the meta-batch composition varies per ep).
                rmse_z = v ** 0.5
                w.writerow([i + 1, "train_loc_err_z_rmse", "", "", f"{rmse_z:.6f}"])
            for ck in val_checkpoints:
                w.writerow([ck["episode"], "val_loc_err_m",
                            ck["k"], ck["env_id"], f"{ck['loc_err_m']:.6f}"])
        print(f"history → {args.out_history}")


if __name__ == "__main__":
    main()
