"""Non-destructive FOMAML run that logs per-episode INNER-loop losses.

This is a thin wrapper around `scripts.train_tier1_sionna` that:

  1. Saves the trained model to a caller-supplied `--out` path
     (NEVER overwriting `checkpoints/tier1_sionna.pt` /
     `checkpoints/tier1_mixed.pt` / any existing ckpt).
  2. Adds per-episode INNER-loop logging:
       - pre_inner_q_loss  : query loss BEFORE any inner adaptation
       - inner_step_{i}_support : support loss at each inner step
       - post_inner_q_loss : query loss AFTER all inner steps
     Each goes into the history CSV with a new `phase` value.
  3. Limits the iteration count to a short test (default 20 episodes).

Use this when you want to see the per-task inner-loop dynamics
of FOMAML without disturbing the trained models in checkpoints/.

Usage:
    # short test, all default ckpts
    python3.11 -m scripts.train_tier1_sionna_inner \\
        --n_episodes 20 \\
        --out checkpoints/_tmp_tier1_inner_test

    # inherit Tier 0 encoder transfer + save to a different path
    python3.11 -m scripts.train_tier1_sionna_inner \\
        --tier0_ckpt checkpoints/tier0_multiscene_f2_3to1_100ep.pt \\
        --n_episodes 20 \\
        --out_history figures/tier1_inner_test_history.csv \\
        --out checkpoints/_tmp_tier1_inner_test_t0
"""
from __future__ import annotations
import argparse
import time
import csv as _csv
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as TF
from torch.func import functional_call

from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer
from ultraclear.tier1_maml import MAMLCore, adapt_and_evaluate

# Reuse the multi-scene dataset + sampling helpers from the
# canonical trainer.
from scripts.train_tier1_sionna import (
    make_core, sample_task_multi, few_shot_eval,
)
from scripts.multi_scene_dataset import build_default_multi_scene


def adapt_with_inner_log(
    core: MAMLCore,
    sx: torch.Tensor, sy: torch.Tensor,
    qx: torch.Tensor, qy: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Inner loop that ALSO returns the per-step support loss and the
    pre-adaptation query loss. Same first-order mechanics as
    `adapt_and_evaluate` — see comments there for the FOMAML trick.

    Returns
    -------
    q_loss_post : torch.Tensor
        Query loss after all `inner_steps` of inner adaptation
        (this is what FOMAML backprops through).
    info : dict with keys
        "q_pred"        : post-adaptation query prediction (Tensor)
        "pre_q_loss"    : query loss BEFORE inner adaptation (Tensor, float)
        "inner_support_{i}" for i in 0..inner_steps : support loss at
                          each inner step, recorded AFTER that step's
                          gradient update (so step 0 = loss after the
                          first gradient step).
    """
    params = dict(core.named_parameters())
    names = list(params.keys())
    fast = {n: p.clone() for n, p in params.items()}

    info: Dict[str, torch.Tensor] = {}

    # Pre-adaptation query loss (no inner steps taken yet).
    q_pred_pre = functional_call(core, fast, (qx,))
    T_q = qx.size(1)
    q_y_mid = qy[:, T_q // 2, :]
    pre_q_loss = TF.mse_loss(q_pred_pre, q_y_mid)
    info["pre_q_loss"] = pre_q_loss.detach()

    for step in range(inner_steps):
        s_pred = functional_call(core, fast, (sx,))
        T_s = sx.size(1)
        s_y_mid = sy[:, T_s // 2, :]
        s_loss = TF.mse_loss(s_pred, s_y_mid)
        # FOMAML: keep create_graph=True so the first-order chain
        # from fast -> original params survives, but allow_unused
        # because some params (e.g. unused layer biases) may not
        # appear in this forward pass.
        grads = torch.autograd.grad(
            s_loss, list(fast.values()),
            create_graph=True, allow_unused=True,
        )
        new_fast = {}
        for n, g in zip(names, grads):
            if g is None:
                # parameter wasn't used; keep it as-is
                new_fast[n] = fast[n]
                continue
            new_fast[n] = fast[n] - inner_lr * g
        fast = new_fast
        info[f"inner_support_{step}"] = s_loss.detach()

    q_pred_post = functional_call(core, fast, (qx,))
    q_loss_post = TF.mse_loss(q_pred_post, q_y_mid)
    info["q_pred"] = q_pred_post
    return q_loss_post, info


def fomaml_train_with_inner_log(
    core: MAMLCore,
    train_scene_env_ids: list[int],
    ds,
    cfg: Config,
    device: str,
    n_episodes: int,
    log_every: int = 5,
) -> List[Dict]:
    """FOMAML outer loop that ALSO captures per-task inner-loop losses
    for every env in the meta-batch (so the support-loss plot can
    show one line per task across training).

    Returns a list of dicts, one per episode, with keys:
        episode, outer_q_loss, log_env_id_{eid}, pre_q_loss_{eid},
        inner_support_{i}_{eid} for i in 0..inner_steps-1
    """
    opt = torch.optim.Adam(core.parameters(), lr=cfg.maml_outer_lr)
    rng = np.random.default_rng(cfg.seed)
    history: List[Dict] = []
    t0 = time.time()
    for ep in range(n_episodes):
        opt.zero_grad()
        env_sample = list(rng.choice(
            train_scene_env_ids,
            size=min(cfg.n_way, len(train_scene_env_ids)),
            replace=False,
        ))
        total_q = 0.0
        # We log the inner trajectory of EVERY env in the meta-batch
        # so the support-loss plot can show one curve per task.
        per_task_info: Dict[int, Dict[str, torch.Tensor]] = {}
        for idx, eid in enumerate(env_sample):
            sx, sy = sample_task_multi(ds, int(eid), cfg.k_shot, cfg.seed + ep)
            qx, qy = sample_task_multi(ds, int(eid), cfg.q_query, cfg.seed + ep + 1)
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)
            q_loss, info = adapt_with_inner_log(
                core, sx, sy, qx, qy,
                inner_lr=cfg.maml_inner_lr,
                inner_steps=cfg.maml_inner_steps,
            )
            per_task_info[int(eid)] = info
            total_q = total_q + q_loss
        total_q = total_q / max(1, len(env_sample))
        total_q.backward()
        opt.step()
        # Record: outer meta-loss + per-task (pre + each inner step).
        rec: Dict = {
            "episode": ep + 1,
            "outer_q_loss": float(total_q.item()),
            "sampled_envs": list(env_sample),
        }
        for eid, info in per_task_info.items():
            rec[f"pre_q_loss_e{eid}"] = float(info["pre_q_loss"].item())
            for k, v in info.items():
                if k.startswith("inner_support_"):
                    rec[f"{k}_e{eid}"] = float(v.item())
        history.append(rec)
        if (ep + 1) % log_every == 0 or ep == 0:
            # Print the mean per-task support loss at the final step
            # for a quick "are they all learning?" glance.
            finals = [
                rec.get(f"inner_support_{cfg.maml_inner_steps-1}_e{eid}")
                for eid in env_sample
                if f"inner_support_{cfg.maml_inner_steps-1}_e{eid}" in rec
            ]
            finals = [v for v in finals if v is not None]
            final_mean = float(np.mean(finals)) if finals else float("nan")
            print(f"  [maml+inner] ep {ep+1:3d}/{n_episodes}  "
                  f"outer_q = {total_q.item():.4f}  "
                  f"mean_task_support = {final_mean:.4f}  "
                  f"({time.time()-t0:.1f}s)")
    return history


def main():
    p = argparse.ArgumentParser(
        description="Non-destructive FOMAML test with inner-loss logging.",
    )
    p.add_argument("--tier0_ckpt", default=None)
    p.add_argument("--n_episodes", type=int, default=20,
                   help="Number of FOMAML outer-loop episodes (default 20 "
                        "for a short test).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_scenes", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                   help="Default 10 envs so we can plot one support-loss "
                        "line per task in the viz. Pass fewer for a "
                        "faster smoke test.")
    p.add_argument("--val_scenes", type=int, nargs="+", default=[4, 5])
    p.add_argument("--k_shots", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    p.add_argument("--n_eval_trials", type=int, default=10,
                   help="Reduced trial count for the few-shot eval at the end.")
    p.add_argument("--all_scenes", action="store_true",
                   help="(kept for compatibility — actually now always "
                        "loads all_scenes so the 10-env default works.)")
    # ----- temp / non-destructive output paths -----
    p.add_argument("--out", required=True,
                   help="Directory to save the trained model. WILL NOT "
                        "be created if it already exists unless "
                        "--overwrite is passed. The intent is that this "
                        "is a throwaway / temp dir.")
    p.add_argument("--overwrite", action="store_true",
                   help="If set, allows --out to be an existing directory.")
    p.add_argument("--out_history", default=None,
                   help="If set, save the per-episode inner-loss history "
                        "CSV here.")
    p.add_argument("--no_few_shot", action="store_true",
                   help="Skip the few-shot eval at the end (faster).")
    args = p.parse_args()

    # ----- non-destructive guard: refuse to overwrite a known good ckpt -----
    out_dir = Path(args.out)
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                f"refusing to overwrite existing dir: {out_dir}\n"
                f"pass --overwrite to allow, or pick a different --out."
            )
    # Belt-and-braces: if the resolved path matches a known canonical
    # ckpt, refuse unconditionally.
    canonical = {
        Path("checkpoints/tier1_sionna.pt"),
        Path("checkpoints/tier1_mixed.pt"),
        Path("checkpoints/tier1_sionna_best.pt"),
        Path("checkpoints/tier1_maml_sionna.pt"),
    }
    try:
        out_abs = out_dir.resolve()
        for c in canonical:
            try:
                c_abs = c.resolve()
            except FileNotFoundError:
                continue
            if out_abs == c_abs:
                raise SystemExit(
                    f"refusing to write to canonical checkpoint {c_abs}. "
                    f"Use a temp path like checkpoints/_tmp_tier1_inner."
                )
    except SystemExit:
        raise
    except Exception:
        pass  # resolution best-effort; canonical check is the real one

    cfg = Config()
    cfg.n_channels = 2
    cfg.device = args.device
    cfg.seed = args.seed

    print(f"device: {cfg.device}  seed: {cfg.seed}  "
          f"n_episodes: {args.n_episodes}")
    print(f"Tier 0 ckpt: {args.tier0_ckpt or '(none — random init)'}")
    print(f"train scenes: {args.train_scenes}  val scenes: {args.val_scenes}")
    print(f"output dir: {out_dir}  (will be created if missing)")
    print()

    print("building multi-scene dataset...")
    ds, scene_info = build_default_multi_scene(
        n_channels=2, seq_len=32, stride=32,
        all_scenes=True,
    )
    print(f"total windows: {len(ds)}")
    train_env_ids = set(args.train_scenes)
    norm_mean, norm_std = ds.compute_norm_stats(train_env_ids)
    print(f"per-channel norm (from train scenes):")
    for f in range(2):
        print(f"  ch {f}: mean={norm_mean[f]:.3f}  std={norm_std[f]:.3f}")
    print()

    core, info = make_core(cfg, args.tier0_ckpt, args.device)
    print(f"encoder init: {info['encoder_init']}")
    print(f"  total params: {sum(p.numel() for p in core.parameters()):,}")
    print()

    print("=" * 60)
    print("  Tier 1 FOMAML training (with inner-loss logging)")
    print("=" * 60)
    history = fomaml_train_with_inner_log(
        core, args.train_scenes, ds, cfg, args.device,
        n_episodes=args.n_episodes,
        log_every=max(1, args.n_episodes // 5),
    )

    # Few-shot eval at the end (small trial count, optional).
    if not args.no_few_shot:
        print()
        print("=" * 60)
        print("  Few-shot evaluation on held-out scenes (small)")
        print("=" * 60)
        fs_results = few_shot_eval(
            core, args.val_scenes, ds, cfg, args.device,
            ks=args.k_shots, n_trials=args.n_eval_trials,
        )
        print()
        print(f"  {'K':>4} | " +
              " | ".join(f"env {e}" for e in args.val_scenes))
        print("  " + "-" * 60)
        for k in args.k_shots:
            row = f"  {k:>4} |"
            for eid in args.val_scenes:
                row += f" {fs_results[(k, eid)]:>10.4f} |"
            print(row)
    else:
        fs_results = {}

    # Save the model to the temp dir.
    out_dir.mkdir(parents=True, exist_ok=True)
    fs_serial = {f"k={k}_env={e}": v for (k, e), v in fs_results.items()}
    torch.save({
        "model_state": core.state_dict(),
        "train_history": history,
        "fs_results": fs_serial,
        "cfg": cfg.__dict__,
        "info": info,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "train_scenes": args.train_scenes,
        "val_scenes": args.val_scenes,
        "n_episodes": args.n_episodes,
    }, out_dir / "tier1_sionna_inner.pt")
    print(f"\nsaved model → {out_dir / 'tier1_sionna_inner.pt'}")

    # Save the per-episode inner-loss history CSV.
    if args.out_history:
        out_hist = Path(args.out_history)
        out_hist.parent.mkdir(parents=True, exist_ok=True)
        if not history:
            print(f"(no history to save)")
        else:
            # Union of all keys: episode, outer_q_loss, sampled_envs,
            # then per-task pre_q_loss and inner_support_{i} for each
            # env in the meta-batch.
            keys = ["episode", "outer_q_loss", "sampled_envs"]
            for rec in history:
                for k in rec:
                    if k not in keys and k not in ("sampled_envs",):
                        keys.append(k)
            with open(out_hist, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(keys)
                for rec in history:
                    row = []
                    for k in keys:
                        v = rec.get(k, "")
                        if isinstance(v, list):
                            v = ";".join(str(x) for x in v)
                        row.append(v)
                    w.writerow(row)
            print(f"inner history → {out_hist}")
            # Also save a z-scored RMSE history compatible with the
            # existing tier1_maml_history.csv reader (so the
            # viz_ssl_vs_meta script can be reused if desired).
            rmse_hist = out_hist.with_name(
                out_hist.stem + "_zrmse_compat.csv"
            )
            with open(rmse_hist, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["episode", "phase", "k", "env_id", "value"])
                for rec in history:
                    rmse_z = rec["outer_q_loss"] ** 0.5
                    w.writerow([rec["episode"], "train_loc_err_z_rmse", "",
                                "", f"{rmse_z:.6f}"])
            print(f"z-rmse compat history → {rmse_hist}")


if __name__ == "__main__":
    main()
