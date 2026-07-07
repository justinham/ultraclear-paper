"""Tier 1 (B) — Per-NLOS-condition error breakdown.

For each (variant, val scene, NLOS bin), run K-shot adaptation
and report query MSE.

The paper's claim: the Tier-0 pretrained encoder should help
*more* on NLOS trials (where the model needs cross-anchor
context the most) than on pure-LOS trials. This script tests
that by binning windows by their per-trial NLOS status
(0 = all 4 anchors LOS, 1 = at least one NLOS).

The default val scenes are the heavy-NLOS pillars scenes
(outdoor_blockages_pillars, indoor_blockages_pillars) which
have 39-42% any-NLOS rate, so both bins have non-trivial
counts.

Usage:
    python3.11 -m scripts.viz_tier1_nlos_breakdown \\
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
from scripts.multi_scene_dataset import build_default_multi_scene, _per_trial_cond_id


def sample_task_by_cond(
    ds, env_id: int, n: int, seed: int, cond_target: int | None = None
):
    """Like sample_task_multi but optionally filter to a specific cond_id.

    If cond_target is None, returns a uniformly random sample.
    If cond_target is 0 or 1, returns only windows whose cond_id matches.
    """
    indices = [i for i, (s_idx, _, _) in enumerate(ds.windows)
                if ds.scene_info[s_idx].env_id == env_id]
    if cond_target is not None:
        # filter by cond_id at the middle timestep
        indices = [i for i in indices if int(ds[i][3]) == cond_target]
    if not indices:
        raise RuntimeError(f"no windows for env_id={env_id} cond={cond_target}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=min(n, len(indices)), replace=False)
    xs, ys = [], []
    for i in chosen:
        x, y, _, _ = ds[int(i)]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def few_shot_eval_by_cond(
    core_state, cfg, val_scenes, ds, K, n_trials, cond_target
):
    """Run K-shot adaptation per (scene, cond), average over n_trials.
    Returns dict[(scene_id, cond)] -> mean query MSE."""
    out = {}
    for eid in val_scenes:
        errs = []
        for trial in range(n_trials):
            try:
                sx, sy = sample_task_by_cond(ds, eid, K, cfg.seed + 7000 + trial,
                                              cond_target=cond_target)
                qx, qy = sample_task_by_cond(ds, eid, 20, cfg.seed + 8000 + trial,
                                              cond_target=cond_target)
            except RuntimeError:
                # not enough windows in this cond bin
                continue
            fresh = MAMLCore(cfg)
            fresh.load_state_dict(core_state)
            q_loss, _ = adapt_and_evaluate(
                fresh, sx, sy, qx, qy,
                inner_lr=cfg.maml_inner_lr,
                inner_steps=cfg.maml_inner_steps,
            )
            errs.append(float(q_loss.item()))
        if errs:
            out[(eid, cond_target)] = float(np.mean(errs))
            out[(eid, cond_target, "se")] = float(np.std(errs) / np.sqrt(len(errs)))
            out[(eid, cond_target, "n")] = len(errs)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_random", required=True)
    p.add_argument("--ckpt_tier0", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--val_scenes", type=int, nargs="+", default=[2, 3],
                   help="env_ids for evaluation; default 2,3 (heavy-NLOS pillars scenes)")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--n_trials", type=int, default=30)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt_random).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading multi-scene dataset...")
    ds, scene_info = build_default_multi_scene(n_channels=2, seq_len=32, stride=32)
    train_scenes = set([0, 1, 2, 3])
    norm_mean, norm_std = ds.compute_norm_stats(train_scenes)
    ds.norm_mean = norm_mean
    ds.norm_std = norm_std

    # check cond distribution in val scenes
    print(f"\ncond distribution in val scenes (env {args.val_scenes}):")
    cond_counts = {0: 0, 1: 0}
    for i in range(len(ds)):
        _, _, eid, c = ds[i]
        if int(eid) in args.val_scenes:
            cond_counts[int(c)] += 1
    print(f"  cond=0 (all LOS): {cond_counts[0]}")
    print(f"  cond=1 (any NLOS): {cond_counts[1]}")
    print()

    # load both checkpoints
    ckpt_r = torch.load(args.ckpt_random, weights_only=False)
    ckpt_t = torch.load(args.ckpt_tier0, weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt_r["cfg"].items()
                    if k in Config.__dataclass_fields__})

    # run per-cond eval
    res = {"random": {}, "tier0": {}}
    for variant, state in [("random", ckpt_r["model_state"]),
                            ("tier0", ckpt_t["model_state"])]:
        for cond in [0, 1]:
            r = few_shot_eval_by_cond(state, cfg, args.val_scenes, ds,
                                       K=args.K, n_trials=args.n_trials,
                                       cond_target=cond)
            for k, v in r.items():
                res[variant][k] = v
        # also run the un-binned eval
        r_all = few_shot_eval_by_cond(state, cfg, args.val_scenes, ds,
                                       K=args.K, n_trials=args.n_trials,
                                       cond_target=None)
        for k, v in r_all.items():
            res[variant][k] = v

    # print table
    print(f"K = {args.K}, n_trials = {args.n_trials}, val scenes = {args.val_scenes}\n")
    print("=" * 78)
    print("  Per-NLOS-condition query MSE")
    print("=" * 78)
    for cond_label, cond in [("all (any cond)", None), ("LOS only (cond=0)", 0),
                              ("NLOS (cond=1)", 1)]:
        print(f"\n  {cond_label}:")
        print(f"    {'scene':<8} {'random':>12} {'tier0':>12} {'reduction':>14} {'n':>6}")
        for eid in args.val_scenes:
            key = (eid, cond)
            if key in res["random"] and key in res["tier0"]:
                r = res["random"][key]
                t = res["tier0"][key]
                red = (r - t) / r * 100 if r > 0 else 0
                n = res["random"].get((eid, cond, "n"), "?")
                print(f"    env {eid:<3} {r:>12.3f} {t:>12.3f} {red:>+13.1f}% {n:>6}")

    # ----- bar chart -----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # 3 groups: all, LOS, NLOS.  2 variants per group.
    group_labels = ["all", "LOS\n(cond=0)", "NLOS\n(cond=1)"]
    conds = [None, 0, 1]
    width = 0.35
    x = np.arange(len(group_labels))
    rvals, tvals, rerrs, terrs = [], [], [], []
    for c in conds:
        # average across val scenes
        r_means = [res["random"].get((eid, c), np.nan) for eid in args.val_scenes]
        t_means = [res["tier0"].get((eid, c), np.nan) for eid in args.val_scenes]
        # filter NaN
        r_means = [v for v in r_means if not np.isnan(v)]
        t_means = [v for v in t_means if not np.isnan(v)]
        rvals.append(np.mean(r_means) if r_means else np.nan)
        tvals.append(np.mean(t_means) if t_means else np.nan)
        rerrs.append(np.std(r_means) / np.sqrt(len(r_means)) if len(r_means) > 1 else 0)
        terrs.append(np.std(t_means) / np.sqrt(len(t_means)) if len(t_means) > 1 else 0)
    ax.bar(x - width/2, rvals, width, yerr=rerrs, capsize=4,
            color="C0", alpha=0.85, label="random init")
    ax.bar(x + width/2, tvals, width, yerr=terrs, capsize=4,
            color="C1", alpha=0.85, label="Tier-0 pretrained")
    # value labels on top
    for xi, (rv, tv) in enumerate(zip(rvals, tvals)):
        if not np.isnan(rv):
            ax.text(xi - width/2, rv, f"{rv:.1f}", ha="center", va="bottom", fontsize=9)
        if not np.isnan(tv):
            ax.text(xi + width/2, tv, f"{tv:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(f"query MSE (m²)")
    ax.set_title(f"Per-NLOS-condition query MSE  (K={args.K}, val scenes {args.val_scenes})\n"
                  f"NLOS bin = trials where at least one of 4 anchors is NLOS")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = out_dir / "tier1_nlos_breakdown.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nsaved {out}")

    # save CSV
    csv_out = out_dir / "tier1_nlos_breakdown.csv"
    with open(csv_out, "w") as f:
        f.write("variant,val_scene,cond_label,query_mse,se,n\n")
        cond_labels = {None: "all", 0: "LOS", 1: "NLOS"}
        for variant in ["random", "tier0"]:
            for eid in args.val_scenes:
                for c, lbl in cond_labels.items():
                    mse = res[variant].get((eid, c), float("nan"))
                    se = res[variant].get((eid, c, "se"), float("nan"))
                    n = res[variant].get((eid, c, "n"), 0)
                    f.write(f"{variant},{eid},{lbl},{mse},{se},{n}\n")
    print(f"saved {csv_out}")


if __name__ == "__main__":
    main()
