"""Tier 1 (F) — Inner-loop adaptation curves.

For each held-out val scene and each K (number of support shots),
run K-shot adaptation on the model and record the query MSE at
each intermediate inner-loop step (step 0 = no adaptation,
step 1, 2, ..., 5 = full adaptation).

The figure plots query MSE vs inner-loop step for the two
variants (random init vs Tier-0 pretrained). The Tier-0
pretrained curve should:
  - start LOWER at step 0 (the SSL features are a better init)
  - drop faster / converge in fewer steps

Usage:
    python3.11 -m scripts.viz_tier1_adaptation_curve \\
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
from ultraclear.tier1_maml import MAMLCore
from scripts.train_tier1_sionna import sample_task_multi
from scripts.multi_scene_dataset import build_default_multi_scene


def adapt_stepwise(
    core: MAMLCore,
    sx: torch.Tensor, sy: torch.Tensor,
    qx: torch.Tensor, qy: torch.Tensor,
    inner_lr: float,
    max_steps: int,
) -> list[float]:
    """Run inner-loop adaptation step-by-step and return query MSE after
    each step (including step 0 = no adaptation, before any gradient).
    """
    params = dict(core.named_parameters())
    names = list(params.keys())
    fast = {n: p.detach().clone().requires_grad_(True) for n, p in params.items()}

    losses = []
    # step 0: query MSE before any adaptation
    with torch.no_grad():
        q_pred = functional_call(core, fast, (qx,))
        losses.append(float(TF.mse_loss(q_pred, qy).item()))

    for step in range(1, max_steps + 1):
        pred = functional_call(core, fast, (sx,))
        loss = TF.mse_loss(pred, sy)
        grads = torch.autograd.grad(
            loss, list(fast.values()), create_graph=False, allow_unused=True,
        )
        for n, g in zip(names, grads):
            if g is None:
                continue
            fast[n] = fast[n] - inner_lr * g
        # query MSE after this step (no grad)
        with torch.no_grad():
            q_pred = functional_call(core, fast, (qx,))
            losses.append(float(TF.mse_loss(q_pred, qy).item()))
    return losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_random", required=True)
    p.add_argument("--ckpt_tier0", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--k_shots", type=int, nargs="+", default=[1, 5, 20])
    p.add_argument("--n_trials", type=int, default=20,
                   help="how many independent support/query draws per scene per K")
    p.add_argument("--max_steps", type=int, default=5)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt_random).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # rebuild the dataset (we need its per-scene env_ids for sampling)
    print("loading multi-scene dataset...")
    ds, scene_info = build_default_multi_scene(n_channels=2, seq_len=32, stride=32)
    val_scenes = [4, 5]   # held out per default
    train_scenes = set([0, 1, 2, 3])
    norm_mean, norm_std = ds.compute_norm_stats(train_scenes)
    ds.norm_mean = norm_mean
    ds.norm_std = norm_std

    # load both checkpoints
    ckpt_r = torch.load(args.ckpt_random, weights_only=False)
    ckpt_t = torch.load(args.ckpt_tier0, weights_only=False)
    cfg_r = Config(**{k: v for k, v in ckpt_r["cfg"].items()
                      if k in Config.__dataclass_fields__})
    cfg_t = Config(**{k: v for k, v in ckpt_t["cfg"].items()
                      if k in Config.__dataclass_fields__})
    assert cfg_r.n_channels == 2 and cfg_t.n_channels == 2

    # build a fresh model and load each ckpt's weights
    def build_with_state(state):
        m = MAMLCore(cfg_r)
        m.load_state_dict(state)
        return m

    core_r = build_with_state(ckpt_r["model_state"])
    core_t = build_with_state(ckpt_t["model_state"])
    core_r.eval()
    core_t.eval()
    print(f"loaded {Path(args.ckpt_random).name} and {Path(args.ckpt_tier0).name}")

    # collect stepwise loss curves: dict[(k, scene_id, variant)] -> list[float]
    # store ALL trial curves; we'll median-average for robustness to
    # FOMAML instability at low K.
    raw_curves: dict = {}
    rng = np.random.default_rng(cfg_r.seed)
    for k in args.k_shots:
        for eid in val_scenes:
            tier0_trials = []
            rand_trials = []
            for trial in range(args.n_trials):
                sx, sy = sample_task_multi(ds, eid, k, cfg_r.seed + 5000 + trial)
                qx, qy = sample_task_multi(ds, eid, 20, cfg_r.seed + 6000 + trial)
                fresh_t = build_with_state(ckpt_t["model_state"])
                fresh_t.eval()
                fresh_r = build_with_state(ckpt_r["model_state"])
                fresh_r.eval()
                t_losses = adapt_stepwise(
                    fresh_t, sx, sy, qx, qy,
                    inner_lr=cfg_t.maml_inner_lr,
                    max_steps=args.max_steps,
                )
                r_losses = adapt_stepwise(
                    fresh_r, sx, sy, qx, qy,
                    inner_lr=cfg_r.maml_inner_lr,
                    max_steps=args.max_steps,
                )
                tier0_trials.append(t_losses)
                rand_trials.append(r_losses)
            raw_curves[("tier0", k, eid)] = np.array(tier0_trials)  # (n_trials, n_steps+1)
            raw_curves[("random", k, eid)] = np.array(rand_trials)

    # median + IQR for robust aggregation across trials
    curves: dict = {}
    for (variant, k, eid), arr in raw_curves.items():
        curves[(variant, k, eid)] = np.median(arr, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        curves[(variant + "_iqr", k, eid)] = (q75 - q25) / 2  # half-IQR as error bar
        curves[(variant, k, eid)] = np.median(arr, axis=0)

    # ----- plot: one panel per K -----
    n_panels = len(args.k_shots)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.6),
                              sharey=True)
    if n_panels == 1:
        axes = [axes]

    steps = np.arange(0, args.max_steps + 1)
    for ax, k in zip(axes, args.k_shots):
        # median across val scenes (each scene's per-step median)
        rand_mean = np.mean([curves[("random", k, e)] for e in val_scenes], axis=0)
        rand_iqr = np.mean([curves[("random_iqr", k, e)] for e in val_scenes], axis=0)
        tier0_mean = np.mean([curves[("tier0", k, e)] for e in val_scenes], axis=0)
        tier0_iqr = np.mean([curves[("tier0_iqr", k, e)] for e in val_scenes], axis=0)
        # per-scene (faint) — show per-scene medians
        for e in val_scenes:
            ax.plot(steps, curves[("random", k, e)], "-", color="C0", alpha=0.3, lw=1)
            ax.plot(steps, curves[("tier0", k, e)], "-", color="C1", alpha=0.3, lw=1)
        # mean with half-IQR error bars
        ax.errorbar(steps, rand_mean, yerr=rand_iqr, marker="o", capsize=3,
                     lw=2, color="C0", label="random init")
        ax.errorbar(steps, tier0_mean, yerr=tier0_iqr, marker="s", capsize=3,
                     lw=2, color="C1", label="Tier-0 pretrained")
        ax.set_yscale("log")   # log y: handles K=1 spikes + shows K=20 clearly
        ax.set_xlabel("inner-loop step")
        ax.set_title(f"K = {k} shot{'s' if k > 1 else ''}")
        ax.grid(True, alpha=0.3, which="both")
        if k == args.k_shots[0]:
            ax.set_ylabel("query MSE (m², log)")
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Inner-loop adaptation curves on held-out scenes\n"
                  "(averaged over 2 val scenes × 20 independent support/query draws)",
                  fontsize=11)
    fig.tight_layout()
    out = out_dir / "tier1_adaptation_curves.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")

    # save CSV (per-trial curves, for downstream analysis)
    csv_out = out_dir / "tier1_adaptation_curves.csv"
    with open(csv_out, "w") as f:
        f.write("variant,K_shot,val_env,trial,inner_step,query_mse\n")
        for (variant, k, eid), arr in raw_curves.items():
            for trial in range(arr.shape[0]):
                for s in range(arr.shape[1]):
                    f.write(f"{variant},{k},{eid},{trial},{s},{arr[trial, s]:.4f}\n")
    print(f"saved {csv_out}")

    # ----- print a summary table -----
    print()
    print("=" * 70)
    print("  Inner-loop adaptation: query MSE (MEDIAN over val scenes × trials)")
    print("=" * 70)
    print(f"  {'K':>3}  {'variant':<18} " + "  ".join(f"step {s}" for s in range(args.max_steps + 1)))
    for k in args.k_shots:
        for variant in ["random", "tier0"]:
            mean = np.mean([curves[(variant, k, e)] for e in val_scenes], axis=0)
            label = "Tier-0 pretrained" if variant == "tier0" else "random init"
            print(f"  {k:>3}  {label:<18} " + "  ".join(f"{v:>7.2f}" for v in mean))


if __name__ == "__main__":
    main()
