"""Joint Tier 1+2 — K-shot adaptation curves, matching the tier1_sionna style.

Compared figures from `figures/tier1_sionna/`:

  - `compare_tier1_few_shot.png` -- K-shot curve, log y-axis, per-env
                                   faded markers + median IQR on top.
  - `tier1_adaptation_curves.png` -- inner-loop step curves, one panel
                                     per K, per-trial faint lines +
                                     median IQR.

This script produces the joint equivalent:

  - `viz_joint_k_shot.png` -- 4-panel K-shot curve, one panel per held-
                              out val env, all 4 variants overlaid.
  - `viz_joint_step_curve.png` -- 4-panel inner-loop step curve at the
                                 smallest K and the largest K, per-trial
                                 faint lines + median IQR.

Variants compared:
  1. Random init (no SSL, no MAML) -- baseline encoder + head.
  2. Tier 1 (SSL + MAML)            -- Phase A: SSL + 100 FOMAML eps.
  3. Tier 2 (SSL + disentangle)     -- Phase B: disentangle, enc frozen.
  4. Joint Tier 1+2                 -- Phase C: enc unfrozen, joint.

Usage:
  python3.11 -m scripts.viz_joint_adaptation_curve \\
      --baseline_ckpt /tmp/joint_with_baseline/baseline.pt \\
      --phase_a_ckpt  /tmp/joint_with_baseline/phase_A.pt \\
      --phase_b_ckpt  /tmp/joint_with_baseline/phase_B.pt \\
      --phase_c_ckpt  /tmp/joint_with_baseline/phase_C.pt \\
      --data sionna \\
      --out_dir figures/joint_meta
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as TF
from torch.func import functional_call
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.config import Config
from ultraclear.asta import ASTAEncoder
from scripts.train_tier12_joint import LocationHead, LocationHeadPerTimestep
from scripts.sionna_joint_dataset import (
    build_sionna_joint_dataset, sionna_to_items,
)
from scripts.viz_tier2_helpers import TIER2_FIG


@torch.no_grad()
def _query_mse(core, fast_params, qx, qy, denorm=None):
    q_pred = functional_call(core, fast_params, (qx,))
    if denorm is not None:
        mean, std = denorm
        q_pred = q_pred * torch.from_numpy(std).to(q_pred.device) + torch.from_numpy(mean).to(q_pred.device)
        qy = qy * torch.from_numpy(std).to(qy.device) + torch.from_numpy(mean).to(qy.device)
    return float(TF.mse_loss(q_pred, qy).item())


def adapt_stepwise(encoder, head, sx, sy, qx, qy,
                   inner_lr, max_steps, denorm=None):
    class Core(torch.nn.Module):
        def __init__(self, e, h):
            super().__init__()
            self.encoder = e
            self.head = h
        def forward(self, x):
            return self.head(self.encoder(x))
    core = Core(encoder, head)
    params = dict(core.named_parameters())
    names = list(params.keys())
    fast = {n: p.detach().clone().requires_grad_(True) for n, p in params.items()}

    losses = [_query_mse(core, fast, qx, qy, denorm)]
    for _ in range(max_steps):
        pred = functional_call(core, fast, (sx,))
        loss = TF.mse_loss(pred, sy)
        grads = torch.autograd.grad(loss, list(fast.values()), create_graph=False, allow_unused=True)
        for n, g in zip(names, grads):
            if g is None: continue
            fast[n] = fast[n] - inner_lr * g
        losses.append(_query_mse(core, fast, qx, qy, denorm))
    return losses


def sample_task(items, env_id, n, seed, rng):
    pool = [it for it in items if it[2] == env_id]
    idx = (rng.permutation(len(pool))[:n] if len(pool) >= n
           else rng.integers(0, len(pool), size=n).tolist())
    X = torch.from_numpy(np.stack([pool[i][0] for i in idx])).float()
    y = torch.from_numpy(np.stack([pool[i][1] for i in idx])).float()
    return X, y


def build_encoder_from_ckpt(ckpt_path, cfg, device, per_timestep=False):
    """Build ASTAEncoder + location head and load weights.

    Supports {encoder_state, head_state} (Phase A, baseline) and
    {model_state} (Phase B/C, DisentangledTrainer) formats.

    If per_timestep=True, uses LocationHeadPerTimestep; otherwise
    the per-window LocationHead. When loading a model_state ckpt,
    the head class is auto-detected from the saved keys (anchor_attn
    = per-timestep, pool_attn = per-window) and the per_timestep
    flag is overridden accordingly.
    """
    encoder = ASTAEncoder(cfg).to(device)
    HeadCls = LocationHeadPerTimestep if per_timestep else LocationHead
    head = HeadCls(cfg.d_model).to(device)
    if ckpt_path is None:
        return encoder, head
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "encoder_state" in ck and "head_state" in ck:
        encoder.load_state_dict(ck["encoder_state"])
        try:
            head.load_state_dict(ck["head_state"])
        except RuntimeError:
            # Try the other head class — the ckpt might use the
            # other head shape than the one we built.
            other_cls = LocationHead if per_timestep else LocationHeadPerTimestep
            try:
                head2 = other_cls(cfg.d_model).to(device)
                head2.load_state_dict(ck["head_state"])
                head = head2
            except RuntimeError:
                print(f"  WARNING: ckpt head doesn't match either head shape, "
                      f"using random init")
    elif "model_state" in ck:
        ms = ck["model_state"]
        enc_state = {k.replace("disent.encoder.", ""): v
                     for k, v in ms.items() if k.startswith("disent.encoder.")}
        encoder.load_state_dict(enc_state)
        head_state = {k.replace("loc_head.", ""): v
                      for k, v in ms.items() if k.startswith("loc_head.")}
        # Auto-detect: per-timestep uses `anchor_attn`, per-window uses
        # `pool_attn`. Override the per_timestep flag if needed.
        auto_per_ts = any(k.startswith("anchor_attn") for k in head_state)
        if auto_per_ts and not per_timestep:
            head = LocationHeadPerTimestep(cfg.d_model).to(device)
        elif not auto_per_ts and per_timestep:
            head = LocationHead(cfg.d_model).to(device)
        try:
            head.load_state_dict(head_state)
        except RuntimeError:
            print(f"  WARNING: ckpt loc_head doesn't match head shape")
    return encoder, head


# Per-variant plotting style (colors and markers)
VARIANT_STYLES = {
    "Random init (no SSL, no MAML)":  {"color": "C0", "marker": "o", "ls": "-"},
    "MAML only (no SSL)":             {"color": "C4", "marker": "P", "ls": "-"},
    "Tier 1 (SSL + MAML)":            {"color": "C1", "marker": "s", "ls": "-"},
    "Tier 2 (SSL + disentangle)":     {"color": "C2", "marker": "^", "ls": "-"},
    "Joint Tier 1+2":                 {"color": "C3", "marker": "D", "ls": "-"},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_ckpt", default=None)
    p.add_argument("--maml_only_ckpt", default=None,
                   help="MAML only baseline (no SSL, with MAML). "
                        "Complements the random-init baseline by isolating "
                        "what the MAML outer loop contributes alone.")
    p.add_argument("--phase_a_ckpt", default=None)
    p.add_argument("--phase_b_ckpt", default=None)
    p.add_argument("--phase_c_ckpt", default=None)
    p.add_argument("--no_random", action="store_true",
                   help="drop the random-init baseline from the comparison")
    p.add_argument("--no_maml_only", action="store_true",
                   help="drop the MAML-only baseline from the comparison")
    p.add_argument("--no_tier2", action="store_true",
                   help="drop the Tier 2 (SSL + disentangle) variant from the comparison")
    p.add_argument("--data", default="sionna", choices=["synthetic", "sionna"])
    p.add_argument("--csv_dir", default=None)
    p.add_argument("--k_list", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", default="figures/joint_meta")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_envs", type=int, default=4,
                   help="max number of val envs to plot (top by NLOS rate "
                        "for visual diversity, or first N if --sort_by=nlos)")
    p.add_argument("--sort_by", default="nlos", choices=["nlos", "order"],
                   help="how to pick which val envs to plot")
    p.add_argument("--per_timestep", action="store_true",
                   help="use LocationHeadPerTimestep (per-timestep output) "
                        "instead of the per-window LocationHead. The head "
                        "class is also auto-detected from saved state-dict "
                        "keys (anchor_attn vs pool_attn), so this is mostly "
                        "a hint for the Random init baseline variant.")
    p.add_argument("--n_channels", type=int, default=2,
                   help="number of input channels for the encoder/dataset "
                        "(1=ToA only, 2=ToA+AoA). Default 2 to match the "
                        "joint Tier 1+2 ckpts and the per-timestep baseline.")
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = args.device
    print(f"device: {device}")

    # Build dataset
    cfg = Config()
    # Default to n_channels=2 (ToA + AoA) for the joint Tier 1+2 / baseline
    # ckpts from the 7/10 runs. Can be overridden by --n_channels for
    # per-window Tier 1 ckpts that were trained with n_channels=1.
    cfg.n_channels = args.n_channels
    cfg.device = device
    if args.data == "sionna":
        from pathlib import Path as _P
        csv_paths = None
        if args.csv_dir:
            csv_paths = sorted(str(p) for p in _P(args.csv_dir).glob("*_A1A4swapped.csv"))
        ds, scene_info = build_sionna_joint_dataset(
            csv_paths=csv_paths, n_channels=cfg.n_channels,
            seq_len=cfg.seq_len, stride=cfg.seq_len,
        )
        env_ids = [info.env_id for info in scene_info]
        train_envs = set(env_ids[:10])
        val_env_ids_full = sorted(set(env_ids[10:]) - {14})
        train_items, val_items, scene_info, denorm_info = sionna_to_items(
            ds, train_env_ids=train_envs, val_env_ids=val_env_ids_full,
        )
        ds.compute_norm_stats(train_envs)
        train_items, val_items, scene_info, denorm_info = sionna_to_items(
            ds, train_env_ids=train_envs, val_env_ids=val_env_ids_full,
        )
        # Pick which val envs to plot. Sort by NLOS rate descending to
        # get visual diversity (high NLOS envs are the interesting ones).
        if args.sort_by == "nlos":
            env_nlos = {}
            for eid in val_env_ids_full:
                nlos_in_eid = [it[4].mean() for it in val_items if it[2] == eid]
                env_nlos[eid] = float(np.mean(nlos_in_eid)) if nlos_in_eid else 0.0
            val_env_ids = sorted(env_nlos.keys(),
                                 key=lambda e: -env_nlos[e])[:args.max_envs]
        else:
            val_env_ids = val_env_ids_full[:args.max_envs]
        print(f"data: sionna, train_envs={sorted(train_envs)}, "
              f"val_envs_to_plot={val_env_ids} (nlos rates: "
              + ", ".join(f"{e}={env_nlos[e]:.2f}" if args.sort_by == "nlos" else f"{e}=?" for e in val_env_ids) + ")")
    else:
        from ultraclear.data import build_synthetic_dataset
        all_items = build_synthetic_dataset(cfg, seed=args.seed)
        env_ids = sorted({it[2] for it in all_items})
        n_train = int(0.8 * len(env_ids))
        train_envs = set(env_ids[:n_train])
        val_env_ids_full = sorted(set(env_ids) - train_envs)
        train_items = [it for it in all_items if it[2] in train_envs]
        val_items = [it for it in all_items if it[2] not in train_envs]
        denorm_info = {eid: None for eid in val_env_ids_full}
        val_env_ids = val_env_ids_full[:args.max_envs]
        print(f"data: synthetic, val_envs_to_plot={val_env_ids}")

    # Build the variants. --no_random / --no_maml_only / --no_tier2
    # drop the corresponding variant if its checkpoint is provided.
    variants = []
    if not args.no_random:
        if args.baseline_ckpt is None:
            print("  --no_random not set but no --baseline_ckpt; skipping random init")
        else:
            print(f"  loading Random init (no SSL, no MAML) from {args.baseline_ckpt}...")
            encoder, head = build_encoder_from_ckpt(args.baseline_ckpt, cfg, device, per_timestep=args.per_timestep)
            variants.append(("Random init (no SSL, no MAML)", encoder, head))
    if not args.no_maml_only:
        if args.maml_only_ckpt is None:
            print("  --no_maml_only not set but no --maml_only_ckpt; skipping MAML only")
        else:
            print(f"  loading MAML only (no SSL) from {args.maml_only_ckpt}...")
            encoder, head = build_encoder_from_ckpt(args.maml_only_ckpt, cfg, device, per_timestep=args.per_timestep)
            variants.append(("MAML only (no SSL)", encoder, head))
    if not args.no_tier2:
        for label, ckpt_path in [("Tier 2 (SSL + disentangle)", args.phase_b_ckpt)]:
            if ckpt_path is None:
                print(f"  skipping {label}: no checkpoint provided")
                continue
            print(f"  loading {label} from {ckpt_path}...")
            encoder, head = build_encoder_from_ckpt(ckpt_path, cfg, device, per_timestep=args.per_timestep)
            variants.append((label, encoder, head))
    for label, ckpt_path in [
        ("Tier 1 (SSL + MAML)",            args.phase_a_ckpt),
        ("Joint Tier 1+2",                 args.phase_c_ckpt),
    ]:
        if ckpt_path is None:
            print(f"  skipping {label}: no checkpoint provided")
            continue
        print(f"  loading {label} from {ckpt_path}...")
        encoder, head = build_encoder_from_ckpt(ckpt_path, cfg, device, per_timestep=args.per_timestep)
        variants.append((label, encoder, head))

    if not variants:
        print("ERROR: no variants loaded")
        return

    # Run K-shot evaluation for each variant
    # results[(variant, k, eid)] -> (n_trials, max_steps+1) array
    rng = np.random.default_rng(args.seed)
    results = {}
    for label, encoder, head in variants:
        print(f"  evaluating {label}...")
        for k in args.k_list:
            for eid in val_env_ids:
                denorm = denorm_info[eid]
                trials = []
                for trial in range(args.n_trials):
                    enc = copy.deepcopy(encoder)
                    h = copy.deepcopy(head)
                    sx, sy = sample_task(val_items, eid, k, args.seed + trial, rng)
                    qx, qy = sample_task(val_items, eid, 20, args.seed + 1000 + trial, rng)
                    sx, sy = sx.to(device), sy.to(device)
                    qx, qy = qx.to(device), qy.to(device)
                    losses = adapt_stepwise(enc, h, sx, sy, qx, qy,
                                            inner_lr=cfg.maml_inner_lr,
                                            max_steps=args.max_steps,
                                            denorm=denorm)
                    trials.append(losses)
                results[(label, k, eid)] = np.array(trials)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- plot 1: K-shot curve, one panel per val env, all 4 variants -----
    n_envs = len(val_env_ids)
    n_cols = min(n_envs, 4)
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.6 * n_rows),
                              sharey=False)
    axes = np.array(axes).reshape(-1)
    for ax_idx, eid in enumerate(val_env_ids):
        ax = axes[ax_idx]
        for label, _, _ in variants:
            style = VARIANT_STYLES[label]
            # median over trials, at max_steps, for each K
            ks_arr = np.array(args.k_list, dtype=float)
            medians = np.array([np.median(results[(label, k, eid)][:, -1])
                                 for k in args.k_list])
            iqrs = np.array([(np.percentile(results[(label, k, eid)][:, -1], 75) -
                              np.percentile(results[(label, k, eid)][:, -1], 25)) / 2
                             for k in args.k_list])
            # per-trial faint lines
            for trial in range(args.n_trials):
                y_per_trial = [results[(label, k, eid)][trial, -1] for k in args.k_list]
                ax.plot(args.k_list, y_per_trial, color=style["color"], alpha=0.15, lw=0.8)
            # median with IQR error bars
            ax.errorbar(args.k_list, medians, yerr=iqrs,
                        marker=style["marker"], capsize=3, lw=2,
                        color=style["color"], label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("K (support shots)")
        ax.set_title(f"val env {eid}")
        ax.grid(True, alpha=0.3, which="both")
        if ax_idx == 0:
            ax.set_ylabel(f"query MSE (m², log)\n"
                          f"after {args.max_steps} inner-loop steps")
            ax.legend(loc="upper right", fontsize=8)
    # hide unused panels
    for j in range(len(val_env_ids), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Joint Tier 1+2 — K-shot adaptation to held-out Sionna envs\n"
                 f"({n_envs} val envs, {args.n_trials} trials, "
                 f"median ± half-IQR)", fontsize=12)
    fig.tight_layout()
    out = out_dir / "viz_joint_k_shot.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")

    # ----- plot 2: inner-loop step curve, one panel per K, all envs aggregated -----
    n_k = len(args.k_list)
    n_cols2 = min(n_k, 3)
    n_rows2 = (n_k + n_cols2 - 1) // n_cols2
    fig, axes = plt.subplots(n_rows2, n_cols2, figsize=(4.5 * n_cols2, 3.6 * n_rows2),
                              sharey=True)
    axes = np.array(axes).reshape(-1)
    steps = np.arange(0, args.max_steps + 1)
    for ax_idx, k in enumerate(args.k_list):
        ax = axes[ax_idx]
        for label, _, _ in variants:
            style = VARIANT_STYLES[label]
            # per-env, per-trial curves as faint lines
            for eid in val_env_ids:
                for trial in range(args.n_trials):
                    y_per_trial = results[(label, k, eid)][trial]
                    ax.plot(steps, y_per_trial, color=style["color"], alpha=0.05, lw=0.7)
            # median over (envs × trials) at each step
            step_arrays = [results[(label, k, eid)][:, s].flatten()
                            for s in range(args.max_steps + 1)]
            medians = [np.median(a) for a in step_arrays]
            iqrs = [(np.percentile(a, 75) - np.percentile(a, 25)) / 2 for a in step_arrays]
            ax.errorbar(steps, medians, yerr=iqrs,
                        marker=style["marker"], capsize=3, lw=2,
                        color=style["color"], label=label)
        ax.set_yscale("log")
        ax.set_xlabel("inner-loop step")
        ax.set_title(f"K = {k} shot{'s' if k > 1 else ''}")
        ax.grid(True, alpha=0.3, which="both")
        if ax_idx == 0:
            ax.set_ylabel(f"query MSE (m², log)\n"
                          f"median over {n_envs} val envs × {args.n_trials} trials")
            ax.legend(loc="upper right", fontsize=8)
    for j in range(len(args.k_list), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Joint Tier 1+2 — inner-loop adaptation curves on held-out Sionna envs\n"
                 f"(step 0 = no adaptation, then {args.max_steps} inner-loop steps)",
                 fontsize=12)
    fig.tight_layout()
    out = out_dir / "viz_joint_step_curve.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")

    # ----- save raw CSV -----
    csv_out = out_dir / "viz_joint_adaptation.csv"
    with open(csv_out, "w") as f:
        f.write("variant,K_shot,val_env,trial,inner_step,query_mse\n")
        for (label, k, eid), arr in results.items():
            for trial in range(arr.shape[0]):
                for s in range(arr.shape[1]):
                    f.write(f"{label},{k},{eid},{trial},{s},{arr[trial, s]:.4f}\n")
    print(f"saved {csv_out}")

    # ----- print summary table -----
    print()
    print("=" * 80)
    print(f"  K-shot adaptation summary (median query MSE at step={args.max_steps}, m²)")
    print("=" * 80)
    print(f"  {'variant':<40} " + "  ".join(f"K={k:<4}" for k in args.k_list))
    for label, _, _ in variants:
        line = f"  {label:<40} "
        for k in args.k_list:
            arrs = [results[(label, k, e)][:, -1] for e in val_env_ids]
            v = np.median(np.concatenate(arrs))
            line += f"  {v:<7.2f}"
        print(line)


if __name__ == "__main__":
    main()
