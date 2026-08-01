"""Compare Tier 1 MAMLCore vs Tier 2 (window-head) DisentangledTrainer
on the same held-out Sionna val scenes (3, 8, 13, 18).

Why this script exists:
  - `eval_tier1_kshot.py`     → MAMLCore only (per-window (B, 2))
  - `eval_tier2_sionna_reference.py` → DisentangledTrainer only (assumes
    per-timestep (B, T, 2) location head)
  Our new `train_tier2_synthetic_window.py` uses a Tier-2-style model
  with a Tier-1-style per-window head (output (B, 2)), so neither eval
  drops in cleanly. This script handles BOTH contracts explicitly.

Usage:
    python -m scripts.eval_tier1_vs_tier2_window
    python -m scripts.eval_tier1_vs_tier2_window --ks 0 10 20
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.config import Config
from ultraclear.tier1_maml import MAMLCore, adapt_and_evaluate
from scripts.train_tier1_sionna import sample_task_multi
from scripts.train_tier2_synthetic_window import DisentangledTrainer
from scripts.multi_scene_dataset import build_default_multi_scene

import torch.nn.functional as TF
from torch.func import functional_call


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def denormalize_xy_per_scene(z_xy: torch.Tensor, sinfo) -> torch.Tensor:
    """Convert z-scored (x, y) back to meters using per-scene stats."""
    if sinfo.tag_xy_mean is None or sinfo.tag_xy_std is None:
        return z_xy
    if isinstance(z_xy, torch.Tensor):
        return (
            z_xy * torch.from_numpy(sinfo.tag_xy_std).float().to(z_xy.device)
            + torch.from_numpy(sinfo.tag_xy_mean).float().to(z_xy.device)
        )
    return z_xy * sinfo.tag_xy_std + sinfo.tag_xy_mean


def load_mamlcore(ckpt_path: str, cfg: Config, device: str) -> MAMLCore:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    core = MAMLCore(cfg).to(device)
    core.load_state_dict(ck["model_state"])
    core.eval()
    return core


def load_disentangled(ckpt_path: str, cfg: Config, device: str) -> DisentangledTrainer:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # n_envs: prefer the explicit 'n_envs' field; fall back to
    # inferring from the adversary head's output dim, or 12.
    n_envs = ck.get("n_envs")
    if n_envs is None:
        # Infer from the adversary head's output dim (last layer weight)
        ms = ck.get("model_state", {})
        for k, v in ms.items():
            if k.endswith("disent.adv.net.2.weight") or k.endswith("adv.net.2.weight"):
                n_envs = v.shape[0]
                break
    if n_envs is None:
        n_envs = 12
    n_conds = 3
    ms = ck.get("model_state", {})
    for k, v in ms.items():
        if k.endswith("cond_head.weight"):
            n_conds = v.shape[0]
            break
    model = DisentangledTrainer(cfg, n_envs=n_envs, n_conds=n_conds).to(device)
    missing, unexpected = model.load_state_dict(ms, strict=False)
    if missing:
        print(f"  WARNING missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  WARNING unexpected keys (first 5): {unexpected[:5]}")
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Tier 1 eval (MAMLCore, with K-shot adaptation)
# ----------------------------------------------------------------------------
def eval_tier1_Kshot(
    env_id: int, K: int, n_trials: int, ds, cfg: Config,
    core: MAMLCore, sinfo, device: str,
) -> float:
    """Per-scene K-shot RMSE in meters."""
    loc_errs = []
    for trial in range(n_trials):
        if K == 0:
            # zero-shot: just forward query through meta-init
            qx, qy = sample_task_multi(ds, env_id, cfg.q_query,
                                        cfg.seed + 2000 + trial)
            qx, qy = qx.to(device), qy.to(device)
            with torch.no_grad():
                q_pred = core(qx)
            T = qx.size(1)
            q_y_mid = qy[:, T // 2, :]
        else:
            sx, sy = sample_task_multi(ds, env_id, K, cfg.seed + 1000 + trial)
            qx, qy = sample_task_multi(ds, env_id, cfg.q_query,
                                        cfg.seed + 2000 + trial)
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)
            adapted = MAMLCore(cfg).to(device)
            adapted.load_state_dict(core.state_dict())
            q_loss, info = adapt_and_evaluate(
                adapted, sx, sy, qx, qy,
                inner_lr=cfg.maml_inner_lr,
                inner_steps=cfg.maml_inner_steps,
            )
            q_pred = info.get("q_pred")
            T = qx.size(1)
            q_y_mid = qy[:, T // 2, :]
            if q_pred is None:
                continue

        # q_pred is (B, 2), z-scored. Denormalize per sample.
        per_sample_err = []
        for i in range(q_pred.size(0)):
            p_m = denormalize_xy_per_scene(q_pred[i], sinfo)
            y_m = denormalize_xy_per_scene(q_y_mid[i], sinfo)
            per_sample_err.append(float(torch.linalg.norm(p_m - y_m).item()))
        loc_errs.append(float(np.sqrt(np.mean(np.square(per_sample_err)))))
    return float(np.mean(loc_errs))


def _adapt_tier2_inner(
    model: DisentangledTrainer,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    inner_lr: float = 1e-2,
    inner_steps: int = 5,
) -> dict:
    """Adapt FiLM + h_phi + loc_head on support set. Returns fast-weights dict.

    Mirrors tier1_maml.adapt_and_evaluate but operates on the
    DisentangledTrainer (6-tuple forward) and only touches the params we
    want to adapt: disent.film.*, disent.h_phi.*, loc_head.*. Everything
    else (encoder, adversary, cond_head, nlos head) stays frozen.

    Returns the fast-weights dict so the caller can pass it to
    functional_call on the query set.
    """
    adapt_prefixes = ("disent.film.", "disent.h_phi.", "loc_head.")
    params = {n: p for n, p in model.named_parameters()
              if n.startswith(adapt_prefixes)}
    fast = {n: p.clone().requires_grad_(True) for n, p in params.items()}
    for _ in range(inner_steps):
        # functional_call: invoke the model's forward with overridden params.
        # The DisentangledTrainer.forward signature is (x, detach_c_for_adv=False),
        # but functional_call handles kwargs the same way.
        h_prime, c, adv_logits, loc, nlos_logits, cond_logits = functional_call(
            model, fast, (support_x,), {"detach_c_for_adv": False},
        )
        T = support_x.size(1)
        y_mid = support_y[:, T // 2, :]
        # loc is (B, 2) per-window
        loss = TF.mse_loss(loc, y_mid)
        grads = torch.autograd.grad(
            loss, list(fast.values()), create_graph=False, allow_unused=True,
        )
        new_fast = {}
        for n, g in zip(fast.keys(), grads):
            if g is None:
                new_fast[n] = fast[n]
            else:
                new_fast[n] = fast[n] - inner_lr * g
        fast = new_fast
    return fast


def eval_tier2_Kshot(
    env_id: int, K: int, n_trials: int, ds, cfg: Config,
    model: DisentangledTrainer, sinfo, device: str,
    inner_lr: float = 1e-2,
    inner_steps: int = 5,
) -> float:
    """K-shot adaptation for Tier 2 (window-head).

    For each trial: sample K support + Q query windows from env_id,
    adapt (FiLM, h_phi, loc_head) on support, evaluate query RMSE
    in meters (per-scene denormalization).
    K=0: zero-shot — just forward the query, no adaptation.
    """
    loc_errs = []
    for trial in range(n_trials):
        if K == 0:
            qx, qy = sample_task_multi(ds, env_id, cfg.q_query,
                                        cfg.seed + 2000 + trial)
            qx, qy = qx.to(device), qy.to(device)
            with torch.no_grad():
                _h, _c, _adv, loc, _nlos, _cond = model(qx)
        else:
            sx, sy = sample_task_multi(ds, env_id, K, cfg.seed + 1000 + trial)
            qx, qy = sample_task_multi(ds, env_id, cfg.q_query,
                                        cfg.seed + 2000 + trial)
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)
            fast = _adapt_tier2_inner(model, sx, sy, inner_lr, inner_steps)
            with torch.no_grad():
                _h, _c, _adv, loc, _nlos, _cond = functional_call(
                    model, fast, (qx,), {"detach_c_for_adv": False},
                )
        # loc is (B, 2) z-scored
        if loc.dim() == 3:
            loc = loc[:, qx.size(1) // 2, :]
        T = qx.size(1)
        q_y_mid = qy[:, T // 2, :]
        per_sample_err = []
        for i in range(loc.size(0)):
            p_m = denormalize_xy_per_scene(loc[i], sinfo)
            y_m = denormalize_xy_per_scene(q_y_mid[i], sinfo)
            per_sample_err.append(float(torch.linalg.norm(p_m - y_m).item()))
        loc_errs.append(float(np.sqrt(np.mean(np.square(per_sample_err)))))
    return float(np.mean(loc_errs))


# ----------------------------------------------------------------------------
# Tier 2 eval (DisentangledTrainer, zero-shot only — no inner-loop support
# in the disentangled model)
# ----------------------------------------------------------------------------
def eval_tier2_zero_shot(
    env_id: int, ds, sinfo, model: DisentangledTrainer, device: str,
) -> dict:
    """Per-scene RMSE/MAE/median/p90 in meters. Returns summary stats."""
    indices = [i for i, (s_idx, _, _) in enumerate(ds.windows)
               if ds.scene_info[s_idx].env_id == env_id]
    if not indices:
        return {"rmse": float("nan"), "mae": float("nan"),
                "median": float("nan"), "p90": float("nan"), "n": 0}
    Xs, ys = [], []
    for i in indices:
        X, y, _, _, _ = ds[i]
        Xs.append(X); ys.append(y)
    X = torch.stack(Xs).to(device)
    y = torch.stack(ys).to(device)
    with torch.no_grad():
        _h, _c, _adv, loc, _nlos, _cond = model(X)
    T = X.size(1)
    y_mid = y[:, T // 2, :]
    # loc is (B, 2) for our window-head variant
    if loc.dim() == 3:
        loc_z = loc[:, T // 2, :]
    else:
        loc_z = loc
    per_sample_err = []
    for i in range(X.size(0)):
        p_m = denormalize_xy_per_scene(loc_z[i], sinfo)
        y_m = denormalize_xy_per_scene(y_mid[i], sinfo)
        per_sample_err.append(float(torch.linalg.norm(p_m - y_m).item()))
    errs = np.asarray(per_sample_err)
    return {
        "rmse": float(np.sqrt(np.mean(errs ** 2))),
        "mae": float(errs.mean()),
        "median": float(np.median(errs)),
        "p90": float(np.percentile(errs, 90)),
        "n": len(errs),
        "errs": errs.tolist(),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier1_ckpt", required=True)
    p.add_argument("--tier2_ckpt", default=None,
                   help="optional: Tier 2 (window-head) ckpt; if omitted, "
                        "tier 2 column is left blank in the table.")
    p.add_argument("--val_envs", type=int, nargs="+", default=[3, 8, 13, 18])
    p.add_argument("--ks", type=int, nargs="+", default=[0, 10, 20],
                   help="K-shot values for the Tier 1 sweep.")
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--out_csv", default="figures/tier1_vs_tier2_window.csv")
    p.add_argument("--out_fig", default="figures/tier1_vs_tier2_window.png")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    # Load ckpts to get cfg from one of them
    print(f"loading Tier 1 ckpt: {args.tier1_ckpt}")
    ck1 = torch.load(args.tier1_ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ck1["cfg"].items()
                    if k in Config.__dataclass_fields__})
    print(f"  cfg: n_channels={cfg.n_channels}  seq_len={cfg.seq_len}  "
          f"d_model={cfg.d_model}  n_anchors={cfg.n_anchors}")

    print("\nloading Sionna data (all 21 scenes, env 14 skipped)...")
    ds, scene_info = build_default_multi_scene(
        n_channels=cfg.n_channels, seq_len=cfg.seq_len, stride=32,
        all_scenes=True,
    )
    train_scenes = set(ck1.get("train_scenes") or list(range(10)))
    norm_mean, norm_std = ds.compute_norm_stats(train_scenes)
    ds.norm_mean = norm_mean
    ds.norm_std = norm_std
    print(f"  {len(ds)} windows across {len(scene_info)} scenes")

    # Build Tier 1 model
    core = load_mamlcore(args.tier1_ckpt, cfg, args.device)

    # Build Tier 2 model (if ckpt provided)
    model2 = None
    if args.tier2_ckpt:
        print(f"\nloading Tier 2 ckpt: {args.tier2_ckpt}")
        # Match cfg to the tier 2 ckpt's architecture params
        ck2 = torch.load(args.tier2_ckpt, map_location="cpu", weights_only=False)
        if "cfg" in ck2:
            for k, v in ck2["cfg"].items():
                if k in {"n_channels", "d_model", "seq_len", "n_anchors",
                         "micro_dim", "adv_hidden"}:
                    setattr(cfg, k, v)
            print(f"  cfg overridden by Tier 2 ckpt: n_channels={cfg.n_channels}")
        model2 = load_disentangled(args.tier2_ckpt, cfg, args.device)

    # ---- Tier 1 sweep ----
    print(f"\n{'='*72}")
    print(f"Tier 1 MAMLCore K-shot sweep (envs={args.val_envs})")
    print(f"{'='*72}")
    tier1_results = {eid: {} for eid in args.val_envs}
    t0 = time.time()
    for eid in args.val_envs:
        sinfo = next(s for s in scene_info if s.env_id == eid)
        for K in args.ks:
            rmse = eval_tier1_Kshot(eid, K, args.n_trials, ds, cfg,
                                     core, sinfo, args.device)
            tier1_results[eid][K] = rmse
            print(f"  env {eid:>2} K={K:>2}: {rmse:.3f} m")
    print(f"  tier 1 done in {time.time()-t0:.1f}s")

    # ---- Tier 2 K-shot sweep ----
    tier2_results = {eid: {} for eid in args.val_envs}
    tier2_full = {}  # extra metrics (mae/median/p90) for the zero-shot only
    if model2 is not None:
        print(f"\n{'='*72}")
        print(f"Tier 2 window-head K-shot sweep (envs={args.val_envs})")
        print(f"{'='*72}")
        t0 = time.time()
        for eid in args.val_envs:
            sinfo = next(s for s in scene_info if s.env_id == eid)
            for K in args.ks:
                if K == 0:
                    r = eval_tier2_zero_shot(eid, ds, sinfo, model2, args.device)
                    rmse = r["rmse"]
                    tier2_full[eid] = r
                else:
                    rmse = eval_tier2_Kshot(
                        eid, K, args.n_trials, ds, cfg,
                        model2, sinfo, args.device,
                    )
                tier2_results[eid][K] = rmse
                print(f"  env {eid:>2} K={K:>2}: {rmse:.3f} m")
        print(f"  tier 2 done in {time.time()-t0:.1f}s")

    # ---- Pretty table ----
    print(f"\n{'='*72}")
    print(f"Side-by-side: Tier 1 (MAMLCore, K-shot) vs Tier 2 (window-head, K-shot)")
    print(f"{'='*72}")
    # Header
    cols = []
    for K in args.ks:
        cols.append(f"T1 K={K}")
    for K in args.ks:
        cols.append(f"T2 K={K}")
    col_w = max(8, max(len(c) for c in cols) + 2)
    print(f"{'env':>4} | " + " | ".join(f"{c:>{col_w}}" for c in cols))
    print("-" * (6 + (col_w + 3) * len(cols)))
    for eid in args.val_envs:
        row_vals = []
        for K in args.ks:
            row_vals.append(f"{tier1_results[eid][K]:.3f}")
        for K in args.ks:
            row_vals.append(f"{tier2_results[eid][K]:.3f}")
        print(f"{eid:>4} | " + " | ".join(f"{v:>{col_w}}" for v in row_vals))

    # ---- Save CSV ----
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["env_id", "role"]
        for K in args.ks:
            hdr.append(f"tier1_K{K}_rmse_m")
        for K in args.ks:
            hdr.append(f"tier2_K{K}_rmse_m")
        # zero-shot extras (mae/median/p90)
        if 0 in args.ks:
            hdr.append("tier2_K0_mae_m")
            hdr.append("tier2_K0_median_m")
            hdr.append("tier2_K0_p90_m")
        w.writerow(hdr)
        for eid in args.val_envs:
            role = "val"
            row = [eid, role]
            for K in args.ks:
                row.append(f"{tier1_results[eid][K]:.4f}")
            for K in args.ks:
                row.append(f"{tier2_results[eid][K]:.4f}")
            if 0 in args.ks and eid in tier2_full:
                r = tier2_full[eid]
                row += [f"{r['mae']:.4f}", f"{r['median']:.4f}",
                        f"{r['p90']:.4f}"]
            w.writerow(row)
    print(f"\nsaved CSV → {out_csv}")

    # ---- Figure: per-env comparison, 2-panel (Tier 1 / Tier 2) ----
    # Side-by-side panels avoid the cramped 12-bars-per-group layout
    # that resulted from packing all K values for both models in a
    # single axes.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                              constrained_layout=True)
    ks = list(args.ks)
    n_k = len(ks)
    width = 0.8 / n_k  # total 0.8 of group width for the K-bars
    x = np.arange(len(args.val_envs))
    colors_t1 = plt.cm.Blues(np.linspace(0.4, 0.9, n_k))
    colors_t2 = plt.cm.Oranges(np.linspace(0.4, 0.9, n_k))

    # ---- Panel 1: Tier 1 ----
    ax = axes[0]
    for i, K in enumerate(ks):
        vals = [tier1_results[e][K] for e in args.val_envs]
        offset = (i - (n_k - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, color=colors_t1[i],
                       edgecolor="white", linewidth=0.5,
                       label=f"K={K}")
        # Value labels on top of each bar (compact)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                     f"{v:.2f}", ha="center", va="bottom",
                     fontsize=7, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"env {e}" for e in args.val_envs], fontsize=10)
    ax.set_title(f"Tier 1 (MAMLCore, K-shot adapt)\n"
                  f"{Path(args.tier1_ckpt).parent.name}",
                  fontsize=11)
    ax.set_ylabel("loc RMSE (m)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=8, ncol=2, title="K (support)",
               title_fontsize=8)

    # ---- Panel 2: Tier 2 ----
    ax = axes[1]
    for i, K in enumerate(ks):
        vals = [tier2_results[e][K] for e in args.val_envs]
        offset = (i - (n_k - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, color=colors_t2[i],
                       edgecolor="white", linewidth=0.5,
                       label=f"K={K}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                     f"{v:.2f}", ha="center", va="bottom",
                     fontsize=7, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"env {e}" for e in args.val_envs], fontsize=10)
    tier2_label = (Path(args.tier2_ckpt).parent.name
                   if args.tier2_ckpt else "(none)")
    ax.set_title(f"Tier 2 (DisentangledTrainer, window-head, K-shot)\n"
                  f"{tier2_label}",
                  fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=8, ncol=2, title="K (support)",
               title_fontsize=8)

    fig.suptitle(
        "Tier 1 vs Tier 2 — K-shot localization RMSE on held-out Sionna val envs\n"
        "(Tier 1 = encoder + head meta-learned;  Tier 2 = frozen Tier 1 encoder + FiLM + per-window head)",
        fontsize=12,
    )
    fig.savefig(args.out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure → {args.out_fig}")


if __name__ == "__main__":
    main()