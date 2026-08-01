"""Tier 2 reference eval — single model, single cond, loc RMSE on trained + untrained envs.

Companion to `eval_tier1_reference.py`. Runs the Tier 2
DisentangledTrainer (FiLM + adversary on top of frozen MAML encoder)
on the SYNTHETIC dataset (Tier 2 was trained on synthetic, not
Sionna). Reports loc RMSE on:
  (a) one specific TRAINED environment
  (b) one specific UNTRAINED environment (held-out during Tier 2 training)

For this first pass: one micro condition (cond 0 = low-NLOS by default).
Later passes can extend to all 3 conds and add NLOS detection.

Usage:
    python -m scripts.eval_tier2_reference
    python -m scripts.eval_tier2_reference --train_env 0 --val_env 11 --cond 0
    python -m scripts.eval_tier2_reference --cond 1  # mid-NLOS
    python -m scripts.eval_tier2_reference --cond 2  # high-NLOS
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.config import make_default
from ultraclear.data import build_synthetic_dataset, TensorUWB
from torch.utils.data import DataLoader
from scripts.train_tier2_synthetic import DisentangledTrainer


def load_model_and_data(ckpt_path: str, device: str = "cpu"):
    """Load Tier 2 ckpt and reconstruct the matching synthetic dataset.

    The Tier 2 ckpt is self-describing: it has `cfg` saved. We re-derive
    the synthetic data with the SAME cfg (especially n_channels which
    was 2 when trained on tier1_sionna_best.pt). The cfg may have
    n_synthetic_envs=8 in the saved ckpt (because that was the value
    at the time the run started); we use the saved cfg exactly to
    reproduce the data distribution.
    """
    print(f"loading ckpt: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # The cfg in the ckpt may have n_synthetic_envs=8 (old default
    # before our 8->12 change today). Use cfg as-is to reproduce.
    saved_cfg = ck["cfg"]
    print(f"  saved cfg n_channels = {saved_cfg.get('n_channels')}")
    print(f"  saved cfg n_synthetic_envs = {saved_cfg.get('n_synthetic_envs')}")
    print(f"  saved cfg n_micro_conditions = {saved_cfg.get('n_micro_conditions')}")
    print(f"  saved cfg seq_len = {saved_cfg.get('seq_len')}")
    print(f"  saved cfg d_model = {saved_cfg.get('d_model')}")

    # Build a Config object with the saved values. Some keys may not
    # be in the Config dataclass (e.g. training-loop params); setattr
    # handles that.
    from dataclasses import fields
    from ultraclear.config import Config
    cfg = Config()
    valid_keys = {f.name for f in fields(Config)}
    for k, v in saved_cfg.items():
        if k in valid_keys:
            setattr(cfg, k, v)
    # Also stash training-loop params that downstream code needs
    for k in ("nlos_loss_weight", "cond_loss_weight"):
        if k in saved_cfg:
            setattr(cfg, k, saved_cfg[k])
    print(f"  using cfg n_channels = {cfg.n_channels}, n_synthetic_envs = {cfg.n_synthetic_envs}")

    # Build data
    items = build_synthetic_dataset(cfg, seed=cfg.seed)
    # Train/val split: must match how it was done in the run.
    # The run used the 10/2 split for n_synthetic_envs=12, else 80/20.
    env_ids = sorted({it[2] for it in items})
    if len(env_ids) == 12:
        train_envs = set(env_ids[:10])
    else:
        n_train_envs = max(1, int(0.8 * len(env_ids)))
        train_envs = set(env_ids[:n_train_envs])

    # Build the model. The ckpt was saved with the model's state_dict
    # which has keys like 'disent.encoder.*', 'loc_head.*', 'nlos_mlp.*',
    # 'cond_head.*'. DisentangledTrainer takes (cfg, n_envs) — n_envs
    # is used only for the adversary head which has shape (adv_hidden, n_envs).
    # Use the n_envs from the saved ckpt to match.
    n_envs_in_ckpt = ck.get("n_envs", len(env_ids))
    print(f"  n_envs in ckpt = {n_envs_in_ckpt}")
    model = DisentangledTrainer(cfg, n_envs=n_envs_in_ckpt).to(device)
    # Load with strict=False because the n_envs might mismatch the
    # current data env count, but the adversary's last layer is just
    # dropped (which is fine for inference — we don't use adv_logits).
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    if missing:
        print(f"  WARNING missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval()
    return model, cfg, items, train_envs


def filter_by_cond_env(items, env_id: int, cond: int):
    """Return items where env_id matches and cond_id matches."""
    return [it for it in items if it[2] == env_id and int(it[3]) == cond]


def run_eval(
    env_id: int,
    cond: int,
    items,
    model: DisentangledTrainer,
    cfg,
    device: str = "cpu",
) -> dict:
    """Eval the model on (env_id, cond). No K-shot — Tier 2 is a
    regular supervised model (frozen encoder, no MAML). Forward pass
    only. Denormalize the predicted (x, y) to meters for reporting.

    Returns per-window loc error in meters and the raw pred/gt arrays.
    """
    # Filter to (env_id, cond). The synthetic data is already in
    # meters (no z-scoring), so no denormalization is needed for y.
    sub = filter_by_cond_env(items, env_id, cond)
    if not sub:
        raise RuntimeError(f"no items for env_id={env_id} cond={cond}")
    print(f"  eval set: {len(sub)} windows for env {env_id}, cond {cond}")

    # The X channels: the synthetic data stores ToA (and AoA if
    # n_channels=2) in raw meters. Forward directly.
    # We need to forward the entire batch at once to be efficient.
    X = torch.from_numpy(np.stack([it[0] for it in sub])).float().to(device)
    y = torch.from_numpy(np.stack([it[1] for it in sub])).float().to(device)
    nlos_gt = torch.from_numpy(np.stack([it[4] for it in sub])).float().to(device)
    cond_gt = torch.tensor([int(it[3]) for it in sub], dtype=torch.long).to(device)

    with torch.no_grad():
        h_prime, c, adv_logits, loc, nlos_logits, cond_logits = model(X)
    # loc: (B, T, 2); y: (B, T, 2). Per-timestep MSE / RMSE in meters.
    err = torch.linalg.norm(loc - y, dim=-1)  # (B, T)
    loc_rmse_m = float(torch.sqrt((err ** 2).mean()).item())
    loc_mae_m = float(err.mean().item())
    loc_errs_per_window = err.mean(dim=-1).cpu().numpy()  # (B,) avg over T per window
    loc_errs_all = err.flatten().cpu().numpy()            # (B*T,) every timestep

    # Also compute the per-cond NLOS rate (for context)
    nlos_rate = float(nlos_gt.mean().item())

    return {
        "env_id": env_id,
        "cond": cond,
        "n_windows": len(sub),
        "loc_rmse_m": loc_rmse_m,
        "loc_mae_m": loc_mae_m,
        "loc_errs_per_window": loc_errs_per_window.tolist(),
        "loc_errs_all": loc_errs_all.tolist(),
        "pred_xy": loc.cpu().numpy().tolist(),   # (B, T, 2)
        "gt_xy": y.cpu().numpy().tolist(),       # (B, T, 2)
        "nlos_rate": nlos_rate,
    }


def make_figure_one_env(
    env_id: int,
    r: dict,
    train_envs: set,
    cond: int,
    out_path: str,
):
    """3-panel figure for a single env at a single cond."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    is_trained = env_id in train_envs

    # Panel 1: bar of loc error statistics
    ax = axes[0]
    errs = np.asarray(r["loc_errs_per_window"])
    metrics = [("RMSE", r["loc_rmse_m"], "C0"),
               ("MAE", r["loc_mae_m"], "C1"),
               ("median", float(np.median(errs)), "C2"),
               ("p90", float(np.percentile(errs, 90)), "C3")]
    names = [m[0] for m in metrics]
    vals = [m[1] for m in metrics]
    colors = [m[2] for m in metrics]
    bars = ax.bar(names, vals, color=colors, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}m",
                 ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("loc error (m)")
    role = "TRAINED" if is_trained else "untrained"
    ax.set_title(f"env {env_id} ({role}, cond {cond}, n_q={r['n_windows']})\n"
                  f"loc error summary (per-window avg over T timesteps)")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: per-window loc error CDF
    ax = axes[1]
    errs_sorted = np.sort(errs)
    cdf = np.arange(1, len(errs_sorted) + 1) / len(errs_sorted)
    ax.plot(errs_sorted, cdf, color="C0", lw=2,
             label=f"env {env_id} (n={len(errs_sorted)})")
    for thresh in [0.5, 1.0, 2.0, 5.0]:
        frac = float((errs_sorted <= thresh).mean())
        ax.axvline(thresh, color="grey", linestyle=":", alpha=0.5)
        ax.text(thresh, 0.02, f"{thresh:.1f}m\n{frac*100:.0f}%",
                 ha="center", va="bottom", fontsize=7, color="grey")
    ax.set_xlabel("loc error (m)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Per-window loc error CDF\n"
                  f"mean = {r['loc_mae_m']:.2f}m, median = {float(np.median(errs)):.2f}m, "
                  f"p90 = {float(np.percentile(errs, 90)):.2f}m")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Panel 3: scatter of last-timestep pred vs GT (one point per window)
    ax = axes[2]
    pred = np.asarray(r["pred_xy"])  # (B, T, 2)
    gt = np.asarray(r["gt_xy"])
    T = pred.shape[1]
    # Use the middle timestep (matches how the loss is computed in training)
    mid = T // 2
    pred_mid = pred[:, mid, :]  # (B, 2)
    gt_mid = gt[:, mid, :]      # (B, 2)
    ax.scatter(gt_mid[:, 0], gt_mid[:, 1], s=12, alpha=0.5, color="C0",
                label="GT (mid-timestep)", edgecolors="none")
    ax.scatter(pred_mid[:, 0], pred_mid[:, 1], s=12, alpha=0.5, color="C3",
                marker="x", label="pred (mid-timestep)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    nlos_pct = r["nlos_rate"] * 100
    ax.set_title(f"Pred vs GT at middle timestep (● GT, × pred)\n"
                  f"n = {len(gt_mid)} windows, NLOS rate in input = {nlos_pct:.1f}%")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right")

    fig.suptitle(f"Tier 2 reference eval — env {env_id} ({role}), cond {cond} (low-NLOS)",
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure → {out_path}")


def print_summary(results: dict, ckpt_path: str, train_envs: set):
    print(f"\n{'='*72}")
    print(f"Tier 2 reference eval — {ckpt_path}")
    print(f"{'='*72}")
    print(f"{'env':<6} {'role':<10} {'cond':>4} {'n_w':>5} {'RMSE':>8} {'MAE':>8}  "
          f"{'nlos%':>7}")
    print("-" * 72)
    for env_id, r in results.items():
        role = "TRAINED" if env_id in train_envs else "untrained"
        print(f"{env_id:<6} {role:<10} {r['cond']:>4} {r['n_windows']:>5} "
              f"{r['loc_rmse_m']:>7.3f}m {r['loc_mae_m']:>7.3f}m  "
              f"{r['nlos_rate']*100:>6.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/tier2_frozen_tier1/tier2_synthetic.pt")
    p.add_argument("--train_env", type=int, default=0,
                    help="one trained env (default: 0)")
    p.add_argument("--val_env", type=int, nargs="+", default=None,
                    help="one or more untrained envs. If not specified, "
                         "uses ALL envs not in the train set (auto-detected).")
    p.add_argument("--cond", type=int, default=0,
                    help="micro condition to evaluate on: 0=low, 1=mid, 2=high NLOS")
    p.add_argument("--out_csv", default=None)
    p.add_argument("--out_fig", default=None,
                    help="output figure template. If the path contains '{env}', "
                         "one figure per env is written.")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model, cfg, items, train_envs = load_model_and_data(args.ckpt, device=args.device)

    # Auto-detect val envs if not specified
    all_envs = sorted({it[2] for it in items})
    if args.val_env is None:
        val_envs = [e for e in all_envs if e not in train_envs]
        print(f"\nauto-detected val envs: {val_envs}")
    else:
        val_envs = list(args.val_env)

    env_ids = [args.train_env] + val_envs
    results = {}
    for env_id in env_ids:
        is_train = env_id in train_envs
        print(f"\n--- env {env_id} ({'TRAINED' if is_train else 'UNTRAINED'}) "
              f"cond={args.cond} ---")
        r = run_eval(env_id, args.cond, items, model, cfg, device=args.device)
        results[env_id] = r
        print(f"  loc RMSE = {r['loc_rmse_m']:.3f} m, MAE = {r['loc_mae_m']:.3f} m "
              f"(n_windows={r['n_windows']}, nlos_rate={r['nlos_rate']*100:.1f}%)")

    print_summary(results, args.ckpt, train_envs)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w") as f:
            f.write("env_id,is_trained,cond,n_windows,loc_rmse_m,loc_mae_m,nlos_rate\n")
            for env_id, r in results.items():
                is_train = env_id in train_envs
                f.write(f"{env_id},{int(is_train)},{r['cond']},{r['n_windows']},"
                        f"{r['loc_rmse_m']:.4f},{r['loc_mae_m']:.4f},"
                        f"{r['nlos_rate']:.4f}\n")
        print(f"\nsaved CSV → {out_csv}")

    if args.out_fig:
        out_fig = Path(args.out_fig)
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        if "{env}" in args.out_fig:
            for env_id, r in results.items():
                per_env_path = str(out_fig).format(env=env_id)
                make_figure_one_env(env_id, r, train_envs, args.cond, per_env_path)
        else:
            raise NotImplementedError("combined figure for Tier 2 not implemented; "
                                       "use --out_fig with {env} placeholder")


if __name__ == "__main__":
    main()
