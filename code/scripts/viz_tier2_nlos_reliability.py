"""Tier 2 — NLOS head reliability: per-cond calibration + accuracy.

Sanity check that the NLOS head is actually learning the micro signal
(since c is supposed to be informative about the micro condition,
the NLOS head attached to h' should also be).

Reports:
  - Per-cond NLOS rate (ground truth) — for context
  - Per-cond NLOS accuracy (predicted vs ground truth)
  - Per-cond predicted NLOS rate (calibration check)
  - Overall: confusion-matrix-style counts

Usage:
  python -m scripts.viz_tier2_nlos_reliability
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ultraclear.config import Config
from ultraclear.data import build_synthetic_dataset, TensorUWB
from scripts.train_tier2_synthetic import DisentangledTrainer


@torch.no_grad()
def collect_predictions(model, loader, device):
    nlos_logits_buf, nlos_buf, cond_buf = [], [], []
    for X, _y, _env, cond, nlos in loader:
        X = X.to(device)
        _h, _c, _adv, _loc, nlos_logits, _cond = model(X)
        nlos_logits_buf.append(nlos_logits.cpu())
        nlos_buf.append(nlos)
        cond_buf.append(cond)
    return (torch.cat(nlos_logits_buf, 0),
            torch.cat(nlos_buf, 0).flatten().numpy(),
            torch.cat(cond_buf, 0).numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/tier2_synthetic/tier2_synthetic.pt")
    p.add_argument("--out", type=str, default="figures/tier2_synthetic/nlos_reliability.png")
    p.add_argument("--max_points", type=int, default=800)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    print(f"loading {args.ckpt}...")
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__dataclass_fields__})
    cfg.device = args.device
    n_envs = ck["n_envs"]
    model = DisentangledTrainer(cfg, n_envs=n_envs).to(args.device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    print("building data...")
    all_items = build_synthetic_dataset(cfg, seed=cfg.seed)
    if len(all_items) > args.max_points:
        rng = np.random.default_rng(1)
        idx = rng.permutation(len(all_items))[:args.max_points]
        items = [all_items[i] for i in idx]
    else:
        items = all_items
    loader = DataLoader(TensorUWB(items), batch_size=64)
    nlos_logits, nlos, conds = collect_predictions(model, loader, args.device)
    probs = torch.sigmoid(nlos_logits).numpy().flatten()
    preds = (probs > 0.5).astype(int)
    n_cells = len(nlos)
    print(f"evaluated {n_cells} (window × time × anchor) cells")

    # Per-cond metrics
    rows = []
    for ci in range(3):
        mask = np.repeat(conds == ci, cfg.seq_len * cfg.n_anchors)
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == nlos[mask]).mean()
        gt_rate = nlos[mask].mean()
        pred_rate = preds[mask].mean()
        rows.append({"cond": ci, "n": int(mask.sum()), "acc": acc,
                     "gt_rate": gt_rate, "pred_rate": pred_rate,
                     "mean_prob": probs[mask].mean()})
    for r in rows:
        print(f"  cond {r['cond']}: n={r['n']}  acc={r['acc']:.3f}  "
              f"gt_NLOS_rate={r['gt_rate']:.3f}  pred_NLOS_rate={r['pred_rate']:.3f}  "
              f"mean_pred_prob={r['mean_prob']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    conds_arr = np.array([r["cond"] for r in rows])
    accs = np.array([r["acc"] for r in rows])
    gt_rates = np.array([r["gt_rate"] for r in rows])
    pred_rates = np.array([r["pred_rate"] for r in rows])

    x = np.arange(len(rows))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width/2, gt_rates, width, label="ground truth NLOS rate", color="C0", alpha=0.8)
    ax.bar(x + width/2, pred_rates, width, label="predicted NLOS rate", color="C3", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"cond {r['cond']}" for r in rows])
    ax.set_ylabel("fraction NLOS")
    ax.set_title("NLOS rate by micro condition\n(GT vs predicted)")
    ax.set_ylim(0, max(gt_rates.max(), pred_rates.max()) * 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x, accs, color="C2", alpha=0.8)
    ax.axhline(0.5, color="grey", linestyle="--", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"cond {r['cond']}" for r in rows])
    ax.set_ylabel("per-cell accuracy")
    ax.set_title("NLOS head accuracy by micro condition")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate acc values
    for xi, a in zip(x, accs):
        ax.text(xi, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)

    fig.suptitle("Tier 2 — NLOS head reliability (sanity check: micro signal preserved)", fontsize=14)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
