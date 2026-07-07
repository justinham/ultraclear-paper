"""Loss-curve plot for a single SSL checkpoint, with train/val on
linear and log scales + reference lines.

Usage:
    python3.11 -m scripts.plot_loss_curve --ckpt <ckpt.pt> [--out path.png]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=False)
    tr = ckpt["train_losses"]
    va = np.array(ckpt["val_losses"])
    ep = np.arange(1, len(tr) + 1)

    cfg = ckpt.get("cfg", {})
    out = Path(args.out) if args.out else Path(args.ckpt).parent / "loss_curve.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(ep, tr, "-o", color="C0", label="train", markersize=4)
    ax1.plot(ep, np.where(np.isnan(va), np.nan, va),
              "-o", color="C1", label="val", markersize=4)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("MSE on masked cells")
    ax1.set_title("Train/val loss (linear)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(ep, tr, "-o", color="C0", label="train", markersize=4)
    ax2.semilogy(ep, np.where(np.isnan(va), np.nan, va),
                  "-o", color="C1", label="val", markersize=4)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("MSE (log scale)")
    ax2.set_title("Train/val loss (log y)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    title = args.title or f"Tier 0 SSL — loss curve  (F={cfg.get('n_channels', 1)}, mask={cfg.get('mask_ratio', 0.25):.2f})"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")
    print(f"  final train MSE: {tr[-1]:.4f}  (RMSE = {np.sqrt(tr[-1]):.4f})")
    if not np.isnan(va[-1]):
        print(f"  final val   MSE: {va[-1]:.4f}  (RMSE = {np.sqrt(va[-1]):.4f})")


if __name__ == "__main__":
    main()
