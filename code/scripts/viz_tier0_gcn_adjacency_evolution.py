"""Short Tier 0 SSL run that snapshots the Dynamic GCN adjacency at every epoch.

Purpose: visualize how the learned per-channel anchor-to-anchor adjacency
of the DynamicDualEdgeGCN module evolves during pretraining. The GCN's
self.A_ch is a parameter of shape (n_channels, M, M); after sigmoid +
row-normalize (as in DynamicDualEdgeGCN.forward) it becomes the effective
edge weight between anchor i and anchor j on each channel.

Self-contained: builds the same model + data as the canonical
train_tier0_sionna_multiscene trainer but does NOT touch any existing
checkpoint or modify the canonical trainer.

Usage:
    # default 5-epoch quick run
    python3.11 -m scripts.viz_tier0_gcn_adjacency_evolution

    # longer
    python3.11 -m scripts.viz_tier0_gcn_adjacency_evolution --epochs 20

Writes:
  - figures/tier0_gcn_adj_evo/gcn_adjacency.csv
  - figures/tier0_gcn_adj_evo/gcn_adjacency_evolution.png
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer, ssl_loss
from ultraclear.asta import DynamicDualEdgeGCN
from scripts.multi_scene_dataset import build_multi_scene_dataset


# Same 4-scene mix as the canonical trainer (see
# scripts/train_tier0_sionna_multiscene.py, default branch).
DEFAULT_SCENE_PATHS = [
    "/Users/justin/Downloads/srp_research/grid2/"
    "sionna_uwb_grid_empty_open_A1A4swapped.csv",
    "/Users/justin/Downloads/srp_research/grid2/"
    "sionna_uwb_grid_custom_room_large_A1A4swapped.csv",
    "/Users/justin/Downloads/srp_research/grid2/"
    "sionna_uwb_grid_outdoor_blockages_pillars_A1A4swapped.csv",
    "/Users/justin/Downloads/srp_research/grid2/"
    "sionna_uwb_grid_indoor_blockages_pillars_A1A4swapped.csv",
]


def effective_adjacency(gcn: DynamicDualEdgeGCN) -> np.ndarray:
    """Reproduce DynamicDualEdgeGCN.forward's adjacency computation
    (sigmoid + self-loop + row-normalize) but return the matrix
    instead of aggregating features. Shape: (F, M, M)."""
    A = torch.sigmoid(gcn.A_ch.detach())            # (F, M, M)
    I = torch.eye(gcn.M, device=A.device).expand_as(A)
    A_hat = A + gcn.lambda_self * I
    A_hat = A_hat / (A_hat.sum(-1, keepdim=True) + 1e-6)
    return A_hat.cpu().numpy()


def run_short_ssl(cfg: Config, n_epochs: int, ds, device: str) -> List[dict]:
    """Run a short SSL training loop and capture per-epoch snapshots
    of the GCN adjacency.

    Snapshots are taken at the START of each epoch (so the first row
    in the CSV reflects the random initialization, before any
    training) plus one final snapshot at the end.
    """
    model = SSLPretrainer(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ssl_lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params:,}")
    print(f"  encoder.asta_blocks: {cfg.n_asta_blocks}  "
          f"M={cfg.n_anchors}  F={cfg.n_channels}")

    g = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        ds, batch_size=cfg.ssl_batch_size, shuffle=True,
        num_workers=0, generator=g, drop_last=True,
    )
    cw = torch.tensor(cfg.channel_weights, dtype=torch.float32, device=device)

    snapshots: List[dict] = []

    def capture(epoch: int) -> None:
        for blk_idx, blk in enumerate(model.encoder.blocks):
            A = effective_adjacency(blk.gcn)
            for f_ch in range(cfg.n_channels):
                for i in range(cfg.n_anchors):
                    for j in range(cfg.n_anchors):
                        snapshots.append({
                            "epoch": epoch, "block": blk_idx,
                            "channel": f_ch, "i": i, "j": j,
                            "w": float(A[f_ch, i, j]),
                        })

    t0 = time.time()
    for ep in range(n_epochs):
        capture(ep)  # snapshot at the START of each epoch (includes init)

        # Train one epoch
        model.train()
        total, n = 0.0, 0
        for X, _, _, _, _ in loader:
            X = X.to(device)
            opt.zero_grad()
            x_pred, mask, _x = model(X)
            loss = ssl_loss(x_pred, X, mask, channel_weights=cw)
            loss.backward()
            opt.step()
            total += float(loss.item()) * X.size(0)
            n += X.size(0)
        ep_loss = total / max(1, n)
        print(f"  ep {ep+1:2d}/{n_epochs}  ssl_loss={ep_loss:.4f}  "
              f"({time.time()-t0:.1f}s)")

    # Final snapshot AFTER the last training epoch
    capture(n_epochs)
    return snapshots


def plot_evolution(snapshots: List[dict], out_path: Path,
                    cfg: Config, max_cols: int = 8) -> None:
    """Plot the GCN adjacency evolution as a grid:
        rows:     n_asta_blocks x n_channels
        columns:  up to `max_cols` evenly-spaced epochs (including
                  the initial random-initialization snapshot at
                  ep=0 and the final ep)

    Spreading the columns evenly instead of showing every epoch
    keeps the figure readable when the run is long (e.g. 20
    epochs → 8 column snapshots is plenty to see the evolution).
    """
    import pandas as pd
    df = pd.DataFrame(snapshots)
    all_epochs = sorted(df["epoch"].unique())
    if len(all_epochs) > max_cols:
        # Pick `max_cols` evenly-spaced indices across all_epochs.
        idx = np.linspace(0, len(all_epochs) - 1, num=max_cols,
                          dtype=int)
        epochs = [all_epochs[i] for i in idx]
    else:
        epochs = all_epochs
    block_chans = sorted(
        {(int(b), int(c)) for b, c in zip(df["block"], df["channel"])},
        key=lambda x: (x[0], x[1]),
    )
    n_rows = len(block_chans)
    n_cols = len(epochs)
    M = cfg.n_anchors

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.5 * n_cols, 1.5 * n_rows + 0.8),  # extra top room
        constrained_layout=True,
    )
    if n_rows == 1: axes = axes[np.newaxis, :]
    if n_cols == 1: axes = axes[:, np.newaxis]

    vmin = float(df["w"].min())
    vmax = float(df["w"].max())

    for r, (b, c) in enumerate(block_chans):
        for col, ep in enumerate(epochs):
            ax = axes[r, col]
            cell = df[(df["epoch"] == ep) & (df["block"] == b) & (df["channel"] == c)]
            Mmat = np.zeros((M, M))
            for _, row in cell.iterrows():
                Mmat[int(row["i"]), int(row["j"])] = row["w"]
            im = ax.imshow(Mmat, vmin=vmin, vmax=vmax, cmap="viridis",
                            aspect="equal")
            ax.set_xticks(range(M)); ax.set_yticks(range(M))
            # Use subscript-style anchor labels and $\mathrm{ToA}$ /
            # $\mathrm{AoA}$ for the row labels so the default font
            # doesn't ligature the "ToA" / "AoA" into "TeA" / "AbA".
            ax.set_xticklabels([f"$A_{{{j+1}}}$" for j in range(M)],
                                fontsize=9)
            ax.set_yticklabels([f"$A_{{{i+1}}}$" for i in range(M)],
                                fontsize=9)
            if r == 0:
                ax.set_title(f"ep {ep}", fontsize=10, pad=4)
            if col == 0:
                ch_lbl = r"$\mathrm{ToA}$" if c == 0 else (
                    r"$\mathrm{AoA}$" if c == 1 else f"ch{c+1}"
                )
                ax.set_ylabel(f"blk {b+1}, {ch_lbl}", fontsize=10)
            for i in range(M):
                for j in range(M):
                    ax.text(j, i, f"{Mmat[i, j]:.2f}",
                             ha="center", va="center",
                             color="white" if Mmat[i, j] < (vmin + vmax) / 2 else "black",
                             fontsize=7)
    fig.suptitle(
        f"GCN adjacency evolution during Tier 0 SSL pretraining\n"
        f"(rows = ASTA block × channel; columns = epochs; "
        f"cell = {M}×{M} anchor-to-anchor adjacency, post-sigmoid+normalize)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5,
                   help="Number of SSL epochs. Default 5 for a quick look.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_cols", type=int, default=8,
                   help="Maximum number of epoch columns to show in the "
                        "figure. The script evenly subsamples the run so "
                        "long runs stay readable.")
    p.add_argument("--out_dir", default="figures/tier0_gcn_adj_evo")
    args = p.parse_args()

    cfg = Config()
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.n_channels = 2
    cfg.ssl_epochs = args.epochs
    cfg.ssl_batch_size = args.batch_size
    cfg.ssl_lr = 1e-3
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"device: {cfg.device}  seed: {cfg.seed}  "
          f"epochs: {args.epochs}  batch: {args.batch_size}")
    print(f"Building 4-scene mix (the same default as the canonical trainer)...")
    ds, scene_info = build_multi_scene_dataset(
        DEFAULT_SCENE_PATHS, seq_len=cfg.seq_len, stride=32, n_channels=2,
    )
    print(f"  total windows: {len(ds)}")

    print(f"\n=== short SSL run (capturing GCN adjacency per epoch) ===")
    snapshots = run_short_ssl(cfg, args.epochs, ds, args.device)

    out_dir = Path("/Users/justin/ultraclear-paper/code") / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "gcn_adjacency.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["epoch", "block", "channel", "i", "j", "w"]
        )
        w.writeheader()
        for s in snapshots:
            w.writerow(s)
    print(f"snapshot CSV → {csv_path}")

    fig_path = out_dir / "gcn_adjacency_evolution.png"
    plot_evolution(snapshots, fig_path, cfg, max_cols=args.max_cols)


if __name__ == "__main__":
    main()
