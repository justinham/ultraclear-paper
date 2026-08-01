"""Tier 0 SSL pretraining on the Sionna `empty_open` grid dataset.

We take a real Sionna-emulated CSV (one row = one (grid position, single
noisy trial) with 4 anchors' ToA + AoA + LOS/NLOS) and build sliding
windows of T=32 trials at a single grid position. The SSL task is
masked prediction of ToA: hide 25% of (trial, anchor) cells, predict the
original ToA from context.

This is the same SSLPretrainer / ssl_loss used in `train.py` -- only the
data loader changes (CSV instead of synthetic generator).

Usage:
    python3.11 -m scripts.train_tier0_sionna
    python3.11 -m scripts.train_tier0_sionna --csv /path/to/other.csv
    python3.11 -m scripts.train_tier0_sionna --quick
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer, ssl_loss

# Anchor UUIDs → column-name mapping (metadata-order = training order)
# This is the metadata order baked into the training script's
# `self.anchor_pos` -- see train_dynamic_graph_toa_aoa.py.
ANCHOR_COLS = ["A1", "A2", "A3", "A4"]
DEFAULT_CSV = (
    "/Users/justin/Downloads/srp_research/grid2/"
    "sionna_uwb_grid_empty_open_A1A4swapped.csv"
)


class SionnaGridWindowDataset(Dataset):
    """Sliding-window dataset over the Sionna grid CSV.

    Each window is shape (T, M, F) = (T trials, 4 anchors, F channels).
    The label y is the GT (x, y) at the middle trial (paper convention;
    for SSL we don't actually use y, but the dataset exposes it for
    downstream tiers).

    Args:
        csv_path: path to the Sionna grid CSV.
        seq_len: window length T.
        n_channels: F (1 for ToA-only, 2 for ToA+AoA).
        stride: stride between windows. With stride=32 we get
            non-overlapping windows; smaller stride = more data.
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: int = 32,
        n_channels: int = 1,
        stride: int = 32,
    ):
        df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_anchors = 4

        # build a tensor per (grid_id) of shape (n_trials, M, F+gt_pos)
        # using only valid (grid_id, trial_id) where all 4 anchors have
        # valid ToA readings.
        per_grid = []
        self.grid_ids = sorted(df.grid_id.unique())
        for gid in self.grid_ids:
            sub = df[df.grid_id == gid].sort_values("trial_id")
            n_trials = len(sub)
            # skip grids where we don't have at least seq_len trials
            if n_trials < seq_len:
                per_grid.append(None)
                continue
            # build (n_trials, M, F) feature tensor
            toa = np.stack(
                [sub[f"A{m}_measured_dist_m"].values for m in range(1, 5)],
                axis=1,
            ).astype(np.float32)  # (n_trials, 4)
            if n_channels >= 2:
                aoa = np.stack(
                    [sub[f"A{m}_aoa_azimuth_deg"].values for m in range(1, 5)],
                    axis=1,
                ).astype(np.float32)  # (n_trials, 4)
                feat = np.stack([toa, aoa], axis=-1)  # (n_trials, 4, 2)
            else:
                feat = toa[..., None]  # (n_trials, 4, 1)
            tag_xy = sub[["tag_x", "tag_y"]].values.astype(np.float32)
            per_grid.append((feat, tag_xy, n_trials))

        # build window index
        self.windows: list[tuple[int, int]] = []  # (grid_idx, start_trial)
        for gi, item in enumerate(per_grid):
            if item is None:
                continue
            _, _, n_trials = item
            for s in range(0, n_trials - seq_len + 1, stride):
                self.windows.append((gi, s))

        # store features as a single stack for fast indexing
        self._per_grid = per_grid

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        gi, s = self.windows[i]
        feat, tag_xy, _ = self._per_grid[gi]
        x = feat[s : s + self.seq_len]  # (T, M, F)
        y = tag_xy[s : s + self.seq_len]  # (T, 2) per-timestep
        # env_id / cond_id: this CSV is a single env, single LOS-only
        # condition (empty_open is 100% LOS per the README), so we use
        # constants. Keeps the (X, y, env, cond) shape contract the
        # paper's pipeline assumes.
        # No per-cell NLOS mask in Sionna CSVs; only the coarse any-NLOS
        # cond flag is available. nlos is all-False to keep the tuple shape
        # contract consistent with the synthetic 5-tuple format.
        T_, M_, _ = x.shape
        nlos = np.zeros((T_, M_), dtype=bool)
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.tensor(0, dtype=torch.long),  # env_id
            torch.tensor(0, dtype=torch.long),  # cond_id
            torch.from_numpy(nlos),            # (T, M) all-False
        )


def run_tier0(
    cfg: Config,
    train_ds: SionnaGridWindowDataset,
    val_ds: SionnaGridWindowDataset | None,
    device: str,
) -> tuple[SSLPretrainer, list[float], list[float]]:
    """Pretrain SSLPretrainer and return (model, train_losses, val_losses)."""
    model = SSLPretrainer(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ssl_lr)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.ssl_batch_size, shuffle=True,
        num_workers=0,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.ssl_batch_size, num_workers=0)
        if val_ds is not None else None
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    t0 = time.time()
    for ep in range(cfg.ssl_epochs):
        # ---- train ----
        model.train()
        total, n = 0.0, 0
        for X, _, _, _, _ in train_loader:
            X = X.to(device)
            opt.zero_grad()
            x_pred, mask, _x = model(X)
            loss = ssl_loss(x_pred, X, mask,
                            loss_on_visible=cfg.loss_on_visible,
                            visible_weight=cfg.visible_weight)
            loss.backward()
            opt.step()
            total += loss.item() * X.size(0)
            n += X.size(0)
        train_loss = total / max(1, n)
        train_losses.append(train_loss)

        # ---- val ----
        if val_loader is not None:
            model.eval()
            vtotal, vn = 0.0, 0
            with torch.no_grad():
                for X, _, _, _, _ in val_loader:
                    X = X.to(device)
                    x_pred, mask, _x = model(X)
                    vloss = ssl_loss(x_pred, X, mask,
                                     loss_on_visible=cfg.loss_on_visible,
                                     visible_weight=cfg.visible_weight)
                    vtotal += vloss.item() * X.size(0)
                    vn += X.size(0)
            val_loss = vtotal / max(1, vn)
            val_losses.append(val_loss)
            vstr = f"  val={val_loss:.4f}"
        else:
            val_loss = float("nan")
            val_losses.append(val_loss)
            vstr = ""

        if (ep + 1) % max(1, cfg.ssl_epochs // 10) == 0 or ep == 0:
            print(f"  [tier0] ep {ep+1:3d}/{cfg.ssl_epochs}  "
                  f"train={train_loss:.4f}{vstr}  "
                  f"({time.time()-t0:.1f}s)")
    print(f"  done in {time.time()-t0:.1f}s")
    return model, train_losses, val_losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="tiny run: 3 epochs, 64 windows")
    p.add_argument("--n_channels", type=int, default=1, choices=[1, 2])
    p.add_argument("--stride", type=int, default=32,
                   help="sliding-window stride; stride=seq_len gives "
                        "non-overlapping windows")
    p.add_argument("--out", default=None,
                   help="if set, save the trained model + losses here")
    p.add_argument("--epochs", type=int, default=None,
                   help="override cfg.ssl_epochs for this run")
    p.add_argument("--mask_ratio", type=float, default=None,
                   help="override cfg.mask_ratio for this run "
                        "(e.g., 0.15 for a 15%% random mask)")
    p.add_argument("--loss_on_visible", action="store_true",
                   help="include visible (unmasked) cells in the loss "
                        "(DAE-style full-reconstruction loss, in addition "
                        "to the masked loss). Use --visible_weight to "
                        "down-weight the visible term relative to masked.")
    p.add_argument("--visible_weight", type=float, default=1.0,
                   help="scale factor for the visible-cell loss term. "
                        "Only used when --loss_on_visible is set. "
                        "Use 0.1-0.3 to make visible cells a regularizer "
                        "rather than the main signal.")
    args = p.parse_args()

    cfg = Config()
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.n_channels = args.n_channels
    if args.epochs is not None:
        cfg.ssl_epochs = args.epochs
    if args.mask_ratio is not None:
        cfg.mask_ratio = args.mask_ratio
    if args.loss_on_visible:
        cfg.loss_on_visible = True
    if args.visible_weight != 1.0:
        cfg.visible_weight = args.visible_weight
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if args.quick:
        cfg.ssl_epochs = 3
        cfg.ssl_batch_size = 16

    print(f"device: {cfg.device}  seed: {cfg.seed}  n_channels: {cfg.n_channels}  "
          f"mask_ratio: {cfg.mask_ratio}  loss_on_visible={getattr(cfg, 'loss_on_visible', False)}  "
          f"visible_weight={getattr(cfg, 'visible_weight', 1.0)}")
    print(f"csv: {args.csv}")
    print("building sliding-window dataset (T=32 over trials)...")
    full_ds = SionnaGridWindowDataset(
        args.csv, seq_len=cfg.seq_len,
        n_channels=cfg.n_channels, stride=args.stride,
    )
    n = len(full_ds)
    print(f"  total windows: {n}  (seq_len={cfg.seq_len}, stride={args.stride})")

    # 80/20 split by *grid_id* so val holds out grid cells (geometry OOD)
    n_grid = len(full_ds.grid_ids)
    n_train_grid = max(1, int(0.8 * n_grid))
    train_grid_ids = set(full_ds.grid_ids[:n_train_grid])
    val_grid_ids = set(full_ds.grid_ids[n_train_grid:])
    print(f"  train grids: {len(train_grid_ids)}  val grids: {len(val_grid_ids)}")

    # rebuild two datasets (the window index references grid_idx, so
    # the simplest path is to keep a single Dataset and filter indices).
    # We'll do an in-place filter:
    train_indices = [
        i for i, (gi, _) in enumerate(full_ds.windows)
        if full_ds.grid_ids[gi] in train_grid_ids
    ]
    val_indices = [
        i for i, (gi, _) in enumerate(full_ds.windows)
        if full_ds.grid_ids[gi] in val_grid_ids
    ]
    print(f"  train windows: {len(train_indices)}  val windows: {len(val_indices)}")

    train_ds = torch.utils.data.Subset(full_ds, train_indices)
    val_ds = torch.utils.data.Subset(full_ds, val_indices)

    # quick mode: subsample
    if args.quick:
        train_ds = torch.utils.data.Subset(
            full_ds, train_indices[:64])
        val_ds = torch.utils.data.Subset(
            full_ds, val_indices[:32])

    print()
    model, train_losses, val_losses = run_tier0(cfg, train_ds, val_ds, cfg.device)

    print()
    print("=" * 60)
    print("  Per-epoch loss summary")
    print("=" * 60)
    print(f"  ep   train   val")
    for ep, (tr, va) in enumerate(zip(train_losses, val_losses), 1):
        vastr = f"{va:.4f}" if not np.isnan(va) else "  n/a"
        print(f"  {ep:3d}  {tr:.4f}  {vastr}")

    if args.out:
        out_path = Path(args.out)
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "cfg": cfg.__dict__,
            "csv": args.csv,
        }, out_path / "tier0_sionna.pt")
        print(f"\nsaved → {out_path / 'tier0_sionna.pt'}")


if __name__ == "__main__":
    main()
