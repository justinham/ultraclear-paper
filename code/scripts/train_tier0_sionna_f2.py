"""Tier 0 SSL pretraining on the Sionna `empty_open` grid dataset
**with F=2 (ToA + horizontal AoA azimuth)**.

This is a sibling of `train_tier0_sionna.py` for the 2-channel case.
The key differences:
  - Sliding windows are (T, M, F=2) = (32, 4, [ToA_m, AoA_azimuth_deg]).
  - Per-channel normalization: z-score each channel using stats
    computed on the training set (mean/std per (channel)), so the
    model doesn't have to learn wildly different output scales for
    ToA (0-15 m) vs AoA (-180 to 180 deg).
  - Mask is JOINT across channels: if (t, m) is masked, both ToA and
    AoA are zeroed at that cell. The model can't cheat by using the
    visible AoA at the same (t, m) to predict the masked ToA there.
  - Normalization stats are saved in the checkpoint so inference /
    viz scripts can denormalize predictions back to original units.

Usage:
    python3.11 -m scripts.train_tier0_sionna_f2
    python3.11 -m scripts.train_tier0_sionna_f2 --csv /path/to/empty_open.csv
    python3.11 -m scripts.train_tier0_sionna_f2 --quick
    python3.11 -m scripts.train_tier0_sionna_f2 --mask_ratio 0.25
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

# Anchor column layout (metadata-order = training order)
ANCHOR_COLS = ["A1", "A2", "A3", "A4"]
DEFAULT_CSV = (
    "/Users/justin/ultraclear-paper/code/data/empty_open.csv"
)
# Channel 0 = ToA (meters), Channel 1 = horizontal AoA (degrees).
CHANNEL_NAMES = ["ToA (m)", "AoA azimuth (deg)"]


class SionnaGridWindowF2Dataset(Dataset):
    """Sliding-window dataset over the Sionna grid CSV, F=2 channels.

    Each window is shape (T, M, F=2) = (32, 4, 2). The label y is the
    GT (x, y) at the middle trial. We also expose env_id / cond_id
    constants (single env, single LOS condition) for the (X, y, e, c)
    shape contract downstream tiers expect.

    Returns NORMALIZED windows; pass `denorm` to recover the original
    units (handy for viz). The normalization stats are also exposed
    as `self.norm_mean` and `self.norm_std` (each shape (F,)) and
    are computed from the training grid_ids only (so val is truly
    held-out).
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: int = 32,
        stride: int = 32,
    ):
        df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.n_anchors = 4
        self.n_channels = 2

        per_grid = []
        self.grid_ids = sorted(df.grid_id.unique())
        for gid in self.grid_ids:
            sub = df[df.grid_id == gid].sort_values("trial_id")
            n_trials = len(sub)
            if n_trials < seq_len:
                per_grid.append(None)
                continue
            # channel 0: ToA (meters)
            toa = np.stack(
                [sub[f"A{m}_measured_dist_m"].values for m in range(1, 5)],
                axis=1,
            ).astype(np.float32)  # (n_trials, 4)
            # channel 1: horizontal AoA azimuth (degrees, wrap-around ignored)
            aoa = np.stack(
                [sub[f"A{m}_aoa_azimuth_deg"].values for m in range(1, 5)],
                axis=1,
            ).astype(np.float32)  # (n_trials, 4)
            feat = np.stack([toa, aoa], axis=-1)  # (n_trials, 4, 2)
            tag_xy = sub[["tag_x", "tag_y"]].values.astype(np.float32)
            per_grid.append((feat, tag_xy, n_trials))

        # build window index
        self.windows: list[tuple[int, int]] = []
        for gi, item in enumerate(per_grid):
            if item is None:
                continue
            _, _, n_trials = item
            for s in range(0, n_trials - seq_len + 1, stride):
                self.windows.append((gi, s))
        self._per_grid = per_grid

    def compute_norm_stats(self, train_grid_ids: set[int]):
        """Compute per-channel mean/std from the *training* grid cells.
        Run this *before* wrapping in a Subset so that the train/val
        split is consistent with the rest of the pipeline.
        """
        all_train = []
        for gi, item in enumerate(self._per_grid):
            if item is None:
                continue
            if self.grid_ids[gi] not in train_grid_ids:
                continue
            feat, _, _ = item  # (n_trials, M, F=2)
            all_train.append(feat.reshape(-1, self.n_channels))
        if not all_train:
            raise RuntimeError("no training windows for norm stats")
        cat = np.concatenate(all_train, axis=0)  # (n_train_cells, 2)
        self.norm_mean = cat.mean(axis=0).astype(np.float32)
        self.norm_std = (cat.std(axis=0) + 1e-6).astype(np.float32)
        return self.norm_mean, self.norm_std

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """x: (..., F=2) -> normalized."""
        return (x - self.norm_mean) / self.norm_std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """x: (..., F=2) -> original units."""
        return x * self.norm_std + self.norm_mean

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        gi, s = self.windows[i]
        feat, tag_xy, _ = self._per_grid[gi]
        x = feat[s : s + self.seq_len]                              # (T, M, 2)
        if hasattr(self, "norm_mean"):
            x = self.normalize(x)
        y = tag_xy[s + self.seq_len // 2]                          # (2,)
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.tensor(0, dtype=torch.long),  # env_id
            torch.tensor(0, dtype=torch.long),  # cond_id
            torch.zeros(x.shape[:2], dtype=torch.bool),  # (T, M) all-False
        )


def run_tier0(
    cfg: Config,
    train_ds: SionnaGridWindowF2Dataset,
    val_ds: SionnaGridWindowF2Dataset | None,
    device: str,
) -> tuple[SSLPretrainer, list[float], list[float]]:
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

    # per-channel loss weights (e.g., [3.0, 1.0] upweights ToA over AoA
    # because ToA is the geometrically meaningful channel that flows
    # into the downstream location head). Default: equal weight.
    cw = torch.tensor(cfg.channel_weights, dtype=torch.float32)

    train_losses: list[float] = []
    val_losses: list[float] = []
    t0 = time.time()
    for ep in range(cfg.ssl_epochs):
        model.train()
        total, n = 0.0, 0
        for X, _, _, _, _ in train_loader:
            X = X.to(device)
            opt.zero_grad()
            x_pred, mask = model(X)
            # masked weighted MSE, normalized units. After z-score the
            # per-channel scale is matched, so weights here are pure
            # *task* preferences (not unit-balancing hacks).
            loss = ssl_loss(x_pred, X, mask, channel_weights=cw)
            loss.backward()
            opt.step()
            total += loss.item() * X.size(0)
            n += X.size(0)
        train_loss = total / max(1, n)
        train_losses.append(train_loss)

        if val_loader is not None:
            model.eval()
            vtotal, vn = 0.0, 0
            with torch.no_grad():
                for X, _, _, _, _ in val_loader:
                    X = X.to(device)
                    x_pred, mask = model(X)
                    vloss = ssl_loss(x_pred, X, mask, channel_weights=cw)
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
            print(f"  [tier0 f2] ep {ep+1:3d}/{cfg.ssl_epochs}  "
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
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument("--channel_weights", type=float, nargs="+", default=None,
                   help="per-channel loss weights, e.g. '3.0 1.0' to "
                        "upweight ToA over AoA by 3x. Default: equal "
                        "weights (1.0 for each channel).")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = Config()
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.n_channels = 2  # F=2 (ToA + AoA)
    if args.epochs is not None:
        cfg.ssl_epochs = args.epochs
    if args.mask_ratio is not None:
        cfg.mask_ratio = args.mask_ratio
    if args.channel_weights is not None:
        if len(args.channel_weights) != cfg.n_channels:
            raise ValueError(
                f"--channel_weights must have {cfg.n_channels} values, "
                f"got {len(args.channel_weights)}"
            )
        cfg.channel_weights = tuple(args.channel_weights)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if args.quick:
        cfg.ssl_epochs = 3
        cfg.ssl_batch_size = 16

    print(f"device: {cfg.device}  seed: {cfg.seed}  n_channels: {cfg.n_channels}  "
          f"mask_ratio: {cfg.mask_ratio}  channel_weights: {list(cfg.channel_weights)}")
    print(f"csv: {args.csv}")
    print("building sliding-window dataset (T=32 over trials, F=2)...")
    full_ds = SionnaGridWindowF2Dataset(
        args.csv, seq_len=cfg.seq_len, stride=args.stride,
    )
    n = len(full_ds)
    print(f"  total windows: {n}  (seq_len={cfg.seq_len}, stride={args.stride})")

    n_grid = len(full_ds.grid_ids)
    n_train_grid = max(1, int(0.8 * n_grid))
    train_grid_ids = set(full_ds.grid_ids[:n_train_grid])
    val_grid_ids = set(full_ds.grid_ids[n_train_grid:])
    print(f"  train grids: {len(train_grid_ids)}  val grids: {len(val_grid_ids)}")

    # compute per-channel norm stats from the training grids only
    norm_mean, norm_std = full_ds.compute_norm_stats(train_grid_ids)
    print(f"  channel 0 (ToA):    mean={norm_mean[0]:.3f}  std={norm_std[0]:.3f}")
    print(f"  channel 1 (AoA az): mean={norm_mean[1]:.3f}  std={norm_std[1]:.3f}")

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

    if args.quick:
        train_ds = torch.utils.data.Subset(full_ds, train_indices[:64])
        val_ds = torch.utils.data.Subset(full_ds, val_indices[:32])

    print()
    model, train_losses, val_losses = run_tier0(cfg, train_ds, val_ds, cfg.device)

    print()
    print("=" * 60)
    print("  Per-epoch loss summary (z-score normalized units)")
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
            "norm_mean": full_ds.norm_mean,
            "norm_std": full_ds.norm_std,
        }, out_path / "tier0_sionna_f2.pt")
        print(f"\nsaved → {out_path / 'tier0_sionna_f2.pt'}")


if __name__ == "__main__":
    main()
