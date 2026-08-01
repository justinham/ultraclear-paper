"""Tier 0 SSL pretraining on a MIX of Sionna scenes (LOS + NLOS).

This is the multi-scene counterpart of `train_tier0_sionna_f2.py`.
The motivation: the previous Tier 0 ckpt was trained on `empty_open`
(100% LOS) only, and Tier 1 FOMAML with that init *diverged* on
NLOS trials (see `viz_tier1_nlos_breakdown.py`). Retraining Tier 0
on a mix of LOS + NLOS scenes should give the encoder exposure to
the propagation patterns it will see in OOD deployment, and
eliminate the NLOS failure mode at the FOMAML stage.

Uses `MultiSceneSionnaDataset` from `multi_scene_dataset.py`, which
already loads multiple CSVs, groups by (grid, trial), and exposes
(scene, cond) metadata.

Differences vs `train_tier0_sionna_f2.py`:
  - Multiple CSVs instead of one
  - Train/val split is by *scene* (held-out scenes are the val set)
    instead of by grid_id
  - Per-channel norm stats are computed across all training scenes
    (not per-scene)
  - Default scene mix is 4 train scenes (2 LOS-only + 2 NLOS-heavy)
    + 2 val scenes (the parking lots)

Usage:
  # default 4-scene mix, F=2 3:1 ToA:AOA
  python3.11 -m scripts.train_tier0_sionna_multiscene

  # custom scene list (comma-separated)
  python3.11 -m scripts.train_tier0_sionna_multiscene \\
      --scenes empty_open,outdoor_blockages_pillars,indoor_blockages_pillars,custom_room_large
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer, ssl_loss
from scripts.multi_scene_dataset import (
    build_multi_scene_dataset, MultiSceneSionnaDataset,
    SceneInfo, build_default_multi_scene,
)


def run_tier0(
    cfg: Config,
    train_ds: Subset,
    val_ds: Subset | None,
    device: str,
) -> tuple[SSLPretrainer, list[float], list[float]]:
    model = SSLPretrainer(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ssl_lr)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.ssl_batch_size, shuffle=True, num_workers=0,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.ssl_batch_size, num_workers=0)
        if val_ds is not None else None
    )

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
            x_pred, mask, _x = model(X)
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
                    x_pred, mask, _x = model(X)
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
            print(f"  [tier0 multi] ep {ep+1:3d}/{cfg.ssl_epochs}  "
                  f"train={train_loss:.4f}{vstr}  "
                  f"({time.time()-t0:.1f}s)")
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"  done in {time.time()-t0:.1f}s")
    return model, train_losses, val_losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=str, default=None,
                   help="comma-separated scene names. default = the "
                        "4-scene mix (empty_open, custom_room_large, "
                        "outdoor_blockages_pillars, indoor_blockages_pillars).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument("--channel_weights", type=float, nargs="+", default=None)
    p.add_argument("--out", default=None,
                   help="output directory for the checkpoint. default: "
                        "code/checkpoints/tier0_multiscene_f2_3to1_<N>ep/")
    args = p.parse_args()

    cfg = Config()
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.n_channels = 2
    if args.epochs is not None:
        cfg.ssl_epochs = args.epochs
    if args.mask_ratio is not None:
        cfg.mask_ratio = args.mask_ratio
    if args.channel_weights is not None:
        if len(args.channel_weights) != cfg.n_channels:
            raise ValueError(
                f"--channel_weights must have {cfg.n_channels} values"
            )
        cfg.channel_weights = tuple(args.channel_weights)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if args.quick:
        cfg.ssl_epochs = 3
        cfg.ssl_batch_size = 16

    print(f"device: {cfg.device}  seed: {cfg.seed}  n_channels: {cfg.n_channels}  "
          f"mask_ratio: {cfg.mask_ratio}  channel_weights: {list(cfg.channel_weights)}")

    # ---- Build dataset ----
    if args.scenes is not None:
        # Custom scene list: use the same paths as build_default_multi_scene
        base = "/Users/justin/Downloads/srp_research/grid2"
        scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]
        paths = [f"{base}/sionna_uwb_grid_{s}_A1A4swapped.csv" for s in scene_names]
        print(f"Building multi-scene dataset with {len(scene_names)} custom scenes:")
        ds, scene_info = build_multi_scene_dataset(
            paths, seq_len=cfg.seq_len, stride=args.stride, n_channels=2,
        )
    else:
        # Default 4-scene mix. We pass the paths explicitly so we can
        # also keep two extra parking scenes held out for the FOMAML
        # val stage.
        base = "/Users/justin/Downloads/srp_research/grid2"
        train_scenes = [
            "empty_open",                       # 100% LOS
            "custom_room_large",                # 100% LOS
            "outdoor_blockages_pillars",        # 39% any-NLOS
            "indoor_blockages_pillars",         # 42% any-NLOS
        ]
        paths = [f"{base}/sionna_uwb_grid_{s}_A1A4swapped.csv" for s in train_scenes]
        print(f"Building multi-scene dataset with {len(train_scenes)} default scenes:")
        ds, scene_info = build_multi_scene_dataset(
            paths, seq_len=cfg.seq_len, stride=args.stride, n_channels=2,
        )

    print(f"  total windows: {len(ds)}")

    # ---- Train/val split ----
    # For multi-scene training we use a small per-scene val split:
    # 10% of windows per scene go to val. The val windows come from
    # *grids not in train* per scene (to keep the spatial OOD
    # discipline we used in the single-scene trainer).
    n_grid = len({g for s_idx in range(len(scene_info))
                  for g in range(scene_info[s_idx].n_grids)})
    # We need to know which grid_ids belong to which scene. Re-derive
    # from the dataset's window index.
    n_train, n_val = 0, 0
    train_indices, val_indices = [], []
    for i, (s_idx, g, s) in enumerate(ds.windows):
        # Per-scene 80/20 by grid
        scene = ds._per_scene[s_idx]
        n_g = scene["tag_xy"].shape[0]
        train_cutoff = int(0.8 * n_g)
        if g < train_cutoff:
            train_indices.append(i)
            n_train += 1
        else:
            val_indices.append(i)
            n_val += 1
    print(f"  train windows: {n_train}  val windows: {n_val}  "
          f"({100 * n_train / max(1, n_train + n_val):.0f}% train)")

    # compute per-channel norm stats from training windows
    all_train = []
    for i in train_indices:
        s_idx, g, s = ds.windows[i]
        feat = ds._per_scene[s_idx]["feat"]   # (n_grids, n_trials, M, F)
        x = feat[g, s : s + cfg.seq_len]
        all_train.append(x.reshape(-1, cfg.n_channels))
    cat = np.concatenate(all_train, axis=0)
    norm_mean = cat.mean(axis=0).astype(np.float32)
    norm_std = (cat.std(axis=0) + 1e-6).astype(np.float32)
    print(f"  channel 0 (ToA):    mean={norm_mean[0]:.3f}  std={norm_std[0]:.3f}")
    print(f"  channel 1 (AoA az): mean={norm_mean[1]:.3f}  std={norm_std[1]:.3f}")
    ds.norm_mean = norm_mean
    ds.norm_std = norm_std

    train_ds = Subset(ds, train_indices)
    val_ds = Subset(ds, val_indices)

    if args.quick:
        train_ds = Subset(ds, train_indices[:64])
        val_ds = Subset(ds, val_indices[:32])

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

    if args.out is None:
        n_ep = cfg.ssl_epochs
        args.out = f"checkpoints/tier0_multiscene_f2_3to1_{n_ep}ep"
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "cfg": cfg.__dict__,
        "scenes": [s.name for s in scene_info],
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }, out_path / "tier0_multiscene_f2.pt")
    print(f"\nsaved → {out_path / 'tier0_multiscene_f2.pt'}")

    # Persist per-epoch loss to CSV so downstream viz scripts
    # (e.g. the all-tiers training-curve figure) can read it
    # without re-running training.
    import csv
    csv_path = out_path / "tier0_history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        for ep, (t, v) in enumerate(zip(train_losses, val_losses), 1):
            w.writerow([ep, f"{t:.6f}", f"{v:.6f}"])
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
