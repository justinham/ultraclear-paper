"""MAML-only baseline: no SSL pretraining, just the FOMAML outer loop.

Trains an ASTA encoder + LocationHead from scratch with the MAML
outer loop on the Sionna 10/11 train/val split. This is the
"no SSL, with MAML" baseline that complements the joint trainer's
Tier 1 (SSL + MAML).

Why this baseline:
  - Random init baseline: no SSL, no MAML. Trained with supervised
    MSE on the location task.
  - MAML-only baseline (this script): no SSL, with MAML. Trained
    with the FOMAML outer loop on the location task.
  - Tier 1 (joint trainer Phase A): SSL + MAML.
  - Joint Tier 1+2: SSL + disentangle + MAML.

The MAML-only baseline answers: "does the MAML outer loop alone
help, without SSL?" If yes, then the MAML machinery is doing real
work and the SSL addition can be attributed separately.

Usage:
    python3.11 -m scripts.train_maml_only_sionna \\
        --out checkpoints/maml_only.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from ultraclear.config import Config
from ultraclear.tier1_maml import MAMLCore, maml_train
from ultraclear.asta import ASTAEncoder
from scripts.train_tier12_joint import LocationHead
from scripts.sionna_joint_dataset import (
    build_sionna_joint_dataset, sionna_to_items,
)
from ultraclear.data import TensorUWB


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="checkpoints/maml_only.pt")
    p.add_argument("--data", default="sionna", choices=["sionna", "synthetic"])
    p.add_argument("--csv_dir", default=None)
    p.add_argument("--episodes", type=int, default=200,
                   help="number of FOMAML outer-loop episodes")
    p.add_argument("--inner_lr", type=float, default=None,
                   help="override cfg.maml_inner_lr (default: use cfg)")
    p.add_argument("--outer_lr", type=float, default=None,
                   help="override cfg.maml_outer_lr (default: use cfg)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sionna_train_envs", type=int, nargs="+", default=None)
    p.add_argument("--sionna_val_envs", type=int, nargs="+", default=None)
    args = p.parse_args()

    device = resolve_device(args.device)
    print(f"device: {device}, seed: {args.seed}")

    cfg = Config()
    cfg.n_channels = 1
    cfg.device = device
    cfg.seed = args.seed
    if args.inner_lr is not None:
        cfg.maml_inner_lr = args.inner_lr
    if args.outer_lr is not None:
        cfg.maml_outer_lr = args.outer_lr
    cfg.maml_episodes = args.episodes
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Build dataset (Sionna 10/11 split)
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
        train_set = set(args.sionna_train_envs) if args.sionna_train_envs else set(env_ids[:10])
        val_set = set(args.sionna_val_envs) if args.sionna_val_envs else set(env_ids[10:]) - {14}
        train_items, val_items, scene_info, denorm_info = sionna_to_items(
            ds, train_env_ids=train_set, val_env_ids=val_set,
        )
        ds.compute_norm_stats(train_set)
        train_items, val_items, scene_info, denorm_info = sionna_to_items(
            ds, train_env_ids=train_set, val_env_ids=val_set,
        )
        print(f"data: sionna, train_envs={sorted(train_set)}, val_envs={sorted(val_set)}")
    else:
        from ultraclear.data import build_synthetic_dataset
        all_items = build_synthetic_dataset(cfg, seed=args.seed)
        env_ids = sorted({it[2] for it in all_items})
        n_train = int(0.8 * len(env_ids))
        train_set = set(env_ids[:n_train])
        val_set = set(env_ids) - train_set
        train_items = [it for it in all_items if it[2] in train_set]
        val_items = [it for it in all_items if it[2] not in train_set]
        denorm_info = {eid: None for eid in val_set}
        print(f"data: synthetic")

    # FOMAML outer loop (no SSL, just MAML)
    print(f"  MAML outer-loop with {args.episodes} episodes, "
          f"inner_lr={cfg.maml_inner_lr}, outer_lr={cfg.maml_outer_lr}")
    t0 = time.time()
    core = maml_train(cfg, train_items, device=device, log_every=max(1, args.episodes // 10))
    print(f"  MAML training done in {time.time()-t0:.1f}s")

    # Save the encoder + head in the same format as Phase A / baseline
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Extract the encoder (which is core.encoder) and the head (core.head)
    torch.save({
        "encoder_state": core.encoder.state_dict(),
        "head_state": core.head.state_dict(),
        "cfg": cfg.__dict__,
        "n_episodes": args.episodes,
    }, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
