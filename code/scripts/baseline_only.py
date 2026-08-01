"""Quick standalone baseline: ASTAEncoder + LocationHeadPerTimestep
trained from scratch with plain supervised MSE. Used as a comparison
point in the 3-way K-shot ablation (joint Tier 1+2 vs Tier 1 only vs
no-Tier-1 baseline).

This is a thin wrapper around `run_baseline_for_joint` that builds the
dataset itself (so we don't have to fake a Phase A ckpt to get past
the trainer's gating). Keeps the eval logic identical to the joint
trainer.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultraclear.config import Config
from ultraclear.asta import ASTAEncoder
from ultraclear.tier1_maml import LocationHeadPerTimestep
from scripts.sionna_joint_dataset import build_sionna_joint_dataset, sionna_to_items
from scripts.train_tier12_joint import TensorUWB


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", required=True)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_ckpt", default=None)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config()
    cfg.n_channels = 2   # ToA + AoA, matching the joint trainer
    cfg.device = args.device

    csv_paths = sorted(str(p) for p in Path(args.csv_dir).glob("*_A1A4swapped.csv"))
    if not csv_paths:
        sys.exit(f"no CSVs in {args.csv_dir}")

    print("building dataset...")
    ds, scene_info = build_sionna_joint_dataset(
        csv_paths=csv_paths, n_channels=cfg.n_channels,
        seq_len=cfg.seq_len, stride=cfg.seq_len,
    )
    env_ids = [info.env_id for info in scene_info]
    train_set = set(env_ids[:10])
    val_set = set(env_ids[10:]) - {14}
    train_items, val_items, scene_info, denorm_info = sionna_to_items(
        ds, train_env_ids=train_set, val_env_ids=val_set,
    )
    train_env_ids = sorted({it[2] for it in train_items})
    ds.compute_norm_stats(set(train_env_ids))
    train_items, val_items, scene_info, denorm_info = sionna_to_items(
        ds, train_env_ids=train_set, val_env_ids=val_set,
    )
    n_envs = len(scene_info)
    print(f"  n_envs={n_envs}  train_items={len(train_items)}  val_items={len(val_items)}")

    enc = ASTAEncoder(cfg).to(args.device)
    head = LocationHeadPerTimestep(cfg.d_model).to(args.device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    loader = DataLoader(TensorUWB(train_items), batch_size=32, shuffle=True)

    print(f"  Training baseline (no Tier 2) for {args.n_epochs} epochs...")
    enc.train(); head.train()
    for ep in range(args.n_epochs):
        total, n = 0.0, 0
        for X, y, _env, _c, _nlos in loader:
            X = X.to(args.device); y = y.to(args.device)
            opt.zero_grad()
            pred = head(enc(X))
            loss = TF.mse_loss(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * X.size(0)
            n += X.size(0)
        if (ep + 1) % max(1, args.n_epochs // 5) == 0 or ep == 0:
            print(f"    [baseline] ep {ep+1}/{args.n_epochs}  loc_mse = {total/n:.4f}")

    enc.eval(); head.eval()
    val_loader = DataLoader(TensorUWB(val_items), batch_size=64)
    errs_per_cond = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for X, y, env, _c, nlos in val_loader:
            X = X.to(args.device); y = y.to(args.device); env = env.to(args.device)
            pred = head(enc(X))
            pred_orig = torch.zeros_like(pred)
            y_orig = torch.zeros_like(y)
            for i in range(len(env)):
                eid = int(env[i].item())
                mean, std = denorm_info[eid]
                pred_orig[i] = pred[i] * torch.from_numpy(std).to(args.device) + torch.from_numpy(mean).to(args.device)
                y_orig[i] = y[i] * torch.from_numpy(std).to(args.device) + torch.from_numpy(mean).to(args.device)
            pred, y = pred_orig, y_orig
            err = torch.linalg.norm(pred - y, dim=-1).cpu().numpy()
            nlos_rate = nlos.float().mean(dim=(1, 2)).numpy()
            for rate, e in zip(nlos_rate.tolist(), err.tolist()):
                if rate < 0.3:
                    errs_per_cond[0].append(e)
                elif rate < 0.7:
                    errs_per_cond[1].append(e)
                else:
                    errs_per_cond[2].append(e)
    per_cond = {ci: float(np.mean(es)) for ci, es in errs_per_cond.items()}
    print(f"\n  baseline per-cond val loc err: {per_cond}")

    if args.out_ckpt:
        torch.save({
            "encoder_state": enc.state_dict(),
            "head_state": head.state_dict(),
            "cfg": cfg.__dict__,
        }, args.out_ckpt)
        print(f"  baseline ckpt → {args.out_ckpt}")
    if args.out_csv:
        import csv as _csv
        with open(args.out_csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["cond_id", "baseline_bucket_err"])
            for ci in sorted(per_cond.keys()):
                w.writerow([ci, per_cond[ci]])
        print(f"  baseline CSV → {args.out_csv}")


if __name__ == "__main__":
    main()
