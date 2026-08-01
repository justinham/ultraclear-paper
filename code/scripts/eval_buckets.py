"""Re-eval a joint-tier ckpt with the 5%/25% NLOS-rate bucketing.

Usage:
  python3.11 -m scripts.eval_buckets --ckpt path/to/phase_C.pt
  python3.11 -m scripts.eval_buckets --ckpt path/to/baseline.pt

Loads the ckpt's encoder + loc_head, runs on the val set, and reports
per-bucket (low < 5%, mid < 25%, high >= 25%) mean loc err. The
bucket rules live in train_tier12_joint.eval_per_nlos_bucket.
"""
from __future__ import annotations
import argparse
import torch
import numpy as np

from ultraclear.config import make_default
from ultraclear.asta import ASTAEncoder
from ultraclear.tier1_maml import LocationHeadPerTimestep
from scripts.sionna_joint_dataset import build_sionna_joint_dataset, sionna_to_items
from scripts.train_tier12_joint import eval_per_nlos_bucket


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", default="/Users/justin/Downloads/srp_research/grid2")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = "cpu"
    else:
        device = args.device

    cfg = make_default()
    print(f"Loading ckpt {args.ckpt}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Update cfg from ckpt so the encoder's input dim matches.
    if "cfg" in ck:
        for k, v in ck["cfg"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    else:
        # The progressive-unfreeze ckpts (saved 2026-07-15) don't
        # include cfg — infer n_channels from the input_proj weight
        # shape and n_anchors from the A_ch shape.
        ms = ck.get("model_state", {})
        w = ms.get("disent.encoder.input_proj.weight")
        if w is not None and hasattr(cfg, "n_channels"):
            cfg.n_channels = int(w.shape[1])
            print(f"  inferred cfg.n_channels = {cfg.n_channels} from ckpt")
        a_ch = ms.get("disent.encoder.blocks.0.gcn.A_ch")
        if a_ch is not None and hasattr(cfg, "n_anchors"):
            cfg.n_anchors = int(a_ch.shape[1])
            print(f"  inferred cfg.n_anchors = {cfg.n_anchors} from ckpt")
    # The joint trainer saves {'model_state', 'per_cond', ...}.
    # The baseline trainer saves {'encoder_state', 'head_state', 'cfg'}.
    if "model_state" in ck:
        # Need DisentangledTrainer to get encoder + loc_head, but the
        # bare encoder+head are enough for eval.
        enc = ASTAEncoder(cfg).to(device)
        head = LocationHeadPerTimestep(cfg.d_model).to(device)
        ms = ck["model_state"]
        # Strip 'disent.encoder.' and 'loc_head.' prefixes
        enc_state = {k.replace("disent.encoder.", "", 1): v
                     for k, v in ms.items() if k.startswith("disent.encoder.")}
        head_state = {k.replace("loc_head.", "", 1): v
                      for k, v in ms.items() if k.startswith("loc_head.")}
        enc.load_state_dict(enc_state, strict=False)
        head.load_state_dict(head_state, strict=False)
    elif "encoder_state" in ck and "head_state" in ck:
        enc = ASTAEncoder(cfg).to(device)
        head = LocationHeadPerTimestep(cfg.d_model).to(device)
        enc.load_state_dict(ck["encoder_state"])
        head.load_state_dict(ck["head_state"])
    else:
        raise RuntimeError(f"unknown ckpt format: keys={list(ck.keys())}")
    enc.eval(); head.eval()

    # Build Sionna val set (same 21-env split as joint trainer)
    from pathlib import Path
    csv_paths = sorted(str(p) for p in Path(args.data_root).glob("*_A1A4swapped.csv"))
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

    # Wrap encoder + head in a tiny model that mimics DisentangledTrainer
    # for the eval function. eval_per_nlos_bucket calls model(X) and
    # destructures the 6-tuple (h, c, adv, loc, nlos, cond).
    class LocOnly(torch.nn.Module):
        def __init__(self, enc, head):
            super().__init__()
            self.enc = enc
            self.head = head
        def forward(self, x):
            loc = self.head(self.enc(x))
            # Return 6-tuple of None/empty placeholders that match
            # DisentangledTrainer's interface. eval_per_nlos_bucket only
            # uses `loc` and the bucket accumulation.
            T = x.size(1)
            B = x.size(0)
            D = self.enc.cfg.d_model if hasattr(self.enc, 'cfg') else 64
            M = 4
            return (
                torch.zeros(B, T, M, D, device=x.device),  # h
                torch.zeros(B, D, device=x.device),        # c
                torch.zeros(B, 21, device=x.device),       # adv (n_envs=21)
                loc,                                        # loc
                torch.zeros(B, T, M, device=x.device),     # nlos
                torch.zeros(B, 3, device=x.device),        # cond
            )

    model = LocOnly(enc, head)
    per_bucket = eval_per_nlos_bucket(model, val_items, device, denorm_info=denorm_info)
    print(f"  per-NLOS-bucket val loc err (5%/25% thresholds):")
    print(f"    low (< 5%):   {per_bucket[0]:.3f} m")
    print(f"    mid (< 25%):  {per_bucket[1]:.3f} m")
    print(f"    high (>=25%): {per_bucket[2]:.3f} m")
    valid = [v for v in per_bucket.values() if not np.isnan(v)]
    if valid:
        print(f"    mean:         {sum(valid)/len(valid):.3f} m")


if __name__ == "__main__":
    main()
