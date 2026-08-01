"""Shared CLI helpers for Tier 2 + joint-trainer viz scripts.

The 4 model-loading Tier 2 viz scripts (c_space, film, nlos_reliability,
loss_curves, adv_accuracy) and the comparison script (loc_comparison)
all need a small amount of plumbing to work with both the standalone
Tier 2 checkpoints (checkpoints/tier2_synthetic/) and the joint
Tier 1+2 checkpoints (checkpoints/joint/phase_*.pt).

The key insight: the model architecture is the same (DisentangledTrainer),
so any script that loads a checkpoint and forwards through the model
works as-is on either. The only differences are:
  - file paths
  - output paths (we want figures/joint/ not figures/tier2_synthetic/)
  - history CSV format (joint phase_C has different column names)

This module provides:
  - add_joint_flag(parser): adds --joint and --ckpt flags. If --joint
    is set, --ckpt defaults to the joint phase_C checkpoint and
    --out defaults to figures/joint/...
  - load_model_from_ckpt(ckpt_path, device): returns
    (model, cfg, ck_dict) ready to forward through.
  - load_history_csv(path, kind): returns a list of dicts with
    standardized keys (episode/eps, task_mse, adv_ce, nlos_ce, cond_ce,
    probe_env_acc, lambda) regardless of whether the CSV came from
    standalone Tier 2 or joint phase C.

Usage in a viz script:
    from scripts.viz_tier2_helpers import add_joint_flag, load_model_from_ckpt

    p = argparse.ArgumentParser()
    add_joint_flag(p, default_ckpt="checkpoints/tier2_synthetic/...")
    args = p.parse_args()
    model, cfg, ck = load_model_from_ckpt(args.ckpt, args.device)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from ultraclear.config import Config
from scripts.train_tier2_synthetic import DisentangledTrainer


# Default paths
TIER2_CKPT = "checkpoints/tier2_synthetic/tier2_synthetic.pt"
JOINT_C_CKPT = "checkpoints/joint/phase_C.pt"
TIER2_FIG = "figures/tier2_synthetic"
JOINT_FIG = "figures/joint"


def add_joint_flag(
    parser: argparse.ArgumentParser,
    default_ckpt: str = TIER2_CKPT,
    default_out: str | None = None,
    joint_ckpt: str = JOINT_C_CKPT,
    joint_out_prefix: str = JOINT_FIG,
) -> None:
    """Add --ckpt, --out, and --joint flags to a parser.

    --joint: switch the default checkpoint to the joint phase_C ckpt
             and the default output dir to figures/joint/. Both --ckpt
             and --out can still be overridden explicitly.
    """
    parser.add_argument("--ckpt", type=str, default=default_ckpt,
                        help=f"checkpoint to load (default: {default_ckpt})")
    if default_out is None:
        default_out = default_ckpt.replace("checkpoints/", "figures/").replace(
            "tier2_synthetic/tier2_synthetic", "tier2_synthetic"
        ).rsplit(".", 1)[0] + ".png"
    parser.add_argument("--out", type=str, default=default_out)
    parser.add_argument("--joint", action="store_true",
                        help="use the joint phase_C checkpoint and "
                             f"figures/joint/ output dir (ckpt={joint_ckpt})")
    parser.add_argument("--device", type=str, default="cpu")


def resolve_paths(args) -> tuple[str, str]:
    """Apply --joint flag to args.ckpt and args.out, return (ckpt, out)."""
    ckpt = args.ckpt
    out = args.out
    if getattr(args, "joint", False):
        ckpt = args.joint_ckpt if hasattr(args, "joint_ckpt") else JOINT_C_CKPT
        if "tier2_synthetic" in out and "joint" not in out:
            out = out.replace("tier2_synthetic", "joint").replace(
                "figures/joint", "figures/tier2_synthetic"  # safety: don't double-replace
            )
        # Make sure the output dir exists
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    return ckpt, out


def load_model_from_ckpt(
    ckpt_path: str,
    device: str = "cpu",
) -> tuple[DisentangledTrainer, Config, dict]:
    """Load a DisentangledTrainer from a checkpoint file.

    Works for both standalone Tier 2 (train_tier2_synthetic.py output)
    and joint trainer (train_tier12_joint.py phase_B or phase_C output),
    because they save the same model architecture.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ck["cfg"]
    cfg = Config(**{k: v for k, v in cfg_dict.items()
                    if k in Config.__dataclass_fields__})
    cfg.device = device
    n_envs = ck["n_envs"]
    # Joint phase_B and phase_C save the same n_envs, but standalone
    # Tier 2 also saves n_envs. Be defensive: if missing, default to 8.
    model = DisentangledTrainer(cfg, n_envs=n_envs).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg, ck


def load_history_csv(path: str, kind: str = "tier2") -> list[dict]:
    """Load a history CSV and return rows as dicts.

    kind="tier2": standalone Tier 2 CSV. Columns:
        epoch, lambda, train_task_mse, train_adv_ce, train_nlos_ce,
        train_cond_ce, val_task_mse, probe_env_acc, probe_chance, probe_n_envs

    kind="joint_B": joint phase B CSV. Columns:
        epoch, lambda, train_task_mse, train_adv_ce, train_nlos_ce, train_cond_ce

    kind="joint_C": joint phase C CSV. Columns:
        episode, lambda, meta_loss, task_loss_disent, adv_ce, nlos_ce,
        cond_ce, feat_loss, probe_env_acc

    Returns rows in a standardized form with keys:
        ep, task_mse, adv_ce, nlos_ce, cond_ce, probe_env_acc, lam
    Missing values are NaN.
    """
    import csv
    import math
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    row[k] = float(v) if v not in ("", "NA", "nan", None) else float("nan")
                except (ValueError, TypeError):
                    pass
            std = {}
            std["ep"] = row.get("epoch", row.get("episode", float("nan")))
            if kind == "joint_C":
                std["task_mse"] = row.get("meta_loss", float("nan"))  # MAML query loss
                std["task_disent"] = row.get("task_loss_disent", float("nan"))
                std["feat_loss"] = row.get("feat_loss", float("nan"))
            else:
                std["task_mse"] = row.get("train_task_mse", float("nan"))
                std["task_disent"] = float("nan")
                std["feat_loss"] = float("nan")
            std["val_task_mse"] = row.get("val_task_mse", float("nan"))
            std["adv_ce"] = row.get("train_adv_ce", row.get("adv_ce", float("nan")))
            std["nlos_ce"] = row.get("train_nlos_ce", row.get("nlos_ce", float("nan")))
            std["cond_ce"] = row.get("train_cond_ce", row.get("cond_ce", float("nan")))
            std["probe_env_acc"] = row.get("probe_env_acc", float("nan"))
            std["lam"] = row.get("lambda", float("nan"))
            rows.append(std)
    return rows
