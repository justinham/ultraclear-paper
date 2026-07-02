"""Shared config for UltraClear 4-tier implementation.

All four tiers share this hyperparameter object so that shapes line up.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ---- data shape ----
    n_anchors: int = 4            # M (number of anchors)
    n_channels: int = 1           # 1 for ToA-only, 2 for ToA+AoA
    seq_len: int = 32             # T (time window length)
    mask_ratio: float = 0.25      # fraction of (anchor, time) cells masked in Tier 0

    # ---- ASTA encoder ----
    d_model: int = 64             # hidden width inside ASTA
    n_asta_blocks: int = 2        # how many ASTA blocks to stack
    n_attention_heads: int = 4
    tcn_kernel: int = 3           # causal TCN kernel
    anchor_topo_dropout: float = 0.0

    # ---- micro disentanglement ----
    micro_dim: int = 32           # dimension of condition vector c
    film_hidden: int = 64         # FiLM MLP hidden
    adv_hidden: int = 64          # adversary hidden

    # ---- training ----
    device: str = "cpu"           # change to "cuda" if you have a GPU
    seed: int = 0

    # Tier 0
    ssl_lr: float = 1e-3
    ssl_epochs: int = 30
    ssl_batch_size: int = 32

    # Tier 1 (MAML)
    maml_inner_lr: float = 1e-2
    maml_inner_steps: int = 5
    maml_outer_lr: float = 1e-3
    maml_episodes: int = 200      # outer iterations
    n_way: int = 4                # sample N envs per episode
    k_shot: int = 10              # support size per env
    q_query: int = 20             # query size per env

    # Tier 2 (adversarial)
    adv_lr: float = 1e-3
    adv_lambda: float = 1.0       # weight of adv loss (grad reversal strength)
    finetune_epochs: int = 50
    finetune_lr: float = 1e-3
    finetune_batch_size: int = 16

    # synthetic data
    n_synthetic_envs: int = 8     # how many macro environments to simulate
    samples_per_env: int = 200    # traces per environment
    n_micro_conditions: int = 3   # distinct NLOS/multi-path severities


def make_default() -> Config:
    return Config()
