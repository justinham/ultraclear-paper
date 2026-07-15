"""Synthetic UWB-like data for the 4-tier pipeline.

Each *environment* is a macro domain (e.g. "office", "garage") with:
  - a random anchor layout in 2D
  - a random tag-trajectory generator (a slow random walk)

Each *trace* within an environment is one tag path; we sample distances
from each anchor to each tag position, then add measurement noise whose
*intensity* depends on a *micro condition* (NLOS / multipath severity).

Output per sample: a tensor of shape (T, M, F) where
  T  = sequence length (time steps)
  M  = number of anchors
  F  = number of channels (1 for ToA only, 2 for ToA + AoA)
The y label is the 2D (x, y) tag position at *each* timestep, i.e. the
full (T, 2) trajectory. The model is a per-timestep location head that
outputs (B, T, 2) — one (x, y) per timestep.
We also expose env_id (macro) and condition_id (micro) for meta-learning.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config


@dataclass
class SyntheticEnv:
    env_id: int
    anchor_xy: np.ndarray          # (M, 2)
    tag_init: np.ndarray           # (2,)
    walk_scale: float              # how fast the tag moves per step
    nlos_bias: np.ndarray          # (M,) per-anchor positive bias under NLOS


def _make_random_env(rng: np.random.Generator, env_id: int, M: int) -> SyntheticEnv:
    # Anchors at the corners of a random axis-aligned rectangle in [-5, 5]^2
    xs = rng.uniform(-5.0, 5.0, size=M)
    ys = rng.uniform(-5.0, 5.0, size=M)
    anchor_xy = np.stack([xs, ys], axis=1).astype(np.float32)
    tag_init = rng.uniform(-3.0, 3.0, size=2).astype(np.float32)
    walk_scale = float(rng.uniform(0.1, 0.4))
    nlos_bias = rng.uniform(0.0, 0.5, size=M).astype(np.float32)
    return SyntheticEnv(env_id, anchor_xy, tag_init, walk_scale, nlos_bias)


def _micro_params(condition_id: int) -> Tuple[float, float, float]:
    """Return (nlos_rate, nlos_strength, multipath_std) for a micro condition.

    Micro = blockage severity. NLOS event rate and NLOS bias magnitude
    co-vary (more frequent -> stronger). Multipath std is a small
    thermal-noise floor.

    Rates follow the "person walking by" dynamic-noise regime:
      cond 0: rare, mild  (3% NLOS, 0.3m bias) -- 1 anchor briefly
      cond 1: occasional  (10% NLOS, 0.4m bias) -- 1 anchor sometimes
      cond 2: frequent    (20% NLOS, 0.5m bias) -- 1 anchor often,
                                                  2 anchors sometimes
    These match the simple_baseline's gen_simple_synthetic.py scene
    regime so the per-cond buckets across scripts are consistent.
    """
    table = {
        0: (0.03, 0.3, 0.02),   # rare, mild blockages
        1: (0.10, 0.4, 0.02),   # occasional, moderate
        2: (0.20, 0.5, 0.02),   # frequent, strong (still realistic)
    }
    return table[condition_id % 3]


def simulate_trace(
    env: SyntheticEnv,
    condition_id: int,
    T: int,
    rng: np.random.Generator,
    n_channels: int,
) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    """Return (X[T,M,F], y[2], env_id, condition_id, nlos[T,M] bool).

    nlos is the per-anchor, per-timestep ground-truth NLOS mask — used as
    the NLOS head target and to verify the micro-encoder learned a
    condition vector that *is* informative about the micro state.
    """
    M = env.anchor_xy.shape[0]
    nlos_rate, nlos_strength, mp_std = _micro_params(condition_id)

    # random walk
    pos = np.zeros((T, 2), dtype=np.float32)
    pos[0] = env.tag_init + rng.normal(0, env.walk_scale, size=2).astype(np.float32)
    for t in range(1, T):
        pos[t] = pos[t - 1] + rng.normal(0, env.walk_scale, size=2).astype(np.float32)

    # distances
    diff = pos[:, None, :] - env.anchor_xy[None, :, :]   # (T, M, 2)
    dist = np.linalg.norm(diff, axis=-1)                 # (T, M)

    # NLOS events: per-anchor, per-timestep bernoulli
    nlos = (rng.random(size=(T, M)) < nlos_rate)         # (T, M) bool
    bias = nlos.astype(np.float32) * (env.nlos_bias[None, :] * nlos_strength)

    noise = rng.normal(0.0, mp_std, size=(T, M)).astype(np.float32) + bias
    toa = dist + noise

    if n_channels >= 2:
        # simple AoA: angle from anchor to tag, with noise
        ang = np.arctan2(diff[..., 1], diff[..., 0])     # (T, M)
        ang_noise = rng.normal(0.0, 0.05, size=(T, M)).astype(np.float32)
        aoa = ang + ang_noise
        X = np.stack([toa, aoa], axis=-1)
    else:
        X = toa[..., None]

    # Per-timestep label = full trajectory (T, 2).
    # The tag does a random walk of T steps, so each timestep has a
    # different (x, y). The per-timestep location head outputs (B, T, 2)
    # and is trained against this full trajectory.
    y = pos.astype(np.float32)        # (T, 2)
    return (
        X.astype(np.float32),
        y,
        env.env_id,
        condition_id,
        nlos,                                # (T, M) bool
    )


def build_synthetic_dataset(
    cfg: Config, seed: int = 0
) -> List[Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]]:
    """Build a flat list of (X, y, env_id, cond_id, nlos) over all envs × traces × conds."""
    rng = np.random.default_rng(seed)
    envs: List[SyntheticEnv] = [
        _make_random_env(rng, i, cfg.n_anchors) for i in range(cfg.n_synthetic_envs)
    ]
    data: List[Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]] = []
    for env in envs:
        for _ in range(cfg.samples_per_env):
            cond = int(rng.integers(0, cfg.n_micro_conditions))
            X, y, eid, cid, nlos = simulate_trace(
                env, cond, cfg.seq_len, rng, cfg.n_channels
            )
            data.append((X, y, eid, cid, nlos))
    return data


class TensorUWB(Dataset):
    """Wrap a list of (X, y, env_id, cond_id[, nlos]) as tensors for DataLoader use.

    The 5-tuple form (with NLOS mask) is the new canonical format. Legacy
    4-tuple inputs are still accepted for backward compatibility with
    existing scripts; NLOS will be reported as an all-False mask.
    """

    def __init__(self, items):
        n = len(items)
        # Detect tuple arity
        if items and len(items[0]) == 6:
            Xs, ys, es, cs, ns, rs = zip(*items)
        elif items and len(items[0]) == 5:
            Xs, ys, es, cs, ns = zip(*items)
            T, M = Xs[0].shape[:2]
            rs = [np.zeros((T, M), dtype=np.float32) for _ in range(n)]
        elif items and len(items[0]) == 4:
            Xs, ys, es, cs = zip(*items)
            T, M = Xs[0].shape[:2]
            ns = [np.zeros((T, M), dtype=bool) for _ in range(n)]
            rs = [np.zeros((T, M), dtype=np.float32) for _ in range(n)]
        else:
            raise ValueError(f"unexpected item tuple arity: {len(items[0])}")
        self.X = torch.from_numpy(np.stack(list(Xs))).float()
        self.y = torch.from_numpy(np.stack(list(ys))).float()
        self.env = torch.tensor(list(es), dtype=torch.long)
        self.cond = torch.tensor(list(cs), dtype=torch.long)
        self.nlos = torch.from_numpy(np.stack(list(ns))).bool()    # (N, T, M)
        self.restore = torch.from_numpy(np.stack(list(rs))).float()  # (N, T, M)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return (self.X[i], self.y[i], self.env[i], self.cond[i],
                self.nlos[i], self.restore[i])


def make_synthetic(cfg: Config, seed: int = 0) -> Tuple[List, List]:
    """Split 80/20 by *environment* (so val sees unseen envs) for the standard
    supervised baseline path. Tier 1 (MAML) handles its own env-aware split."""
    items = build_synthetic_dataset(cfg, seed=seed)
    env_ids = sorted({it[2] for it in items})
    n_train_envs = max(1, int(0.8 * len(env_ids)))
    train_envs = set(env_ids[:n_train_envs])
    train = [it for it in items if it[2] in train_envs]
    val = [it for it in items if it[2] not in train_envs]
    return train, val
