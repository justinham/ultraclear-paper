"""Moving-tag Sionna dataset for the joint Tier 1+2 trainer.

Loads the per-timestep walking-path CSV
`~/Downloads/srp_research/sionna_walking_path_v2.csv`
(4001 timesteps, 4 anchors, ToA + AoA + per-anchor NLOS) and builds a
sliding-window dataset with the same 5-tuple contract the joint
trainer expects: (X, y, env_id, cond_id, nlos).

The walking-path data is **non-stationary** by construction: the tag
walks through 12 waypoints, dwelling 300 timesteps at each and
transitioning for the remainder. No two consecutive timesteps share
the *exact* (x, y) (the data has small pose jitter even at dwells),
so per-timestep labels are meaningful throughout the file.

Differences from `sionna_joint_dataset.py`:
  - There is no `grid_id` / `trial_id` structure. The data is a
    continuous 4001-step time series.
  - The sliding window is just `s in [0, N - seq_len]` (consecutive
    timesteps), with no "stay within one location" constraint.
    The user's spec: the path is non-stationary, so any 32-step
    window is a valid sample.
  - "Envs" for FOMAML: by default the whole path is one env. The
    `--n_envs` arg splits the path into N roughly equal temporal
    chunks; each chunk is its own env for FOMAML's task sampling.
    This is a pragmatic approximation of "different scenes" for the
    walking-path data, since we only have one trajectory. With one
    env, FOMAML degenerates to standard supervised learning.

API matches `sionna_joint_dataset.py`:
    from scripts.walking_path_dataset import (
        build_walking_path_dataset, walking_path_to_items,
    )
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ANCHOR_COLS = ["A1", "A2", "A3", "A4"]


@dataclass
class WalkingPathSceneInfo:
    name: str           # human-readable scene name
    env_id: int         # 0..N-1, used by FOMAML as the task identity
    nlos_rate: float    # fraction of (T, M) cells flagged as NLOS in this segment
    n_timesteps: int    # number of timesteps in this segment
    t_start: int        # first timestep index in the original CSV
    t_end: int          # last timestep index (exclusive) in the original CSV
    # Per-segment coordinate normalization (so the model can predict
    # location in a common reference frame across segments).
    tag_xy_mean: np.ndarray = None   # (2,)
    tag_xy_std: np.ndarray = None    # (2,)


def default_csv_path() -> Path:
    return Path("/Users/justin/Downloads/srp_research/sionna_walking_path_v2.csv")


class WalkingPathDataset(Dataset):
    """Sliding-window dataset over a per-timestep moving-tag CSV.

    Each window is shape (T, M, F) with per-timestep label (T, 2).
    The window at index `i` covers timesteps [i * stride, i * stride + seq_len).
    """

    def __init__(
        self,
        feat: np.ndarray,           # (N_timesteps, M, F) raw features
        tag_xy: np.ndarray,         # (N_timesteps, 2) tag positions
        nlos: np.ndarray,           # (N_timesteps, M) bool NLOS mask
        windows: List[Tuple[int, int]],  # list of (env_id, t_start)
        scene_info: List[WalkingPathSceneInfo],
        seq_len: int = 32,
        norm_mean: np.ndarray | None = None,
        norm_std: np.ndarray | None = None,
    ):
        self.feat = feat
        self.tag_xy = tag_xy
        self.nlos = nlos
        self.windows = windows
        self.scene_info = scene_info
        self.seq_len = seq_len
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        env_id, t = self.windows[i]
        x = self.feat[t : t + self.seq_len].copy()      # (T, M, F)
        if self.norm_mean is not None:
            x = (x - self.norm_mean) / self.norm_std
        y = self.tag_xy[t : t + self.seq_len].copy()    # (T, 2)
        if self.norm_mean is not None:
            y = (y - self.norm_mean[:2]) / self.norm_std[:2]
        nlos = self.nlos[t : t + self.seq_len].copy()    # (T, M) bool
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.tensor(env_id, dtype=torch.long),
            torch.tensor(-1, dtype=torch.long),     # no cond label for real data
            torch.from_numpy(nlos).bool(),
        )


def build_walking_path_dataset(
    csv_path: str | None = None,
    seq_len: int = 32,
    stride: int = 32,
    n_channels: int = 2,
    n_envs: int = 1,
) -> Tuple[WalkingPathDataset, List[WalkingPathSceneInfo]]:
    """Build a moving-tag dataset from the walking-path CSV.

    Args:
        csv_path: path to the CSV. Defaults to
            `~/Downloads/srp_research/sionna_walking_path_v2.csv`.
        seq_len: window size.
        stride: window stride. stride=seq_len gives non-overlapping windows.
        n_channels: 1 for ToA only, 2 for ToA + AoA azimuth.
        n_envs: how many envs to split the trajectory into for FOMAML.
            Default 1 (whole path = one env). With more envs, the path
            is split into N roughly equal temporal chunks; each chunk
            is its own env for the joint trainer's MAML task sampling.
            n_envs=1 means FOMAML degenerates (only one task to sample).
            Recommended: 4-8 for proper FOMAML transfer.
    """
    if csv_path is None:
        csv_path = str(default_csv_path())
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Walking-path CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    N = len(df)
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")
    n_envs = min(n_envs, N // seq_len)   # at least seq_len per env

    # Build features
    toa_cols = [f"A{m}_measured_dist_m" for m in range(1, 5)]
    aoa_cols = [f"A{m}_aoa_azimuth_deg" for m in range(1, 5)]
    nlos_cols = [f"A{m}_los_nlos" for m in range(1, 5)]
    toa = df[toa_cols].values.astype(np.float32)              # (N, 4)
    nlos = df[nlos_cols].values.astype(np.int8)               # (N, 4)
    tag_xy = df[["tag_x", "tag_y"]].values.astype(np.float32)  # (N, 2)
    if n_channels >= 2:
        aoa = df[aoa_cols].values.astype(np.float32)          # (N, 4)
        feat = np.stack([toa, aoa], axis=-1)                 # (N, 4, 2)
    else:
        feat = toa[..., None]                                  # (N, 4, 1)

    # Split into n_envs temporal chunks (contiguous, non-overlapping)
    chunk_size = N // n_envs
    scene_info: List[WalkingPathSceneInfo] = []
    windows: List[Tuple[int, int]] = []
    for env_id in range(n_envs):
        t_start_chunk = env_id * chunk_size
        t_end_chunk = (env_id + 1) * chunk_size if env_id < n_envs - 1 else N
        seg_t = slice(t_start_chunk, t_end_chunk)
        seg_toa = toa[seg_t]
        seg_nlos = nlos[seg_t]
        # Per-segment normalization: each segment has different
        # coordinate statistics; we normalize within-segment.
        xy_mean = tag_xy[seg_t].mean(axis=0).astype(np.float32)
        xy_std = (tag_xy[seg_t].std(axis=0) + 1e-6).astype(np.float32)
        toa_mean = np.concatenate([xy_mean, np.zeros(2, dtype=np.float32)])
        toa_std = np.concatenate([xy_std, np.ones(2, dtype=np.float32) * 100.0])
        if n_channels == 1:
            toa_mean = xy_mean
            toa_std = xy_std
        # Sliding windows within this segment
        n_windows = (t_end_chunk - t_start_chunk - seq_len) // stride + 1
        for w in range(n_windows):
            t = t_start_chunk + w * stride
            windows.append((env_id, t))
        info = WalkingPathSceneInfo(
            name=f"walking_path_seg{env_id}",
            env_id=env_id,
            nlos_rate=float(seg_nlos.mean()),
            n_timesteps=t_end_chunk - t_start_chunk,
            t_start=t_start_chunk,
            t_end=t_end_chunk,
            tag_xy_mean=xy_mean,
            tag_xy_std=xy_std,
        )
        scene_info.append(info)

    # Global per-channel norm for the features (ToA + AoA), computed
    # from the entire file. This matches the Sionna joint dataset
    # behavior where the encoder sees a common z-scored feature space
    # across envs.
    n_feats = feat.shape[-1]
    feat_mean = np.zeros(n_feats, dtype=np.float32)
    feat_std = np.ones(n_feats, dtype=np.float32)
    if n_feats >= 1:
        feat_mean[0] = toa.mean()
        feat_std[0] = toa.std() + 1e-6
    if n_feats >= 2:
        feat_mean[1] = aoa.mean()
        feat_std[1] = aoa.std() + 1e-6

    return (
        WalkingPathDataset(
            feat=feat, tag_xy=tag_xy, nlos=nlos.astype(bool),
            windows=windows, scene_info=scene_info, seq_len=seq_len,
            norm_mean=feat_mean, norm_std=feat_std,
        ),
        scene_info,
    )


def walking_path_to_items(
    dataset: WalkingPathDataset,
    train_env_ids: set[int] | None = None,
    val_env_ids: set[int] | None = None,
) -> Tuple[List[Tuple], List[Tuple], List[WalkingPathSceneInfo], dict]:
    """Convert the dataset into the (items, ...) list format the joint
    trainer expects, with a 80/20-by-env train/val split if not given.

    Returns (train_items, val_items, scene_info, denorm_info) where
    denorm_info maps env_id -> (mean, std) so the trainer can denormalize
    predictions back to original meters for the eval phase.
    """
    env_ids = sorted({info.env_id for info in dataset.scene_info})
    if train_env_ids is None and val_env_ids is None:
        n_train = max(1, int(0.8 * len(env_ids)))
        train_env_ids = set(env_ids[:n_train])
        val_env_ids = set(env_ids[n_train:])
    elif train_env_ids is None:
        train_env_ids = set(env_ids) - set(val_env_ids)
    elif val_env_ids is None:
        val_env_ids = set(env_ids) - set(train_env_ids)

    train_items = []
    val_items = []
    denorm_info = {}
    for i in range(len(dataset)):
        env_id, _t = dataset.windows[i]
        x, y, env_t, cond_t, nlos_t = dataset[i]
        item = (x.numpy(), y.numpy(), int(env_id), int(cond_t.item()), nlos_t.numpy())
        if env_id in train_env_ids:
            train_items.append(item)
        else:
            val_items.append(item)
    for info in dataset.scene_info:
        denorm_info[info.env_id] = (info.tag_xy_mean, info.tag_xy_std)
    return train_items, val_items, dataset.scene_info, denorm_info
