"""Multi-scene Sionna sliding-window dataset for Tier 1 FOMAML.

Loads N CSV files (one per scene) and builds sliding windows of
T=32 trials at each grid cell, carrying the scene env_id through
into each window.

Each window is shape (T, M, F=2) and exposes (X, y, env_id, cond_id):
  - X: the (T, M, F) window
  - y: tag (x, y) at the middle timestep
  - env_id: which scene the window came from (0..N-1)
  - cond_id: per-(grid, anchor) LOS/NLOS indicator at the middle
    timestep (a simple proxy for "micro-condition" — see below)

Per-channel z-score normalization is computed across *all training
scenes* (and saved so the inference / viz scripts can denormalize).

The "cond_id" we use is the average `A_m_los_nlos` flag at the
middle timestep: 0 if all 4 anchors are LOS, 1 if any is NLOS. This
is a coarse micro-condition proxy; the paper's Tier 2 will refine
this via FiLM.

Usage:
    from scripts.train_tier1_sionna import (
        MultiSceneSionnaDataset, build_multi_scene_dataset
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
class SceneInfo:
    name: str           # human-readable scene name (e.g. "empty_open")
    env_id: int         # 0..N-1, used by FOMAML as the task identity
    nlos_rate: float    # 0..1, fraction of trials with at least one NLOS anchor
    n_grids: int        # number of grid cells
    n_trials: int       # trials per grid


def _per_trial_cond_id(df: pd.DataFrame) -> np.ndarray:
    """For each trial, return 0 (all LOS) or 1 (any NLOS). Kept for
    backward compatibility but no longer used by build_multi_scene_dataset."""
    los = np.stack([df[f"A{m}_los_nlos"].values for m in range(1, 5)], axis=1)
    return (los == 0).any(axis=1).astype(np.int64)


def build_multi_scene_dataset(
    csv_paths: List[str],
    seq_len: int = 32,
    stride: int = 32,
    n_channels: int = 2,
) -> Tuple["MultiSceneSionnaDataset", List[SceneInfo]]:
    """Build a dataset from a list of scene CSVs.

    Each CSV must be 400 grids × 100 trials × 4 anchors. We group rows
    by (grid_id, trial_id) after sorting, then build per-scene feature
    tensors of shape (n_grids, n_trials, M=4, F).

    Returns (dataset, scene_info_list). The dataset exposes
    .scene_info for downstream FOMAML task sampling.
    """
    per_scene_features: List[np.ndarray] = []
    per_scene_tag_xy: List[np.ndarray] = []
    per_scene_cond: List[np.ndarray] = []
    scene_info: List[SceneInfo] = []

    for env_id, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        name = Path(path).stem.replace("_A1A4swapped", "").replace("sionna_uwb_grid_", "")

        # sort by (grid_id, trial_id) so the reshape to (n_grids, n_trials, ...) is well-defined
        df = df.sort_values(["grid_id", "trial_id"]).reset_index(drop=True)
        n_grids = int(df["grid_id"].nunique())
        n_trials = int((df["grid_id"] == df["grid_id"].iloc[0]).sum())
        assert n_grids * n_trials == len(df), \
            f"scene {name}: expected {n_grids * n_trials} rows, got {len(df)}"

        # channel 0: ToA (meters)
        toa = np.stack(
            [df[f"A{m}_measured_dist_m"].values for m in range(1, 5)],
            axis=1,
        ).astype(np.float32)  # (n_grids * n_trials, 4)
        toa = toa.reshape(n_grids, n_trials, 4)

        # channel 1: AoA azimuth (degrees)
        if n_channels >= 2:
            aoa = np.stack(
                [df[f"A{m}_aoa_azimuth_deg"].values for m in range(1, 5)],
                axis=1,
            ).astype(np.float32)
            aoa = aoa.reshape(n_grids, n_trials, 4)
            feat = np.stack([toa, aoa], axis=-1)             # (n_grids, n_trials, 4, 2)
        else:
            feat = toa[..., None]                              # (n_grids, n_trials, 4, 1)

        # tag positions
        tag_xy = df[["tag_x", "tag_y"]].values.astype(np.float32).reshape(n_grids, n_trials, 2)
        # cond_id: per-trial NLOS indicator (0 = all LOS, 1 = any NLOS)
        los = np.stack([df[f"A{m}_los_nlos"].values for m in range(1, 5)], axis=1)
        cond = (los == 0).any(axis=1).astype(np.int64).reshape(n_grids, n_trials)

        per_scene_features.append(feat)
        per_scene_tag_xy.append(tag_xy)
        per_scene_cond.append(cond)
        scene_info.append(SceneInfo(
            name=name, env_id=env_id,
            nlos_rate=float(cond.any(axis=1).mean()) if cond.size else 0.0,
            n_grids=n_grids, n_trials=n_trials,
        ))
        print(f"  env {env_id} ({name:35s}): {n_grids} grids x {n_trials} trials, "
              f"any-NLOS rate {scene_info[-1].nlos_rate*100:5.1f}%")

    ds = MultiSceneSionnaDataset.from_per_scene(
        per_scene_features, per_scene_tag_xy, per_scene_cond,
        scene_info, seq_len=seq_len, stride=stride,
    )
    return ds, scene_info


class MultiSceneSionnaDataset(Dataset):
    """Sliding-window dataset over multiple Sionna scenes.

    Construct via `MultiSceneSionnaDataset.from_per_scene(...)` or via
    the higher-level `build_multi_scene_dataset(csv_paths)`.

    Stores per-scene features (raw, un-normalized) in `self._per_scene`,
    then builds a window index: each entry is (scene_idx, grid_idx,
    start_trial). Windows are *normalized* on demand using the dataset's
    per-channel norm stats (set via `compute_norm_stats`).
    """

    def __init__(self):
        # required for pickling / Dataset inheritance; the real init is in
        # from_per_scene. Don't call this directly.
        raise NotImplementedError("use from_per_scene or build_multi_scene_dataset")

    @classmethod
    def from_per_scene(
        cls,
        per_scene_features: List[np.ndarray],   # list of (n_grids, n_trials, M, F)
        per_scene_tag_xy: List[np.ndarray],     # list of (n_grids, n_trials, 2)
        per_scene_cond: List[np.ndarray],       # list of (n_grids, n_trials) int
        scene_info: List[SceneInfo],
        seq_len: int = 32,
        stride: int = 32,
    ):
        """Direct constructor that takes already-grouped per-scene arrays."""
        ds = cls.__new__(cls)
        ds.seq_len = seq_len
        ds.n_anchors = 4
        ds.n_channels = per_scene_features[0].shape[-1]
        ds.scene_info = scene_info
        ds._per_scene = []
        for f, xy, c in zip(per_scene_features, per_scene_tag_xy, per_scene_cond):
            ds._per_scene.append({
                "feat": f,         # (n_grids, n_trials, M, F)
                "tag_xy": xy,      # (n_grids, n_trials, 2)
                "cond": c,         # (n_grids, n_trials)
            })
        # build window index: (scene_idx, grid_idx, start_trial)
        ds.windows = []
        for s_idx, info in enumerate(scene_info):
            n_trials = info.n_trials
            for g in range(info.n_grids):
                for s in range(0, n_trials - seq_len + 1, stride):
                    ds.windows.append((s_idx, g, s))
        return ds

    def compute_norm_stats(self, train_scene_ids: set[int]):
        """Compute per-channel mean/std from the *training* scenes.

        The caller is expected to pass a set of env_ids that are
        designated for training (so val scenes are truly held out).
        """
        all_train = []
        for s_idx, info in enumerate(self.scene_info):
            if info.env_id not in train_scene_ids:
                continue
            feat = self._per_scene[s_idx]["feat"]  # (n_grids, n_trials, M, F)
            all_train.append(feat.reshape(-1, self.n_channels))
        if not all_train:
            raise RuntimeError("no training scenes for norm stats")
        cat = np.concatenate(all_train, axis=0)
        self.norm_mean = cat.mean(axis=0).astype(np.float32)
        self.norm_std = (cat.std(axis=0) + 1e-6).astype(np.float32)
        return self.norm_mean, self.norm_std

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        s_idx, g, s = self.windows[i]
        scene = self._per_scene[s_idx]
        x = scene["feat"][g, s : s + self.seq_len]              # (T, M, F)
        if hasattr(self, "norm_mean"):
            x = (x - self.norm_mean) / self.norm_std
        y = scene["tag_xy"][g, s : s + self.seq_len]          # (T, 2) per-timestep
        c = int(scene["cond"][g, s + self.seq_len // 2])        # 0 or 1
        env_id = self.scene_info[s_idx].env_id
        T_, M_, _ = x.shape
        nlos = np.zeros((T_, M_), dtype=bool)
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.tensor(env_id, dtype=torch.long),
            torch.tensor(c, dtype=torch.long),
            torch.from_numpy(nlos),            # (T, M) all-False
        )


def build_default_multi_scene(
    n_channels: int = 2, seq_len: int = 32, stride: int = 32,
    all_scenes: bool = False,
):
    """Default scene list: pick a representative mix of LOS + NLOS scenes
    from the grid2 directory.

    If all_scenes=True, load every *_A1A4swapped.csv in the grid2
    directory (typically 20 scenes, with the missing-data scene env
    14 included). This gives FOMAML the full env diversity for the
    paper's main runs; the default 6-scene list is only useful for
    quick smoke tests.

    Returns (dataset, scene_info). See build_multi_scene_dataset for
    the path-to-dataset logic; this function just supplies the CSVs.
    """
    from pathlib import Path as _Path
    base = _Path("/Users/justin/Downloads/srp_research/grid2")
    if all_scenes:
        paths = sorted(str(p) for p in base.glob("*_A1A4swapped.csv"))
        print(f"Building multi-scene dataset with {len(paths)} scenes (all):")
    else:
        scenes = [
            "empty_open",                       # 100% LOS, the easy one
            "custom_room_large",                # 100% LOS, smaller
            "outdoor_blockages_pillars",        # 64% LOS, 14% NLOS, 21% no-paths
            "indoor_blockages_pillars",         # 62% LOS, 5% NLOS, 33% no-paths
            "outdoor_parking_01",               # 97% LOS
            "indoor_garage_parking_01",         # 93% LOS
        ]
        paths = [str(base / f"sionna_uwb_grid_{s}_A1A4swapped.csv") for s in scenes]
        print(f"Building multi-scene dataset with {len(scenes)} scenes:")
    return build_multi_scene_dataset(paths, seq_len=seq_len, stride=stride,
                                      n_channels=n_channels)
