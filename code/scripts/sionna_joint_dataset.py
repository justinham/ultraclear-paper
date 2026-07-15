"""Sionna real-data dataset for the joint Tier 1+2 trainer.

Loads the 20 Sionna ray-tracing CSVs in `~/Downloads/srp_research/grid2/`
(each file is 400 grids * 100 trials * 4 anchors with ToA + AoA + per-cell
NLOS flags) and builds a sliding-window dataset with the same 5-tuple
contract the joint trainer expects: (X, y, env_id, cond_id, nlos).

Differences from the synthetic `build_synthetic_dataset`:
  - The 20 CSV files are loaded as 20 different envs.
  - `cond_id` is *not* cleanly defined for real data (the `los_nlos`
    flag is a per-anchor propensity, not a clean 3-class micro state).
    We return `cond_id = -1` to signal "no cond label" to the trainer.
    The trainer should drop the `cond_loss` term in that case.
  - `nlos` is the per-cell (T, M) bool mask from the Sionna
    `A{m}_los_nlos` columns. This IS per-cell ground truth, so the
    NLOS head can still be trained.
  - Per-channel z-score normalization is computed from *train* scenes
    only (matching the existing Tier 1 multi-scene pattern).

This file is the real-data counterpart to `sionna_uwb_track_*` style
data that the joint trainer used to require synthetic generation for.
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
class SionnaSceneInfo:
    name: str           # human-readable scene name
    env_id: int         # 0..N-1, used by FOMAML as the task identity
    nlos_rate: float    # fraction of (trial, anchor) cells flagged as NLOS
    n_grids: int
    n_trials: int
    csv_path: str
    # Per-scene coordinate normalization (so the model can predict
    # location in a common reference frame across scenes with
    # different scales and offsets).
    tag_xy_mean: np.ndarray = None   # (2,)
    tag_xy_std: np.ndarray = None    # (2,)


def default_csv_dir() -> Path:
    """Default location for the Sionna ray-tracing CSVs.

    Looks first in `code/data/` (the canonical in-repo location,
    gitignored) and falls back to `~/Downloads/srp_research/grid2/`
    for backward compatibility with earlier runs.
    """
    repo_data = Path(__file__).resolve().parent.parent / "data"
    if any(repo_data.glob("*_A1A4swapped.csv")):
        return repo_data
    return Path("/Users/justin/Downloads/srp_research/grid2")


def discover_sionna_scenes(csv_dir: Path | None = None,
                           pattern: str = "*_A1A4swapped.csv") -> List[Path]:
    """Find all Sionna scene CSVs in the given directory."""
    if csv_dir is None:
        csv_dir = default_csv_dir()
    return sorted(csv_dir.glob(pattern))


def build_sionna_joint_dataset(
    csv_paths: List[str] | None = None,
    seq_len: int = 32,
    stride: int = 32,
    n_channels: int = 2,
) -> Tuple["SionnaJointDataset", List[SionnaSceneInfo]]:
    """Build the joint-trainer dataset from a list of Sionna scene CSVs.

    Returns (dataset, scene_info_list). The dataset exposes .scene_info
    for downstream FOMAML task sampling.
    """
    if csv_paths is None:
        csv_paths = [str(p) for p in discover_sionna_scenes()]

    per_scene_features: List[np.ndarray] = []
    per_scene_tag_xy: List[np.ndarray] = []
    per_scene_nlos: List[np.ndarray] = []
    per_scene_restore: List[np.ndarray] = []
    scene_info: List[SionnaSceneInfo] = []

    for env_id, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        name = Path(path).stem.replace("_A1A4swapped", "").replace("sionna_uwb_grid_", "")
        df = df.sort_values(["grid_id", "trial_id"]).reset_index(drop=True)
        n_grids = int(df["grid_id"].nunique())
        n_trials = int((df["grid_id"] == df["grid_id"].iloc[0]).sum())
        assert n_grids * n_trials == len(df), \
            f"scene {name}: expected {n_grids * n_trials} rows, got {len(df)}"

        # channel 0: ToA (meters)
        toa = np.stack(
            [df[f"A{m}_measured_dist_m"].values for m in range(1, 5)],
            axis=1,
        ).astype(np.float32).reshape(n_grids, n_trials, 4)

        if n_channels >= 2:
            aoa = np.stack(
                [df[f"A{m}_aoa_azimuth_deg"].values for m in range(1, 5)],
                axis=1,
            ).astype(np.float32).reshape(n_grids, n_trials, 4)
            feat = np.stack([toa, aoa], axis=-1)             # (n_grids, n_trials, 4, 2)
        else:
            feat = toa[..., None]                              # (n_grids, n_trials, 4, 1)

        # Ground-truth geometric distance per cell (clean signal for the
        # restoration head to predict). For NLOS / NaN cells we fall
        # back to the measured ToA so the target is always defined.
        # Shape: (n_grids, n_trials, M) float32 in meters.
        gt_dist = np.stack(
            [df[f"A{m}_gt_dist_m"].values for m in range(1, 5)],
            axis=1,
        ).astype(np.float32).reshape(n_grids, n_trials, 4)

        # tag positions
        tag_xy = df[["tag_x", "tag_y"]].values.astype(np.float32).reshape(n_grids, n_trials, 2)
        # Per-scene normalization: center and scale to a common reference
        # frame. Different Sionna scenes have wildly different scales
        # (custom_room ~3m, empty_open ~20m, parking ~30m), so without
        # this the model would need to learn a different scale per scene.
        tag_xy_mean = tag_xy.reshape(-1, 2).mean(axis=0)
        tag_xy_std = tag_xy.reshape(-1, 2).std(axis=0) + 1e-6
        tag_xy = (tag_xy - tag_xy_mean) / tag_xy_std
        # per-cell NLOS mask: A_m_los_nlos == 0 means LOS, == 1 means NLOS
        # We use the los_nlos flag directly as the binary nlos target.
        nlos = np.stack([df[f"A{m}_los_nlos"].values for m in range(1, 5)], axis=1) \
                   .astype(np.float32).reshape(n_grids, n_trials, 4)
        # Drop trials that have NaN in any ToA measurement (e.g. the
        # 'outdoor_empty_missing20' scene has 20% deliberately-missing
        # ToA rows). We replace with 0 and mark a per-cell invalid mask
        # in `nlos` (NLOS=1 since the reading is unusable for distance).
        feat_raw = feat  # (n_grids, n_trials, M, F)
        toa_cols = [f"A{m}_measured_dist_m" for m in range(1, 5)]
        nan_mask = df[toa_cols].isna().any(axis=1).values.reshape(n_grids, n_trials)
        n_nan = int(nan_mask.sum())
        if n_nan > 0:
            print(f"    {n_nan} trials with NaN ToA -> marked NLOS=1, ToA set to 0")
        # Set ToA=0 and NLOS=1 for any NaN trial
        if n_channels >= 2:
            feat_raw[nan_mask, :, 0] = 0.0   # ToA = 0
        else:
            feat_raw[nan_mask, :, :] = 0.0
        nlos[nan_mask, :] = 1.0  # NLOS = 1 (unusable)
        feat = feat_raw

        # For the restore target, replace NaN gt_dist with the
        # measured ToA (or 0 where measured was also NaN). We also
        # fall back to measured where the cell is NLOS-flagged, since
        # "clean distance" isn't a meaningful concept in NLOS.
        restore_target = gt_dist.copy()
        restore_target = np.where(np.isnan(restore_target), toa, restore_target)
        # NLOS cells: use measured (biased) as the "best available"
        # target. The restoration head will then learn to output the
        # measured value for NLOS cells and the true distance for LOS.
        nlos_mask = nlos.astype(bool)        # (n_grids, n_trials, M)
        restore_target[nlos_mask] = toa[nlos_mask]
        # NaN-trial guard (matches feat: ToA=0 there)
        restore_target[nan_mask, :] = 0.0

        per_scene_features.append(feat)
        per_scene_tag_xy.append(tag_xy)
        per_scene_nlos.append(nlos)
        per_scene_restore.append(restore_target)
        scene_info.append(SionnaSceneInfo(
            name=name, env_id=env_id,
            nlos_rate=float(nlos.mean()),
            n_grids=n_grids, n_trials=n_trials,
            csv_path=path,
            tag_xy_mean=tag_xy_mean,
            tag_xy_std=tag_xy_std,
        ))
        print(f"  env {env_id:2d} ({name:35s}): {n_grids} grids x {n_trials} trials, "
              f"per-cell NLOS rate {scene_info[-1].nlos_rate*100:5.1f}%")

    ds = SionnaJointDataset.from_per_scene(
        per_scene_features, per_scene_tag_xy, per_scene_nlos,
        per_scene_restore, scene_info, seq_len=seq_len, stride=stride,
    )
    return ds, scene_info


class SionnaJointDataset(Dataset):
    """Sliding-window dataset over multiple Sionna scenes.

    Returns (X, y, env_id, cond_id, nlos) where:
      - X: (T, M, F) measurement window (z-score normalized if norm stats set)
      - y: (2,) tag (x, y) at the middle timestep
      - env_id: int scene index
      - cond_id: int, ALWAYS -1 for real data (no clean 3-class micro label)
      - nlos: (T, M) bool per-cell NLOS mask

    Construct via `SionnaJointDataset.from_per_scene(...)` or via the
    higher-level `build_sionna_joint_dataset(csv_paths)`.

    Per-channel z-score normalization is set via `compute_norm_stats(train_env_ids)`.
    """

    def __init__(self):
        raise NotImplementedError("use from_per_scene or build_sionna_joint_dataset")

    @classmethod
    def from_per_scene(
        cls,
        per_scene_features: List[np.ndarray],   # (n_grids, n_trials, M, F)
        per_scene_tag_xy: List[np.ndarray],     # (n_grids, n_trials, 2)
        per_scene_nlos: List[np.ndarray],       # (n_grids, n_trials, M) per-cell NLOS
        per_scene_restore: List[np.ndarray],    # (n_grids, n_trials, M) restore target
        scene_info: List[SionnaSceneInfo],
        seq_len: int = 32,
        stride: int = 32,
    ):
        ds = cls.__new__(cls)
        ds.seq_len = seq_len
        ds.n_anchors = 4
        ds.n_channels = per_scene_features[0].shape[-1]
        ds.scene_info = scene_info
        ds._per_scene = []
        for f, xy, nl, rt in zip(per_scene_features, per_scene_tag_xy, per_scene_nlos, per_scene_restore):
            ds._per_scene.append({
                "feat": f,         # (n_grids, n_trials, M, F)
                "tag_xy": xy,      # (n_grids, n_trials, 2)
                "nlos": nl,        # (n_grids, n_trials, M) per-cell
                "restore": rt,     # (n_grids, n_trials, M) clean signal target
            })
        ds.windows = []  # (scene_idx, grid_idx, start_trial)
        for s_idx, info in enumerate(scene_info):
            for g in range(info.n_grids):
                for s in range(0, info.n_trials - seq_len + 1, stride):
                    ds.windows.append((s_idx, g, s))
        ds.norm_mean = None
        ds.norm_std = None
        return ds

    def compute_norm_stats(self, train_env_ids: set[int]):
        """Per-channel mean/std from training scenes only."""
        all_train = []
        for s_idx, info in enumerate(self.scene_info):
            if info.env_id not in train_env_ids:
                continue
            feat = self._per_scene[s_idx]["feat"]
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
        sc = self._per_scene[s_idx]
        x = sc["feat"][g, s : s + self.seq_len].copy()  # (T, M, F)
        if self.norm_mean is not None:
            x = (x - self.norm_mean) / self.norm_std
        # Per-timestep label: the full (T, 2) trajectory at the same grid.
        # NOTE: the Sionna CSVs are *stationary* per grid (all 100 trials
        # at a given (grid_id) share the same (tag_x, tag_y)), so the
        # T labels in any window from this data are all (approximately)
        # equal. The per-timestep head on this data will learn to output
        # the same (x, y) at every timestep. For genuinely moving-tag
        # data, see sionna_walking_path_v2.csv.
        # The window [s, s+seq_len) is constrained to stay within a
        # single grid_id, so labels at all T timesteps come from the
        # same underlying (x, y).
        y = sc["tag_xy"][g, s : s + self.seq_len].copy()  # (T, 2)
        env_id = self.scene_info[s_idx].env_id
        cond_id = -1   # no clean 3-class micro label for real data
        nlos = sc["nlos"][g, s : s + self.seq_len].copy()  # (T, M) float
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.tensor(env_id, dtype=torch.long),
            torch.tensor(cond_id, dtype=torch.long),
            torch.from_numpy(nlos).bool(),
        )

    def getitem_with_restore(self, i: int):
        """Same as __getitem__ but also returns the (T, M) restore target
        as a 6th element. Used by the supervised baseline trainer so
        the restoration head has a real signal to learn from. The
        5-tuple __getitem__ contract is preserved for the joint
        trainer and other downstream consumers."""
        s_idx, g, s = self.windows[i]
        sc = self._per_scene[s_idx]
        x = sc["feat"][g, s : s + self.seq_len].copy()
        if self.norm_mean is not None:
            x = (x - self.norm_mean) / self.norm_std
        y = sc["tag_xy"][g, s : s + self.seq_len].copy()
        env_id = self.scene_info[s_idx].env_id
        nlos = sc["nlos"][g, s : s + self.seq_len].copy()
        restore = sc["restore"][g, s : s + self.seq_len].copy()
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.tensor(env_id, dtype=torch.long),
            torch.tensor(-1, dtype=torch.long),
            torch.from_numpy(nlos).bool(),
            torch.from_numpy(restore).float(),
        )


def sionna_to_items(
    dataset: SionnaJointDataset,
    train_env_ids: set[int] | None = None,
    val_env_ids: set[int] | None = None,
) -> Tuple[List[Tuple], List[Tuple], List[SionnaSceneInfo], dict]:
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
    for info in dataset.scene_info:
        denorm_info[info.env_id] = (info.tag_xy_mean, info.tag_xy_std)
    for i in range(len(dataset)):
        x, y, env, _cond, nlos = dataset[i]
        # y is in normalized coords; we keep it that way for training.
        # Cond is always -1 for real data; the trainer will skip cond_loss.
        item = (x.numpy(), y.numpy(), int(env.item()), -1, nlos.numpy().astype(bool))
        if int(env.item()) in train_env_ids:
            train_items.append(item)
        else:
            val_items.append(item)
    return train_items, val_items, dataset.scene_info, denorm_info


def sionna_to_items_with_restore(
    dataset: SionnaJointDataset,
    train_env_ids: set[int] | None = None,
    val_env_ids: set[int] | None = None,
) -> Tuple[List[Tuple], List[Tuple], List[SionnaSceneInfo], dict]:
    """Like `sionna_to_items` but returns 6-tuples that include the
    (T, M) restore target window. Used by the supervised baseline
    trainer to give the restoration head a real signal.

    Returns (train_items, val_items, scene_info, denorm_info).
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

    train_items, val_items, denorm_info = [], [], {}
    for info in dataset.scene_info:
        denorm_info[info.env_id] = (info.tag_xy_mean, info.tag_xy_std)
    for i in range(len(dataset)):
        x, y, env, _cond, nlos, restore = dataset.getitem_with_restore(i)
        item = (x.numpy(), y.numpy(), int(env.item()), -1,
                nlos.numpy().astype(bool), restore.numpy())
        if int(env.item()) in train_env_ids:
            train_items.append(item)
        else:
            val_items.append(item)
    return train_items, val_items, dataset.scene_info, denorm_info
