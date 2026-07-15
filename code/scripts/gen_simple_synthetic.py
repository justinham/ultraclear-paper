"""Simple controlled-noise synthetic UWB dataset.

Generates a multi-scene CSV dataset that matches the format of the
Sionna ray-traced data (`sionna_uwb_grid_*_A1A4swapped.csv`) so the
supervised baseline trainer can consume it unchanged via
`--csv_dir`. The whole point is to have a *clean, controlled*
noise distribution we can reason about, instead of the messy
real-data distribution where the "NLOS" flag is actually a multipath
artifact and the "LOS" cells are the biased ones.

Noise model (per-cell):
  - LOS cells:   ToA = true_dist + N(0, sigma_toa=0.05 m)
                 AoA = true_aoa + N(0, sigma_aoa=2 deg)
  - NLOS cells:  ToA = true_dist + U(0, 0.5) + N(0, 0.05) m
                 AoA = U(0, 360) deg  (no reliable AoA in NLOS)
  - Missing:     0.5% of cells (rare NaN measurement, marked NLOS=1)

Dynamic noise (per-window NLOS rate distribution):
  We model "a person walking by" as: most of the time the path is
  clear, occasionally 1-2 anchors are blocked. Per-scene we draw
  the per-window NLOS rate from one of three regimes:
    - low  (3 scenes):  rate ~ 0.03  (occasional 1-anchor blockage)
    - mid  (4 scenes):  rate ~ 0.10  (regular 1-anchor, rare 2-anchor)
    - high (3 scenes):  rate ~ 0.20  (frequent 1-anchor, regular 2-anchor)
  This gives a realistic, *unimodal* per-window NLOS rate
  distribution (instead of the bimodal 0%/100% the real Sionna
  data produces because its NLOS flag is per-scene, not per-walk-by).

Scene geometry:
  - 10 rectangular rooms, sides drawn from {4, 6, 8, 12, 16, 20} m
  - 4 anchors at the corners at height 2.0 m
  - Tag at height 1.0 m, 100 grid points per scene
  - 32 trials per (scene, grid) point = 1 window per point at seq_len=32

CSV format (per row = one (grid, trial)):
  grid_id, trial_id, grid_ix, grid_iy,
  tag_x, tag_y, tag_z,
  A{m}_measured_dist_m, A{m}_gt_dist_m, A{m}_error_m, A{m}_los_nlos,
  A{m}_aoa_azimuth_deg, A{m}_aoa_elevation_deg,
  A{m}_aoa_gt_azimuth_deg, A{m}_aoa_gt_elevation_deg

Output goes to `<out_dir>/sionna_uwb_grid_<scene>_A1A4swapped.csv`,
matching the naming convention the baseline expects.

Usage:
    python3.11 -m scripts.gen_simple_synthetic --out_dir /tmp/simple_synth
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Constants for the noise model
SIGMA_TOA_LOS = 0.05      # m, 5 cm
SIGMA_TOA_NLOS_EXTRA = 0.05  # additional Gaussian on top of the bias
SIGMA_AOA_LOS_DEG = 2.0   # deg
BIAS_MAX_M = 0.5          # m, max NLOS bias
MISSING_RATE = 0.005       # 0.5% of cells have NaN ToA
N_GRID = 10               # tag positions per side per scene
N_TRIALS = 32             # trials per (scene, grid) point (= seq_len)
TAG_Z = 1.0               # m
ANCHOR_Z = 2.0            # m


# Per-scene regime: side length (m), NLOS rate (mean), regime name
SCENES = [
    # low: 3 scenes, ~3% per-window NLOS
    {"name": "small_low_1",   "side": 4.0,  "nlos_mean": 0.03, "regime": "low"},
    {"name": "small_low_2",   "side": 6.0,  "nlos_mean": 0.03, "regime": "low"},
    {"name": "small_low_3",   "side": 8.0,  "nlos_mean": 0.03, "regime": "low"},
    # mid: 4 scenes, ~10% per-window NLOS
    {"name": "mid_1",         "side": 8.0,  "nlos_mean": 0.10, "regime": "mid"},
    {"name": "mid_2",         "side": 12.0, "nlos_mean": 0.10, "regime": "mid"},
    {"name": "mid_3",         "side": 12.0, "nlos_mean": 0.10, "regime": "mid"},
    {"name": "mid_4",         "side": 16.0, "nlos_mean": 0.10, "regime": "mid"},
    # high: 3 scenes, ~20% per-window NLOS
    {"name": "large_high_1",  "side": 16.0, "nlos_mean": 0.20, "regime": "high"},
    {"name": "large_high_2",  "side": 16.0, "nlos_mean": 0.20, "regime": "high"},
    {"name": "large_high_3",  "side": 20.0, "nlos_mean": 0.20, "regime": "high"},
]


def aoa_azimuth_deg(tag_xy, anchor_xy):
    """AoA azimuth in degrees: angle from anchor to tag, measured
    CCW from +x (atan2 convention). 0 deg = tag is to the +x of anchor,
    90 deg = tag is to the +y of anchor. Range [-180, 180]."""
    dx = tag_xy[0] - anchor_xy[0]
    dy = tag_xy[1] - anchor_xy[1]
    return math.degrees(math.atan2(dy, dx))


def aoa_elevation_deg(tag_xyz, anchor_xyz):
    """AoA elevation in degrees: angle above horizontal."""
    dx = tag_xyz[0] - anchor_xyz[0]
    dy = tag_xyz[1] - anchor_xyz[1]
    dz = tag_xyz[2] - anchor_xyz[2]
    horiz = math.hypot(dx, dy)
    return math.degrees(math.atan2(dz, horiz))


def make_scene(scene_def: dict, seed: int) -> pd.DataFrame:
    """Generate one scene as a long DataFrame."""
    rng = np.random.default_rng(seed)
    name = scene_def["name"]
    side = scene_def["side"]
    nlos_mean = scene_def["nlos_mean"]

    # Anchors at the corners at ANCHOR_Z
    anchors = [
        (0.0, 0.0),         # A1 bottom-left
        (side, 0.0),        # A2 bottom-right
        (side, side),       # A3 top-right
        (0.0, side),        # A4 top-left
    ]

    # Tag grid: N_GRID x N_GRID positions in the room interior
    grid_xs = np.linspace(0.5, side - 0.5, N_GRID)
    grid_ys = np.linspace(0.5, side - 0.5, N_GRID)

    rows = []
    grid_id = 0
    for gx in grid_xs:
        for gy in grid_ys:
            tag = (float(gx), float(gy), TAG_Z)
            # Per-window NLOS rate: Beta-distributed around the scene's mean.
            # Beta(2, K) where K = 2/nlos_mean - 2 has the right mean and a
            # realistic variance (most windows near the mean, some lower/higher).
            # Clamp to a reasonable range so we never get 0% or 100% per window.
            n = max(0.5, 2.0 / nlos_mean - 2.0)
            window_nlos_rate = float(np.clip(
                rng.beta(2.0, n), 0.01, 0.50,
            ))
            # Per-window NLOS pattern: each (anchor, trial) is NLOS
            # independently with probability = window_nlos_rate. This
            # captures "1-2 anchors briefly blocked" within a single window.
            for trial in range(N_TRIALS):
                row = {
                    "grid_id": grid_id,
                    "trial_id": trial,
                    "grid_ix": int(gx / side * N_GRID),
                    "grid_iy": int(gy / side * N_GRID),
                    "tag_x": tag[0],
                    "tag_y": tag[1],
                    "tag_z": tag[2],
                }
                for m, anc in enumerate(anchors, start=1):
                    anchor_xyz = (anc[0], anc[1], ANCHOR_Z)
                    dx = tag[0] - anc[0]
                    dy = tag[1] - anc[1]
                    dz = tag[2] - ANCHOR_Z
                    true_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    true_az = aoa_azimuth_deg(tag, anc)
                    true_el = aoa_elevation_deg(tag, anchor_xyz)

                    # NLOS or LOS?
                    is_nlos = rng.random() < window_nlos_rate
                    # Forced missing (rare, mimics sensor dropouts)
                    is_missing = rng.random() < MISSING_RATE
                    if is_missing:
                        # Leave ToA NaN, mark NLOS=1, AoA random
                        row[f"A{m}_measured_dist_m"] = np.nan
                        row[f"A{m}_gt_dist_m"] = true_dist
                        row[f"A{m}_error_m"] = np.nan
                        row[f"A{m}_los_nlos"] = 1
                        row[f"A{m}_aoa_azimuth_deg"] = rng.uniform(-180, 180)
                        row[f"A{m}_aoa_elevation_deg"] = rng.uniform(-45, 45)
                        row[f"A{m}_aoa_gt_azimuth_deg"] = true_az
                        row[f"A{m}_aoa_gt_elevation_deg"] = true_el
                        continue

                    if is_nlos:
                        # Bias uniform in [0, BIAS_MAX_M], plus Gaussian
                        bias = rng.uniform(0.0, BIAS_MAX_M)
                        noise = rng.normal(0.0, SIGMA_TOA_LOS)
                        measured = true_dist + bias + noise
                        aoa_az = rng.uniform(-180, 180)
                        aoa_el = rng.uniform(-30, 30)
                    else:
                        # Pure Gaussian noise
                        noise = rng.normal(0.0, SIGMA_TOA_LOS)
                        measured = true_dist + noise
                        aoa_az = true_az + rng.normal(0.0, SIGMA_AOA_LOS_DEG)
                        aoa_el = true_el + rng.normal(0.0, SIGMA_AOA_LOS_DEG)

                    row[f"A{m}_measured_dist_m"] = float(measured)
                    row[f"A{m}_gt_dist_m"] = float(true_dist)
                    row[f"A{m}_error_m"] = float(measured - true_dist)
                    row[f"A{m}_los_nlos"] = int(is_nlos)
                    row[f"A{m}_aoa_azimuth_deg"] = float(aoa_az)
                    row[f"A{m}_aoa_elevation_deg"] = float(aoa_el)
                    row[f"A{m}_aoa_gt_azimuth_deg"] = float(true_az)
                    row[f"A{m}_aoa_gt_elevation_deg"] = float(true_el)
                rows.append(row)
            grid_id += 1

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, scene in enumerate(SCENES):
        df = make_scene(scene, seed=args.seed * 1000 + i)
        # Per-scene sanity check
        nlos_rate = df["A1_los_nlos"].mean()
        los_err = df.loc[df["A1_los_nlos"] == 0, "A1_error_m"].abs().mean()
        nlos_err = df.loc[df["A1_los_nlos"] == 1, "A1_error_m"].abs().mean()
        side = scene["side"]
        path = out_dir / f"sionna_uwb_grid_{scene['name']}_A1A4swapped.csv"
        df.to_csv(path, index=False)
        print(f"  env {i:2d} ({scene['name']:20s} {side:5.1f}m, "
              f"regime={scene['regime']:4s}): "
              f"nlos={nlos_rate*100:4.1f}%  "
              f"|A1 LOS err|={los_err*100:5.2f}cm  "
              f"|A1 NLOS err|={nlos_err*100:5.2f}cm  "
              f"-> {path.name}")

    # Cross-scene summary
    print(f"\nWrote {len(SCENES)} scenes to {out_dir}")
    print("Use with the supervised baseline:")
    print(f"  python3.11 -m scripts.train_supervised_baseline --csv_dir {out_dir} ...")


if __name__ == "__main__":
    main()
