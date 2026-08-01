"""Convert track2 CSVs (one-row-per-track with waypoint axis) into the
grid_id × trial_id format that `sionna_joint_dataset.build_sionna_joint_dataset`
expects.

track2 source shape: 1800 rows = 12 waypoints × 150 samples.
Each waypoint has 100 dwell samples (tag at the waypoint) + 50 transition
samples (tag moving to the next waypoint). Per the joint trainer, the
per-timestep label is stationary per grid, so we use (wp_x, wp_y) as the
grid's true position. This matches the sionna-grid semantics: each grid is
"a different (x, y) we want to localize at", and trial_id is sequential
samples at that location.

Output: *_grid.csv next to each *_A1A4swapped.csv, with grid_id/trial_id
columns added. Original columns are preserved.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def convert_one(path: Path) -> Path:
    df = pd.read_csv(path)
    if "grid_id" in df.columns and "trial_id" in df.columns:
        # Already in grid format — skip.
        return path

    # Group by (wp_x, wp_y) to define grids. Sort by wp_x, then wp_y for
    # deterministic grid_id assignment.
    waypoints = (
        df[["wp_x", "wp_y"]]
        .drop_duplicates()
        .sort_values(["wp_x", "wp_y"])
        .reset_index(drop=True)
    )
    wp_to_grid = {(r.wp_x, r.wp_y): i for i, r in waypoints.iterrows()}

    grid_ids = np.empty(len(df), dtype=np.int64)
    for i, (x, y) in enumerate(zip(df["wp_x"].values, df["wp_y"].values)):
        grid_ids[i] = wp_to_grid[(x, y)]

    # Within a grid, order by sample_id so trial_id 0..149 is sequential.
    df = df.copy()
    df["grid_id"] = grid_ids
    df["trial_id"] = df.groupby("grid_id").cumcount()

    # Force tag_x, tag_y to the static waypoint target. The track2 CSVs have
    # 100 dwell + 50 transition samples per waypoint — during transition the
    # tag is moving toward the next wp, so tag_x, tag_y deviate from wp_x,
    # wp_y. The joint trainer assumes stationary per grid, so we collapse
    # the label to the waypoint. This matches the sionna-grid semantics and
    # is consistent with what 7/8 used.
    df["tag_x"] = df["wp_x"]
    df["tag_y"] = df["wp_y"]

    # Reorder columns: scene, grid_id, trial_id, then the rest.
    front = ["scene", "grid_id", "trial_id"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    # Default: write *next to* the source with a "_grid" infix so the original
    # is preserved. Use --inplace to overwrite the source directly.
    out = path.with_name(path.name.replace("_A1A4swapped", "_A1A4swapped_grid"))
    df.to_csv(out, index=False)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", required=True)
    p.add_argument(
        "--pattern", default="*_A1A4swapped.csv",
        help="Glob to match source CSVs (already swapped into metadata order).",
    )
    p.add_argument(
        "--inplace", action="store_true",
        help="Overwrite the source CSV in place (no '_grid' infix).",
    )
    args = p.parse_args()

    paths = sorted(Path(args.csv_dir).glob(args.pattern))
    if not paths:
        sys.exit(f"no CSVs matched {args.pattern} in {args.csv_dir}")

    for path in paths:
        out = convert_one(path)
        if args.inplace:
            # Move the just-written "_grid" file over the original.
            path.unlink()
            out.rename(path)
            out = path
        print(f"  {path.name} → {out.name}")


if __name__ == "__main__":
    main()
