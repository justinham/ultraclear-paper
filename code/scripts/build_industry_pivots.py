"""Build the pivoted_superframe_{distances,rxpower}_<scene>.csv files
from the IDLAB industry1 / industry2 .bag exports.

For each sub-folder under Downloads/industry{1,2} (each one is one
.bag recording, already converted to processed.csv + gt.csv by the
IDLAB export tooling) we read:

    processed.csv  -- one row per UWB ranging event. Schema:
        superframe, Logtime, data, version, anchor (0xXXXX),
        tag, distance (mm), ..., rx_power (dBm), fp_power (dBm), ...
        Already filtered by the IDLAB export:
          - dropped on ERRORS
          - dropped on poll-request rssi
          - dropped on b_anchor (so each row is a real range)
          - dropped on distance > 50m
          - TWR superframe added (superframe column populated)
        Confirmed by the dataset_information.txt that ships with
        each folder.

    gt.csv         -- bot ground-truth pose at 10 Hz, columns:
        Logtime, x, y, z
        (1 row per ~100 ms, units = meters in the local anchor frame)

We group the ranging rows by `superframe`, then for each superframe
we record:
    logtime_start = first Logtime in the group
    logtime_mid   = (logtime_start + logtime_end) / 2
    logtime_end   = last Logtime in the group

Then we pivot to one row per superframe, with one column per anchor
(dist_0xXXXX for distance-in-mm, rxp_0xXXXX for rx_power-in-dBm).
Anchors that didn't get a ranging in a given superframe leave the
cell empty (NaN).

Finally we snap gt.csv to each logtime_mid (nearest neighbour) and
add gt_x, gt_y, gt_z.

Output naming convention (matches the existing files in
figure/anc20/):

  industry1 (TWR, 2 scenes):
    pivoted_superframe_distances.csv          <- industry1/_rh
    pivoted_superframe_distances_sl.csv       <- industry1/_sl

  industry2 (TDOA, 5 scenes, RX power only):
    pivoted_superframe_rxpower_rh.csv         <- industry2/_rh
    pivoted_superframe_rxpower_sllong.csv     <- industry2/_sllong
    pivoted_superframe_rxpower_ro.csv         <- industry2/_ro
    pivoted_superframe_rxpower_ls_los.csv     <- industry2/_ls_los
    pivoted_superframe_rxpower_grid_los.csv   <- industry2/_grid_los

We derive the suffix from the sub-folder name (the bit between
"app_" and the date) so the script generalizes if more recordings
are dropped in.

Run:
    python3.11 -m scripts.build_industry_pivots
    python3.11 -m scripts.build_industry_pivots --industry1_root .../industry1
    python3.11 -m scripts.build_industry_pivots --out_root .../figure/anc20
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_INDUSTRY1_ROOT = Path(
    "/Users/justin/Downloads/industry1"
)
DEFAULT_INDUSTRY2_ROOT = Path(
    "/Users/justin/Downloads/industry2"
)
DEFAULT_OUT_ROOT = Path(
    "/Users/justin/ultraclear-paper/figure/anc20"
)

# Each sub-folder maps to a scene tag. We parse the tag from the
# folder name (between "app_" and the date) but keep an explicit
# override map so the output filenames stay stable if the user
# renames the folders.
SCENE_TAG_OVERRIDES: dict[str, str] = {
    # industry1
    "2023_IDLAB_JULY_CIR-TWR-app_rh_2023_Jul_10_15_32_36_2023-07-10-13-34-25":
        "rh",                                # no suffix on the dist file
    "2023_IDLAB_JULY_CIR-TWR-app_sl_2023_Jul_10_15_21_28_2023-07-10-13-23-07":
        "sl",
    # industry2
    "OKT23_FL2_E5CE_2-TDOA-app_rh_2023_Nov_21_11_26_15_2023-11-21-10-26-28":
        "rh",
    "OKT23_FL2_E5CE-TDOA-app_sllong_2023_Nov_22_10_54_37_2023-11-22-09-54-49":
        "sllong",
    "OKT23_FL2_E5CE-TDOA-app_ro_2023_Nov_22_11_38_10_2023-11-22-10-38-21":
        "ro",
    "TDOA-app-1_LS_LOS_2023_Jul_11_10_08_08_2023-07-11-08-09-46":
        "ls_los",
    "TDOA-app-1_GRID_LOS_2023_Jul_11_10_24_20_2023-07-11-08-26-09":
        "grid_los",
}

# Industry1 (CIR-TWR) emits distance pivots for both sub-folders.
# It also emits an rx-power pivot, but ONLY for the _sl recording
# — the existing `pivoted_superframe_rxpower_rh.csv` in
# figure/anc20/ is actually from industry2/_rh (the OKT23 TDOA
# recording), so the industry1/_rh rx-power pivot would clobber
# that and is skipped.
INDUSTRY1_DEFAULT_PIVOTS = ("distances",)
INDUSTRY1_FOLDERS = {
    # folder-name-prefix -> pivots tuple
    "2023_IDLAB_JULY_CIR-TWR-app_rh_": ("distances",),
    "2023_IDLAB_JULY_CIR-TWR-app_sl_": ("distances", "rxpower"),
}
# Industry2 (TDOA) emits rx-power pivots only (no per-anchor
# distance; TDOA gives time-difference-of-arrival).
INDUSTRY2_PIVOTS = ("rxpower",)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def parse_scene_tag(folder_name: str) -> str:
    """Extract the scene tag from a sub-folder name.

    Example: '..._app_rh_2023_Jul_10_15_32_36_2023-07-10-...'
             -> 'rh'

    Falls back to the parent override map (preferred) before
    falling back to the regex.
    """
    if folder_name in SCENE_TAG_OVERRIDES:
        return SCENE_TAG_OVERRIDES[folder_name]
    # Generic: 'app_<tag>_<date>' -> tag
    m = re.search(r"app_([A-Za-z0-9]+)_", folder_name)
    if m:
        return m.group(1).lower()
    # Last resort: hash to a short stable id
    return "scene" + str(abs(hash(folder_name)) % 10_000)


def load_canonical_anchors(folder: Path) -> list[str] | None:
    """Return the 23 anchor hexes from AnchorsIIoT-28march23.csv if it
    ships in the sub-folder. This is the canonical anchor set used
    to define the column order of the output pivots; anchors that
    don't appear in this recording still get a column (all-NaN).

    Returns None if the file isn't there (falls back to per-file
    unique anchors in the caller)."""
    csv = folder / "AnchorsIIoT-28march23.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    return df["hex"].tolist()


def distance_filename(scene_tag: str) -> str:
    """industry1 distance pivots: only the 'sl' variant carries a suffix
    in the existing files (`pivoted_superframe_distances_sl.csv`); the
    first one (`rh`) has no suffix. We match that convention so re-runs
    overwrite the same files."""
    if scene_tag == "sl":
        return "pivoted_superframe_distances_sl.csv"
    return "pivoted_superframe_distances.csv"


def rxpower_filename(scene_tag: str) -> str:
    return f"pivoted_superframe_rxpower_{scene_tag}.csv"


# --------------------------------------------------------------------------- #
# Core: build a pivot for one (folder, kind) pair
# --------------------------------------------------------------------------- #

def build_pivot(folder: Path,
                kind: str,
                out_path: Path) -> dict:
    """kind in {"distances", "rxpower"}.

    Returns a small dict of stats for logging."""
    proc_csv = folder / "processed.csv"
    gt_csv = folder / "gt.csv"
    if not proc_csv.exists():
        raise FileNotFoundError(proc_csv)
    if not gt_csv.exists():
        raise FileNotFoundError(gt_csv)

    proc = pd.read_csv(proc_csv)
    gt = pd.read_csv(gt_csv)

    # Sanity-check: every row should have an anchor address. If
    # not, drop them but log.
    n_no_anchor = int(proc["anchor"].isna().sum())
    if n_no_anchor:
        print(f"  [warn] dropping {n_no_anchor} rows with NaN anchor")
        proc = proc.dropna(subset=["anchor"]).reset_index(drop=True)

    # Canonical anchor list, if available, defines the column order
    # AND guarantees we always emit the same 23 columns even when
    # an anchor is silent in this recording. Fall back to the union
    # of anchors that actually appear in this file's processed.csv.
    canonical = load_canonical_anchors(folder)
    if canonical is not None:
        anchors = list(canonical)
    else:
        anchors = sorted(proc["anchor"].unique().tolist())

    # Group by superframe
    grouped = proc.groupby("superframe", sort=True)

    rows = []
    for sf, sub in grouped:
        logtime_start = float(sub["Logtime"].min())
        logtime_end = float(sub["Logtime"].max())
        logtime_mid = 0.5 * (logtime_start + logtime_end)

        # One row per anchor in this superframe. If an anchor has
        # multiple ranges in the same superframe we average them
        # (the IDLAB export shouldn't produce that, but guard
        # anyway).
        per_anchor = sub.groupby("anchor")
        row: dict = {
            "logtime_start": logtime_start,
            "logtime_mid": logtime_mid,
            "logtime_end": logtime_end,
        }
        if kind == "distances":
            for a in anchors:
                if a in per_anchor.groups:
                    row[f"dist_{a}"] = float(per_anchor.get_group(a)["distance"].mean())
                else:
                    row[f"dist_{a}"] = np.nan
        elif kind == "rxpower":
            for a in anchors:
                if a in per_anchor.groups:
                    row[f"rxp_{a}"] = float(per_anchor.get_group(a)["rx_power"].mean())
                else:
                    row[f"rxp_{a}"] = np.nan
        else:
            raise ValueError(f"unknown kind: {kind}")
        rows.append(row)

    df = pd.DataFrame(rows)

    # Snap GT (nearest neighbour) to each logtime_mid. We sort both
    # by time, then use searchsorted for O(N+M) matching.
    gt_sorted = gt.sort_values("Logtime").reset_index(drop=True)
    gt_times = gt_sorted["Logtime"].to_numpy()
    idx = np.searchsorted(gt_times, df["logtime_mid"].to_numpy())
    # searchsorted gives the *insertion* point, so the nearest is
    # either idx or idx-1. Pick whichever is closer.
    idx = np.clip(idx, 0, len(gt_times) - 1)
    nearest = df["logtime_mid"].to_numpy() - gt_times[idx]
    idx_prev = np.clip(idx - 1, 0, len(gt_times) - 1)
    nearer = np.abs(df["logtime_mid"].to_numpy() - gt_times[idx_prev]) < np.abs(nearest)
    idx = np.where(nearer, idx_prev, idx)

    df["gt_x"] = gt_sorted["x"].to_numpy()[idx]
    df["gt_y"] = gt_sorted["y"].to_numpy()[idx]
    df["gt_z"] = gt_sorted["z"].to_numpy()[idx]

    # Prepend id = the actual superframe number (NOT a fresh row
    # index). Gaps in the superframe sequence are preserved so the
    # id values match the originals bit-for-bit. Two superframes in
    # the sl recording (34, 168) have zero ranging rows in the
    # processed.csv, which is why the file has 256 rows but the id
    # range is 0..257.
    sf_ids = [int(sf) for sf, _ in grouped]
    df.insert(0, "id", sf_ids)
    df = df.sort_values("id").reset_index(drop=True)

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use the gt CSV's natural float format (no float_format kwarg)
    # so the output matches the original pivots to the last digit
    # instead of being rounded to 6 decimal places.
    df.to_csv(out_path, index=False)

    n_anchors = len(anchors)
    n_superframes = len(df)
    coverage = float(df.filter(like=("dist_" if kind == "distances" else "rxp_"))
                       .notna().sum().sum() / (n_anchors * n_superframes))
    return {
        "out": str(out_path),
        "n_superframes": n_superframes,
        "n_anchors": n_anchors,
        "coverage": coverage,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def pivots_for_industry1(folder_name: str) -> tuple[str, ...] | None:
    """Per-folder pivot selection for industry1. Returns the tuple
    of pivot kinds to emit, or None to skip the folder entirely."""
    for prefix, pivots in INDUSTRY1_FOLDERS.items():
        if folder_name.startswith(prefix):
            return pivots
    # Unknown industry1 sub-folder — emit distance only.
    return ("distances",)


def process_root(root: Path,
                 default_pivots: tuple[str, ...],
                 out_root: Path) -> list[dict]:
    # If this is industry1, use the per-folder override map.
    # Otherwise, every sub-folder gets default_pivots.
    if root.name == "industry1":
        def pivots_for_folder(_root_name, folder_name, _dflt):
            return pivots_for_industry1(folder_name)
    else:
        def pivots_for_folder(_root_name, _folder_name, dflt):
            return dflt

    if not root.exists():
        print(f"[skip] {root} (not found)")
        return []
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    print(f"[root] {root}: {len(subdirs)} sub-folder(s)")
    results = []
    for folder in subdirs:
        scene = parse_scene_tag(folder.name)
        # Pick which pivots to emit for this sub-folder
        pivots = pivots_for_folder(root.name, folder.name, default_pivots)
        if not pivots:
            print(f"\n[{scene}] {folder.name}  [skip — no pivots configured]")
            continue
        print(f"\n[{scene}] {folder.name}")
        for kind in pivots:
            if kind == "distances":
                out = out_root / distance_filename(scene)
            elif kind == "rxpower":
                out = out_root / rxpower_filename(scene)
            else:
                raise ValueError(kind)
            stat = build_pivot(folder, kind=kind, out_path=out)
            stat["scene"] = scene
            stat["kind"] = kind
            results.append(stat)
            print(f"  {kind:9s} -> {out.name}  "
                  f"({stat['n_superframes']} superframes, "
                  f"{stat['n_anchors']} anchors, "
                  f"coverage {stat['coverage']*100:.1f}%)")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--industry1_root", type=Path, default=DEFAULT_INDUSTRY1_ROOT)
    p.add_argument("--industry2_root", type=Path, default=DEFAULT_INDUSTRY2_ROOT)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--force", action="store_true",
                   help="overwrite existing pivots (default: skip if file exists)")
    args = p.parse_args()

    print("=" * 70)
    print("build_industry_pivots")
    print(f"  industry1: {args.industry1_root}")
    print(f"  industry2: {args.industry2_root}")
    print(f"  out:       {args.out_root}")
    print("=" * 70)

    all_results: list[dict] = []
    all_results += process_root(args.industry1_root, INDUSTRY1_DEFAULT_PIVOTS, args.out_root)
    all_results += process_root(args.industry2_root, INDUSTRY2_PIVOTS, args.out_root)

    print()
    print("=" * 70)
    print(f"done: {len(all_results)} pivot file(s) generated")
    for r in all_results:
        print(f"  {r['kind']:9s}  {r['out']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
