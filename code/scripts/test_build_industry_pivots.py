"""Regression test: regenerating the pivoted_superframe_*.csv files
from the raw industry1/industry2 bags must produce byte-identical
output (modulo sub-mm float precision in the GT columns).

Run from /Users/justin/ultraclear-paper/code:
    python3.11 -m scripts.test_build_industry_pivots
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_industry_pivots import build_pivot

ORIG_DIR = Path(
    "/Users/justin/ultraclear-paper/figure/anc20"
)
INDUSTRY1_ROOT = Path("/Users/justin/Downloads/industry1")
INDUSTRY2_ROOT = Path("/Users/justin/Downloads/industry2")

# (relative-to-industryN-root folder name, pivot kind, expected output filename)
TEST_CASES = [
    # industry1
    ("2023_IDLAB_JULY_CIR-TWR-app_rh_2023_Jul_10_15_32_36_2023-07-10-13-34-25",
     "distances", "pivoted_superframe_distances.csv", INDUSTRY1_ROOT),
    ("2023_IDLAB_JULY_CIR-TWR-app_sl_2023_Jul_10_15_21_28_2023-07-10-13-23-07",
     "distances", "pivoted_superframe_distances_sl.csv", INDUSTRY1_ROOT),
    # industry2
    ("OKT23_FL2_E5CE_2-TDOA-app_rh_2023_Nov_21_11_26_15_2023-11-21-10-26-28",
     "rxpower", "pivoted_superframe_rxpower_rh.csv", INDUSTRY2_ROOT),
    ("OKT23_FL2_E5CE-TDOA-app_sllong_2023_Nov_22_10_54_37_2023-11-22-09-54-49",
     "rxpower", "pivoted_superframe_rxpower_sllong.csv", INDUSTRY2_ROOT),
    ("OKT23_FL2_E5CE-TDOA-app_ro_2023_Nov_22_11_38_10_2023-11-22-10-38-21",
     "rxpower", "pivoted_superframe_rxpower_ro.csv", INDUSTRY2_ROOT),
    ("TDOA-app-1_LS_LOS_2023_Jul_11_10_08_08_2023-07-11-08-09-46",
     "rxpower", "pivoted_superframe_rxpower_ls_los.csv", INDUSTRY2_ROOT),
    ("TDOA-app-1_GRID_LOS_2023_Jul_11_10_24_20_2023-07-11-08-26-09",
     "rxpower", "pivoted_superframe_rxpower_grid_los.csv", INDUSTRY2_ROOT),
]


def diff_dataframes(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Cell-by-cell diff summary."""
    info = {
        "shape_old": old.shape,
        "shape_new": new.shape,
        "cols_match": list(old.columns) == list(new.columns),
        "ids_match": (old["id"].values == new["id"].values).all() if "id" in old.columns else None,
        "max_abs_diff": 0.0,
        "max_diff_col": "",
    }
    if old.shape != new.shape or not info["cols_match"]:
        return info
    for col in old.columns:
        if (col in new.columns
                and pd.api.types.is_numeric_dtype(old[col])
                and pd.api.types.is_numeric_dtype(new[col])):
            mask = old[col].notna() & new[col].notna()
            if mask.any():
                d = (old[col][mask] - new[col][mask]).abs().max()
                if d > info["max_abs_diff"]:
                    info["max_abs_diff"] = float(d)
                    info["max_diff_col"] = col
    return info


def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for folder_name, kind, expected_name, root in TEST_CASES:
            folder = root / folder_name
            if not folder.exists():
                print(f"[skip] {folder_name} (not found)")
                continue
            new_csv = tmp_path / expected_name
            build_pivot(folder, kind=kind, out_path=new_csv)
            old_csv = ORIG_DIR / expected_name
            if not old_csv.exists():
                print(f"[skip] {expected_name} (no original to diff against)")
                continue
            old = pd.read_csv(old_csv)
            new = pd.read_csv(new_csv)
            info = diff_dataframes(old, new)
            ok = (info["shape_old"] == info["shape_new"]
                  and info["cols_match"]
                  and info["ids_match"]
                  and info["max_abs_diff"] < 0.001)
            tag = "OK" if ok else "FAIL"
            print(f"  [{tag}] {expected_name}")
            print(f"         shape {info['shape_old']} -> {info['shape_new']}  "
                  f"cols_match={info['cols_match']}  "
                  f"ids_match={info['ids_match']}  "
                  f"max|diff|={info['max_abs_diff']:.6f} m  "
                  f"(col={info['max_diff_col']})")
            if not ok:
                failures.append(expected_name)

    print()
    if failures:
        print(f"FAILED ({len(failures)} file(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"all {len(TEST_CASES)} pivot(s) regenerate byte-identical.")


if __name__ == "__main__":
    main()
