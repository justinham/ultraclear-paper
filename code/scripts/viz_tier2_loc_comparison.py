"""Tier 2 + joint — Headline numbers: location error comparison.

The paper's main Tier 2 result. Compares per-cond location error on
the val set (unseen envs) for:
  - baseline:   no Tier 2 (ASTA encoder + location head, supervised)
  - tier2:      full DisentangledModel + FiLM + location head
                (standalone Tier 2 training)
  - joint:      full DisentangledModel after joint Tier 1+2 training
                (phase C of train_tier12_joint.py)

Each per-cond CSV has columns: cond_id, baseline_loc_err, tier2_loc_err
(standalone) or cond_id, baseline_loc_err, joint_loc_err (joint).

Usage:
  python -m scripts.viz_tier2_loc_comparison                          # standalone Tier 2 vs baseline
  python -m scripts.viz_tier2_loc_comparison --joint                  # joint vs baseline
  python -m scripts.viz_tier2_loc_comparison --compare_all            # 3-way comparison
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scripts.viz_tier2_helpers import TIER2_FIG, JOINT_FIG


# Cond/bucket labels. The synthetic data uses cond_id 0/1/2 to mean
# "mostly LOS / moderate NLOS / heavy NLOS" (matches data.py _micro_params).
# The Sionna real-data path uses cond_id 0/1/2 to mean NLOS-rate buckets
# (low <0.3, mid <0.7, high >=0.7). The cond_id column in the per-cond
# CSV doesn't tell us which is which, so we use neutral labels that are
# accurate for both interpretations.
COND_LABELS = {
    0: "cond 0\n(bucket 0)",
    1: "cond 1\n(bucket 1)",
    2: "cond 2\n(bucket 2)",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per_cond", type=str, default="checkpoints/tier2_synthetic/tier2_per_cond.csv",
                   help="standalone Tier 2 per-cond CSV")
    p.add_argument("--joint_per_cond", type=str, default="checkpoints/joint/phase_C_per_cond.csv",
                   help="joint phase C per-cond CSV")
    p.add_argument("--out", type=str, default=f"{TIER2_FIG}/loc_error_comparison.png")
    p.add_argument("--joint", action="store_true",
                   help="use the joint per-cond CSV instead of standalone")
    p.add_argument("--compare_all", action="store_true",
                   help="3-way comparison: baseline, standalone Tier 2, joint")
    args = p.parse_args()

    if args.compare_all:
        tier2_rows = _read_per_cond(args.per_cond, key="tier2_loc_err")
        joint_rows = _read_per_cond(args.joint_per_cond, key="joint_loc_err")
        merged = {}
        for r in tier2_rows:
            merged[r["cond_id"]] = {"cond_id": r["cond_id"],
                                    "baseline": r["baseline"],
                                    "tier2": r["tier2"]}
        for r in joint_rows:
            if r["cond_id"] not in merged:
                merged[r["cond_id"]] = {"cond_id": r["cond_id"], "baseline": r["baseline"]}
            merged[r["cond_id"]]["joint"] = r["tier2"]
        merged = {k: v for k, v in merged.items() if "tier2" in v and "joint" in v}
        rows = [{"cond_id": v["cond_id"], "baseline": v["baseline"],
                 "tier2": v["tier2"], "joint": v["joint"]}
                for v in sorted(merged.values(), key=lambda x: x["cond_id"])]
        if "tier2_synthetic" in args.out:
            args.out = args.out.replace("tier2_synthetic", "joint")
        _bar_plot_3way(rows, args.out)
        return

    if args.joint:
        if "tier2_synthetic" in args.out:
            args.out = args.out.replace("tier2_synthetic", "joint")
        rows = _read_per_cond(args.joint_per_cond, key="joint_loc_err")
        _bar_plot_2way(rows, args.out, methods=["baseline", "joint"],
                       title_suffix=" (joint Tier 1+2)", data_source="sionna")
        return

    rows = _read_per_cond(args.per_cond, key="tier2_loc_err")
    _bar_plot_2way(rows, args.out, methods=["baseline", "tier2"], title_suffix="")


def _read_per_cond(path: str, key: str) -> list[dict]:
    """Read a per-cond CSV. Returns list of {cond_id, baseline, tier2} dicts."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {"cond_id": int(r["cond_id"])}
            row["baseline"] = float(r["baseline_loc_err"]) if r.get("baseline_loc_err") not in ("NA", "", "nan", None) else float("nan")
            row["tier2"] = float(r[key]) if r.get(key) not in ("NA", "", "nan", None) else float("nan")
            rows.append(row)
    return rows


def _bar_plot_2way(rows: list[dict], out_path: str,
                   methods: list[str], title_suffix: str = "",
                   data_source: str = "synthetic") -> None:
    """Two-bar comparison."""
    conds = [r["cond_id"] for r in rows]
    baseline = [r["baseline"] for r in rows]
    method_vals = [r["tier2"] for r in rows]
    method_label = "joint (Tier 1+2)" if "joint" in methods else "Tier 2 (DisentangledModel)"

    # Pick bucket labels that match the data source.
    if data_source == "sionna":
        bucket_labels = {
            0: "low NLOS\n(<30% cells)",
            1: "mid NLOS\n(30-70% cells)",
            2: "high NLOS\n(>70% cells)",
        }
    else:
        bucket_labels = COND_LABELS

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(conds))
    width = 0.35
    ax.bar(x - width/2, baseline, width, label="baseline (no Tier 2)", color="grey", alpha=0.8)
    ax.bar(x + width/2, method_vals, width, label=method_label, color="C0", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([bucket_labels.get(c, f"cond {c}") for c in conds], fontsize=10)
    ax.set_ylabel("mean Euclidean location error\n(val envs, lower = better)")
    data_label = "Sionna real data (NLOS-bucketed)" if data_source == "sionna" else "synthetic data"
    ax.set_title(f"location error on unseen environments\n"
                 f"{data_label}{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for xi, b, t in zip(x, baseline, method_vals):
        if not np.isnan(b) and b > 0:
            delta = (b - t) / b * 100
            label = f"{b:.2f}\n{t:.2f}\n({delta:+.1f}%)"
        else:
            label = f"NA\n{t:.2f}"
        ax.text(xi, max(b, t) + 0.05, label, ha="center", fontsize=8)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    print(f"\nPer-cond location error summary (val envs):")
    print(f"  {'cond':<6}  {'baseline':>10}  {method_label:>20}  {'Δ':>8}")
    for c, b, t in zip(conds, baseline, method_vals):
        if not np.isnan(b):
            print(f"  {c:<6}  {b:>10.3f}  {t:>20.3f}  {(t-b)/b*100:>+7.1f}%")


def _bar_plot_3way(rows: list[dict], out_path: str) -> None:
    """Three-bar comparison: baseline vs standalone Tier 2 vs joint."""
    conds = [r["cond_id"] for r in rows]
    baseline = [r["baseline"] for r in rows]
    tier2 = [r["tier2"] for r in rows]
    joint = [r["joint"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(conds))
    width = 0.27
    ax.bar(x - width, baseline, width, label="baseline (no Tier 2)", color="grey", alpha=0.8)
    ax.bar(x, tier2, width, label="Tier 2 (standalone)", color="C0", alpha=0.8)
    ax.bar(x + width, joint, width, label="Tier 1+2 (joint)", color="C3", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS.get(c, f"cond {c}") for c in conds])
    ax.set_ylabel("mean Euclidean location error\n(val envs, lower = better)")
    ax.set_title("3-way comparison: baseline vs Tier 2 vs Tier 1+2 (joint)\n"
                 "on unseen environments")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for xi, b, t, j in zip(x, baseline, tier2, joint):
        if not np.isnan(b) and b > 0:
            d_t = (b - t) / b * 100
            d_j = (b - j) / b * 100
            label = f"{b:.2f}\n{t:.2f} ({d_t:+.1f}%)\n{j:.2f} ({d_j:+.1f}%)"
        else:
            label = f"NA\n{t:.2f}\n{j:.2f}"
        ax.text(xi, max([b, t, j]) + 0.05, label, ha="center", fontsize=7)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    print(f"\n3-way per-cond location error summary (val envs):")
    print(f"  {'cond':<6}  {'baseline':>10}  {'tier2':>10}  {'joint':>10}")
    for c, b, t, j in zip(conds, baseline, tier2, joint):
        if not np.isnan(b):
            print(f"  {c:<6}  {b:>10.3f}  {t:>10.3f}  {j:>10.3f}")


if __name__ == "__main__":
    main()
