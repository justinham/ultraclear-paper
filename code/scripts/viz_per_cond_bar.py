"""Bar chart of per-cond val loc err for the 3-way joint training comparison.

Reads the per-cond CSVs from:
  - 7/8 per-window joint T1+2: /tmp/joint_with_baseline/phase_C_per_cond.csv
  - 7/9 per-timestep smoke (10 ep): hard-coded
  - 7/10 per-timestep full (bounded adv): checkpoints/.../phase_C_per_cond.csv
  - 7/10 baseline: checkpoints/.../baseline_per_cond.csv

Outputs a 3-bar-per-cond grouped bar chart to figures/joint_per_timestep/.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path("/Users/justin/ultraclear-paper/code")


def load_per_cond(path):
    out = {}
    if not Path(path).exists():
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            ci = int(row["cond_id"])
            for k, v in row.items():
                if k == "cond_id":
                    continue
                try:
                    out.setdefault(ci, {})[k] = float(v) if v not in ("NA", "", "nan", None) else float("nan")
                except ValueError:
                    pass
    return out


def main():
    baseline = load_per_cond(REPO / "checkpoints/joint_per_timestep_bounded_adv/baseline_per_cond.csv")
    joint_full = load_per_cond(REPO / "checkpoints/joint_per_timestep_bounded_adv/phase_C_per_cond.csv")
    joint_78 = load_per_cond("/tmp/joint_with_baseline/phase_C_per_cond.csv")
    # 7/9 smoke (10 phase-C episodes, per-timestep)
    joint_smoke = {0: 8.607, 1: 3.607, 2: 2.196}
    # 7/10 early-stop: best snapshot was ep 40, see phase_C_snapshots/phase_C_ep040.pt
    # (manually copied to phase_C_per_cond.csv format)
    import csv as _csv
    ep40 = {0: 8.61, 1: 2.80, 2: 1.99}

    cond_labels = ["low NLOS", "mid NLOS", "high NLOS"]
    conds = [0, 1, 2]
    variants = [
        ("Baseline (no T2, per-timestep)", baseline, "C0"),
        ("Joint T1+2, per-window, 7/8",   joint_78,   "C1"),
        ("Joint T1+2, per-timestep smoke, 7/9",  joint_smoke, "C2"),
        ("Joint T1+2, per-timestep full 100ep, 7/10 (bounded adv)", joint_full, "C3"),
        ("Joint T1+2, per-timestep EARLY-STOP ep40, 7/10", ep40, "C4"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_v = len(variants)
    width = 0.8 / n_v
    x = np.arange(len(conds))

    for i, (label, data, color) in enumerate(variants):
        # Detect format: 7/9 smoke is {ci: float}, others are {ci: {key: float}}.
        def get_v(d, ci):
            entry = d.get(ci, float("nan"))
            if isinstance(entry, dict):
                if "joint" in label.lower() and "baseline" not in label.lower():
                    return entry.get("joint_bucket_err", float("nan"))
                return entry.get("baseline_bucket_err", float("nan"))
            return entry
        vals = [get_v(data, ci) for ci in conds]
        offset = (i - n_v / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("val loc err (m, lower is better)")
    ax.set_title("Per-cond val loc err — joint Tier 1+2 ablations on 21-env Sionna")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    def ymax(d):
        vals = []
        for c in conds:
            entry = d.get(c, float("nan"))
            if isinstance(entry, dict):
                v = entry.get("joint_bucket_err", entry.get("baseline_bucket_err", 0))
            else:
                v = entry
            vals.append(v if not np.isnan(v) else 0)
        return max(vals) if vals else 0
    ax.set_ylim(0, max(ymax(d) for _, d, _ in variants) * 1.15)

    out = REPO.parent / "figures/joint_per_timestep/per_cond_val_loc_err.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
