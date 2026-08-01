"""Draw the per-env K-shot figure with the y-axis converted from meters to inches.

Reads `figures/tier1_kshot_per_env_16train_4val_1000ep.csv` (meters),
multiplies each numeric value by 39.37 (1 m = 39.37 in), and writes:
  - figures/tier1_kshot_per_env_16train_4val_1000ep_in.csv
  - figures/tier1_kshot_per_env_16train_4val_1000ep_in.png

Mirrors the structure of `scripts/eval_tier1_kshot.py` so the two
figures can be dropped into the same comparison document.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.config import Config
from ultraclear.tier1_maml import MAMLCore
from scripts.train_tier1_sionna import sample_task_multi
from scripts.multi_scene_dataset import build_default_multi_scene


M_TO_IN = 39.37
SRC_CSV = "figures/tier1_kshot_per_env_16train_4val_1000ep.csv"
SRC_CKPT = "checkpoints/tier1_sionna_16train_4val_1000ep/tier1_sionna.pt"
OUT_CSV = "figures/tier1_kshot_per_env_16train_4val_1000ep_in.csv"
OUT_FIG = "figures/tier1_kshot_per_env_16train_4val_1000ep_in.png"


def main():
    # ---- Load source CSV (meters) ----
    src = Path(SRC_CSV)
    rows = []
    with open(src) as f:
        r = csv.DictReader(f)
        header = r.fieldnames  # ['env_id','role','K=0','K=1','K=5','K=10','K=20']
        for row in r:
            rows.append(row)
    Ks = [k for k in header if k.startswith("K=")]

    # ---- Write inches CSV ----
    out_csv = Path(OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([
                row["env_id"],
                row["role"],
                *[f"{float(row[k]) * M_TO_IN:.4f}" for k in Ks],
            ])
    print(f"saved CSV → {out_csv}")

    # ---- Build plot data ----
    train_envs = [int(r["env_id"]) for r in rows if r["role"] == "TRAIN"]
    val_envs = [int(r["env_id"]) for r in rows if r["role"] == "VAL"]
    train_envs.sort()
    val_envs.sort()

    def vals(eid, K):
        row = next(r for r in rows if int(r["env_id"]) == eid)
        return float(row[K]) * M_TO_IN

    K_colors = ["C7", "C0", "C2", "C1", "C3"]
    K_labels = [f"K={K.split('=')[1]}" for K in Ks]
    n_K = len(Ks)
    width = 0.8 / n_K

    # Mirror env tag convention from eval_tier1_kshot.py
    train_set = set(train_envs)
    def _xtick(eid):
        tag = "trn" if eid in train_set else "val"
        return f"e{eid}\n{tag}"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    ymax = max(vals(int(r["env_id"]), K)
               for r in rows for K in Ks) * 1.15

    def panel(ax, envs, title):
        x_pos = np.arange(len(envs))
        for ki, K in enumerate(Ks):
            vs = [vals(e, K) for e in envs]
            offset = (ki - (n_K - 1) / 2) * width
            bars = ax.bar(x_pos + offset, vs, width,
                          color=K_colors[ki % len(K_colors)],
                          label=K_labels[ki], edgecolor="white", linewidth=0.3)
            if K == "K=10":
                for b, v in zip(bars, vs):
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                            ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([_xtick(e) for e in envs], rotation=0, fontsize=8)
        ax.set_xlabel("(trn = ckpt train, val = held out)", fontsize=9, style="italic")
        ax.set_ylabel("loc RMSE (in)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, ncol=n_K, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, ymax)

    panel(axes[0], train_envs, "TRAIN — Tier 1 K-shot adaptation curve (16/4 split, 1000 ep)")
    panel(axes[1], val_envs,   "VAL   — Tier 1 K-shot adaptation curve (16/4 split, 1000 ep)")

    fig.suptitle(
        f"Per-env Tier 1 K-shot adaptation curve — converted meters → inches (×{M_TO_IN})\n"
        f"ckpt: tier1_sionna_16train_4val_1000ep  |  TRAIN={sorted(train_envs)}  VAL={val_envs}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure → {OUT_FIG}")

    # ---- Quick summary ----
    print()
    print("=== Per-K summary (inches) ===")
    for K in Ks:
        trn = [vals(e, K) for e in train_envs]
        val = [vals(e, K) for e in val_envs]
        all_v = trn + val
        print(f"  {K:>5}: TRAIN mean={np.mean(trn):.1f}in  VAL mean={np.mean(val):.1f}in  "
              f"OVERALL mean={np.mean(all_v):.1f}in  max={max(all_v):.1f}in")


if __name__ == "__main__":
    main()
