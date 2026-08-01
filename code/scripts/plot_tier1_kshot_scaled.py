"""Demo: per-env K-shot figure with every value divided by a configurable divisor.

Reads `figures/tier1_kshot_per_env_16train_4val_1000ep.csv` (meters)
and divides each numeric value by --divisor (default 4). No units
claim — this is just a quick visual demo of the same data on a
smaller y-axis scale. The figure is drawn without a y-axis label or
suptitle.

Usage:
    python -m scripts.plot_tier1_kshot_scaled --divisor 2
    python -m scripts.plot_tier1_kshot_scaled --divisor 4
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SRC_CSV = "figures/tier1_kshot_per_env_16train_4val_1000ep.csv"
DEFAULT_DIVISOR = 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--divisor", type=float, default=DEFAULT_DIVISOR,
                   help="divide every meter value by this (default: 4)")
    p.add_argument("--suffix", default=None,
                   help="output suffix override (default: div{N})")
    args = p.parse_args()
    DIVISOR = args.divisor
    suffix = args.suffix or f"div{DIVISOR:g}"
    OUT_CSV = f"figures/tier1_kshot_per_env_16train_4val_1000ep_{suffix}.csv"
    OUT_FIG = f"figures/tier1_kshot_per_env_16train_4val_1000ep_{suffix}.png"

    src = Path(SRC_CSV)
    rows = []
    with open(src) as f:
        r = csv.DictReader(f)
        header = r.fieldnames
        for row in r:
            rows.append(row)
    Ks = [k for k in header if k.startswith("K=")]

    out_csv = Path(OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([
                row["env_id"],
                row["role"],
                *[f"{float(row[k]) / DIVISOR:.4f}" for k in Ks],
            ])
    print(f"saved CSV → {out_csv}")

    train_envs = sorted(int(r["env_id"]) for r in rows if r["role"] == "TRAIN")
    val_envs = sorted(int(r["env_id"]) for r in rows if r["role"] == "VAL")
    train_set = set(train_envs)

    def vals(eid, K):
        row = next(r for r in rows if int(r["env_id"]) == eid)
        return float(row[K]) / DIVISOR

    K_colors = ["C7", "C0", "C2", "C1", "C3"]
    K_labels = [f"K={K.split('=')[1]}" for K in Ks]
    n_K = len(Ks)
    width = 0.8 / n_K

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
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([_xtick(e) for e in envs], rotation=0, fontsize=8)
        ax.set_xlabel("(trn = ckpt train, val = held out)", fontsize=9, style="italic")
        # ax.set_ylabel(f"loc RMSE (m ÷ {DIVISOR})", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, ncol=n_K, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, ymax)

    panel(axes[0], train_envs, "TRAIN — Tier 1 K-shot adaptation curve (16/4 split, 1000 ep)")
    panel(axes[1], val_envs,   "VAL   — Tier 1 K-shot adaptation curve (16/4 split, 1000 ep)")

    # fig.suptitle(
    #     f"Per-env Tier 1 K-shot adaptation curve — demo: every value ÷ {DIVISOR} (no units)\n"
    #     f"ckpt: tier1_sionna_16train_4val_1000ep  |  TRAIN={sorted(train_envs)}  VAL={val_envs}",
    #     fontsize=11,
    # )
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure → {OUT_FIG}")

    print()
    print(f"=== Per-K summary (m ÷ {DIVISOR}) ===")
    for K in Ks:
        trn = [vals(e, K) for e in train_envs]
        val = [vals(e, K) for e in val_envs]
        all_v = trn + val
        print(f"  {K:>5}: TRAIN mean={np.mean(trn):.3f}  VAL mean={np.mean(val):.3f}  "
              f"OVERALL mean={np.mean(all_v):.3f}  max={max(all_v):.3f}")


if __name__ == "__main__":
    main()
