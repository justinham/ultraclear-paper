"""Top-5 best envs from the K-shot per-env figure.

Reads `figures/tier1_kshot_per_env.csv` and produces a 2-subplot
figure showing the 5 best TRAIN envs and 5 best VAL envs by
their K=10 loc RMSE (the headline metric). Bars are grouped by K,
so each subplot shows the full K=0..20 curve for the top 5 envs.

Usage:
    python -m scripts.viz_tier1_top5
    python -m scripts.viz_tier1_top5 --csv ... --out_fig ...
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="figures/tier1_kshot_per_env.csv")
    p.add_argument("--out_fig", default="figures/tier1_top5_per_env.png")
    p.add_argument("--top_k", type=int, default=3,
                   help="How many best envs to show per side (default 3)")
    p.add_argument("--rank_by", default="K=10",
                   help="Which K column to rank by (default K=10)")
    p.add_argument("--ks", nargs="+", type=int, default=None,
                   help="Which K columns to plot (default: all K= columns in CSV)")
    args = p.parse_args()

    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            rows.append({
                "env_id": int(r["env_id"]),
                "role": r["role"],
                **{k: float(v) for k, v in r.items() if k.startswith("K=")},
            })
    Ks = sorted(
        (k for k in rows[0] if k.startswith("K=")),
        key=lambda k: int(k.split("=")[1]),
    )
    if args.ks is not None:
        wanted = {f"K={k}" for k in args.ks}
        missing = wanted - set(Ks)
        if missing:
            raise SystemExit(
                f"--ks {args.ks} requested but CSV is missing: {sorted(missing)}; "
                f"re-run eval_tier1_kshot with these K values first."
            )
        Ks = [k for k in Ks if k in wanted]
    if args.rank_by not in rows[0]:
        raise SystemExit(f"--rank_by={args.rank_by!r} not in CSV (have {list(rows[0])})")
    if args.rank_by not in Ks:
        raise SystemExit(f"--rank_by={args.rank_by!r} is not in --ks {args.ks}; ranking and plotting use the same set")

    train_rows = [r for r in rows if r["role"] == "TRAIN"]
    val_rows = [r for r in rows if r["role"] == "VAL"]
    train_best = sorted(train_rows, key=lambda r: r[args.rank_by])[:args.top_k]
    val_best = sorted(val_rows, key=lambda r: r[args.rank_by])[:args.top_k]
    print(f"top {args.top_k} TRAIN (by {args.rank_by}): {[r['env_id'] for r in train_best]}")
    print(f"top {args.top_k} VAL   (by {args.rank_by}): {[r['env_id'] for r in val_best]}")

    # Per-env label combining env_id and the train/val membership
    # on a single line, e.g. "env #3 (val)" — keeps the x-axis
    # compact while still telling the reader which envs the FOMAML
    # has seen during meta-training.
    train_scenes = set(r["env_id"] for r in rows if r["role"] == "TRAIN")

    def _xtick(eid):
        if eid in train_scenes:
            tag = "trn"
        else:
            tag = "val"
        return f"env #{eid} ({tag})"

    K_colors = ["C7", "C0", "C2", "C1", "C3", "C5", "C4"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, side_rows, side_label in [
        (axes[0], train_best, "TRAIN"),
        (axes[1], val_best, "VAL"),
    ]:
        n_envs = len(side_rows)
        n_K = len(Ks)
        x_pos = np.arange(n_envs)
        width = 0.8 / n_K
        for ki, K in enumerate(Ks):
            vals = [r[K] for r in side_rows]
            offset = (ki - (n_K - 1) / 2) * width
            bars = ax.bar(x_pos + offset, vals, width, color=K_colors[ki % len(K_colors)],
                          label=K, edgecolor="white", linewidth=0.3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                         ha="center", va="bottom", fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [_xtick(r["env_id"]) for r in side_rows],
            fontsize=15,
        )
        ax.set_ylabel("loc RMSE (m)", fontsize=18, labelpad=10)
        # Side label: "TRAIN MACRO ENV" = the FOMAML training set
        # (the model has seen these during MAML outer-loop training);
        # "UNSEEN MACRO ENV" = held-out envs the FOMAML has not seen
        # at training time and must adapt to via K-shot at eval.
        if side_label == "TRAIN":
            title_side = "TRAIN MACRO ENV"
        else:
            title_side = "UNSEEN MACRO ENV"
        ax.set_title(title_side, fontsize=15)
        ax.legend(fontsize=13, ncol=n_K, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        all_vals = [v for r in side_rows for v in [r[K] for K in Ks]]
        ax.set_ylim(0, max(all_vals) * 1.2)
        ax.tick_params(axis="y", labelsize=15)

    fig.suptitle(
        f"Per-env Tier 1 K-shot adaptation curve — top {args.top_k} best envs from TRAIN and VAL "
        f"(ranked by {args.rank_by} RMSE)",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure → {args.out_fig}")


if __name__ == "__main__":
    main()
