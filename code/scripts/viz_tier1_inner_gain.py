"""Plot per-episode pre-adapt query loss vs. post-adapt (outer) query loss.

Reads the inner-loss history CSV written by
`scripts/train_tier1_sionna_inner.py` and shows, per episode, how
much the inner-loop adaptation actually buys you.

  - `pre_q_loss`  : query loss BEFORE any inner adaptation
  - `outer_q_loss`: query loss AFTER `maml_inner_steps` of inner
                    adaptation (this is what FOMAML backprops on)
  - inner-support trajectory: optional secondary panel

Usage:
    python3.11 -m scripts.viz_tier1_inner_gain \\
        --history figures/tier1_inner_test_history.csv \\
        --out figures/tier1_inner_gain.png
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_history(path: Path) -> dict:
    with open(path) as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    cols: dict[str, list] = {}
    for r in rows:
        for k, v in r.items():
            cols.setdefault(k, []).append(float(v) if v != "" else float("nan"))
    return {k: np.asarray(v) for k, v in cols.items()}


def make_figure(out_path: Path, h: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    fig.suptitle(
        "FOMAML inner-loop gain: pre-adapt vs. post-adapt query loss",
        fontsize=14, y=1.04,
    )

    ep = h["episode"]
    pre = h["pre_q_loss"]
    outer = h["outer_q_loss"]
    gain = pre - outer
    rel_gain = np.where(pre > 1e-9, (pre - outer) / np.maximum(pre, 1e-9) * 100.0, 0.0)

    # ---- Left: pre vs outer per episode ---------------------------------
    ax = axes[0]
    ax.plot(ep, pre, color="tab:gray", lw=1.6, linestyle="--",
            label="pre-adapt query loss")
    ax.plot(ep, outer, color="tab:blue", lw=1.8,
            label="post-adapt query loss (FOMAML outer target)")
    ax.fill_between(ep, outer, pre, where=(pre > outer),
                     color="tab:green", alpha=0.15,
                     label="inner-loop gain")
    ax.set_xlabel("episode")
    ax.set_ylabel("query MSE (z-scored xy)")
    ax.set_title("Per-episode: pre-adapt vs. post-adapt")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")

    # ---- Right: gain (absolute + relative) ------------------------------
    ax = axes[1]
    ax.plot(ep, gain, color="tab:green", lw=1.6,
            label="absolute gain (pre - outer)")
    ax2 = ax.twinx()
    ax2.plot(ep, rel_gain, color="tab:purple", lw=1.6, linestyle=":",
              label="relative gain (%)")
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
    ax.set_xlabel("episode")
    ax.set_ylabel("absolute gain (pre - outer, MSE units)",
                   color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax2.set_ylabel("relative gain (%)", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_title("Inner-loop gain over training")
    # Combined legend from both axes (filter out non-data artists
    # like axhline so they don't leak in as orphan entries).
    def _data_lines(axis):
        out = []
        for ln in axis.get_lines():
            lbl = ln.get_label()
            if lbl.startswith("_"):
                continue
            out.append(ln)
        return out
    lns = _data_lines(ax) + _data_lines(ax2)
    ax.legend(lns, [l.get_label() for l in lns], loc="upper right",
               fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")
    # Y-axis padding for absolute gain
    if len(gain) > 0:
        pad = 0.05 * (gain.max() - gain.min() + 1e-9)
        ax.set_ylim(gain.min() - pad, gain.max() + pad)
    if len(rel_gain) > 0:
        ax2.set_ylim(0, max(100, rel_gain.max() * 1.1))

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True,
                   help="path to tier1_inner_*_history.csv")
    p.add_argument("--out", default="figures/tier1_inner_gain.png")
    args = p.parse_args()

    h = load_history(Path(args.history))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure(out_path, h)


if __name__ == "__main__":
    main()
