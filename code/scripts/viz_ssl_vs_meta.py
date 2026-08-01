"""Two-panel figure: SSL pretraining (Tier 0) and FOMAML meta-learning (Tier 1).

Left:  SSL reconstruction loss per epoch, train + val, 4-scene Sionna mix.
Right: FOMAML per-episode z-score RMSE location error, raw + rolling mean.

Reads:
  - checkpoints/tier0_multiscene_f2_3to1_100ep/tier0_history.csv
  - figures/tier1_maml_history.csv  (filters phase == train_loc_err_z_rmse)

Writes:
  - figures/ssl_vs_meta_training.png
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        cols: dict[str, list] = {h: [] for h in header}
        for row in rdr:
            for h, v in zip(header, row):
                if v == "" or v == "nan":
                    continue
                try:
                    cols[h].append(float(v))
                except ValueError:
                    pass
    return {h: np.asarray(v) for h, v in cols.items()}


def load_tier0(ckpt_dir: Path) -> dict:
    d = load_csv(ckpt_dir / "tier0_history.csv")
    return {"epoch": d["epoch"], "train": d["train_loss"], "val": d["val_loss"]}


def load_tier1(csv_path: Path) -> dict:
    """Filter the multi-purpose Tier 1 history to train_loc_err_z_rmse only."""
    with open(csv_path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        ep_col = header.index("episode")
        phase_col = header.index("phase")
        val_col = header.index("value")
    ep, val = [], []
    with open(csv_path) as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            if row[phase_col] != "train_loc_err_z_rmse":
                continue
            if row[val_col] == "" or row[val_col] == "nan":
                continue
            ep.append(float(row[ep_col]))
            val.append(float(row[val_col]))
    return {"episode": np.asarray(ep), "train_loc": np.asarray(val)}


def make_figure(out_path: Path, tier0: dict, tier1: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    fig.suptitle(
        "SSL pretraining vs. FOMAML meta-learning",
        fontsize=14, y=1.04,
    )

    # ---- Left: Tier 0 SSL ------------------------------------------------
    ax = axes[0]
    ep = tier0["epoch"]; tr = tier0["train"]; va = tier0["val"]
    ax.plot(ep, tr, color="tab:blue", lw=1.6, label="train (SSL recon)")
    ax.plot(ep, va, color="tab:orange", lw=1.6, linestyle="--",
            label="val (held-out scenes)")
    if len(va) > 0 and not np.isnan(va[-1]):
        ax.axhline(va[-1], color="tab:orange", alpha=0.2, linestyle=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("SSL reconstruction loss")
    ax.set_title("Tier 0: SSL pretraining\n"
                 "(4-scene Sionna mix, 25% mask, F=2 channels)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")

    # ---- Right: Tier 1 FOMAML -------------------------------------------
    ax = axes[1]
    ep = tier1["episode"]; tr = tier1["train_loc"]
    window = 50
    if len(tr) > window:
        kernel = np.ones(window) / window
        tr_smooth = np.convolve(tr, kernel, mode="valid")
        ep_smooth = ep[window - 1:]
    else:
        tr_smooth = tr; ep_smooth = ep
    ax.plot(ep, tr, color="tab:blue", alpha=0.25, lw=0.7, label="per-episode")
    ax.plot(ep_smooth, tr_smooth, color="tab:blue", lw=1.8,
            label=f"rolling mean (w={window})")
    ax.set_xlabel("episode")
    ax.set_ylabel("train loc err (z-score RMSE)")
    ax.set_title("Tier 1: FOMAML outer loop\n"
                 "(per-episode meta-loss on multi-scene FOMAML)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code_dir", default="/Users/justin/ultraclear-paper/code")
    p.add_argument("--out", default="figures/ssl_vs_meta_training.png")
    args = p.parse_args()

    code = Path(args.code_dir)
    tier0 = load_tier0(code / "checkpoints/tier0_multiscene_f2_3to1_100ep")
    tier1 = load_tier1(code / "figures/tier1_maml_history.csv")

    out_path = code / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure(out_path, tier0, tier1)


if __name__ == "__main__":
    main()
