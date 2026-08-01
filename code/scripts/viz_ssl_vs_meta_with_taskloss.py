"""Two-panel figure: SSL pretraining + FOMAML with per-task support losses.

Left  : Tier 0 SSL recon loss (train + val).
Right : Tier 1 FOMAML
          - meta loss (overall query loss, post-adaptation) — blue thick
          - one single-task support-loss curve per env — thin colored,
            one line per env across all the episodes it appears in.

The right panel uses the per-episode inner-loss CSV from
`scripts/train_tier1_sionna_inner.py`. Each episode samples a
subset of the train envs (size = cfg.n_way, default 4), so the
support-loss curve for a given env is only present in the
episodes where that env was sampled. Missing values are simply
not plotted for that episode.

Usage:
    python3.11 -m scripts.viz_ssl_vs_meta_with_taskloss \\
        --ssl_history checkpoints/tier0_multiscene_f2_3to1_100ep/tier0_history.csv \\
        --inner_history figures/tier1_inner_test_history.csv \\
        --out figures/ssl_vs_meta_with_taskloss.png
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Human-readable environment names (mirrors the sionna env metadata
# table: ID, Environment, Area (m^2), NLOS Rate). Used to label the
# per-env support-loss curves in the right-panel legend, so a reader
# can tell which curve is which without cross-referencing env IDs.
ENV_NAMES: dict[int, str] = {
    0:  "Indoor small room",
    1:  "Indoor large room",
    2:  "Indoor private garage",
    3:  "Indoor private garage",
    4:  "Public garage building #1",
    5:  "Public garage building #2",
    6:  "Public garage building #3",
    7:  "Public garage building #4",
    8:  "Public garage building #5",
    9:  "Outdoor roadside #1",
    10: "Outdoor roadside #2",
    11: "Outdoor roadside #3",
    12: "Outdoor roadside #4",
    13: "Outdoor roadside #5",
    14: "Outdoor parking lot #1",
    15: "Outdoor parking lot #2",
    16: "Outdoor parking lot #3",
    17: "Outdoor parking lot #4",
    18: "Outdoor parking lot #5",
    19: "Outdoor roadside",
    20: "Warehouse with corridor #1",
    21: "Warehouse with corridor #2",
    22: "Warehouse with corridor #3",
    23: "Warehouse with corridor #4",
}


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
    return {"epoch": d["epoch"], "train": d["train_loss"],
            "val": d["val_loss"]}


def load_tier1_per_task(csv_path: Path) -> dict:
    """Load the inner-loss CSV and pull out the per-env support-loss
    curves (final inner step by default).

    Returns dict with:
        episode        : (N,) outer-loop episode indices
        outer_q_loss   : (N,) meta-loss
        per_env        : dict[env_id] -> (M,) episode-aligned support
                         loss (with NaN where the env wasn't sampled)
    """
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows:
        raise ValueError(f"empty CSV: {csv_path}")

    # Find the per-env support-loss column for the final inner step.
    # The column naming is `inner_support_{i}_e{eid}`.
    inner_step_cols = sorted(
        [k for r in rows for k in r if k.startswith("inner_support_")],
        key=lambda s: int(s.split("_")[2]),
    )
    if not inner_step_cols:
        raise ValueError(
            f"no inner_support_* columns found in {csv_path}"
        )
    # Use the last step (most adapted).
    final_col_pool = {}
    for c in inner_step_cols:
        step = int(c.split("_")[2])
        eid = int(c.split("_e")[-1])
        if step == max(int(s.split("_")[2]) for s in inner_step_cols):
            final_col_pool[eid] = c
    if not final_col_pool:
        raise ValueError("could not parse any inner_support_*_e* columns")

    episodes = np.array([int(r["episode"]) for r in rows], dtype=float)
    outer = np.array([float(r["outer_q_loss"]) for r in rows], dtype=float)
    per_env: dict[int, np.ndarray] = {}
    for eid, col in final_col_pool.items():
        vals = []
        for r in rows:
            v = r.get(col, "")
            vals.append(float(v) if v != "" else np.nan)
        per_env[eid] = np.asarray(vals, dtype=float)

    return {
        "episode": episodes,
        "outer_q_loss": outer,
        "per_env": per_env,
        "n_inner_steps": max(int(s.split("_")[2]) for s in inner_step_cols) + 1,
    }


def _set_rc():
    """Bump rcParams for a readable single-figure output, then return the
    fig-level font sizes so subplots can override as needed."""
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "legend.title_fontsize": 13,
    })


def _reset_rc():
    """Restore rcParams so later figures aren't polluted."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
    })


def _draw_left(ax, tier0: dict) -> None:
    """Tier-0 SSL reconstruction loss (train + val)."""
    ep = tier0["epoch"]; tr = tier0["train"]; va = tier0["val"]
    ax.plot(ep, tr, color="tab:blue", lw=2.2, label="train (SSL recon)")
    ax.plot(ep, va, color="tab:orange", lw=2.2, linestyle="--",
            label="val (held-out scenes)")
    if len(va) > 0 and not np.isnan(va[-1]):
        ax.axhline(va[-1], color="tab:orange", alpha=0.2, linestyle=":")
    ax.set_xlabel("epoch", fontsize=15)
    ax.set_ylabel("SSL reconstruction loss", fontsize=15)
    ax.set_title("SSL encoder pre-training", fontsize=16)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=":")


def _draw_right(ax, tier1: dict, env_id_for_col: dict[int, int]) -> None:
    """Tier-1 FOMAML: meta loss + per-env support loss."""
    ep = tier1["episode"]
    outer = tier1["outer_q_loss"]
    per_env = tier1["per_env"]
    env_ids = sorted(per_env.keys())
    cmap = plt.get_cmap("tab10")

    # Meta loss: thick dark-navy line, deliberately distinct from
    # any colormap we'll use for per-task curves.
    ax.plot(ep, outer, color="#0a1f44", lw=2.6,
            label="overall query loss (meta loss)")

    # Per-task support loss: one thin line per env, drawn from a
    # color cycle that EXCLUDES the dark navy used for the meta
    # line. We build a 10-color list from tab10's non-blue slots
    # and append a few extra hues to handle >9 envs.
    exclude = {0}  # tab10[0] is the saturated matplotlib blue
    n_needed = len(env_ids)
    palette = [cmap(i) for i in range(10) if i not in exclude]
    if n_needed > len(palette):
        # Add a few extra colors from Set2 if we have more envs
        # than colors in the trimmed tab10.
        extra = list(plt.get_cmap("Set2").colors)
        palette = (palette + extra)[:n_needed]
    for idx, eid in enumerate(env_ids):
        vals = per_env[eid]
        mask = ~np.isnan(vals)
        if not mask.any():
            continue
        x = ep[mask]
        y = vals[mask]
        color = palette[idx % len(palette)]
        # `eid` here is the CSV column index (0..N-1). Map it to the
        # real sionna env id so the legend matches the mixed model's
        # TRAIN set ({0, 1, 2, 5, 9, 11, 13, 15, 17, 19}).
        real_eid = env_id_for_col.get(eid, eid)
        ax.plot(x, y, color=color, lw=1.4, alpha=0.85,
                linestyle=":",  # dotted, like the SSL val curve
                label=f"env{real_eid} ({ENV_NAMES.get(real_eid, '?')})")

    ax.set_xlabel("epoch", fontsize=15)
    ax.set_ylabel("query / support MSE", fontsize=15)
    ax.set_title("Multi-task Meta Training", fontsize=16)
    # Legend: inside the plot, upper-RIGHT. The lines start high on
    # the left and converge low on the right, so the upper-right
    # corner is empty space in the data.
    ax.legend(loc="upper right", fontsize=11, ncol=1, frameon=True)
    ax.grid(True, alpha=0.3, linestyle=":")


def make_figure(out_path: Path, tier0: dict, tier1: dict,
                env_id_for_col: dict[int, int],
                panels: str = "both") -> None:
    """Render one or both panels.

    panels:
      "left"  — Tier-0 SSL only
      "right" — Tier-1 FOMAML only
      "both"  — side-by-side (default; original behavior)
    """
    _set_rc()
    try:
        if panels == "both":
            fig, axes = plt.subplots(
                1, 2, figsize=(15, 4.8), constrained_layout=True
            )
            _draw_left(axes[0], tier0)
            _draw_right(axes[1], tier1, env_id_for_col)
        elif panels == "left":
            fig, ax = plt.subplots(
                1, 1, figsize=(7.5, 4.8), constrained_layout=True
            )
            _draw_left(ax, tier0)
        elif panels == "right":
            # Wider so the 11-entry legend (meta + 10 envs) doesn't
            # crowd the plot area.
            fig, ax = plt.subplots(
                1, 1, figsize=(8.5, 5.2), constrained_layout=True
            )
            _draw_right(ax, tier1, env_id_for_col)
        else:
            raise ValueError(
                f"unknown --panels value {panels!r} (use left|right|both)"
            )
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")
    finally:
        _reset_rc()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ssl_history",
                   default="checkpoints/tier0_multiscene_f2_3to1_100ep/tier0_history.csv")
    p.add_argument("--inner_history",
                   default="figures/tier1_inner_test_history.csv")
    p.add_argument("--out",
                   default="figures/ssl_vs_meta_with_taskloss.png")
    p.add_argument("--train_env_ids", type=int, nargs="+", default=None,
                   help="Real sionna env IDs for the 10 train scenes in "
                        "the order they appear as CSV columns _e0.._e9. "
                        "Default: [0,1,2,5,9,11,13,15,17,19] (the mixed "
                        "model's TRAIN set).")
    p.add_argument("--panels", choices=("both", "left", "right"),
                   default="both",
                   help="Which panel(s) to render. 'both' = the original "
                        "side-by-side figure; 'left' = SSL pre-training "
                        "only; 'right' = FOMAML meta + per-env support "
                        "loss only.")
    args = p.parse_args()

    code = Path("/Users/justin/ultraclear-paper/code")
    tier0 = load_tier0(code / "checkpoints/tier0_multiscene_f2_3to1_100ep")
    tier1 = load_tier1_per_task(code / args.inner_history)

    # Map CSV column index (0..N-1) -> real sionna env id, for
    # legend labels. Without this the figure would call every curve
    # "env0..env9", which is meaningless when the CSV was generated
    # with a non-trivial env-id ordering.
    if args.train_env_ids is None:
        args.train_env_ids = [0, 1, 2, 5, 9, 11, 13, 15, 17, 19]
    env_id_for_col = {i: eid for i, eid in enumerate(args.train_env_ids)}

    out_path = code / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure(out_path, tier0, tier1, env_id_for_col,
                panels=args.panels)


if __name__ == "__main__":
    main()
