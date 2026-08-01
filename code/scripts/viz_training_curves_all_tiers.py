"""All-tiers training-curve figure.

One figure, four subplots (one per tier), each showing the most
informative training-curve view of that tier:

  - Tier 0: SSL pretraining. Train + val reconstruction loss per epoch
    on the 4-scene Sionna mix.
  - Tier 1: FOMAML outer-loop meta-loss per episode (train) + best
    K-shot RMSE on a held-out val env. Shows the meta-learning
    trajectory.
  - Tier 2: Two stacked axes. (a) task loss + val task loss on the
    primary objective, (b) the disentanglement diagnostics (adversary
    CE approaching chance, env-probe accuracy).
  - Joint Tier 1+2 trainer: meta-loss + disentangle head losses +
    env-probe accuracy. Shows the joint optimization curve.

Each subplot is annotated with the key event (e.g. λ ramp start
on the Tier 2 panel, the K-shot inflection on Tier 1).

Output:
  - figures/training_curves_all_tiers.png
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- data loading ----------------------------------------------------------

def load_csv(path: Path, header_row: int = 0) -> dict[str, np.ndarray]:
    """Read a CSV with a header row, return {col_name: np.ndarray}."""
    with open(path) as f:
        rdr = csv.reader(f)
        for _ in range(header_row):
            next(rdr)
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


# --- per-tier loaders ------------------------------------------------------

def load_tier0(ckpt_dir: Path) -> dict:
    """Tier 0 SSL pretraining. train/val loss per epoch."""
    csv_path = ckpt_dir / "tier0_history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Re-run train_tier0_sionna_multiscene "
            f"to produce it.")
    d = load_csv(csv_path)
    return {"epoch": d["epoch"], "train": d["train_loss"], "val": d["val_loss"]}


def load_tier1(csv_path: Path) -> dict:
    """Tier 1 FOMAML. We have per-episode entries in the multi-purpose
    history CSV with `phase` column (`train_loc_err_z_rmse` etc.).
    Aggregate to per-episode mean and overlay the K-shot val RMSE
    points we have separately.
    """
    d = load_csv(csv_path)
    return {"episode": d["episode"], "train_loc": d["value"]}


def load_tier2(history_path: Path) -> dict:
    """Tier 2 v2 (sionna) or frozen-tier1 (synthetic). Same column
    shape, so a single loader works for both."""
    d = load_csv(history_path)
    return d


def load_joint(history_path: Path) -> dict:
    d = load_csv(history_path)
    return d


# --- the figure ------------------------------------------------------------

def make_figure(out_path: Path,
                 tier0: dict,
                 tier1: dict,
                 tier2: dict,
                 joint: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "UltraClear training dynamics, per tier\n"
        "(each panel = one tier; y-axis is the relevant loss for that stage)",
        fontsize=14, y=1.02,
    )

    # ---- Tier 0 ----------------------------------------------------------
    ax = axes[0, 0]
    ep = tier0["epoch"]; tr = tier0["train"]; va = tier0["val"]
    ax.plot(ep, tr, color="tab:blue", label="train (SSL recon)", lw=1.5)
    ax.plot(ep, va, color="tab:orange", label="val (held-out scenes)",
            lw=1.5, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("SSL reconstruction loss")
    ax.set_title("Tier 0: SSL pretraining\n"
                 "(4-scene Sionna mix, 25% mask, F=2 channels)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")
    # Annotate convergence region
    if len(va) > 10 and not np.isnan(va[-1]):
        ax.axhline(va[-1], color="tab:orange", alpha=0.2, linestyle=":")

    # ---- Tier 1 ----------------------------------------------------------
    ax = axes[0, 1]
    ep = tier1["episode"]; tr = tier1["train_loc"]
    # Use a rolling mean to make the noisy per-episode curve readable
    window = 50
    if len(tr) > window:
        kernel = np.ones(window) / window
        tr_smooth = np.convolve(tr, kernel, mode="valid")
        ep_smooth = ep[window - 1:]
    else:
        tr_smooth = tr; ep_smooth = ep
    ax.plot(ep, tr, color="tab:blue", alpha=0.25, lw=0.7, label="per-episode")
    ax.plot(ep_smooth, tr_smooth, color="tab:blue", lw=1.5,
            label=f"rolling mean (w={window})")
    ax.set_xlabel("episode")
    ax.set_ylabel("train loc err (z-score RMSE)")
    ax.set_title("Tier 1: FOMAML outer loop\n"
                 "(per-episode meta-loss on multi-scene FOMAML)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")

    # ---- Tier 2 ----------------------------------------------------------
    # Two-line plot on a single axis with a twin y-axis: task loss
    # on the left, adversary CE and env-probe on the right. The
    # disentanglement story (probe→chance, adv CE→chance) lives
    # on the right axis.
    ax = axes[1, 0]
    ax2 = ax.twinx()

    ep = tier2["epoch"]
    train_task = tier2["train_task_mse"]; val_task = tier2["val_task_mse"]
    adv_ce = tier2["train_adv_ce"]
    probe = tier2["probe_env_acc"]
    n_envs = int(tier2["probe_n_envs"][0])
    chance = 1.0 / n_envs

    ln1 = ax.plot(ep, train_task, color="tab:blue", lw=1.5,
                   label="train task MSE")
    ax.plot(ep, val_task, color="tab:blue", lw=1.5, linestyle="--",
            label="val task MSE")
    ln2 = ax2.plot(ep, adv_ce, color="tab:red", lw=1.5,
                    label="train adv CE")
    ax2.plot(ep, probe, color="tab:green", lw=1.5, linestyle=":",
              label=f"env-probe acc (chance={chance:.2f})")
    ax2.axhline(chance, color="tab:green", alpha=0.3, linestyle=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("task MSE", color="tab:blue")
    ax2.set_ylabel("adv CE / probe acc", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    # Mark the λ ramp region. Default in this script is 0 for 5 epochs
    # then linear ramp to 1.0 over 10 epochs.
    ramp_start = 5; ramp_end = 15
    if ep[-1] >= ramp_end:
        ax.axvspan(ramp_start, ramp_end, alpha=0.12, color="tab:purple")
        # Add the legend entry by hand since axvspan doesn't create one
        from matplotlib.patches import Patch
        ramp_patch = Patch(facecolor="tab:purple", alpha=0.12,
                            label="λ ramp (0→1)")
        ax.add_patch(ramp_patch) if False else None
        lns = ln1 + ln2 + [ramp_patch]
    else:
        lns = ln1 + ln2
    ax.set_title("Tier 2 v2: Sionna disentanglement\n"
                 "(blue = task loss, red = adv CE, green = env-probe)")
    ax.legend(lns, [l.get_label() for l in lns], loc="upper right",
               fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=":")

    # ---- Joint Tier 1+2 trainer -----------------------------------------
    ax = axes[1, 1]
    ep = joint["episode"]
    meta = joint["meta_loss"]
    if "probe_env_acc" in joint:
        probe = joint["probe_env_acc"]
        ax2 = ax.twinx()
        ax.plot(ep, meta, color="tab:blue", lw=1.5, label="meta_loss")
        ax2.plot(ep, probe, color="tab:green", lw=1.5, linestyle=":",
                  label="env-probe acc")
        ax.set_ylabel("meta loss", color="tab:blue")
        ax2.set_ylabel("probe acc", color="tab:green")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        lns = [ax.lines[0], ax2.lines[0]]
        ax.legend(lns, [l.get_label() for l in lns], loc="upper right",
                   fontsize=8)
    else:
        ax.plot(ep, meta, color="tab:blue", lw=1.5, label="meta_loss")
        ax.set_ylabel("meta loss")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("episode")
    ax.set_title("Joint Tier 1+2 trainer (phase C)\n"
                 "(meta-loss + env-probe accuracy)")
    ax.grid(True, alpha=0.3, linestyle=":")

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code_dir", default="/Users/justin/ultraclear-paper/code")
    p.add_argument("--out_dir", default="figures")
    args = p.parse_args()

    code = Path(args.code_dir)

    # Tier 0
    tier0 = load_tier0(code / "checkpoints/tier0_multiscene_f2_3to1_100ep")
    # Tier 1
    tier1 = load_tier1(code / "figures/tier1_maml_history.csv")
    # Tier 2 v2 (sionna) — the headline Tier 2 model
    tier2 = load_tier2(code / "checkpoints/tier2_sionna_v2_500" / "tier2_sionna_history.csv")
    # Joint tier 1+2 trainer (phase C)
    joint = load_joint(code / "checkpoints/joint_per_timestep_bounded_adv" / "phase_C_history.csv")

    out_path = code / args.out_dir / "training_curves_all_tiers.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure(out_path, tier0, tier1, tier2, joint)


if __name__ == "__main__":
    main()
