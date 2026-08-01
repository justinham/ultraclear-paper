"""Plot the per-cond val loc err as a function of Phase C episode,
showing the trajectory across snapshots. This is the early-stopping
diagnostic figure: the meta loss / val loc err is non-monotonic in
episode, and the best snapshot is often at ep 30-50, not the final
ep 100.

Reads per-cond values from the trainer's log lines and produces a
2-row figure: top row = cond 0/1/2 trajectories, bottom row = meta
loss + task_disent over the same episodes.
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path("/Users/justin/ultraclear-paper/code")


def parse_log(log_path: Path):
    """Parse the trainer log to get per-snapshot per-cond numbers and
    per-episode meta/task_disent losses."""
    snap_re = re.compile(
        r"\[phaseC\]\s+snapshot at ep (\d+): per_cond="
        r"\{0: ([\d.]+), 1: ([\d.]+), 2: ([\d.]+)\}"
    )
    ep_re = re.compile(
        r"\[phaseC\]\s+ep\s+(\d+)/\d+\s+lam=[\d.]+\s+meta=([\d.]+)\s+"
        r"task_d=([\d.]+)\s+adv=[\d.]+\s+cond=[\d.]+\s+probe=[\d.]+"
    )
    snapshots = []
    episodes = []
    with open(log_path) as f:
        for line in f:
            m = snap_re.search(line)
            if m:
                ep = int(m.group(1))
                snapshots.append({
                    "ep": ep,
                    "cond0": float(m.group(2)),
                    "cond1": float(m.group(3)),
                    "cond2": float(m.group(4)),
                })
                continue
            m = ep_re.search(line)
            if m:
                episodes.append({
                    "ep": int(m.group(1)),
                    "meta": float(m.group(2)),
                    "task_d": float(m.group(3)),
                })
    return snapshots, episodes


def main():
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO.parent.parent / "tmp/ultraclear_earlystop.log"
    )
    if not log.exists():
        sys.exit(f"log not found: {log}")
    snapshots, episodes = parse_log(log)
    if not snapshots:
        sys.exit(f"no snapshots found in {log}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top: per-cond val loc err vs episode
    ax = axes[0]
    eps = [s["ep"] for s in snapshots]
    means = [(s["cond0"] + s["cond1"] + s["cond2"]) / 3 for s in snapshots]
    ax.plot(eps, [s["cond0"] for s in snapshots], "o-", color="C0",
            label="cond 0 (low NLOS)", lw=2, markersize=6)
    ax.plot(eps, [s["cond1"] for s in snapshots], "s-", color="C1",
            label="cond 1 (mid NLOS)", lw=2, markersize=6)
    ax.plot(eps, [s["cond2"] for s in snapshots], "^-", color="C2",
            label="cond 2 (high NLOS)", lw=2, markersize=6)
    ax.plot(eps, means, "D--", color="k", label="mean (across conds)",
            lw=1.5, markersize=5, alpha=0.7)
    # Mark the best (lowest mean) snapshot
    best_idx = int(np.argmin(means))
    best_ep = eps[best_idx]
    ax.axvline(best_ep, color="grey", ls=":", alpha=0.6)
    ax.annotate(f"best: ep {best_ep}\nmean = {means[best_idx]:.2f} m",
                xy=(best_ep, means[best_idx]),
                xytext=(best_ep + 5, means[best_idx] + 1.0),
                fontsize=10, ha="left",
                arrowprops=dict(arrowstyle="->", color="grey"))
    ax.set_ylabel("val loc err (m, lower is better)")
    ax.set_title("Phase C trajectory — per-cond val loc err at each snapshot",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom: meta loss + task_disent over Phase C episodes
    ax = axes[1]
    if episodes:
        eps_e = [e["ep"] for e in episodes]
        ax.plot(eps_e, [e["meta"] for e in episodes], "-", color="C0",
                label="meta loss (FOMAML query)", lw=1.5, alpha=0.8)
        ax.plot(eps_e, [e["task_d"] for e in episodes], "-", color="C3",
                label="task_disent (encoder fwd on query)", lw=1.5, alpha=0.8)
        # Mark each snapshot with a vertical dotted line
        for ep in eps:
            ax.axvline(ep, color="grey", ls=":", alpha=0.3)
    ax.set_xlabel("Phase C episode")
    ax.set_ylabel("loss (MSE)")
    ax.set_title("Phase C training losses", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    out_dir = REPO.parent / "figures/joint_per_timestep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "phase_c_trajectory.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
