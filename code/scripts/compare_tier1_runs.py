"""Compare Tier 1 FOMAML runs: random init vs Tier 0 pretrained.

Loads both checkpoints (same FOMAML outer-loop, different encoder
init) and produces a few-shot curve plot.

Usage:
    python3.11 -m scripts.compare_tier1_runs \\
        --ckpt_random /tmp/tier1_random/tier1_sionna.pt \\
        --ckpt_tier0  /tmp/tier1_t0init/tier1_sionna.pt \\
        --out_dir /path/to/save/figures
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_random", required=True)
    p.add_argument("--ckpt_tier0", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--val_scene_names", nargs="+",
                   default=["outdoor_parking_01", "indoor_garage_parking_01"])
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt_random).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_r = torch.load(args.ckpt_random, weights_only=False)
    ckpt_t = torch.load(args.ckpt_tier0, weights_only=False)

    # Reconstruct the k_shot axis from the keys
    def get_ks_and_results(ckpt):
        fs = ckpt["fs_results"]
        # keys are "k=N_env=M" strings
        ks = sorted({int(k.split("_")[0].split("=")[1]) for k in fs.keys()})
        envs = sorted({int(k.split("_")[1].split("=")[1]) for k in fs.keys()})
        # build a matrix (n_ks, n_envs)
        mat = np.zeros((len(ks), len(envs)))
        for i, k in enumerate(ks):
            for j, e in enumerate(envs):
                mat[i, j] = fs[f"k={k}_env={e}"]
        return ks, envs, mat

    ks_r, envs_r, mat_r = get_ks_and_results(ckpt_r)
    ks_t, envs_t, mat_t = get_ks_and_results(ckpt_t)
    assert ks_r == ks_t, f"k_shots differ: {ks_r} vs {ks_t}"
    assert envs_r == envs_t, f"val envs differ: {envs_r} vs {envs_t}"
    ks, envs, mat_r, mat_t = ks_r, envs_r, mat_r, mat_t
    n_envs = len(envs)

    # average across val scenes for the headline number
    mean_r = mat_r.mean(axis=1)
    mean_t = mat_t.mean(axis=1)
    se_r = mat_r.std(axis=1) / np.sqrt(n_envs)
    se_t = mat_t.std(axis=1) / np.sqrt(n_envs)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ks, mean_r, yerr=se_r, marker="o", capsize=3, linewidth=2,
                color="C0", label="random init")
    ax.errorbar(ks, mean_t, yerr=se_t, marker="s", capsize=3, linewidth=2,
                color="C1", label="Tier 0 pretrained")
    # also plot per-env points (smaller markers)
    for e_idx, eid in enumerate(envs):
        ax.plot(ks, mat_r[:, e_idx], "o", color="C0", alpha=0.3, markersize=4)
        ax.plot(ks, mat_t[:, e_idx], "s", color="C1", alpha=0.3, markersize=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K (number of support shots per env)")
    ax.set_ylabel("query MSE (averaged over val scenes)")
    ax.set_title("Tier 1 FOMAML — few-shot adaptation to held-out scenes\n"
                  "(F=2 ToA+AoA, 6 scenes, 100 FOMAML episodes)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    out1 = out_dir / "compare_tier1_few_shot.png"
    fig.savefig(out1, dpi=140)
    plt.close(fig)
    print(f"saved {out1}")

    # also print a clean table
    print()
    print("=" * 72)
    print("  Tier 1 FOMAML — few-shot query MSE on held-out scenes")
    print("=" * 72)
    print(f"  train scenes: {ckpt_r['train_scenes']}  "
          f"val scenes: {ckpt_r['val_scenes']}  "
          f"n_episodes: {ckpt_r['n_episodes']}")
    print()
    print(f"  {'K':>4}  {'random':>10}  {'tier0':>10}  "
          f"{'rel. reduction':>15}  {'per-val-scene (rnd)':>22}")
    print("  " + "-" * 72)
    for i, k in enumerate(ks):
        red = (mean_r[i] - mean_t[i]) / mean_r[i] * 100
        per_env = "  ".join(f"{mat_r[i, j]:7.2f}" for j in range(n_envs))
        print(f"  {k:>4}  {mean_r[i]:>10.3f}  {mean_t[i]:>10.3f}  "
              f"{red:>+14.1f}%  {per_env:>22}")
    print("  " + "-" * 72)

    # save CSV
    csv_out = out_dir / "compare_tier1_few_shot.csv"
    with open(csv_out, "w") as f:
        f.write("k_shot,random_mean,random_se,tier0_mean,tier0_se,reduction_pct\n")
        for i, k in enumerate(ks):
            red = (mean_r[i] - mean_t[i]) / mean_r[i] * 100
            f.write(f"{k},{mean_r[i]:.4f},{se_r[i]:.4f},"
                    f"{mean_t[i]:.4f},{se_t[i]:.4f},{red:.2f}\n")
    print(f"\nsaved {csv_out}")


if __name__ == "__main__":
    main()
