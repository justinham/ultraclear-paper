"""Per-window NLOS classification accuracy box plot for the v2 unfrozen
Tier 2 ckpt.

For each of the 21 envs, computes per-window NLOS classification
accuracy (% of T*M cells correct per window) over the 1,200 items in
that env. Then plots:

  Panel 1: per-env box plot, x = env_id (color-coded by TRAIN/VAL/EXCL),
           y = per-window accuracy. Overlay a strip of mean markers
           and a calibration scatter (p_actual_nlos vs mean accuracy).
  Panel 2: TRAIN vs VAL summary box (2 boxes).
  Panel 3: per-class breakdown — one box per env for acc_LOS cells and
           one for acc_NLOS cells (TWO boxes per env, side by side).

Outputs:
  - figures/tier2_v2_nlos_boxplot.png
  - checkpoints/.../nlos_per_window.json (per-window accuracies, in
    case you want to make a different plot later)

Usage:
    python3.11 -m scripts.viz_tier2_v2_nlos_boxplot \
      --ckpt checkpoints/tier2_sionna_v2_500/tier2_sionna_best.pt
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ultraclear.config import Config, make_default
from ultraclear.data import TensorUWB
from scripts.multi_scene_dataset import build_default_multi_scene
from scripts.train_tier2_synthetic import DisentangledTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_fig", default=None)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = ck["cfg"]
    cfg = make_default()
    arch_keys = {"n_channels", "d_model", "seq_len", "n_anchors",
                 "n_synthetic_envs", "n_micro_conditions",
                 "samples_per_env", "ssl_hidden", "adv_hidden",
                 "micro_dim", "k_shot"}
    for k, v in cfg_dict.items():
        if k in arch_keys and hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    cfg.device = args.device
    n_envs = ck["n_envs"]
    train_scenes = set(ck["train_scenes"])
    val_scenes = set(ck["val_scenes"])
    print(f"Loaded ckpt: {args.ckpt}")
    print(f"  train_scenes: {sorted(train_scenes)}")
    print(f"  val_scenes: {sorted(val_scenes)}")

    print(f"\nloading Sionna multi-scene data...")
    ds, scene_info = build_default_multi_scene(
        n_channels=cfg.n_channels, seq_len=cfg.seq_len, stride=32,
        all_scenes=True,
    )
    ds.compute_norm_stats(train_scenes)

    model = DisentangledTrainer(cfg, n_envs=n_envs, n_conds=2).to(args.device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    # Index items by env_id
    print(f"\nindexing items by env_id...")
    items_by_env: dict[int, list] = {e: [] for e in range(21)}
    for i in range(len(ds)):
        item = ds[i]
        eid = int(item[2])
        items_by_env[eid].append(item)

    # Per-window accuracy per env
    print(f"\ncomputing per-window accuracy...")
    per_env_window_acc: dict[int, list[float]] = {e: [] for e in range(21)}
    per_env_los_acc: dict[int, list[float]] = {e: [] for e in range(21)}
    per_env_nlos_acc: dict[int, list[float]] = {e: [] for e in range(21)}
    per_env_actual_rate: dict[int, float] = {}
    per_env_pred_rate: dict[int, float] = {}
    for eid in range(21):
        items = items_by_env[eid]
        if not items:
            continue
        loader = DataLoader(TensorUWB(items), batch_size=64)
        n_total = 0
        n_correct = 0
        n_nlos_cells = 0
        sum_pred_prob = 0.0
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 6:
                    X, y, env, cond, nlos, _ = batch
                else:
                    X, y, env, cond, nlos = batch
                X = X.to(args.device)
                nlos = nlos.to(args.device)
                _h, _c, _adv, _loc, nlos_logits, _cond = model(X)
                pred = (nlos_logits > 0).long()
                gt = nlos.long()
                # per-window accuracy
                B = X.size(0)
                for b in range(B):
                    per_env_window_acc[eid].append(
                        float((pred[b] == gt[b]).float().mean().item())
                    )
                # per-window class-conditional accuracy
                for b in range(B):
                    los_mask_b = (gt[b] == 0)
                    nlos_mask_b = (gt[b] == 1)
                    if los_mask_b.any():
                        per_env_los_acc[eid].append(
                            float((pred[b][los_mask_b] == 0).float().mean().item())
                        )
                    if nlos_mask_b.any():
                        per_env_nlos_acc[eid].append(
                            float((pred[b][nlos_mask_b] == 1).float().mean().item())
                        )
                n_total += pred.numel()
                n_correct += (pred == gt).sum().item()
                n_nlos_cells += (gt == 1).sum().item()
                sum_pred_prob += torch.sigmoid(nlos_logits).sum().item()
        per_env_actual_rate[eid] = n_nlos_cells / n_total
        per_env_pred_rate[eid] = sum_pred_prob / n_total
        print(f"  env {eid:2d}: n_items={len(items):4d}, "
              f"acc={n_correct/n_total:.3f}, "
              f"p_actual_nlos={per_env_actual_rate[eid]:.3f}, "
              f"p_pred_nlos={per_env_pred_rate[eid]:.3f}")

    # Save per-window accuracies as JSON
    out_json = args.out_json or str(Path(args.ckpt).parent / "nlos_per_window.json")
    with open(out_json, "w") as f:
        json.dump({
            "ckpt": str(args.ckpt),
            "per_env_window_acc": {str(e): v for e, v in per_env_window_acc.items()},
            "per_env_los_acc": {str(e): v for e, v in per_env_los_acc.items()},
            "per_env_nlos_acc": {str(e): v for e, v in per_env_nlos_acc.items()},
            "per_env_actual_rate": {str(k): v for k, v in per_env_actual_rate.items()},
            "per_env_pred_rate": {str(k): v for k, v in per_env_pred_rate.items()},
            "train_scenes": sorted(train_scenes),
            "val_scenes": sorted(val_scenes),
        }, f)
    print(f"\nPer-window data saved to {out_json}")

    # ---- Plot ----
    # The data is heavily saturated at 1.0 (most windows have 100%
    # accuracy), so a regular boxplot collapses. Use a violin plot
    # that explicitly shows the kernel density, plus overlay the
    # per-env mean (red diamond) and the bottom-tail outliers.
    # Plus a 2nd panel for the per-class (LOS vs NLOS cells) bar
    # comparison, and a 3rd panel for the envs with most spread
    # (the envs with NLOS > 5%).

    env_ids = list(range(21))
    plot_envs = [e for e in env_ids if e != 14]
    role_color = {
        "TRAIN": "tab:blue",
        "VAL":   "tab:orange",
        "EXCL":  "lightgray",
    }
    env_role = []
    for e in plot_envs:
        if e in train_scenes:
            env_role.append("TRAIN")
        elif e in val_scenes:
            env_role.append("VAL")
        else:
            env_role.append("EXCL")
    bar_colors = [role_color[r] for r in env_role]

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], hspace=0.30, wspace=0.20)

    # --- Panel 1 (top, spans both columns): violin plot ---
    ax1 = fig.add_subplot(gs[0, :])
    data_main = [np.array(per_env_window_acc[e]) for e in plot_envs]
    parts = ax1.violinplot(
        data_main,
        positions=plot_envs,
        widths=0.8,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], bar_colors):
        body.set_facecolor(color)
        body.set_alpha(0.5)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
    # Median lines: matplotlib draws them as horizontal segments at
    # the median y across the body. Make them visible.
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)
    # Per-env mean (red diamond)
    means = [np.mean(per_env_window_acc[e]) if per_env_window_acc[e] else np.nan
             for e in plot_envs]
    ax1.scatter(plot_envs, means, color="red", s=40, zorder=5, marker="D",
                edgecolor="black", linewidth=0.5, label="per-env mean")
    # Vertical boundary between TRAIN and VAL
    if train_scenes and val_scenes:
        boundary = (max([e for e in train_scenes if e != 14])
                    + min([e for e in val_scenes if e != 14])) / 2
        ax1.axvline(boundary, color="black", linestyle="--",
                    alpha=0.5, linewidth=1.0)
    # Reference line for 0.75 (visible lower-tail band in the data)
    ax1.axhline(0.75, color="purple", linestyle=":", alpha=0.5, linewidth=1.0)
    ax1.text(plot_envs[0] - 0.5, 0.752, "0.75 ref", fontsize=8,
             color="purple", va="bottom", ha="left")
    ax1.set_xticks(plot_envs)
    ax1.set_xticklabels(plot_envs, fontsize=9)
    ax1.set_xlabel("env_id (0..20, excl. env 14)")
    ax1.set_ylabel("per-window NLOS classification accuracy")
    # Floor at 0.5 so all lower tails are visible (env 4 and 12 reach
    # down to ~0.65 in this run)
    ax1.set_ylim(0.0, 1.005)
    ax1.set_title(
        f"Per-window NLOS classification accuracy — Tier 2 v2 (unfrozen encoder)\n"
        f"violin shows kernel density; black bar = median; red ◇ = per-env mean; "
        f"n=1200 windows per env",
        fontsize=11
    )
    legend_elements = [
        Patch(facecolor="tab:blue", alpha=0.5, label="TRAIN env (n=10)"),
        Patch(facecolor="tab:orange", alpha=0.5, label="VAL env (n=10)"),
        plt.Line2D([0], [0], marker="D", color="red", linestyle="",
                   markeredgecolor="black", label="per-env mean"),
    ]
    ax1.legend(handles=legend_elements, loc="lower left",
               bbox_to_anchor=(0.0, 0.0), fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    # --- Panel 2 (bottom-left): TRAIN vs VAL summary violin ---
    ax2 = fig.add_subplot(gs[1, 0])
    train_acc = []
    val_acc = []
    for e in plot_envs:
        if e in train_scenes:
            train_acc.extend(per_env_window_acc[e])
        elif e in val_scenes:
            val_acc.extend(per_env_window_acc[e])
    parts2 = ax2.violinplot(
        [train_acc, val_acc],
        positions=[1, 2],
        widths=0.7,
        showmeans=True,
        showmedians=True,
        showextrema=False,
    )
    parts2["bodies"][0].set_facecolor("tab:blue")
    parts2["bodies"][1].set_facecolor("tab:orange")
    for body in parts2["bodies"]:
        body.set_alpha(0.5)
        body.set_edgecolor("black")
    if "cmedians" in parts2:
        parts2["cmedians"].set_color("black")
        parts2["cmedians"].set_linewidth(1.5)
    if "cmeans" in parts2:
        parts2["cmeans"].set_color("red")
        parts2["cmeans"].set_linewidth(1.5)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([f"TRAIN\n(10 envs, 12,000 windows)",
                          f"VAL\n(10 envs, 12,000 windows)"], fontsize=9)
    ax2.set_ylabel("per-window accuracy")
    ax2.set_xlim(0.5, 2.5)
    ax2.set_ylim(0.0, 1.005)
    ax2.set_title(
        f"Summary: TRAIN vs VAL\n"
        f"mean TRAIN = {np.mean(train_acc):.3f}, mean VAL = {np.mean(val_acc):.3f}",
        fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)

    # --- Panel 3 (bottom-right): per-env per-class bar chart ---
    ax3 = fig.add_subplot(gs[1, 1])
    los_means = []
    nlos_means = []
    n_nlos_cells_per_env = []
    for e in plot_envs:
        los_means.append(np.mean(per_env_los_acc[e])
                          if per_env_los_acc[e] else np.nan)
        nlos_means.append(np.mean(per_env_nlos_acc[e])
                           if per_env_nlos_acc[e] else np.nan)
        n_nlos_cells_per_env.append(per_env_actual_rate.get(e, 0) * 128 * 1200)
    x = np.array(plot_envs)
    width = 0.4
    ax3.bar(x - width/2, los_means, width=width, color="tab:cyan",
            alpha=0.75, label="acc on LOS cells")
    ax3.bar(x + width/2, nlos_means, width=width, color="tab:red",
            alpha=0.75, label="acc on NLOS cells")
    # Mark NaN NLOS bars (envs with 0% NLOS)
    for i, nlos_v in enumerate(nlos_means):
        if np.isnan(nlos_v):
            ax3.text(x[i] + width/2, 0.68, "no NLOS", ha="center", va="bottom",
                     fontsize=6, color="gray", rotation=90)
    if train_scenes and val_scenes:
        boundary = (max([e for e in train_scenes if e != 14])
                    + min([e for e in val_scenes if e != 14])) / 2
        ax3.axvline(boundary, color="black", linestyle="--",
                    alpha=0.5, linewidth=1.0)
    ax3.set_xticks(plot_envs)
    ax3.set_xticklabels(plot_envs, fontsize=8)
    ax3.set_xlabel("env_id (0..20, excl. env 14)")
    ax3.set_ylabel("per-class accuracy")
    ax3.set_ylim(0.0, 1.005)
    ax3.set_title("Per-class: LOS cells (cyan) vs NLOS cells (red)",
                  fontsize=10)
    ax3.legend(loc="lower left", fontsize=8)
    ax3.grid(True, axis="y", alpha=0.3)

    out_fig = args.out_fig or str(Path(args.ckpt).parent / "tier2_v2_nlos_boxplot.png")
    if not out_fig.startswith("/"):
        out_fig = str(Path("/Users/justin/ultraclear-paper") / out_fig)
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_fig}")


if __name__ == "__main__":
    main()
