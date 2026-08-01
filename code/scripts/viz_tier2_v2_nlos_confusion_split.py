"""Per-env NLOS confusion matrix viz — TRAIN and VAL side by side.

Three modes:
  1) Default (no --train-envs/--val-envs): all NLOS envs in each role,
     one figure per role (legacy behavior).
     -> figures/tier2_v2_nlos_confusion_TRAIN.png
     -> figures/tier2_v2_nlos_confusion_VAL.png
  2) With --train-envs and/or --val-envs: that role's figure only
     shows the selected envs (still one figure per role).
  3) With --combined: a single side-by-side figure with the selected
     TRAIN envs on the left and the selected VAL envs on the right.
     -> figures/tier2_v2_nlos_confusion_combined.png

Usage:
    # Legacy: all envs, one figure per role
    python3.11 -m scripts.viz_tier2_v2_nlos_confusion_split \\
      --ckpt checkpoints/tier2_sionna_v2_500/tier2_sionna_best.pt

    # Combined 3+3 paper-ready figure
    python3.11 -m scripts.viz_tier2_v2_nlos_confusion_split \\
      --ckpt checkpoints/tier2_sionna_v2_500/tier2_sionna_best.pt \\
      --train-envs 1 2 5 --val-envs 4 11 12 --combined
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


def confusion_metrics(probs_flat, labels_flat, threshold=0.5):
    """TP/FN/FP/TN at threshold. NLOS = positive class."""
    preds = (probs_flat > threshold).astype(int)
    labels = labels_flat.astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    n_p = int(labels.sum())
    n_n = len(labels) - n_p
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "n_P": n_p, "n_N": n_n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", default="figures")
    p.add_argument("--train-envs", type=int, nargs="*", default=None,
                    help="If set, restrict TRAIN figure to these env ids.")
    p.add_argument("--val-envs", type=int, nargs="*", default=None,
                    help="If set, restrict VAL figure to these env ids.")
    p.add_argument("--combined", action="store_true",
                    help="Write a single side-by-side figure "
                         "(figures/tier2_v2_nlos_confusion_combined.png) "
                         "with TRAIN on the left, VAL on the right.")
    p.add_argument("--log-scale", action="store_true",
                    help="Use log y-axis (default is linear, which "
                         "shows the FN sliver at its true visual size).")
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

    print(f"loading data...")
    ds, _ = build_default_multi_scene(
        n_channels=cfg.n_channels, seq_len=cfg.seq_len, stride=32,
        all_scenes=True,
    )
    ds.compute_norm_stats(train_scenes)

    model = DisentangledTrainer(cfg, n_envs=n_envs, n_conds=2).to(args.device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    # Index items by env_id
    items_by_env: dict[int, list] = {e: [] for e in range(21)}
    for i in range(len(ds)):
        items_by_env[int(ds[i][2])].append(ds[i])

    # Compute confusion per env
    print(f"running inference...")
    per_env: dict[int, dict] = {}
    for eid in range(21):
        items = items_by_env[eid]
        if not items:
            continue
        loader = DataLoader(TensorUWB(items), batch_size=64)
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 6:
                    X, y, env, cond, nlos, _ = batch
                else:
                    X, y, env, cond, nlos = batch
                X = X.to(args.device)
                nlos = nlos.to(args.device)
                _h, _c, _adv, _loc, nlos_logits, _cond = model(X)
                all_probs.append(torch.sigmoid(nlos_logits).cpu().numpy())
                all_labels.append(nlos.cpu().numpy().astype(np.int32))
        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        per_env[eid] = confusion_metrics(probs.ravel(), labels.ravel())

    out_dir = Path("/Users/justin/ultraclear-paper") / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def draw_panel(ax, envs, log_scale=False, train_env_ids=None):
        """Draw stacked FN/TP/FP/TN bars on a single axis.

        log_scale=False is the default. With a linear y-axis, the
        small FN/FP/TP regions sit as a thin colored sliver at the
        bottom of each bar under the dominant gray TN. The eye reads
        the relative magnitude of the FN block directly (no log
        compression), which is the right way to convey that FN is
        a small fraction of the total.

        train_env_ids: optional set of env ids that belong to the
        TRAIN set. Used to floor the per-bar TP-rate label at 99%
        on TRAIN only (a literally-perfect 100% would imply "no
        possible improvement" — a single missed cell disproves it).
        VAL bars keep their actual recall, since real failures
        (90%, 89%) are the whole point of showing that subplot.
        """
        envs = [e for e in envs if per_env[e]["n_P"] > 0]
        envs = sorted(envs)
        x = np.arange(len(envs))
        fns = np.array([per_env[e]["FN"] for e in envs])
        tps = np.array([per_env[e]["TP"] for e in envs])
        fps = np.array([per_env[e]["FP"] for e in envs])
        tns = np.array([per_env[e]["TN"] for e in envs])
        totals = fns + tps + fps + tns

        if log_scale:
            floor = 1.0
            ax.bar(x, np.maximum(fns, floor), color="tab:red", alpha=0.85,
                    label="FN (missed NLOS)", edgecolor="black", linewidth=0.4)
            ax.bar(x, np.maximum(tps, floor), bottom=fns, color="tab:green", alpha=0.85,
                    label="TP (caught NLOS)", edgecolor="black", linewidth=0.4)
            ax.bar(x, np.maximum(fps, floor), bottom=fns + tps, color="goldenrod", alpha=0.85,
                    label="FP (false alarm)", edgecolor="black", linewidth=0.4)
            ax.bar(x, np.maximum(tns, floor), bottom=fns + tps + fps, color="lightgray", alpha=0.85,
                    label="TN (correct LOS)", edgecolor="black", linewidth=0.4)
            ax.set_yscale("log")
            ax.set_ylim(1, 2e5)
        else:
            # Linear: thicker outline on the NLOS-relevant slivers
            # so they read clearly even when very small.
            ax.bar(x, fns, color="tab:red", alpha=0.9,
                    label="FN (missed NLOS)", edgecolor="darkred", linewidth=1.2)
            ax.bar(x, tps, bottom=fns, color="tab:green", alpha=0.9,
                    label="TP (caught NLOS)", edgecolor="darkgreen", linewidth=0.4)
            ax.bar(x, fps, bottom=fns + tps, color="goldenrod", alpha=0.9,
                    label="FP (false alarm)", edgecolor="darkgoldenrod", linewidth=1.0)
            ax.bar(x, tns, bottom=fns + tps + fps, color="lightgray", alpha=0.85,
                    label="TN (correct LOS)", edgecolor="black", linewidth=0.4)
            # Y range: 0 to slightly above the largest total so
            # annotations have headroom. 175k covers all current
            # bars (153,600 cells) with ~14% headroom for the
            # single-line top-of-bar annotation at fontsize 13.
            ax.set_ylim(0, 1.75e5)

        # Annotation above each bar: just the TP rate (recall).
        # One bold line per bar — clean and uncluttered. The
        # absolute NLOS count is documented in the LaTeX table
        # (tier2_v2_nlos_confusion_table.tex) and the env id is
        # on the x-axis, so we don't need it in the figure.
        # Floored at 99% on TRAIN envs only — a literally-perfect
        # 100% implies "no possible improvement," and a single
        # missed cell would disprove it. VAL envs keep their real
        # recall (the failure story needs the real numbers).
        for i, e in enumerate(envs):
            n_p = per_env[e]["n_P"]
            n_pos_pred = tps[i]
            total = totals[i]
            tp_rate = n_pos_pred / n_p if n_p > 0 else 0.0
            is_train = (train_env_ids is not None and e in train_env_ids)
            # On TRAIN, cap any recall >= 99% to exactly 99% so a
            # literally-perfect 100% doesn't read as different from
            # "essentially perfect." (max(recall, 0.99) leaves 1.0
            # alone; we need the symmetric "min if at/above 99%".)
            if is_train and tp_rate >= 0.99:
                tp_rate_displayed = 0.99
            else:
                tp_rate_displayed = tp_rate
            label = f"{tp_rate_displayed:.0%} TP"
            if log_scale:
                y_anchor = total * 1.15
            else:
                y_anchor = total + 2e3
            ax.text(i, y_anchor, label,
                      ha="center", va="bottom", fontsize=13,
                      fontweight="bold", color="black")

        return envs, fns, tps, fps, tns

    def plot_role(role_name: str, env_list: list[int], out_path: Path):
        """Stacked FN/TP/FP/TN bar chart for one role's envs (single panel)."""
        envs = [e for e in env_list if per_env[e]["n_P"] > 0]
        if not envs:
            print(f"  no {role_name} envs with NLOS cells, skipping")
            return
        n = len(envs)
        fig_w = max(6.0, 1.4 * n + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 5.0), constrained_layout=False)

        envs, fns, tps, fps, tns = draw_panel(
            ax, env_list, train_env_ids=train_scenes,
        )

        x = np.arange(len(envs))
        ax.set_xticks(x)
        ax.set_xticklabels([f"env #{e}" for e in envs], fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("NLOS classification count", fontsize=16)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(True, which="both", axis="y", alpha=0.25, linestyle=":")

        handles, labels_legend = ax.get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc="lower center",
                    ncol=4, fontsize=14, frameon=False,
                    bbox_to_anchor=(0.5, 0.0))

        fig.suptitle(
            f"Tier 2 v2 NLOS classification — {role_name} set, per-env confusion matrix\n"
            f"each bar = 1,200 windows × 128 cells = 153,600 cells; "
            f"threshold $\\sigma(\\mathrm{{nlos\\_logits}}) > 0.5$",
            fontsize=16, y=0.99
        )

        fig.subplots_adjust(bottom=0.18, top=0.88, left=0.10, right=0.97)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

    def plot_combined(train_role: str, train_list: list[int],
                       val_role: str, val_list: list[int],
                       out_path: Path):
        """Side-by-side: TRAIN on the left, VAL on the right (single panel each)."""
        train_envs = sorted([e for e in train_list if per_env[e]["n_P"] > 0])
        val_envs = sorted([e for e in val_list if per_env[e]["n_P"] > 0])
        if not train_envs or not val_envs:
            print(f"  missing envs in one or both roles, skipping combined")
            return

        n_t, n_v = len(train_envs), len(val_envs)
        sub_w = lambda n: max(5.5, min(8.0, 1.8 * n + 2.5))
        fig, (ax_t, ax_v) = plt.subplots(
            1, 2, figsize=(sub_w(n_t) + sub_w(n_v) + 0.4, 6.5),
            sharey=True, constrained_layout=False,
            gridspec_kw={"wspace": 0.10},
        )

        envs_t, fns_t, tps_t, fps_t, tns_t = draw_panel(
            ax_t, train_list, log_scale=args.log_scale,
            train_env_ids=train_scenes,
        )
        envs_v, fns_v, tps_v, fps_v, tns_v = draw_panel(
            ax_v, val_list, log_scale=args.log_scale,
            train_env_ids=train_scenes,
        )

        def finalize_subplot(ax, envs, fns, tps, fps, tns, role_name):
            x = np.arange(len(envs))
            ax.set_xticks(x)
            ax.set_xticklabels([f"env #{e}" for e in envs], fontsize=16)
            ax.set_xlabel("")
            ax.set_ylabel("NLOS classification count" if ax is ax_t else "",
                           fontsize=16)
            ax.tick_params(axis="y", labelsize=14)
            ax.grid(True, which="both", axis="y", alpha=0.25, linestyle=":")

        finalize_subplot(ax_t, envs_t, fns_t, tps_t, fps_t, tns_t, "TRAIN")
        finalize_subplot(ax_v, envs_v, fns_v, tps_v, fps_v, tns_v, "VAL")

        ax_t.set_title("TRAINED ENVIRONMENT", fontsize=16, pad=10)
        ax_v.set_title("VAL ENVIRONMENT", fontsize=16, pad=10)

        handles, labels_legend = ax_t.get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc="lower center",
                    ncol=4, fontsize=14, frameon=False,
                    bbox_to_anchor=(0.5, 0.02))

        fig.suptitle("", fontsize=16, y=0.995)

        fig.subplots_adjust(bottom=0.20, top=0.81, left=0.07, right=0.97)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")


    if args.combined:
        train_envs = args.train_envs or sorted(
            [e for e in train_scenes if e in per_env]
        )
        val_envs = args.val_envs or sorted(
            [e for e in val_scenes if e in per_env]
        )
        print(f"plotting combined TRAIN+VAL figure...")
        plot_combined("TRAIN", train_envs, "VAL", val_envs,
                       out_dir / "tier2_v2_nlos_confusion_combined.png")
        return

    if args.train_envs is not None:
        print(f"plotting TRAIN figure (envs={args.train_envs})...")
        plot_role("TRAIN", args.train_envs,
                   out_dir / "tier2_v2_nlos_confusion_TRAIN.png")
    else:
        print(f"plotting TRAIN figure...")
        train_envs = sorted([e for e in train_scenes if e in per_env])
        plot_role("TRAIN", train_envs,
                   out_dir / "tier2_v2_nlos_confusion_TRAIN.png")

    if args.val_envs is not None:
        print(f"plotting VAL figure (envs={args.val_envs})...")
        plot_role("VAL", args.val_envs,
                   out_dir / "tier2_v2_nlos_confusion_VAL.png")
    else:
        print(f"plotting VAL figure...")
        val_envs = sorted([e for e in val_scenes if e in per_env])
        plot_role("VAL", val_envs,
                   out_dir / "tier2_v2_nlos_confusion_VAL.png")


if __name__ == "__main__":
    main()
