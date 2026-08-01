"""Tier 1 — GT vs predicted (x, y) overlay, per checkpoint, per scene.

For a chosen tier-1 checkpoint and a set of held-out val scenes,
adapt the model to each scene (FOMAML with K shots) and then
plot the GT tag positions and the model's predicted positions
on the same floor plan. Two colors:
  - GT positions in dark grey, small marker
  - Predicted positions in colored, larger marker

Per-position error is also written to a CSV. This is the
"where does the model think the tag is" diagnostic for the
paper: it makes the bias of the predictions directly visible
without needing a colormap.

Usage:
    python3.11 -m scripts.viz_tier1_gt_vs_pred \\
        --ckpt checkpoints/tier1_sionna_best.pt \\
        --scenes 4 5 11 16 \\
        --out_dir /path/to/save
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.config import Config
from ultraclear.tier1_maml import MAMLCore, adapt_and_evaluate
from scripts.train_tier1_sionna import sample_task_multi
from scripts.multi_scene_dataset import build_default_multi_scene


def adapt_and_predict(core_state, cfg, eid, ds, K=20,
                        denorm_xy=False):
    """Adapt the model to scene eid using K shots, then return
    (gt_xy, pred_xy) for every window in the scene.

    If denorm_xy=True, denormalize the (x, y) coordinates back to
    real meters using the per-scene z-score stats. The dataset
    z-scores tag_xy per scene (so the model sees unit-variance
    targets), but for the paper we want to show positions in the
    real-world coordinate frame.

    Returns two arrays of shape (N, 2).
    """
    fresh = MAMLCore(cfg)
    fresh.load_state_dict(core_state)
    fresh.train()
    sx, sy = sample_task_multi(ds, eid, K, cfg.seed + 9000)
    qx, qy = sample_task_multi(ds, eid, 20, cfg.seed + 9100)
    adapt_and_evaluate(fresh, sx, sy, qx, qy,
                        inner_lr=cfg.maml_inner_lr,
                        inner_steps=cfg.maml_inner_steps)
    fresh.eval()
    indices = [i for i, (s_idx, _, _) in enumerate(ds.windows)
                if ds.scene_info[s_idx].env_id == eid]
    gts, preds = [], []
    with torch.no_grad():
        for i in indices:
            x, y, _, _, _ = ds[i]
            xb = x.unsqueeze(0)
            pred = functional_call(fresh, dict(fresh.named_parameters()), (xb,))
            # y has shape (T, 2) — tag position at each timestep,
            # in z-scored units. Use the mean as the window's GT.
            # pred has shape (1, 2) — one (x, y) for the whole window.
            gt_xy = y.mean(dim=0)
            pred_xy = pred[0]
            gts.append((float(gt_xy[0]), float(gt_xy[1])))
            preds.append((float(pred_xy[0]), float(pred_xy[1])))
    gts = np.array(gts)
    preds = np.array(preds)
    if denorm_xy:
        info = ds.scene_info[eid]
        if info.tag_xy_mean is not None and info.tag_xy_std is not None:
            gts = gts * info.tag_xy_std + info.tag_xy_mean
            preds = preds * info.tag_xy_std + info.tag_xy_mean
    return gts, preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                    help="path to tier1 .pt; can also point at a "
                         "bundle directory like checkpoints/tier1_mixed.pt "
                         "if --ckpt_inner is given")
    p.add_argument("--ckpt_inner", default="tier1_sionna.pt",
                    help="filename inside a bundle directory")
    p.add_argument("--scenes", type=int, nargs="+", default=None,
                    help="env_ids to plot; if omitted, plots all "
                         "scenes in the dataset")
    p.add_argument("--all_scenes", action="store_true",
                    help="if set, load every *_A1A4swapped.csv from "
                         "the grid2 directory (~20 scenes) instead of "
                         "the default 6-scene mix")
    p.add_argument("--K", type=int, default=20,
                    help="number of support shots for FOMAML adaptation")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--denorm_xy", action="store_true",
                    help="if set, denormalize (x, y) positions from "
                         "z-scored back to real meters using the "
                         "per-scene tag_xy_mean / tag_xy_std. Without "
                         "this flag, the figure shows positions in "
                         "the model's normalized coordinate space "
                         "(unit-variance per scene).")
    p.add_argument("--ckpt_tag", default="",
                    help="if given, appended to output filenames so "
                         "two checkpoints can be plotted to the same "
                         "out_dir without overwriting each other "
                         "(e.g. 'sionna_best' -> gt_vs_pred_env0_<name>_sionna_best.png)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # the ckpt path may point at a bundle dir; the actual file is
    # <ckpt>/<ckpt_inner>
    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / args.ckpt_inner
    print(f"loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), weights_only=False)

    print("loading multi-scene dataset...")
    ds, scene_info = build_default_multi_scene(n_channels=2, seq_len=32,
                                                  stride=32,
                                                  all_scenes=args.all_scenes)
    # if no scenes specified, plot all
    if args.scenes is None:
        args.scenes = list(range(len(scene_info)))
    # train/val split: prefer what's stored in the checkpoint
    # (each tier1 run may use a different split), else default to
    # the 6-scene convention.
    train_scenes = set(ckpt.get("train_scenes", [0, 1, 2, 3]))
    print(f"train_scenes (from ckpt): {sorted(train_scenes)}")
    norm_mean, norm_std = ds.compute_norm_stats(train_scenes)
    ds.norm_mean = norm_mean
    ds.norm_std = norm_std

    cfg = Config(**{k: v for k, v in ckpt["cfg"].items()
                    if k in Config.__dataclass_fields__})
    print(f"ckpt inner_lr={cfg.maml_inner_lr} inner_steps={cfg.maml_inner_steps}")

    # one figure per scene
    for eid in args.scenes:
        info = scene_info[eid]
        print(f"\nscene {eid} ({info.name}): {info.n_grids} grids, "
              f"NLOS rate {info.nlos_rate*100:.1f}%")

        gts, preds = adapt_and_predict(ckpt["model_state"], cfg, eid, ds,
                                          K=args.K,
                                          denorm_xy=args.denorm_xy)
        # per-position Euclidean error (m)
        errs = np.linalg.norm(preds - gts, axis=1)
        median_err = float(np.median(errs))
        mean_err = float(np.mean(errs))
        p95 = float(np.percentile(errs, 95))

        fig, ax = plt.subplots(figsize=(7, 6))
        # GT: dark grey, small
        ax.scatter(gts[:, 0], gts[:, 1], s=8, c="black", alpha=0.6,
                    label="GT", edgecolors="none")
        # predicted: red, larger
        ax.scatter(preds[:, 0], preds[:, 1], s=14, c="C3", alpha=0.7,
                    label="predicted", edgecolors="none")
        ax.set_aspect("equal")
        ax.set_xlabel("tag x (m)", fontsize=12)
        ax.set_ylabel("tag y (m)", fontsize=12)
        units_note = " (m, real-world)" if args.denorm_xy else " (z-scored, per scene)"
        ax.set_title(f"{info.name}\n"
                      f"K={args.K}, median err = {median_err:.2f} m, "
                      f"mean = {mean_err:.2f} m, p95 = {p95:.2f} m{units_note}",
                      fontsize=12)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        fig.tight_layout()
        out = out_dir / f"gt_vs_pred_env{eid}_{info.name}{('_' + args.ckpt_tag) if args.ckpt_tag else ''}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")

        # CSV with per-position GT, pred, and error
        csv_out = out_dir / f"gt_vs_pred_env{eid}_{info.name}{('_' + args.ckpt_tag) if args.ckpt_tag else ''}.csv"
        with open(csv_out, "w") as fcsv:
            fcsv.write("gt_x,gt_y,pred_x,pred_y,err_m\n")
            for i in range(len(gts)):
                fcsv.write(f"{gts[i, 0]:.4f},{gts[i, 1]:.4f},"
                            f"{preds[i, 0]:.4f},{preds[i, 1]:.4f},"
                            f"{errs[i]:.4f}\n")
        print(f"  saved {csv_out}")


if __name__ == "__main__":
    main()
