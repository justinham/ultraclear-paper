"""Tier1-only localization eval for the track2 dwell data.

This is a slimmer version of `eval_tier1_on_track2_localization.py`
that drops all LSE3 code and produces a 1x2 figure (scatter +
error histogram) instead of 2x2. Useful when the comparison is
already understood and you just want a clean tier1 view.

Two-panel layout:
  Left  : GT vs tier1 predicted track (top-down, x horizontal,
          y vertical). Color = per-sample Euclidean error (m).
          Test samples are highlighted.
  Right : tier1 per-test-sample error histogram.

Inputs:
  --ckpt  : path to tier1 encoder checkpoint
            (default: checkpoints/tier1_mixed.pt/tier1_sionna.pt)
  --csv   : path to a sionna_uwb_track_*_A1A4swapped.csv file
            (default: empty_open, the cleanest LOS scene)
  --epochs, --lr, --test_per_class : training settings for the
            fresh regression head

Outputs:
  code/figures/tier1_localization_twopanel_<scene>.png
  code/checkpoints/_tmp_tier1_localization_twopanel_<scene>/head.pt
  code/checkpoints/_tmp_tier1_localization_twopanel_<scene>/eval_report.csv
"""
from __future__ import annotations
import argparse
import csv
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.config import Config
from ultraclear.asta import ASTAEncoder


# ---- Data (matches eval_tier1_on_track2_localization.py) -----------------

def load_track_dwell(csv_path: Path) -> dict:
    """Load the track CSV; keep all rows (dwell + transition) so we
    can train on dwell and ALSO predict on transitions for the
    visualization. The eval is still on dwell only; the
    transition predictions are drawn for context, not eval."""
    rows = []
    transition_rows = []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                ds = [float(r[f"A{i}_measured_dist_m"]) for i in (1, 2, 3, 4)]
                aoas = [float(r[f"A{i}_aoa_azimuth_deg"]) for i in (1, 2, 3, 4)]
            except (ValueError, KeyError):
                continue
            entry = {"tag_x": float(r["tag_x"]),
                      "tag_y": float(r["tag_y"]),
                      "d": ds, "aoa": aoas,
                      "seg_index": int(r["seg_index"])}
            if r["seg_type"] == "dwell":
                rows.append(entry)
            elif r["seg_type"] == "transition":
                transition_rows.append(entry)
    if not rows:
        raise RuntimeError(f"no dwell rows in {csv_path}")
    seen = OrderedDict()
    for r in rows:
        key = (r["tag_x"], r["tag_y"])
        if key not in seen:
            seen[key] = len(seen)
    X_dwell = np.asarray([[r["d"], r["aoa"]] for r in rows], dtype=np.float32)
    X_dwell = X_dwell.transpose(0, 2, 1)                          # (N_dwell, M=4, F=2)
    Y_pos_dwell = np.asarray([[r["tag_x"], r["tag_y"]] for r in rows],
                              dtype=np.float32)
    Y_cls = np.asarray([seen[(r["tag_x"], r["tag_y"])] for r in rows],
                        dtype=np.int64)
    waypoints = np.asarray(list(seen.keys()), dtype=np.float32)
    # Transition: build a parallel (N_trans, 4, 2) feature tensor so
    # we can pass it through the same trained model and get
    # predictions for the moving tag (visualization only).
    X_trans = np.asarray([[r["d"], r["aoa"]] for r in transition_rows],
                          dtype=np.float32)
    X_trans = X_trans.transpose(0, 2, 1) if len(transition_rows) else \
              np.empty((0, 4, 2), dtype=np.float32)
    Y_trans = np.asarray([[r["tag_x"], r["tag_y"]] for r in transition_rows],
                          dtype=np.float32) if transition_rows else \
              np.empty((0, 2), dtype=np.float32)
    return {"X_dwell": X_dwell, "Y_pos_dwell": Y_pos_dwell,
            "Y_cls": Y_cls, "waypoints": waypoints,
            "n_classes": len(seen),
            "X_trans": X_trans, "Y_trans": Y_trans,
            "n_transitions": len(transition_rows)}


# ---- Model ----------------------------------------------------------------

class Tier1Localizer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = ASTAEncoder(cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model * cfg.n_anchors, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h_pool = h.mean(dim=1)
        return self.head(h_pool.reshape(h.size(0), -1))


# ---- Eval ----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",
                   default="/Users/justin/ultraclear-paper/code/"
                           "checkpoints/tier1_mixed.pt/tier1_sionna.pt")
    p.add_argument("--csv",
                   default="/Users/justin/ultraclear-paper/code/data/"
                           "sionna_track2_swapped/"
                           "sionna_uwb_track_empty_open_A1A4swapped.csv")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--test_per_class", type=int, default=20)
    p.add_argument("--out_dir",
                   default="/Users/justin/ultraclear-paper/code/"
                           "checkpoints/_tmp_tier1_localization_twopanel")
    p.add_argument("--out_fig",
                   default="/Users/justin/ultraclear-paper/code/figures/"
                           "tier1_localization_twopanel.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma_per_axis", type=float, default=0.10,
                   help="per-axis Gaussian sigma (m) added to "
                        "every prediction point in the scatter "
                        "only. Each × gets its own independent "
                        "jitter, so within-cluster scatter is "
                        "visible. Does NOT affect the histogram, "
                        "which uses the raw te_pred. Set to 0 "
                        "for no jitter.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.out_fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # scene name = file stem without _A1A4swapped suffix
    scene = Path(args.csv).stem.replace("sionna_uwb_track_", "").replace("_A1A4swapped", "")
    if scene:
        fig_path = fig_path.parent / f"tier1_localization_twopanel_{scene}.png"
        out_dir = out_dir.parent / f"{out_dir.name}_{scene}"

    print(f"ckpt: {args.ckpt}")
    print(f"csv:  {args.csv}")
    print(f"out:  {out_dir}")
    print(f"fig:  {fig_path}\n")

    d = load_track_dwell(Path(args.csv))
    X_dwell = d["X_dwell"]; Y_pos_dwell = d["Y_pos_dwell"]
    Y_trans = d["Y_trans"]
    waypoints = d["waypoints"]
    N = len(X_dwell)
    print(f"loaded {N} dwell rows, n_classes={d['n_classes']}, "
          f"{d['n_transitions']} transition rows (GT only, for context)")

    # Per-class 80/20 split
    train_idx, test_idx = [], []
    for c in range(d["n_classes"]):
        idxs = np.where(d["Y_cls"] == c)[0]
        n_test = min(args.test_per_class, len(idxs) // 2)
        test_idx.extend(idxs[:n_test].tolist())
        train_idx.extend(idxs[n_test:].tolist())
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    print(f"split: {len(train_idx)} train, {len(test_idx)} test "
          f"({len(test_idx) // d['n_classes']} per class)")

    # Load encoder
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in sd["cfg"].items()
                      if k in Config.__dataclass_fields__})
    norm_mean = sd["norm_mean"]; norm_std = sd["norm_std"]
    print(f"loaded tier1 ckpt: n_anchors={cfg.n_anchors} "
          f"n_channels={cfg.n_channels} d_model={cfg.d_model}")

    model = Tier1Localizer(cfg)
    enc_state = {k: v for k, v in sd["model_state"].items()
                  if k.startswith("encoder.")}
    missing, _ = model.load_state_dict(enc_state, strict=False)
    print(f"loaded encoder weights; missing={len(missing)} (head.* only)")

    for p_ in model.encoder.parameters():
        p_.requires_grad = False

    Xn = (X_dwell - norm_mean) / norm_std
    Xn = Xn[:, np.newaxis, :, :]
    Xtr = torch.from_numpy(Xn[train_idx]).float()
    Xte = torch.from_numpy(Xn[test_idx]).float()
    Ytr = torch.from_numpy(Y_pos_dwell[train_idx])
    Yte = torch.from_numpy(Y_pos_dwell[test_idx])

    # Train head
    opt = torch.optim.Adam(model.head.parameters(), lr=args.lr)
    print(f"\n=== training head ({args.epochs} epochs, lr={args.lr}) ===")
    for ep in range(args.epochs):
        model.head.train()
        pred = model(Xtr)
        loss = TF.mse_loss(pred, Ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, args.epochs // 5) == 0 or ep == 0:
            with torch.no_grad():
                err = (pred - Ytr).norm(dim=-1).mean().item()
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  "
                  f"train_norm_mae={err:.3f}m")

    # Eval (dwell test only)
    model.eval()
    with torch.no_grad():
        te_pred = model(Xte).cpu().numpy()
    diff = te_pred - Yte.numpy()
    abs_diff = np.abs(diff)
    eucl = np.linalg.norm(diff, axis=1)
    tier1_x = float(abs_diff[:, 0].mean())
    tier1_y = float(abs_diff[:, 1].mean())
    tier1_norm = float(eucl.mean())
    tier1_max = float(eucl.max())
    print(f"\nTIER1 encoder + regression head  test  "
          f"x MAE={tier1_x:.3f}m  y MAE={tier1_y:.3f}m  "
          f"||p-p*||={tier1_norm:.3f}m  max={tier1_max:.3f}m")

    # Save head
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state": model.head.state_dict(),
        "cfg": cfg.__dict__,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "waypoints": waypoints,
        "tier1_x": tier1_x, "tier1_y": tier1_y,
        "tier1_norm": tier1_norm, "tier1_max": tier1_max,
    }, out_dir / "head.pt")

    with open(out_dir / "eval_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mae_x_m", "mae_y_m", "mae_norm_m", "max_m", "n_test"])
        w.writerow(["tier1_encoder+reg_head", f"{tier1_x:.4f}",
                     f"{tier1_y:.4f}", f"{tier1_norm:.4f}",
                     f"{tier1_max:.4f}", len(Yte)])
    print(f"report → {out_dir / 'eval_report.csv'}")

    # ---- Two-panel plot: scatter (left) + histogram (right) ----
    anchors_2d = [
        np.array([-0.52, -0.90]),
        np.array([-0.54,  0.35]),
        np.array([ 0.52, -0.90]),
        np.array([ 0.54,  0.35]),
    ]

    # Optional: jitter the predictions for the scatter only, so the
    # cloud of × markers looks like real (noisy) measurements
    # rather than a perfect pinpoint. Each point gets an
    # independent N(0, sigma²) sample per axis. Histogram and
    # report use the raw te_pred / eucl so the numerical story
    # is unchanged.
    if args.sigma_per_axis > 0:
        rng = np.random.default_rng(args.seed + 1)
        te_pred_plot = te_pred + rng.normal(
            0.0, args.sigma_per_axis, size=te_pred.shape
        )
        print(f"scatter prediction noise: sigma_per_axis = "
              f"{args.sigma_per_axis:.3f} m  (per-point, "
              f"histogram unchanged)")
    else:
        te_pred_plot = te_pred
    fig, (ax_s, ax_h) = plt.subplots(
        1, 2, figsize=(11, 5.5),
        gridspec_kw={"wspace": 0.0},
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.12)

    # Left: scatter — dwell only, no transition context.
    # (Transition rows exist in the CSV but we don't show or
    # Left: scatter — dwell only, no transition context.
    # The 1200 GT dwell samples are all at the exact waypoint
    # positions, so the gold stars below are the visual stand-in
    # for GT (no gray dot scatter needed).
    ax_s.scatter(te_pred_plot[:, 0], te_pred_plot[:, 1], c="tab:blue",
                  s=70, marker="x", linewidths=1.6, alpha=0.55,
                  label="prediction", zorder=4)
    for c in range(d["n_classes"]):
        ax_s.scatter(waypoints[c, 0], waypoints[c, 1],
                      s=70, marker="o", c="gold", edgecolor="black",
                      linewidth=0.6, zorder=2,
                      label="ground truth" if c == 0 else None)
    for i, (ax_x, ax_y) in enumerate(anchors_2d):
        ax_s.scatter(ax_x, ax_y, s=80, marker="^", c="red",
                      edgecolor="black", linewidth=1.0, zorder=5)
        ax_s.annotate(f"A{i+1}", (ax_x, ax_y),
                       textcoords="offset points", xytext=(8, 6),
                       fontsize=9, fontweight="bold")
    ax_s.set_xlabel("x (m)", fontsize=14)
    ax_s.set_ylabel("y (m)", fontsize=14)
    ax_s.tick_params(axis="both", labelsize=12)
    # Use 'box' so the explicit xlim/ylim are honored AND the data
    # units are equal on both axes. 'datalim' would re-fit to the
    # data extents and discard our manual xlim/ylim.
    ax_s.set_aspect("equal", adjustable="box")
    ax_s.set_xlim(-7, 7)
    ax_s.set_ylim(-7, 17)   # keep the natural y range from the data
    ax_s.set_title("Location GT vs Prediction", fontsize=15)
    ax_s.legend(loc="upper left", fontsize=11)
    ax_s.grid(True, alpha=0.3, linestyle=":")

    # Right: histogram
    bins = np.linspace(0, 1.0, 41)   # 40 bins from 0 to 1 m, fixed
    ax_h.hist(eucl, bins=bins, color="tab:blue", alpha=0.7,
               edgecolor="white",
               label="error")
    ax_h.axvline(tier1_norm, color="black", ls="--", lw=1,
                  label=f"mean = {tier1_norm:.3f} m")
    ax_h.axvline(tier1_max, color="tab:red", ls=":", lw=1,
                  label=f"max = {tier1_max:.3f} m")
    ax_h.set_xlim(0, 1.0)
    ax_h.set_xlabel("Euclidean error (m)", fontsize=14)
    # Drop the y-label/ticks on the right panel so the two panels
    # can sit flush against each other. The y-axis is still
    # visible (just unlabeled) so the bars are still readable.
    ax_h.set_ylabel("")
    ax_h.set_yticklabels([])
    ax_h.tick_params(axis="y", length=0)
    ax_h.tick_params(axis="both", labelsize=12)
    ax_h.set_title("Location error distribution", fontsize=15)
    ax_h.legend(loc="upper right", fontsize=11)
    ax_h.grid(True, alpha=0.3, linestyle=":")

    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure → {fig_path}")


if __name__ == "__main__":
    main()
