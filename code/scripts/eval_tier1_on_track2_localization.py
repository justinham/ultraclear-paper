"""Validate the track2 12-waypoint tracking using the pretrained
tier1 encoder, this time for CONTINUOUS localization (x, y)
regression instead of 12-waypoint classification.

The 12-class eval in `eval_tier1_on_track2_12waypoint.py`
quantized the localization task to nearest-waypoint. This
script goes one level deeper: train a head that outputs
(x, y) directly, and report per-axis + Euclidean error.

Same setup as the classification version, but:
  - label: (tag_x, tag_y) from the dwell row
  - head: 2-dim linear (or small MLP) output
  - loss: MSE on (x, y)
  - eval: per-axis MAE, Euclidean norm MAE, max error
  - baseline: LSE3 directly (no nearest-waypoint snap)

The interesting question: does the tier1 encoder features
give a model that beats LSE3 in absolute position error,
not just in classification accuracy?

Usage:
    python3.11 -m scripts.eval_tier1_on_track2_localization
    python3.11 -m scripts.eval_tier1_on_track2_localization \
        --csv data/sionna_track2_swapped/sionna_uwb_track_indoor_blockages_pillars_A1A4swapped.csv
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


# ---- Data (same schema as the 12-class script) ---------------------------

def load_track_dwell(csv_path: Path) -> dict:
    rows = []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if r["seg_type"] != "dwell":
                continue
            try:
                ds = [float(r[f"A{i}_measured_dist_m"]) for i in (1, 2, 3, 4)]
                aoas = [float(r[f"A{i}_aoa_azimuth_deg"]) for i in (1, 2, 3, 4)]
                rows.append({"tag_x": float(r["tag_x"]),
                              "tag_y": float(r["tag_y"]),
                              "d": ds, "aoa": aoas,
                              "seg_index": int(r["seg_index"])})
            except (ValueError, KeyError):
                continue
    if not rows:
        raise RuntimeError(f"no dwell rows in {csv_path}")
    seen = OrderedDict()
    for r in rows:
        key = (r["tag_x"], r["tag_y"])
        if key not in seen:
            seen[key] = len(seen)
    X = np.asarray([[r["d"], r["aoa"]] for r in rows], dtype=np.float32)  # (N, 4, 2)
    X = X.transpose(0, 2, 1)                                               # (N, M=4, F=2)
    Y_pos = np.asarray([[r["tag_x"], r["tag_y"]] for r in rows], dtype=np.float32)
    Y_cls = np.asarray([seen[(r["tag_x"], r["tag_y"])] for r in rows], dtype=np.int64)
    waypoints = np.asarray(list(seen.keys()), dtype=np.float32)
    return {
        "X": X, "Y_pos": Y_pos, "Y_cls": Y_cls,
        "waypoints": waypoints, "n_classes": len(seen),
    }


# ---- Model: same encoder, regression head --------------------------------

class Tier1Localizer(nn.Module):
    """Frozen tier1 encoder + a fresh 2-dim regression head."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = ASTAEncoder(cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model * cfg.n_anchors, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, M, F). Returns (B, 2) (x, y)."""
        h = self.encoder(x)
        h_pool = h.mean(dim=1)
        return self.head(h_pool.reshape(h.size(0), -1))


# ---- LSE baseline ---------------------------------------------------------

def lse_2d(d: np.ndarray, anchor_list) -> np.ndarray:
    valid = ~np.isnan(d)
    if valid.sum() < 3:
        return np.array([np.nan, np.nan])
    valid_idx = np.where(valid)[0]
    p0 = anchor_list[valid_idx[0]]
    d0 = d[valid_idx[0]]
    A, b = [], []
    for k in valid_idx[1:]:
        pi, di = anchor_list[k], d[k]
        A.append(2.0 * (pi - p0))
        b.append(d0 ** 2 - di ** 2 + np.dot(pi, pi) - np.dot(p0, p0))
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    xy, *_ = np.linalg.lstsq(A, b, rcond=None)
    return xy


# ---- Eval -----------------------------------------------------------------

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
                           "checkpoints/_tmp_tier1_localization")
    p.add_argument("--out_fig",
                   default="/Users/justin/ultraclear-paper/code/figures/"
                           "tier1_localization_eval.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.out_fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"ckpt: {args.ckpt}")
    print(f"csv:  {args.csv}\n")

    d = load_track_dwell(Path(args.csv))
    X = d["X"]; Y_pos = d["Y_pos"]; waypoints = d["waypoints"]
    N = len(X)
    print(f"loaded {N} dwell rows, n_classes={d['n_classes']}")

    # Per-class 80/20 split (same scheme as the 12-class script)
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
          f"n_channels={cfg.n_channels} d_model={cfg.d_model} "
          f"n_asta_blocks={cfg.n_asta_blocks}")
    print(f"normalization: mean={norm_mean}, std={norm_std}")

    model = Tier1Localizer(cfg)
    enc_state = {k: v for k, v in sd["model_state"].items()
                  if k.startswith("encoder.")}
    missing, _ = model.load_state_dict(enc_state, strict=False)
    print(f"loaded encoder weights; missing={len(missing)} (head.* only)")

    # Freeze encoder
    for p_ in model.encoder.parameters():
        p_.requires_grad = False

    # Normalize + T=1 window
    Xn = (X - norm_mean) / norm_std
    Xn = Xn[:, np.newaxis, :, :]
    Xtr = torch.from_numpy(Xn[train_idx]).float()
    Xte = torch.from_numpy(Xn[test_idx]).float()
    Ytr = torch.from_numpy(Y_pos[train_idx])
    Yte = torch.from_numpy(Y_pos[test_idx])

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
                  f"train_rmse={err**0.5:.3f}m  train_norm_mae={err:.3f}m")

    # Eval
    model.eval()
    with torch.no_grad():
        te_pred = model(Xte).cpu().numpy()
    diff = te_pred - Yte.numpy()
    abs_diff = np.abs(diff)
    eucl = np.linalg.norm(diff, axis=1)
    asta_x = float(abs_diff[:, 0].mean())
    asta_y = float(abs_diff[:, 1].mean())
    asta_norm = float(eucl.mean())
    asta_max = float(eucl.max())
    print(f"\nTIER1 encoder + regression head  test  "
          f"x MAE={asta_x:.3f}m  y MAE={asta_y:.3f}m  "
          f"||p-p*||={asta_norm:.3f}m  max={asta_max:.3f}m")

    # LSE baseline
    anchors_2d = [
        np.array([-0.52, -0.90]),
        np.array([-0.54,  0.35]),
        np.array([ 0.52, -0.90]),
        np.array([ 0.54,  0.35]),
    ]
    lse_pred = np.zeros_like(Yte.numpy())
    for j, i in enumerate(test_idx):
        d_meas = d["X"][i, :, 0]
        lse_pred[j] = lse_2d(d_meas, anchors_2d)
    diff_l = lse_pred - Yte.numpy()
    abs_diff_l = np.abs(diff_l)
    eucl_l = np.linalg.norm(diff_l, axis=1)
    lse_x = float(np.nanmean(abs_diff_l[:, 0]))
    lse_y = float(np.nanmean(abs_diff_l[:, 1]))
    lse_norm = float(np.nanmean(eucl_l))
    lse_max = float(np.nanmax(eucl_l))
    print(f"LSE3 (4-anchor closed-form)         test  "
          f"x MAE={lse_x:.3f}m  y MAE={lse_y:.3f}m  "
          f"||p-p*||={lse_norm:.3f}m  max={lse_max:.3f}m")

    # Save head
    torch.save({
        "head_state": model.head.state_dict(),
        "cfg": cfg.__dict__,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "waypoints": waypoints,
        "asta_x": asta_x, "asta_y": asta_y,
        "asta_norm": asta_norm, "asta_max": asta_max,
        "lse_x": lse_x, "lse_y": lse_y,
        "lse_norm": lse_norm, "lse_max": lse_max,
    }, out_dir / "head.pt")

    with open(out_dir / "eval_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mae_x_m", "mae_y_m", "mae_norm_m", "max_m", "n_test"])
        w.writerow(["tier1_encoder+reg_head", f"{asta_x:.4f}",
                     f"{asta_y:.4f}", f"{asta_norm:.4f}",
                     f"{asta_max:.4f}", len(Yte)])
        w.writerow(["lse3_4anchor", f"{lse_x:.4f}", f"{lse_y:.4f}",
                     f"{lse_norm:.4f}", f"{lse_max:.4f}", len(Yte)])
    print(f"report → {out_dir / 'eval_report.csv'}")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)

    # Top-left: predicted vs GT track, color by error
    # Convention here: x on horizontal axis, y on vertical axis.
    # (The track2 path is 8m wide in x and 14m tall in y, so a
    # natural top-down view has the long y axis as vertical.)
    ax = axes[0, 0]
    # All GT points in light gray for context
    ax.scatter(Y_pos[:, 0], Y_pos[:, 1], c="lightgray", s=10,
                alpha=0.5, zorder=1, label="GT (all 1200)")
    # Test GT in black
    ax.scatter(Yte[:, 0], Yte[:, 1], c="black", s=40, marker="o",
                label="GT (test)", zorder=3)
    # Tier1 predicted in blue
    sc = ax.scatter(te_pred[:, 0], te_pred[:, 1], c=eucl, cmap="viridis",
                     s=80, marker="^", edgecolor="navy", lw=0.6,
                     label="tier1 predicted", zorder=4)
    # LSE predicted in red
    ax.scatter(lse_pred[:, 0], lse_pred[:, 1], c=eucl_l, cmap="plasma",
                s=50, marker="x", linewidths=1.2,
                label="LSE3 predicted", zorder=4)
    # Waypoint stars
    for c in range(d["n_classes"]):
        ax.scatter(waypoints[c, 0], waypoints[c, 1],
                    s=180, marker="*", c="gold", edgecolor="black",
                    linewidth=0.8, zorder=2)
    for i, (ax_x, ax_y) in enumerate(anchors_2d):
        ax.scatter(ax_x, ax_y, s=200, marker="^", c="red",
                    edgecolor="black", linewidth=1.0, zorder=5)
        ax.annotate(f"A{i+1}", (ax_x, ax_y),
                     textcoords="offset points", xytext=(8, 6),
                     fontsize=8, fontweight="bold")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("GT vs predicted (test set, 240 samples)\n"
                  "color = Euclidean error (m)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=":")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("tier1 err (m)")

    # Top-right: error histogram
    ax = axes[0, 1]
    bins = np.linspace(0, max(asta_max, lse_max) * 1.05, 40)
    ax.hist(eucl, bins=bins, color="tab:blue", alpha=0.65,
            label=f"tier1 (mean {asta_norm:.2f}m)", edgecolor="white")
    ax.hist(eucl_l, bins=bins, color="tab:red", alpha=0.55,
            label=f"LSE3 (mean {lse_norm:.2f}m)", edgecolor="white")
    ax.set_xlabel("Euclidean error (m)")
    ax.set_ylabel("count")
    ax.set_title("per-test-sample error distribution")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")

    # Bottom-left: per-axis error bars
    ax = axes[1, 0]
    methods = ["tier1 encoder\n+ reg head", "LSE3\n(4-anchor)"]
    x_errs = [asta_x, lse_x]
    y_errs = [asta_y, lse_y]
    norm_errs = [asta_norm, lse_norm]
    width = 0.25
    x_pos = np.arange(len(methods))
    ax.bar(x_pos - width, x_errs, width, color="tab:cyan", label="x MAE")
    ax.bar(x_pos,         y_errs, width, color="tab:olive", label="y MAE")
    ax.bar(x_pos + width, norm_errs, width, color="tab:purple", label="||p-p*|| MAE")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("MAE (m)")
    ax.set_title("per-axis + norm MAE (lower is better)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":", axis="y")
    for i, (x, y, n) in enumerate(zip(x_errs, y_errs, norm_errs)):
        ax.text(i - width, x + 0.01, f"{x:.2f}", ha="center", va="bottom",
                 fontsize=9, color="tab:cyan", fontweight="bold")
        ax.text(i, y + 0.01, f"{y:.2f}", ha="center", va="bottom",
                 fontsize=9, color="tab:olive", fontweight="bold")
        ax.text(i + width, n + 0.01, f"{n:.2f}", ha="center", va="bottom",
                 fontsize=9, color="tab:purple", fontweight="bold")

    # Bottom-right: per-waypoint error
    ax = axes[1, 1]
    wp_errs_asta = []
    wp_errs_lse = []
    for c in range(d["n_classes"]):
        m = (d["Y_cls"][test_idx] == c)
        if m.sum() == 0:
            wp_errs_asta.append(0); wp_errs_lse.append(0); continue
        wp_errs_asta.append(float(eucl[m].mean()))
        wp_errs_lse.append(float(np.nanmean(eucl_l[m])))
    x_pos = np.arange(d["n_classes"])
    width = 0.4
    ax.bar(x_pos - width / 2, wp_errs_asta, width, color="tab:blue",
            label="tier1")
    ax.bar(x_pos + width / 2, wp_errs_lse, width, color="tab:red",
            label="LSE3")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"wp{c}" for c in range(d["n_classes"])],
                        fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("per-waypoint mean Euclidean error (m)")
    ax.set_title("per-waypoint error (which waypoints is each method worst at?)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":", axis="y")

    fig.suptitle(
        f"tier1 encoder on track2 12-waypoint localization (regression)\n"
        f"{Path(args.csv).name}",
        fontsize=13,
    )
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure → {fig_path}")


if __name__ == "__main__":
    main()
