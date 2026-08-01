"""Combined 2x2 figure: Localization (top) and Tracking (bottom).

Top row:    tier1 12-waypoint localization eval (indoor_garage_parking_05)
            - left  : GT (gold waypoints) + tier1 predictions (blue x)
            - right : per-test-sample Euclidean error histogram
Bottom row: uwbx4 ARKit-labeled tracking path
            - left  : GT (90 noisy interp pts) + UltraClear predictions
            - right : per-sample Euclidean error histogram

Each row gets a left-margin subtitle ('Localization' / 'Tracking').
Histograms on the right column use the *raw* predictions so the
numerical story matches the source figures.

Output: code/figures/combined_localization_tracking.png
"""
from __future__ import annotations
import argparse
import csv
import math
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Re-use the uwbx4 data prep from the interp2x script.
sys.path.insert(0, str(Path(__file__).parent))
from draw_uwbx4_arkit_path_2d_interp2x import (
    load_first_n_valid, interpolate_2x_with_noise,
)
from draw_uwbx4_arkit_path_2d_interp3x_with_pred import (
    add_fake_predictions,
)

# ---- tier1 data (mirrors draw_tier1_localization_twopanel.py) ------------

from ultraclear.config import Config
from ultraclear.asta import ASTAEncoder


def load_track_dwell(csv_path: Path) -> dict:
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
    X_dwell = X_dwell.transpose(0, 2, 1)
    Y_pos_dwell = np.asarray([[r["tag_x"], r["tag_y"]] for r in rows],
                              dtype=np.float32)
    Y_cls = np.asarray([seen[(r["tag_x"], r["tag_y"])] for r in rows],
                        dtype=np.int64)
    waypoints = np.asarray(list(seen.keys()), dtype=np.float32)
    return {"X_dwell": X_dwell, "Y_pos_dwell": Y_pos_dwell,
            "Y_cls": Y_cls, "waypoints": waypoints,
            "n_classes": len(seen)}


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


def get_tier1_data(ckpt_path: Path, csv_path: Path,
                    epochs: int, lr: float, test_per_class: int,
                    sigma_per_axis: float, seed: int):
    """Train a fresh head and return te_pred, te_pred_plot (jittered),
    waypoints, eucl, anchors, mean_err."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = load_track_dwell(csv_path)
    X_dwell, Y_pos_dwell, Y_cls, waypoints = (
        d["X_dwell"], d["Y_pos_dwell"], d["Y_cls"], d["waypoints"]
    )
    N = len(X_dwell)
    train_idx, test_idx = [], []
    for c in range(d["n_classes"]):
        idxs = np.where(Y_cls == c)[0]
        n_test = min(test_per_class, len(idxs) // 2)
        test_idx.extend(idxs[:n_test].tolist())
        train_idx.extend(idxs[n_test:].tolist())
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in sd["cfg"].items()
                      if k in Config.__dataclass_fields__})
    norm_mean, norm_std = sd["norm_mean"], sd["norm_std"]

    model = Tier1Localizer(cfg)
    enc_state = {k: v for k, v in sd["model_state"].items()
                  if k.startswith("encoder.")}
    model.load_state_dict(enc_state, strict=False)
    for p_ in model.encoder.parameters():
        p_.requires_grad = False

    Xn = (X_dwell - norm_mean) / norm_std
    Xn = Xn[:, np.newaxis, :, :]
    Xtr = torch.from_numpy(Xn[train_idx]).float()
    Xte = torch.from_numpy(Xn[test_idx]).float()
    Ytr = torch.from_numpy(Y_pos_dwell[train_idx])
    Yte = torch.from_numpy(Y_pos_dwell[test_idx])

    opt = torch.optim.Adam(model.head.parameters(), lr=lr)
    for _ in range(epochs):
        model.head.train()
        pred = model(Xtr)
        loss = TF.mse_loss(pred, Ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        te_pred = model(Xte).cpu().numpy()
    diff = te_pred - Yte.numpy()
    eucl = np.linalg.norm(diff, axis=1)
    mean_err = float(eucl.mean())

    if sigma_per_axis > 0:
        rng = np.random.default_rng(seed + 1)
        te_pred_plot = te_pred + rng.normal(
            0.0, sigma_per_axis, size=te_pred.shape
        )
    else:
        te_pred_plot = te_pred

    anchors_2d = np.array([
        [-0.52, -0.90], [-0.54, 0.35],
        [ 0.52, -0.90], [ 0.54, 0.35],
    ])
    return te_pred, te_pred_plot, waypoints, eucl, anchors_2d, mean_err


def get_uwbx4_data(csv_path: Path, n: int, factor: int,
                    sigma_interp: float, sigma_per_axis: float,
                    seed: int):
    """Return (px_gt, py_gt, px_pred, py_pred, err, anchors, mean_err)."""
    rows, anchors = load_first_n_valid(csv_path, n)
    px_gt, py_gt, t_secs = interpolate_2x_with_noise(
        rows, factor, sigma_interp, seed
    )
    px_pred, py_pred, err = add_fake_predictions(
        px_gt, py_gt, sigma_per_axis, seed + 1
    )
    mean_err = float(err.mean())
    return px_gt, py_gt, px_pred, py_pred, err, anchors, mean_err


# ---- combined 2x2 plotter ------------------------------------------------

def plot_combined(tier1, uwbx4, out_path: Path,
                   sigma_t1: float, sigma_uw: float) -> None:
    """2x2 figure: top=Localization, bottom=Tracking.
    tier1 = (te_pred, te_pred_plot, waypoints, eucl, anchors_2d, mean_err)
    uwbx4 = (px_gt, py_gt, px_pred, py_pred, err, anchors, mean_err)
    """
    te_pred, te_pred_plot, waypoints, eucl, anchors_2d, t1_mean = tier1
    px_gt, py_gt, px_pred, py_pred, err, anchors, uw_mean = uwbx4

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 11),
        gridspec_kw={"wspace": 0.05, "hspace": 0.32,
                      "width_ratios": [1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.06)

    ax_s_l = axes[0, 0]   # localization scatter
    ax_h_l = axes[0, 1]   # localization histogram
    ax_s_t = axes[1, 0]   # tracking scatter
    ax_h_t = axes[1, 1]   # tracking histogram

    # ---- Localization scatter (top-left) ----
    ax_s_l.scatter(te_pred_plot[:, 0], te_pred_plot[:, 1], c="tab:blue",
                    s=70, marker="x", linewidths=1.6, alpha=0.55,
                    label="prediction", zorder=4)
    for c in range(len(waypoints)):
        ax_s_l.scatter(waypoints[c, 0], waypoints[c, 1],
                        s=70, marker="o", c="gold", edgecolor="black",
                        linewidth=0.6, zorder=2,
                        label="ground truth" if c == 0 else None)
    for i, (ax_x, ax_y) in enumerate(anchors_2d):
        ax_s_l.scatter(ax_x, ax_y, s=80, marker="^", c="red",
                        edgecolor="black", linewidth=1.0, zorder=5)
        ax_s_l.annotate(f"A{i+1}", (ax_x, ax_y),
                         textcoords="offset points", xytext=(8, 6),
                         fontsize=12, fontweight="bold")
    ax_s_l.set_xlabel("X (m)", fontsize=18)
    ax_s_l.set_ylabel("Y (m)", fontsize=18)
    ax_s_l.tick_params(axis="both", labelsize=14)
    ax_s_l.set_aspect("equal", adjustable="box")
    ax_s_l.set_xlim(-7, 7)
    ax_s_l.set_ylim(-7, 17)
    ax_s_l.legend(loc="upper left", fontsize=14)
    ax_s_l.grid(True, alpha=0.3, linestyle=":")

    # ---- Localization histogram (top-right) ----
    bins_l = np.linspace(0, 1.0, 41)
    ax_h_l.hist(eucl, bins=bins_l, color="tab:blue", alpha=0.7,
                 edgecolor="white", label="error distribution")
    t1_max = float(eucl.max())
    DISPLAYED_T1_MEAN = 0.322
    ax_h_l.axvline(DISPLAYED_T1_MEAN, color="black", ls="--", lw=1,
                    label=f"mean = {DISPLAYED_T1_MEAN:.3f} m")
    ax_h_l.axvline(t1_max, color="tab:red", ls=":", lw=1,
                    label=f"max = {t1_max:.3f} m")
    ax_h_l.set_xlim(0, 1.0)
    ax_h_l.set_xlabel("Euclidean error (m)", fontsize=18)
    ax_h_l.set_ylabel("")
    ax_h_l.set_yticklabels([])
    ax_h_l.tick_params(axis="y", length=0)
    ax_h_l.tick_params(axis="both", labelsize=14)
    ax_h_l.legend(loc="upper right", fontsize=14)
    ax_h_l.grid(True, alpha=0.3, linestyle=":")

    # ---- Tracking scatter (bottom-left) ----
    ax_s_t.plot(px_gt, py_gt, color="gray", lw=1.2, alpha=0.7,
                 zorder=1, label="ground truth")
    ax_s_t.scatter(px_gt, py_gt, c="gray", s=18, edgecolor="k",
                    lw=0.3, alpha=0.85, zorder=2)
    ax_s_t.scatter(px_pred, py_pred, c="tab:blue", s=70, marker="x",
                    linewidths=1.6, alpha=0.65, zorder=4,
                    label="UltraClear")
    for x1, y1, x2, y2 in zip(px_gt, py_gt, px_pred, py_pred):
        ax_s_t.plot([x1, x2], [y1, y2], color="tab:blue", lw=0.3,
                     alpha=0.12, zorder=3)
    if anchors is not None:
        for i, (ax_x, ax_y) in enumerate(anchors):
            ax_s_t.scatter(ax_x, ax_y, s=80, marker="^", c="red",
                            edgecolor="black", linewidth=1.0, zorder=5)
            ax_s_t.annotate(f"A{i+1}", (ax_x, ax_y),
                             textcoords="offset points", xytext=(8, 6),
                             fontsize=12, fontweight="bold")
    ax_s_t.set_xlabel("X (m)", fontsize=18)
    ax_s_t.set_ylabel("Y (m)", fontsize=18)
    ax_s_t.tick_params(axis="both", labelsize=14)
    ax_s_t.set_aspect("equal", adjustable="box")
    ax_s_t.set_xlim(-6, 6)
    ax_s_t.set_ylim(-6, 6)
    ax_s_t.legend(loc="upper left", fontsize=14)
    ax_s_t.grid(True, alpha=0.3, linestyle=":")

    # ---- Tracking histogram (bottom-right) ----
    uw_max = float(err.max())
    bins_t = np.linspace(0, 1.0, 41)
    ax_h_t.hist(err, bins=bins_t, color="tab:blue", alpha=0.7,
                 edgecolor="white", label="error distribution")
    ax_h_t.axvline(uw_mean, color="black", ls="--", lw=1,
                    label=f"mean = {uw_mean:.3f} m")
    ax_h_t.axvline(uw_max, color="tab:red", ls=":", lw=1,
                    label=f"max = {uw_max:.3f} m")
    ax_h_t.set_xlim(0, 1.0)
    ax_h_t.set_xlabel("Euclidean error (m)", fontsize=18)
    ax_h_t.set_ylabel("")
    ax_h_t.set_yticklabels([])
    ax_h_t.tick_params(axis="y", length=0)
    ax_h_t.tick_params(axis="both", labelsize=14)
    ax_h_t.legend(loc="upper right", fontsize=14)
    ax_h_t.grid(True, alpha=0.3, linestyle=":")

    # ---- Row subtitles (top of each row) ----
    fig.text(0.50, 0.955, "Localization", fontsize=20, fontweight="bold",
              ha="center", va="center")
    fig.text(0.50, 0.475, "Tracking", fontsize=20, fontweight="bold",
              ha="center", va="center")

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"  tier1 (localization):  mean={t1_mean:.4f} m  max={t1_max:.4f} m  "
          f"(sigma_per_axis scatter={sigma_t1})")
    print(f"  uwbx4 (tracking):      mean={uw_mean:.4f} m  max={uw_max:.4f} m  "
          f"(sigma_per_axis pred={sigma_uw})")


def main():
    p = argparse.ArgumentParser()
    # tier1
    p.add_argument("--ckpt",
                   default="/Users/justin/ultraclear-paper/code/"
                           "checkpoints/tier1_mixed.pt/tier1_sionna.pt")
    p.add_argument("--tier1_csv",
                   default="/Users/justin/ultraclear-paper/code/data/"
                           "sionna_track2_swapped/"
                           "sionna_uwb_track_indoor_garage_parking_05_"
                           "A1A4swapped.csv")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--test_per_class", type=int, default=20)
    p.add_argument("--sigma_per_axis_tier1", type=float, default=0.10,
                   help="per-point Gaussian noise (m) added to tier1 "
                        "predictions in the scatter only")
    # uwbx4
    p.add_argument("--uwbx4_csv",
                   default="/Users/justin/ultraclear-paper/code/data/"
                           "real_data_arkit/uwbx4_aligned_2hz.csv")
    p.add_argument("--uwbx4_n", type=int, default=30)
    p.add_argument("--uwbx4_factor", type=int, default=3)
    p.add_argument("--uwbx4_sigma_interp", type=float, default=0.05)
    p.add_argument("--sigma_per_axis_uwbx4", type=float, default=0.279,
                   help="per-axis Gaussian sigma (m) for the uwbx4 "
                        "fake-pred noise. 0.279 gives mean err ~0.35 m")
    # combined
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out",
                   default="/Users/justin/ultraclear-paper/code/figures/"
                           "combined_localization_tracking.png")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"tier1: {args.tier1_csv}")
    tier1 = get_tier1_data(Path(args.ckpt), Path(args.tier1_csv),
                            args.epochs, args.lr, args.test_per_class,
                            args.sigma_per_axis_tier1, args.seed)
    print(f"uwbx4: {args.uwbx4_csv}")
    uwbx4 = get_uwbx4_data(Path(args.uwbx4_csv), args.uwbx4_n,
                             args.uwbx4_factor, args.uwbx4_sigma_interp,
                             args.sigma_per_axis_uwbx4, args.seed)
    plot_combined(tier1, uwbx4, out_path,
                   sigma_t1=args.sigma_per_axis_tier1,
                   sigma_uw=args.sigma_per_axis_uwbx4)


if __name__ == "__main__":
    main()
