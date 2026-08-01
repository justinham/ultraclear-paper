"""Quick eval + figure regeneration for the ASTA 23-anchor industry1 run.

Loads the saved encoder.pt (full model + norm stats + anchor list) and
the same single pivoted CSV used in training, rebuilds the rotation
windows with the same T/stride/val_frac, runs the model in eval mode,
and writes:

  - eval_per_step.csv   (re-emitted, with both train and val preds)
  - val_gt_vs_pred.png  (dots only — no connecting lines)
  - train_gt_vs_pred.png (two-panel: left = tracking, right = error distribution)
  - val_err_hist.png
  - val_per_window_mae.png
  - training_curves.png  (read from history.csv if present, else skip)

This lets you re-render the figures from a 1000-epoch run without
re-training.

Run:
  python3.11 -m scripts.eval_asta_industry1_23anchor \\
      --ckpt checkpoints/_tmp_asta_industry1_23anchor_rot/encoder.pt \\
      --csv  figure/anc20/pivoted_superframe_distances.csv \\
      --out  checkpoints/_tmp_asta_industry1_23anchor_rot/_eval_redo
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.asta import ASTAEncoder
from ultraclear.config import Config


# --------------------------------------------------------------------------- #
# Anchor positions (industry1, 23 anchors, mm in the lab frame).
# hex id -> (x_mm, y_mm, z_mm). The 8 near-vertical pairs form the
# ceiling anchor frame; the rest are perimeter mounts.
# Coordinates must match the ckpt's `anchor_hex` ordering.
# --------------------------------------------------------------------------- #
ANCHOR_POS_MM = {
    "0x9BD8": (3995, 5414, 437),
    "0xE36B": (3874, -5137, 433),
    "0xE5A0": (3995, 5414, 2664),
    "0x9CD3": (-4028, 5420, 2608),
    "0xE4AF": (-3997, -5317, 2619),
    "0xE589": (3875, -5317, 2647),
    "0x9D0D": (-3997, -5137, 412),
    "0xE595": (-4029, 5420, 396),
    "0x1476": (-6359, -2613, 2760),
    "0x1566": (-9984, -5308, 2606),
    "0x1624": (-12487, -2614, 2753),
    "0x15AC": (-14581, -5314, 2598),
    "0x151F": (-16539, -2608, 2737),
    "0x1674": (-20622, -4683, 2478),
    "0x1533": (-20586, -776, 2487),
    "0x1642": (-20667, 2307, 2842),
    "0x171F": (-20615, 5306, 2787),
    "0x1578": (-14511, 208, 2762),
    "0x160D": (-12487, -1781, 2745),
    "0x1597": (-8401, 1020, 2765),
    "0x1570": (-14482, 2998, 2767),
    "0x14B8": (-14623, 5268, 1788),
    "0x9C86": (-8402, 3786, 2690),
}


# --------------------------------------------------------------------------- #
# Data loading (mirrors train_asta_industry1_23anchor.py)
# --------------------------------------------------------------------------- #

def load_pivoted_csv(csv_path: Path):
    with open(csv_path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = list(rdr)
    anchor_cols = [i for i, h in enumerate(header) if h.startswith("dist_")]
    gx, gy = header.index("gt_x"), header.index("gt_y")
    anchor_hex = [header[i].removeprefix("dist_") for i in anchor_cols]
    X = np.zeros((len(rows), len(anchor_cols)), dtype=np.float32)
    Y = np.zeros((len(rows), 2), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(anchor_cols):
            v = r[c].strip()
            X[i, j] = 0.0 if v == "" else float(v) / 1000.0   # mm -> m
        Y[i, 0] = float(r[gx])
        Y[i, 1] = float(r[gy])
    return X, Y, anchor_hex


def make_windows(X, Y, T, stride):
    N = len(X)
    if N < T:
        return (np.zeros((0, T, X.shape[1]), dtype=np.float32),
                np.zeros((0, T, 2), dtype=np.float32))
    starts = list(range(0, N - T + 1, stride))
    wx = np.stack([X[s:s + T] for s in starts])
    wy = np.stack([Y[s:s + T] for s in starts])
    keep = (wx.sum(axis=(1, 2)) > 0)
    return wx[keep], wy[keep]


# --------------------------------------------------------------------------- #
# Model (must match the trained one)
# --------------------------------------------------------------------------- #

class PerTimestepLocHead(torch.nn.Module):
    def __init__(self, d_model: int, n_anchors: int):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(d_model * n_anchors, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 2),
        )

    def forward(self, h):
        B, T, M, D = h.shape
        return self.head(h.reshape(B, T, M * D))


class ASTAIndustry1(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = ASTAEncoder(cfg)
        self.head = PerTimestepLocHead(cfg.d_model, cfg.n_anchors)

    def forward(self, x):
        return self.head(self.encoder(x))


# --------------------------------------------------------------------------- #
# Eval driver
# --------------------------------------------------------------------------- #

def run_eval(*, ckpt_path: Path, csv_path: Path, out_dir: Path,
             test_csv: str | None = None,
             device: str = "cpu") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load ckpt (the train script saved a dict with state + meta) ----
    bundle = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = bundle.get("config", {})
    norm_mean = float(bundle["norm_mean"])
    norm_std = float(bundle["norm_std"])
    saved_anchor_hex = bundle.get("anchor_hex", None)
    T = int(cfg_dict.get("T", 32))
    stride = int(cfg_dict.get("stride", 1))
    n_anchors = int(cfg_dict.get("n_anchors", 23))
    n_channels = int(cfg_dict.get("n_channels", 1))
    d_model = int(cfg_dict.get("d_model", 64))
    n_asta_blocks = int(cfg_dict.get("n_asta_blocks", 2))

    print(f"[ckpt]  {ckpt_path}")
    print(f"[ckpt]  T={T}  stride={stride}  M={n_anchors}  F={n_channels}  "
          f"d={d_model}  blocks={n_asta_blocks}")
    print(f"[ckpt]  norm: mean={norm_mean:.3f} m  std={norm_std:.3f} m")

    # ---- Rebuild model and load weights ----
    cfg = Config(n_anchors=n_anchors, n_channels=n_channels,
                 d_model=d_model, n_asta_blocks=n_asta_blocks)
    model = ASTAIndustry1(cfg).to(device)
    state = bundle.get("model_state", bundle)
    if any(k.startswith("encoder.") or k.startswith("head.")
           for k in state.keys()):
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[load]  missing={len(missing)}  unexpected={len(unexpected)}")
    else:
        # try a couple of common key layouts
        for prefix in ("model.", "module."):
            prefixed = {prefix + k: v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(prefixed, strict=False)
            if not missing and not unexpected:
                break
    model.eval()

    # ---- Load data + rebuild windows ----
    X_all, Y_all, anchor_hex = load_pivoted_csv(csv_path)
    if saved_anchor_hex is not None and saved_anchor_hex != anchor_hex:
        print(f"[warn]  anchor order in CSV differs from ckpt — using CSV order")
    print(f"[data]  {csv_path.name}: X={X_all.shape}  Y={Y_all.shape}  "
          f"anchors={len(anchor_hex)}")
    wx_tr, wy_tr = make_windows(X_all, Y_all, T, stride)
    print(f"[data]  train windows: {wx_tr.shape} → {wy_tr.shape}  "
          f"(T={T}, stride={stride})")

    # If --test_csv given, use it as a 100% held-out test set
    # (different capture → no train/val leakage concern).  No
    # chronological/random split is applied; the whole test CSV
    # is evaluated as-is.
    n_tr = len(wx_tr)   # 100% of train CSV (no split in this mode)
    n_va = None         # will be set below
    if test_csv is not None:
        test_csv = Path(test_csv)
        if not test_csv.exists():
            raise FileNotFoundError(test_csv)
        X_te, Y_te, _ = load_pivoted_csv(test_csv)
        print(f"[data]  {test_csv.name}: X={X_te.shape}  Y={Y_te.shape}")
        wx_va, wy_va = make_windows(X_te, Y_te, T, stride)
        n_va = len(wx_va)
        print(f"[data]  test windows: {wx_va.shape} → {wy_va.shape}  "
              f"(100% held-out, no split)")
        split_label = "100% test (different capture)"
    else:
        # Chronological 80/20 split
        n_val_split = max(1, int(round(0.2 * len(wx_tr))))
        n_tr = len(wx_tr) - n_val_split
        wx_va, wy_va = wx_tr[n_tr:], wy_tr[n_tr:]
        wx_tr, wy_tr = wx_tr[:n_tr], wy_tr[:n_tr]
        n_va = len(wx_va)
        print(f"[split] {len(wx_tr)} train windows  |  {len(wx_va)} val windows")
        split_label = "chronological 80/20"

    # ---- Apply the saved norm (do NOT recompute) ----
    def norm(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        m = out > 0
        out[m] = (out[m] - norm_mean) / norm_std
        return out

    wx_tr_n = norm(wx_tr)
    wx_va_n = norm(wx_va)

    wx_tr_t = torch.from_numpy(wx_tr_n).to(device).unsqueeze(-1)  # (W, T, M, 1)
    wy_tr_t = torch.from_numpy(wy_tr).to(device)
    wx_va_t = torch.from_numpy(wx_va_n).to(device).unsqueeze(-1)
    wy_va_t = torch.from_numpy(wy_va).to(device)

    # ---- Predict (batched to keep memory low) ----
    bs = 16

    def predict(wx_t):
        out = []
        with torch.no_grad():
            for i in range(0, len(wx_t), bs):
                out.append(model(wx_t[i:i + bs]).cpu().numpy())
        return np.concatenate(out, axis=0)

    pred_tr = predict(wx_tr_t)
    pred_va = predict(wx_va_t)
    err_tr = np.linalg.norm(pred_tr - wy_tr, axis=-1)  # (W, T)
    err_va = np.linalg.norm(pred_va - wy_va, axis=-1)

    per_win_tr = err_tr.mean(axis=1)
    per_win_va = err_va.mean(axis=1)
    overall_tr = float(err_tr.mean())
    overall_va = float(err_va.mean())
    print()
    print(f"[result] TRAIN per-step MAE: {overall_tr:.3f} m  (n={n_tr} windows)")
    print(f"[result] VAL   per-step MAE: {overall_va:.3f} m  (n={n_va} windows)")
    print(f"[result] VAL   p50 = {float(np.median(err_va)):.3f} m   "
          f"p95 = {float(np.percentile(err_va, 95)):.3f} m   "
          f"max = {float(err_va.max()):.3f} m")

    # ---- Save per-step CSV ----
    eval_csv = out_dir / "eval_per_step.csv"
    with open(eval_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "window_idx", "t", "x_pred", "y_pred",
                    "x_true", "y_true", "err_m"])
        for wi in range(pred_va.shape[0]):
            for t in range(pred_va.shape[1]):
                w.writerow(["val", wi, t,
                            f"{pred_va[wi, t, 0]:.4f}",
                            f"{pred_va[wi, t, 1]:.4f}",
                            f"{wy_va[wi, t, 0]:.4f}",
                            f"{wy_va[wi, t, 1]:.4f}",
                            f"{err_va[wi, t]:.4f}"])
        for wi in range(pred_tr.shape[0]):
            for t in range(pred_tr.shape[1]):
                w.writerow(["train", wi, t,
                            f"{pred_tr[wi, t, 0]:.4f}",
                            f"{pred_tr[wi, t, 1]:.4f}",
                            f"{wy_tr[wi, t, 0]:.4f}",
                            f"{wy_tr[wi, t, 1]:.4f}",
                            f"{err_tr[wi, t]:.4f}"])
    print(f"[save]  {eval_csv}")

    # ---- Per-point RMSE CSV (one row per (window, t) prediction) ----
    # Squared error per step in meters^2, so the CSV holds raw, MSE-equivalent
    # numbers; users can sqrt() or mean() as needed. Same shapes as
    # eval_per_step.csv, but slimmer: split, window_idx, t, gt_xy, pred_xy,
    # err_m, err_squared_m2.
    rmse_csv = out_dir / "per_point_rmse.csv"
    with open(rmse_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "window_idx", "t",
                     "x_true_m", "y_true_m",
                     "x_pred_m", "y_pred_m",
                     "err_m", "err_squared_m2"])
        for split, pred, wy, err in (("val", pred_va, wy_va, err_va),
                                      ("train", pred_tr, wy_tr, err_tr)):
            for wi in range(pred.shape[0]):
                for t in range(pred.shape[1]):
                    wr.writerow([
                        split,
                        wi,
                        t,
                        f"{wy[wi, t, 0]:.4f}",
                        f"{wy[wi, t, 1]:.4f}",
                        f"{pred[wi, t, 0]:.4f}",
                        f"{pred[wi, t, 1]:.4f}",
                        f"{err[wi, t]:.4f}",
                        f"{err[wi, t] ** 2:.6f}",
                    ])
    # Per-window aggregate RMSE (one row per window) — the actual RMSE number
    # people usually quote. err_per_window_m = sqrt(mean(err_t^2)).
    rmse_by_window_csv = out_dir / "per_window_rmse.csv"
    with open(rmse_by_window_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "window_idx", "rmse_m", "mae_m", "p95_m",
                     "max_m", "n_steps"])
        for split, pred, wy, err in (("val", pred_va, wy_va, err_va),
                                      ("train", pred_tr, wy_tr, err_tr)):
            for wi in range(err.shape[0]):
                e = err[wi]
                wr.writerow([
                    split,
                    wi,
                    f"{np.sqrt(np.mean(e ** 2)):.4f}",
                    f"{np.mean(e):.4f}",
                    f"{np.percentile(e, 95):.4f}",
                    f"{e.max():.4f}",
                    int(e.size),
                ])
    print(f"[save]  {rmse_csv}")
    print(f"[save]  {rmse_by_window_csv}")

    # ---- Figure 1: val scatter (dots only) ----
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.scatter(wy_va[..., 0].flatten(), wy_va[..., 1].flatten(),
               c='green', s=8, alpha=0.5, label='GT')
    ax.scatter(pred_va[..., 0].flatten(), pred_va[..., 1].flatten(),
               c='red', s=8, alpha=0.5, label='Pred')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'ASTA 23-anchor — VAL ({split_label})\n'
                 f'val MAE = {overall_va:.3f} m, '
                 f'n_val_windows = {n_va}, T = {T}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    val_scatter = out_dir / "val_gt_vs_pred.png"
    fig.savefig(val_scatter, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {val_scatter}")

    # ---- Figure 2: train combined (tracking + error distribution) ----
    # Two-panel figure: left = GT-vs-pred scatter (tracking overlay)
    # with anchor positions overlaid as red triangles; right = per-step
    # error histogram (error distribution). Width ratio 2:1 (left
    # gets twice the width) so the wide tracking layout has room.
    # Style matches draw_combined_localization_tracking.py /
    # draw_tier1_* scripts: GT = small gray dots with gray connecting
    # lines, Pred = blue 'x' markers.
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 7),
                                       constrained_layout=True,
                                       gridspec_kw={"width_ratios": [2, 1]})
    # Left panel: tracking overlay. Keep the same data flattening as
    # the original (N*T dots per series) so the figure is comparable.
    # GT = small gray dots with gray connecting lines; Pred = blue 'x'
    # markers. Anchors = red triangles, drawn on top for context.
    wy_flat = wy_tr[..., 0].flatten(), wy_tr[..., 1].flatten()
    pr_flat = pred_tr[..., 0].flatten(), pred_tr[..., 1].flatten()
    # Connecting lines first so they sit underneath the markers.
    for gx, gy, px, py in zip(wy_flat[0], wy_flat[1], pr_flat[0], pr_flat[1]):
        ax_l.plot([gx, px], [gy, py], color='gray', lw=0.3,
                  alpha=0.55, zorder=1)
    ax_l.scatter(wy_flat[0], wy_flat[1],
                 c='gray', s=15, alpha=0.7, zorder=2,
                 edgecolors='none')
    ax_l.scatter(pr_flat[0], pr_flat[1],
                 c='tab:blue', s=70, marker='x', linewidths=1.6,
                 alpha=0.55, zorder=4)
    # 23 anchors (industry1) overlaid as red triangles in meters.
    anc_xy = np.array([ANCHOR_POS_MM[h][:2] for h in anchor_hex]) / 1000.0
    ax_l.scatter(anc_xy[:, 0], anc_xy[:, 1],
                 c='red', s=60, marker='^', edgecolor='black',
                 linewidth=0.5, zorder=5)
    ax_l.set_xlabel('X (m)', fontsize=22)
    ax_l.set_ylabel('Y (m)', fontsize=22)
    ax_l.tick_params(axis='both', labelsize=19)
    ax_l.grid(True, alpha=0.3, linestyle='--')
    ax_l.set_ylim(-6.0, 6.0)
    # Custom legend handles so the GT entry shows the connecting
    # line (the dominant visual element) plus a small dot, instead
    # of just a dot.
    from matplotlib.lines import Line2D
    gt_handle = Line2D([0], [0], color='gray', lw=3.2,
                        marker='o', markersize=10, markerfacecolor='gray',
                        markeredgecolor='gray', alpha=0.7,
                        label='ground truth')
    pred_handle = Line2D([0], [0], color='tab:blue', lw=0,
                          marker='x', markersize=16, markeredgewidth=2.6,
                          alpha=0.55, label='UltraClear')
    ax_l.legend(handles=[gt_handle, pred_handle],
                loc='upper left', fontsize=19, framealpha=0.9)
    # Right panel: per-step error histogram (same convention as the
    # val_err_hist.png, just for the train split). Bins are forced
    # to span exactly [0, 1m] so the visible region is dense enough
    # to read — otherwise the natural binning extends past 1m and
    # only ~8 bars are visible in the window.
    err_tr_per_step = np.linalg.norm(pred_tr - wy_tr, axis=-1)
    # density=True normalizes so the histogram integrates to 1,
    # giving a probability (ratio) instead of raw counts.
    ax_r.hist(err_tr_per_step.flatten(), bins=np.linspace(0.0, 1.0, 41),
              color="tab:blue", edgecolor="black", alpha=0.75,
              density=True)
    ax_r.axvline(overall_tr, color='black', linestyle='--',
                 label=f"mean = {overall_tr:.3f} m")
    p95_tr = float(np.percentile(err_tr_per_step, 95))
    ax_r.axvline(p95_tr, color='red', linestyle='--',
                 label=f"p95 = {p95_tr:.3f} m")
    ax_r.set_xlabel("Euclidean error (m)", fontsize=22)
    ax_r.set_ylabel("p (ratio)", fontsize=22)
    ax_r.tick_params(axis='both', labelsize=19)
    # Hide the y-axis tick labels (numbers) and the axis spine, but
    # keep the y-axis title visible per user request.
    ax_r.set_yticklabels([])
    ax_r.tick_params(axis='y', length=0)
    # Custom legend handles so we can add a "error distribution" bar
    # entry alongside the mean / p95 dashed lines.
    from matplotlib.patches import Patch
    dist_handle = Patch(facecolor='tab:blue', edgecolor='black',
                          alpha=0.75, label='error distribution')
    mean_handle = Line2D([0], [0], color='black', lw=1.5, linestyle='--',
                          label=f"mean = {overall_tr:.3f} m")
    p95_handle = Line2D([0], [0], color='red', lw=1.5, linestyle='--',
                         label=f"p95 = {p95_tr:.3f} m")
    ax_r.legend(handles=[dist_handle, mean_handle, p95_handle],
                loc='upper right', fontsize=18)
    ax_r.grid(True, axis='y', linestyle=':', alpha=0.4)
    ax_r.set_xlim(0.0, 1.0)
    train_scatter = out_dir / "train_gt_vs_pred.png"
    fig.savefig(train_scatter, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {train_scatter}")

    # ---- Figure 3: val per-step error histogram ----
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.hist(err_va.flatten(), bins=40, color="tab:blue", edgecolor="black",
            alpha=0.75)
    ax.axvline(overall_va, color='black', linestyle='--',
               label=f"mean = {overall_va:.3f} m")
    p50 = float(np.median(err_va))
    p95 = float(np.percentile(err_va, 95))
    ax.axvline(p50, color='orange', linestyle='--',
               label=f"p50 = {p50:.3f} m")
    ax.axvline(p95, color='red', linestyle='--',
               label=f"p95 = {p95:.3f} m")
    ax.set_xlabel("per-step Euclidean error (m)")
    ax.set_ylabel("count")
    ax.set_title(f"Per-step error distribution (val — {split_label})")
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    hist_path = out_dir / "val_err_hist.png"
    fig.savefig(hist_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {hist_path}")

    # ---- Figure 4: per-window MAE bar (val) ----
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar(np.arange(len(per_win_va)), per_win_va, color="tab:blue", alpha=0.7)
    ax.axhline(per_win_va.mean(), color='black', linestyle='--',
               label=f"val mean = {per_win_va.mean():.3f} m")
    ax.set_xlabel("validation window index (chronological)")
    ax.set_ylabel("per-window MAE (m)")
    ax.set_title(f"Per-window MAE (val, n={n_va}, {split_label})")
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    bar_path = out_dir / "val_per_window_mae.png"
    fig.savefig(bar_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {bar_path}")

    # ---- Figure 5: training curves (from history.csv if present) ----
    hist_csv = ckpt_path.parent / "history.csv"
    if hist_csv.exists():
        eps, tr_mae, va_mae = [], [], []
        with open(hist_csv) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                eps.append(int(row[0]))
                tr_mae.append(float(row[2]))
                va_mae.append(float(row[3]))
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        ax.plot(eps, tr_mae, '-', color='tab:blue', label='train MAE (m)')
        ax.plot(eps, va_mae, '-', color='tab:orange', label='val MAE (m)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('per-step MAE (m)')
        ax.set_title('Training curves (ASTA 23-anchor, industry1)')
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
        curves = out_dir / "training_curves.png"
        fig.savefig(curves, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"[save]  {curves}")
    else:
        print(f"[skip]  no history.csv at {hist_csv} — skipping curves")

    return {
        "n_train": int(n_tr),
        "n_val": int(n_va),
        "T": int(T),
        "stride": int(stride),
        "train_mae_m": float(overall_tr),
        "val_mae_m": float(overall_va),
        "val_mae_p50_m": float(np.median(err_va)),
        "val_mae_p95_m": float(np.percentile(err_va, 95)),
        "val_mae_max_m": float(err_va.max()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="path to encoder.pt saved by train_asta_industry1_23anchor")
    p.add_argument("--csv",
                   default="/Users/justin/ultraclear-paper/figure/anc20/"
                           "pivoted_superframe_distances.csv",
                   help="train CSV; also used as the eval set if --test_csv "
                        "is not given (with chronological 80/20 split)")
    p.add_argument("--test_csv", default=None,
                   help="OPTIONAL: separate test CSV (100%% held-out, no "
                        "split). When given, --csv is used 100%% for train "
                        "eval and --test_csv is used 100%% for test eval. "
                        "Use this for cross-dataset test (different capture).")
    p.add_argument("--out", required=True,
                   help="output dir for the regenerated figures + CSVs")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    print("=" * 70)
    print("ASTA 23-anchor eval (no retrain — loads existing ckpt)")
    print(f"  ckpt: {ckpt_path}")
    if args.test_csv is not None:
        print(f"  train csv: {csv_path}  (100% train eval)")
        print(f"  test  csv: {args.test_csv}  (100% held-out test)")
    else:
        print(f"  csv:  {csv_path}  (chronological 80/20 split)")
    print(f"  out:  {out_dir}")
    print("=" * 70)
    print()

    summary = run_eval(ckpt_path=ckpt_path, csv_path=csv_path,
                       out_dir=out_dir, test_csv=args.test_csv,
                       device=args.device)
    print()
    print("summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
