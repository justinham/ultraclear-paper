"""Quick eval + figure regeneration for the ASTA 23-anchor
office-rack sionna track run.

Loads the saved encoder.pt (full model + per-channel norm stats
+ config) and the same train/val scene CSVs used during training,
rebuilds the rotation windows with the same T/stride, runs the
model in eval mode, and writes:

  - eval_per_step.csv   (per-step preds for both train and val)
  - val_gt_vs_pred.png  (val scatter, dots only)
  - train_gt_vs_pred.png (train scatter, dots only)
  - val_err_hist.png   (val per-step error histogram)
  - val_per_window_mae.png
  - train_val_comparison.png   (NEW: 2-panel side-by-side: train on
                               the left, val on the right, both with
                               GT (green) and Pred (red))
  - per_window_train_vs_val.png  (NEW: bar chart comparing per-window
                                  MAE for train and val side by side
                                  at the same window indices — useful
                                  for spotting overfit windows)
  - training_curves.png  (read from history.csv if present)

Diffs vs scripts/eval_asta_industry1_23anchor.py:
  * 2 input channels (ToA + AoA_azimuth) instead of 1 (ToA only)
  * Per-channel z-score norm (mean/std are length-2 arrays)
  * 2 CSVs: --train_csv (train scene) and --val_csv (val scene),
    not 1 CSV with chronological 80/20 split
  * Anchor list is A1..A23 (not hex IDs)
  * Already in meters / degrees (no mm->m conversion)
  * No need to drop tag_z; we only read tag_x and tag_y directly

Run (defaults match the trainer's default big->right setup):
  python3.11 -m scripts.eval_asta_office_rack_track \\
      --ckpt  checkpoints/_tmp_asta_office_rack_big_to_right/encoder.pt \\
      --out   checkpoints/_tmp_asta_office_rack_big_to_right/_eval_redo
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
# Data loading (mirrors train_asta_office_rack_track.py)
# --------------------------------------------------------------------------- #

N_ANCHORS = 23
N_CHANNELS = 2
CHANNEL_NAMES = ("toa", "aoa_az")
TOA_CAP_M = 50.0
AOA_CAP_DEG = 180.0
DEFAULT_DATA_ROOT = Path(
    "/Users/justin/ultraclear-paper/code/data/office_rack_track_sionna"
)
SCENES = ("big", "middle", "right")


def load_office_rack_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a sionna_uwb_track_office_rack.csv.

    Returns:
        X:  (N, M=23, F=2) float32 — (ToA_m, AoA_azimuth_deg)
        Y:  (N, 2)         float32 — (tag_x, tag_y) in meters
    """
    df = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None,
                       encoding="utf-8")
    n_rows = len(df)
    X = np.zeros((n_rows, N_ANCHORS, N_CHANNELS), dtype=np.float32)
    Y = np.zeros((n_rows, 2), dtype=np.float32)

    Y[:, 0] = df["tag_x"].astype(np.float32)
    Y[:, 1] = df["tag_y"].astype(np.float32)

    for m in range(N_ANCHORS):
        a = m + 1
        toa = df[f"A{a}_measured_dist_m"].astype(np.float32)
        aoa = df[f"A{a}_aoa_azimuth_deg"].astype(np.float32)
        toa = np.where(np.isnan(toa), 0.0, toa)
        aoa = np.where(np.isnan(aoa), 0.0, aoa)
        toa = np.clip(toa, 0.0, TOA_CAP_M)
        aoa = np.clip(aoa, -AOA_CAP_DEG, AOA_CAP_DEG)
        X[:, m, 0] = toa
        X[:, m, 1] = aoa

    return X, Y


def make_windows(X, Y, T, stride):
    N = len(X)
    if N < T:
        return (np.zeros((0, T, X.shape[1], X.shape[2]), dtype=np.float32),
                np.zeros((0, T, 2), dtype=np.float32))
    starts = list(range(0, N - T + 1, stride))
    wx = np.stack([X[s:s + T] for s in starts])
    wy = np.stack([Y[s:s + T] for s in starts])
    keep = (wx[..., 0].sum(axis=(1, 2)) > 0)   # drop all-zero ToA windows
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


class ASTAOfficeRack(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = ASTAEncoder(cfg)
        self.head = PerTimestepLocHead(cfg.d_model, cfg.n_anchors)

    def forward(self, x):
        return self.head(self.encoder(x))


# --------------------------------------------------------------------------- #
# Eval driver
# --------------------------------------------------------------------------- #

def run_eval(*, ckpt_path: Path,
             train_csv: Path, val_csv: Path,
             out_dir: Path,
             device: str = "cpu") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load ckpt (the train script saved a dict with state + meta) ----
    bundle = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = bundle.get("config", {})
    norm_mean = np.asarray(bundle["norm_mean"], dtype=np.float32)  # (F,)
    norm_std = np.asarray(bundle["norm_std"], dtype=np.float32)    # (F,)
    T = int(cfg_dict.get("T", 32))
    stride = int(cfg_dict.get("stride", 1))
    n_anchors = int(cfg_dict.get("n_anchors", N_ANCHORS))
    n_channels = int(cfg_dict.get("n_channels", N_CHANNELS))
    d_model = int(cfg_dict.get("d_model", 64))
    n_asta_blocks = int(cfg_dict.get("n_asta_blocks", 2))
    train_scene = bundle.get("train_scene", "?")
    val_scene = bundle.get("val_scene", "?")

    print(f"[ckpt]  {ckpt_path}")
    print(f"[ckpt]  train={train_scene}  val={val_scene}  "
          f"T={T}  stride={stride}  M={n_anchors}  F={n_channels}  "
          f"d={d_model}  blocks={n_asta_blocks}")
    print(f"[ckpt]  norm: mean={norm_mean.tolist()}  std={norm_std.tolist()}")

    # ---- Rebuild model and load weights ----
    cfg = Config(n_anchors=n_anchors, n_channels=n_channels,
                 d_model=d_model, n_asta_blocks=n_asta_blocks)
    model = ASTAOfficeRack(cfg).to(device)
    state = bundle.get("model_state", bundle)
    if any(k.startswith("encoder.") or k.startswith("head.")
           for k in state.keys()):
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[load]  missing={len(missing)}  unexpected={len(unexpected)}")
    else:
        for prefix in ("model.", "module."):
            prefixed = {prefix + k: v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(prefixed, strict=False)
            if not missing and not unexpected:
                break
    model.eval()

    # ---- Load train + val CSVs, rebuild windows ----
    X_tr_raw, Y_tr_raw = load_office_rack_csv(train_csv)
    X_va_raw, Y_va_raw = load_office_rack_csv(val_csv)
    print(f"[data]  train ({train_scene}): X={X_tr_raw.shape}  Y={Y_tr_raw.shape}")
    print(f"[data]  val   ({val_scene}):   X={X_va_raw.shape}  Y={Y_va_raw.shape}")
    wx_tr, wy_tr = make_windows(X_tr_raw, Y_tr_raw, T, stride)
    wx_va, wy_va = make_windows(X_va_raw, Y_va_raw, T, stride)
    print(f"[data]  train windows: {wx_tr.shape} → {wy_tr.shape}")
    print(f"[data]  val   windows: {wx_va.shape} → {wy_va.shape}")
    n_tr, n_va = len(wx_tr), len(wx_va)

    # ---- Apply the saved per-channel norm ----
    def norm(x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for c in range(N_CHANNELS):
            ch = out[..., c]
            m = ch > 0   # ignore 0-fill cells
            ch[m] = (ch[m] - norm_mean[c]) / norm_std[c]
            out[..., c] = ch
        return out

    wx_tr_t = torch.from_numpy(norm(wx_tr)).to(device)   # (W, T, M, F)
    wy_tr_t = torch.from_numpy(wy_tr).to(device)
    wx_va_t = torch.from_numpy(norm(wx_va)).to(device)
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
    err_tr = np.linalg.norm(pred_tr - wy_tr, axis=-1)   # (W, T)
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

    # ==================================================================== #
    # Figures
    # ==================================================================== #

    # ---- Figure 1: val scatter (dots only) ----
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.scatter(wy_va[..., 0].flatten(), wy_va[..., 1].flatten(),
               c='green', s=8, alpha=0.5, label='GT')
    ax.scatter(pred_va[..., 0].flatten(), pred_va[..., 1].flatten(),
               c='red', s=8, alpha=0.5, label='Pred')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'ASTA 23-anchor ToA+AoA — VAL ({val_scene})\n'
                 f'val MAE = {overall_va:.3f} m, '
                 f'n_val_windows = {n_va}, T = {T}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    val_scatter = out_dir / "val_gt_vs_pred.png"
    fig.savefig(val_scatter, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {val_scatter}")

    # ---- Figure 2: train scatter (dots only) ----
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.scatter(wy_tr[..., 0].flatten(), wy_tr[..., 1].flatten(),
               c='green', s=8, alpha=0.5, label='GT')
    ax.scatter(pred_tr[..., 0].flatten(), pred_tr[..., 1].flatten(),
               c='red', s=8, alpha=0.5, label='Pred')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'ASTA 23-anchor ToA+AoA — TRAIN ({train_scene})\n'
                 f'train MAE = {overall_tr:.3f} m, '
                 f'n_train_windows = {n_tr}, T = {T}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
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
    ax.set_title(f"Per-step error distribution "
                 f"(val — {val_scene}, n={n_va})")
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
    ax.set_title(f"Per-window MAE (val, n={n_va}, scene={val_scene})")
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    bar_path = out_dir / "val_per_window_mae.png"
    fig.savefig(bar_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {bar_path}")

    # ---- Figure 5: SIDE-BY-SIDE train vs val (1x2 panel) ----
    #    This is the headline comparison figure: same plot style on
    #    both panels, GT (green) + Pred (red) dots only, equal aspect
    #    ratio. Title rows report MAE for each split.
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    for ax, gt, pred, mae, label, n in [
        (axes[0], wy_tr, pred_tr, overall_tr,
         f"TRAIN ({train_scene})", n_tr),
        (axes[1], wy_va, pred_va, overall_va,
         f"VAL ({val_scene})", n_va),
    ]:
        ax.scatter(gt[..., 0].flatten(), gt[..., 1].flatten(),
                   c='green', s=8, alpha=0.5, label='GT')
        ax.scatter(pred[..., 0].flatten(), pred[..., 1].flatten(),
                   c='red', s=8, alpha=0.5, label='Pred')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{label} — MAE = {mae:.3f} m  (n={n} windows)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    fig.suptitle(
        f"ASTA 23-anchor ToA+AoA — train {train_scene} → val {val_scene}\n"
        f"train MAE = {overall_tr:.3f} m  |  val MAE = {overall_va:.3f} m  |  "
        f"gap = {overall_va - overall_tr:+.3f} m  |  T = {T}, stride = {stride}",
        fontsize=12, y=1.02,
    )
    cmp_path = out_dir / "train_val_comparison.png"
    fig.savefig(cmp_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {cmp_path}")

    # ---- Figure 6: per-window MAE overlay (train vs val) ----
    #    Useful for spotting overfit windows: x = window index, two
    #    bars per index (train in blue, val in orange), or if the
    #    val/train window counts differ, a 2-panel layout.
    if n_tr == n_va:
        fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
        x = np.arange(n_tr)
        ax.bar(x - 0.2, per_win_tr, width=0.4, color='tab:blue', alpha=0.7,
               label=f'train ({train_scene}) mean={per_win_tr.mean():.3f} m')
        ax.bar(x + 0.2, per_win_va, width=0.4, color='tab:orange', alpha=0.7,
               label=f'val ({val_scene}) mean={per_win_va.mean():.3f} m')
        ax.set_xlabel("window index")
        ax.set_ylabel("per-window MAE (m)")
        ax.set_title(f"Per-window MAE — train vs val "
                     f"(train MAE={overall_tr:.3f}, val MAE={overall_va:.3f})")
        ax.legend(loc='upper right')
        ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True,
                                 sharex=False)
        axes[0].bar(np.arange(n_tr), per_win_tr, color='tab:blue', alpha=0.7)
        axes[0].axhline(per_win_tr.mean(), color='black', linestyle='--',
                        label=f"train mean = {per_win_tr.mean():.3f} m")
        axes[0].set_ylabel("per-window MAE (m)")
        axes[0].set_title(f"TRAIN ({train_scene}) — MAE = {overall_tr:.3f} m, "
                          f"n={n_tr}")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, axis='y', linestyle=':', alpha=0.4)

        axes[1].bar(np.arange(n_va), per_win_va, color='tab:orange', alpha=0.7)
        axes[1].axhline(per_win_va.mean(), color='black', linestyle='--',
                        label=f"val mean = {per_win_va.mean():.3f} m")
        axes[1].set_xlabel("window index (chronological)")
        axes[1].set_ylabel("per-window MAE (m)")
        axes[1].set_title(f"VAL ({val_scene}) — MAE = {overall_va:.3f} m, "
                          f"n={n_va}")
        axes[1].legend(loc='upper right')
        axes[1].grid(True, axis='y', linestyle=':', alpha=0.4)

    fig.suptitle(f"Per-window MAE (T={T}, stride={stride})",
                 fontsize=12, y=1.02)
    pw_path = out_dir / "per_window_train_vs_val.png"
    fig.savefig(pw_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {pw_path}")

    # ---- Figure 7: training curves (from history.csv if present) ----
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
        ax.set_title(f"Training curves "
                     f"(ASTA 23-anchor ToA+AoA, train={train_scene}, "
                     f"val={val_scene})")
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
        "train_scene": str(train_scene),
        "val_scene": str(val_scene),
        "train_mae_m": float(overall_tr),
        "val_mae_m": float(overall_va),
        "val_mae_p50_m": float(np.median(err_va)),
        "val_mae_p95_m": float(np.percentile(err_va, 95)),
        "val_mae_max_m": float(err_va.max()),
        "train_val_gap_m": float(overall_va - overall_tr),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="path to encoder.pt saved by "
                        "train_asta_office_rack_track.py")
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--train_scene", choices=SCENES, default="big",
                   help="must match the --train_scene used during training")
    p.add_argument("--val_scene", choices=SCENES, default="right",
                   help="must match the --val_scene used during training")
    p.add_argument("--out", required=True,
                   help="output dir for the regenerated figures + CSVs")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    train_csv = (args.data_root / f"office_rack_track_{args.train_scene}"
                 / "sionna_uwb_track_office_rack.csv")
    val_csv = (args.data_root / f"office_rack_track_{args.val_scene}"
               / "sionna_uwb_track_office_rack.csv")
    out_dir = Path(args.out)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    if not val_csv.exists():
        raise FileNotFoundError(val_csv)

    print("=" * 70)
    print("ASTA 23-anchor ToA+AoA eval (no retrain — loads existing ckpt)")
    print(f"  ckpt:        {ckpt_path}")
    print(f"  train scene: {args.train_scene}  ({train_csv.name})")
    print(f"  val scene:   {args.val_scene}  ({val_csv.name})")
    print(f"  out:         {out_dir}")
    print("=" * 70)
    print()

    summary = run_eval(ckpt_path=ckpt_path,
                       train_csv=train_csv, val_csv=val_csv,
                       out_dir=out_dir, device=args.device)
    print()
    print("summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
