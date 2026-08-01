"""Multi-scene trajectory visualization for the ASTA 23-anchor
office-rack sionna track model.

Loads the saved encoder.pt and runs it on:
  - the training scene (default big)
  - the canonical val scene (default right)
  - one or more additional test scenes (default middle)

For each scene, computes per-step predictions, then renders:

  - all3_scenes_gt_vs_pred.png   (1x3 panel: GT vs predicted for
                                  every scene, dots + thin connecting
                                  line so the trajectory shape is
                                  visible)
  - per_scene_mae.png            (bar chart of per-step MAE per scene)
  - per_scene_mae_table.csv      (per-scene summary metrics)
  - <scene>_gt_vs_pred.png       (one detailed per-scene figure, with
                                  the connecting line, window boundaries
                                  shown as vertical separators, MAE in
                                  title) — for each scene
  - <scene>_err_vs_t.png         (per-step error time series for each
                                  scene, so you can see WHERE on the
                                  track the model fails)

Diffs vs scripts/eval_asta_office_rack_track.py:
  * Supports an arbitrary number of test scenes (--test_scenes
    comma-separated) in addition to the train scene and canonical
    val scene.
  * Each scene gets a dedicated figure with the trajectory LINE
    (the per-step eval just shows dots; this one connects them so
    the shape of the track is visible).
  * Per-scene error time series (where in the trajectory the model
    is failing).
  * Headline 1x3 figure with all scenes side-by-side.

Run (after training big -> right, evaluating on middle too):
  python3.11 -m scripts.eval_office_rack_all_scenes \\
      --ckpt  checkpoints/_tmp_asta_office_rack_big_to_right/encoder.pt \\
      --out   checkpoints/_tmp_asta_office_rack_big_to_right/_all_scenes
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
from matplotlib.lines import Line2D

from ultraclear.asta import ASTAEncoder
from ultraclear.config import Config


# --------------------------------------------------------------------------- #
# Data loading + model — keep in sync with the train script
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
    keep = (wx[..., 0].sum(axis=(1, 2)) > 0)
    return wx[keep], wy[keep]


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


def load_model(ckpt_path: Path, device: str = "cpu"):
    """Load the saved bundle, rebuild the model, return
    (model, bundle, cfg_dict, norm_mean, norm_std)."""
    bundle = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = bundle.get("config", {})
    norm_mean = np.asarray(bundle["norm_mean"], dtype=np.float32)
    norm_std = np.asarray(bundle["norm_std"], dtype=np.float32)
    n_anchors = int(cfg_dict.get("n_anchors", N_ANCHORS))
    n_channels = int(cfg_dict.get("n_channels", N_CHANNELS))
    d_model = int(cfg_dict.get("d_model", 64))
    n_asta_blocks = int(cfg_dict.get("n_asta_blocks", 2))
    cfg = Config(n_anchors=n_anchors, n_channels=n_channels,
                 d_model=d_model, n_asta_blocks=n_asta_blocks)
    model = ASTAOfficeRack(cfg).to(device)
    state = bundle.get("model_state", bundle)
    if any(k.startswith("encoder.") or k.startswith("head.")
           for k in state.keys()):
        model.load_state_dict(state, strict=False)
    else:
        for prefix in ("model.", "module."):
            prefixed = {prefix + k: v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(prefixed, strict=False)
            if not missing and not unexpected:
                break
    model.eval()
    return model, bundle, cfg_dict, norm_mean, norm_std


def predict_scene(model, X_raw, Y_raw, T, stride, norm_mean, norm_std,
                 device="cpu", batch_size: int = 16):
    """Build windows, normalize, predict, return (gt, pred, err, per_window)."""
    wx, wy = make_windows(X_raw, Y_raw, T, stride)
    # Per-channel norm
    wx_n = wx.copy()
    for c in range(N_CHANNELS):
        ch = wx_n[..., c]
        m = ch > 0
        ch[m] = (ch[m] - norm_mean[c]) / norm_std[c]
        wx_n[..., c] = ch
    wx_t = torch.from_numpy(wx_n).to(device)
    pred_list = []
    with torch.no_grad():
        for i in range(0, len(wx_t), batch_size):
            pred_list.append(model(wx_t[i:i + batch_size]).cpu().numpy())
    pred = np.concatenate(pred_list, axis=0)        # (W, T, 2)
    err = np.linalg.norm(pred - wy, axis=-1)         # (W, T)
    per_win = err.mean(axis=1)
    overall = float(err.mean())
    return wy, pred, err, per_win, overall


# --------------------------------------------------------------------------- #
# Plotting helpers
# --------------------------------------------------------------------------- #

def _plot_scene_panel(ax, gt, pred, err, scene_name, role, mae,
                      p50, p95, max_err, n_windows, T):
    """Draw a single scene panel: GT trajectory (green line) +
    predicted trajectory (red dashed line) + scatter dots so the
    line and points are both visible."""
    # Flatten across windows — order is chronological by window start
    gt_x = gt[..., 0].flatten()
    gt_y = gt[..., 1].flatten()
    pr_x = pred[..., 0].flatten()
    pr_y = pred[..., 1].flatten()

    ax.plot(gt_x, gt_y, '-', color='tab:green', alpha=0.45, lw=1.2,
            label='GT')
    ax.plot(pr_x, pr_y, '--', color='tab:red', alpha=0.55, lw=1.0,
            label='Pred')
    ax.scatter(gt_x, gt_y, c='tab:green', s=4, alpha=0.6, zorder=3)
    ax.scatter(pr_x, pr_y, c='tab:red', s=4, alpha=0.6, zorder=3)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{role} ({scene_name}) — MAE = {mae:.3f} m\n'
                 f'p50 = {p50:.3f}  p95 = {p95:.3f}  max = {max_err:.3f}  '
                 f'(n={n_windows} win, T={T})',
                 fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)


def _plot_err_time_series(ax, gt, err, scene_name, role, mae, n_windows, T):
    """Per-step Euclidean error vs chronological step index.
    Useful for spotting WHERE on the track the model is failing."""
    n_steps = err.shape[0] * err.shape[1]
    err_flat = err.flatten()
    x = np.arange(n_steps)
    ax.plot(x, err_flat, '-', color='tab:blue', alpha=0.5, lw=0.7)
    # Window boundaries as faint vertical lines
    for w in range(1, n_windows):
        ax.axvline(w * T, color='gray', alpha=0.15, lw=0.5)
    ax.axhline(mae, color='black', linestyle='--',
               label=f'mean = {mae:.3f} m')
    ax.set_xlabel('chronological step (across all windows)')
    ax.set_ylabel('per-step Euclidean error (m)')
    ax.set_title(f'{role} ({scene_name}) — error vs step  '
                 f'(n={n_windows} win × T={T})', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=':')


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run(*, ckpt_path: Path,
        train_scene: str, val_scene: str, test_scenes: list[str],
        data_root: Path,
        out_dir: Path,
        device: str = "cpu") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model + norm stats ----
    model, bundle, cfg_dict, norm_mean, norm_std = load_model(ckpt_path, device)
    T = int(cfg_dict.get("T", 32))
    stride = int(cfg_dict.get("stride", 1))
    train_scene_ckpt = bundle.get("train_scene", train_scene)
    val_scene_ckpt = bundle.get("val_scene", val_scene)
    print(f"[ckpt]  {ckpt_path}")
    print(f"[ckpt]  trained on '{train_scene_ckpt}', canonical val "
          f"'{val_scene_ckpt}', T={T} stride={stride}")
    print(f"[ckpt]  norm mean = {norm_mean.tolist()}, std = {norm_std.tolist()}")

    # ---- Define which scenes to evaluate ----
    scenes_to_eval: list[tuple[str, str]] = []  # (scene, role)
    scenes_to_eval.append((train_scene_ckpt, "TRAIN"))
    if val_scene_ckpt != train_scene_ckpt:
        scenes_to_eval.append((val_scene_ckpt, "VAL"))
    for s in test_scenes:
        if s in (train_scene_ckpt, val_scene_ckpt):
            continue   # already in the list
        scenes_to_eval.append((s, "TEST"))

    # ---- Run on each scene ----
    results: dict[str, dict] = {}
    for scene, role in scenes_to_eval:
        csv_path = data_root / f"office_rack_track_{scene}" / \
                   "sionna_uwb_track_office_rack.csv"
        if not csv_path.exists():
            print(f"[skip] {scene} (no CSV at {csv_path})")
            continue
        print(f"\n[eval] {role:5s} scene '{scene}'  ({csv_path.name})")
        X_raw, Y_raw = load_office_rack_csv(csv_path)
        print(f"        raw: X={X_raw.shape}  Y={Y_raw.shape}")
        gt, pred, err, per_win, mae = predict_scene(
            model, X_raw, Y_raw, T, stride, norm_mean, norm_std, device,
        )
        n_win = err.shape[0]
        p50 = float(np.median(err))
        p95 = float(np.percentile(err, 95))
        max_err = float(err.max())
        results[scene] = {
            "role": role,
            "csv": str(csv_path),
            "n_raw": int(X_raw.shape[0]),
            "n_windows": int(n_win),
            "T": int(T),
            "stride": int(stride),
            "gt": gt,
            "pred": pred,
            "err": err,
            "per_window_mae": per_win,
            "mae_m": mae,
            "p50_m": p50,
            "p95_m": p95,
            "max_m": max_err,
        }
        print(f"        MAE = {mae:.3f} m  p50 = {p50:.3f}  "
              f"p95 = {p95:.3f}  max = {max_err:.3f}  "
              f"(n_windows={n_win})")

    # ---- Per-scene detailed figures ----
    for scene, r in results.items():
        # Per-scene GT vs pred with line + dots
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        _plot_scene_panel(ax, r["gt"], r["pred"], r["err"],
                          scene, r["role"],
                          mae=r["mae_m"], p50=r["p50_m"],
                          p95=r["p95_m"], max_err=r["max_m"],
                          n_windows=r["n_windows"], T=r["T"])
        out_path = out_dir / f"{scene}_{r['role'].lower()}_gt_vs_pred.png"
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"[save]  {out_path}")

        # Per-scene error time series
        fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
        _plot_err_time_series(ax, r["gt"], r["err"],
                              scene, r["role"],
                              mae=r["mae_m"],
                              n_windows=r["n_windows"], T=r["T"])
        out_path = out_dir / f"{scene}_{r['role'].lower()}_err_vs_t.png"
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"[save]  {out_path}")

    # ---- 1xN all-scenes trajectory figure ----
    n_scenes = len(results)
    if n_scenes == 0:
        print("[warn] no scenes evaluated; nothing to plot")
        return results
    fig, axes = plt.subplots(1, n_scenes, figsize=(7 * n_scenes, 7),
                             constrained_layout=True)
    if n_scenes == 1:
        axes = [axes]
    for ax, (scene, r) in zip(axes, results.items()):
        _plot_scene_panel(ax, r["gt"], r["pred"], r["err"],
                          scene, r["role"],
                          mae=r["mae_m"], p50=r["p50_m"],
                          p95=r["p95_m"], max_err=r["max_m"],
                          n_windows=r["n_windows"], T=r["T"])
    # Suptitle: headline MAE per scene
    parts = [f"{r['role']} {s}: {r['mae_m']:.2f}m"
             for s, r in results.items()]
    fig.suptitle(
        f"ASTA 23-anchor ToA+AoA — GT vs Predicted trajectory (all scenes)\n"
        f"{'  |  '.join(parts)}",
        fontsize=13, y=1.04,
    )
    all_path = out_dir / "all_scenes_gt_vs_pred.png"
    fig.savefig(all_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {all_path}")

    # ---- Per-scene MAE bar chart ----
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    scenes_order = list(results.keys())
    roles = [results[s]["role"] for s in scenes_order]
    maes = [results[s]["mae_m"] for s in scenes_order]
    p50s = [results[s]["p50_m"] for s in scenes_order]
    p95s = [results[s]["p95_m"] for s in scenes_order]
    colors = ['tab:blue' if r == 'TRAIN' else
              'tab:orange' if r == 'VAL' else
              'tab:green' for r in roles]
    x = np.arange(len(scenes_order))
    ax.bar(x, maes, color=colors, alpha=0.7,
           label='per-step MAE (mean)')
    ax.scatter(x, p50s, color='black', marker='_', s=200, lw=2,
               label='p50', zorder=5)
    ax.scatter(x, p95s, color='red', marker='_', s=200, lw=2,
               label='p95', zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n({s})" for s, r in zip(scenes_order, roles)],
                       fontsize=10)
    ax.set_ylabel('per-step error (m)')
    ax.set_title(f"Per-scene error summary  |  T={T} stride={stride}")
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    bar_path = out_dir / "per_scene_mae.png"
    fig.savefig(bar_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[save]  {bar_path}")

    # ---- Per-scene summary table CSV ----
    table_csv = out_dir / "per_scene_mae_table.csv"
    with open(table_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "role", "n_raw", "n_windows", "T", "stride",
                    "mae_m", "p50_m", "p95_m", "max_m"])
        for s, r in results.items():
            w.writerow([s, r["role"], r["n_raw"], r["n_windows"],
                        r["T"], r["stride"],
                        f"{r['mae_m']:.4f}", f"{r['p50_m']:.4f}",
                        f"{r['p95_m']:.4f}", f"{r['max_m']:.4f}"])
    print(f"[save]  {table_csv}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--train_scene", default="big",
                   help="train scene (the one used during training). "
                        "Default 'big' matches the trainer's default.")
    p.add_argument("--val_scene", default="right",
                   help="canonical val scene (used during training). "
                        "Default 'right' matches the trainer's default.")
    p.add_argument("--test_scenes", default="middle",
                   help="comma-separated list of additional held-out "
                        "scenes to evaluate. Default 'middle'. Pass an "
                        "empty string '' to skip.")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    test_scenes: list[str] = []
    if args.test_scenes:
        test_scenes = [s.strip() for s in args.test_scenes.split(",") if s.strip()]

    print("=" * 70)
    print("ASTA 23-anchor ToA+AoA — multi-scene trajectory eval")
    print(f"  ckpt:        {ckpt_path}")
    print(f"  data root:   {args.data_root}")
    print(f"  train scene: {args.train_scene}")
    print(f"  val scene:   {args.val_scene}")
    print(f"  test scenes: {test_scenes or '(none)'}")
    print(f"  out:         {out_dir}")
    print("=" * 70)
    print()

    results = run(ckpt_path=ckpt_path,
                  train_scene=args.train_scene,
                  val_scene=args.val_scene,
                  test_scenes=test_scenes,
                  data_root=args.data_root,
                  out_dir=out_dir,
                  device=args.device)
    print()
    print("summary:")
    for s, r in results.items():
        print(f"  {r['role']:5s} {s:6s}  MAE={r['mae_m']:.3f}m  "
              f"p50={r['p50_m']:.3f}  p95={r['p95_m']:.3f}  "
              f"max={r['max_m']:.3f}  (n_raw={r['n_raw']}, "
              f"n_win={r['n_windows']})")


if __name__ == "__main__":
    main()
