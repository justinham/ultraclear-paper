"""Visualize ASTA predictions on uwbx4_aligned_2hz.csv.

Two-panel figure for assessing the trained 3-anchor / 1-channel
ASTA model on the real-world ARKit-labeled data.

  Left  : the 2D horizontal track in car frame.
            ARKit GT (black), ASTA prediction (blue), LSE3
            baseline (red, dashed). Markers numbered by window
            index so you can see the per-window error.

  Right : per-window Euclidean error. ASTA in blue, LSE3 in
            red, both as a bar chart. Lets you see WHERE the
            model is off without squinting at the trajectory.

Reuses the loaders from `scripts.train_asta_uwbx4_3anchor` so
the windows and LSE baseline are byte-identical to what the
trainer produced.

Usage:
    python3.11 -m scripts.viz_uwbx4_asta_pred
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the trainer's data + LSE helpers
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train_asta_uwbx4_3anchor import (
    load_uwbx4_aligned, make_windows, normalize_toa,
    ASTALocator, lse3,
)
from ultraclear.config import Config


def predict_all(model, X_win: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X_win)).cpu().numpy()
    return out


def lse_all(X_win: np.ndarray, anchor_pos: np.ndarray,
             toa_cap: float) -> np.ndarray:
    preds = []
    for w in range(len(X_win)):
        last_d = X_win[w, -1, :, 0] * toa_cap
        try:
            p = lse3(last_d, anchor_pos)
            preds.append([p[0], p[1]])
        except Exception:
            preds.append([np.nan, np.nan])
    return np.asarray(preds, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",
                   default="/Users/justin/ultraclear-paper/code/"
                           "checkpoints/_tmp_asta_uwbx4_3anchor/"
                           "asta_uwbx4_3anchor.pt")
    p.add_argument("--csv",
                   default="/Users/justin/ultraclear-paper/code/"
                           "data/real_data_arkit/uwbx4_aligned_2hz.csv")
    p.add_argument("--out",
                   default="/Users/justin/ultraclear-paper/code/figures/"
                           "uwbx4_asta_pred_eval.png")
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt["cfg"].items()
                     if k in Config.__dataclass_fields__})
    anchor_pos = np.asarray(ckpt["anchor_pos_world"], dtype=np.float64)
    toa_cap = float(ckpt["toa_cap"])
    T = int(ckpt.get("T", args.T))
    stride = int(ckpt.get("stride", args.stride))
    print(f"ckpt: {Path(args.ckpt).name}")
    print(f"  T={T}  stride={stride}  toa_cap={toa_cap}")
    print(f"  cfg: n_anchors={cfg.n_anchors}  n_channels={cfg.n_channels}  "
          f"d_model={cfg.d_model}  n_asta_blocks={cfg.n_asta_blocks}")
    print(f"  anchor pos (world):\n  {anchor_pos}")

    # Re-load data and build the same windows the trainer used
    d = load_uwbx4_aligned(Path(args.csv))
    toa_norm = normalize_toa(d["toa"], toa_cap=toa_cap)
    X_win, Y_win, idx = make_windows(toa_norm, d["y"], T=T, stride=stride)
    X_win = X_win[..., np.newaxis]                                  # (W, T, 3, 1)
    print(f"\n{len(X_win)} windows, X={X_win.shape}, Y={Y_win.shape}")

    # Load model + predict
    model = ASTALocator(cfg)
    model.load_state_dict(ckpt["model_state"])
    asta_preds = predict_all(model, X_win)
    lse_preds = lse_all(X_win, anchor_pos, toa_cap)

    # Per-window errors
    asta_err = np.linalg.norm(asta_preds - Y_win, axis=1)
    lse_err = np.linalg.norm(lse_preds - Y_win, axis=1)
    valid = ~np.isnan(lse_err)
    asta_mae = float(asta_err.mean())
    lse_mae = float(np.nanmean(lse_err[valid]))
    print(f"\nASTA: per-window norm MAE = {asta_mae:.3f} m  "
          f"(min {asta_err.min():.3f}, max {asta_err.max():.3f})")
    print(f"LSE3: per-window norm MAE = {lse_mae:.3f} m  "
          f"(min {np.nanmin(lse_err):.3f}, max {np.nanmax(lse_err):.3f})")

    # ---- Two-panel figure ----
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(13, 5.5),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
        constrained_layout=True,
    )
    W = np.arange(len(Y_win))

    # Left: track with numbered markers. Plot order matters so
    # ARKit GT ends up ON TOP (the others can sit underneath
    # since they're farther from the truth).
    ax_l.plot(asta_preds[:, 1], asta_preds[:, 0], "b^-", lw=1.4, ms=8,
               alpha=0.6, label=f"ASTA (MAE {asta_mae:.2f} m)", zorder=2)
    ax_l.plot(lse_preds[valid, 1], lse_preds[valid, 0], "rs--", lw=0.9,
               ms=5, alpha=0.5,
               label=f"LSE3 (MAE {lse_mae:.2f} m)", zorder=1)
    # GT last so it sits on top
    ax_l.plot(Y_win[:, 1], Y_win[:, 0], "ko-", lw=2.0, ms=10,
               label="ARKit GT", zorder=5)
    for w in W:
        ax_l.annotate(str(w), (Y_win[w, 1], Y_win[w, 0]),
                       textcoords="offset points", xytext=(8, 4),
                       fontsize=9, fontweight="bold", color="black",
                       zorder=6)
    ax_l.set_xlabel("py_car (m, lateral)")
    ax_l.set_ylabel("px_car (m, forward)")
    ax_l.set_title(
        f"uwbx4_aligned_2hz — ASTA vs LSE3 (car frame, {len(W)} windows)"
    )
    ax_l.legend(loc="best", fontsize=10)
    ax_l.grid(True, alpha=0.3, linestyle=":")
    ax_l.set_aspect("equal", adjustable="datalim")

    # Right: per-window error bars
    width = 0.4
    bars_a = ax_r.bar(W - width / 2, asta_err, width, color="tab:blue",
                        label="ASTA", alpha=0.85)
    bars_l = ax_r.bar(W + width / 2, lse_err, width, color="tab:red",
                        label="LSE3", alpha=0.75)
    # Add per-bar value labels so the small ASTA errors are readable
    # (otherwise they vanish at the resolution of the plot).
    for bar, v in zip(bars_a, asta_err):
        ax_r.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.15, f"{v:.2f}",
                   ha="center", va="bottom", fontsize=8, color="tab:blue")
    for bar, v in zip(bars_l, lse_err):
        ax_r.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.15, f"{v:.2f}",
                   ha="center", va="bottom", fontsize=8, color="tab:red")
    ax_r.axhline(asta_mae, color="tab:blue", ls=":", lw=1,
                  label=f"ASTA mean ({asta_mae:.2f} m)")
    ax_r.axhline(lse_mae, color="tab:red", ls=":", lw=1,
                  label=f"LSE3 mean ({lse_mae:.2f} m)")
    ax_r.set_xticks(W)
    ax_r.set_xticklabels([str(w) for w in W])
    ax_r.set_xlabel("window index")
    ax_r.set_ylabel("Euclidean error vs ARKit GT (m)")
    ax_r.set_title("per-window error")
    ax_r.set_ylim(0, max(float(np.nanmax(lse_err)), float(asta_err.max())) * 1.15)
    ax_r.legend(loc="upper left", fontsize=9)
    ax_r.grid(True, alpha=0.3, linestyle=":", axis="y")

    fig.suptitle(
        f"ASTA (3 anchors, 1 channel ToA-only) on uwbx4_aligned_2hz",
        fontsize=12,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
