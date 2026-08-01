"""Train ASTA on BOTH uwbx4 CSVs combined, rotating window for tracking.

This is the tracking-framing version of `train_asta_uwbx4_3anchor.py`:

  - Concatenates the two ARKit-labeled CSVs (uwbx4_aligned_2hz.csv
    with 45 valid rows + uwbx4_2_real_test_aligned_2hz.csv with
    22 valid rows) end-to-end so the rotating window can slide
    over ~67 rows instead of 16.
  - Default T=5, stride=1 → 63 windows (vs 7 with T=10 on
    just the 1st CSV).
  - Label = last (current) row of the window, as Justin asked.
  - 80/20 time-ordered split, in-sample MAE on the same windows
    is also reportable via --val_frac 0.

Writes (separate dir from the previous run, never touches
canonical checkpoints):

  code/checkpoints/_tmp_asta_uwbx4_3anchor_both/
    asta_uwbx4_3anchor.pt
    mae_table.csv
    pred_track.png
    train_history.csv
    train_history.png

Usage:
    python3.11 -m scripts.train_asta_uwbx4_3anchor_both
    python3.11 -m scripts.train_asta_uwbx4_3anchor_both --val_frac 0
    python3.11 -m scripts.train_asta_uwbx4_3anchor_both --T 10
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.config import Config
from ultraclear.asta import ASTAEncoder


# ---- Data ------------------------------------------------------------------

def load_uwbx4_csv(csv_path: Path) -> dict:
    """Read one aligned CSV; return per-row arrays of (d1, d2, d3) ToA
    and (px_car, py_car) labels. Drops rows with any NaN in those
    5 fields.

    Anchor positions are constant across rows; we just grab them
    from the first non-NaN row.
    """
    d_cols = ["d1", "d2", "d3"]
    y_cols = ["px_car", "py_car"]
    times, toa, y = [], [], []
    anchor_pos = None
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        header = rdr.fieldnames
        for r in rdr:
            try:
                ds = [float(r[c]) for c in d_cols]
                ys = [float(r[c]) for c in y_cols]
            except (ValueError, KeyError):
                continue
            if any(np.isnan(v) for v in ds + ys):
                continue
            if anchor_pos is None:
                anchor_pos = []
                for i in [1, 2, 3]:
                    anchor_pos.append([float(r[f"anc{i}_wx"]),
                                        float(r[f"anc{i}_wy"]),
                                        float(r[f"anc{i}_wz"])])
            times.append(r["t_grid"])
            toa.append(ds)
            y.append(ys)
    return {
        "times": times, "toa": np.asarray(toa, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "anchor_pos": np.asarray(anchor_pos, dtype=np.float64),
    }


def make_windows(toa: np.ndarray, y: np.ndarray, T: int, stride: int):
    """Sliding window over ToA; label = y at the LAST row of the
    window. Returns X (W, T, M), Y (W, 2), and the starting row
    index for each window (so we can plot against absolute time)."""
    N = len(toa)
    starts = list(range(0, N - T + 1, stride))
    X = np.stack([toa[i:i + T] for i in starts])
    Y = np.stack([y[i + T - 1] for i in starts])
    return X, Y, np.asarray(starts, dtype=int)


def normalize_toa(toa: np.ndarray, toa_cap: float = 10.0) -> np.ndarray:
    return np.clip(toa / toa_cap, 0.0, 1.0).astype(np.float32)


def time_ordered_split(n: int, val_frac: float = 0.2):
    if val_frac <= 0.0:
        return list(range(n)), []
    n_val = max(1, int(round(n * val_frac)))
    n_val = min(n_val, max(1, n // 2))
    n_train = max(1, n - n_val)
    n_val = n - n_train
    return list(range(0, n_train)), list(range(n_train, n))


# ---- Model -----------------------------------------------------------------

class ASTALocator(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = ASTAEncoder(cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model * cfg.n_anchors, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, toa: torch.Tensor) -> torch.Tensor:
        h = self.encoder(toa)
        h_pool = h.mean(dim=1)
        return self.head(h_pool.reshape(h.size(0), -1))


# ---- LSE baseline ----------------------------------------------------------

def lse3(d: np.ndarray, A: np.ndarray) -> np.ndarray:
    p0, p1, p2 = A[0], A[1], A[2]
    A_mat = 2 * (A[1:, :2] - p0[:2])
    b = np.array([
        d[0] ** 2 - d[1] ** 2 + (A[1] ** 2).sum() - (p0 ** 2).sum(),
        d[0] ** 2 - d[2] ** 2 + (A[2] ** 2).sum() - (p0 ** 2).sum(),
    ], dtype=np.float64)
    sol, *_ = np.linalg.lstsq(A_mat, b, rcond=None)
    z = d[0] - np.linalg.norm(A[0, :2] - sol)
    return np.array([sol[0], sol[1], z])


# ---- Training --------------------------------------------------------------

def train(model, Xtr, Ytr, Xva, Yva, n_epochs, lr):
    import torch.nn.functional as TF
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_t = torch.from_numpy(Xtr)
    Ytr_t = torch.from_numpy(Ytr)
    Xva_t = torch.from_numpy(Xva) if len(Xva) > 0 else None
    Yva_t = torch.from_numpy(Yva) if len(Yva) > 0 else None
    history = {"epoch": [], "train_mae": [], "val_mae": []}
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        pred = model(Xtr_t)
        loss = TF.mse_loss(pred, Ytr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_mae = float((pred - Ytr_t).abs().mean().item())
        val_mae = float("nan")
        if Xva_t is not None:
            model.eval()
            with torch.no_grad():
                pred_v = model(Xva_t)
                val_mae = float((pred_v - Yva_t).abs().mean().item())
        history["epoch"].append(ep + 1)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        if (ep + 1) % max(1, n_epochs // 10) == 0 or ep == 0:
            v = f"{val_mae:.3f}" if not np.isnan(val_mae) else "  n/a"
            print(f"  ep {ep+1:3d}/{n_epochs}  "
                  f"train_mae={train_mae:.3f}  val_mae={v}  "
                  f"({time.time()-t0:.1f}s)")
    return history


# ---- Eval + plot -----------------------------------------------------------

def evaluate_and_plot(model, X, Y, anchor_pos, toa_cap, history,
                       out_dir: Path, T: int, stride: int,
                       starts: np.ndarray, csv_label: str):
    import torch.nn.functional as TF
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X)).cpu().numpy()
    lse_preds = []
    for w in range(len(X)):
        last_d = X[w, -1, :, 0] * toa_cap
        try:
            lse_preds.append(lse3(last_d, anchor_pos)[:2])
        except Exception:
            lse_preds.append([np.nan, np.nan])
    lse_preds = np.asarray(lse_preds, dtype=np.float32)

    def mae_per_axis(P, Y):
        d = P - Y
        return (float(np.nanmean(np.abs(d[:, 0]))),
                float(np.nanmean(np.abs(d[:, 1]))),
                float(np.nanmean(np.linalg.norm(d, axis=1))))
    ax_, ay_, an = mae_per_axis(preds, Y)
    lx, ly, ln = mae_per_axis(lse_preds, Y)
    print(f"\nMAE on all {len(X)} windows (car frame, m):")
    print(f"  ASTA : x={ax_:.3f}  y={ay_:.3f}  ||p-p*||={an:.3f}")
    print(f"  LSE3 : x={lx:.3f}  y={ly:.3f}  ||p-p*||={ln:.3f}")

    # Save MAE table
    mae_path = out_dir / "mae_table.csv"
    with open(mae_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mae_x_m", "mae_y_m", "mae_norm_m", "n_windows"])
        w.writerow(["ASTA", f"{ax_:.4f}", f"{ay_:.4f}", f"{an:.4f}", len(X)])
        w.writerow(["LSE3", f"{lx:.4f}", f"{ly:.4f}", f"{ln:.4f}", len(X)])

    # Per-window history
    with open(out_dir / "train_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mae", "val_mae"])
        for ep, t, v in zip(history["epoch"], history["train_mae"],
                            history["val_mae"]):
            w.writerow([ep, f"{t:.6f}", f"{v:.6f}"])

    # Per-window error
    asta_err = np.linalg.norm(preds - Y, axis=1)
    lse_err = np.linalg.norm(lse_preds - Y, axis=1)

    # ---- 3-panel figure: track, error curve, error histogram ----
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 9),
        gridspec_kw={"height_ratios": [1.4, 1.0]},
        constrained_layout=True,
    )
    ax_t, ax_e = axes
    W = np.arange(len(Y))
    t_secs = starts * 0.5   # 2 Hz, relative to start of the combined data

    # Top: track
    ax_t.plot(Y[:, 1], Y[:, 0], "ko-", lw=2.0, ms=8, label="ARKit GT",
               zorder=5)
    ax_t.plot(preds[:, 1], preds[:, 0], "b^-", lw=1.4, ms=7, alpha=0.7,
               label=f"ASTA (MAE {an:.2f} m)", zorder=2)
    valid = ~np.isnan(lse_err)
    ax_t.plot(lse_preds[valid, 1], lse_preds[valid, 0], "rs--", lw=0.9,
               ms=5, alpha=0.5,
               label=f"LSE3 (MAE {ln:.2f} m)", zorder=1)
    ax_t.set_xlabel("py_car (m, lateral)")
    ax_t.set_ylabel("px_car (m, forward)")
    ax_t.set_title(
        f"uwbx4 combined: ASTA vs LSE3 (car frame, rotating window)\n"
        f"{len(X)} windows, T={T}, stride={stride} — {csv_label}"
    )
    ax_t.legend(loc="best", fontsize=10)
    ax_t.grid(True, alpha=0.3, linestyle=":")
    ax_t.set_aspect("equal", adjustable="datalim")

    # Bottom: per-window error over time
    width = max(0.4, 0.8 * (t_secs[-1] - t_secs[0]) / max(1, len(t_secs)))
    # Simple bar plot on the absolute time axis
    ax_e.bar(t_secs - 0.15, asta_err, width=0.3, color="tab:blue",
              alpha=0.8, label="ASTA per-window err")
    ax_e.bar(t_secs + 0.15, lse_err, width=0.3, color="tab:red",
              alpha=0.7, label="LSE3 per-window err")
    ax_e.axhline(an, color="tab:blue", ls=":", lw=1,
                  label=f"ASTA mean ({an:.2f} m)")
    ax_e.axhline(ln, color="tab:red", ls=":", lw=1,
                  label=f"LSE3 mean ({ln:.2f} m)")
    ax_e.set_xlabel("time (s, relative to first window)")
    ax_e.set_ylabel("Euclidean error vs ARKit GT (m)")
    ax_e.set_title("per-window tracking error (rotating window, T={}, stride={})"
                    .format(T, stride))
    ax_e.legend(loc="upper left", fontsize=9)
    ax_e.grid(True, alpha=0.3, linestyle=":", axis="y")
    ax_e.set_ylim(0, max(float(np.nanmax(lse_err)), float(asta_err.max())) * 1.15)

    fig.suptitle(
        f"ASTA (3 anchors, 1 channel) tracking on uwbx4 combined — "
        f"{len(X)} windows",
        fontsize=12,
    )
    fig.savefig(out_dir / "pred_track.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  pred_track.png → {out_dir / 'pred_track.png'}")


# ---- Main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv1",
                   default="/Users/justin/ultraclear-paper/code/data/"
                           "real_data_arkit/uwbx4_aligned_2hz.csv")
    p.add_argument("--csv2",
                   default="/Users/justin/ultraclear-paper/code/data/"
                           "real_data_arkit/uwbx4_2_real_test_aligned_2hz.csv")
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--toa_cap", type=float, default=10.0)
    p.add_argument("--out_dir",
                   default="/Users/justin/ultraclear-paper/code/"
                           "checkpoints/_tmp_asta_uwbx4_3anchor_both")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"T={args.T}  stride={args.stride}  epochs={args.epochs}  "
          f"val_frac={args.val_frac}  toa_cap={args.toa_cap}")
    print(f"out_dir: {out_dir}\n")

    # ---- Load both CSVs ----
    parts = []
    for cp in [args.csv1, args.csv2]:
        cp = Path(cp)
        if not cp.exists():
            print(f"  (skipping missing: {cp})")
            continue
        d = load_uwbx4_csv(cp)
        print(f"loaded {len(d['y'])} valid rows from {cp.name}")
        parts.append(d)
    if not parts:
        raise SystemExit("no input CSVs found")
    toa_all = np.concatenate([p["toa"] for p in parts], axis=0)
    y_all = np.concatenate([p["y"] for p in parts], axis=0)
    ANCHOR_POS = parts[0]["anchor_pos"]
    print(f"\ncombined: {len(y_all)} rows, "
          f"x range [{y_all[:, 0].min():.2f}, {y_all[:, 0].max():.2f}], "
          f"y range [{y_all[:, 1].min():.2f}, {y_all[:, 1].max():.2f}] m")
    print(f"anchor pos (world):\n  {ANCHOR_POS}")

    # ---- Build rotating windows ----
    toa_norm = normalize_toa(toa_all, toa_cap=args.toa_cap)
    X_win, Y_win, starts = make_windows(
        toa_norm, y_all, T=args.T, stride=args.stride)
    X_win = X_win[..., np.newaxis]                   # (W, T, 3, 1)
    print(f"built {len(X_win)} rotating windows of T={args.T} "
          f"(stride={args.stride})")
    print(f"  X_win: {X_win.shape}, Y_win: {Y_win.shape}")
    if len(X_win) < 2:
        raise RuntimeError("only {} windows — try smaller --T".format(len(X_win)))

    # ---- Split ----
    train_idx, val_idx = time_ordered_split(len(X_win), val_frac=args.val_frac)
    Xtr, Ytr = X_win[train_idx], Y_win[train_idx]
    Xva, Yva = X_win[val_idx], Y_win[val_idx]
    print(f"split: {len(train_idx)} train, {len(val_idx)} val")

    # ---- Model ----
    cfg = Config()
    cfg.n_anchors = 3; cfg.n_channels = 1
    cfg.n_asta_blocks = 2; cfg.d_model = 64
    cfg.n_attention_heads = 4; cfg.tcn_kernel = 3
    cfg.device = "cpu"
    print(f"\nmodel: anchors=3 channels=1 d_model=64 blocks=2 heads=4")
    model = ASTALocator(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    # ---- Train ----
    print(f"\n=== train ({args.epochs} epochs) ===")
    history = train(model, Xtr, Ytr, Xva, Yva,
                     n_epochs=args.epochs, lr=args.lr)

    # ---- Save ckpt ----
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
        "history": history,
        "anchor_pos_world": ANCHOR_POS,
        "toa_cap": args.toa_cap,
        "T": args.T, "stride": args.stride,
        "train_idx": train_idx, "val_idx": val_idx,
        "X_win": X_win, "Y_win": Y_win, "starts": starts,
    }, out_dir / "asta_uwbx4_3anchor.pt")
    print(f"\nsaved → {out_dir / 'asta_uwbx4_3anchor.pt'}")

    # ---- Eval + plot ----
    print(f"\n=== evaluate (all {len(X_win)} windows) ===")
    csv_label = "uwbx4 + uwbx4_2 (combined, 2 sessions)"
    evaluate_and_plot(model, X_win, Y_win, ANCHOR_POS, args.toa_cap,
                       history, out_dir,
                       T=args.T, stride=args.stride,
                       starts=starts, csv_label=csv_label)


if __name__ == "__main__":
    main()
