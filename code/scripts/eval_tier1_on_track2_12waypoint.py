"""Validate the track2 12-waypoint tracking hypothesis using the
pretrained tier1 encoder.

Setup
-----
- Source CSV: sionna_uwb_track_empty_open_A1A4swapped.csv (the
  same scene the ASTA model was trained on, in the same
  A1A4swapped column order).
- Ground truth: seg_index 0..11 = the 12 waypoints.
- Inputs: (d1, d2, d3, d4) ToA + (aoa1, aoa2, aoa3, aoa4) AoA,
  shape (N, 4, 2). Filter to seg_type=='dwell' (100 samples
  per waypoint, 1200 total).
- Tier1 encoder checkpoint: checkpoints/tier1_mixed.pt/tier1_sionna.pt
  (4 anchors, 2 channels, d_model=64, 2 blocks, no head).
- A new 12-class linear head is trained on top of the frozen
  encoder using the (ToA, AoA) -> seg_index mapping.
- Test mode: time-ordered 80/20 split per waypoint (80 train
  + 20 test per class, deterministic).

The point of this experiment is to see how well a *trained*
encoder (from tier1 meta-learning) supports the 12-waypoint
classification task, with as little new training as possible.

Comparisons
-----------
We report:
  1. Encoder-frozen + linear head (the main result)
  2. Random / uniform baseline (8.3%)
  3. LSE-nearest-waypoint (a strong classical baseline)

Outputs
-------
- code/figures/tier1_12waypoint_eval.png
  2x2 panel: GT path w/ correct vs wrong, confusion matrix,
  per-class accuracy bar, model vs LSE vs random.
- code/checkpoints/_tmp_tier1_12waypoint/eval_report.csv
- code/checkpoints/_tmp_tier1_12waypoint/head.pt
  (the trained 12-class head so we can re-run eval without
  re-training)

Usage:
    python3.11 -m scripts.eval_tier1_on_track2_12waypoint
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


# ---- Data ------------------------------------------------------------------

def load_track_dwell(csv_path: Path) -> dict:
    """Load the track2 CSV; keep only the dwell rows. Returns
    arrays of (T, M, F) inputs and (T, 2) GT positions + the
    waypoint class index per row (0..11, based on first-occurrence)."""
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

    # Waypoint class index by first-occurrence of (wp_x, wp_y)
    seen = OrderedDict()
    for r in rows:
        key = (r["tag_x"], r["tag_y"])
        if key not in seen:
            seen[key] = len(seen)
    n_classes = len(seen)
    print(f"  found {n_classes} unique waypoint classes")

    X = np.asarray([[r["d"], r["aoa"]] for r in rows], dtype=np.float32)  # (N, 4, 2)
    X = X.transpose(0, 2, 1)                                               # (N, M=4, F=2)
    Y_pos = np.asarray([[r["tag_x"], r["tag_y"]] for r in rows], dtype=np.float32)
    Y_cls = np.asarray([seen[(r["tag_x"], r["tag_y"])] for r in rows], dtype=np.int64)
    waypoints = np.asarray(list(seen.keys()), dtype=np.float32)
    return {
        "X": X,                          # (N, 4, 2)
        "Y_pos": Y_pos,                 # (N, 2)
        "Y_cls": Y_cls,                 # (N,)
        "waypoints": waypoints,         # (n_classes, 2)
        "n_classes": n_classes,
    }


# ---- Model -----------------------------------------------------------------

class Tier1Classifier(nn.Module):
    """Frozen tier1 encoder + a fresh 12-class linear head."""

    def __init__(self, cfg: Config, n_classes: int):
        super().__init__()
        self.encoder = ASTAEncoder(cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model * cfg.n_anchors, 128),
            nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, M, F). Returns (B, n_classes) logits."""
        h = self.encoder(x)                          # (B, T, M, d)
        h_pool = h.mean(dim=1)                       # (B, M, d)
        h_flat = h_pool.reshape(h.size(0), -1)       # (B, M*d)
        return self.head(h_flat)


# ---- LSE baseline ----------------------------------------------------------

def lse_2d(d: np.ndarray, anchor_list) -> np.ndarray:
    """4-anchor closed-form (handles NaN by skipping). Returns (x, y)."""
    valid = ~np.isnan(d)
    n_valid = int(valid.sum())
    if n_valid < 3:
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


# ---- Eval ------------------------------------------------------------------

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
                           "checkpoints/_tmp_tier1_12waypoint")
    p.add_argument("--out_fig",
                   default="/Users/justin/ultraclear-paper/code/figures/"
                           "tier1_12waypoint_eval.png")
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

    # ---- Load data ----
    d = load_track_dwell(Path(args.csv))
    X = d["X"]                                 # (N, 4, 2)  [M=4 anchors, F=2 channels]
    Y_cls = d["Y_cls"]                         # (N,)
    waypoints = d["waypoints"]                 # (12, 2)
    N = len(X)
    print(f"loaded {N} dwell rows, n_classes={d['n_classes']}")

    # ---- Time-ordered per-class split ----
    # For each class, take the first --test_per_class rows for test,
    # rest for train. This is time-ordered within each class (rows
    # are already time-ordered in the file).
    train_idx, test_idx = [], []
    for c in range(d["n_classes"]):
        idxs = np.where(Y_cls == c)[0]
        n_test = min(args.test_per_class, len(idxs) // 2)
        test_idx.extend(idxs[:n_test].tolist())
        train_idx.extend(idxs[n_test:].tolist())
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    print(f"split: {len(train_idx)} train, {len(test_idx)} test "
          f"({len(test_idx) // d['n_classes']} per class)")

    # ---- Load pretrained tier1 encoder + add a fresh head ----
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in sd["cfg"].items()
                      if k in Config.__dataclass_fields__})
    print(f"loaded tier1 ckpt: n_anchors={cfg.n_anchors} "
          f"n_channels={cfg.n_channels} d_model={cfg.d_model} "
          f"n_asta_blocks={cfg.n_asta_blocks}")

    norm_mean = sd["norm_mean"]                  # (2,)  [ToA, AoA]
    norm_std = sd["norm_std"]                    # (2,)
    print(f"normalization: mean={norm_mean}, std={norm_std}")

    # Build model and load the encoder weights (skip any head weights)
    model = Tier1Classifier(cfg, n_classes=d["n_classes"])
    enc_state = {k: v for k, v in sd["model_state"].items()
                  if k.startswith("encoder.")}
    missing, unexpected = model.load_state_dict(enc_state, strict=False)
    print(f"loaded encoder weights; missing={len(missing)} "
          f"(expected: head.*), unexpected={len(unexpected)}")

    # Freeze encoder
    for p_ in model.encoder.parameters():
        p_.requires_grad = False

    # ---- Normalize inputs and add the T=1 window dim ----
    # X is (N, 4, 2). Normalize per-channel using norm_mean/norm_std.
    Xn = (X - norm_mean) / norm_std
    # Add T=1 axis -> (N, T=1, M=4, F=2)
    Xn = Xn[:, np.newaxis, :, :]
    Xtr = torch.from_numpy(Xn[train_idx]).float()
    Xte = torch.from_numpy(Xn[test_idx]).float()
    Ytr = torch.from_numpy(Y_cls[train_idx])
    Yte = torch.from_numpy(Y_cls[test_idx])

    # ---- Train the head ----
    head_only = model.head
    opt = torch.optim.Adam(head_only.parameters(), lr=args.lr)
    print(f"\n=== training head ({args.epochs} epochs, lr={args.lr}) ===")
    for ep in range(args.epochs):
        head_only.train()
        logits = model(Xtr)
        loss = TF.cross_entropy(logits, Ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, args.epochs // 5) == 0 or ep == 0:
            with torch.no_grad():
                tr_acc = (logits.argmax(-1) == Ytr).float().mean().item()
            print(f"  ep {ep+1:3d}/{args.epochs}  loss={loss.item():.4f}  "
                  f"train_acc={tr_acc:.3f}")

    # ---- Evaluate ----
    model.eval()
    with torch.no_grad():
        te_logits = model(Xte)
        te_pred = te_logits.argmax(-1).cpu().numpy()
    te_acc = float((te_pred == Yte.numpy()).mean())
    print(f"\nTIER1 encoder + linear head  test acc = {te_acc:.3f} "
          f"({(te_pred == Yte.numpy()).sum()}/{len(Yte)})")

    # ---- LSE baseline (uses last / only row) ----
    # Anchors in training order (matches the ASTA trained model)
    anchors_2d = [
        np.array([-0.52, -0.90]),
        np.array([-0.54,  0.35]),
        np.array([ 0.52, -0.90]),
        np.array([ 0.54,  0.35]),
    ]
    lse_pred = []
    for i in test_idx:
        d_meas = d["X"][i, :, 0]                 # (4,) ToA only
        xy = lse_2d(d_meas, anchors_2d)
        # Nearest of 12 waypoints
        dists = np.linalg.norm(waypoints - xy, axis=1)
        lse_pred.append(int(np.argmin(dists)))
    lse_pred = np.asarray(lse_pred)
    lse_acc = float((lse_pred == Yte.numpy()).mean())
    print(f"LSE3 nearest-waypoint             test acc = {lse_acc:.3f}")

    # Random baseline
    rng = np.random.default_rng(args.seed)
    rand_pred = rng.integers(0, d["n_classes"], size=len(Yte))
    rand_acc = float((rand_pred == Yte.numpy()).mean())
    print(f"random guess (uniform 1/{d['n_classes']})  test acc = {rand_acc:.3f}")

    # ---- Save head ckpt + report ----
    torch.save({
        "head_state": head_only.state_dict(),
        "cfg": cfg.__dict__,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "n_classes": d["n_classes"],
        "waypoints": waypoints,
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "test_acc": te_acc,
        "lse_acc": lse_acc,
        "rand_acc": rand_acc,
    }, out_dir / "head.pt")
    print(f"\nsaved head → {out_dir / 'head.pt'}")

    with open(out_dir / "eval_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "test_accuracy", "n_test"])
        w.writerow(["tier1_encoder+linear_head", f"{te_acc:.4f}", len(Yte)])
        w.writerow(["lse3_nearest_waypoint", f"{lse_acc:.4f}", len(Yte)])
        w.writerow(["random_uniform", f"{rand_acc:.4f}", len(Yte)])
    print(f"report → {out_dir / 'eval_report.csv'}")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)

    # Top-left: GT path colored by class, with pred markers
    ax = axes[0, 0]
    # Use ALL rows (not just test) for the GT path so the geometry
    # is clear
    cmap = plt.get_cmap("tab20")
    for c in range(d["n_classes"]):
        idxs = np.where(Y_cls == c)[0]
        ax.scatter(d["Y_pos"][idxs, 1], d["Y_pos"][idxs, 0],
                    c=[cmap(c)], s=40, alpha=0.6, zorder=2)
    # Predicted test points (correct vs wrong)
    te_X = d["X"][test_idx]
    te_Y_pos = d["Y_pos"][test_idx]
    correct = (te_pred == Yte.numpy())
    ax.scatter(te_Y_pos[correct, 1], te_Y_pos[correct, 0],
                c="lime", s=80, edgecolor="black", lw=0.7, marker="o",
                label=f"correct ({correct.sum()})", zorder=4)
    ax.scatter(te_Y_pos[~correct, 1], te_Y_pos[~correct, 0],
                c="red", s=80, edgecolor="black", lw=0.7, marker="X",
                label=f"wrong ({(~correct).sum()})", zorder=4)
    # Waypoint stars
    for c in range(d["n_classes"]):
        ax.scatter(waypoints[c, 1], waypoints[c, 0],
                    s=240, marker="*", c="gold",
                    edgecolor="black", linewidth=1.0, zorder=3)
        ax.annotate(f"wp{c}", (waypoints[c, 1], waypoints[c, 0]),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=8, fontweight="bold")
    # Anchors
    for i, (ax_x, ax_y) in enumerate(anchors_2d):
        ax.scatter(ax_x, ax_y, s=200, marker="^", c="red",
                    edgecolor="black", linewidth=1.0, zorder=5)
        ax.annotate(f"A{i+1}", (ax_x, ax_y), textcoords="offset points",
                     xytext=(8, 6), fontsize=8, fontweight="bold")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("GT dwell path + test predictions\n"
                  "(green=correct, red=wrong)")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=9)

    # Top-right: confusion matrix
    ax = axes[0, 1]
    cm = np.zeros((d["n_classes"], d["n_classes"]), dtype=int)
    for t, p in zip(Yte.numpy(), te_pred):
        cm[t, p] += 1
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    ax.set_xticks(range(d["n_classes"]))
    ax.set_yticks(range(d["n_classes"]))
    ax.set_xticklabels([f"wp{c}" for c in range(d["n_classes"])], fontsize=8,
                        rotation=45, ha="right")
    ax.set_yticklabels([f"wp{c}" for c in range(d["n_classes"])], fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (test acc = {te_acc:.2f})")
    for i in range(d["n_classes"]):
        for j in range(d["n_classes"]):
            v = cm[i, j]
            color = "white" if v > cm.max() * 0.6 else "black"
            ax.text(j, i, str(v), ha="center", va="center",
                     fontsize=8, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="count")

    # Bottom-left: per-class accuracy bars
    ax = axes[1, 0]
    per_class_acc = []
    for c in range(d["n_classes"]):
        m = (Yte.numpy() == c)
        if m.sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(float((te_pred[m] == c).mean()))
    ax.bar(range(d["n_classes"]), per_class_acc, color="steelblue")
    ax.set_xticks(range(d["n_classes"]))
    ax.set_xticklabels([f"wp{c}" for c in range(d["n_classes"])],
                        fontsize=8, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(te_acc, color="red", ls=":", lw=1,
                label=f"mean = {te_acc:.2f}")
    ax.set_ylabel("per-class accuracy")
    ax.set_title("per-class accuracy (tier1 encoder + linear head)")
    ax.grid(True, alpha=0.3, linestyle=":", axis="y")
    ax.legend(loc="lower right", fontsize=9)

    # Bottom-right: method comparison
    ax = axes[1, 1]
    methods = ["random\nuniform", "LSE3\nnearest wp", "tier1 encoder\n+ linear head"]
    accs = [rand_acc, lse_acc, te_acc]
    colors = ["lightgray", "tab:orange", "tab:blue"]
    bars = ax.bar(methods, accs, color=colors, edgecolor="black", lw=0.7)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, a + 0.02, f"{a:.2f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0 / d["n_classes"], color="gray", ls=":", lw=0.8,
                label=f"chance = 1/{d['n_classes']} = {1/d['n_classes']:.3f}")
    ax.set_ylabel("test accuracy (12-class)")
    ax.set_title("method comparison (240 test samples, 20 per class)")
    ax.grid(True, alpha=0.3, linestyle=":", axis="y")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"tier1 encoder on track2 12-waypoint tracking emulation\n"
        f"{N} dwell samples from {Path(args.csv).name}",
        fontsize=13,
    )
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure → {fig_path}")


if __name__ == "__main__":
    main()
