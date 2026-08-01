"""Option C (F=2): quantitative metrics per channel.

Same structure as the F=1 metrics script but reports per-channel:
  - RMSE, MAE, bias
  - per-anchor RMSE
  - per-|true-mean-deviation| bin RMSE
  - shrink slope
  - corr(pred_dev, true_dev)

Units are restored to original (m for ToA, deg for AoA) before
computing errors.

Usage:
    python3.11 -m scripts.viz_tier0_metrics_f2 --ckpt <ckpt.pt>
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer, random_mask
from scripts.train_tier0_sionna_f2 import SionnaGridWindowF2Dataset, CHANNEL_NAMES


def shrink_slope(pred, true, center):
    dev_pred = pred - center
    dev_true = true - center
    if dev_true.std() < 1e-9:
        return float("nan")
    slope, _, _, _, _ = sp_stats.linregress(dev_true, dev_pred)
    return float(slope)


def evaluate(model, full_ds, cfg, norm_mean, norm_std, n_windows, seed=0):
    """Run the model on the val set and return per-channel arrays.

    Returns dict of arrays, each (n_masked,) per channel.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    if n_windows is not None:
        idx_list = list(rng.choice(len(full_ds), size=min(n_windows, len(full_ds)), replace=False))
    else:
        idx_list = list(range(len(full_ds)))

    per_ch = {
        "pred":  [list() for _ in range(2)],
        "true":  [list() for _ in range(2)],
        "anchor": [list() for _ in range(2)],
        "center": [list() for _ in range(2)],
    }
    for idx in idx_list:
        X, _, _, _, _ = full_ds[int(idx)]
        Xb = X.unsqueeze(0)
        torch.manual_seed(seed + int(idx))
        with torch.no_grad():
            # Pass the un-masked X. The model does its own internal
            # masking (via SSLPretrainer.forward -> random_mask), and the
            # skip-connection passthrough needs the original values to
            # be correct at visible cells. Double-masking here would
            # zero out the passthrough at cells the model re-marks as
            # visible, producing garbage predictions.
            x_pred, model_mask, _x_in = model(Xb)
        # use the model's internal mask as the eval mask, since the
        # model's head was trained to predict at *its* mask positions
        mask_n = model_mask.squeeze(0).numpy()  # (T, M)
        # denormalize (pred is already in normalized space from the model)
        Xd = X.squeeze(-1).numpy() * norm_std + norm_mean  # (T, M, 2)
        x_pred_d = x_pred.squeeze(0).squeeze(-1).numpy() * norm_std + norm_mean
        # per-(grid, anchor, channel) center: mean over trials of the
        # 100-trial grid
        gi, s = full_ds.windows[int(idx)]
        feat, _, _ = full_ds._per_grid[gi]  # (n_trials, M, F=2)
        center = feat.mean(axis=0)          # (M, F=2)
        if mask_n.sum() == 0:
            continue
        ti, ai = np.where(mask_n)           # (time_idx, anchor_idx)
        for f in range(2):
            per_ch["pred"][f].append(x_pred_d[mask_n, f])
            per_ch["true"][f].append(Xd[mask_n, f])
            per_ch["anchor"][f].append(ai)
            per_ch["center"][f].append(center[ai, f])
    out = {}
    for k in per_ch:
        out[k] = [np.concatenate(per_ch[k][f]) if per_ch[k][f] else np.array([]) for f in range(2)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_windows", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=False)
    cfg = Config(**{k: v for k, v in ckpt["cfg"].items() if k in Config.__dataclass_fields__})
    csv = args.csv or ckpt["csv"]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.n_channels != 2:
        raise ValueError(f"this script requires n_channels=2, got {cfg.n_channels}")

    norm_mean = ckpt["norm_mean"]
    norm_std = ckpt["norm_std"]
    print(f"norm stats: ToA  mean={norm_mean[0]:.3f} std={norm_std[0]:.3f}  "
          f"AoA  mean={norm_mean[1]:.3f} std={norm_std[1]:.3f}")

    full_ds = SionnaGridWindowF2Dataset(csv, seq_len=cfg.seq_len, stride=cfg.seq_len)
    n_grid = len(full_ds.grid_ids)
    n_train_grid = max(1, int(0.8 * n_grid))
    train_grid_ids = set(full_ds.grid_ids[:n_train_grid])
    full_ds.compute_norm_stats(train_grid_ids)
    full_ds.norm_mean = norm_mean
    full_ds.norm_std = norm_std
    val_grid_ids = set(full_ds.grid_ids[n_train_grid:])
    val_indices = [i for i, (gi, _) in enumerate(full_ds.windows)
                    if full_ds.grid_ids[gi] in val_grid_ids]
    print(f"val windows: {len(val_indices)}")

    model = SSLPretrainer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_windows = args.n_windows if args.n_windows is not None else len(val_indices)
    print(f"evaluating on {n_windows} windows...")
    res = evaluate(model, full_ds, cfg, norm_mean, norm_std, n_windows=n_windows, seed=args.seed)

    # visible-cell noise std per channel
    all_resid = [list() for _ in range(2)]
    for gi in range(len(full_ds._per_grid)):
        item = full_ds._per_grid[gi]
        if item is None:
            continue
        feat, _, _ = item
        center_ga = feat.mean(axis=0, keepdims=True)  # (1, M, F=2)
        for f in range(2):
            all_resid[f].append((feat[..., f] - center_ga[..., f]).ravel())
    visible_noise_std = [float(np.std(np.concatenate(r))) if r else float("nan") for r in all_resid]

    units = ["m", "deg"]
    print()
    print("=" * 78)
    print("  Tier 0 SSL (F=2) — quantitative refill-quality metrics (val)")
    print("=" * 78)
    print(f"  windows evaluated:    {n_windows}")
    summary = []
    for f in range(2):
        pred = res["pred"][f]
        true = res["true"][f]
        anchor = res["anchor"][f]
        center = res["center"][f]
        err = pred - true
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        bias = float(err.mean())
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        slope = shrink_slope(pred, true, center)
        dev_pred = pred - center
        dev_true = true - center
        corr = float(np.corrcoef(dev_pred, dev_true)[0, 1]) if dev_true.std() > 0 else float("nan")
        per_anchor_rmse = [float(np.sqrt(np.mean(err[anchor == m] ** 2)))
                            for m in range(cfg.n_anchors)]
        per_anchor_mae = [float(np.mean(np.abs(err[anchor == m])))
                            for m in range(cfg.n_anchors)]
        per_anchor_bias = [float(err[anchor == m].mean())
                            for m in range(cfg.n_anchors)]
        print()
        print(f"  channel {f}: {CHANNEL_NAMES[f]}")
        print(f"    total masked cells:      {len(pred)}")
        print(f"    overall RMSE:            {rmse:.4f} {units[f]}")
        print(f"    overall MAE:             {mae:.4f} {units[f]}")
        print(f"    overall bias:            {bias:+.4f} {units[f]}")
        print(f"    R^2 vs. constant-mean:   {r2:.4f}")
        print(f"    shrink slope:            {slope:.4f}")
        print(f"    corr(pred_dev, true_dev):{corr:.4f}")
        print(f"    visible-cell noise std:  {visible_noise_std[f]:.4f} {units[f]}")
        print(f"    per-anchor breakdown:")
        print(f"      {'anchor':<8} {'RMSE':>10} {'MAE':>10} {'bias':>12}")
        for m in range(cfg.n_anchors):
            print(f"      A{m+1:<7} {per_anchor_rmse[m]:>10.4f} {per_anchor_mae[m]:>10.4f} {per_anchor_bias[m]:>+12.4f}")
        summary.append({
            "channel": f, "channel_name": CHANNEL_NAMES[f],
            "units": units[f],
            "n_masked": len(pred),
            "rmse": rmse, "mae": mae, "bias": bias, "r2": r2,
            "shrink_slope": slope, "corr": corr,
            "visible_noise_std": visible_noise_std[f],
            "per_anchor_rmse": per_anchor_rmse,
            "per_anchor_mae": per_anchor_mae,
            "per_anchor_bias": per_anchor_bias,
        })

    # CSV
    csv_out = out_dir / "viz_C_metrics_f2.csv"
    with open(csv_out, "w") as f:
        f.write("channel,channel_name,units,n_masked_cells,rmse,mae,bias,r2,shrink_slope,corr_pred_dev_true_dev,visible_noise_std\n")
        for s in summary:
            f.write(f"{s['channel']},{s['channel_name']},{s['units']},"
                    f"{s['n_masked']},{s['rmse']:.6f},{s['mae']:.6f},"
                    f"{s['bias']:.6f},{s['r2']:.6f},{s['shrink_slope']:.6f},"
                    f"{s['corr']:.6f},{s['visible_noise_std']:.6f}\n")
        for s in summary:
            for m in range(cfg.n_anchors):
                f.write(f"rmse_A{m+1}_{s['channel_name'].split()[0]},{s['per_anchor_rmse'][m]:.6f}\n")
                f.write(f"mae_A{m+1}_{s['channel_name'].split()[0]},{s['per_anchor_mae'][m]:.6f}\n")
                f.write(f"bias_A{m+1}_{s['channel_name'].split()[0]},{s['per_anchor_bias'][m]:.6f}\n")
    print(f"\nsaved {csv_out}")

    # 4-panel figure: 2x2 with channel 0 / channel 1
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for f in range(2):
        pred = res["pred"][f]
        true = res["true"][f]
        anchor = res["anchor"][f]
        center = res["center"][f]
        err = pred - true
        # (a) pred vs true scatter
        ax = axes[f, 0]
        ax.scatter(true, pred, s=3, alpha=0.3)
        lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y=x")
        ax.set_xlabel(f"true {CHANNEL_NAMES[f]}")
        ax.set_ylabel(f"predicted {CHANNEL_NAMES[f]}")
        ax.set_title(f"(a) ch{f}: pred vs true  (RMSE = {np.sqrt(np.mean(err**2)):.3f} {units[f]})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # (b) per-anchor RMSE
        ax = axes[f, 1]
        per_a = [np.sqrt(np.mean(err[anchor == m] ** 2)) for m in range(cfg.n_anchors)]
        ax.bar(range(cfg.n_anchors), per_a, color="C0", alpha=0.8)
        ax.axhline(visible_noise_std[f], color="green", linestyle="--", alpha=0.6,
                    label=f"visible noise std ({visible_noise_std[f]:.3f} {units[f]})")
        ax.set_xticks(range(cfg.n_anchors))
        ax.set_xticklabels([f"A{m+1}" for m in range(cfg.n_anchors)])
        ax.set_ylabel(f"RMSE ({units[f]})")
        ax.set_title(f"(b) ch{f}: per-anchor RMSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Tier 0 SSL (F=2) — quantitative refill diagnostics (val)",
                  fontsize=11)
    fig.tight_layout()
    out_png = out_dir / "viz_C_metrics_f2.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
