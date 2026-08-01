"""Linear Least-Squares (LSE) trilateration for the 23-anchor
office-rack sionna track datasets.

For each row we have:
  - 23 ToA measurements (in meters) from anchors A1..A23
  - The 23 anchor positions (X, Y, Z) in meters
  - GT tag position (tag_x, tag_y, tag_z)

We solve for the tag position (x, y, z) using linearized
trilateration:

  d_i^2 = (x - x_i)^2 + (y - y_i)^2 + (z - z_i)^2

Subtracting the equation for an arbitrary reference anchor r
from every other anchor i, the squared terms cancel and we
get a linear system A * p = b in p = (x, y, z):

  2 (x_r - x_i) x + 2 (y_r - y_i) y + 2 (z_r - z_i) z
      = d_i^2 - d_r^2 - x_i^2 + x_r^2 - y_i^2 + y_r^2 - z_i^2 + z_r^2

We use anchor A1 as the reference (any of the 23 would do;
A1 is at the +X+Y corner, so it usually has decent geometry).
With 22 equations in 3 unknowns, the system is heavily
overdetermined; we solve it with np.linalg.lstsq.

Notes
-----
- The linearization is a standard trick and gives a closed-form
  point estimate. It is NOT the same as iterative nonlinear
  least-squares (Gauss-Newton) which would minimize
  sum_i (|p - a_i| - d_i)^2 directly — that would be a
  follow-up if you want a stronger baseline.
- For 2D localization we drop the z-row of A and just solve
  for (x, y). The tag in this dataset has z=1.0 (constant),
  so 2D is the natural target.
- We also fit a per-anchor bias term (calibration constant
  added to d_i) as an optional extension — left out for now
  to keep the baseline clean.

Run:
  python3.11 -m scripts.lse_office_rack --scene big
  python3.11 -m scripts.lse_office_rack --scene middle
  python3.11 -m scripts.lse_office_rack --scene right
  python3.11 -m scripts.lse_office_rack --scene all
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultraclear.uwb_transformer_correction import (
    load_office_rack_anchor_positions,
)

# Optional scipy import for nonlinear LSE
try:
    from scipy.optimize import least_squares
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

DEFAULT_DATA_ROOT = Path(
    "/Users/justin/ultraclear-paper/code/data/office_rack_track_sionna"
)
SCENES = ("big", "middle", "right")


# --------------------------------------------------------------------------- #
# Core LSE solver
# --------------------------------------------------------------------------- #

def lse_2d(anchor_xyz: np.ndarray,
           distances: np.ndarray,
           ref_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Linearized LSE trilateration, 2D (x, y), z is held at the
    median tag-z if known, else dropped from the system.

    Args:
        anchor_xyz: (N, 3) anchor positions in meters.
        distances:  (N,) ToA distances in meters. Must be > 0.
                    Assumed to be 3D Euclidean distances from tag
                    to anchor. If you have 2D horizontal distances
                    already, just pass them in — the projection
                    step will leave them untouched.
        ref_idx:    index of the reference anchor (whose equation
                    is subtracted from the others to make the
                    system linear).

    Returns:
        (x, y):     (2,) estimated tag position in meters.
        residuals:  (N-1,) per-equation residual after the linear
                    solve (b - A @ p). Useful for diagnosing
                    outliers / NLOS.
    """
    N = anchor_xyz.shape[0]
    if distances.shape[0] != N:
        raise ValueError(f"distances {distances.shape} != anchor_xyz {anchor_xyz.shape}")
    # We allow 0 here because the 3D->2D projection can clip
    # NLOS-corrupted distances to exactly 0 if d_3d < |delta_z|.
    # A 0 distance just means "this anchor's equation is useless
    # for this row"; np.linalg.lstsq will still produce a solution.
    if (distances < 0).any():
        raise ValueError(f"negative distances at indices "
                         f"{np.where(distances < 0)[0].tolist()}")

    # 3D distances go in; we project to 2D before solving.
    # Caller is expected to pass a tag_z via _z_tag_known; we read
    # it off anchor_xyz[:, 2] and the global default.
    # (The actual projection happens in run_scene which knows z_tag.)
    # Here we just build the linear system.
    other = [i for i in range(N) if i != ref_idx]
    ai = anchor_xyz[other]      # (N-1, 3)
    ar = anchor_xyz[ref_idx]    # (3,)
    di = distances[other]       # (N-1,)
    dr = distances[ref_idx]     # ()

    # 2D: drop the z column from A.
    # 2 (x_r - x_i) x + 2 (y_r - y_i) y
    #   = d_i^2 - d_r^2 - (x_i^2 - x_r^2) - (y_i^2 - y_r^2)
    A = 2.0 * (ar[None, :2] - ai[:, :2])             # (N-1, 2)
    di2 = di ** 2
    dr2 = dr ** 2
    b = (di2 - dr2
         - (ai[:, 0] ** 2 - ar[0] ** 2)
         - (ai[:, 1] ** 2 - ar[1] ** 2))              # (N-1,)
    # Solve A p = b
    p, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    return p, b - A @ p


def lse_nonlinear_2d(anchor_xyz: np.ndarray,
                     distances: np.ndarray,
                     init: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Nonlinear LSE (Gauss-Newton / Levenberg-Marquardt) on the
    L2 distance error in 2D (x, y) only.

    Minimizes `sum_i (|p - a_i|_2d - d_i)^2` using scipy's
    `least_squares` with method='lm'. Much more robust to large
    NLOS bias than the linearized version: the L2 loss is
    linear in the bias, not quadratic.

    Args:
        anchor_xyz: (N, 3) anchor positions in meters. Only
                    the first 2 columns (x, y) are used.
        distances:  (N,) projected 2D horizontal distances in
                    meters.
        init:       (2,) optional initial guess. Defaults to
                    the linearized LSE solution.

    Returns:
        (x, y):     (2,) estimated tag position in meters.
        residuals:  (N,) per-anchor residual `d_i - |p - a_i|_2d`.
    """
    if not _HAS_SCIPY:
        raise ImportError(
            "scipy is required for nonlinear LSE; install with "
            "`pip install scipy` or use the linearized LSE (no --nonlinear)."
        )

    def resid_fn(p):
        diff = anchor_xyz[:, :2] - p[None, :]
        return distances - np.sqrt(np.sum(diff ** 2, axis=1))

    if init is None:
        init, _ = lse_2d(anchor_xyz, distances, ref_idx=0)
    res = least_squares(resid_fn, init, method='lm')
    return res.x, distances - np.sqrt(np.sum(
        (anchor_xyz[:, :2] - res.x[None, :]) ** 2, axis=1
    ))


def lse_nonlinear_3d(anchor_xyz: np.ndarray,
                     distances: np.ndarray,
                     init: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Nonlinear LSE in 3D (x, y, z), letting z be a free variable.

    Minimizes `sum_i (|p - a_i|_3d - d_i)^2` with the Levenberg-
    Marquardt method. Initial guess defaults to (0, 0, 1.0).
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for nonlinear LSE")

    def resid_fn(p):
        diff = anchor_xyz - p[None, :]
        return distances - np.sqrt(np.sum(diff ** 2, axis=1))

    if init is None:
        init = np.array([0.0, 0.0, 1.0])
    res = least_squares(resid_fn, init, method='lm')
    return res.x, distances - np.sqrt(np.sum(
        (anchor_xyz - res.x[None, :]) ** 2, axis=1
    ))


def project_to_2d(distances_3d: np.ndarray,
                 anchor_z: np.ndarray,
                 tag_z: float) -> np.ndarray:
    """Project 3D Euclidean distances to 2D horizontal distances
    using a known tag elevation.

        d_2d = sqrt(d_3d^2 - (z_anchor - z_tag)^2)

    Args:
        distances_3d: (N,) 3D Euclidean distances in meters.
        anchor_z:      (N,) per-anchor z in meters.
        tag_z:         known tag elevation in meters.

    Returns:
        distances_2d: (N,) horizontal distances. Negative
                       under-the-radical inputs (e.g. NLOS
                       distance shorter than z offset) are
                       clipped to 0 to keep np.sqrt happy.
    """
    delta_z_sq = (anchor_z - tag_z) ** 2
    radicand = distances_3d ** 2 - delta_z_sq
    # NLOS can make d_3d < |delta_z| for some anchors; clip.
    radicand = np.clip(radicand, 0.0, None)
    return np.sqrt(radicand)


def lse_3d(anchor_xyz: np.ndarray,
           distances: np.ndarray,
           ref_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """3D version. Same linearization, but keeps z in the system.
    Returns (x, y, z)."""
    other = [i for i in range(anchor_xyz.shape[0]) if i != ref_idx]
    ai = anchor_xyz[other]
    ar = anchor_xyz[ref_idx]
    di = distances[other]
    dr = distances[ref_idx]

    A = 2.0 * (ar[None, :] - ai)                      # (N-1, 3)
    di2 = di ** 2
    dr2 = dr ** 2
    b = (di2 - dr2
         - np.sum(ai ** 2 - ar[None, :] ** 2, axis=1))  # (N-1,)
    p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return p, b - A @ p


# --------------------------------------------------------------------------- #
# Per-scene driver
# --------------------------------------------------------------------------- #

def run_scene(scene: str,
              data_root: Path,
              anchor_xyz: np.ndarray,
              mode: str = "2d",
              top_k_anchors: int | None = None,
              tag_z: float = 1.0,
              project_3d_to_2d: bool = True,
              nonlinear: bool = False) -> dict:
    csv_path = data_root / f"office_rack_track_{scene}" / \
               "sionna_uwb_track_office_rack.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)

    # Load all 23 ToA columns in A1..A23 order
    toa_cols = [f"A{i}_measured_dist_m" for i in range(1, 24)]
    distances = df[toa_cols].to_numpy()   # (N, 23) — already in meters
    tag_x = df["tag_x"].to_numpy()
    tag_y = df["tag_y"].to_numpy()
    # NB: do NOT name this local 'tag_z' — it would shadow the
    # function parameter. We need the parameter (a float) for
    # the 3D->2D ToA projection, and the column (an array) for
    # the GT z (only used in 3D mode).
    if "tag_z" in df.columns:
        gt_tag_z = df["tag_z"].to_numpy()
    else:
        gt_tag_z = np.full(len(df), 1.0)

    # Optional anchor selection by abs-mean ToA noise (using GT as oracle,
    # so this is a CEILING for LSE — it shows what LSE could do if it
    # could perfectly identify the cleanest anchors). If --top_k is set,
    # we keep only the K anchors with the smallest per-row |measured - gt|
    # error. Default = all 23.
    n_total = distances.shape[1]
    if top_k_anchors is not None and top_k_anchors < n_total:
        gt_cols = [f"A{i}_gt_dist_m" for i in range(1, 24)]
        if not all(c in df.columns for c in gt_cols):
            raise ValueError(
                f"--top_k_anchors requires per-anchor A{{i}}_gt_dist_m "
                f"columns, which are missing from {csv_path}"
            )
        gt_d = df[gt_cols].to_numpy()
        noise = np.abs(distances - gt_d).mean(axis=0)   # (23,)
        keep_idx = np.argsort(noise)[:top_k_anchors]    # indices into A1..A23
        keep_anchors = [int(i) + 1 for i in keep_idx]
        print(f"        top-{top_k_anchors} anchor selection by "
              f"abs-mean noise (oracle — uses GT):  "
              f"A{keep_anchors}")
        distances = distances[:, keep_idx]
        anchor_xyz = anchor_xyz[keep_idx]

    n = len(df)
    # We apply the 3D->2D ToA projection inside the per-row loop
    # below so we use the (possibly top-K-sliced) anchor_xyz[:, 2].
    if mode == "2d":
        # Solve (x, y) only. The 3D->2D ToA projection
        # `d_2d = sqrt(d_3d^2 - (z_a - z_t)^2)` is applied
        # per-row so the right anchor-z slice is used after
        # any top_k_anchors selection. The 9.26m unfixable
        # floor error we saw in the un-projected 2D LSE came
        # from ignoring this 1.85m vertical offset (tag at
        # z=1.0, ceiling anchors at z=2.85).
        pred = np.zeros((n, 2), dtype=np.float64)
        n_resid = anchor_xyz.shape[0] - 1
        resid_all = np.zeros((n, n_resid), dtype=np.float64)
        anchor_z = anchor_xyz[:, 2]
        for i in range(n):
            d_i = distances[i]
            if project_3d_to_2d:
                d_i = project_to_2d(d_i, anchor_z, tag_z)
            if nonlinear and _HAS_SCIPY:
                # Warm-start with the linearized solution
                p_lin, _ = lse_2d(anchor_xyz, d_i, ref_idx=0)
                p, r = lse_nonlinear_2d(anchor_xyz, d_i, init=p_lin)
                # Nonlinear residuals are per-anchor (N), not
                # N-1 (which is what the linearized LSE returns).
                # Just truncate to the first N-1 to fit the
                # shared (N, N-1) array shape.
                r = r[:n_resid]
            else:
                p, r = lse_2d(anchor_xyz, d_i, ref_idx=0)
            pred[i, 0] = p[0]
            pred[i, 1] = p[1]
            resid_all[i] = r
        gt = np.stack([tag_x, tag_y], axis=1)         # (N, 2)
    else:
        pred = np.zeros((n, 3), dtype=np.float64)
        n_resid = anchor_xyz.shape[0] - 1
        resid_all = np.zeros((n, n_resid), dtype=np.float64)
        for i in range(n):
            if nonlinear and _HAS_SCIPY:
                p, r = lse_nonlinear_3d(anchor_xyz, distances[i])
                r = r[:n_resid]
            else:
                p, r = lse_3d(anchor_xyz, distances[i], ref_idx=0)
            pred[i] = p
            resid_all[i] = r
        gt = np.stack([tag_x, tag_y, gt_tag_z], axis=1)

    # Per-axis MAE
    err = pred - gt                                       # (N, 2 or 3)
    abs_err = np.abs(err)
    mae_per_axis = abs_err.mean(axis=0)                  # (2 or 3,)
    err_norm = np.linalg.norm(err, axis=-1)               # (N,)
    overall_mae = float(err_norm.mean())
    p50 = float(np.median(err_norm))
    p95 = float(np.percentile(err_norm, 95))
    max_err = float(err_norm.max())
    rmse = float(np.sqrt((err ** 2).mean()))

    return {
        "scene": scene,
        "n": n,
        "pred": pred,
        "gt": gt,
        "err": err,
        "abs_err_per_axis": mae_per_axis,
        "err_norm": err_norm,
        "residuals": resid_all,
        "overall_mae_m": overall_mae,
        "rmse_m": rmse,
        "p50_m": p50,
        "p95_m": p95,
        "max_m": max_err,
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_scene_traj(result: dict,
                    anchor_xyz: np.ndarray,
                    out_path: Path,
                    mode: str):
    pred = result["pred"]
    gt = result["gt"]
    scene = result["scene"]

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)

    # Anchor geometry (top-down: x horizontal, y vertical)
    ceiling = anchor_xyz[anchor_xyz[:, 2] > 1.0]
    floor = anchor_xyz[anchor_xyz[:, 2] <= 1.0]
    ax.scatter(ceiling[:, 0], ceiling[:, 1], marker='^', s=110,
               c='tab:blue', edgecolor='black', linewidth=0.6,
               zorder=4, label='ceiling anchors (z=2.85)')
    ax.scatter(floor[:, 0], floor[:, 1], marker='s', s=110,
               c='tab:red', edgecolor='black', linewidth=0.6,
               zorder=4, label='floor anchors (z=0.30)')

    # GT trajectory (green line) + LSE (red dashed)
    ax.plot(gt[:, 0], gt[:, 1], '-', color='tab:green',
            alpha=0.7, lw=1.4, label='GT', zorder=2)
    ax.plot(pred[:, 0], pred[:, 1], '--', color='tab:red',
            alpha=0.6, lw=1.0, label='LSE predicted', zorder=2)
    ax.scatter(gt[0, 0], gt[0, 1], c='tab:green', s=80,
               marker='o', edgecolor='black', zorder=5,
               label='GT start')
    ax.scatter(gt[-1, 0], gt[-1, 1], c='tab:green', s=80,
               marker='X', edgecolor='black', zorder=5,
               label='GT end')
    ax.scatter(pred[0, 0], pred[0, 1], c='tab:red', s=80,
               marker='o', edgecolor='black', zorder=5)
    ax.scatter(pred[-1, 0], pred[-1, 1], c='tab:red', s=80,
               marker='X', edgecolor='black', zorder=5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"LSE trilateration — scene '{scene}' ({mode} localization, "
        f"23 anchors, ref=A1)\n"
        f"MAE = {result['overall_mae_m']:.3f} m   "
        f"p50 = {result['p50_m']:.3f}   "
        f"p95 = {result['p95_m']:.3f}   "
        f"max = {result['max_m']:.3f}   "
        f"RMSE = {result['rmse_m']:.3f} m\n"
        f"per-axis MAE:  x = {result['abs_err_per_axis'][0]:.3f} m   "
        f"y = {result['abs_err_per_axis'][1]:.3f} m"
    )
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=2)
    ax.set_xlim(-17, 17)
    ax.set_ylim(-7, 7)

    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_per_step_err(result: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(len(result["err_norm"]))
    ax.plot(x, result["err_norm"], '-', color='tab:blue', alpha=0.7, lw=0.8)
    mae = result["overall_mae_m"]
    ax.axhline(mae, color='black', linestyle='--',
               label=f"mean = {mae:.3f} m")
    p95 = result["p95_m"]
    ax.axhline(p95, color='red', linestyle='--',
               label=f"p95 = {p95:.3f} m")
    ax.set_xlabel("sample index (chronological)")
    ax.set_ylabel("LSE per-step error (m)")
    ax.set_title(f"LSE per-step Euclidean error — scene '{result['scene']}'")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle=':')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--scene", choices=(*SCENES, "all"), default="big",
                   help="which scene to run (default 'big'; 'all' for all 3)")
    p.add_argument("--mode", choices=("2d", "3d"), default="2d",
                   help="2D solves for (x, y); 3D adds z (default 2d)")
    p.add_argument("--top_k_anchors", type=int, default=None,
                   help="if set, use only the K cleanest anchors "
                        "(selected by per-anchor abs-mean ToA noise "
                        "against the GT — oracle, CEILING for LSE). "
                        "Default = all 23.")
    p.add_argument("--tag_z", type=float, default=1.0,
                   help="known tag elevation in meters. Used for the "
                        "3D->2D ToA projection when --mode=2d "
                        "(default 1.0).")
    p.add_argument("--no_project", action="store_true",
                   help="disable the 3D->2D ToA projection in 2D mode. "
                        "Use this to reproduce the un-projected "
                        "baseline.")
    p.add_argument("--nonlinear", action="store_true",
                   help="use nonlinear LSE (Gauss-Newton / "
                        "Levenberg-Marquardt) on the L2 distance "
                        "error instead of the linearized LSE. Much "
                        "more robust to large NLOS bias. Requires "
                        "scipy.")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default code/checkpoints/_lse_office_rack_<scene>)")
    args = p.parse_args()

    # Load anchor positions (always 23)
    anchor_xyz_t = load_office_rack_anchor_positions()
    anchor_xyz = anchor_xyz_t.cpu().numpy()              # (23, 3)
    print(f"[anchors] 23 anchors loaded:")
    print(f"          X range: [{anchor_xyz[:, 0].min()}, {anchor_xyz[:, 0].max()}] m")
    print(f"          Y range: [{anchor_xyz[:, 1].min()}, {anchor_xyz[:, 1].max()}] m")
    print(f"          Z values: {sorted(set(anchor_xyz[:, 2].tolist()))} m")

    scenes = SCENES if args.scene == "all" else (args.scene,)
    if args.out is None:
        if len(scenes) == 1:
            out_dir = Path("/Users/justin/ultraclear-paper/code/checkpoints"
                           ) / f"_lse_office_rack_{scenes[0]}"
        else:
            out_dir = Path("/Users/justin/ultraclear-paper/code/checkpoints"
                           ) / f"_lse_office_rack_all"
    else:
        out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print(f"LSE trilateration on office_rack sionna track")
    print(f"  mode:    {args.mode}")
    print(f"  scene:   {scenes}")
    print(f"  out:     {out_dir}")
    print("=" * 70)
    print()

    summary: list[dict] = []
    for scene in scenes:
        result = run_scene(scene, args.data_root, anchor_xyz,
                           mode=args.mode,
                           top_k_anchors=args.top_k_anchors,
                           tag_z=args.tag_z,
                           project_3d_to_2d=not args.no_project,
                           nonlinear=args.nonlinear)
        summary.append(result)
        print(f"[{scene:6s}] n={result['n']:4d}  "
              f"MAE = {result['overall_mae_m']:.3f} m   "
              f"RMSE = {result['rmse_m']:.3f} m   "
              f"p50 = {result['p50_m']:.3f}   "
              f"p95 = {result['p95_m']:.3f}   "
              f"max = {result['max_m']:.3f}")
        if args.mode == "2d":
            print(f"         per-axis MAE:  x = {result['abs_err_per_axis'][0]:.3f} m   "
                  f"y = {result['abs_err_per_axis'][1]:.3f} m")
        else:
            print(f"         per-axis MAE:  x = {result['abs_err_per_axis'][0]:.3f} m   "
                  f"y = {result['abs_err_per_axis'][1]:.3f} m   "
                  f"z = {result['abs_err_per_axis'][2]:.3f} m")

        # Per-scene figures
        plot_scene_traj(result, anchor_xyz,
                        out_dir / f"{scene}_lse_vs_gt.png", args.mode)
        plot_per_step_err(result, out_dir / f"{scene}_err_vs_t.png")

        # Per-scene per-step CSV
        per_step_csv = out_dir / f"{scene}_per_step.csv"
        with open(per_step_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_idx", "x_pred", "y_pred", "x_true", "y_true",
                        "err_x", "err_y", "err_norm"])
            for i in range(result["n"]):
                w.writerow([
                    i,
                    f"{result['pred'][i, 0]:.4f}",
                    f"{result['pred'][i, 1]:.4f}",
                    f"{result['gt'][i, 0]:.4f}",
                    f"{result['gt'][i, 1]:.4f}",
                    f"{result['err'][i, 0]:.4f}",
                    f"{result['err'][i, 1]:.4f}",
                    f"{result['err_norm'][i]:.4f}",
                ])

    # Summary table
    table_csv = out_dir / "lse_summary.csv"
    with open(table_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "n", "mode", "MAE_m", "RMSE_m", "p50_m",
                    "p95_m", "max_m",
                    "MAE_x_m", "MAE_y_m", "MAE_z_m_or_blank"])
        for r in summary:
            row = [r["scene"], r["n"], args.mode,
                   f"{r['overall_mae_m']:.4f}",
                   f"{r['rmse_m']:.4f}",
                   f"{r['p50_m']:.4f}",
                   f"{r['p95_m']:.4f}",
                   f"{r['max_m']:.4f}",
                   f"{r['abs_err_per_axis'][0]:.4f}",
                   f"{r['abs_err_per_axis'][1]:.4f}"]
            if args.mode == "3d" and len(r["abs_err_per_axis"]) >= 3:
                row.append(f"{r['abs_err_per_axis'][2]:.4f}")
            else:
                row.append("")
            w.writerow(row)
    print(f"\n[save]  {table_csv}")

    # If multiple scenes, also write a 1xN comparison figure
    if len(summary) > 1:
        fig, axes = plt.subplots(1, len(summary),
                                 figsize=(9 * len(summary), 7),
                                 constrained_layout=True)
        for ax, r in zip(axes, summary):
            gt = r["gt"]
            pred = r["pred"]
            # Anchor geometry
            ceiling = anchor_xyz[anchor_xyz[:, 2] > 1.0]
            floor = anchor_xyz[anchor_xyz[:, 2] <= 1.0]
            ax.scatter(ceiling[:, 0], ceiling[:, 1], marker='^', s=70,
                       c='tab:blue', edgecolor='black', linewidth=0.5,
                       zorder=4)
            ax.scatter(floor[:, 0], floor[:, 1], marker='s', s=70,
                       c='tab:red', edgecolor='black', linewidth=0.5,
                       zorder=4)
            ax.plot(gt[:, 0], gt[:, 1], '-', color='tab:green',
                    alpha=0.7, lw=1.2, zorder=2)
            ax.plot(pred[:, 0], pred[:, 1], '--', color='tab:red',
                    alpha=0.6, lw=0.8, zorder=2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_title(f"{r['scene']} — MAE = {r['overall_mae_m']:.3f} m\n"
                         f"p50 = {r['p50_m']:.3f}  p95 = {r['p95_m']:.3f}",
                         fontsize=10)
            ax.set_xlim(-17, 17)
            ax.set_ylim(-7, 7)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
        # Legend on the first axis only
        axes[0].plot([], [], '-', color='tab:green', label='GT')
        axes[0].plot([], [], '--', color='tab:red', label='LSE')
        axes[0].scatter([], [], marker='^', c='tab:blue', label='ceiling')
        axes[0].scatter([], [], marker='s', c='tab:red', label='floor')
        axes[0].legend(loc='upper right', fontsize=8, framealpha=0.9)

        parts = [f'{r["scene"]}: {r["overall_mae_m"]:.2f}m' for r in summary]
        fig.suptitle(
            f"LSE trilateration on all 3 office-rack sionna scenes "
            f"({args.mode}, 23 anchors, ref=A1)\n"
            f"  |  ".join(parts),
            fontsize=12, y=1.02,
        )
        all_path = out_dir / "lse_all_scenes.png"
        fig.savefig(all_path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f"[save]  {all_path}")

    print()
    print("summary:")
    for r in summary:
        print(f"  {r['scene']:6s}  MAE = {r['overall_mae_m']:.3f} m  "
              f"(n={r['n']}, p50={r['p50_m']:.3f}, p95={r['p95_m']:.3f})")


if __name__ == "__main__":
    main()
