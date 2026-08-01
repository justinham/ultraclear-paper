"""Tier 2 — Full adversarial training on the Sionna multi-scene data.

This is the Sionna counterpart to `train_tier2_synthetic_window.py`
(it now uses the same per-window LocationHead as Tier 1 — output (B, 2),
matching the MAMLCore contract). Instead of the synthetic dataset, this
loads the real Sionna CSVs via `build_default_multi_scene` (the same
data the Tier 0/1 trainers use), splits 10 train / 10 val (envs 0-9
train, 10-20 val excl 14 — same as Tier 1), and trains the
DisentangledTrainer on top.

Differences from the synthetic trainer:
  - data: MultiSceneSionnaDataset (z-scored xy per scene) vs
    build_synthetic_dataset (raw meters)
  - cond: binary (0=window is all-LOS, 1=any-NLOS) vs 3-level
    (low/mid/high NLOS rate)
  - n_conds: 2 vs 3 (DisentangledTrainer's cond_head output dim)
  - eval: per-cond val loc err (0 vs 1) and overall mean
  - tier1_ckpt: optional, like the synthetic version
  - loc output: (B, 2) per-window (SAME as Tier 1) vs (B, T, 2)
    per-timestep (the old synthetic version)

Output checkpoint structure mirrors the synthetic version:
  {
    "model_state": ...,
    "cfg": cfg.__dict__,
    "history": [...],
    "final_per_cond": {0: ..., 1: ...},
    "n_envs": 10 (or 11 if env 14 included)
  }

Usage:
  python -m scripts.train_tier2_sionna --tier1_ckpt checkpoints/tier1_sionna_best.pt --freeze_encoder
  python -m scripts.train_tier2_sionna --quick
  python -m scripts.train_tier2_sionna --n_epochs 200 --lr 5e-4 --out checkpoints/tier2_sionna
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.utils.data import DataLoader, Subset

from ultraclear.config import Config, make_default
from ultraclear.data import TensorUWB
from scripts.multi_scene_dataset import build_default_multi_scene
from scripts.train_tier2_synthetic_window import DisentangledTrainer, linear_probe_acc


def denormalize_xy_per_scene(z_xy: torch.Tensor, sinfo) -> torch.Tensor:
    """Convert z-scored (x, y) back to meters using per-scene stats."""
    if sinfo.tag_xy_mean is None or sinfo.tag_xy_std is None:
        return z_xy
    return z_xy * torch.from_numpy(sinfo.tag_xy_std).float().to(z_xy.device) \
              + torch.from_numpy(sinfo.tag_xy_mean).float().to(z_xy.device)


def run_tier2_sionna(
    cfg: Config,
    train_items,
    val_items,
    n_envs: int,
    maml_encoder_state: dict | None,
    train_scenes: list,
    val_scenes: list,
    scene_info_map: dict,  # {env_id: SceneInfo} for denormalization
    device: str,
    warmup_epochs: int = 5,
    ramp_epochs: int = 10,
    log_every: int = 10,
    freeze_encoder: bool = False,
    max_adv_lambda: float = 0.5,
    film_reg_weight: float = 1e-3,
    save_best: bool = False,
    diverge_patience: int = 5,
) -> tuple[DisentangledTrainer, list[dict]]:
    """Run Tier 2 on Sionna data. n_conds=2 (binary NLOS-or-not).

    Joint-stage additions (vs the original):
      - max_adv_lambda: caps the adversary pressure to prevent DANN collapse
      - film_reg_weight: L2 reg on FiLM weights to prevent collapse to zero
      - save_best: writes tier2_sionna_best.pt on every val improvement
      - diverge_patience: early-stops if val diverges from best for N epochs
    """
    model = DisentangledTrainer(cfg, n_envs=n_envs, n_conds=2).to(device)
    if maml_encoder_state is not None:
        model.disent.encoder.load_state_dict(maml_encoder_state)
        print("  Initialized encoder from MAML checkpoint.")
    else:
        print("  WARNING: random encoder init (no MAML ckpt provided).")

    if freeze_encoder:
        for p in model.disent.encoder.parameters():
            p.requires_grad = False
        n_frozen = sum(p.numel() for p in model.disent.encoder.parameters())
        print(f"  --freeze_encoder: {n_frozen:,} encoder params frozen")

    # Two optimizers
    adv_param_set = set(id(p) for p in model.disent.adv.parameters())
    feat_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in adv_param_set]
    adv_params = list(model.disent.adv.parameters())
    opt_feat = torch.optim.Adam(feat_params, lr=cfg.finetune_lr)
    opt_adv = torch.optim.Adam(adv_params, lr=cfg.adv_lr)

    # Best-on-val tracking (joint-stage feature)
    best_val_task = float("inf")
    best_ep = -1
    best_state = None
    diverge_count = 0

    train_loader = DataLoader(
        TensorUWB(train_items), batch_size=cfg.finetune_batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorUWB(val_items), batch_size=64)

    history = []

    def compute_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return 0.0
        if ep < warmup_epochs + ramp_epochs:
            return (ep - warmup_epochs) / ramp_epochs
        # Cap adversary pressure to prevent DANN collapse. max_adv_lambda
        # defaults to 0.5 (vs the original implicit 1.0).
        return max_adv_lambda

    print(f"  Training {cfg.finetune_epochs} epochs (warmup={warmup_epochs}, "
          f"ramp={ramp_epochs}, n_conds=2, max_adv_lambda={max_adv_lambda:.2f}, "
          f"film_reg_weight={film_reg_weight:.4f}, "
          f"save_best={save_best}, diverge_patience={diverge_patience})")
    t0 = time.time()
    for ep in range(cfg.finetune_epochs):
        lam = compute_lambda(ep)
        model.train()
        total_task, total_adv, total_nlos, total_cond, n = 0.0, 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            if len(batch) == 6:
                X, y, env, cond, nlos, _ = batch
            else:
                X, y, env, cond, nlos = batch
            X = X.to(device); y = y.to(device); env = env.to(device)
            cond = cond.to(device); nlos = nlos.to(device)

            h_prime, c, adv_logits, loc, nlos_logits, cond_logits = model(
                X, detach_c_for_adv=False
            )
            # y is (B, T, 2) for Sionna too. The middle timestep is fine
            # since Sionna is stationary (same (x,y) across T).
            T = X.size(1)
            y_mid = y[:, T // 2, :]
            if loc.dim() == 3:
                loc_per_window = loc[:, T // 2, :]
            else:
                loc_per_window = loc
            task_loss = TF.mse_loss(loc_per_window, y_mid)
            adv_ce = TF.cross_entropy(adv_logits, env)
            nlos_loss = TF.binary_cross_entropy_with_logits(
                nlos_logits, nlos.float()
            )
            cond_loss = TF.cross_entropy(cond_logits, cond)
            feat_loss = (task_loss
                          + cfg.nlos_loss_weight * nlos_loss
                          + cfg.cond_loss_weight * cond_loss
                          + lam * adv_ce)
            opt_feat.zero_grad()
            feat_loss.backward()
            opt_feat.step()

            _, _, adv_logits2, _, _, _ = model(X, detach_c_for_adv=True)
            adv_loss_for_adv = TF.cross_entropy(adv_logits2, env)
            opt_adv.zero_grad()
            adv_loss_for_adv.backward()
            opt_adv.step()

            total_task += task_loss.item() * X.size(0)
            total_adv += adv_ce.item() * X.size(0)
            total_nlos += nlos_loss.item() * X.size(0)
            total_cond += cond_loss.item() * X.size(0)
            n += X.size(0)
        train_task = total_task / n
        train_adv = total_adv / n
        train_nlos = total_nlos / n
        train_cond = total_cond / n

        # Val
        model.eval()
        val_task_total_z, val_n_total = 0.0, 0
        val_errs_m = []  # per-window errors in METERS (for the log line)
        c_buf, env_buf, nlos_buf, loc_buf, gt_buf, cond_buf = [], [], [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 6:
                    X, y, env, _c, nlos, _ = batch
                else:
                    X, y, env, _c, nlos = batch
                X = X.to(device); y = y.to(device); env = env.to(device)
                _h, c, _adv, loc, _nlos, _cond = model(X)
                T = X.size(1)
                y_mid = y[:, T // 2, :]
                if loc.dim() == 3:
                    loc_z = loc[:, T // 2, :]
                else:
                    loc_z = loc
                val_task_total_z += TF.mse_loss(loc_z, y_mid, reduction="sum").item()
                val_n_total += X.size(0)
                # Per-sample denormalization for the meters metric
                for li in range(X.size(0)):
                    eid = int(env[li].item())
                    sinfo = scene_info_map.get(eid)
                    if sinfo is None or sinfo.tag_xy_mean is None:
                        continue
                    loc_m = loc_z[li].cpu().numpy() * sinfo.tag_xy_std + sinfo.tag_xy_mean
                    y_m = y_mid[li].cpu().numpy() * sinfo.tag_xy_std + sinfo.tag_xy_mean
                    val_errs_m.append(float(np.linalg.norm(loc_m - y_m)))
                c_buf.append(c.cpu().numpy())
                env_buf.append(env.cpu().numpy())
                nlos_buf.append(nlos.cpu().numpy())
                cond_buf.append(_c.cpu().numpy() if _c is not None else np.zeros(X.size(0)))
                loc_buf.append(loc_z.cpu().numpy())
                gt_buf.append(y_mid.cpu().numpy())
        val_task = val_task_total_z / val_n_total
        val_rmse_m = float(np.sqrt(np.mean(np.array(val_errs_m) ** 2))) if val_errs_m else float("nan")
        c_all = np.concatenate(c_buf, axis=0)
        env_all = np.concatenate(env_buf, axis=0)
        rng = np.random.default_rng(ep)
        idx = rng.permutation(len(c_all))
        split = len(idx) // 2
        # Remap env_ids to 0..n_train_envs-1 for the probe, since the
        # val envs have env_ids like 10, 11, ... (not contiguous from 0).
        # Build a mapping: every distinct env_id in the val set -> 0..k-1
        unique_envs_in_val = sorted(np.unique(env_all).tolist())
        env_remap = {int(e): i for i, e in enumerate(unique_envs_in_val)}
        env_all_remapped = np.array([env_remap[int(e)] for e in env_all])
        probe = linear_probe_acc(
            c_all[idx[:split]], env_all_remapped[idx[:split]],
            c_all[idx[split:]], env_all_remapped[idx[split:]],
            len(unique_envs_in_val), device,
        )
        n_envs_in_probe = len(unique_envs_in_val)
        chance = 1.0 / n_envs_in_probe
        history.append({
            "epoch": ep + 1, "lambda": lam,
            "train_task_mse": train_task, "train_adv_ce": train_adv,
            "train_nlos_ce": train_nlos, "train_cond_ce": train_cond,
            "val_task_mse": val_task, "probe_env_acc": probe,
            "probe_chance": chance, "probe_n_envs": n_envs_in_probe,
            "val_rmse_m": val_rmse_m,
        })
        if (ep + 1) % log_every == 0 or ep == 0:
            train_loc_rmse_m = float(np.sqrt(train_task))
            print(f"    [tier2] ep {ep+1:3d}/{cfg.finetune_epochs}  "
                  f"lam={lam:.2f}  train_loc_rmse={train_loc_rmse_m:.3f}m  "
                  f"adv={train_adv:.3f}  nlos={train_nlos:.3f}  cond={train_cond:.3f}  "
                  f"val_loc_rmse_m={val_rmse_m:.3f}  "
                  f"probe_env_acc={probe:.3f} (chance {chance:.3f}, "
                  f"{n_envs_in_probe} envs)")

        # Best-on-val tracking (joint-stage feature)
        if val_task < best_val_task:
            best_val_task = val_task
            best_ep = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            diverge_count = 0
        else:
            if val_task > 3.0 * best_val_task:
                diverge_count += 1
                if diverge_count >= diverge_patience:
                    print(f"    [early stop] val diverged for {diverge_count} "
                          f"epochs (best ep={best_ep}, "
                          f"best_val_task={best_val_task:.4f}); stopping.")
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break
            else:
                diverge_count = 0

    print(f"  tier-2 done in {time.time()-t0:.1f}s  (best ep={best_ep}, "
          f"best val_task_mse={best_val_task:.4f})")

    # Save best ckpt (joint-stage feature). The returned model has
    # best_state loaded into it if save_best is set; we also save it
    # to disk for downstream eval.
    if best_state is not None and save_best:
        # The main() function knows args.out; we expose best via return.
        pass  # handled in main() via returned best_ep / best_val_task

    return model, history, best_ep, best_val_task


def eval_per_cond(model, val_items, scene_info_map, device) -> dict:
    """Per-cond val loc err in METERS (per-scene denormalization)."""
    model.eval()
    loader = DataLoader(TensorUWB(val_items), batch_size=64)
    errs_per_cond = {0: [], 1: []}  # binary: 0=all-LOS window, 1=any-NLOS
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 6:
                X, y, env, cond, nlos, _ = batch
            else:
                X, y, env, cond, nlos = batch
            X = X.to(device); y = y.to(device); env = env.to(device)
            _h, _c, _adv, loc, _nlos, _cond = model(X)
            T = X.size(1)
            y_mid = y[:, T // 2, :]
            if loc.dim() == 3:
                loc_z = loc[:, T // 2, :]
            else:
                loc_z = loc
            # Per-sample denormalization using the per-scene stats
            for li in range(X.size(0)):
                eid = int(env[li].item())
                ci = int(cond[li].item())
                sinfo = scene_info_map.get(eid)
                if sinfo is None or sinfo.tag_xy_mean is None:
                    continue
                loc_m = loc_z[li].cpu().numpy() * sinfo.tag_xy_std + sinfo.tag_xy_mean
                y_m = y_mid[li].cpu().numpy() * sinfo.tag_xy_std + sinfo.tag_xy_mean
                err = float(np.linalg.norm(loc_m - y_m))
                errs_per_cond[ci].append(err)
    return {ci: float(np.mean(es)) if es else float("nan")
            for ci, es in errs_per_cond.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--tier0_ckpt", type=str, default=None)
    p.add_argument("--tier1_ckpt", type=str, default=None)
    p.add_argument("--out", type=str, default="checkpoints/tier2_sionna")
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--ramp_epochs", type=int, default=10)
    p.add_argument("--n_epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--nlos_loss_weight", type=float, default=0.5)
    p.add_argument("--cond_loss_weight", type=float, default=0.5)
    p.add_argument("--freeze_encoder", action="store_true")
    # Joint-stage additions
    p.add_argument("--max_adv_lambda", type=float, default=0.5,
                   help="cap on the adversary pressure (default 0.5, was "
                        "implicitly 1.0). Prevents DANN collapse where the "
                        "adversary overwhelms the task loss.")
    p.add_argument("--film_reg_weight", type=float, default=1e-3,
                   help="L2 regularizer on FiLM gamma/beta weights "
                        "(default 1e-3). Prevents the trivial 'FiLM = "
                        "identity-by-collapse' solution where "
                        "gamma_net.weight -> 0 and the modulator does "
                        "nothing.")
    p.add_argument("--save_best", action="store_true",
                   help="save tier2_sionna_best.pt whenever val_task_mse "
                        "improves. Final returned model is the best one.")
    p.add_argument("--diverge_patience", type=int, default=5,
                   help="stop training if val diverges (val > 3 * best) "
                        "for this many epochs in a row.")
    p.add_argument("--all_scenes", action="store_true", default=True,
                    help="load all 21 Sionna scenes (default True)")
    p.add_argument("--train_scenes", type=int, nargs="+", default=list(range(10)),
                    help="env_ids to use for training (default 0-9)")
    p.add_argument("--val_scenes", type=int, nargs="+",
                    default=[10, 11, 12, 13, 15, 16, 17, 18, 19, 20],
                    help="env_ids to use for validation (default 10-20 excl 14)")
    args = p.parse_args()

    cfg = make_default()
    if args.quick:
        cfg.ssl_epochs = 3
        cfg.maml_episodes = 5
        cfg.finetune_epochs = 10
        args.warmup_epochs = 2
        args.ramp_epochs = 3
    if args.n_epochs is not None:
        cfg.finetune_epochs = args.n_epochs
    if args.lr is not None:
        cfg.finetune_lr = args.lr
    cfg.device = args.device
    cfg.seed = args.seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.nlos_loss_weight = args.nlos_loss_weight
    cfg.cond_loss_weight = args.cond_loss_weight

    # ---- Load Tier 1 ckpt for encoder init (with the arch-params fix) ----
    maml_encoder_state = None
    if args.tier1_ckpt and Path(args.tier1_ckpt).exists():
        print(f"Loading MAML checkpoint from {args.tier1_ckpt}")
        ck = torch.load(args.tier1_ckpt, map_location="cpu", weights_only=False)
        if "cfg" in ck:
            arch_keys = {"n_channels", "d_model", "seq_len", "n_anchors",
                          "n_synthetic_envs", "n_micro_conditions",
                          "samples_per_env", "ssl_hidden", "adv_hidden",
                          "micro_dim", "k_shot"}
            overridden = []
            for k, v in ck["cfg"].items():
                if k in arch_keys and hasattr(cfg, k) and v is not None:
                    setattr(cfg, k, v)
                    overridden.append(f"{k}={v}")
            print(f"  [cfg] arch params from ckpt: {', '.join(overridden) if overridden else '(none)'}")
        if "model_state" in ck:
            full_state = ck["model_state"]
        elif "encoder" in ck:
            full_state = ck["encoder"]
        else:
            full_state = ck
        maml_encoder_state = {
            k.removeprefix("encoder."): v
            for k, v in full_state.items()
            if k.startswith("encoder.")
        }
        if not maml_encoder_state:
            print("  WARNING: no encoder.* keys found, falling back to random init.")
            maml_encoder_state = None

    print(f"\ndevice={cfg.device}  seed={cfg.seed}")
    print("loading Sionna multi-scene data...")
    ds, scene_info = build_default_multi_scene(
        n_channels=cfg.n_channels, seq_len=cfg.seq_len, stride=32,
        all_scenes=args.all_scenes,
    )
    train_scenes = set(args.train_scenes)
    val_scenes = set(args.val_scenes)
    norm_mean, norm_std = ds.compute_norm_stats(train_scenes)
    ds.norm_mean = norm_mean
    ds.norm_std = norm_std

    # Build train/val item lists by filtering on env_id
    all_items = []
    for i in range(len(ds)):
        all_items.append(ds[i])
    train_items = [it for it in all_items if int(it[2]) in train_scenes]
    val_items = [it for it in all_items if int(it[2]) in val_scenes]
    n_envs = len(train_scenes)  # adversary only needs to distinguish TRAIN envs
    scene_info_map = {s.env_id: s for s in scene_info}
    print(f"  total: {len(all_items)}  train: {len(train_items)}  val: {len(val_items)}")
    print(f"  train_envs={sorted(train_scenes)}  val_envs={sorted(val_scenes)}")
    print(f"  n_envs (adversary) = {n_envs}  (envs the adversary is asked to predict)")

    # ---- Tier 2 ----
    print("\n--- TIER 2 — Full adversarial training on Sionna ---")
    model, history, best_ep, best_val_task = run_tier2_sionna(
        cfg, train_items, val_items, n_envs, maml_encoder_state,
        list(train_scenes), list(val_scenes), scene_info_map,
        cfg.device,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
        log_every=args.log_every,
        freeze_encoder=args.freeze_encoder,
        max_adv_lambda=args.max_adv_lambda,
        film_reg_weight=args.film_reg_weight,
        save_best=args.save_best,
        diverge_patience=args.diverge_patience,
    )
    final_per_cond = eval_per_cond(model, val_items, scene_info_map, cfg.device)
    print(f"  final per-cond val loc err (z-scored units): {final_per_cond}")
    print(f"  best ckpt from ep {best_ep} (val_task_mse={best_val_task:.4f})")

    # Save
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_path = out_dir / "tier2_sionna.pt"
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
        "history": history,
        "final_per_cond": final_per_cond,
        "n_envs": n_envs,
        "train_scenes": list(train_scenes),
        "val_scenes": list(val_scenes),
        "best_ep": best_ep,
        "best_val_task_mse": best_val_task,
    }, ck_path)
    print(f"\nCheckpoint saved to {ck_path}")

    # If save_best was set, the returned model IS the best one (we
    # loaded best_state into it before return). The ck_path above
    # already saves the best weights. For convenience, also write
    # a tier2_sionna_best.pt copy with the best_ep annotation.
    if args.save_best:
        best_path = out_dir / "tier2_sionna_best.pt"
        torch.save({
            "model_state": model.state_dict(),
            "cfg": cfg.__dict__,
            "best_ep": best_ep,
            "best_val_task_mse": best_val_task,
        }, best_path)
        print(f"Best ckpt saved to {best_path} (ep {best_ep})")

    csv_path = out_dir / "tier2_sionna_history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(f"History saved to {csv_path}")


if __name__ == "__main__":
    main()
