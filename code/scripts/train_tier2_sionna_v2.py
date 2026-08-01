"""Tier 2 — Full adversarial training on the Sionna multi-scene data
(v2: unfrozen-encoder experiment + DANN cap + early-stopping).

V2 differs from `train_tier2_sionna.py` in three concrete ways:

  (1) **Encoder is NOT frozen by default.** The v1 default was to
      freeze the MAML encoder; v2 trains end-to-end so the encoder
      can adapt to disentanglement pressure. This is the
      experiment. Caveat: 7/10 memory note says freezing is the
      "critical trick" that keeps the location head alive when
      the adversary is active. v2 is therefore expected to be
      unstable for some fraction of runs. We mitigate with:
        - DANN cap (point 2)
        - gradient clipping on the feature optimizer
        - per-epoch encoder-grad-norm logging so we can see
          the instability if it happens
        - early-stopping on val_mse (point 3)
      Pass `--freeze_encoder` to reproduce the v1 behavior.

  (2) **DANN cap is now applied** to the inlined feat_loss. The cap
      (clamp `adv_ce - 2*ln(n_envs)` above zero) lives in
      `tier2_micro.disentangle_loss` but v1 inlined the loss and
      never applied the cap, so once lambda reached 1.0 the
      adversary gradient blew through the encoder and destroyed
      the location head (val_mse 0.27 -> 2.0 over epochs 4-100).
      v2 ports the cap to the inlined loss.

  (3) **Early-stopping on val_mse**: the best ckpt (by val_mse in
      meters) is saved separately as `tier2_sionna_best.pt` while
      the final epoch is saved as `tier2_sionna_final.pt`. v1
      only saved the final epoch, which on its last run was the
      destroyed one.

The v1 default train/val split is also replaced with the
new mixed split (the same as the Tier 1 mixed ckpt) so the
val numbers are directly comparable to the new Tier 1 K-shot
curve.

Differences from the synthetic trainer:
  - data: MultiSceneSionnaDataset (z-scored xy per scene) vs
    build_synthetic_dataset (raw meters)
  - cond: binary (0=window is all-LOS, 1=any-NLOS) vs 3-level
    (low/mid/high NLOS rate)
  - n_conds: 2 vs 3 (DisentangledTrainer's cond_head output dim)
  - eval: per-cond val loc err (0 vs 1) and overall mean
  - tier1_ckpt: optional, like the synthetic version
  - default tier1_ckpt: checkpoints/tier1_mixed.pt/tier1_sionna.pt
    (the new mixed-split MAML ckpt, so the encoder we adapt
    is consistent with the K-shot curve)

Output checkpoint structure mirrors the synthetic version:
  {
    "model_state": ...,
    "cfg": cfg.__dict__,
    "history": [...],
    "final_per_cond": {0: ..., 1: ...},
    "n_envs": 10 (or 11 if env 14 included),
    "train_scenes": [...],
    "val_scenes": [...],
    "best_epoch": int,
  }

Usage:
  # default: unfreeze encoder, DANN cap, mixed split, early stop
  python -m scripts.train_tier2_sionna_v2
  # ablation: freeze encoder (v1 behavior + cap)
  python -m scripts.train_tier2_sionna_v2 --freeze_encoder
  # smoke test
  python -m scripts.train_tier2_sionna_v2 --quick
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
from scripts.train_tier2_synthetic import DisentangledTrainer, linear_probe_acc


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
    save_best: bool = True,
) -> tuple[DisentangledTrainer, list[dict], dict]:
    """Run Tier 2 on Sionna data. n_conds=2 (binary NLOS-or-not).

    Returns (model, history, best_info) where best_info is:
      {"epoch": int, "val_rmse_m": float, "val_task_mse": float,
       "model_state": dict, "train_scenes": list, "val_scenes": list}
    If save_best=False, best_info has epoch=-1 and val_rmse_m=NaN
    and the final-epoch state is returned as model_state.
    """
    """Run Tier 2 on Sionna data. n_conds=2 (binary NLOS-or-not)."""
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
    else:
        # v2 default: encoder is trainable. Per the joint trainer's
        # 7/10 finding, the encoder gradient is the main failure
        # mode when disentanglement is active. We clip aggressively
        # (max_norm=1.0) and log the encoder grad norm so we can
        # see if/when it explodes.
        n_enc = sum(p.numel() for p in model.disent.encoder.parameters())
        print(f"  encoder TRAINABLE ({n_enc:,} params); grad clip max_norm=1.0")

    # Two optimizers
    adv_param_set = set(id(p) for p in model.disent.adv.parameters())
    feat_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in adv_param_set]
    adv_params = list(model.disent.adv.parameters())
    opt_feat = torch.optim.Adam(feat_params, lr=cfg.finetune_lr)
    opt_adv = torch.optim.Adam(adv_params, lr=cfg.adv_lr)

    train_loader = DataLoader(
        TensorUWB(train_items), batch_size=cfg.finetune_batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorUWB(val_items), batch_size=64)

    history = []

    # v2: DANN cap. adv_ce can hit 200-300 on real data with 10+ envs
    # (because the featurizer wins so hard the adversary can't predict
    # above chance), and the *raw* adv_ce dominates the gradient and
    # destroys the location head. Cap the disentanglement pressure to
    # fire only when the adversary is above 2*chance (DANN standard).
    # chance = 1/n_envs for the cross-entropy log-uniform prior; for
    # n_envs=10, ln(10) ~= 2.30, so 2*chance ~= 4.61 nats.
    chance_log = float(np.log(n_envs))
    adv_cap = 2.0 * chance_log

    def compute_lambda(ep: int) -> float:
        if ep < warmup_epochs: return 0.0
        if ep < warmup_epochs + ramp_epochs: return (ep - warmup_epochs) / ramp_epochs
        return 1.0

    print(f"  Training {cfg.finetune_epochs} epochs (warmup={warmup_epochs}, "
          f"ramp={ramp_epochs}, n_conds=2, adv_cap=2*ln({n_envs})={adv_cap:.2f} nats)")
    t0 = time.time()
    # v2: best-epoch tracking by val_rmse_m (meters, not z-scored MSE).
    # The v1 final epoch on its last run had val_mse=2.0 (RMSE 1.41m)
    # while ep 4 had val_mse=0.27 (RMSE 0.52m). Without early-stopping,
    # the saved ckpt is the worst one. v2 saves the best ckpt state in
    # memory and returns it via best_info so main() can write it to
    # tier2_sionna_best.pt alongside the final-epoch state.
    best_val_rmse = float("inf")
    best_epoch = -1
    best_model_state = None
    for ep in range(cfg.finetune_epochs):
        lam = compute_lambda(ep)
        model.train()
        total_task, total_adv, total_nlos, total_cond, n = 0.0, 0.0, 0.0, 0.0, 0
        total_enc_gnorm, total_adv_gnorm = 0.0, 0.0
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
            # v2: DANN cap. Only push the featurizer AWAY from
            # being predictive of env when the adversary is doing
            # better than 2*chance; otherwise the cap zeros the
            # gradient and the featurizer only sees task + aux losses.
            adv_capped = torch.clamp(adv_ce - adv_cap, min=0.0)
            feat_loss = (task_loss
                          + cfg.nlos_loss_weight * nlos_loss
                          + cfg.cond_loss_weight * cond_loss
                          + lam * adv_capped)
            opt_feat.zero_grad()
            feat_loss.backward()
            # v2: gradient clipping on the feature optimizer. Without
            # this, the raw adv_ce spike at the start of the ramp can
            # push the encoder gradient norm into the thousands, and
            # the location head never recovers. max_norm=1.0 is a
            # conservative default; increase if training is too slow.
            feat_gnorm = torch.nn.utils.clip_grad_norm_(feat_params, max_norm=1.0)
            # Track encoder-specific grad norm separately, so we can
            # see whether the instability is coming from the encoder
            # or the heads. encoder grad norm is the union of all
            # encoder param grads (one float per optimizer step).
            enc_gnorm = float('nan')
            if not freeze_encoder:
                enc_sq = 0.0
                for p in model.disent.encoder.parameters():
                    if p.grad is not None:
                        enc_sq += float(p.grad.detach().pow(2).sum().item())
                enc_gnorm = enc_sq ** 0.5
            opt_feat.step()

            _, _, adv_logits2, _, _, _ = model(X, detach_c_for_adv=True)
            adv_loss_for_adv = TF.cross_entropy(adv_logits2, env)
            opt_adv.zero_grad()
            adv_loss_for_adv.backward()
            adv_gnorm = torch.nn.utils.clip_grad_norm_(adv_params, max_norm=5.0)
            opt_adv.step()

            total_task += task_loss.item() * X.size(0)
            total_adv += adv_ce.item() * X.size(0)
            total_nlos += nlos_loss.item() * X.size(0)
            total_cond += cond_loss.item() * X.size(0)
            total_enc_gnorm += enc_gnorm * X.size(0)
            total_adv_gnorm += float(adv_gnorm) * X.size(0)
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
        # v2: also compute mean per-batch encoder/adv grad norms
        # so we can see if the encoder gradient is exploding
        # (the main failure mode for the unfrozen-encoder setup).
        mean_enc_gnorm = total_enc_gnorm / n if not freeze_encoder else float('nan')
        mean_adv_gnorm = total_adv_gnorm / n
        # v2: track best ckpt by val_rmse_m (in meters). Save the
        # model state so main() can write tier2_sionna_best.pt.
        # Only consider epochs after warmup to avoid the trivial
        # early-epoch minimum before disentanglement kicks in.
        if save_best and val_rmse_m < best_val_rmse and (ep + 1) >= warmup_epochs:
            best_val_rmse = val_rmse_m
            best_epoch = ep + 1
            best_model_state = {k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items()}
            if (ep + 1) % log_every == 0 or ep == 0:
                print(f"      [best] ep {ep+1} new best val_rmse_m={val_rmse_m:.3f}")
        history.append({
            "epoch": ep + 1, "lambda": lam,
            "train_task_mse": train_task, "train_adv_ce": train_adv,
            "train_nlos_ce": train_nlos, "train_cond_ce": train_cond,
            "val_task_mse": val_task, "probe_env_acc": probe,
            "probe_chance": chance, "probe_n_envs": n_envs_in_probe,
            "enc_grad_norm": mean_enc_gnorm, "adv_grad_norm": mean_adv_gnorm,
        })
        if (ep + 1) % log_every == 0 or ep == 0:
            train_loc_rmse_m = float(np.sqrt(train_task))
            enc_str = f"  enc_gnorm={mean_enc_gnorm:6.2f}" if not freeze_encoder else ""
            print(f"    [tier2] ep {ep+1:3d}/{cfg.finetune_epochs}  "
                  f"lam={lam:.2f}  train_loc_rmse={train_loc_rmse_m:.3f}m  "
                  f"adv={train_adv:.3f}  nlos={train_nlos:.3f}  cond={train_cond:.3f}  "
                  f"val_loc_rmse_m={val_rmse_m:.3f}  "
                  f"probe_env_acc={probe:.3f} (chance {chance:.3f}, "
                  f"{n_envs_in_probe} envs){enc_str}")
    print(f"  tier-2 done in {time.time()-t0:.1f}s")
    if save_best and best_model_state is not None:
        print(f"  [best] saved best ckpt from epoch {best_epoch} "
              f"(val_rmse_m={best_val_rmse:.3f})")
        best_info = {
            "epoch": best_epoch,
            "val_rmse_m": best_val_rmse,
            "val_task_mse": best_val_rmse ** 2,
            "model_state": best_model_state,
            "train_scenes": list(train_scenes),
            "val_scenes": list(val_scenes),
        }
    else:
        best_info = {
            "epoch": -1,
            "val_rmse_m": float("nan"),
            "val_task_mse": float("nan"),
            "model_state": None,
            "train_scenes": list(train_scenes),
            "val_scenes": list(val_scenes),
        }
    return model, history, best_info


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
    p.add_argument("--tier1_ckpt", type=str,
                    default="checkpoints/tier1_mixed.pt/tier1_sionna.pt",
                    help="default to the new mixed-split MAML ckpt "
                         "(set to '' to disable MAML init)")
    p.add_argument("--out", type=str, default="checkpoints/tier2_sionna_v2")
    p.add_argument("--warmup_epochs", type=int, default=10,
                    help="v2 default: 10 (was 5 in v1) so the location "
                         "head has time to settle before disentanglement "
                         "pressure arrives")
    p.add_argument("--ramp_epochs", type=int, default=20,
                    help="v2 default: 20 (was 10 in v1) so the lambda "
                         "ramp is gentler when the encoder is unfrozen")
    p.add_argument("--n_epochs", type=int, default=80,
                    help="v2 default: 80 (v1 ran 100 with the location "
                         "head collapsing by ep 50; the useful training "
                         "is over by ~ep 40-50 with the slower ramp)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--nlos_loss_weight", type=float, default=0.5)
    p.add_argument("--cond_loss_weight", type=float, default=0.5)
    p.add_argument("--freeze_encoder", action="store_true",
                    help="v2 default is NO freeze (encoder is trainable). "
                         "Pass this flag to reproduce the v1 frozen setup.")
    p.add_argument("--no_save_best", action="store_true",
                    help="v2 default is to save both best and final ckpts. "
                         "Pass this to only save the final epoch.")
    p.add_argument("--all_scenes", action="store_true", default=True,
                    help="load all 21 Sionna scenes (default True)")
    p.add_argument("--train_scenes", type=int, nargs="+",
                    default=[0, 1, 2, 5, 9, 11, 13, 15, 17, 19],
                    help="v2 default: new mixed split (5 indoor + 5 outdoor) "
                         "matching tier1_mixed.pt")
    p.add_argument("--val_scenes", type=int, nargs="+",
                    default=[3, 4, 6, 7, 8, 10, 12, 16, 18, 20],
                    help="v2 default: new mixed split (5 indoor + 5 outdoor) "
                         "matching tier1_mixed.pt")
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

    # Build train/val item lists by filtering on env_id. We do NOT
    # remap env_ids to a contiguous range here, because the
    # MultiSceneSionnaDataset stores the original env_id (0..20)
    # in each item, and the per-scene denormalization in
    # eval_per_cond needs the original env_id to look up the
    # scene_info_map. The adversary's class set is the full union
    # of train+val envs (n_envs_total = len(train_scenes) +
    # len(val_scenes)), so env_id == target works as a class
    # index without any remapping.
    all_items = []
    for i in range(len(ds)):
        all_items.append(ds[i])
    train_items = [it for it in all_items if int(it[2]) in train_scenes]
    val_items = [it for it in all_items if int(it[2]) in val_scenes]
    # n_envs is the adversary's class-set size = the union of all
    # envs the model will see across both train and val. (For the
    # new mixed split, train ∩ val = ∅, so this is len(train) +
    # len(val) = 20.) v1's `n_envs = len(train_scenes)` was a latent
    # bug that only worked because train/val were disjoint
    # (train 0-9, val 10-20) so the val envs never appeared in
    # the train loader. With the new mixed split, val envs ARE
    # in the train loader, and the adversary must be able to
    # classify them too.
    n_envs_total = len(train_scenes) + len(val_scenes)
    n_envs = n_envs_total
    scene_info_map = {s.env_id: s for s in scene_info}
    print(f"  total: {len(all_items)}  train: {len(train_items)}  val: {len(val_items)}")
    print(f"  train_envs={sorted(train_scenes)}  val_envs={sorted(val_scenes)}")
    print(f"  n_envs (adversary) = {n_envs}  (union of train+val = {len(train_scenes)}+{len(val_scenes)})")

    # ---- Tier 2 ----
    print("\n--- TIER 2 — Full adversarial training on Sionna (v2) ---")
    print(f"  freeze_encoder={args.freeze_encoder}  "
          f"save_best={not args.no_save_best}  "
          f"warmup={args.warmup_epochs}  ramp={args.ramp_epochs}  "
          f"n_epochs={cfg.finetune_epochs}")
    model, history, best_info = run_tier2_sionna(
        cfg, train_items, val_items, n_envs, maml_encoder_state,
        list(train_scenes), list(val_scenes), scene_info_map,
        cfg.device,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
        log_every=args.log_every,
        freeze_encoder=args.freeze_encoder,
        save_best=not args.no_save_best,
    )
    final_per_cond = eval_per_cond(model, val_items, scene_info_map, cfg.device)
    print(f"  final per-cond val loc err (z-scored units): {final_per_cond}")
    if not args.no_save_best and best_info["epoch"] > 0:
        print(f"  best per-cond val loc err (ep {best_info['epoch']}, "
              f"val_rmse_m={best_info['val_rmse_m']:.3f}): "
              f"re-evaluating on best ckpt...")
        model.load_state_dict(best_info["model_state"])
        best_per_cond = eval_per_cond(model, val_items, scene_info_map, cfg.device)
        print(f"  best  per-cond val loc err (z-scored units): {best_per_cond}")
    else:
        best_per_cond = None

    # Save
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_path_final = out_dir / "tier2_sionna_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
        "history": history,
        "final_per_cond": final_per_cond,
        "n_envs": n_envs,
        "train_scenes": list(train_scenes),
        "val_scenes": list(val_scenes),
        "freeze_encoder": args.freeze_encoder,
        "warmup_epochs": args.warmup_epochs,
        "ramp_epochs": args.ramp_epochs,
    }, ck_path_final)
    print(f"\nFinal-epoch checkpoint saved to {ck_path_final}")

    # v2: save the best ckpt (by val_rmse_m) separately so it survives
    # even if the final-epoch ckpt is the destroyed one (which is what
    # happened on the v1 last run).
    if not args.no_save_best and best_info["model_state"] is not None:
        ck_path_best = out_dir / "tier2_sionna_best.pt"
        torch.save({
            "model_state": best_info["model_state"],
            "cfg": cfg.__dict__,
            "history": history,
            "best_epoch": best_info["epoch"],
            "best_val_rmse_m": best_info["val_rmse_m"],
            "best_per_cond": best_per_cond,
            "n_envs": n_envs,
            "train_scenes": list(train_scenes),
            "val_scenes": list(val_scenes),
            "freeze_encoder": args.freeze_encoder,
            "warmup_epochs": args.warmup_epochs,
            "ramp_epochs": args.ramp_epochs,
        }, ck_path_best)
        print(f"Best-epoch checkpoint saved to {ck_path_best} "
              f"(epoch {best_info['epoch']}, val_rmse_m={best_info['val_rmse_m']:.3f})")

    csv_path = out_dir / "tier2_sionna_history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(f"History saved to {csv_path}")


if __name__ == "__main__":
    main()
