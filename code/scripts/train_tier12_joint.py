"""Tier 1 + Tier 2 — joint FOMAML + adversarial disentanglement training.

This is the "macro + micro" combined training pipeline. It runs in three
phases, each producing its own checkpoint so the script can be stopped
and resumed between phases.

The motivation (per discussion in 2026-07-08): the three tiers should
not be trained in strict isolation, because:
  - Tier 1's MAML outer loop alone is macro-aware but micro-blind.
    It adapts the encoder to new envs but doesn't disentangle the
    micro signal.
  - Tier 2's adversarial disentanglement alone has nothing to push
    against unless the encoder is also being meta-learned across envs.
  - Naive joint training (sum all losses, single optimizer) leads to
    scale collapse, adversarial oscillation, and head-vs-encoder tug
    of war.

The fix: staged unfreeze. Phase A trains just the location head on
top of a frozen SSL encoder. Phase B freezes the encoder and trains
all the new modules (h_phi, FiLM, loc head, nlos head, cond head,
adversary) with the full disentanglement loss. Phase C unfreezes
the encoder and runs the joint training with a curriculum on the
disentanglement pressure.

Phases:
  A. Tier 1 FOMAML warmup. Encoder frozen at SSL init. Location head
     trained via MAML inner loop. 30-50 episodes. No disentanglement.
  B. Tier 2 disentangle, encoder frozen. DisentangledTrainer built on
     top of the Phase A encoder. Featurizer + adversary trained with
     full loss (task + nlos + cond + adv) and a curriculum on lam.
     ~30-50 epochs.
  C. Joint FOMAML + disentangle. Encoder unfrozen. MAML outer loss +
     full disentanglement loss flow through the encoder. Adversary
     and featurizer are two optimizers. ~50-100 episodes.

Each phase saves its own checkpoint; later phases load the earlier
ones. The script can be run for any single phase via --phase A|B|C
(default: A then B then C).

Usage:
  # Full pipeline (A -> B -> C)
  python3.11 -m scripts.train_tier12_joint

  # Just phase C, loading existing A and B checkpoints
  python3.11 -m scripts.train_tier12_joint --phase C \\
      --phase_a_ckpt checkpoints/joint/phase_A.pt \\
      --phase_b_ckpt checkpoints/joint/phase_B.pt

  # Quick smoke test
  python3.11 -m scripts.train_tier12_joint --quick
"""
from __future__ import annotations
import argparse
import copy
import csv
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.func import functional_call
from torch.utils.data import DataLoader

from ultraclear.config import Config, make_default
from ultraclear.data import build_synthetic_dataset, TensorUWB
from ultraclear.tier0_ssl import SSLPretrainer, ssl_loss
from ultraclear.asta import ASTAEncoder
from ultraclear.tier2_micro import DisentangledModel, disentangle_loss
from ultraclear.tier3_heads import AttentionUWB

from scripts.train_tier2_synthetic import (
    DisentangledTrainer,
    AttentionDisentangledTrainer,
    LocationHead,
    LocationHeadPerTimestep,
    linear_probe_acc,
)


def resolve_device(requested: str) -> str:
    """Resolve a device string to an actually-available device.

    Accepts 'cpu', 'mps', 'cuda', 'cuda:N', 'auto'. With 'auto' picks
    CUDA > MPS > CPU based on availability. Falls back to CPU if the
    requested accelerator is unavailable.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"  WARNING: {requested} requested but CUDA is not available, falling back to CPU")
        return "cpu"
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        print(f"  WARNING: MPS requested but not available, falling back to CPU")
        return "cpu"
    return requested


# ----------------------------------------------------------------------------
# Phase A: MAML warmup with frozen SSL encoder
# ----------------------------------------------------------------------------
def run_phase_A(
    cfg: Config,
    items: list,
    device: str,
    n_episodes: int = 40,
    ssl_epochs: int = 20,
) -> tuple[ASTAEncoder, LocationHead, list[float]]:
    """Train SSL encoder from scratch, then run a few FOMAML episodes with
    the encoder frozen. Returns (encoder, loc_head, maml_losses).

    The point of this phase is to get the location head calibrated
    *before* the disentanglement pressure arrives. We don't need many
    FOMAML episodes here -- just enough to get the head into a
    reasonable basin. The encoder's MAML init comes from the SSL
    pretraining, not from the FOMAML outer loop yet.
    """
    print("  Phase A: SSL pretraining...")
    ssl = SSLPretrainer(cfg).to(device)
    opt = torch.optim.Adam(ssl.parameters(), lr=cfg.ssl_lr)
    loader = DataLoader(TensorUWB(items), batch_size=cfg.ssl_batch_size, shuffle=True)
    ssl.train()
    for ep in range(ssl_epochs):
        total, n = 0.0, 0
        for X, _, _, _, _ in loader:
            X = X.to(device)
            opt.zero_grad()
            x_pred, mask, _x = ssl(X)
            loss = ssl_loss(x_pred, X, mask)
            loss.backward()
            opt.step()
            total += loss.item() * X.size(0)
            n += X.size(0)
        if (ep + 1) % max(1, ssl_epochs // 3) == 0 or ep == 0:
            print(f"    [ssl] ep {ep+1}/{ssl_epochs}  loss = {total/n:.4f}")

    encoder = ASTAEncoder(cfg).to(device)
    encoder.load_state_dict(ssl.encoder.state_dict())
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    head = LocationHeadPerTimestep(cfg.d_model).to(device)
    print(f"  Phase A: {n_episodes} FOMAML episodes with FROZEN encoder...")
    losses = _fomaml_loop(
        encoder, head, items, cfg, device, n_episodes,
        log_every=max(1, n_episodes // 5),
    )
    return encoder, head, losses


def run_phase_A_from_tier0(
    cfg: Config,
    tier0_ckpt_path: str,
    items: list,
    device: str,
    n_episodes: int = 40,
) -> tuple[ASTAEncoder, LocationHead, list[float]]:
    """Phase A variant: load a pre-trained SSL encoder from a Tier 0 ckpt
    (skipping the in-loop SSL pretraining), then run the FOMAML warmup.

    Use this when you already have a saved SSL encoder (e.g. from
    `train_tier0_sionna_multiscene.py` or `train_tier0_sionna.py`) and
    want to feed it directly into the joint Tier 1+2 trainer as Phase A.

    The Tier 0 ckpt format is expected to be a dict with:
      - 'model_state': a state_dict for SSLPretrainer (encoder + head)
      - 'cfg': the config used during SSL pretraining
    We extract the encoder.* weights and load them into a fresh
    ASTAEncoder (the SSL head is discarded; the joint trainer builds
    its own heads).

    **Important:** the cfg's `n_channels` and `channel_weights` are
    updated from the ckpt's stored config before building the encoder,
    so the encoder's input projection matches the F=2 (ToA + AoA)
    channels of the saved weights. This is a *side effect on cfg*;
    the caller's cfg object is modified in place.

    Returns (encoder, loc_head, maml_losses), same contract as
    run_phase_A.
    """
    print(f"  Phase A: loading Tier 0 SSL encoder from {tier0_ckpt_path}")
    ckpt = torch.load(tier0_ckpt_path, map_location=device, weights_only=False)
    ssl_state = ckpt.get("model_state", ckpt.get("encoder_state"))
    if ssl_state is None:
        raise RuntimeError(
            f"Tier 0 ckpt at {tier0_ckpt_path} has no 'model_state' or "
            f"'encoder_state' key; got keys: {list(ckpt.keys())}"
        )
    # Update cfg in place to match the ckpt's training config. This is
    # the only way to get the encoder's input projection to match the
    # F=2 / channel_weights from the SSL pretraining.
    if "cfg" in ckpt:
        for k, v in ckpt["cfg"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        print(f"  [cfg] updated to match ckpt: n_channels={cfg.n_channels}, "
              f"channel_weights={list(cfg.channel_weights)}, "
              f"mask_ratio={cfg.mask_ratio}")
    # Build a temporary SSLPretrainer (with the updated cfg) just to load
    # the encoder weights. The head is discarded; we only need the encoder.* keys.
    ssl_tmp = SSLPretrainer(cfg).to(device)
    try:
        ssl_tmp.load_state_dict(ssl_state, strict=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Tier 0 ckpt: {e}\n"
            f"  cfg: n_channels={cfg.n_channels}, d_model={cfg.d_model}\n"
            f"  expected keys: encoder.input_proj.*, encoder.tcn_in.*, "
            f"encoder.blocks.*\n"
            f"  got keys: {[k for k in ssl_state.keys()][:5]}..."
        )
    encoder = ASTAEncoder(cfg).to(device)
    # Load only the encoder.* keys; SSL head is discarded.
    encoder_state = {
        k.replace("encoder.", "", 1): v
        for k, v in ssl_state.items()
        if k.startswith("encoder.")
    }
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        print(f"  [warn] encoder load: missing={missing}, unexpected={unexpected}")
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    head = LocationHeadPerTimestep(cfg.d_model).to(device)
    print(f"  Phase A: {n_episodes} FOMAML episodes with FROZEN encoder (from Tier 0 ckpt)...")
    losses = _fomaml_loop(
        encoder, head, items, cfg, device, n_episodes,
        log_every=max(1, n_episodes // 5),
    )
    return encoder, head, losses


def _fomaml_loop(
    encoder: ASTAEncoder,
    head: LocationHead,
    items: list,
    cfg: Config,
    device: str,
    n_episodes: int,
    log_every: int = 5,
) -> list[float]:
    """FOMAML inner loop. The encoder may be frozen (Phase A) or trainable
    (Phase C); the head is always trainable.

    For Phase A the encoder has requires_grad=False, so the meta-gradient
    only updates the head. For Phase C the encoder participates in the
    meta-gradient.
    """
    opt = torch.optim.Adam(
        [p for p in encoder.parameters() if p.requires_grad] + list(head.parameters()),
        lr=cfg.maml_outer_lr,
    )
    rng = np.random.default_rng(cfg.seed)
    env_ids = sorted({it[2] for it in items})
    losses: list[float] = []
    encoder.train() if any(p.requires_grad for p in encoder.parameters()) else encoder.eval()
    head.train()
    t0 = time.time()
    for ep in range(n_episodes):
        opt.zero_grad()
        env_sample = list(rng.choice(
            env_ids, size=min(cfg.n_way, len(env_ids)), replace=False
        ))
        total_q = 0.0
        for eid in env_sample:
            sx, sy = _sample_task(items, int(eid), cfg.k_shot, cfg.seed + ep, rng)
            qx, qy = _sample_task(items, int(eid), cfg.q_query, cfg.seed + ep + 1, rng)
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)
            q_loss, _ = _adapt_and_eval(
                encoder, head, sx, sy, qx, qy,
                inner_lr=cfg.maml_inner_lr, inner_steps=cfg.maml_inner_steps,
            )
            total_q = total_q + q_loss
        total_q = total_q / max(1, len(env_sample))
        total_q.backward()
        opt.step()
        losses.append(float(total_q.item()))
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"    [maml] ep {ep+1:3d}/{n_episodes}  query_loss = {total_q.item():.4f}  "
                  f"({time.time()-t0:.1f}s)")
    return losses


def _sample_task(items, env_id, n, seed, rng):
    """Sample n (X, y) pairs from one env. Returns (X, y) tensors."""
    pool = [it for it in items if it[2] == env_id]
    if len(pool) < n:
        idx = rng.integers(0, len(pool), size=n).tolist()
    else:
        idx = rng.permutation(len(pool))[:n].tolist()
    X = torch.from_numpy(np.stack([pool[i][0] for i in idx])).float()
    y = torch.from_numpy(np.stack([pool[i][1] for i in idx])).float()
    return X, y


def _adapt_and_eval(encoder, head, sx, sy, qx, qy, inner_lr, inner_steps):
    """FOMAML inner loop using functional_call. Returns (query_loss, info)."""
    # Wrap encoder + head in a single "core" for functional_call
    class Core(nn.Module):
        def __init__(self, enc, h):
            super().__init__()
            self.encoder = enc
            self.head = h
        def forward(self, x):
            return self.head(self.encoder(x))
    core = Core(encoder, head)
    params = dict(core.named_parameters())
    names = list(params.keys())
    fast = {n: p.detach().clone().requires_grad_(True) for n, p in params.items()}

    for _ in range(inner_steps):
        pred = functional_call(core, fast, (sx,))
        loss = TF.mse_loss(pred, sy)
        grads = torch.autograd.grad(
            loss, list(fast.values()), create_graph=False, allow_unused=True
        )
        for n, g in zip(names, grads):
            if g is None:
                continue
            fast[n] = fast[n] - inner_lr * g

    q_pred = functional_call(core, fast, (qx,))
    q_loss = TF.mse_loss(q_pred, qy)
    return q_loss, {"q_pred": q_pred.detach()}


# ----------------------------------------------------------------------------
# Phase B: Disentangle with frozen encoder
# ----------------------------------------------------------------------------
def run_phase_B(
    cfg: Config,
    encoder: ASTAEncoder,
    phase_a_head: LocationHead | None,
    items: list,
    n_envs: int,
    n_conds: int,
    device: str,
    n_epochs: int = 40,
    warmup_epochs: int = 5,
    model_kind: str = "downstream",
    ramp_epochs: int = 10,
) -> tuple[DisentangledTrainer, list[dict]]:
    """Train h_phi + FiLM + heads + adversary with the encoder frozen.

    The encoder is frozen at the Phase A weights. All the new modules
    are trained with the full disentanglement loss (task + nlos +
    cond + adv) and a curriculum on lam (warmup -> ramp -> 1.0).

    If phase_a_head is provided, we initialize the new loc_head from
    it (so the disentangle phase starts from a calibrated location
    head rather than random).
    """
    if model_kind == "attention":
        # Attention model has no Phase A (no compatible SSL ckpt), so
        # the encoder is randomly initialized. Just build the model.
        model = AttentionDisentangledTrainer(cfg, n_envs=n_envs, n_conds=n_conds).to(device)
        return model, [], []
    model = DisentangledTrainer(cfg, n_envs=n_envs, n_conds=n_conds).to(device)
    model.disent.encoder.load_state_dict(encoder.state_dict())
    for p in model.disent.encoder.parameters():
        p.requires_grad_(False)

    if phase_a_head is not None:
        # Copy Phase A's location head weights into the new model
        model.loc_head.load_state_dict(phase_a_head.state_dict())
        print("  Initialized loc_head from Phase A.")

    feat_params = (
        list(model.disent.h_phi.parameters())
        + list(model.disent.film.parameters())
        + list(model.loc_head.parameters())
        + list(model.nlos_mlp.parameters())
        + list(model.cond_head.parameters())
    )
    adv_params = list(model.disent.adv.parameters())
    opt_feat = torch.optim.Adam(feat_params, lr=cfg.finetune_lr)
    opt_adv = torch.optim.Adam(adv_params, lr=cfg.adv_lr)

    train_loader = DataLoader(
        TensorUWB(items), batch_size=cfg.finetune_batch_size, shuffle=True
    )

    history = []

    def compute_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return 0.0
        if ep < warmup_epochs + ramp_epochs:
            return (ep - warmup_epochs) / ramp_epochs
        return 1.0

    print(f"  Phase B: {n_epochs} epochs (warmup={warmup_epochs}, ramp={ramp_epochs}), "
          f"encoder FROZEN")
    t0 = time.time()
    for ep in range(n_epochs):
        lam = compute_lambda(ep)
        model.train()
        total_task, total_adv, total_nlos, total_cond, n = 0.0, 0.0, 0.0, 0.0, 0
        for X, y, env, cond, nlos in train_loader:
            X = X.to(device); y = y.to(device); env = env.to(device)
            cond = cond.to(device); nlos = nlos.to(device)
            h_prime, c, adv_logits, loc, nlos_logits, cond_logits = model(
                X, detach_c_for_adv=False
            )
            task_loss = TF.mse_loss(loc, y)
            adv_ce = TF.cross_entropy(adv_logits, env)
            # Bounded disentangle surrogate (DANN-style). We push the encoder
            # to make the adversary worse than chance, but the gradient is
            # *capped* at +chance so it can never dominate the task loss.
            # The old 7/8 form was `clamp(adv_ce - 2*chance, min=0)` which
            # is unbounded from above -- adv_ce=200 contributed 194 to the
            # loss, blew up the encoder, and Phase C went to NaN.
            adv_chance = math.log(n_envs)
            # Bounded surrogate: ramps from 0 at adv_ce=chance to chance
            # at adv_ce=2*chance, then saturates. Gradient w.r.t. encoder
            # is bounded by ~1.0 so the disentangle term can never
            # dominate the task loss (which is O(0.1) on stationary data).
            adv_term = (
                torch.clamp(adv_ce - adv_chance, max=adv_chance) - adv_chance
            )
            nlos_loss = TF.binary_cross_entropy_with_logits(nlos_logits, nlos.float())
            # cond_loss is only meaningful when we have real cond labels
            # (synthetic data has 3-class cond; Sionna real data uses
            # cond_id = -1 to signal "no label"). Skip the cond loss
            # entirely when no labeled cond is present.
            if (cond >= 0).all():
                cond_loss = TF.cross_entropy(cond_logits, cond)
            else:
                cond_loss = torch.zeros((), device=device)
            feat_loss = (
                task_loss
                + 0.5 * nlos_loss
                + 2.0 * cond_loss
                + lam * adv_term
            )
            opt_feat.zero_grad()
            feat_loss.backward()
            opt_feat.step()
            _h, _c, adv_logits2, _loc, _nlos, _cond2 = model(X, detach_c_for_adv=True)
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
        history.append({
            "epoch": ep + 1,
            "lambda": lam,
            "train_task_mse": train_task,
            "train_adv_ce": train_adv,
            "train_nlos_ce": train_nlos,
            "train_cond_ce": train_cond,
        })
        if (ep + 1) % max(1, n_epochs // 10) == 0 or ep == 0:
            print(f"    [phaseB] ep {ep+1:3d}/{n_epochs}  lam={lam:.2f}  "
                  f"task={train_task:.3f}  adv={train_adv:.3f}  "
                  f"nlos={train_nlos:.3f}  cond={train_cond:.3f}  "
                  f"({time.time()-t0:.1f}s)")
    print(f"  Phase B done in {time.time()-t0:.1f}s")
    return model, history


# ----------------------------------------------------------------------------
# Phase C: Joint FOMAML + disentangle (encoder trainable)
# ----------------------------------------------------------------------------
def _attention_supervised_loop(
    model, items, cfg, n_episodes,
    warmup_episodes, ramp_episodes, val_items, eval_fn, device,
    n_envs,
):
    """Plain supervised training loop for the AttentionDisentangledTrainer.

    Drops the FOMAML inner loop (which is an ASTA-specific design that
    doesn't fit the per-anchor pos head) but keeps the rest of the
    disentangled objective: task (loc) + cond (auxiliary) + adv
    (adversary on c) + nlos (per-anchor). Curriculum on lam (adv weight)
    follows the same warmup/ramp/constant schedule as Phase C.
    """
    from ultraclear.tier2_micro import disentangle_loss

    def compute_lambda(ep: int) -> float:
        if ep < warmup_episodes:
            return 0.0
        if ep < warmup_episodes + ramp_episodes:
            return (ep - warmup_episodes) / ramp_episodes
        return 1.0

    # Single optimizer over all params; we just adjust the adv loss
    # weight via lambda.
    opt = torch.optim.Adam(model.parameters(), lr=cfg.finetune_lr)
    rng = np.random.default_rng(cfg.seed + 999)
    env_ids = sorted({it[2] for it in items})
    history = []

    print(f"  Attention Phase C: {n_episodes} episodes (warmup={warmup_episodes}, "
          f"ramp={ramp_episodes})")
    t0 = time.time()
    for ep in range(n_episodes):
        lam = compute_lambda(ep)
        opt.zero_grad()
        # Sample a batch (env-mixed) from items
        batch_size = min(32, len(items))
        idx = rng.choice(len(items), size=batch_size, replace=False)
        X = torch.from_numpy(np.stack([items[i][0] for i in idx])).float().to(device)
        y = torch.from_numpy(np.stack([items[i][1] for i in idx])).float().to(device)
        env = torch.tensor([items[i][2] for i in idx], dtype=torch.long).to(device)
        nlos = torch.from_numpy(np.stack([items[i][4] for i in idx])).bool().to(device)
        # Forward
        _, c, adv_logits, loc, nlos_logits, cond_logits = model(X)
        # Loc loss (mean over batch; y is (B, T, 2) for the joint
        # trainer's items, take the middle timestep)
        T = X.size(1)
        y_mid = y[:, T // 2, :]
        loc_loss = TF.mse_loss(loc, y_mid)
        # nlos loss (per-cell BCE)
        nlos_loss = TF.binary_cross_entropy_with_logits(nlos_logits, nlos.float())
        # cond loss: c is supposed to be informative about the
        # auxiliary cond_id. items have -1 for real data; skip in that
        # case.
        cond = torch.tensor([items[i][3] for i in idx], dtype=torch.long).to(device)
        if (cond >= 0).any():
            cond_loss = TF.cross_entropy(cond_logits, cond.clamp(min=0))
        else:
            cond_loss = torch.tensor(0.0, device=device)
        # adv loss: adversary tries to predict env from c
        adv_loss = TF.cross_entropy(adv_logits, env)
        # Total = task + 0.5*cond + lam*adv (the disentangle_loss
        # helper expects task_loss as the task; the + sign is correct
        # because grad reversal is inside the model)
        total = loc_loss + 0.5 * cond_loss + lam * adv_loss + 0.5 * nlos_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (ep + 1) % max(1, n_episodes // 20) == 0 or ep == 0:
            print(f"    ep {ep+1:3d}/{n_episodes}  loc={loc_loss.item():.4f}  "
                  f"nlos={nlos_loss.item():.4f}  adv={adv_loss.item():.4f}  "
                  f"lam={lam:.2f}")

    # Final eval
    hist = {"final_loc_loss": loc_loss.item(),
            "final_nlos_loss": nlos_loss.item(),
            "final_adv_loss": adv_loss.item(),
            "n_episodes": n_episodes, "elapsed_s": time.time() - t0}
    if eval_fn is not None and val_items is not None:
        try:
            per_cond = eval_fn(model, val_items, device)
            hist["per_cond"] = per_cond
        except Exception as e:
            print(f"  eval failed: {e}")
    return model, [hist], []


# ----------------------------------------------------------------------------
def run_phase_C(
    cfg: Config,
    model: DisentangledTrainer,
    items: list,
    n_envs: int,
    device: str,
    n_episodes: int = 60,
    warmup_episodes: int = 10,
    ramp_episodes: int = 20,
    save_every: int = 0,
    val_items: list = None,
    eval_fn = None,
    freeze_episodes: int = 0,
    unfreeze_lr_scale: float = 1.0,
) -> tuple[DisentangledTrainer, list[dict], list[dict]]:
    """Joint training with progressive unfreezing.

    Stage 1 (ep in [0, freeze_episodes)): encoder FROZEN. Only
    h_phi/FiLM/heads/adversary train. No FOMAML inner loop (the
    encoder isn't getting adapted anyway). This is the "Phase B
    continued" stage — it lets the new modules settle into a
    reasonable basin before the encoder is allowed to move.

    Stage 2 (ep in [freeze_episodes, n_episodes)): encoder UNFROZEN
    at LR = cfg.finetune_lr * unfreeze_lr_scale. Heads + FiLM train
    at the full cfg.finetune_lr. MAML inner loop is enabled. This is
    the full joint training that previously destroyed the FOMAML
    features (per the 7/10 memory, meta loss rose 0.24 -> 0.37 with
    encoder+head at the same LR).

    Curriculum on lam (adv weight) spans the WHOLE Phase C, not just
    stage 2, so the disentangle pressure is already at its target
    value when the encoder unfreezes. (Stage 1 trains the adversary
    alongside the heads, so the adversary is calibrated by the time
    the encoder comes online.)

    Args:
      freeze_episodes: how many episodes to keep the encoder frozen
        at the start of Phase C. 0 = old behavior (immediate
        unfreeze). Default 20.
      unfreeze_lr_scale: multiplier on cfg.finetune_lr for the
        encoder params during stage 2. 0.1 = 10x lower than heads.
        Default 0.1.
    """
    # ---- Attention model: skip the FOMAML inner-loop (maml_head
    # state-dict shapes don't match) and just do plain supervised
    # training of the disentangled objective (task + cond + adv + nlos).
    # The whole point of the joint trainer for the attention model is
    # to test whether FiLM + adversary help on top of the simpler
    # transformer backbone, not the MAML inner loop (which is the ASTA
    # model's design).
    if isinstance(model, AttentionDisentangledTrainer):
        print("  --model attention: skipping FOMAML inner loop (maml_head "
              "shapes don't match). Running plain supervised training of "
              "the disentangled objective.")
        return _attention_supervised_loop(
            model, items, cfg, n_episodes,
            warmup_episodes, ramp_episodes, val_items, eval_fn, device,
            n_envs,
        )

    # ---- Progressive-unfreeze setup ----
    # Stage 1 (ep < freeze_episodes): encoder FROZEN. Only the new
    # modules (h_phi, FiLM, heads, adversary) train. No MAML inner loop
    # because adapting maml_head to a frozen encoder is wasted compute.
    # Stage 2 (ep >= freeze_episodes): encoder UNFROZEN at
    # cfg.finetune_lr * unfreeze_lr_scale. Heads + FiLM keep the full
    # cfg.finetune_lr. MAML inner loop is enabled.

    if freeze_episodes > 0:
        for p in model.disent.encoder.parameters():
            p.requires_grad_(False)
        model.disent.encoder.eval()
        print(f"  Phase C progressive-unfreeze: {freeze_episodes} frozen-encoder "
              f"episodes first, then unfreeze at LR*{unfreeze_lr_scale}")
    else:
        for p in model.disent.encoder.parameters():
            p.requires_grad_(True)
        model.disent.encoder.train()
        print(f"  Phase C: encoder UNFROZEN from ep 1 (legacy behavior)")

    # Build a separate "core" (encoder + loc head) for the MAML inner loop.
    # Note: this is a DIFFERENT LocationHead from model's loc_head -- it's
    # a fresh copy of the head for the MAML outer loop. The model's loc_head
    # continues to be the disentangled version used for inference.
    # Why: the FOMAML inner loop wants to adapt a head to a specific env;
    # using the disentangled head here would mean adapting a head that
    # was trained to ignore macro info, which defeats the purpose.
    maml_head = LocationHeadPerTimestep(cfg.d_model).to(device)
    maml_head.load_state_dict(model.loc_head.state_dict())

    def compute_lambda(ep: int) -> float:
        if ep < warmup_episodes:
            return 0.0
        if ep < warmup_episodes + ramp_episodes:
            return (ep - warmup_episodes) / ramp_episodes
        return 1.0

    # Two parameter groups in the featurizer optimizer:
    #   - encoder params: stage 1 excluded (requires_grad=False),
    #     stage 2 trained at cfg.finetune_lr * unfreeze_lr_scale
    #   - everything else: full cfg.finetune_lr, both stages
    head_params = (
        list(model.disent.h_phi.parameters())
        + list(model.disent.film.parameters())
        + list(model.loc_head.parameters())
        + list(model.nlos_mlp.parameters())
        + list(model.cond_head.parameters())
        + list(maml_head.parameters())
    )
    encoder_params = list(model.disent.encoder.parameters())
    # Optimizer 1: featurizer side. We use param_groups so the encoder
    # gets a different LR from the heads during stage 2. During stage 1
    # the encoder params are excluded from this optimizer (their
    # requires_grad is False, so Adam just won't update them anyway).
    feat_param_groups = [
        {"params": head_params, "lr": cfg.finetune_lr},
    ]
    if any(p.requires_grad for p in encoder_params):
        feat_param_groups.append({
            "params": encoder_params,
            "lr": cfg.finetune_lr * unfreeze_lr_scale,
        })
    opt_feat = torch.optim.Adam(feat_param_groups)
    # Optimizer 2: adversary only
    adv_params = list(model.disent.adv.parameters())
    opt_adv = torch.optim.Adam(adv_params, lr=cfg.adv_lr)

    rng = np.random.default_rng(cfg.seed + 999)  # different stream from A
    env_ids = sorted({it[2] for it in items})
    history = []

    print(f"  Phase C: {n_episodes} episodes (warmup={warmup_episodes}, "
          f"ramp={ramp_episodes})"
          f"{f', saving every {save_every} eps' if save_every > 0 else ''}")
    t0 = time.time()
    snapshots = []  # list of (ep, state_dict, per_cond_or_None)
    for ep in range(n_episodes):
        # ---- Stage transition: at freeze_episodes, unfreeze encoder ----
        if freeze_episodes > 0 and ep == freeze_episodes:
            for p in model.disent.encoder.parameters():
                p.requires_grad_(True)
            model.disent.encoder.train()
            # Adam tracks param groups; since encoder_params was already
            # in opt_feat as a param group (just with requires_grad
            # False), all we need is to flip the flag. Adam will start
            # updating on the next step.
            print(f"    [phaseC] >>> unfreezing encoder at ep {ep+1} "
                  f"(LR={cfg.finetune_lr * unfreeze_lr_scale:.1e})")

        lam = compute_lambda(ep)
        opt_feat.zero_grad()
        encoder_trainable = any(p.requires_grad for p in model.disent.encoder.parameters())

        # ---- MAML inner loop (only when encoder is trainable) ----
        meta_loss = torch.tensor(0.0, device=device)
        all_query_x, all_query_y, all_query_env, all_query_cond, all_query_nlos = [], [], [], [], []
        if encoder_trainable:
            env_sample = list(rng.choice(
                env_ids, size=min(cfg.n_way, len(env_ids)), replace=False
            ))
            for eid in env_sample:
                sx, sy = _sample_task(items, int(eid), cfg.k_shot, cfg.seed + ep, rng)
                qx, qy = _sample_task(items, int(eid), cfg.q_query, cfg.seed + ep + 1, rng)
                # also pull labels for disentangle on query
                q_pool = [it for it in items if it[2] == int(eid)]
                q_idx = rng.permutation(len(q_pool))[:cfg.q_query].tolist()
                q_env = torch.tensor([q_pool[i][2] for i in q_idx], dtype=torch.long)
                q_cond = torch.tensor([q_pool[i][3] for i in q_idx], dtype=torch.long)
                q_nlos = torch.from_numpy(np.stack([q_pool[i][4] for i in q_idx])).bool()

                sx, sy = sx.to(device), sy.to(device)
                qx, qy = qx.to(device), qy.to(device)
                q_env, q_cond, q_nlos = q_env.to(device), q_cond.to(device), q_nlos.to(device)

                q_loss, _ = _adapt_and_eval(
                    model.disent.encoder, maml_head, sx, sy, qx, qy,
                    inner_lr=cfg.maml_inner_lr, inner_steps=cfg.maml_inner_steps,
                )
                meta_loss = meta_loss + q_loss

                all_query_x.append(qx)
                all_query_y.append(qy)
                all_query_env.append(q_env)
                all_query_cond.append(q_cond)
                all_query_nlos.append(q_nlos)
            meta_loss = meta_loss / max(1, len(env_sample))

        # ---- Disentangle pass on the query set (using the CURRENT encoder,
        #      not the MAML-adapted one -- MAML adapts the encoder to a
        #      specific env, but disentanglement should be on the
        #      shared representation, not the env-specific one) ----
        # In stage 1 (encoder frozen), we just sample a single batch of
        # train items since the MAML inner loop is skipped.
        if encoder_trainable:
            qx_all = torch.cat(all_query_x, dim=0)
            qy_all = torch.cat(all_query_y, dim=0)
            qenv_all = torch.cat(all_query_env, dim=0)
            qcond_all = torch.cat(all_query_cond, dim=0)
            qnlos_all = torch.cat(all_query_nlos, dim=0)
        else:
            # Stage 1: just grab a random batch from the train pool.
            # Keep the batch size comparable to one env's query set so
            # the per-batch losses are on the same scale as stage 2.
            n = min(cfg.q_query, len(items))
            qidx = rng.choice(len(items), size=n, replace=False)
            qx_all = torch.from_numpy(np.stack([items[i][0] for i in qidx])).float().to(device)
            qy_all = torch.from_numpy(np.stack([items[i][1] for i in qidx])).float().to(device)
            qenv_all = torch.tensor([items[i][2] for i in qidx], dtype=torch.long).to(device)
            qcond_all = torch.tensor([items[i][3] for i in qidx], dtype=torch.long).to(device)
            qnlos_all = torch.from_numpy(np.stack([items[i][4] for i in qidx])).bool().to(device)

        # Forward through the model (NOT the adapted one) for disentangle
        h_prime, c, adv_logits, loc, nlos_logits, cond_logits = model(
            qx_all, detach_c_for_adv=False
        )
        task_loss_d = TF.mse_loss(loc, qy_all)
        adv_ce = TF.cross_entropy(adv_logits, qenv_all)
        # Capped adv loss (DANN): only disentangle above 2*chance. Prevents
        # the adv_ce explosion on real data with 21+ envs.
        adv_chance = math.log(n_envs)
        # Bounded disentangle surrogate (DANN-style). See Phase B for the
        # full rationale -- the old `clamp(adv_ce - 2*chance, min=0)` was
        # unbounded from above and exploded Phase C at ~ep 40 with 21 envs.
        adv_term = (
            torch.clamp(adv_ce - adv_chance, max=adv_chance) - adv_chance
        )
        nlos_loss = TF.binary_cross_entropy_with_logits(nlos_logits, qnlos_all.float())
        # cond_loss skipped when no labeled cond (e.g. Sionna real data)
        if (qcond_all >= 0).all():
            cond_loss = TF.cross_entropy(cond_logits, qcond_all)
        else:
            cond_loss = torch.zeros((), device=device)
        # In stage 1, no meta_loss (encoder frozen, no FOMAML). The
        # disentangle pass is still useful -- it lets the new modules
        # and the adversary calibrate against the frozen encoder.
        feat_loss = (
            meta_loss
            + task_loss_d
            + 0.5 * nlos_loss
            + 2.0 * cond_loss
            + lam * adv_term
        )
        feat_loss.backward()
        opt_feat.step()

        # ---- Adversary update on detached c ----
        _h, _c, adv_logits2, _loc, _nlos, _cond2 = model(qx_all, detach_c_for_adv=True)
        adv_loss_for_adv = TF.cross_entropy(adv_logits2, qenv_all)
        opt_adv.zero_grad()
        adv_loss_for_adv.backward()
        opt_adv.step()

        # ---- Linear probe on c (diagnostic) ----
        # We need to forward the model under no_grad to extract c as
        # numpy (the probe should not backprop into the encoder), but
        # the probe training itself needs grad enabled.
        with torch.no_grad():
            _h, c_val, _adv, _loc, _nlos, _cond = model(qx_all)
            c_np = c_val.detach().cpu().numpy()
            env_np = qenv_all.detach().cpu().numpy()
        split = len(c_np) // 2
        if split > 0 and len(np.unique(env_np[:split])) >= 2:
            idx = np.random.default_rng(ep).permutation(len(c_np))
            probe = linear_probe_acc(
                c_np[idx[:split]], env_np[idx[:split]],
                c_np[idx[split:]], env_np[idx[split:]],
                n_envs, device, epochs=10,
            )
        else:
            probe = float("nan")

        history.append({
            "episode": ep + 1,
            "stage": "frozen" if not encoder_trainable else "unfrozen",
            "lambda": lam,
            "meta_loss": float(meta_loss.item()),
            "task_loss_disent": float(task_loss_d.item()),
            "adv_ce": float(adv_ce.item()),
            "nlos_ce": float(nlos_loss.item()),
            "cond_ce": float(cond_loss.item()),
            "feat_loss": float(feat_loss.item()),
            "probe_env_acc": probe,
        })
        if (ep + 1) % max(1, n_episodes // 10) == 0 or ep == 0:
            stage_tag = "F" if not encoder_trainable else "U"
            print(f"    [phaseC] ep {ep+1:3d}/{n_episodes} [{stage_tag}]  lam={lam:.2f}  "
                  f"meta={meta_loss.item():.3f}  task_d={task_loss_d.item():.3f}  "
                  f"adv={adv_ce.item():.3f}  cond={cond_loss.item():.3f}  "
                  f"probe={probe:.3f}  ({time.time()-t0:.1f}s)")

        # Snapshot the model state_dict at chosen episodes. Skip ep 0
        # since the encoder hasn't moved yet. Caller passes val_items
        # + eval_fn if they want a per-cond score attached to the snapshot.
        if save_every > 0 and (ep + 1) % save_every == 0 and (ep + 1) < n_episodes:
            sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            per_cond_snap = None
            if val_items is not None and eval_fn is not None:
                per_cond_snap = eval_fn(model, val_items, device)
            snapshots.append({"ep": ep + 1, "state_dict": sd, "per_cond": per_cond_snap})
            print(f"    [phaseC]   snapshot at ep {ep+1}: per_cond={per_cond_snap}")

    # Sync: copy MAML-trained loc_head weights back into the model's loc_head
    # so inference uses the best one. Not strictly required (model's
    # loc_head is already disentangled and used for inference), but useful
    # for downstream Tier 3 attachment.
    model.loc_head.load_state_dict(maml_head.state_dict())
    print(f"  Phase C done in {time.time()-t0:.1f}s")
    return model, history, snapshots


# ----------------------------------------------------------------------------
# Per-cond evaluation
# ----------------------------------------------------------------------------
def eval_per_cond(model: DisentangledTrainer, items: list, device: str,
                  denorm_info: dict | None = None) -> dict:
    model.eval()
    loader = DataLoader(TensorUWB(items), batch_size=64)
    errs_per_cond = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for batch in loader:
            # TensorUWB returns either 5-tuple (X, y, env, cond, nlos)
            # or 6-tuple (X, y, env, cond, nlos, restore). Accept both.
            if len(batch) == 6:
                X, y, env, cond, _nlos, _restore = batch
            else:
                X, y, env, cond, _nlos = batch
            X = X.to(device); y = y.to(device); env = env.to(device)
            _h, _c, _adv, loc, _nlos, _cond = model(X)
            # The ASTA model's loc is (B, T, 2) (per-timestep). The
            # attention model's loc is (B, 2) (per-window from the
            # middle-timestep per-anchor head). Normalize both to
            # (B, 2) using y's middle timestep before computing err.
            T = X.size(1)
            if loc.dim() == 3:
                loc_per_window = loc[:, T // 2, :]
            else:
                loc_per_window = loc
            y_per_window = y[:, T // 2, :]
            # Denormalize predictions and labels if denorm_info given
            # (Sionna data is trained in per-scene-normalized space).
            if denorm_info is not None:
                loc_orig = torch.zeros_like(loc_per_window)
                y_orig = torch.zeros_like(y_per_window)
                for i in range(len(env)):
                    eid = int(env[i].item())
                    mean, std = denorm_info[eid]
                    loc_orig[i] = loc_per_window[i] * torch.from_numpy(std).to(device) + torch.from_numpy(mean).to(device)
                    y_orig[i] = y_per_window[i] * torch.from_numpy(std).to(device) + torch.from_numpy(mean).to(device)
                loc_per_window, y_per_window = loc_orig, y_orig
            # Per-window Euclidean error: (B, 2) -> (B,)
            # over T to get one error per window, so the cond bucketing
            # sees a scalar per sample.
            err = torch.linalg.norm(loc_per_window - y_per_window, dim=-1).cpu().numpy()
            for ci, e in zip(cond.tolist(), err.tolist()):
                # cond_id == -1 means "no cond label" (real data, walking
                # path, etc.). Skip those samples rather than crashing.
                if ci < 0:
                    continue
                errs_per_cond[ci].append(e)
    return {ci: float(np.mean(es)) if es else float("nan")
            for ci, es in errs_per_cond.items()}


def eval_per_nlos_bucket(model: DisentangledTrainer, items: list, device: str,
                          denorm_info: dict | None = None) -> dict:
    """Per-NLOS-rate-bucket evaluation for the Sionna real-data path.

    For each window, compute the per-cell NLOS rate (from the `nlos`
    field), bucket into 3 classes (low < 0.3, mid < 0.7, high >= 0.7),
    and report per-bucket mean loc error. Per-timestep errors are
    averaged within each window before bucketing.
    """
    model.eval()
    loader = DataLoader(TensorUWB(items), batch_size=64)
    errs_per_bucket = {0: [], 1: [], 2: []}   # low / mid / high NLOS
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 6:
                X, y, env, _cond, nlos, _restore = batch
            else:
                X, y, env, _cond, nlos = batch
            X = X.to(device); y = y.to(device); env = env.to(device)
            _h, _c, _adv, loc, _nlos, _cond = model(X)
            if denorm_info is not None:
                loc_orig = torch.zeros_like(loc)
                y_orig = torch.zeros_like(y)
                for i in range(len(env)):
                    eid = int(env[i].item())
                    mean, std = denorm_info[eid]
                    loc_orig[i] = loc[i] * torch.from_numpy(std).to(device) + torch.from_numpy(mean).to(device)
                    y_orig[i] = y[i] * torch.from_numpy(std).to(device) + torch.from_numpy(mean).to(device)
                loc, y = loc_orig, y_orig
            # Per-timestep errors, then averaged per window.
            err = torch.linalg.norm(loc - y, dim=-1)         # (B, T)
            err_per_window = err.mean(dim=-1).cpu().numpy()  # (B,)
            nlos_rate = nlos.float().mean(dim=(1, 2)).numpy()  # (B,)
            # Bucket thresholds: 5% / 25% NLOS rate (per window).
            # These are the thresholds Justin uses to split low/mid/high
            # NLOS conditions for the per-cond val report. The old 0.3 /
            # 0.7 thresholds were way too lax for this dataset -- with
            # 13/21 envs at 100% per-cell NLOS, almost every window
            # ended up in "high" and "low" was an empty bucket.
            for rate, e in zip(nlos_rate.tolist(), err_per_window.tolist()):
                if rate < 0.05:
                    errs_per_bucket[0].append(e)
                elif rate < 0.25:
                    errs_per_bucket[1].append(e)
                else:
                    errs_per_bucket[2].append(e)
    return {bi: float(np.mean(es)) if es else float("nan")
            for bi, es in errs_per_bucket.items()}


def eval_for_data(model, items, device, data_source, denorm_info=None):
    """Route to the right eval function based on the data source.

    For data sources that have per-cell NLOS labels (sionna, walking_path),
    use the per-NLOS-bucket eval. For synthetic data (which has clean
    3-class cond labels), use the per-cond eval.
    """
    if data_source in ("sionna", "walking_path"):
        return eval_per_nlos_bucket(model, items, device, denorm_info=denorm_info)
    return eval_per_cond(model, items, device, denorm_info=denorm_info)


# ----------------------------------------------------------------------------
# Optional: train a no-Tier-2 baseline (for the 3-way comparison)
# ----------------------------------------------------------------------------
def run_baseline_for_joint(
    cfg: Config,
    train_items: list,
    val_items: list,
    device: str,
    n_epochs: int = 50,
    denorm_info: dict | None = None,
    save_ckpt_path: str | None = None,
) -> dict:
    """Train an ASTA encoder + LocationHead with plain supervised MSE.
    Returns {cond_id: mean_loc_error} on val for the 3-way comparison.
    If save_ckpt_path is provided, also saves the encoder + head to
    that path in the standard {encoder_state, head_state} format
    that viz_joint_adaptation_curve.py can load."""
    from ultraclear.asta import ASTAEncoder
    print("  Training baseline (no Tier 2) for the 3-way comparison...")
    enc = ASTAEncoder(cfg).to(device)
    head = LocationHeadPerTimestep(cfg.d_model).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    loader = DataLoader(TensorUWB(train_items), batch_size=32, shuffle=True)
    enc.train(); head.train()
    for ep in range(n_epochs):
        total, n = 0.0, 0
        for X, y, _, _, _ in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = head(enc(X))
            loss = TF.mse_loss(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * X.size(0)
            n += X.size(0)
        if (ep + 1) % max(1, n_epochs // 3) == 0 or ep == 0:
            print(f"    [baseline] ep {ep+1}/{n_epochs}  loc_mse = {total/n:.4f}")
    enc.eval(); head.eval()
    val_loader = DataLoader(TensorUWB(val_items), batch_size=64)
    # If items have cond_id = -1 (Sionna data), bucket by NLOS rate instead.
    sample_c = val_items[0][3] if val_items else 0
    use_nlos_buckets = (sample_c == -1)
    if use_nlos_buckets:
        errs_per_cond = {0: [], 1: [], 2: []}  # low / mid / high NLOS
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 6:
                    X, y, env, _c, nlos, _restore = batch
                else:
                    X, y, env, _c, nlos = batch
                X = X.to(device); y = y.to(device); env = env.to(device)
                pred = head(enc(X))
                if denorm_info is not None:
                    pred_orig = torch.zeros_like(pred)
                    y_orig = torch.zeros_like(y)
                    for i in range(len(env)):
                        eid = int(env[i].item())
                        mean, std = denorm_info[eid]
                        pred_orig[i] = pred[i] * torch.from_numpy(std).to(device) + torch.from_numpy(mean).to(device)
                        y_orig[i] = y[i] * torch.from_numpy(std).to(device) + torch.from_numpy(mean).to(device)
                    pred, y = pred_orig, y_orig
                err = torch.linalg.norm(pred - y, dim=-1).cpu().numpy()
                nlos_rate = nlos.float().mean(dim=(1, 2)).numpy()
                for rate, e in zip(nlos_rate.tolist(), err.tolist()):
                    if rate < 0.05:
                        errs_per_cond[0].append(e)
                    elif rate < 0.25:
                        errs_per_cond[1].append(e)
                    else:
                        errs_per_cond[2].append(e)
    else:
        errs_per_cond = {0: [], 1: [], 2: []}
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 6:
                    X, y, _, c, _, _restore = batch
                else:
                    X, y, _, c, _ = batch
                X, y = X.to(device), y.to(device)
                pred = head(enc(X))
                err = torch.linalg.norm(pred - y, dim=-1).cpu().numpy()
                for ci, e in zip(c.tolist(), err):
                    errs_per_cond[ci].append(e)
    if save_ckpt_path is not None:
        torch.save({
            "encoder_state": enc.state_dict(),
            "head_state": head.state_dict(),
            "cfg": cfg.__dict__,
        }, save_ckpt_path)
        print(f"  baseline ckpt saved → {save_ckpt_path}")
    return {ci: float(np.mean(es)) if es else float("nan")
            for ci, es in errs_per_cond.items()}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="ABC",
                   help="which phases to run: A, B, C, AB, ABC (default: ABC)")
    p.add_argument("--phase_a_ckpt", default=None,
                   help="load Phase A ckpt (skips A)")
    p.add_argument("--tier0_ckpt", default=None,
                   help="load a raw SSL pretraining ckpt (just the encoder) "
                        "as the Phase A encoder init. Skips the in-loop SSL "
                        "pretraining but still runs the FOMAML warmup. The "
                        "ckpt should have a 'model_state' key with encoder.* "
                        "weights, like code/checkpoints/tier0_multiscene_f2_3to1_100ep/tier0_multiscene_f2.pt.")
    p.add_argument("--phase_b_ckpt", default=None,
                   help="load Phase B ckpt (skips A and B)")
    p.add_argument("--device", default="auto",
                   help="device: 'cpu', 'mps', 'cuda', 'cuda:N', or 'auto' "
                        "(default: auto, picks CUDA > MPS > CPU)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="tiny run for smoke testing")
    p.add_argument("--out", default="checkpoints/joint")
    p.add_argument("--data", default="synthetic", choices=["synthetic", "sionna", "walking_path"],
                   help="data source: 'synthetic' uses build_synthetic_dataset (random walk + Bernoulli); "
                        "'sionna' uses the 21 real ray-tracing CSVs in ~/Downloads/srp_research/grid2/")
    p.add_argument("--sionna_csv_dir", default=None,
                   help="override the Sionna CSV directory (default: ~/Downloads/srp_research/grid2/)")
    p.add_argument("--sionna_glob", default=None,
                   help="override the default '*_A1A4swapped.csv' glob (e.g. "
                        "'*_A1A4swapped_grid.csv' for grid-converted track2 data)")
    p.add_argument("--sionna_train_envs", type=int, nargs="+", default=None,
                   help="explicit list of env_ids for the train split "
                        "(default: first 10 sorted envs -- indoor scenes, "
                        "tests indoor-to-outdoor generalization)")
    p.add_argument("--sionna_val_envs", type=int, nargs="+", default=None,
                   help="explicit list of env_ids for the val split "
                        "(default: sorted envs 10-20 except 14, which has "
                        "20% missing data and produces degenerate per-bucket "
                        "errors. Override to keep 14 if you want the missing-"
                        "data scene in the val set.)")
    p.add_argument("--walking_path_csv", default=None,
                   help="path to the walking-path CSV (default: "
                        "~/Downloads/srp_research/sionna_walking_path_v2.csv). "
                        "Used when --data walking_path.")
    p.add_argument("--walking_path_n_envs", type=int, default=4,
                   help="number of envs to split the walking path into for "
                        "FOMAML task sampling (default 4). The path is "
                        "temporally chunked; each chunk is its own env.")
    p.add_argument("--run_baseline", action="store_true",
                   help="also train a no-Tier-2 baseline (ASTA + loc head) "
                        "and merge its per-cond errors into the joint per-cond CSV "
                        "for the 3-way comparison viz")
    p.add_argument("--baseline_epochs", type=int, default=50,
                   help="number of epochs to train the baseline (default 50, "
                        "matches the standalone Tier 2 finetune_epochs)")
    # Phase A
    p.add_argument("--phase_a_episodes", type=int, default=40)
    p.add_argument("--phase_a_ssl_epochs", type=int, default=20)
    # Phase B
    p.add_argument("--phase_b_epochs", type=int, default=40)
    p.add_argument("--phase_b_warmup", type=int, default=5)
    p.add_argument("--phase_b_ramp", type=int, default=10)
    # Phase C
    p.add_argument("--phase_c_episodes", type=int, default=60)
    p.add_argument("--phase_c_save_every", type=int, default=0,
                   help="if > 0, save a Phase C state-dict snapshot every N "
                        "episodes (and skip the final, already-saved one). "
                        "Used to find the best ckpt along the trajectory "
                        "since meta loss can rise over time.")
    p.add_argument("--phase_c_warmup", type=int, default=10)
    p.add_argument("--phase_c_ramp", type=int, default=20)
    p.add_argument("--phase_c_freeze_episodes", type=int, default=0,
                   help="Progressive-unfreezing: keep encoder FROZEN for the "
                        "first N episodes of Phase C. Only h_phi/FiLM/heads/"
                        "adversary train during this period; no FOMAML inner "
                        "loop. At episode N+1, encoder is unfrozen and "
                        "trained at LR = finetune_lr * unfreeze_lr_scale. "
                        "Default 0 = old behavior (immediate unfreeze). "
                        "Try 20 to fix the encoder-drift problem "
                        "(see memory/2026-07-10.md 'Phase C trajectory').")
    p.add_argument("--phase_c_unfreeze_lr_scale", type=float, default=0.1,
                   help="Progressive-unfreezing: LR multiplier for encoder "
                        "params once they are unfrozen. 0.1 = 10x lower LR "
                        "than the heads. Default 0.1. Only takes effect "
                        "when --phase_c_freeze_episodes > 0.")
    p.add_argument("--model", default="downstream", choices=["downstream", "attention"],
                   help="Which backbone to use for the joint Tier 1+2 trainer. "
                        "'downstream' (default) = ASTAEncoder + LocationHeadPerTimestep. "
                        "'attention' = nn.TransformerEncoder + per-anchor pos head "
                        "(matches the simple attention baseline that gave "
                        "the best supervised numbers on the simple synthetic). "
                        "Attention model skips Phase A (no compatible SSL ckpt) "
                        "and uses random init for the transformer backbone.")
    args = p.parse_args()

    cfg = make_default()
    if args.quick:
        cfg.ssl_epochs = 3
        cfg.maml_episodes = 5
        args.phase_a_episodes = 5
        args.phase_a_ssl_epochs = 3
        args.phase_b_epochs = 10
        args.phase_b_warmup = 2
        args.phase_b_ramp = 3
        args.phase_c_episodes = 10
        args.phase_c_warmup = 2
        args.phase_c_ramp = 3
    cfg.device = resolve_device(args.device)
    cfg.seed = args.seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If a Tier 0 ckpt is provided, update cfg from the ckpt's saved
    # config *before* building the dataset. This is important when the
    # ckpt was trained with a different n_channels (e.g. F=2 ToA+AoA)
    # than the joint trainer's default (F=1 ToA only).
    if args.tier0_ckpt:
        print(f"Pre-loading Tier 0 ckpt config from {args.tier0_ckpt}")
        _ckpt = torch.load(args.tier0_ckpt, map_location="cpu", weights_only=False)
        if "cfg" in _ckpt:
            for k, v in _ckpt["cfg"].items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            print(f"  [cfg] updated: n_channels={cfg.n_channels}, "
                  f"channel_weights={list(cfg.channel_weights)}")
    elif args.phase_b_ckpt:
        # Same idea: when skipping straight to Phase C via --phase_b_ckpt,
        # we still need cfg to match the saved ckpt (n_channels, etc.) so
        # the dataset gets built with the right feature width.
        print(f"Pre-loading Phase B ckpt config from {args.phase_b_ckpt}")
        _ckpt = torch.load(args.phase_b_ckpt, map_location="cpu", weights_only=False)
        if "cfg" in _ckpt:
            for k, v in _ckpt["cfg"].items():
                if hasattr(cfg, k) and v is not None:
                    setattr(cfg, k, v)
            print(f"  [cfg] updated: n_channels={cfg.n_channels}, "
                  f"channel_weights={list(cfg.channel_weights)}")
    elif args.phase_a_ckpt:
        # Same idea: when loading a Phase A ckpt to seed Phase B (or to run
        # the standalone baseline), match cfg to the saved ckpt.
        print(f"Pre-loading Phase A ckpt config from {args.phase_a_ckpt}")
        _ckpt = torch.load(args.phase_a_ckpt, map_location="cpu", weights_only=False)
        if "cfg" in _ckpt:
            for k, v in _ckpt["cfg"].items():
                if hasattr(cfg, k) and v is not None:
                    setattr(cfg, k, v)
            print(f"  [cfg] updated: n_channels={cfg.n_channels}, "
                  f"channel_weights={list(cfg.channel_weights)}")

    print(f"device={cfg.device}  seed={cfg.seed}  phase={args.phase}  data={args.data}")
    if args.data == "synthetic":
        print("building synthetic data...")
        all_items = build_synthetic_dataset(cfg, seed=cfg.seed)
        env_ids = sorted({it[2] for it in all_items})
        # 10 train / 2 val: envs 0-9 train, 10-11 val (matches the
        # Sionna 21-env protocol where train=[0,10), val=[10,12)).
        # For other env counts, fall back to 80/20.
        if len(env_ids) == 12:
            train_envs = set(env_ids[:10])
        else:
            n_train_envs = max(1, int(0.8 * len(env_ids)))
            train_envs = set(env_ids[:n_train_envs])
        train_items = [it for it in all_items if it[2] in train_envs]
        val_items = [it for it in all_items if it[2] not in train_envs]
        n_envs = len(env_ids)
        n_conds = cfg.n_micro_conditions
        print(f"  total: {len(all_items)}  train: {len(train_items)}  val: {len(val_items)}  "
              f"n_envs={n_envs}  train_envs={sorted(train_envs)}  val_envs={sorted(set(env_ids)-train_envs)}")
    else:
        # Walking-path or Sionna real data
        if args.data == "walking_path":
            from scripts.walking_path_dataset import (
                build_walking_path_dataset, walking_path_to_items,
            )
            n_envs_walking = args.walking_path_n_envs or 4
            print(f"building walking-path dataset (n_envs={n_envs_walking})...")
            ds, scene_info = build_walking_path_dataset(
                csv_path=args.walking_path_csv, n_channels=cfg.n_channels,
                seq_len=cfg.seq_len, stride=cfg.seq_len,
                n_envs=n_envs_walking,
            )
            env_ids = [info.env_id for info in scene_info]
            # 80/20 split by env
            n_train_envs = max(1, int(0.8 * len(env_ids)))
            train_set = set(env_ids[:n_train_envs])
            val_set = set(env_ids[n_train_envs:])
            train_items, val_items, scene_info, denorm_info = walking_path_to_items(
                ds, train_env_ids=train_set, val_env_ids=val_set,
            )
            n_envs = len(scene_info)
            n_conds = 3   # placeholder, never used since cond_id = -1
            print(f"  total: {len(ds)}  train: {len(train_items)}  val: {len(val_items)}  "
                  f"n_envs={n_envs}  train_envs={sorted(train_set)}  val_envs={sorted(val_set)}")
            print(f"  per-channel norm (global): mean={ds.norm_mean.tolist()}  std={ds.norm_std.tolist()}")
        else:
            # Sionna real data (stationary per grid)
            from scripts.sionna_joint_dataset import (
                build_sionna_joint_dataset, sionna_to_items,
            )
            from pathlib import Path as _Path
            csv_paths = None
            if args.sionna_csv_dir:
                csv_paths = sorted(str(p) for p in _Path(args.sionna_csv_dir).glob("*_A1A4swapped.csv"))
                if not csv_paths:
                    raise RuntimeError(f"no CSVs found in {args.sionna_csv_dir}")
            print("building Sionna real-data dataset...")
            ds, scene_info = build_sionna_joint_dataset(
                csv_paths=csv_paths, n_channels=cfg.n_channels,
                seq_len=cfg.seq_len, stride=cfg.seq_len,
            )
            # Compute env_ids for the sionna branch (used to derive train/val)
            env_ids = [info.env_id for info in scene_info]
            train_set = set(args.sionna_train_envs) if args.sionna_train_envs else set(env_ids[:10])
            val_set = set(args.sionna_val_envs) if args.sionna_val_envs else set(env_ids[10:]) - {14}
            # If the user passed one of train/val explicitly, derive the other.
            if args.sionna_train_envs and not args.sionna_val_envs:
                val_set = set(env_ids) - train_set
            elif args.sionna_val_envs and not args.sionna_train_envs:
                train_set = set(env_ids) - val_set
            train_items, val_items, scene_info, denorm_info = sionna_to_items(
                ds, train_env_ids=train_set, val_env_ids=val_set,
            )
            # Compute per-channel norm stats from train scenes
            train_env_ids = sorted({it[2] for it in train_items})
            ds.compute_norm_stats(set(train_env_ids))
            # Re-fetch items with normalization applied
            train_items, val_items, scene_info, denorm_info = sionna_to_items(
                ds, train_env_ids=train_set, val_env_ids=val_set,
            )
            n_envs = len(scene_info)
            n_conds = 3   # placeholder, never used since cond_id = -1
            print(f"  total: {len(ds)}  train: {len(train_items)}  val: {len(val_items)}  "
                  f"n_envs={n_envs}  train_envs={train_env_ids}  val_envs={sorted({it[2] for it in val_items})}")
            print(f"  per-channel norm (from train scenes): mean={ds.norm_mean.tolist()}  std={ds.norm_std.tolist()}")
    print()

    encoder, head, model, history = None, None, None, {"A": None, "B": None, "C": None}

    # ---- Phase A ----
    if "A" in args.phase.upper() and not args.phase_a_ckpt:
        if args.model == "attention":
            print("  --model attention: skipping Phase A (no compatible "
                  "SSL ckpt for the transformer backbone). Proceeding with "
                  "random init for Phase B / C.")
        elif args.tier0_ckpt:
            print("=" * 60)
            print("  PHASE A — Tier 0 SSL encoder + MAML warmup (encoder frozen)")
            print("=" * 60)
            encoder, head, maml_losses = run_phase_A_from_tier0(
                cfg, args.tier0_ckpt, train_items, cfg.device,
                n_episodes=args.phase_a_episodes,
            )
        else:
            print("=" * 60)
            print("  PHASE A — SSL + MAML warmup (encoder frozen)")
            print("=" * 60)
            encoder, head, maml_losses = run_phase_A(
                cfg, train_items, cfg.device,
                n_episodes=args.phase_a_episodes,
                ssl_epochs=args.phase_a_ssl_epochs,
            )
        history["A"] = {"maml_losses": maml_losses}
        # Save
        torch.save({
            "encoder_state": encoder.state_dict(),
            "head_state": head.state_dict(),
            "maml_losses": maml_losses,
            "cfg": cfg.__dict__,
        }, out_dir / "phase_A.pt")
        print(f"\n  Phase A ckpt → {out_dir / 'phase_A.pt'}")
        print()
    elif args.phase_a_ckpt:
        print(f"Loading Phase A ckpt from {args.phase_a_ckpt}")
        ck = torch.load(args.phase_a_ckpt, map_location="cpu", weights_only=False)
        encoder = ASTAEncoder(cfg).to(cfg.device)
        encoder.load_state_dict(ck["encoder_state"])
        head = LocationHeadPerTimestep(cfg.d_model).to(cfg.device)
        head.load_state_dict(ck["head_state"])
        print()

    # ---- Phase B ----
    if "B" in args.phase.upper() and not args.phase_b_ckpt:
        if encoder is None:
            raise RuntimeError("Phase B requires Phase A encoder; provide --phase_a_ckpt or run A first")
        print("=" * 60)
        print("  PHASE B — Disentangle with frozen encoder")
        print("=" * 60)
        model, hist_b = run_phase_B(
            cfg, encoder, head, train_items,
            n_envs=n_envs, n_conds=n_conds, device=cfg.device,
            n_epochs=args.phase_b_epochs,
            warmup_epochs=args.phase_b_warmup,
            ramp_epochs=args.phase_b_ramp,
            model_kind=args.model,
        )
        history["B"] = hist_b
        if not hist_b:
            # Attention model: Phase B was a no-op (no Phase A to freeze
            # against). Skip the per-cond eval and ckpt save; Phase C
            # will train the model from its random init.
            print("  Phase B was a no-op for the attention model. "
                  "Skipping per-cond eval; Phase C will train from random init.")
        else:
            # Per-cond eval (or per-NLOS-bucket for Sionna)
            per_cond = eval_for_data(model, train_items, cfg.device, args.data,
                                      denorm_info=denorm_info if args.data == "sionna" else None)
            print(f"  Phase B per-cond loc err (train): {per_cond}")
            torch.save({
                "model_state": model.state_dict(),
                "history": hist_b,
                "per_cond": per_cond,
                "cfg": cfg.__dict__,
                "n_envs": n_envs,
            }, out_dir / "phase_B.pt")
            # Save history CSV
            with open(out_dir / "phase_B_history.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(hist_b[0].keys()))
                w.writeheader()
                w.writerows(hist_b)
            print(f"  Phase B ckpt → {out_dir / 'phase_B.pt'}")
        print()
    elif args.phase_b_ckpt:
        print(f"Loading Phase B ckpt from {args.phase_b_ckpt}")
        ck = torch.load(args.phase_b_ckpt, map_location="cpu", weights_only=False)
        # Update cfg from saved ckpt cfg to match n_channels etc.
        if "cfg" in ck:
            for k, v in ck["cfg"].items():
                if hasattr(cfg, k) and v is not None:
                    setattr(cfg, k, v)
            print(f"  [cfg] updated from Phase B ckpt: "
                  f"n_channels={cfg.n_channels}, d_model={cfg.d_model}")
        if args.model == "attention":
            model = AttentionDisentangledTrainer(cfg, n_envs=n_envs, n_conds=n_conds).to(cfg.device)
        else:
            model = DisentangledTrainer(cfg, n_envs=n_envs, n_conds=n_conds).to(cfg.device)
        model.load_state_dict(ck["model_state"])
        history["B"] = ck.get("history")

    # ---- Phase C ----
    if "C" in args.phase.upper():
        if model is None:
            if args.model == "attention":
                # Attention model has no Phase A/B; build a fresh model
                # here so Phase C can train from random init.
                print("  --model attention: building fresh model for Phase C "
                      "(no Phase A/B to load from).")
                model = AttentionDisentangledTrainer(
                    cfg, n_envs=n_envs, n_conds=n_conds,
                ).to(cfg.device)
            else:
                raise RuntimeError("Phase C requires Phase B model; provide --phase_b_ckpt or run B first")
        print("=" * 60)
        print("  PHASE C — Joint FOMAML + disentangle (encoder unfrozen)")
        print("=" * 60)
        # If we want intermediate snapshots, bind the data_source +
        # denorm_info into a closure so run_phase_C can call it with
        # just (model, items, device).
        eval_closure = None
        if args.phase_c_save_every > 0:
            data_source = args.data
            denorm = denorm_info if args.data == "sionna" else None
            def eval_closure(m, items, device, ds=data_source, dn=denorm):
                return eval_for_data(m, items, device, ds, denorm_info=dn)
        model, hist_c, snapshots_c = run_phase_C(
            cfg, model, train_items, n_envs=n_envs, device=cfg.device,
            n_episodes=args.phase_c_episodes,
            warmup_episodes=args.phase_c_warmup,
            ramp_episodes=args.phase_c_ramp,
            save_every=args.phase_c_save_every,
            val_items=val_items if args.phase_c_save_every > 0 else None,
            eval_fn=eval_closure,
            freeze_episodes=args.phase_c_freeze_episodes,
            unfreeze_lr_scale=args.phase_c_unfreeze_lr_scale,
        )
        history["C"] = hist_c
        # Per-cond eval (or per-NLOS-bucket for Sionna)
        per_cond = eval_for_data(model, val_items, cfg.device, args.data,
                                  denorm_info=denorm_info if args.data == "sionna" else None)
        print(f"  Phase C per-cond loc err (val): {per_cond}")
        torch.save({
            "model_state": model.state_dict(),
            "history": hist_c,
            "per_cond": per_cond,
            "cfg": cfg.__dict__,
            "n_envs": n_envs,
        }, out_dir / "phase_C.pt")
        with open(out_dir / "phase_C_history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hist_c[0].keys()))
            w.writeheader()
            w.writerows(hist_c)
        # Per-cond CSV in the same format as standalone Tier 2
        # (for the 3-way viz_tier2_loc_comparison --compare_all).
        # The 'baseline' column is filled in by --run_baseline below.
        # For Sionna data, the column is bucket-based (NLOS rate), not cond-based.
        col_baseline = "baseline_loc_err" if args.data == "synthetic" else "baseline_bucket_err"
        col_joint = "joint_loc_err" if args.data == "synthetic" else "joint_bucket_err"
        with open(out_dir / "phase_C_per_cond.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cond_id", col_baseline, col_joint])
            for ci in sorted(per_cond.keys()):
                w.writerow([ci, "NA", per_cond[ci]])
        print(f"  Phase C ckpt → {out_dir / 'phase_C.pt'}")
        print()

        # If we took intermediate snapshots, save them too and pick the
        # best one by mean per-cond val loc err. This is the early-stop
        # signal: meta loss can rise over Phase C, but a snapshot partway
        # through may be a much better encoder.
        if snapshots_c:
            snap_dir = out_dir / "phase_C_snapshots"
            snap_dir.mkdir(exist_ok=True)
            best = None  # (mean_err, ep, path)
            for s in snapshots_c:
                snap_path = snap_dir / f"phase_C_ep{s['ep']:03d}.pt"
                torch.save({"model_state": s["state_dict"],
                            "per_cond": s["per_cond"],
                            "ep": s["ep"]}, snap_path)
                pc = s["per_cond"] or {}
                mean_err = sum(pc.values()) / max(1, len(pc)) if pc else float("inf")
                if best is None or mean_err < best[0]:
                    best = (mean_err, s["ep"], snap_path)
            if best is not None:
                print(f"  best snapshot: ep={best[1]}  mean per-cond val err = {best[0]:.3f} m  ({best[2]})")

    print("=" * 60)
    print("  DONE")
    print("=" * 60)
    if model is not None:
        per_cond = eval_for_data(model, val_items, cfg.device, args.data,
                                  denorm_info=denorm_info if args.data == "sionna" else None)
        print(f"  final per-cond loc err (val): {per_cond}")

        # Optional: train baseline and merge into the per-cond CSV
        if args.run_baseline:
            print("\n  Training baseline (no Tier 2) for the 3-way comparison...")
            baseline_per_cond = run_baseline_for_joint(
                cfg, train_items, val_items, cfg.device,
                n_epochs=args.baseline_epochs,
                denorm_info=denorm_info if args.data == "sionna" else None,
                save_ckpt_path=str(out_dir / "baseline.pt") if args.data == "sionna" else None,
            )
            print(f"  baseline per-cond loc err: {baseline_per_cond}")
            # Re-write phase_C_per_cond.csv with both columns.
            # For Sionna data, the column is named "baseline_bucket_err" /
            # "joint_bucket_err" to reflect that it's NLOS-bucket, not cond.
            col_baseline = "baseline_loc_err" if args.data == "synthetic" else "baseline_bucket_err"
            col_joint = "joint_loc_err" if args.data == "synthetic" else "joint_bucket_err"
            with open(out_dir / "phase_C_per_cond.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["cond_id", col_baseline, col_joint])
                for ci in sorted(per_cond.keys()):
                    w.writerow([ci, baseline_per_cond.get(ci, "NA"), per_cond[ci]])
            print(f"  updated per-cond CSV: {out_dir / 'phase_C_per_cond.csv'}")


if __name__ == "__main__":
    main()
