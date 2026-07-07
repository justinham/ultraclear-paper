"""Tier 2 — Full adversarial training loop on the synthetic UWB data.

This is the missing piece from `scripts/train.py`'s Tier 2 smoke test.

What it does:
  1. Builds the synthetic dataset (8 envs × 3 micro conditions).
  2. Splits 80/20 by *environment* so the val set has unseen envs
     (this is the macro-OOD protocol the paper uses).
  3. Initializes the DisentangledModel from the MAML-trained encoder
     (loads a Tier-1 checkpoint if provided, else runs MAML from
     scratch with a Tier-0 SSL-pretrained encoder).
  4. Trains the full model with:
       - Task loss:      location MSE (x, y)
       - Adv loss:       cross-entropy on env_id from c, pushed through
                         a gradient-reversal layer (already inside
                         DisentangledModel.forward)
       - NLOS loss:      per-(anchor, time) binary CE (uses the
                         ground-truth nlos mask now exposed by data.py)
  5. Two optimizers: one for the feature extractor (encoder + micro-
     encoder + FiLM + location head), one for the adversary. The
     gradient-reversal layer handles the sign flip.
  6. Curriculum for adv_lambda: 0 for the first `warmup_epochs` epochs,
     then linearly ramped to 1.0 over `ramp_epochs` epochs. Standard
     DANN practice — gives the task loss time to settle before the
     disentanglement pressure kicks in.
  7. Periodic linear-probe of c → env_id accuracy (the canonical
     disentanglement metric from §tier2 of the paper).
  8. Saves a checkpoint + a CSV with per-epoch metrics for the
     visualization scripts to consume.

Run as:
  python -m scripts.train_tier2_synthetic
  python -m scripts.train_tier2_synthetic --quick                # 30s smoke test
  python -m scripts.train_tier2_synthetic --tier1_ckpt PATH.pt   # init from MAML
  python -m scripts.train_tier2_synthetic --tier0_ckpt PATH.pt   # init from SSL
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
from ultraclear.data import build_synthetic_dataset, TensorUWB
from ultraclear.tier0_ssl import SSLPretrainer, ssl_loss
from ultraclear.tier1_maml import MAMLCore, maml_train
from ultraclear.tier2_micro import DisentangledModel
from ultraclear.tier3_heads import NLOSHead, LocationHeadFromRestore, RestorationHead


# ----------------------------------------------------------------------------
# Location head that sits on top of FiLM-modulated features
# (mirrors tier1_maml.LocationHead so the adapter can drop into either
# pipeline without surgery)
# ----------------------------------------------------------------------------
class LocationHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.pool_attn = nn.Linear(d_model, 1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, M, D = h.shape
        a = self.pool_attn(h).squeeze(-1)
        w = TF.softmax(a.view(B, T * M), dim=-1).view(B, T, M).unsqueeze(-1)
        pooled = (h * w).sum(dim=(1, 2))
        return self.mlp(pooled)


# ----------------------------------------------------------------------------
# DisentangledTrainer — wraps the model + heads + adversaries
# ----------------------------------------------------------------------------
class DisentangledTrainer(nn.Module):
    """Encoder (Tier 1 MAML-initialized) + MicroEncoder + FiLM + heads.

    Forward returns the pieces the training loop needs:
      h_prime : (B, T, M, d)  FiLM-modulated features
      c       : (B, micro_dim) micro condition vector
      adv_logits : (B, n_envs) adversary logits (after grad reversal)
      cond_logits : (B, n_conds) condition classifier (auxiliary loss)
      loc     : (B, 2) location prediction
      nlos_logits : (B, T, M) NLOS head logits (computed from c, NOT h')

    Design note: the NLOS head is attached to `c` (the micro condition
    vector), not to `h'`. c is supposed to be informative about the
    micro condition (NLOS rate, severity), so NLOS is a natural target.
    Attaching it to h' would let FiLM wash out the NLOS signal by
    canceling it across channels.
    """

    def __init__(self, cfg: Config, n_envs: int, n_conds: int = 3):
        super().__init__()
        self.cfg = cfg
        self.disent = DisentangledModel(cfg, n_envs=n_envs)
        self.loc_head = LocationHead(cfg.d_model)
        # NLOS head: per-(B, T, M) logits from c. One shared MLP that
        # takes c and emits per-anchor logits, replicated across T.
        self.nlos_mlp = nn.Sequential(
            nn.Linear(cfg.micro_dim, cfg.adv_hidden),
            nn.GELU(),
            nn.Linear(cfg.adv_hidden, cfg.n_anchors),     # per-anchor logit
        )
        # Auxiliary: c -> cond_id. Keeps c informative about the micro
        # condition (a positive pressure to balance the env-adversary's
        # negative pressure). Trained alongside the rest.
        self.cond_head = nn.Linear(cfg.micro_dim, n_conds)

    def forward(self, x: torch.Tensor, detach_c_for_adv: bool = False):
        h_prime, c, adv_logits = self.disent(x, detach_c_for_adv=detach_c_for_adv)
        loc = self.loc_head(h_prime)
        # Per-(B, T, M) NLOS logits from c. Per-anchor logits replicated
        # across T (NLOS within a window is dominated by the micro
        # condition, which is roughly constant within a window).
        per_anchor = self.nlos_mlp(c)                     # (B, M)
        nlos_logits = per_anchor.unsqueeze(1).expand(-1, self.cfg.seq_len, -1)  # (B, T, M)
        cond_logits = self.cond_head(c)                   # (B, n_conds)
        return h_prime, c, adv_logits, loc, nlos_logits, cond_logits


# ----------------------------------------------------------------------------
# Linear probe: a 1-layer linear classifier on c → env_id.
# Trained briefly on a held-out split, reports accuracy. This is the
# "linear classifier should fail to predict env" disentanglement metric.
# ----------------------------------------------------------------------------
def linear_probe_acc(c_train, env_train, c_val, env_val, n_envs, device, epochs=20):
    """Train a fresh linear layer on (c_train, env_train), eval on val."""
    probe = nn.Linear(c_train.shape[1], n_envs).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    Xtr = torch.from_numpy(c_train).float().to(device)
    Ytr = torch.from_numpy(env_train).long().to(device)
    Xva = torch.from_numpy(c_val).float().to(device)
    Yva = torch.from_numpy(env_val).long().to(device)
    for _ in range(epochs):
        opt.zero_grad()
        loss = TF.cross_entropy(probe(Xtr), Ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(Xva).argmax(dim=-1)
        acc = (pred == Yva).float().mean().item()
    return acc


# ----------------------------------------------------------------------------
# Train a baseline (no Tier 2) for the comparison viz later
# ----------------------------------------------------------------------------
def run_tier0_and_tier1_init(cfg: Config, train_items, all_items, device):
    """Run Tier 0 SSL + Tier 1 MAML on synthetic, return MAMLCore."""
    print("  Tier 0 SSL pretraining...")
    ssl = SSLPretrainer(cfg).to(device)
    opt = torch.optim.Adam(ssl.parameters(), lr=cfg.ssl_lr)
    loader = DataLoader(TensorUWB(train_items), batch_size=cfg.ssl_batch_size, shuffle=True)
    ssl.train()
    for ep in range(cfg.ssl_epochs):
        total, n = 0.0, 0
        for X, _, _, _, _ in loader:
            X = X.to(device)
            opt.zero_grad()
            x_pred, mask = ssl(X)
            loss = ssl_loss(x_pred, X, mask)
            loss.backward()
            opt.step()
            total += loss.item() * X.size(0)
            n += X.size(0)
        if (ep + 1) % max(1, cfg.ssl_epochs // 3) == 0 or ep == 0:
            print(f"    [tier0] ep {ep+1}/{cfg.ssl_epochs}  ssl_loss = {total/n:.4f}")

    print("  Tier 1 MAML on top of SSL encoder...")
    maml = maml_train(cfg, all_items, device=device, log_every=max(1, cfg.maml_episodes // 3))
    return maml


def run_baseline(
    cfg: Config, train_items, val_items, device
):
    """No Tier 2: just ASTA encoder + Location head, train supervised."""
    print("  Baseline (no Tier 2): ASTA encoder + Location head...")
    from ultraclear.asta import ASTAEncoder

    enc = ASTAEncoder(cfg).to(device)
    head = LocationHead(cfg.d_model).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    train_loader = DataLoader(TensorUWB(train_items), batch_size=32, shuffle=True)
    enc.train(); head.train()
    for ep in range(cfg.finetune_epochs):
        total, n = 0.0, 0
        for X, y, _, _, _ in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            h = enc(X)
            pred = head(h)
            loss = TF.mse_loss(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * X.size(0)
            n += X.size(0)
        if (ep + 1) % max(1, cfg.finetune_epochs // 3) == 0 or ep == 0:
            print(f"    [baseline] ep {ep+1}/{cfg.finetune_epochs}  loc_mse = {total/n:.4f}")

    # Per-cond eval
    enc.eval(); head.eval()
    val_loader = DataLoader(TensorUWB(val_items), batch_size=64)
    errs_per_cond = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for X, y, _, c, _ in val_loader:
            X, y = X.to(device), y.to(device)
            pred = head(enc(X))
            err = torch.linalg.norm(pred - y, dim=-1).cpu().numpy()
            for ci, e in zip(c.tolist(), err):
                errs_per_cond[ci].append(e)
    return {ci: float(np.mean(es)) for ci, es in errs_per_cond.items()}


# ----------------------------------------------------------------------------
# Main Tier 2 training loop
# ----------------------------------------------------------------------------
def run_tier2(
    cfg: Config,
    train_items,
    val_items,
    n_envs: int,
    maml_encoder_state: dict | None,
    device: str,
    warmup_epochs: int = 5,
    ramp_epochs: int = 10,
) -> tuple[DisentangledTrainer, list[dict]]:
    """Full adversarial training loop. Returns (trained_model, history)."""

    model = DisentangledTrainer(cfg, n_envs=n_envs).to(device)
    if maml_encoder_state is not None:
        model.disent.encoder.load_state_dict(maml_encoder_state)
        print("  Initialized encoder from MAML checkpoint.")
    else:
        print("  WARNING: random encoder init (no MAML ckpt provided).")

    # Two optimizers: one for the feature extractor, one for the adversary.
    # The gradient-reversal layer handles the sign on the adversary's gradient
    # for the feature-extractor side. The adversary's own optimizer sees
    # adv_ce as a normal loss.
    feat_params = (
        list(model.disent.encoder.parameters())
        + list(model.disent.h_phi.parameters())
        + list(model.disent.film.parameters())
        + list(model.loc_head.parameters())
        + list(model.nlos_mlp.parameters())
        + list(model.cond_head.parameters())
    )
    adv_params = list(model.disent.adv.parameters())
    opt_feat = torch.optim.Adam(feat_params, lr=cfg.finetune_lr)
    opt_adv = torch.optim.Adam(adv_params, lr=cfg.adv_lr)

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
        return 1.0

    print(f"  Training {cfg.finetune_epochs} epochs (warmup={warmup_epochs}, "
          f"ramp={ramp_epochs})")
    t0 = time.time()
    for ep in range(cfg.finetune_epochs):
        lam = compute_lambda(ep)
        # The gradient-reversal layer in DisentangledModel uses cfg.adv_lambda
        # (NOT our schedule's lambda). We achieve the schedule by *scaling
        # the GR strength* via a manual lambda parameter passed through
        # `_grad_reverse`. To keep the model API stable, we instead scale
        # the adversary's loss by `lam` * cfg.adv_lambda on the *feat* side
        # via the loss construction below. The model-side GR is set to
        # cfg.adv_lambda=1.0 (constant) and the feat optimizer applies
        # `lam * adv_ce` as a *negative* term in the task loss.
        model.train()
        total_task, total_adv, total_nlos, total_cond, n = 0.0, 0.0, 0.0, 0.0, 0
        for X, y, env, cond, nlos in train_loader:
            X = X.to(device); y = y.to(device); env = env.to(device)
            cond = cond.to(device); nlos = nlos.to(device)
            # ---- Two-optimizer pattern (clean version) ----
            # Pass 1 (featurizer): get c, pass it through adversary with
            # grad enabled (c NOT detached) so the GR layer inside the
            # model pushes c AWAY from being predictive of env. This is
            # the disentanglement pressure on the encoder / h_phi / FiLM.
            # Auxiliary cond loss: c -> cond_id. Keeps c informative
            # about the micro condition (positive pressure balancing
            # the env-adversary's negative pressure).
            h_prime, c, adv_logits, loc, nlos_logits, cond_logits = model(
                X, detach_c_for_adv=False
            )
            task_loss = TF.mse_loss(loc, y)
            adv_ce = TF.cross_entropy(adv_logits, env)
            nlos_loss = TF.binary_cross_entropy_with_logits(
                nlos_logits, nlos.float()
            )
            cond_loss = TF.cross_entropy(cond_logits, cond)
            # `+ lam * adv_ce` is correct (NOT `-`): the GR layer inside
            # the model reverses the gradient w.r.t. c, so the featurizer
            # gets pushed to *maximize* adv_ce, i.e. make c uninformative.
            # `- cond_loss` (yes, MINUS) is the positive pressure to keep
            # c informative about cond. After GR sign flip the effective
            # gradient on c is `+lam * ∂adv_ce/∂c - ∂cond_loss/∂c`,
            # which pushes c away from env-informative and toward
            # cond-informative — the right saddle-point.
            feat_loss = (
                task_loss
                + cfg.nlos_loss_weight * nlos_loss
                + cfg.cond_loss_weight * cond_loss
                + lam * adv_ce
            )
            opt_feat.zero_grad()
            feat_loss.backward()
            opt_feat.step()
            # Pass 2 (adversary only): forward again with c detached so
            # the backward only updates D's own weights, not the encoder.
            _h, _c, adv_logits2, _loc, _nlos, _cond2 = model(
                X, detach_c_for_adv=True
            )
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

        # ---- val ----
        model.eval()
        val_task_total, val_n_total = 0.0, 0
        c_buf, env_buf = [], []
        with torch.no_grad():
            for X, y, env, _cond, nlos in val_loader:
                X = X.to(device); y = y.to(device); env = env.to(device)
                _h, c, _adv, loc, _nlos, _cond = model(X)
                val_task_total += TF.mse_loss(loc, y, reduction="sum").item()
                val_n_total += X.size(0)
                c_buf.append(c.cpu().numpy())
                env_buf.append(env.cpu().numpy())
        val_task = val_task_total / val_n_total
        c_all = np.concatenate(c_buf, axis=0)
        env_all = np.concatenate(env_buf, axis=0)
        # Random 50/50 split for the linear probe (deterministic)
        rng = np.random.default_rng(ep)
        idx = rng.permutation(len(c_all))
        split = len(idx) // 2
        probe = linear_probe_acc(
            c_all[idx[:split]], env_all[idx[:split]],
            c_all[idx[split:]], env_all[idx[split:]],
            n_envs, device,
        )
        # Chance depends on what envs are in c_all. If we probed on train
        # we'd have n_envs; on val we have only 2 (the unseen envs).
        n_envs_in_probe = len(np.unique(env_all))
        chance = 1.0 / n_envs_in_probe
        history.append({
            "epoch": ep + 1,
            "lambda": lam,
            "train_task_mse": train_task,
            "train_adv_ce": train_adv,
            "train_nlos_ce": train_nlos,
            "train_cond_ce": train_cond,
            "val_task_mse": val_task,
            "probe_env_acc": probe,
            "probe_chance": chance,
            "probe_n_envs": n_envs_in_probe,
        })
        if (ep + 1) % max(1, cfg.finetune_epochs // 10) == 0 or ep == 0:
            print(f"    [tier2] ep {ep+1:3d}/{cfg.finetune_epochs}  "
                  f"lam={lam:.2f}  task={train_task:.3f}  adv={train_adv:.3f}  "
                  f"nlos={train_nlos:.3f}  cond={train_cond:.3f}  "
                  f"val_task={val_task:.3f}  "
                  f"probe_env_acc={probe:.3f} (chance {chance:.3f}, "
                  f"{n_envs_in_probe} envs)")
    print(f"  tier-2 done in {time.time()-t0:.1f}s")
    return model, history


# ----------------------------------------------------------------------------
# Final per-cond evaluation on the val set
# ----------------------------------------------------------------------------
def eval_per_cond(model: DisentangledTrainer, val_items, device) -> dict:
    model.eval()
    loader = DataLoader(TensorUWB(val_items), batch_size=64)
    errs_per_cond = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for X, y, _, cond, _ in loader:
            X, y = X.to(device), y.to(device)
            _h, _c, _adv, loc, _nlos, _cond = model(X)
            err = torch.linalg.norm(loc - y, dim=-1).cpu().numpy()
            for ci, e in zip(cond.tolist(), err):
                errs_per_cond[ci].append(e)
    return {ci: float(np.mean(es)) if es else float("nan") for ci, es in errs_per_cond.items()}


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    p.add_argument("--tier0_ckpt", type=str, default=None)
    p.add_argument("--tier1_ckpt", type=str, default=None)
    p.add_argument("--out", type=str, default="checkpoints/tier2_synthetic")
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--ramp_epochs", type=int, default=10)
    p.add_argument("--nlos_loss_weight", type=float, default=0.5)
    p.add_argument("--cond_loss_weight", type=float, default=0.5)
    p.add_argument("--run_baseline", action="store_true",
                   help="also train the no-Tier-2 baseline for the comparison viz")
    args = p.parse_args()

    cfg = make_default()
    if args.quick:
        cfg.ssl_epochs = 3
        cfg.maml_episodes = 5
        cfg.finetune_epochs = 10
        args.warmup_epochs = 2
        args.ramp_epochs = 3
    cfg.device = args.device
    cfg.seed = args.seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    # nlos_loss_weight is a CLI arg, not on Config — attach as a transient attr
    cfg.nlos_loss_weight = args.nlos_loss_weight
    cfg.cond_loss_weight = args.cond_loss_weight

    print(f"device={cfg.device}  seed={cfg.seed}")
    print("building synthetic data...")
    all_items = build_synthetic_dataset(cfg, seed=cfg.seed)
    # 80/20 by env
    env_ids = sorted({it[2] for it in all_items})
    n_train_envs = max(1, int(0.8 * len(env_ids)))
    train_envs = set(env_ids[:n_train_envs])
    train_items = [it for it in all_items if it[2] in train_envs]
    val_items = [it for it in all_items if it[2] not in train_envs]
    n_envs = len(env_ids)
    print(f"  total: {len(all_items)}  train: {len(train_items)}  val: {len(val_items)}  "
          f"n_envs={n_envs}  train_envs={sorted(train_envs)}  val_envs={sorted(set(env_ids)-train_envs)}")

    # ---- Tier 0 + Tier 1 init ----
    if args.tier1_ckpt and Path(args.tier1_ckpt).exists():
        print(f"Loading MAML checkpoint from {args.tier1_ckpt}")
        ck = torch.load(args.tier1_ckpt, map_location="cpu")
        maml_encoder_state = ck.get("encoder", ck) if isinstance(ck, dict) else ck
        # In case it's a full MAMLCore state_dict, extract encoder only:
        if any(k.startswith("encoder.") for k in maml_encoder_state.keys()):
            maml_encoder_state = {
                k[len("encoder."):]: v for k, v in maml_encoder_state.items()
                if k.startswith("encoder.")
            }
    else:
        print("No Tier 1 ckpt; running Tier 0 + Tier 1 from scratch...")
        maml = run_tier0_and_tier1_init(cfg, train_items, all_items, cfg.device)
        maml_encoder_state = maml.encoder.state_dict()

    # ---- Tier 2 ----
    print("\n--- TIER 2 — Full adversarial training ---")
    model, history = run_tier2(
        cfg, train_items, val_items, n_envs, maml_encoder_state, cfg.device,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
    )
    final_per_cond = eval_per_cond(model, val_items, cfg.device)
    print(f"  final per-cond loc error: {final_per_cond}")

    # ---- Baseline (optional) ----
    baseline_per_cond = None
    if args.run_baseline:
        print("\n--- BASELINE — No Tier 2 ---")
        baseline_per_cond = run_baseline(cfg, train_items, val_items, cfg.device)
        print(f"  baseline per-cond loc error: {baseline_per_cond}")

    # ---- Save ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_path = out_dir / "tier2_synthetic.pt"
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
        "history": history,
        "final_per_cond": final_per_cond,
        "baseline_per_cond": baseline_per_cond,
        "n_envs": n_envs,
    }, ck_path)
    print(f"\nCheckpoint saved to {ck_path}")

    # ---- Save history CSV for the viz scripts ----
    csv_path = out_dir / "tier2_history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(f"History saved to {csv_path}")

    # ---- Save per-cond summary ----
    summary_path = out_dir / "tier2_per_cond.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cond_id", "baseline_loc_err", "tier2_loc_err"])
        for ci in sorted(final_per_cond.keys()):
            w.writerow([ci, baseline_per_cond[ci] if baseline_per_cond else "NA",
                        final_per_cond[ci]])
    print(f"Per-cond summary saved to {summary_path}")


if __name__ == "__main__":
    main()
