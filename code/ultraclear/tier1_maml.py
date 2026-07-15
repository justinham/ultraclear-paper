"""Tier 1 — Macro MAML over environments.

A "task" is one deployment environment. We sample N envs per outer
iteration; for each we draw K support samples and Q query samples,
adapt the model on the support set with K inner gradient steps, and
update the *initialization* (the shared parameters) on the query loss.

We use first-order MAML (FOMAML) via torch.func.functional_call to
keep the inner loop cheap and stable. The encoder here is the same
ASTAEncoder from Tier 0; the prediction head is a small regression
head that maps the encoder's pooled output to a 2D (x, y) location.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.func import functional_call

from .config import Config
from .asta import ASTAEncoder


class LocationHead(nn.Module):
    """Pools the (T, M) tokens of the encoder output to a single vector and
    regresses to 2D (x, y)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.pool_attn = nn.Linear(d_model, 1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, M, d_model) -> (B, 2) location."""
        B, T, M, D = h.shape
        # attention-weighted average over (T, M)
        a = self.pool_attn(h).squeeze(-1)         # (B, T, M)
        w = TF.softmax(a.view(B, T * M), dim=-1).view(B, T, M)
        w = w.unsqueeze(-1)                       # (B, T, M, 1)
        pooled = (h * w).sum(dim=(1, 2))          # (B, D)
        return self.mlp(pooled)                   # (B, 2)


class LocationHeadPerTimestep(nn.Module):
    """Per-timestep location head for moving-tag localization.

    Input:  h of shape (B, T, M, d_model) — encoder output
    Output: loc of shape (B, T, 2) — one (x, y) per timestep

    The head pools over the M anchor dimension only (not over T) so
    each timestep gets its own location prediction. The pooling is
    a learnable attention-weighted average across the 4 anchors.

    Why pool over M only (not T): the tag can be at a *different*
    (x, y) at each timestep, so we must not blend information across
    timesteps. Pooling only across M is the contract for a per-
    timestep head.

    For a stationary tag (every timestep has the same (x, y)), the
    T outputs will all be (approximately) equal — the head will
    learn to ignore the time axis. This is the head for
    moving-tag setups where each timestep has a distinct label.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # attention weights over the M anchors, shared across T
        self.anchor_attn = nn.Linear(d_model, 1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, M, d_model) -> (B, T, 2) per-timestep location.

        Per-timestep attention over M anchors:
          a_t: (B, T, M) -- attention scores across anchors for each (b, t)
          w_t: (B, T, M) -- softmax-normalized
          pooled_t: (B, T, D) -- weighted average of h over M
          loc: (B, T, 2) -- per-timestep (x, y)
        """
        B, T, M, D = h.shape
        # attention across M (anchor dim) only, per (b, t)
        a = self.anchor_attn(h).squeeze(-1)      # (B, T, M)
        w = TF.softmax(a, dim=-1)                # (B, T, M) — sum to 1 over M
        w = w.unsqueeze(-1)                      # (B, T, M, 1)
        pooled = (h * w).sum(dim=2)              # (B, T, D) — sum over M
        out = self.mlp(pooled)                   # (B, T, 2)
        return out


class MAMLCore(nn.Module):
    """Encoder + location head. This is what gets meta-learned."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = ASTAEncoder(cfg)
        self.head = LocationHead(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)                       # (B, T, M, d)
        return self.head(h)                       # (B, 2)


def adapt_and_evaluate(
    core: MAMLCore,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One MAML inner loop on a single task.

    Returns (query_loss, info_dict).  We use first-order MAML:
    functional_call gives us a functional view of `core` with named params,
    we take gradient steps on a *local* copy, then evaluate the query loss
    using the *original* core's params and the *inner-adapted* fast weights
    via functional_call again. To keep things simple we evaluate the
    *adapted* model's query loss (still using the post-adaptation fast
    weights) but the *outer* gradient is taken w.r.t. the original
    parameters, treating the adaptation as a stop-grad operation.
    """
    params = dict(core.named_parameters())
    names = list(params.keys())
    # Make fast weights as a leaf tensor copy for each param.
    fast = {n: p.detach().clone().requires_grad_(True) for n, p in params.items()}

    # Inner loop
    for _ in range(inner_steps):
        pred = functional_call(core, fast, (support_x,))
        # LocationHead outputs (B, 2) (per-window, attention-pooled
        # over T and M). support_y is (B, T, 2) (per-timestep from
        # the Sionna dataset). Take the middle timestep to match.
        T = support_x.size(1)
        y_mid = support_y[:, T // 2, :]
        loss = TF.mse_loss(pred, y_mid)
        grads = torch.autograd.grad(
            loss, list(fast.values()), create_graph=False, allow_unused=True
        )
        for n, g in zip(names, grads):
            if g is None:
                # parameter wasn't used in this forward pass; keep it as-is
                continue
            fast[n] = fast[n] - inner_lr * g

    # Query evaluation with adapted weights
    with torch.no_grad():
        # we still need a grad path through fast -> original params for FOMAML
        # by re-running with create_graph=True only on the meta-loss
        pass
    q_pred = functional_call(core, fast, (query_x,))
    # Same per-window head design: q_pred is (B, 2), query_y is
    # (B, T, 2). Take the middle timestep.
    T_q = query_x.size(1)
    q_y_mid = query_y[:, T_q // 2, :]
    q_loss = TF.mse_loss(q_pred, q_y_mid)
    return q_loss, {"q_pred": q_pred.detach()}


def sample_task(
    items: List[Tuple],
    env_id: int,
    n: int,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick `n` random samples from one environment."""
    pool = [it for it in items if it[2] == env_id]
    if len(pool) < n:
        # with replacement if env is small
        idx = torch.randint(0, len(pool), (n,), generator=rng).tolist()
    else:
        idx = torch.randperm(len(pool), generator=rng)[:n].tolist()
    X = torch.from_numpy(__import__("numpy").stack([pool[i][0] for i in idx])).float()
    y = torch.from_numpy(__import__("numpy").stack([pool[i][1] for i in idx])).float()
    return X, y


def maml_train(
    cfg: Config,
    items: List[Tuple],
    device: str = "cpu",
    log_every: int = 10,
) -> MAMLCore:
    """First-order MAML outer loop.

    items: list of (X, y, env_id, cond_id) — for tier 1 we only need env_id.
    Each iteration: sample N envs, build support/query, sum query losses,
    backprop into core's parameters.
    """
    core = MAMLCore(cfg).to(device)
    opt = torch.optim.Adam(core.parameters(), lr=cfg.maml_outer_lr)
    rng = torch.Generator().manual_seed(cfg.seed)
    env_ids = sorted({it[2] for it in items})
    assert len(env_ids) >= cfg.n_way, f"need at least {cfg.n_way} envs, have {len(env_ids)}"

    core.train()
    for ep in range(cfg.maml_episodes):
        opt.zero_grad()
        # sample N envs without replacement
        env_sample = [
            env_ids[i] for i in torch.randperm(len(env_ids), generator=rng)[: cfg.n_way].tolist()
        ]
        total_q = 0.0
        for eid in env_sample:
            sx, sy = sample_task(items, eid, cfg.k_shot, rng)
            qx, qy = sample_task(items, eid, cfg.q_query, rng)
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)
            q_loss, _ = adapt_and_evaluate(
                core, sx, sy, qx, qy,
                inner_lr=cfg.maml_inner_lr,
                inner_steps=cfg.maml_inner_steps,
            )
            total_q = total_q + q_loss
        total_q = total_q / cfg.n_way
        total_q.backward()
        opt.step()
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"  [maml] ep {ep+1:3d}/{cfg.maml_episodes}  query_loss = {total_q.item():.4f}")
    return core
