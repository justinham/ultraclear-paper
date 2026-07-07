"""Tier 2 — Micro condition disentanglement.

We learn a small encoder h_phi that maps ASTA features H to a condition
vector c. A FiLM layer then modulates H with c:  H' = gamma(c) * H + beta(c).
A separate adversary D tries to predict the *macro* environment from c;
a gradient-reversal layer (implemented manually here) forces D to fail,
which means c contains no information about the macro environment —
only about the micro propagation condition.

The `c` vector is the *disentangled* micro representation that lets the
downstream heads compose "outdoor env" + "heavy blockage" at test time.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as TF

from .config import Config
from .asta import ASTAEncoder


def _grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Gradient-reversal layer: forward identity, backward multiply by -lambda."""
    return GradReverse.apply(x, lambd)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class MicroEncoder(nn.Module):
    """h_phi: ASTA features -> condition vector c."""

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        # pool the (T, M) tokens to a vector
        self.pool_attn = nn.Linear(d, 1)
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.micro_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.micro_dim * 2, cfg.micro_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, M, d) -> c: (B, micro_dim)."""
        B, T, M, D = h.shape
        a = self.pool_attn(h).squeeze(-1)         # (B, T, M)
        w = TF.softmax(a.view(B, T * M), dim=-1).view(B, T, M)
        w = w.unsqueeze(-1)
        pooled = (h * w).sum(dim=(1, 2))          # (B, D)
        return self.mlp(pooled)                   # (B, micro_dim)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: H' = gamma(c) * H + beta(c)."""

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.gamma_net = nn.Linear(cfg.micro_dim, d)
        self.beta_net = nn.Linear(cfg.micro_dim, d)
        # init so that gamma=1, beta=0 at the start (FiLM is identity)
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.ones_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """h: (B, T, M, d)  c: (B, micro_dim) -> (B, T, M, d)."""
        gamma = 1.0 + self.gamma_net(c)           # bias starts at 1
        beta = self.beta_net(c)                   # bias starts at 0
        # broadcast over T, M
        gamma = gamma.unsqueeze(1).unsqueeze(1)
        beta = beta.unsqueeze(1).unsqueeze(1)
        return gamma * h + beta


class Adversary(nn.Module):
    """Predicts the macro env from the condition vector c (must fail)."""

    def __init__(self, cfg: Config, n_envs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.micro_dim, cfg.adv_hidden),
            nn.GELU(),
            nn.Linear(cfg.adv_hidden, n_envs),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


class DisentangledModel(nn.Module):
    """Wraps the ASTA encoder, the micro encoder, the FiLM, and the adversary
    so we can train them with a single forward pass.

    The gradient-reversal (GR) layer lives *inside* the adversary path:
    forward = identity, backward = -lambda. So when the featurizer is
    optimized on a loss that *includes* adv_ce (i.e. `L = task + adv_ce`),
    the gradient that flows back into c from the adversary branch is
    REVERSED — pushing c AWAY from being predictive of env. The
    adversary's own weights, by contrast, see the *un-reversed* gradient
    and step in the normal minimizing direction.

    This is the standard DANN pattern. Use this with a single optimizer
    over all parameters and a loss of the form
        L = task_loss + lambda * CE(D(GR(c)), env)
    where the `+` sign is correct (NOT `-`) because the GR sign flip
    happens inside the model.

    For the two-optimizer pattern, pass `detach_c_for_adv=True` in the
    adversary-only forward pass to stop gradients from flowing into the
    encoder / h_phi / FiLM.
    """

    def __init__(self, cfg: Config, n_envs: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = ASTAEncoder(cfg)
        self.h_phi = MicroEncoder(cfg)
        self.film = FiLM(cfg)
        self.adv = Adversary(cfg, n_envs)

    def forward(self, x: torch.Tensor, detach_c_for_adv: bool = False):
        """Returns (h_prime, c, adv_logits).

        detach_c_for_adv: if True, c is detached before going into the
        adversary. Backward from the adversary branch will then only
        update D's own weights, not the encoder / h_phi / FiLM.
        """
        h = self.encoder(x)                       # (B, T, M, d)
        c = self.h_phi(h)                         # (B, micro_dim)
        h_prime = self.film(h, c)                 # (B, T, M, d)
        c_for_adv = c.detach() if detach_c_for_adv else c
        c_rev = _grad_reverse(c_for_adv, lambd=self.cfg.adv_lambda)
        adv_logits = self.adv(c_rev)              # (B, n_envs)
        return h_prime, c, adv_logits


def disentangle_loss(
    adv_logits: torch.Tensor,
    env_ids: torch.Tensor,
    n_envs: int,
    task_loss: torch.Tensor,
    lambda_adv: float = 1.0,
) -> torch.Tensor:
    """Total loss = task_loss + lambda_adv * adv_ce.

    The `+` sign is correct (NOT `-`) because the gradient-reversal
    layer inside `DisentangledModel.forward` flips the sign of the
    adversary's gradient w.r.t. c during the backward pass. So the
    featurizer (encoder / h_phi / FiLM) sees a gradient that pushes c
    AWAY from being predictive of env (DANN disentanglement pressure).
    The adversary's own weights, by contrast, see the un-reversed
    gradient and step in the normal minimizing direction.
    """
    adv_ce = TF.cross_entropy(adv_logits, env_ids)
    return task_loss + lambda_adv * adv_ce
