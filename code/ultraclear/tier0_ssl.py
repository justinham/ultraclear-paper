"""Tier 0 — Self-supervised pretraining via masked prediction.

We mask a fraction of (anchor, timestep) cells in the input, run the
encoder, and ask a small head to predict the original measurement
values at the masked positions. The encoder is the same ASTA encoder
that downstream tiers will use; only the head is added here.

The pretraining loss is MSE on the *masked* cells, so the model is
forced to interpolate from visible context. This is directly useful
for the downstream restoration head.
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as TF

from .config import Config
from .asta import ASTAEncoder


def random_mask(
    x: torch.Tensor, mask_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask (anchor, time) cells uniformly at random.

    x: (B, T, M, F) -> x_masked: same shape, masked cells zeroed;
                     mask: (B, T, M) bool, True at masked positions.
    """
    B, T, M, F_ = x.shape
    mask = torch.rand(B, T, M, device=x.device) < mask_ratio
    x_masked = x.clone()
    x_masked[:, :, :, :] = torch.where(
        mask.unsqueeze(-1), torch.zeros_like(x), x
    )
    return x_masked, mask


class SSLHead(nn.Module):
    """Predicts the original F-channel values at masked positions from the
    encoder output. We pool the encoder state at the masked (T, M) cells
    and run a small MLP.
    """

    def __init__(self, d_model: int, n_channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_channels),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h: (B, T, M, d_model)  mask: (B, T, M) bool.
        Returns predictions of shape (B, T, M, F), at *all* positions;
        caller will select masked positions for the loss.
        """
        return self.net(h)


class SSLPretrainer(nn.Module):
    """Encoder + masked-prediction head."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = ASTAEncoder(cfg)
        self.head = SSLHead(cfg.d_model, cfg.n_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, M, F) raw measurements.
        Returns (x_pred, mask) where x_pred is the prediction at every cell.
        """
        x_masked, mask = random_mask(x, self.cfg.mask_ratio)
        h = self.encoder(x_masked)
        x_pred = self.head(h, mask)
        return x_pred, mask


def ssl_loss(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    mask: torch.Tensor,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted MSE on masked cells only.

    x_pred, x_true: (B, T, M, F).
    mask: (B, T, M) bool.
    channel_weights: (F,) tensor of non-negative weights, applied per
        channel *before* the mean over channels. If None, all channels
        get weight 1.0 (the historical F=1 default; for F=2 this means
        ToA and AoA contribute equally to the loss, which after per-
        channel z-scoring is the natural 50/50 split).

    We intentionally keep the per-channel weights as a multiplicative
    factor on the squared error *before* averaging over channels, so
    that the loss is in the same units regardless of F. The `n_masked`
    denominator is the number of (b, t, m) cells where mask is True
    (NOT the per-channel count, which would change units).
    """
    diff = (x_pred - x_true) ** 2                  # (B, T, M, F)
    if channel_weights is not None:
        # broadcast (F,) over (B, T, M, F)
        diff = diff * channel_weights.view(1, 1, 1, -1).to(diff.device)
    diff = diff.mean(dim=-1)                       # (B, T, M)  weighted avg over F
    diff = diff * mask.float()                     # zero out non-masked
    n_masked = mask.float().sum().clamp_min(1.0)
    return diff.sum() / n_masked
