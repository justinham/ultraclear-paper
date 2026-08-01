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
    encoder output, with a hard skip-connection that preserves the raw
    input at *visible* cells.

    Why the hard skip: in the joint Tier 1+2 trainer, the encoder features
    are passed to the location head, but the head also benefits from the
    raw ToA values at visible cells. If the encoder has to choose between
    being a good "denoiser" (helps masked cells) and a good "passthrough"
    (helps visible cells), it picks one. A 50K-param encoder can't do
    both well simultaneously, and the SSL loss only scores masked cells
    -- so the encoder learns to denoise at the cost of mangling visible
    cells. On held-out scenes, the visible-cell RMSE blows up to
    ~1.8 m for ToA (vs the 0.6 m masked-cell RMSE) and ~52 degrees for
    AoA. The hard skip removes this pathology: at visible cells the head
    just outputs the input (zero reconstruction error), and the encoder
    only has to learn the inference task at masked cells.

    The head is otherwise a 2-layer MLP applied to the encoder features,
    same as the original SSLHead.
    """

    def __init__(self, d_model: int, n_channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_channels),
        )

    def forward(
        self,
        h: torch.Tensor,
        x_input: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """h: (B, T, M, d_model) encoder features.
        x_input: (B, T, M, F) the ORIGINAL (un-masked) input -- the
            skip-connection source.
        mask: (B, T, M) bool, True at masked positions.
        Returns predictions of shape (B, T, M, F) where:
            - at visible cells (~mask): the raw x_input value (passthrough)
            - at masked cells (mask): the head's inference from h
        """
        inferred = self.net(h)                                # (B, T, M, F)
        out = torch.where(mask.unsqueeze(-1), inferred, x_input)
        return out


class SSLPretrainer(nn.Module):
    """Encoder + masked-prediction head with hard skip-connection.

    The skip-connection means the model output equals the input at
    visible cells (zero reconstruction error there) and equals the
    head's prediction at masked cells. This keeps the encoder's
    pretraining loss focused on the inference task only -- the
    passthrough is a freebie.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = ASTAEncoder(cfg)
        self.head = SSLHead(cfg.d_model, cfg.n_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, T, M, F) raw measurements.
        Returns (x_pred, mask, x_input) where:
            - x_pred: prediction at every cell (passthrough at visible,
                head(encoder) at masked)
            - mask: (B, T, M) bool, True at masked positions
            - x_input: the original input, exposed so the loss function
                and downstream code can access the un-masked measurements
        """
        x_masked, mask = random_mask(x, self.cfg.mask_ratio)
        h = self.encoder(x_masked)
        x_pred = self.head(h, x, mask)
        return x_pred, mask, x


def ssl_loss(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    mask: torch.Tensor,
    channel_weights: torch.Tensor | None = None,
    loss_on_visible: bool = False,
    visible_weight: float = 1.0,
) -> torch.Tensor:
    """Weighted MSE on masked cells, optionally plus visible cells.

    With the hard-skip SSLHead, the visible-cell loss is identically zero
    (the head outputs the input). The `loss_on_visible` flag is now mostly
    a no-op for the default model -- it would only matter if the head were
    changed to a soft/gated skip that produces non-input outputs at visible
    cells. We keep the flag for backward compatibility with the
    `train_tier0_sionna.py` CLI and the DAE-style ablation.

    x_pred, x_true: (B, T, M, F).
    mask: (B, T, M) bool.
    channel_weights: (F,) tensor of non-negative weights, applied per
        channel *before* the mean over channels. If None, all channels
        get weight 1.0 (the historical F=1 default; for F=2 this means
        ToA and AoA contribute equally to the loss, which after per-
        channel z-scoring is the natural 50/50 split).
    loss_on_visible: if True, also include the visible cells in the
        loss. With the hard-skip head this contributes zero; with a
        soft-skip head it would penalize deviations from passthrough.
    visible_weight: scale factor on the visible-cell loss term.

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
    n_total = float(diff.numel())                  # B*T*M cells
    n_masked = mask.float().sum().clamp_min(1.0)
    n_visible = (n_total - n_masked).clamp_min(1.0)
    diff_masked = diff * mask.float()               # zero out non-masked
    masked_loss = diff_masked.sum() / n_masked
    if loss_on_visible:
        diff_visible = diff * (~mask).float()      # zero out masked
        visible_loss = diff_visible.sum() / n_visible
        return masked_loss + visible_weight * visible_loss
    return masked_loss
