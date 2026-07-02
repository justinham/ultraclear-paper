"""Tier 3 — Downstream heads on top of disentangled features.

Three heads share the same encoder trunk:
  - NLOS Identification: per-anchor probability (binary classification).
  - Signal Restoration: per-(anchor, time, channel) denoised measurement.
  - Location: final 2D (x, y) estimate.

The location head sits *on top of* the restoration output so the
restoration head is implicitly regularized to produce measurements that
are geometrically consistent with the location.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as TF

from .config import Config
from .asta import ASTAEncoder


class NLOSHead(nn.Module):
    """Per-(anchor, time) binary NLOS probability."""

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, M, d) -> logits (B, T, M)."""
        return self.net(h).squeeze(-1)


class RestorationHead(nn.Module):
    """Predicts the F-channel denoised values at every (anchor, time) cell."""

    def __init__(self, d_model: int, n_channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_channels),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, M, d) -> (B, T, M, F)."""
        return self.net(h)


class LocationHeadFromRestore(nn.Module):
    """Maps the *restored* measurements (B, T, M, F) to a 2D location."""

    def __init__(self, n_anchors: int, n_channels: int, hidden: int = 64):
        super().__init__()
        in_dim = n_anchors * n_channels
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, restored: torch.Tensor) -> torch.Tensor:
        """restored: (B, T, M, F) -> (B, 2). Uses the *middle* timestep
        (consistent with how the synthetic labels are defined)."""
        B, T, M, F = restored.shape
        x = restored[:, T // 2]                   # (B, M, F)
        x = x.reshape(B, M * F)                   # (B, M*F)
        return self.net(x)


class DownstreamModel(nn.Module):
    """End-to-end model: encoder + 3 heads."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = ASTAEncoder(cfg)
        self.nlos_head = NLOSHead(cfg.d_model)
        self.restore_head = RestorationHead(cfg.d_model, cfg.n_channels)
        self.loc_head = LocationHeadFromRestore(cfg.n_anchors, cfg.n_channels)

    def forward(self, x: torch.Tensor):
        """x: (B, T, M, F) -> dict with 'nlos_logits', 'restored', 'loc'."""
        h = self.encoder(x)                       # (B, T, M, d)
        nlos_logits = self.nlos_head(h)           # (B, T, M)
        restored = self.restore_head(h)           # (B, T, M, F)
        loc = self.loc_head(restored)             # (B, 2)
        return {
            "nlos_logits": nlos_logits,
            "restored": restored,
            "loc": loc,
        }
