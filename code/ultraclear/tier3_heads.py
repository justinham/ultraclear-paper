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


class AttentionUWB(nn.Module):
    """Simple self-attention UWB model.

    Direct port of `SimpleAttentionUWB` from
    `/Users/justin/Downloads/srp_research/train_attention_baseline.py`,
    adapted to the DownstreamModel output contract
    `{'nlos_logits', 'restored', 'loc'}` so the supervised baseline
    trainer can swap it in for `DownstreamModel` unchanged.

    Architecture differences from `DownstreamModel`:
      - Backbone: nn.TransformerEncoder over flattened (T*M) tokens
        with learned positional encoding. Simpler than the GCN+STA
        blocks in `ASTAEncoder` and a better fit for stationary-tag
        per-window predictions.
      - Per-anchor heads: mean-pool over the time dimension, then
        per-anchor MLP. Gives one prediction per (anchor, channel)
        that's shared across timesteps — the right inductive bias
        for a stationary tag.
      - Position head: `Linear(M*embed_dim, 2)` over the mean-pooled
        per-anchor embeddings, so the position prediction has direct
        access to all anchors' representations (vs `DownstreamModel`'s
        `LocationHeadFromRestore` which only reads the middle timestep's
        restored signal).

    Output shapes match DownstreamModel so the trainer doesn't change:
      - nlos_logits: (B, T, M)   (broadcasted from per-anchor)
      - restored:    (B, T, M, F) (broadcasted from per-anchor)
      - loc:         (B, 2)
    """

    def __init__(self, cfg: Config,
                 embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Input embedding: project (F) -> embed_dim
        self.input_embed = nn.Linear(cfg.n_channels, embed_dim)

        # Learned positional encoding (T*M tokens max)
        self.pos_encoder = nn.Parameter(torch.randn(1, 256, embed_dim) * 0.1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Per-anchor heads (operate on mean-pooled-over-time embeddings)
        self.nlos_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.restore_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, cfg.n_channels),
        )
        # Position head: flatten all anchors' embeddings -> 2D position
        self.pos_head = nn.Sequential(
            nn.Linear(embed_dim * cfg.n_anchors, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor):
        """x: (B, T, M, F) -> dict with 'nlos_logits', 'restored', 'loc'."""
        B, T, M, F = x.shape

        # Flatten T and M into one token sequence: (B, T*M, F)
        tokens = x.reshape(B, T * M, F)
        # Embed to d_model
        h = self.input_embed(tokens)                     # (B, T*M, d)
        # Add positional encoding
        h = h + self.pos_encoder[:, : T * M, :]
        # Self-attention over all T*M tokens
        h = self.transformer(h)                          # (B, T*M, d)
        # Reshape back to (B, T, M, d) and mean-pool over time
        h = h.reshape(B, T, M, self.embed_dim)
        h_anchor = h.mean(dim=1)                         # (B, M, d)

        # Per-anchor predictions
        nlos_per_anchor = self.nlos_head(h_anchor).squeeze(-1)   # (B, M)
        restored_per_anchor = self.restore_head(h_anchor)        # (B, M, F)

        # Position: flatten all anchors' embeddings
        loc = self.pos_head(h_anchor.reshape(B, -1))     # (B, 2)

        # Broadcast per-anchor outputs across T so the trainer's
        # `restore_target` (B, T, M, F) and `nlos` (B, T, M) match.
        return {
            "nlos_logits": nlos_per_anchor.unsqueeze(1).expand(-1, T, -1),  # (B, T, M)
            "restored": restored_per_anchor.unsqueeze(1).expand(-1, T, -1, -1),  # (B, T, M, F)
            "loc": loc,
        }
