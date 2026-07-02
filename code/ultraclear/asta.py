"""Clean-room simplified ASTA encoder.

Three pieces, in order:
  1. CausalTemporalConv  — per-channel 1D conv with left-only padding.
                            Encodes the temporal stream of each (anchor, channel).
  2. DynamicDualEdgeGCN  — message passing on a learned anchor-adjacency
                            (channel-specific weighting), with self-loops.
  3. MultiHeadSelfAttn   — global attention across the flattened T*M tokens
                            to capture cross-sensor long-range dependencies.

Input:  x of shape (B, T, M, F)
Output: h of shape (B, T, M, d_model)

This is a *simplified* version of the paper's full ASTA-Net — same ideas,
fewer parameters, designed to be small enough to MAML over in seconds on CPU.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as TF

from .config import Config


class CausalTemporalConv(nn.Module):
    """Per-(anchor, channel) causal 1D conv that mixes T with a learned filter."""

    def __init__(self, channels: int, kernel: int, d_out: int):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(channels, d_out, kernel_size=kernel, groups=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, M, F) -> convolve along T per (B, M) with F input channels.
        B, T, M, F_in = x.shape
        # merge B and M, put channels (F_in) and length (T) in the right slots
        x = x.permute(0, 2, 3, 1).contiguous().view(B * M, F_in, T)  # (B*M, F_in, T)
        # left-pad by (kernel-1) so output length == T and no future leakage
        x = TF.pad(x, (self.kernel - 1, 0))
        y = self.conv(x)                                          # (B*M, d_out, T)
        y = y.view(B, M, -1, T).permute(0, 3, 1, 2).contiguous()  # (B, T, M, d_out)
        return y


class DynamicDualEdgeGCN(nn.Module):
    """Lightweight GCN over the M anchor nodes per timestep.

    We learn a *channel-specific* base adjacency W ∈ R^{M×M} per channel,
    then add a self-loop (the "intra-node ToA-AoA coupling" in the paper).
    Adjacency is fixed across the batch (we learn a global graph structure
    tied to the deployment topology, not sample-specific), which keeps this
    module cheap and MAML-friendly.
    """

    def __init__(self, d_in: int, d_out: int, n_anchors: int, n_channels: int,
                 lambda_self: float = 0.5):
        super().__init__()
        self.M = n_anchors
        self.F = n_channels
        self.lambda_self = lambda_self
        # per-channel adjacency; sigmoid keeps edge weights in [0,1]
        self.A_ch = nn.Parameter(torch.randn(n_channels, n_anchors, n_anchors) * 0.1)
        # per-channel linear on the input features
        self.lin = nn.Linear(d_in * n_channels, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, M, d_model) — reshape last dim into (F, d_model/F) for the
        per-channel message passing, then aggregate, then project back to d_model.
        """
        B, T, M, D = x.shape
        assert D % self.F == 0, f"input last dim {D} must be divisible by n_channels {self.F}"
        per_ch = D // self.F
        x_split = x.view(B, T, M, self.F, per_ch)                 # (B,T,M,F,per_ch)
        # adjacency per channel
        A = torch.sigmoid(self.A_ch)                              # (F, M, M) in [0,1]
        I = torch.eye(M, device=x.device).expand(self.F, M, M)    # (F, M, M)
        A_hat = A + self.lambda_self * I                          # (F, M, M)
        A_hat = A_hat / (A_hat.sum(-1, keepdim=True) + 1e-6)      # row-normalize
        # message passing across anchors per channel
        # A_hat: (F, M, M), x_split: (B, T, M, F, per_ch) -> out (B, T, M, F, per_ch)
        # want: out[b, t, i, f, d] = sum_j A_hat[f, i, j] * x_split[b, t, j, f, d]
        agg = torch.einsum('fij,btjfd->btifd', A_hat, x_split)   # (B,T,M,F,per_ch)
        agg = agg.permute(0, 1, 2, 3, 4).contiguous().view(B, T, M, D)
        return self.lin(agg)                                       # (B, T, M, d_out)


class ASTABlock(nn.Module):
    """One stack: GCN -> attention -> residual + LayerNorm.

    The TCN is *only* used once at the input projection (see ASTAEncoder).
    Within a block we apply the GCN over the channel-decomposed features
    (per-channel adjacency weighting), then a multi-head self-attention
    over the flattened T*M tokens, with residuals and a small FFN.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        # GCN splits the d_model dim into (n_channels * per_ch).  We let
        # per_ch be whatever fits cleanly, and the GCN's final linear projects
        # back to d_model.
        per_ch = max(1, cfg.d_model // cfg.n_channels)
        self.gcn = DynamicDualEdgeGCN(
            d_in=per_ch, d_out=cfg.d_model,
            n_anchors=cfg.n_anchors, n_channels=cfg.n_channels,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model, num_heads=cfg.n_attention_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, M, d_model) — already projected from raw F by the encoder
        h = self.gcn(x)                                           # (B,T,M,d)
        h = self.norm1(x + h)                                     # residual
        # attention over (T*M) tokens
        B, T, M, D = h.shape
        flat = h.view(B, T * M, D)
        a, _ = self.attn(flat, flat, flat)
        a = a.view(B, T, M, D)
        h = self.norm2(h + a)
        h = h + self.ffn(h)
        return h


class ASTAEncoder(nn.Module):
    """Input projection (raw F -> d_model, via 1x1 conv-like linear) + a single
    causal TCN to mix time, then a stack of ASTA blocks.

    Causal mixing is split: the input TCN runs once at the front (to get
    past data into each timestep), and per-block we re-attend the global
    token structure. This keeps the per-block path cheap for MAML.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # input projection: per-(anchor) project F raw channels into d_model
        self.input_proj = nn.Linear(cfg.n_channels, cfg.d_model)
        # one causal TCN up front for the temporal mix
        self.tcn_in = CausalTemporalConv(cfg.n_channels, cfg.tcn_kernel, cfg.d_model)
        self.blocks = nn.ModuleList([ASTABlock(cfg) for _ in range(cfg.n_asta_blocks)])
        self.out_dim = cfg.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, M, F)
        h = self.tcn_in(x)                                        # (B,T,M,d)
        for blk in self.blocks:
            h = blk(h)
        return h                                                   # (B,T,M,d)
