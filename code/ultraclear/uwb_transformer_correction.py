"""UWB Transformer for TDoA position-error correction.

This module implements a transformer-based corrector that takes:
  - the raw complex Channel Impulse Response (CIR) for N anchors,
    each represented as a length-L sequence of (I, Q) samples
  - the physical (X, Y, Z) coordinates of those N anchors

and outputs a 3D position offset (Δx, Δy, Δz) that, when added to
a noisy initial position estimate, recovers the true tag location.

PARKED — model class only, no training script yet. The companion
training script will live in scripts/train_uwb_transformer_correction.py
when the corrector work is picked back up. The CIR data for the
office_rack sionna track is essentially zero across all 23 anchors
and all 3 scenes (max |CIR|^2 < 1e-5) as of 2026-07-23, so this
model is not yet trainable on that data; a future task is to
either fix the data generation or use a different CIR source.

Key design choices (matching the original spec):
  - cir_length=152 in the original spec; this implementation
    accepts any cir_length that is divisible by patch_size.
    Use patch_size=6 for cir_length=60 (10 patches), or
    patch_size=8 for cir_length=152 (19 patches).
  - 3D anchor positions are projected through a 2-layer MLP
    into the embedding dim, then broadcast over the patch
    dimension (so each anchor's spatial encoding is shared
    across all of its CIR patches).
  - 1D patch-index positional encoding is added so the model
    knows which delay bin each patch corresponds to.
  - The transformer encoder flattens (N_anchors × Num_Patches)
    into a single sequence, so self-attention can mix information
    across BOTH anchors (inter-anchor multipath correlation) AND
    patches (intra-anchor multipath).
  - Global mean pooling across all (anchor, patch) tokens,
    followed by a small MLP head, outputs the (Δx, Δy, Δz) offset.

Reference (original spec):
  #1D Patching Layer to partition raw Channel Impulse Responses
   (CIRs) into continuous segment tokens.
  #3D Spatial Positional Encodings to project the physical (X,Y,Z)
   coordinates of each anchor directly onto the patch tokens.
  #Transformer Encoder to model inter-anchor multipath correlations.
  #Position Correction Head to directly output a 3D location
   offset (Δx, Δy, Δz).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Sub-modules
# --------------------------------------------------------------------------- #

class AnchorSpatialPositionalEncoding(nn.Module):
    """Project physical 3D anchor coordinates (X, Y, Z) into the
    Transformer's d_model embedding dimension.

    The output is broadcastable over the patch dimension so that
    the same spatial encoding is added to every CIR patch of a
    given anchor.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.coord_proj = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

    def forward(self, anchor_coords: torch.Tensor) -> torch.Tensor:
        # anchor_coords: (B, N_anchors, 3)
        # returns:      (B, N_anchors, 1, embed_dim)
        spatial_emb = self.coord_proj(anchor_coords)
        return spatial_emb.unsqueeze(2)


class CIRPatchEmbedding(nn.Module):
    """Split a 1D CIR sequence into non-overlapping patches of size
    `patch_size` and linearly project each patch to embed_dim.

    Input:  (B, N_anchors, CIR_length)
    Output: (B, N_anchors, Num_Patches, embed_dim)

    `Num_Patches = CIR_length // patch_size`. The trailing taps
    beyond the last full patch are dropped (we require
    `CIR_length % patch_size == 0`).
    """
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, cir: torch.Tensor) -> torch.Tensor:
        # cir: (B, N, L)
        B, N, L = cir.shape
        assert L % self.patch_size == 0, (
            f"CIR length {L} must be divisible by patch_size "
            f"{self.patch_size} (last {(L % self.patch_size)} tap(s) would be dropped)"
        )
        num_patches = L // self.patch_size
        # (B, N, L) -> (B, N, num_patches, patch_size) -> (B, N, num_patches, embed_dim)
        cir_patched = cir.view(B, N, num_patches, self.patch_size)
        return self.projection(cir_patched)


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #

def load_office_rack_anchor_positions(
    csv_path: "str | Path | None" = None,
) -> "torch.Tensor":
    """Load the 23 office-rack anchor (X, Y, Z) positions into a
    (23, 3) float32 tensor, ordered An1..An23 to match the
    office_rack sionna CSVs' `An_measured_dist_m` columns.

    The default path is
    `code/data/office_rack_track_sionna/anchor_positions.csv` —
    hand-curated from the deployment diagram, listing every
    anchor's physical (X, Y, Z) coordinates in meters and a
    short note ('ceiling' / 'floor' / 'rack-near').

    The CSV has two id columns:
      - `col_id`  : An1..An23 — the column label used in the
                    office_rack CSVs (e.g. `A8_measured_dist_m`
                    in the CSV corresponds to the anchor with
                    `col_id == "A8"`)
      - `anchor_id`: a1..a23 — the physical anchor name from
                    the deployment diagram (A1 in the CSV
                    might be physically `a1` or `a10` etc.)

    This helper returns positions in the `col_id` (An1..An23)
    order, so a caller can use:

        anchor_xyz = load_office_rack_anchor_positions()
        d_A8 = df['A8_measured_dist_m'].iloc[0]   # n-th column
        ax, ay, az = anchor_xyz[7]                # n-th row, 0-indexed

    Anchor layout summary (23 anchors total):
      - 19 ceiling (z=2.85) — 15 perimeter + 4 rack-near
        + 2 mid-rack (a22, a23) which are also "rack-near"
      -  4 floor (z=0.30) — A5, A7, A12, A18 (the floor twins
        of A4, A6, A1, A17 respectively)

    Room footprint: 30m × 11m (x ∈ [-15, 15], y ∈ [-5.5, 5.5]).

    Returns:
        anchor_xyz: (23, 3) float32 tensor, An1..An23 order.
    """
    from pathlib import Path
    if csv_path is None:
        csv_path = (Path(__file__).resolve().parent.parent
                    / "data" / "office_rack_track_sionna"
                    / "anchor_positions.csv")
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "col_id" not in df.columns:
        raise ValueError(
            f"expected a 'col_id' column (An1..An23) in {csv_path}; "
            f"found columns {list(df.columns)}. The CSV should be in "
            f"the An1..An23 order matching the office_rack CSVs."
        )
    # Sort by col_id to be safe (we rely on the CSV being in the
    # right order, but explicit sort makes the helper robust to
    # someone re-ordering the rows).
    df = df.copy()
    df["col_num"] = df["col_id"].str.replace("A", "", regex=False).astype(int)
    df = df.sort_values("col_num").reset_index(drop=True)
    if len(df) != 23:
        raise ValueError(
            f"expected 23 anchors in {csv_path}, found {len(df)}"
        )
    out = []
    for _, row in df.iterrows():
        out.append([float(row.X_m), float(row.Y_m), float(row.Z_m)])
    import torch
    return torch.tensor(out, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #

class UWBTransformerCorrection(nn.Module):
    """Transformer encoder for LSE-residual position correction.

    Takes three inputs:
      - raw CIR for N anchors (length-L real-valued sequence each)
      - physical (X, Y, Z) positions of those N anchors
      - an initial LSE position estimate (x, y, z) for the tag

    Outputs a (Δx, Δy, Δz) offset such that
        corrected_pos = lse_pred + delta_pos
    is a better estimate of the true tag location.

    The intuition: the LSE baseline gives a reasonable (but biased)
    starting point. The transformer reads the CIR and learns to
    predict the residual that, when added, gives the true location.
    The model is essentially learning "given this LSE guess and
    this CIR, how do I correct it?"

    Forward signature
    -----------------
        cir_data:      (B, N_anchors, CIR_length) float32
                       Raw CIR — the model's primary input. For a
                       real complex CIR you should pre-process to
                       (I, Q) → real/imag or |CIR|^2 before passing
                       in (this class works on real values only).
        anchor_coords: (B, N_anchors, 3)            float32
                       Physical 3D positions of the anchors in
                       the same coordinate frame as the tag.
        lse_pred:      (B, 3)                       float32
                       Initial LSE position estimate (x, y, z).
                       The model outputs the offset to ADD to
                       this estimate to get the corrected position.

    Output
    ------
        delta_pos:     (B, 3)                       float32
                       Estimated 3D position correction vector
                       (Δx, Δy, Δz). Add this to lse_pred to
                       recover a more accurate tag location.
    """
    def __init__(
        self,
        cir_length: int = 152,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ----- patch geometry -----
        if cir_length % patch_size != 0:
            raise ValueError(
                f"cir_length ({cir_length}) must be divisible by "
                f"patch_size ({patch_size})"
            )
        self.cir_length = cir_length
        self.patch_size = patch_size
        self.num_patches = cir_length // patch_size
        self.embed_dim = embed_dim

        # ----- CIR patching + 3D spatial PE + temporal patch-index PE -----
        self.cir_embedder = CIRPatchEmbedding(patch_size, embed_dim)
        self.spatial_pe = AnchorSpatialPositionalEncoding(embed_dim)
        # (1, 1, num_patches, embed_dim) — added to every anchor's patches
        self.temporal_pe = nn.Parameter(
            torch.zeros(1, 1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        # ----- Transformer encoder stack -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ----- Position-correction MLP head -----
        # Concatenate the pooled CIR features (embed_dim) with the
        # LSE prediction (3) before the MLP head, so the head can
        # learn to adjust the LSE estimate based on what the CIR
        # features suggest.  The head therefore reads
        #     [pooled_features ; lse_pred]    (embed_dim + 3)
        # and outputs (Δx, Δy, Δz) — the residual to add.
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim + 3, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),       # (Δx, Δy, Δz)
        )

    def forward(
        self,
        cir_data: torch.Tensor,
        anchor_coords: torch.Tensor,
        lse_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            cir_data:      (B, N_anchors, CIR_length) float32
            anchor_coords: (B, N_anchors, 3)            float32
            lse_pred:      (B, 3)                       float32
                           Initial LSE position estimate.

        Returns:
            delta_pos:     (B, 3) float32 — offset to add to
                           lse_pred for a corrected estimate.
        """
        B, N, L = cir_data.shape
        if N != anchor_coords.shape[1]:
            raise ValueError(
                f"Number of anchors in CIR ({N}) doesn't match anchor "
                f"coords ({anchor_coords.shape[1]})"
            )
        if L != self.cir_length:
            raise ValueError(
                f"CIR length {L} doesn't match model.cir_length "
                f"{self.cir_length}"
            )
        if lse_pred.shape != (B, 3):
            raise ValueError(
                f"lse_pred shape {tuple(lse_pred.shape)} doesn't match "
                f"expected (B, 3) = ({B}, 3)"
            )

        # A. Patch tokens: (B, N, num_patches, embed_dim)
        tokens = self.cir_embedder(cir_data)

        # B. Spatial PE: (B, N, 1, embed_dim) — broadcast over patches
        spatial_emb = self.spatial_pe(anchor_coords)
        # Temporal PE: (1, 1, num_patches, embed_dim) — broadcast over batch & anchors
        tokens = tokens + spatial_emb + self.temporal_pe

        # C. Flatten anchors and patches into one sequence
        #    (B, N, num_patches, embed_dim) -> (B, N*num_patches, embed_dim)
        sequence = tokens.view(B, N * self.num_patches, -1)

        # D. Transformer encoder
        encoded_tokens = self.transformer(sequence)   # (B, N*num_patches, embed_dim)

        # E. Global mean pooling over (anchor, patch) tokens
        pooled_features = torch.mean(encoded_tokens, dim=1)   # (B, embed_dim)

        # F. Concatenate with the LSE prediction so the head can
        #    condition the residual on both the CIR features and
        #    the current LSE estimate.  This is what makes the
        #    model a *residual corrector* rather than a direct
        #    regressor: the head learns "given that LSE says X,
        #    and the CIR looks like this, what correction Δ would
        #    get me closer to the true location?"
        head_input = torch.cat([pooled_features, lse_pred], dim=-1)  # (B, embed_dim+3)

        # G. Predict (Δx, Δy, Δz)
        delta_pos = self.mlp_head(head_input)             # (B, 3)
        return delta_pos

    def predict(
        self,
        cir_data: torch.Tensor,
        anchor_coords: torch.Tensor,
        lse_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper: returns corrected_pos = lse_pred + delta_pos
        instead of the raw residual. Use this at inference time."""
        return lse_pred + self.forward(cir_data, anchor_coords, lse_pred)

    # ------------------------------------------------------------------ #
    # Convenience constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def for_office_rack_cir(
        cls,
        cir_length: int = 60,
        patch_size: int = 6,
        **kwargs,
    ) -> "UWBTransformerCorrection":
        """Build a model sized for the office_rack sionna track CIR.

        Defaults: cir_length=60, patch_size=6 → 10 patches per
        anchor. Override either argument to use a different geometry.

        Common alternatives for cir_length=60:
            patch_size=4  → 15 patches
            patch_size=5  → 12 patches
            patch_size=10 →  6 patches
        """
        return cls(
            cir_length=cir_length,
            patch_size=patch_size,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# Distance-correction variant (2026-07-23 16:35 EDT)
# --------------------------------------------------------------------------- #
#
# The position-correction model above outputs (Δx, Δy) — a
# *position residual* that, when added to an LSE estimate,
# refines the tag location. It needs the LSE estimate as a
# third input.
#
# The distance-correction variant below is a different
# decomposition: instead of correcting the position directly,
# it predicts a per-anchor *distance residual* (Δd₁, Δd₂, ...,
# Δd₂₃) that, when added to the measured ToA distances, gives
# the "corrected" distances. The position is then recovered
# by running the LSE on the corrected distances. The two
# pieces of work (learn the NLOS bias per anchor, then solve
# for geometry) are now separated:
#
#   measured_dist  +  Δd  →  corrected_dist
#   LSE(corrected_dist, anchor_xyz)  →  tag position
#
# This is closer to the data the simulation gives us — the
# measured ToA is the noisy observation, and the NLOS bias is
# the per-anchor error that needs to be subtracted. Predicting
# the bias directly (per-anchor) is a simpler supervised
# target than predicting the integrated position error.
#
# The input is the same shape as the position model: (B, N,
# CIR_length). The output is (B, N) — one scalar per anchor.
# The architecture is the same CIR-patches + transformer +
# pool, but the head is per-anchor (one logit per anchor) and
# reads back out of the per-anchor tokens BEFORE the global
# mean pool that the position model uses.
#
# Training target:
#   per-anchor residual d_i = gt_dist_i − measured_dist_i
# in meters.
#
# Eval:
#   corrected_dist = measured_dist + model(...)
#   pos = LSE(corrected_dist, anchor_xyz)
# and compare pos to tag (x, y).
# --------------------------------------------------------------------------- #

class UWBTransformerDistanceCorrection(nn.Module):
    """Transformer encoder that predicts a per-anchor distance
    residual (Δd_1, ..., Δd_N) for refining LSE.

    The model's output is a per-anchor correction, not a
    position correction. To get a position estimate you then
    run LSE on the corrected distances (see
    `scripts/lse_office_rack.py`).

    Forward signature
    -----------------
        cir_data:      (B, N_anchors, CIR_length) float32
                       Raw CIR (or any per-tap feature).
        anchor_coords: (B, N_anchors, 3)            float32
                       Physical 3D positions of the anchors.

    Output
    ------
        delta_d:       (B, N_anchors)              float32
                       Per-anchor distance correction
                       vector in meters. Add to the
                       *measured* distance to get a
                       corrected distance, then run LSE
                       on the corrected distances.
    """
    def __init__(
        self,
        cir_length: int = 152,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if cir_length % patch_size != 0:
            raise ValueError(
                f"cir_length ({cir_length}) must be divisible by "
                f"patch_size ({patch_size})"
            )
        self.cir_length = cir_length
        self.patch_size = patch_size
        self.num_patches = cir_length // patch_size
        self.embed_dim = embed_dim

        # Same patch + 3D spatial + temporal PE as the position
        # model. We reuse the implementation by composing the
        # existing sub-modules.
        self.cir_embedder = CIRPatchEmbedding(patch_size, embed_dim)
        self.spatial_pe = AnchorSpatialPositionalEncoding(embed_dim)
        self.temporal_pe = nn.Parameter(
            torch.zeros(1, 1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        # Transformer encoder stack (shared shapes with the
        # position model so we can swap weights if useful).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Per-anchor head: from each anchor's pooled token
        # (embed_dim) -> a single scalar Δd. We use a small
        # 2-layer MLP per anchor, applied independently.
        # (Equivalent to a (1x1) conv over the anchor axis.)
        self.per_anchor_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        cir_data: torch.Tensor,
        anchor_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            cir_data:      (B, N_anchors, CIR_length) float32
            anchor_coords: (B, N_anchors, 3)            float32

        Returns:
            delta_d:       (B, N_anchors) float32 — per-anchor
                           distance correction in meters.
                           `corrected_dist = measured_dist + delta_d`.
        """
        B, N, L = cir_data.shape
        if N != anchor_coords.shape[1]:
            raise ValueError(
                f"Number of anchors in CIR ({N}) doesn't match "
                f"anchor coords ({anchor_coords.shape[1]})"
            )
        if L != self.cir_length:
            raise ValueError(
                f"CIR length {L} doesn't match model.cir_length "
                f"{self.cir_length}"
            )

        # A. Patch tokens: (B, N, num_patches, embed_dim)
        tokens = self.cir_embedder(cir_data)

        # B. Spatial + temporal PE
        spatial_emb = self.spatial_pe(anchor_coords)   # (B, N, 1, embed_dim)
        tokens = tokens + spatial_emb + self.temporal_pe

        # C. Flatten (N, num_patches) into a single sequence so
        #    self-attention can mix information across both
        #    anchors (inter-anchor multipath correlation) AND
        #    patches (intra-anchor multipath).
        #    (B, N, num_patches, embed_dim) -> (B, N*num_patches, embed_dim)
        sequence = tokens.view(B, N * self.num_patches, -1)

        # D. Transformer encoder
        encoded = self.transformer(sequence)           # (B, N*num_patches, embed_dim)

        # E. Reshape back to per-anchor token format, then
        #    pool across patches (mean over the num_patches
        #    axis) to get one (embed_dim) feature vector per
        #    anchor. This is per-anchor pooling, NOT global
        #    pooling — we keep the anchor identity.
        encoded_per_anchor = encoded.view(B, N, self.num_patches, -1)
        per_anchor_features = encoded_per_anchor.mean(dim=2)  # (B, N, embed_dim)

        # F. Per-anchor head: (B, N, embed_dim) -> (B, N, 1)
        #    -> squeeze to (B, N).
        delta_d = self.per_anchor_head(per_anchor_features).squeeze(-1)
        return delta_d

    @classmethod
    def for_office_rack_cir(
        cls,
        cir_length: int = 60,
        patch_size: int = 6,
        **kwargs,
    ) -> "UWBTransformerDistanceCorrection":
        """Build a model sized for the office_rack sionna track CIR.

        Defaults: cir_length=60, patch_size=6 → 10 patches per
        anchor. See `UWBTransformerCorrection.for_office_rack_cir`
        for patch-size alternatives for cir_length=60.
        """
        return cls(
            cir_length=cir_length,
            patch_size=patch_size,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# Self-test (run as `python -m ultraclear.uwb_transformer_correction`)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Smoke test: build a model, run a forward pass, check shapes
    print("=" * 70)
    print("UWBTransformerCorrection self-test")
    print("=" * 70)

    import time

    def smoke(cir_length: int, patch_size: int, n_anchors: int = 23):
        print(f"\n[test] cir_length={cir_length}  patch_size={patch_size}  "
              f"n_anchors={n_anchors}")
        model = UWBTransformerCorrection.for_office_rack_cir(
            cir_length=cir_length, patch_size=patch_size,
        )
        # eval mode so dropout is off — otherwise predict() and
        # forward() produce different outputs (which is correct
        # behavior, but breaks the assert below).
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  total params: {n_params:,}  (num_patches={model.num_patches})")

        B = 4
        cir = torch.randn(B, n_anchors, cir_length)
        coords = torch.randn(B, n_anchors, 3)
        lse = torch.randn(B, 3)   # (x, y, z) initial LSE estimate
        t0 = time.time()
        delta = model(cir, coords, lse)
        dt = time.time() - t0
        print(f"  forward: cir {tuple(cir.shape)} + coords {tuple(coords.shape)} + "
              f"lse {tuple(lse.shape)}  -> delta_pos {tuple(delta.shape)}  "
              f"({dt*1000:.1f} ms)")

        # Also exercise the predict() wrapper (returns corrected pos)
        corrected = model.predict(cir, coords, lse)
        assert torch.allclose(corrected, lse + delta)
        print(f"  predict(): corrected_pos = lse_pred + delta  "
              f"(shape {tuple(corrected.shape)})  ✓")

    # Original spec: cir_length=152, patch_size=8
    smoke(152, 8, n_anchors=23)
    # Office-rack default
    smoke(60, 6, n_anchors=23)
    # Also test 4-anchor (track2 style)
    smoke(60, 6, n_anchors=4)
    # And 152 with 4 anchors (closest to original)
    smoke(152, 8, n_anchors=4)

    # ---- Gradient check ----
    print("\n[gradient check]")
    model = UWBTransformerCorrection.for_office_rack_cir(cir_length=60, patch_size=6)
    B, N, L = 4, 23, 60
    cir = torch.randn(B, N, L, requires_grad=True)
    coords = torch.randn(B, N, 3, requires_grad=True)
    lse = torch.randn(B, 3, requires_grad=True)
    target_delta = torch.randn(B, 3)
    out = model(cir, coords, lse)
    loss = ((out - target_delta) ** 2).mean()
    loss.backward()
    n_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  loss: {loss.item():.4f}")
    print(f"  params with non-zero grad: {n_grads}/{sum(1 for _ in model.parameters())}")
    print(f"  cir.grad norm: {cir.grad.norm():.4f}  "
          f"coords.grad norm: {coords.grad.norm():.4f}  "
          f"lse.grad norm: {lse.grad.norm():.4f}")

    # ---- Distance-correction model smoke test (2026-07-23 16:35 EDT) ----
    print("\n" + "=" * 70)
    print("UWBTransformerDistanceCorrection self-test")
    print("=" * 70)

    def smoke_dist(cir_length: int, patch_size: int, n_anchors: int = 23):
        print(f"\n[test] cir_length={cir_length}  patch_size={patch_size}  "
              f"n_anchors={n_anchors}")
        m = UWBTransformerDistanceCorrection.for_office_rack_cir(
            cir_length=cir_length, patch_size=patch_size,
        )
        m.eval()
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  total params: {n_params:,}  (num_patches={m.num_patches})")

        B = 4
        cir = torch.randn(B, n_anchors, cir_length)
        coords = torch.randn(B, n_anchors, 3)
        t0 = time.time()
        delta_d = m(cir, coords)
        dt = time.time() - t0
        print(f"  forward: cir {tuple(cir.shape)} + coords {tuple(coords.shape)}  "
              f"-> delta_d {tuple(delta_d.shape)}  ({dt*1000:.1f} ms)")
        assert delta_d.shape == (B, n_anchors), f"shape mismatch: {delta_d.shape}"

    smoke_dist(152, 8, n_anchors=23)
    smoke_dist(60, 6, n_anchors=23)
    smoke_dist(60, 6, n_anchors=4)
    smoke_dist(152, 8, n_anchors=4)

    # ---- Distance-corrector gradient check ----
    print("\n[gradient check]")
    m = UWBTransformerDistanceCorrection.for_office_rack_cir(
        cir_length=60, patch_size=6,
    )
    B, N, L = 4, 23, 60
    cir = torch.randn(B, N, L, requires_grad=True)
    coords = torch.randn(B, N, 3, requires_grad=True)
    target_delta_d = torch.randn(B, N)   # (B, N) per-anchor target
    out = m(cir, coords)
    loss = ((out - target_delta_d) ** 2).mean()
    loss.backward()
    n_grads = sum(1 for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  loss: {loss.item():.4f}")
    print(f"  params with non-zero grad: {n_grads}/{sum(1 for _ in m.parameters())}")
    print(f"  cir.grad norm: {cir.grad.norm():.4f}  "
          f"coords.grad norm: {coords.grad.norm():.4f}")
