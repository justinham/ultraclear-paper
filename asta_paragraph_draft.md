# ASTA description — refined draft (rev 1)

Justin's two paragraphs with minimum checks. Changed: 1 missing
module (FFN), 1 wrong module (Feature Concatenation), 1 wrong
location (temporal dependency is in the encoder wrapper, not
inside each block), 2 typos, and 1 unclear phrase about the
training pipeline.

## What's in your draft that I kept

- "Adaptive Spatial-Temporal Multi-Attention (ASTA) Blocks" name
- "Core novelty" framing
- The three dependency names: Dual-Channel Feature, Cross-Sensor
  Spatial, Temporal
- "Adaptive graph convolution and multi-head attention" as the
  mechanism
- The 3-phase training pipeline (SSL → MAML → FiLM)
- "Multiple ASTA blocks can be stacked in sequence to form a deep
  encoder, with each block refining the spatial-temporal
  representations at increasing levels of abstraction."

## What I changed and why

### Paragraph 1 (block design + training)

1. **"jointly models three dependency types (Dual-Channel Feature
   Dependency, Cross-Sensor Spatial Dependency, and Temporal
   Dependency)"** — the source code (`asta.py:85-105`) shows that
   the **ASTA block** does *not* model the temporal dependency
   directly. The block is: `GCN → residual+LN → MHSA → residual+LN
   → FFN`. The temporal dependency is modeled by the **causal TCN
   at the input projection** of `ASTAEncoder` (one-shot, before
   the blocks), not inside each block. The three dependencies
   *do* all exist in the encoder as a whole, but they're not all
   jointly modeled by the block. I rewrote to be accurate: the
   block handles Dual-Channel + Cross-Sensor; the encoder
   wrapper adds the temporal piece. The "jointly models" claim
   was overstated.

2. **"through adaptive graph convolution and multi-head
   attention"** — kept, this is the right description of the
   mechanism.

3. **"sequentially trained though self-supervised learning, Meta
   cross invironment supervised learning, and then added a micro
   encoder to enhance the ability for dynamic random noise through
   FiLM"** — multiple issues:
   - **"though" → "through"** (typo).
   - **"cross invironment" → "cross-environment"** (typo).
   - **"added a micro encoder"** — the micro encoder isn't
     "added" in series with the ASTA. It's stacked *on top* of
     the ASTA encoder at Phase 3, and the ASTA encoder is
     *frozen* during Phase 3 (this is the "frozen-encoder
     trick" that prevents the FiLM+adversary from collapsing
     the location head, per the 7/15-7/16 work). I rewrote
     to: "stacked on top with the ASTA encoder frozen, to
     disentangle dynamic micro-conditions via FiLM." This
     matches what the code actually does.
   - **"dynamic random noise"** — the micro condition is
     about propagation (NLOS rate × strength per
     `code/ultraclear/data.py:_micro_params`), not generic
     "random noise." Rewrote to "dynamic micro-conditions
     (e.g. NLOS rate and strength)."

### Paragraph 2 (block structure)

4. **"Each ASTA block contains the sequential structure including
   a Dynamic Dual-edge GCN, a Feature Concatenation layer, a
   Multi-head Self-Attention layer"** — there is **no Feature
   Concatenation layer** in the ASTA block. The block is:
   `GCN → residual+LN → MHSA → residual+LN → FFN`. The
   concatenation might be a misremembering of either:
   - The residual connection (element-wise add, not concat), or
   - The input projection that brings raw features into
     `d_model` (a 1x1-conv-equivalent linear, not a concat).
   I dropped the "Feature Concatenation layer" and added the
   **FFN** (feed-forward network) and **LayerNorm** residuals
   that are actually in the code. This is a factual fix.

5. **"to generate spatial and temporal features at the end of the
   block"** — the attention generates **spatial** features
   (cross-sensor, via the M×M GCN adjacency) and **long-range
   cross-time** features (the attention is over T×M tokens), but
   the block itself doesn't have a separate "temporal feature"
   output. The temporal stream is mixed at the *encoder* level
   by the causal TCN, then blocks refine spatial+cross-time.
   I rewrote: "to generate spatial and cross-time features at
   the end of the block."

6. **"with each block refining the spatial-temporal representations
   at increasing levels of abstraction"** — kept, this is a
   good description of stacking.

## Refined version (the changes applied)

\textbf{Adaptive Spatial-Temporal Multi-Attention (ASTA) Blocks.}
The core novelty lies in the ASTA block design, which captures
two of the three dependency types at the block level: the
\textbf{Dual-Channel Feature Dependency} (the GCN applies a
per-channel anchor adjacency, so ToA and AoA channels get
different graph weights) and the \textbf{Cross-Sensor Spatial
Dependency} (message passing across anchors via the learned M×M
adjacency). The \textbf{Temporal Dependency} is captured once
at the encoder's input projection by a causal temporal
convolution, before the stack of blocks. Each block then refines
the spatial representations; multiple blocks stack to form a
deep encoder.

The ASTA encoder is sequentially trained through three phases
of the pipeline: \textbf{Phase 1} is self-supervised learning
on abundant unlabeled UWB streams, which trains the GCN and
attention modules to learn the underlying dependency structure
without labels. \textbf{Phase 2} is meta cross-environment
supervised learning, which uses FOMAML over deployment
environments to learn an environment-invariant encoder
initialization. \textbf{Phase 3} adds a micro encoder stacked
on top of the (now frozen) ASTA encoder, which disentangles
dynamic micro-conditions (NLOS rate and strength) via FiLM
conditioning coupled with a gradient-reversal adversary.

Each ASTA block contains the sequential structure: a
\textbf{Dynamic Dual-Edge GCN}, a residual connection with
LayerNorm, a \textbf{Multi-Head Self-Attention} layer over the
flattened (time, anchor) token grid, another residual + LayerNorm,
and a small feed-forward network, to generate spatial and
long-range cross-time features at the end of the block.
Multiple ASTA blocks can be stacked in sequence to form a deep
encoder, with each block refining the spatial-temporal
representations at increasing levels of abstraction.

## Diff vs your draft

| Your draft | Refined |
|---|---|
| "jointly models three dependency types" | "captures two of the three dependency types at the block level … The Temporal Dependency is captured once at the encoder's input projection" — the temporal piece is in the encoder wrapper, not inside each block |
| "Dual-Channel Feature Dependency, Cross-Sensor Spatial Dependency, and Temporal Dependency" | Same names, but each one mapped to its actual mechanism: GCN per-channel adjacency / GCN M×M adjacency / causal TCN at input |
| "sequentially trained though self-supervised learning" | "sequentially trained through three phases of the pipeline" + each phase named explicitly (matches the 3-phase framing we settled on for the abstract) |
| "Meta cross invironment supervised learning" | "meta cross-environment supervised learning" (typo) |
| "added a micro encoder to enhance the ability for dynamic random noise through FiLM" | "adds a micro encoder stacked on top of the (now frozen) ASTA encoder, which disentangles dynamic micro-conditions (NLOS rate and strength) via FiLM conditioning coupled with a gradient-reversal adversary" — accurate to the code and the 7/15-7/16 frozen-encoder finding |
| "a Dynamic Dual-edge GCN, a Feature Concatenation layer, a Multi-head Self-Attention layer" | "a Dynamic Dual-Edge GCN, a residual connection with LayerNorm, a Multi-Head Self-Attention layer … another residual + LayerNorm, and a small feed-forward network" — matches `ASTABlock` in `asta.py:85-105` |
| "to generate spatial and temporal features" | "to generate spatial and long-range cross-time features" — the block generates spatial + cross-time, not a separate temporal feature |

## Open items

1. **"Adaptive" in ASTA.** The A in "Adaptive" — the code has
   `DynamicDualEdgeGCN` (the "D" is Dynamic, learned adjacency),
   not "Adaptive." The three As in ASTA would be: Adaptive,
   Spatial, Temporal, Attention. The "Spatial-Temporal Attention"
   is clear from the architecture. "Adaptive" is the only A
   that's not literally in the source. The graph weights are
   *learned* (adaptive in the sense of input-conditioned would
   be a stretch — they're not input-conditioned, they're
   learned parameters). If you want to keep "Adaptive" in the
   name, that's fine — it's a defensible description of learned
   vs fixed graph weights. But if a reviewer asks "what's
   adaptive about it?" the answer is "the adjacency is learned,
   not fixed" not "the graph is input-conditioned." Tell me
   if you want to keep it.
2. **"micro encoder" vs "h_phi" vs "disentangled micro encoder."**
   The code calls this module `h_phi` in `tier2_micro.py` and
   produces a condition vector `c`. Calling it a "micro encoder"
   is fine, but the body of the paper uses different language
   for it. We should pick one and stick with it. Tell me which.
3. **Block-level vs encoder-level dependency claim.** My rewrite
   splits the dependencies into "block does dual-channel +
   cross-sensor" and "encoder adds temporal." This is
   *accurate* but it might be more pedagogical to say the block
   does all three (treating the TCN as part of the block for
   narrative purposes). The risk: a reviewer reads the code,
   sees TCN is outside the block, and flags the inconsistency.
   I'd prefer the accurate version, but tell me if you want the
   narrative version.
