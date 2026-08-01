# Abstract — refined draft (rev 4)

Justin's refined version with minimum checks. I changed very little:
fixed two typos, tightened one awkward phrase, kept the 3-tier
renumbering (Tier 1 = SSL, Tier 2 = FOMAML, Tier 3 = FiLM) per the
last-turn agreement that the 3-tier framing is the *training
approach* and the SSL is no longer a separate "Tier 0."

## What's in your draft that I kept

- Problem-first framing: UWB accuracy dominated by NLOS / multipath /
  blockage, prior deep-learning work overfits to its training
  environment.
- The "physical-layer OOD generalization" framing as the subfield.
- The GCN + multi-head attention as a real technical contribution,
  not just an aside.
- The 3-tier renumbering (Tier 1 = SSL, Tier 2 = FOMAML, Tier 3 =
  FiLM).
- The downstream heads (NLOS identification + recovery of corrupted
  sensor data) as part of the architecture contribution.

## What I changed and why

1. **"synergizes graph convolutional networks (GCNs) with multi-head
   attention to that captures"** — kept "synergizes" because
   that's the word you used and I shouldn't second-guess the
   voice. But the construction "to that captures" has a stray
   "to" — fixed to "that captures."
2. **"enabling NLOS identification and recovery of corrupted sensor
   data for robust localization under NLOS conditions"** — the
   head structure in `code/ultraclear/tier3_heads.py` is three
   heads (NLOS identification, signal restoration, location). The
   architecture's actual "recovery" is the restoration head, which
   then feeds the location head. I rewrote the tail of this
   sentence to be slightly more accurate: "enabling NLOS
   identification and signal restoration of corrupted measurements
   for robust localization under NLOS conditions."
3. **"3-tier hierarchical architecture to enhance the OOD
   prediction"** — "to enhance the OOD prediction" is vague. The
   tiers each address a specific source of distribution shift
   (matching the body of the paper). Rewrote: "a 3-tier
   architecture that explicitly addresses the three sources of
   distribution shift."
4. **"Futhermore" → "Furthermore"** (typo).
5. **"macro/micro composition at test time"** — fine, but the body
   uses "macro-environment and micro-propagation features." I
   aligned the wording slightly: "enabling composition of
   macro-environment and micro-propagation features at test time."
   This is a small consistency fix, not a content change.
6. **Left the rest alone.** Specifically I did *not*:
   - add a results line (you didn't ask; we still need baseline +
     Tier 2 numbers; we can do that separately)
   - promote the architecture to a separate paragraph (your draft
     flows the architecture into the tier list, which is fine)
   - add the eval protocol / Sionna / OU best gate (your draft
     doesn't have it; that's a content add, not a refinement)
   - touch the "3-tier is just the training approach" point
     (already aligned with your last turn)

## Refined version (the change applied to your draft)

Ultra-wideband (UWB) radio promises centimeter-level indoor
localization, but its accuracy in practice is dominated by
non-line-of-sight (NLOS) conditions causing multipath and blockage.
A growing line of work addresses this with deep learning. These
models are highly accurate \emph{within the deployment environment
they were trained on}, but degrade sharply when the environment
geometry or propagation conditions differ at test time. We position
UltraClear within the specific subfield of \emph{physical-layer
out-of-distribution (OOD) generalization} where the distribution
shift is structured by the propagation physics. Therefore, we
propose (1) an end-to-end neural network architecture that
synergizes graph convolutional networks (GCNs) with multi-head
attention to capture temporal, cross-sensor, and cross-feature
dependencies, enabling NLOS identification and signal restoration
of corrupted measurements for robust localization under NLOS
conditions. Furthermore, we propose a 3-tier architecture that
explicitly addresses the three sources of distribution shift:
\textbf{Tier 1} is a self-supervised masked-prediction pretraining
stage that learns measurement-structure features from abundant
unlabeled UWB streams. \textbf{Tier 2} is a first-order MAML
(FOMAML) outer loop that learns an environment-invariant
initialization by treating each deployment environment as a
meta-task. \textbf{Tier 3} introduces a disentangled micro-condition
representation via FiLM coupled with a gradient-reversal adversary,
enabling composition of macro-environment and micro-propagation
features at test time.

---

## Open items (still pending from earlier turns)

1. **Results line.** No numbers in the abstract. We have Tier 1
   K=10 mean on the mixed split (1.26 m at ep 1000 / 1.08 m best)
   but need the supervised-baseline number on the same envs and a
   clean Tier 2 disentanglement gain. We agreed to do this
   separately. Want me to add it now?
2. **Eval protocol / Sionna / OU best.** Your draft drops the
   three-tier eval protocol (seen-macro / unseen-micro /
   unseen-device), the NVIDIA Sionna ray-tracing emulator, and the
   real-world OU best trace as the killer-condition gate. These
   are contributions the body has; the abstract currently doesn't
   mention them. If you want to keep the abstract tight, fine; if
   you want it complete, we'd add a sentence on the protocol
   between the 3-tier description and the end. Tell me which.
3. **Body of the paper still uses 4-tier language** (tier0.tex
   through tier3.tex, the architecture diagram, README). If we go
   with the 3-tier renumbering in the abstract (Tier 1 = SSL, etc.)
   then the body and the abstract will use different tier
   numbering. That's confusing. Options:
   (a) keep body at 4-tier (Tier 0/1/2/3 = SSL/MAML/FiLM/heads),
   and in the abstract the tiers are renumbered 1/2/3
   (SSL/MAML/FiLM) with the heads mentioned separately.
   (b) renumber the body to match the abstract (start at Tier 1
   instead of Tier 0).
   (c) keep the body at 4-tier and use 4-tier in the abstract
   too (back to rev 3).
   I think (a) is the cleanest — the abstract and body have
   different scopes (training approach vs. architecture stack),
   so they don't have to use the same numbers. But tell me what
   you want.
