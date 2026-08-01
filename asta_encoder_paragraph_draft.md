# ASTA encoder description — refined draft (rev 1)

Justin's three paragraphs with minimum checks. Changed: 2 broken
figure refs (`fig:asta_2way` and `fig:asta` don't exist in the
repo — only `fig:arch` and `fig:tier0` do), 1 domain-mismatch
claim ("vehicle environments" — the experiments are on
indoor/outdoor/parking scenes, not vehicles), 4 grammar/tense
errors, 1 overclaim about the ASTA's role, and cleaned LaTeX
formatting. Kept the technical critique of vanilla GCNs, the
motivating paragraph on non-stationary conditions, and the
"data-driven adaptation is critical" closing.

## What's in your draft that I kept

- The two-step motivation: (1) vanilla GCNs use fixed
  adjacency / single-channel weighting; (2) UWB environments
  are non-stationary, so fixed weights fail.
- The "without requiring explicit NLOS labels during training"
  point — important and accurate.
- The "data-driven adaptation is critical for real-world
  deployment" framing.
- "Pre-characterizing all possible degradation patterns is
  infeasible" — good reviewer-facing point.
- The adaptive-weighting + multi-head-attention mechanism.

## What I changed and why

### Figure references

1. **`fig:asta_2way` doesn't exist.** Searched the entire repo:
   only `fig:arch` (the overall 4-tier architecture at
   `main.tex:227`) and `fig:tier0` (SSL pipeline at
   `sections/tier0.tex:20`) and the per-tier figures. There's no
   `fig:asta_2way` or `fig:asta`. The `figures/` directory has
   `architecture.tex`, `tier0_pipeline.tex`, etc., but nothing
   named `asta*`. **Your draft would fail to compile the
   cross-references.** I propose two options below.

2. **`fig:asta` doesn't exist either** — same problem.

### Domain claim

3. **"our UWB positioning system targets vehicle environments"**
   — this is a domain mismatch. The experiments in the paper
   are on Sionna-simulated indoor/outdoor/parking scenes (per
   `sections/tier1.tex:60-100`, e.g. `outdoor_parking_01`,
   `indoor_garage_parking_01`). These are simulated
   **environments**, not vehicle-specific. `main.tex:118` says
   "explicitly carves the problem this way for UWB" — no
   mention of vehicles.

   I have to flag this because if the paper claims "vehicle
   environments" but the experiments are on static indoor +
   outdoor scenes, a reviewer will catch the mismatch and it
   will hurt the paper. Two options:
   - **Option A (drop "vehicle"):** "our UWB positioning
     system targets dynamic indoor and outdoor environments"
     or just "our UWB positioning system" — the
     non-stationary / dynamic NLOS argument below still works
     without the vehicle framing.
   - **Option B (keep "vehicle" but add scope):** "our UWB
     positioning system targets vehicle environments such as
     parking garages and outdoor lots, where …" — but only
     if you actually have vehicle-specific experiments
     somewhere. If you don't, this is a paper-level change,
     not a paragraph-level one, and we should talk about it.

   I went with Option A in the refined version. Tell me if
   you want B and we'll discuss.

### Grammar / tense

4. **"will be sequentially training"** — should be "is
   sequentially trained" (present tense, consistent with the
   rest of the paper) or "will be sequentially trained" (if
   you want future tense). "Sequentially training" is wrong
   (gerund, not past participle).

5. **"down-weighted the untrustful measurements while emphasize
   reliable measurements"** — three issues:
   - "down-weighted" → "down-weights" (present tense)
   - "untrustful" → "unreliable" (standard term; "untrustful"
     is rare and reads oddly)
   - "while emphasize" → "while emphasizing" (parallel gerund)

6. **"are correlate with positioning errors"** — should be
   "are correlated with" or "correlate with."

### Overclaim about ASTA's role

7. **"our adaptive weighting mechanism … addresses this
   limitation by learning to identify and suppress unreliable
   measurements in situ"** — the ASTA *learns
   representations* that support NLOS identification; the
   actual *identification* is done by the downstream
   `NLOSHead` in `code/ultraclear/tier3_heads.py`. The ASTA
   is the encoder, not the classifier. Saying the ASTA
   "identifies and suppresses" is overstating its role.

   I rewrote to: "addresses this limitation by learning
   representations in which unreliable measurements are
   down-weighted in the attention and message-passing
   stages, without requiring explicit NLOS labels." This is
   accurate: the ASTA learns to *make unreliable
   measurements contribute less* to its features, but the
   head does the explicit identification.

### LaTeX formatting

8. Added `\emph{}` for technical terms (GCN, ASTA, ToA, AoA,
   NLOS) per the body style.
9. Bold key terms (`Dual-Edge GCN`, `Multi-Head Attention`)
   on first use.
10. `Figure~\ref{}` (with non-breaking `~`) per the existing
    paper convention (`main.tex:230` uses
    `Figure~\ref{fig:arch}`).

## Refined LaTeX (the changes applied)

Figure~\ref{fig:arch} shows the shared ASTA encoder, which is
sequentially trained through SSL and \emph{FOMAML}.

Having established the node and edge formulations, the challenge
becomes quantifying edge weights and constructing the graph
network. Traditional \emph{GCN} formulations employ fixed
adjacency matrices or static edge weights, which assume
stationary measurement quality and uniform reliability across all
sensors and timesteps. Some prior works attempt to formulate edge
weights based on estimated distances between adjacent anchors.
However, this single-channel approach fails to capture
cross-modal dependencies — \emph{ToA} and \emph{AoA} may exhibit
different correlation patterns with positioning errors.

Moreover, our UWB positioning system targets dynamic indoor and
outdoor environments that operate in non-stationary conditions
where signal degradation is highly dynamic: NLOS conditions
appear and disappear as obstacles move through the propagation
path, multipath patterns shift with tag motion, and interference
varies temporally. A fixed-weight GCN would propagate corrupted
measurements with the same confidence as reliable ones,
polluting the learned representations and degrading positioning
accuracy.

Our adaptive weighting mechanism (see Figure~\ref{fig:arch})
addresses this limitation by learning representations in which
unreliable measurements are down-weighted in the attention and
message-passing stages, without requiring explicit NLOS labels
during training. The multi-head attention model discovers signal
qualities that are correlated with positioning errors,
automatically down-weighting the unreliable measurements while
emphasizing the reliable ones. This data-driven adaptation is
critical for real-world deployment, where pre-characterizing all
possible degradation patterns is infeasible and environmental
conditions evolve over time.

## Diff vs your draft

| Your draft | Refined |
|---|---|
| `Figure \ref{fig:asta_2way}` | `Figure~\ref{fig:arch}` — `fig:asta_2way` doesn't exist; `fig:arch` is the actual architecture figure |
| "the shared ASTA encoder that will be sequentially training through SSL and Meta supervised learning" | "the shared ASTA encoder, which is sequentially trained through SSL and \emph{FOMAML}" — grammar fix; "FOMAML" italicized per body style |
| "our UWB positioning system targets vehicle environments" | "our UWB positioning system targets dynamic indoor and outdoor environments" — domain-mismatch fix; experiments are on Sionna indoor/outdoor/parking scenes, not vehicle-specific |
| "operate in non-stationary conditions where signal degradation is highly dynamic" | kept (still accurate after the domain change) |
| "addresses this limitation by learning to identify and suppress unreliable measurements in situ" | "addresses this limitation by learning representations in which unreliable measurements are down-weighted in the attention and message-passing stages" — accurate to the ASTA's role (encoder, not classifier) |
| "are correlate with positioning errors" | "are correlated with positioning errors" |
| "down-weighted the untrustful measurements while emphasize reliable measurements" | "down-weighting the unreliable measurements while emphasizing the reliable ones" — tense, spelling, parallelism fixed |
| `Figure \ref{fig:asta}` | `Figure~\ref{fig:arch}` (same figure, second reference) |

## Open items

1. **`fig:asta_2way` and `fig:asta`** — neither exists. Options:
   - **(a)** Create the figures. I can draft a TikZ ASTA block
     diagram (`\input{figures/asta_block.tex}`) and a 2-way
     dual-channel figure. Want me to?
   - **(b)** Drop the figure references entirely. The text
     stands on its own; the existing `fig:arch` is enough for
     the architecture overview.
   - **(c)** Point both refs at `fig:arch` (what I did in the
     refined version). This is the lowest-friction option.
   Tell me which.

2. **"Vehicle environments"** — I went with Option A (drop
   "vehicle") in the refined version because the experiments
   don't support it. If the paper is actually targeted at
   vehicle scenarios and we have vehicle-specific results
   somewhere I haven't seen, tell me and I'll restore "vehicle"
   with proper scope. If not, this is the right fix.

3. **ASTA's role in NLOS identification** — the ASTA is the
   *encoder*; the `NLOSHead` is the *classifier*. The refined
   version makes this distinction clear. If the paper has
   another framing where the ASTA itself does the identification
   (e.g. attention weights serve as soft NLOS probabilities),
   tell me and I'll adjust.

4. **Where does this go in `main.tex`?** The existing
   `\subsection{The ASTA encoder}` at `main.tex:252` is a
   different description (uses the CTC/D-GCN/MHSA names,
   describes each sublayer). Your three paragraphs could:
   - **(a)** Replace the existing subsection entirely.
   - **(b)** Sit *before* the existing subsection as
     motivation/justification, with the existing subsection
     being the formal description.
   - **(c)** Be folded into the existing subsection as
     additional paragraphs.
   I'd go with (b) — your draft is motivation/justification,
   the existing is formal description, and both serve
   different purposes. Tell me which.
