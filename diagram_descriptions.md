# Figure descriptions for the two architecture diagrams

Two diagrams:
- **Macro** (`macro---e5834d51-...png`): the Tier 1 FOMAML pipeline
  (per-environment inner loops + meta-update).
- **Micro** (`micro---91bbc1b5-...png`): the Tier 2 FiLM +
  gradient-reversal disentanglement pipeline.

For each, I provide: (1) a figure caption (what goes under the
figure in the LaTeX `\caption{}`), and (2) supporting prose (what
goes in the body of the paper to walk the reader through the
diagram).

Both diagrams are also referenced in the LaTeX as new labels:
- Macro: `\label{fig:macro}` (currently no such label)
- Micro: `\label{fig:micro}` (currently no such label)

I describe the new labels and where the figure files go.

---

## Figure file layout

Both diagrams are PNGs. To use them in the LaTeX, we need either
(a) `\includegraphics` with the PNGs, or (b) re-create them as
TikZ for vector quality and editability.

**Option (a) — quick path:** drop the PNGs into
`ultraclear-paper/figures/` and use:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.9\linewidth]{figures/macro.png}
  \caption{...}
  \label{fig:macro}
\end{figure}
```

**Option (b) — paper-quality path:** recreate as TikZ so the
labels match the LaTeX style and the figure scales cleanly. I'd
estimate 30-60 min per figure. Tell me if you want this; I can
draft the TikZ.

I went with Option (a) in the captions below since the PNGs
already exist.

---

## Macro diagram — caption + supporting prose

### Caption

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.9\linewidth]{figures/macro.png}
  \caption{Tier~1 FOMAML training over deployment environments.
  Each environment's input is processed by a per-task branch
  (Local ASTA + local decoder) initialized from the shared
  meta-parameters. The branch is adapted on a support set (Local
  BP, the inner loop) and evaluated on a query set. The query
  losses across all environments are aggregated into a meta-loss
  that updates the shared meta-parameters via Global BP (the
  outer loop), so the meta-learner produces an initialization
  from which few-shot adaptation generalizes across
  environments.}
  \label{fig:macro}
\end{figure}
```

### Supporting prose (for the body of the paper)

Figure~\ref{fig:macro} illustrates the Tier~1 \emph{FOMAML}
training procedure over deployment environments. For each
environment $e \in \{1, \dots, N\}$ in the outer loop, the
shared meta-parameters $\bm{\theta}$ produce a per-task
initialization, which is adapted on a support set of $K$ labeled
samples via a small number of inner-loop gradient steps (the
\emph{Local BP} branch). The adapted per-task model is then
evaluated on a held-out query set, and the query loss is
aggregated across environments into a single meta-loss

\[
\bm{\theta} \leftarrow \bm{\theta} - \beta \sum_{e=1}^{N}
\nabla_{\bm{\theta}} \mathcal{L}_{\mathrm{task}}^{(e)}
(\mathrm{supp}^{(e)}, \mathrm{query}^{(e)}),
\]

where $\beta$ is the meta-step size and
$\mathcal{L}_{\mathrm{task}}^{(e)}$ is the per-environment
task loss evaluated on the query set after support-set
adaptation. The meta-update (\emph{Global BP}) adjusts the
shared initialization so that \emph{after} inner-loop
adaptation, the model performs well on each environment's
query set. At test time, the same procedure is applied to a
new (unseen) environment using only $K$ labeled support shots
— no online fine-tuning is required beyond the inner-loop
adaptation.

### Notes on the equation

**Important — please verify:** the equation you wrote on the
diagram is `θ ← θ - β Σ ∇_task(sup(sup))`. I'm reading the
second `sup` as `query` (the standard MAML/FOMAML meta-loss is
computed on the **query** set, not the support set — the
support set is for the inner update, the query set is for the
outer update). If the diagram literally has `sup(sup)`, it
might be a typo, or the second `sup` could be a different
symbol I'm misreading. I wrote the equation with `query` in the
LaTeX above; tell me if your original notation was different and
I'll match it.

The `\mathcal{L}_{\mathrm{task}}^{(e)}(\mathrm{supp},
\mathrm{query})` notation makes the role of each set explicit
even though only `query` appears in the gradient.

---

## Micro diagram — caption + supporting prose

### Caption

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.9\linewidth]{figures/micro.png}
  \caption{Tier~2 FiLM-based micro-condition disentanglement.
  The pretrained ASTA encoder is frozen (dashed) and acts as a
  fixed feature extractor producing per-cell features
  $\bm{H} \in \R^{B \times T \times M \times d}$. A small
  micro encoder pools $\bm{H}$ into a compact condition
  vector $\bm{c} \in \R^{B \times d'}$, which is mapped by the
  $\gamma$ and $\beta$ networks into per-feature multiplicative
  and additive scales. The FiLM-conditioned features
  $\bm{H}' = \bm{\gamma} \odot \bm{H} + \bm{\beta}$ are
  consumed by a multi-head decoder for the supervised task
  (e.g.\ NLOS identification, signal restoration, and location
  estimation) and, through a gradient-reversal layer, by an
  adversary that attempts to recover the macro environment
  identity. The reversal forces $\bm{H}'$ to be informative
  for the task while being uninformative about the macro
  environment, yielding a disentangled micro representation.}
  \label{fig:micro}
\end{figure}
```

### Supporting prose (for the body of the paper)

Figure~\ref{fig:micro} shows the Tier~2 \emph{FiLM}-based
disentanglement pipeline. The pretrained ASTA encoder from
Tier~0 is \textbf{frozen} and acts as a fixed feature extractor,
producing per-cell features
$\bm{H} \in \R^{B \times T \times M \times d}$ for an input
window. A small \emph{micro encoder} $h_\phi$ pools $\bm{H}$ into
a compact condition vector $\bm{c} \in \R^{B \times d'}$.

Two small networks — $\gamma(\bm{c})$ and $\beta(\bm{c})$ — map
$\bm{c}$ into per-feature multiplicative and additive scales of
shape $(B, 1, 1, d)$, broadcast across the $T$ and $M$ axes. The
FiLM-conditioned features are then

\[
\bm{H}' = \bm{\gamma} \odot \bm{H} + \bm{\beta},
\]

where $\odot$ denotes the element-wise (Hadamard) product. The
$\gamma$ network is initialized to output $\bm{1}$ and the
$\beta$ network to output $\bm{0}$, so that FiLM is the identity
at the start of training.

The conditioned features $\bm{H}'$ are consumed by two heads.
The \emph{multi-head decoder} regresses the supervised task
targets (NLOS identification, signal restoration, and location
estimation). Through a \emph{gradient-reversal} layer,
$\bm{H}'$ is also fed to an adversary that attempts to recover
the macro environment identity. The reversal multiplies the
upstream gradient by a negative constant during
back-propagation, so the adversary is trained to predict the
environment while the signal flowing back into the FiLM
parameters \emph{opposes} that prediction. The effect:
$\bm{H}'$ is forced to be informative for the supervised task
but uninformative about the macro environment — i.e.\ the
condition vector $\bm{c}$ carries only micro-propagation
information (NLOS rate, NLOS strength), not macro-environment
information (room geometry, anchor layout).

### Notes on "GT label ×3"

The diagram shows the multi-head decoder producing three
prediction targets. In the codebase, the downstream heads are
NLOS identification, signal restoration, and location
estimation (per `code/ultraclear/tier3_heads.py`). I wrote the
caption to call these out explicitly. If the ×3 in your
diagram means something different (e.g.\ three different
trajectories, or three different NLOS-related targets), tell
me and I'll adjust the caption.

### Note on the "freeze" label

The freeze arrow on the ASTA encoder is doing real work — the
7/15-7/16 work showed that the frozen-encoder trick is what
prevents the FiLM+adversary from collapsing the location head.
If the encoder were trainable, the adversary could push the
encoder itself toward a constant to make the env-ID prediction
impossible, which would kill the location-relevant signal as
a side effect. With the encoder frozen, the adversary is
forced to push only the FiLM and micro-encoder modules,
leaving the location-relevant features intact. The supporting
prose above mentions "frozen" in bold; the body could expand
on this in the discussion if useful.

---

## Where these go in `main.tex`

Both figures naturally fit in the **methods / architecture**
section we're building. The current `main.tex` jumps from the
intro directly to `sections/tier0.tex`. I'd put them as:

1. **Macro diagram (`fig:macro`)** — in the new Tier 1
   discussion. The existing `sections/tier1.tex` describes
   FOMAML in prose; the macro diagram would be a visual
   companion. Place it before the existing figure
   `fig:tier1-fewshot` (the K-shot comparison figure) or as
   the first figure of `sections/tier1.tex`.

2. **Micro diagram (`fig:micro`)** — in the new Tier 2
   discussion. The existing `sections/tier2.tex` describes
   FiLM in prose; the micro diagram would be a visual
   companion. Place it near the existing `fig:tier2-film`
   figure.

If you want me to slot them in, I need:
- Confirm the figure filenames (PNG names: `macro.png` /
  `micro.png`? or do you want to keep the inbound UUID names
  and rename later?)
- Confirm the labels (`fig:macro` / `fig:micro` or different?)
- Confirm the equation notation in the macro diagram
  (`sup(sup)` vs `query`)

---

## Open items

1. **Equation in the macro diagram.** Is the second `sup` in
   `θ ← θ - β Σ ∇_task(sup(sup))` actually `query`? Standard
   MAML uses query-set loss for the meta-update. Tell me
   which is right.
2. **"GT label ×3" in the micro diagram.** I assumed the 3
   targets are NLOS / restoration / location. Confirm or
   correct.
3. **PNG vs TikZ.** I went with PNG since the diagrams
   already exist. For paper-quality vector graphics, I can
   re-create as TikZ. Want me to?
4. **Figure filenames and labels.** I'm proposing `figures/macro.png`,
   `figures/micro.png`, `fig:macro`, `fig:micro`. Tell me if
   you want different names.
5. **The "Meta weight init/update" block in the macro
   diagram.** It dispatches the meta-init to each local
   branch. The code (`tier1_maml.py`) uses
   `torch.func.functional_call` to do this — i.e.\ the meta
   parameters are passed as functional call args to a fresh
   per-task model, not deep-copied. The supporting prose
   says "produces a per-task initialization" which captures
   the intent without going into the functional-call
   implementation detail. Tell me if you want more or less
   detail here.
