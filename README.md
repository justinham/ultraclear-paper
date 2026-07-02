# UltraClear Paper

LaTeX source for the UltraClear paper:
> **UltraClear: A 4-Tier Architecture for Physical-Layer OOD Generalization in UWB Localization**

## Repository layout

```
ultraclear-paper/
├── main.tex              # main file (preamble, abstract, intro, related work, eval, refs)
├── sections/             # one file per tier
│   ├── tier0.tex         # Tier 0: SSL pretraining (masked prediction)
│   ├── tier1.tex         # Tier 1: FOMAML over environments
│   ├── tier2.tex         # Tier 2: Micro-condition disentanglement (FiLM + grad-reversal adv)
│   └── tier3.tex         # Tier 3: Downstream heads (NLOS, restoration, location)
├── figures/
│   └── architecture.tex  # TikZ architecture diagram (4-tier overview)
├── references.bib        # BibTeX references
└── README.md             # this file
```

## How to build locally

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## How to sync with Overleaf

1. Open your Overleaf project.
2. Go to **Menu** (top-left) → **GitHub** → **Link to GitHub**.
3. Select the `justinham/ultraclear-paper` repository and the `main` branch.
4. Overleaf will pull the current state of the repo. After that, every
   `git push` to GitHub appears in Overleaf automatically, and changes
   made in Overleaf can be pushed back to GitHub via the same menu.

## Status

- [x] Title, abstract, introduction
- [x] Related work (4 paragraphs: DA/DG, UWB NLOS, SSL-on-UWB, conditional modulation)
- [x] Problem formulation
- [x] ASTA encoder subsection
- [x] Tier 0: SSL pretraining
- [x] Tier 1: FOMAML
- [x] Tier 2: Micro-condition disentanglement
- [x] Tier 3: Downstream heads
- [x] Three-tier evaluation protocol
- [x] Discussion and conclusion
- [ ] Experiments section: placeholder (camera-ready)
- [ ] Numerical results: pending full-pipeline runs

## Companion code

The architecture is implemented in
[`justinham/ultraclear_v2`](https://github.com/justinham/ultraclear_v2) (or
local path `~/.openclaw/workspace/ultraclear_v2/`).
