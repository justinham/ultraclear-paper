# UltraClear Paper

LaTeX source and reference implementation for the UltraClear paper:
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
├── code/                 # reference implementation (clean-room, self-contained)
│   ├── ultraclear/       # the model package
│   ├── scripts/train.py  # end-to-end training entry point
│   └── README.md         # how to run the code
├── references.bib        # BibTeX references
└── README.md             # this file
```

## How to build the paper

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## How to run the code

```bash
cd code
python3.11 -m scripts.train --quick      # 5-second smoke test
python3.11 -m scripts.train              # full run
```

See [`code/README.md`](code/README.md) for details.

## How to sync with Overleaf

1. Open your Overleaf project.
2. Go to **Menu** (top-left) → **GitHub** → **Link to GitHub**.
3. Select the `justinham/ultraclear-paper` repository and the `main` branch.
4. Overleaf will pull the current state of the repo. After that, every
   `git push` to GitHub appears in Overleaf automatically, and changes
   made in Overleaf can be pushed back to GitHub via the same menu.

## Status

### Paper
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

### Code
- [x] Synthetic UWB-like data with env/cond metadata
- [x] ASTA encoder (CTC + D-GCN + MHSA)
- [x] Tier 0: SSL pretraining (full training loop)
- [x] Tier 1: FOMAML (hand-rolled, no `learn2learn` dep)
- [x] Tier 2: disentangled model wires up (smoke test)
- [x] Tier 3: downstream heads (full training loop)
- [ ] Tier 2: full adversarial training loop
