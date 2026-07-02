# UltraClear — Reference Implementation

Clean-room, self-contained Python implementation of the UltraClear
4-tier architecture described in
[`../main.tex`](../main.tex). Runs end-to-end on CPU in ~1 minute
on synthetic UWB-like data; no external services or datasets
required.

## Layout

```
code/
├── ultraclear/         # the model package
│   ├── config.py       # shared hyperparameters (Config dataclass)
│   ├── data.py         # synthetic UWB-like data with env/cond labels
│   ├── asta.py         # ASTA encoder (CTC + D-GCN + MHSA)
│   ├── tier0_ssl.py    # Tier 0: masked-prediction pretraining
│   ├── tier1_maml.py   # Tier 1: FOMAML over environments
│   ├── tier2_micro.py  # Tier 2: FiLM + gradient-reversal adversary
│   └── tier3_heads.py  # Tier 3: NLOS / restoration / location heads
├── scripts/
│   └── train.py        # end-to-end: 4 tiers + 3-tier eval
└── data/               # placeholder for real (Sionna / OU best) data
```

## How to run

From this directory:

```bash
# Quick smoke test (~5 seconds)
python3.11 -m scripts.train --quick

# Full run (~1 minute on CPU)
python3.11 -m scripts.train
```

You should see output like:

```
TIER 0 — ssl_loss drops ~6.3 → 0.7 in 30 epochs
TIER 1 — maml outer-loop runs end-to-end
TIER 2 — h_prime, c, adv_logits shapes confirmed
TIER 3 — train_loc_mse drops ~3.9 → 0.2 in 50 epochs
EVAL — three-tier localization error printed
```

## Dependencies

```
torch >= 2.0
numpy
```

No `learn2learn`, `higher`, or `torch_geometric` required. The
MAML inner loop is hand-rolled using `torch.func.functional_call`
to keep the dependency surface small.

## Plugging in real data

The synthetic data module in `ultraclear/data.py` exposes the
shape contract every tier assumes:
`(X, y, env_id, cond_id)` where

- `X.shape = (T, M, F)` — sliding window of measurements
- `y.shape = (2,)` — 2D tag position at the middle timestep
- `env_id`, `cond_id` — integer labels for the macro environment
  and micro propagation condition

To use Sionna-generated or real UWB data, replace
`build_synthetic_dataset` in `data.py` with a loader that
returns the same 4-tuple shape. The rest of the pipeline is
unchanged.

## What is and isn't trained end-to-end yet

| Tier | Module | Training loop |
|------|--------|---------------|
| 0 | masked-prediction SSL | ✅ full |
| 1 | FOMAML over environments | ✅ full (FOMAML via functional_call) |
| 2 | FiLM + gradient-reversal adv | ⚠️ smoke-test only — full adversarial training loop is the next iteration |
| 3 | NLOS / restoration / location heads | ✅ full (location loss only; NLOS head is exercised but has no labels in the synthetic data) |

## Known limitations

- **FOMAML stability** — the first-order approximation is known
  to oscillate. For higher stability, consider switching to
  Reptile (a 5-line change in `tier1_maml.py:maml_train`).
- **NLOS head training** — the synthetic data has no
  per-anchor NLOS labels, so the NLOS head is trained only via
  gradient flow from the location head. Real data with
  per-anchor NLOS labels will exercise the full
  multi-task loss (Eq. 13 of the paper).
- **Tier 2 adversarial training** — the model wires up but the
  outer adversarial training loop is the next item on the
  implementation roadmap.
