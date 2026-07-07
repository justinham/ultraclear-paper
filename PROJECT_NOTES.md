# Project Notes — UltraClear paper (Tier 0 / Tier 1 work, 2026-07-03)

This is a working log of the Tier 0 SSL and Tier 1 FOMAML experiments
done on the Sionna ray-tracing data on 2026-07-03. It is the
single best starting point for resuming this work in a future
session; the rest of the project lives under `code/`, `figures/`,
`code/checkpoints/`, and `code/scripts/`.

---

## 1. What was built (in the order it was built)

### Tier 0: SSL pretraining on Sionna data
- `code/scripts/train_tier0_sionna.py` — F=1 trainer, sliding
  windows over the trial axis. CLI flags: `--epochs`, `--mask_ratio`,
  `--n_channels`, `--csv`, `--out`.
- `code/scripts/train_tier0_sionna_f2.py` — F=2 (ToA + AoA) trainer
  with per-channel z-score normalization (stats computed from
  *training* scenes only, saved in the checkpoint). Additional flag:
  `--channel_weights 3.0 1.0` to upweight ToA over AoA in the loss.
- Joint masking across F channels: the existing `random_mask` in
  `ultraclear/tier0_ssl.py` already does `(B, T, M)` mask broadcast
  over `F`, so if (t, m) is masked, both ToA and AoA are zeroed.
  No change needed to that function.

### Tier 0 visualizations
- `code/scripts/viz_tier0_heatmaps.py` (and `_f2.py`) — 4-column
  heatmap stack (masked input / GT / model output / |recon−truth|
  error). v2 design: viridis for value rows, RdBu_r diverging for
  error, magenta mask outlines, per-window RMSE in titles,
  percentile-bounded colormap.
- `code/scripts/viz_tier0_traces.py` (and `_f2.py`) — per-anchor
  time-series traces with color-coded error markers.
- `code/scripts/viz_tier0_metrics.py` (and `_f2.py`) — quantitative
  metrics: RMSE, MAE, bias, R², shrink slope, per-anchor breakdown.
- `code/scripts/plot_tier0_sionna.py` — basic loss curve + pred-vs-true
  scatter.
- `code/scripts/plot_loss_curve.py` — generic single-ckpt loss plot
  (reusable for any ckpt with `train_losses` / `val_losses`).
- `code/scripts/compare_mask_ratios.py` — 25% vs 15% mask comparison.
- `code/scripts/compare_f2_runs.py` — F=2 50/50 vs 3:1 channel-weight
  comparison.

### Tier 1: FOMAML on multi-scene Sionna
- `code/scripts/multi_scene_dataset.py` — `MultiSceneSionnaDataset`
  + `build_default_multi_scene(csv_paths)` loads N scene CSVs and
  groups by (grid_id, trial_id). Exposes (X, y, env_id, cond_id)
  per window. cond_id is coarse 0/1 (all LOS / any NLOS).
- `code/scripts/train_tier1_sionna.py` — FOMAML on multi-scene
  data with optional Tier-0 transfer (one line: `core.encoder.load_state_dict(...)`).
  Few-shot eval at K=1,2,5,10,20 on held-out scenes.
- `code/scripts/compare_tier1_runs.py` — few-shot curve plot.
- `code/scripts/viz_tier1_adaptation_curve.py` — query MSE at
  every inner-loop step (0..5), median-aggregated for robustness.
- `code/scripts/viz_tier1_nlos_breakdown.py` — per-NLOS-bin MSE bars.
- `code/scripts/viz_tier1_error_heatmap.py` — per-position error
  on floor plan.
- `code/scripts/viz_tier1_transfer_matrix.py` — train→val transfer
  matrix.

### Paper figure (TikZ)
- `figures/tier0_pipeline.tex` — standalone TikZ figure for the
  Tier 0 section. Compile via `\input{}` from `main.tex`. Matches
  the style of `figures/architecture.tex`. v2 layout is
  left-to-right: data prep column → ASTA encoder → SSL head →
  loss, with pretrained encoder below. ~241 lines.
- `code/scripts/preview_tier0_pipeline.py` — matplotlib preview
  of the TikZ figure (no pdflatex on build host). The user edited
  the .tex by hand after seeing the v1 preview; the preview script
  was rewritten for v2.

---

## 2. Datasets used

- `code/data/empty_open.csv` — Sionna `empty_open` scene,
  400 grids × 100 trials, 100% LOS. Copied from
  `~/Downloads/srp_research/grid2/sionna_uwb_grid_empty_open_A1A4swapped.csv`.
  ~10.5 MB. **Gitignored** via `code/data/*.csv` rule.
- The 20 other Sionna scenes live in `~/Downloads/srp_research/grid2/`
  (all `_A1A4swapped.csv` filenames). 6 of them are used as the
  Tier 1 multi-scene mix:
  - `empty_open` (0% NLOS, 20×20m outdoor) — train
  - `custom_room_large` (0% NLOS, 12.5×14.5m indoor) — train
  - `outdoor_blockages_pillars` (39% any-NLOS) — train
  - `indoor_blockages_pillars` (42% any-NLOS) — train
  - `outdoor_parking_01` (3% any-NLOS) — **val**
  - `indoor_garage_parking_01` (9% any-NLOS) — **val**

---

## 3. Trained checkpoints (in `code/checkpoints/`)

| File | Size | Description |
|------|------|-------------|
| `tier0_empty_open_25pct_100ep.pt` | 336 KB | F=1 ToA only, 25% mask, 100 ep |
| `tier0_empty_open_15pct_100ep.pt` | 336 KB | F=1 ToA only, 15% mask, 100 ep |
| `tier0_empty_open_f2_25pct_100ep.pt` | 338 KB | F=2, 50/50 channel weights, 100 ep |
| `tier0_empty_open_f2_25pct_3to1_100ep.pt` | 338 KB | F=2, 3:1 ToA:AOA weights, 100 ep **(used for Tier 1 transfer)** |
| `tier1_multi_scene_random_100ep.pt` | 338 KB | FOMAML, 4 train scenes, random init |
| `tier1_multi_scene_tier0init_100ep.pt` | 338 KB | FOMAML, 4 train scenes, Tier-0 transfer from the 3:1 ckpt |

Each `.pt` contains the `model_state`, `train_losses`, `val_losses`,
`cfg` dict, and (for F=2) `norm_mean` / `norm_std` for inference
denormalization. The Tier-1 ckpts also include `fs_results` (the
few-shot eval) and `info` (whether the encoder was Tier-0-init'd
or random).

---

## 4. Headline results

### Tier 0 SSL on `empty_open` (LOS only)
| Run | Final val MSE | RMSE on masked cells | Shrink slope |
|---|---:|---:|---:|
| F=1, 25% mask, 100 ep | 0.033 | 0.181 m | -0.001 |
| F=1, 15% mask, 100 ep | 0.050 | 0.224 m | -0.094 |
| F=2, 50/50, 100 ep | 0.0036 (z-norm) | 0.247 m / 6.32° | -0.001 / 48 |
| F=2, 3:1 ToA:AOA, 100 ep | 0.0090 (z-norm) | **0.211 m** / 6.45° | better |

**Key finding**: model converges to "predict approximately the
per-(window, anchor) mean" rather than tracking true deviations
(shrink slope ≈ 0). The 25% mask beats 15% on training MSE but the
per-masked-cell RMSE is essentially identical. The 3:1 channel
weighting cuts ToA RMSE by 15% vs the 50/50 default.

### Tier 1 FOMAML on 6-scene Sionna
| K (shots) | Random init | Tier-0 pretrained | Reduction |
|---:|---:|---:|---:|
| 1 | 691.2 | 66.1 | **+90%** |
| 2 | 70.7 | 62.1 | +12% |
| 5 | 50.1 | 39.9 | +20% |
| 10 | 36.4 | 31.2 | +14% |
| 20 | 33.1 | 28.3 | +15% |

**Tier-0 init wins at every K**, biggest at K=1. Inner-loop
adaptation curves show: (1) Tier-0 starts 4% lower at step 0,
(2) Tier-0 is more stable at K=1 (random diverges, tier0 stays
flat). Per-NLOS breakdown shows: Tier-0 helps on LOS trials
(+17%) but **diverges on NLOS trials** — an honest limitation
that the paper reports. Spatial heatmaps show Tier-0 produces
uniformly low error across the floor plan. Transfer matrix
shows source NLOS distribution should match target for best
transfer (empty_open → parking is best, NLOS pillars → parking
is worst).

---

## 5. What's in the paper (Tier 1 section, `sections/tier1.tex`)

The Tier 1 section is now 296 lines, 14 begin/end blocks balanced,
4 `\ref{}` cross-references. Contents:
1. FOMAML math (existing, unchanged)
2. Few-shot curve + table (`tab:tier1-fewshot`, `fig:tier1-fewshot`)
3. **Inner-loop adaptation curves** (`fig:tier1-adaptation`) +
   paragraph interpreting the step-0 advantage
4. **Per-NLOS-condition breakdown** (`fig:tier1-nlos`) + paragraph
   honestly reporting the NLOS failure mode + remedies
5. **Spatial error pattern** (`fig:tier1-heatmap`) + paragraph
   about the uniform spatial improvement
6. **Transfer matrix** (`fig:tier1-transfer`) + paragraph
   about source-target NLOS distribution matching

The Tier 0 section has a TikZ pipeline figure (`fig:tier0`)
inserted at the top.

---

## 5. Tier 2 — full adversarial training + viz (added 2026-07-07)

The README's open Status item "Tier 2: full adversarial training
loop" is **done**. We built the full two-optimizer adversarial
training pipeline on the synthetic data, plus 6 visualization
scripts and 6 paper figures.

### 5.1 What was built

**Code changes**
- `code/ultraclear/data.py` — `simulate_trace` now also returns the
  per-(T, M) NLOS mask; `TensorUWB.__getitem__` returns 5-tuples
  (X, y, env, cond, nlos). 4-tuple legacy still works.
- `code/ultraclear/tier2_micro.py` — `DisentangledModel.forward` now
  accepts `detach_c_for_adv=True` for the two-optimizer pattern.
  `disentangle_loss` sign fixed (`+` not `-`, the GR layer does the
  flip in backward).
- `code/scripts/train_tier2_synthetic.py` — full training loop with:
  - 2 optimizers (featurizer + adversary)
  - adversarial lambda schedule: warmup (5ep, λ=0) → ramp (10ep, 0→1) → 1.0
  - 3 featurizer losses: task MSE + nlos BCE + cond CE (auxiliary,
    keeps c informative about micro signal)
  - linear-probe env_acc diagnostic on the val envs every epoch
  - 80/20 env split, baseline (no-Tier-2) ASTA+LocationHead run for
    the comparison
- `code/scripts/viz_tier2_loss_curves.py` — 6-panel training
  dynamics figure
- `code/scripts/viz_tier2_adv_accuracy.py` — disentanglement
  diagnostic (the **headline** Tier 2 figure)
- `code/scripts/viz_tier2_c_space.py` — t-SNE of c, colored by
  env and cond (the **visual** disentanglement claim)
- `code/scripts/viz_tier2_film.py` — FiLM γ/β distribution +
  per-cond modulation profiles
- `code/scripts/viz_tier2_nlos_reliability.py` — NLOS head
  per-cond accuracy and calibration
- `code/scripts/viz_tier2_loc_comparison.py` — **headline numbers**
  bar chart: Tier 2 vs baseline, per-cond

**Paper changes**
- `sections/tier2.tex` — added a 6-paragraph "Experimental
  validation" subsection with all 6 figures. Also fixed the
  sign error in the loss equation from `-λL_adv` to `+λL_adv`
  (the GR layer in the model handles the sign in backward;
  the `+` on the task side is the standard DANN convention).
- Total: 320 lines added, 14 figures referenced, 6 begin/end
  blocks balanced.

### 5.2 Headline results (50 epochs, cond_loss_weight=2.0)

| Metric | Value | Verdict |
|---|---:|---|
| Final task MSE (train) | 0.54 | good convergence |
| Final adv CE | 1.78 | ≈ chance ln(8)=2.08 — adversary failing |
| Final cond CE | 0.06 | way below chance ln(3)=1.10 — c is cond-informative |
| Final nlos BCE | 0.50 | matches per-cell base rate |
| Linear-probe env_acc (val) | 0.56–0.83 | chance=0.50, partial disentanglement |
| c-space t-SNE | 3 cond clusters visible, env blob diffuse | visual disentanglement |
| Per-cond loc err (val, 8 envs total, 2 unseen) | cond 0: 5.83 vs 4.74 baseline (+23%) | tier 2 worse |
| | cond 1: 4.87 vs 4.49 (+9%) | tier 2 worse |
| | cond 2: **4.12 vs 4.47 (-8%)** | tier 2 better |

**The story**: Tier 2 helps on heavy NLOS (cond 2) and is neutral-to-slightly-worse on
easy conditions. This is a coherent (not uniform) result that the
paper reports honestly.

**The caveat**: probe_env_acc hovers 0.5–0.8 on the *unseen* val
envs. Chance is 0.5 (2 val envs), so the model is partially
disentangled — not at chance, but well below the 1.0 of a perfectly
entangled model. We attribute the residual env signal to (i) limited
adversary capacity (2-layer MLP, d_c=32) and (ii) 50-epoch budget.

### 5.3 Compute

- Full Tier 0+1+2 (synthetic, 8 envs, 50 Tier-2 epochs): **~2.5
  min on CPU**
- Each viz script: <30 s
- Checkpoint: `code/checkpoints/tier2_synthetic/tier2_synthetic.pt`
  (418 KB)

### 5.4 What I'd do with more time

1. Bigger adversary (4-layer MLP, d_c=64) + 200 Tier-2 epochs to
   drive probe_env_acc closer to chance
2. Per-anchor NLOS head (currently shared) — the per-anchor
   variation in NLOS is informative
3. Multi-seed runs (we have n=1; needs ≥3 seeds for a paper
   table)
4. Apply the same pipeline to Sionna multi-scene data
   (requires solving the "we don't have per-cell NLOS labels in
   Sionna" problem — coarse cond_id is all we have)

## 6. Open TODOs (ranked by impact)

1. **Re-train Tier 0 on a multi-scene mix** (LOS + NLOS) instead
   of empty_open alone. This is the most impactful next step —
   directly addresses the NLOS failure mode reported in §4.
   `train_tier0_sionna_f2.py` already supports it via `--csv`
   (just point at multiple CSVs or a combined one). The
   multi-scene dataset code in `multi_scene_dataset.py` can be
   adapted to do the sliding windows.
2. **Run Tier 1 FOMAML for 200 episodes** instead of 100 (paper's
   stated budget in the Hyperparameters paragraph; the table
   numbers are from the 100-episode run). 20s on CPU.
3. ~~Tier 2: full adversarial training loop.~~ **DONE 2026-07-07.**
   See §5 above. The remaining item from this is "run on Sionna"
   rather than "build the loop", which is a different problem
   (needs per-cell NLOS labels).
4. **Tier 2 on Sionna (synthetic → real-data transition).**
   The synthetic pipeline works; porting it to Sionna requires
   per-cell NLOS labels which the coarse cond_id doesn't provide.
   Could be solved with a self-supervised NLOS head (RNN predicts
   the next cell, large error = NLOS), but that needs a separate
   sub-project.
5. **Tier 3: numerical results.** The downstream heads
   (NLOS, restoration, location) are implemented; need
   end-to-end training runs to populate the experiments table.
6. **Paper experiments section** (`main.tex` line ~239 placeholder
   text "Full ablation tables... will be reported in the
   camera-ready version"). Could be filled in with the few-shot
   table, transfer matrix, and per-NLOS breakdown.

---

## 7. Compute / runtime notes

- Tier 0 F=1 (100 ep): ~70s on CPU
- Tier 0 F=2 (100 ep): ~75s on CPU
- Tier 1 FOMAML (100 ep): ~20s on CPU
- Tier 1 few-shot eval (5 K-shots × 2 val scenes × 30 trials): ~10s
- All viz scripts: <1 minute each

Total compute for the whole Tier 0+1 experimental campaign: well
under 30 minutes. A 200-episode FOMAML rerun + multi-scene Tier-0
pretrain would add ~5 minutes.

---

## 8. Open questions for future Justin

- Should the Tier 0 SSL loss weight the per-(T, M) error by the
  fraction of visible cells? Currently MSE is uniform over masked
  cells; if masking is biased toward one anchor the loss is
  biased accordingly. Probably fine, but worth a sanity check.
- The FOMAML instability at K=1 (random diverges to 71 m² at
  step 3) is well-known. Reptile (5-line change in
  `tier1_maml.py:maml_train`) is the suggested replacement per
  the docstring. Not tried.
- AoA wrap-around: azimuth is on a circle but MSE is on a line.
  A sin/cos encoding of AoA would fix this. Not tried.
- The `cfloat = np.percentile(arr, 2.0)` for the heatmap
  percentile bounds in `viz_tier0_heatmaps.py` was set empirically;
  if the figure is too dark/light on new data, tweak these.
- The matplotlib preview (`preview_tier0_pipeline.py`) uses
  mathtext instead of full LaTeX, so it has limitations (no `\bm`,
  no `\R`, no `\odot`). The TikZ file is the canonical version.
- The user edited `figures/tier0_pipeline.tex` by hand after the
  v1 preview; I do not know what the final state is. Re-read the
  .tex file before regenerating anything based on it.
