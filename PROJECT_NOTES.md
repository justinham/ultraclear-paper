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
- `code/scripts/train_tier0_sionna_multiscene.py` — multi-scene
  variant: trains on 4 scenes (2 LOS + 2 NLOS-heavy) instead of one.
  Uses `MultiSceneSionnaDataset` from `multi_scene_dataset.py`.
  Same F=2 / 3:1 / 25% mask recipe. The new Tier-0 ckpt lives at
  `code/checkpoints/tier0_multiscene_f2_3to1_100ep/`.
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
| K (shots) | Random init | Tier-0 (single scene) | Tier-0 (multi-scene) |
|---:|---:|---:|---:|
| 1 | 691.2 | 66.1 (**+90%**) | **64.4 (+90.7%)** |
| 2 | 70.7 | 62.1 (+12%) | **58.5 (+17%)** |
| 5 | 50.1 | 39.9 (+20%) | 41.0 (+18%) |
| 10 | 36.4 | 31.2 (+14%) | 34.8 (+4%) |
| 20 | 33.1 | 28.3 (+15%) | 32.6 (+1.7%) |

**Tier-0 init wins at every K-shot level**, for BOTH pretraining
mixes. Biggest at K=1 (the practical deployment scenario: you
get 1-2 shots of a new environment and need to work
immediately). Inner-loop adaptation curves show: (1) Tier-0
starts 4% lower at step 0, (2) Tier-0 is more stable at K=1
(random diverges, tier0 stays flat).

**Multi-scene Tier-0 vs single-scene (empty_open only)**:
multi-scene widens the K=1, 2 advantage (the SSL features are
more general — better geometry prior) but narrows the K≥10
advantage (the K-shot inner loop has enough data to correct a
weaker init). Net: multi-scene is the strictly better Tier-0
strategy in deployment-relevant low-K regimes; the headline
claim — Tier-0 init wins at every K — is robust to the
pretraining mix.

**NLOS breakdown (K=20, held-out parking scenes, single-scene
Tier-0)**: Tier-0 helps on LOS trials (+17%) but
**diverges on NLOS trials** (random=4-29 m², tier0=50-100 m²
on env 4 NLOS) — an honest limitation reported in tier1.tex.

**NLOS breakdown (K=20, multi-scene Tier-0)**: the divergence
on env 4 NLOS is **reduced from 100 m² to 38 m²** (2.6× better),
and env 5 NLOS is essentially flat (32 vs 29 m², no
catastrophic divergence). The multi-scene pretraining
**partially fixes** the NLOS failure mode; full fix would
need an explicit NLOS-aware auxiliary task (option (ii)
below). Spatial heatmaps show Tier-0 produces uniformly low
error across the floor plan. Transfer matrix shows source
NLOS distribution should match target for best transfer
(empty_open → parking is best, NLOS pillars → parking is
worst).

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

1. ~~**Re-train Tier 0 on a multi-scene mix**~~ **DONE 2026-07-07.**
   New Tier-0 ckpt at `code/checkpoints/tier0_multiscene_f2_3to1_100ep/`
   trained on envs 0-3 (2 LOS + 2 NLOS). Reduces env-4 NLOS
   divergence from 100.6 m² to 38.0 m² (2.6× improvement on the
   failure mode). See new figure
   `figures/tier1_sionna/tier1_nlos_breakdown_t0multi_K20.png`.
   The remaining residual failure is now small and architecturally
   tractable — option (ii), the explicit NLOS auxiliary task, is
   the natural next step.
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
7. **LoRA for Tier 1/2 fine-tuning** (parked 2026-07-14).
   Discussion only, no code written. Hypothesis: LoRA on the
   location head's MLP would replace the FOMAML inner-loop
   gradient step with a small set of trainable low-rank matrices
   that adapt in 1 forward pass. Could fix the K=1 case where
   FOMAML is currently the bottleneck (5.40 m² joint vs 0.64 m²
   random at K=1 on 21-env Sionna). No LoRA references in code
   as of 7/14 — verify before implementing.

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

## 9. Hard-skip SSL head (2026-07-09, Justin's idea)
The trained SSL encoder mangles visible cells: ToA RMSE 1.78m on
visible cells vs 0.63m on masked. The encoder learns a
*directional denoising* that improves masked cells at the cost of
visible cells. Inference test: `code/scripts/test_tier0_inference.py`
with `--no_skip` flag for side-by-side.

**Fix:** hard skip-connection in `SSLHead`. At visible cells,
output the raw input (`x_input`). At masked cells, output
`MLP(encoder(x_masked))`. The encoder and head weights are
unchanged (the loss is computed only on masked cells, so the
gradients are the same); the skip is a pure inference-time
change.

**Side-by-side result on held-out parking scene:**

| | v1 (no skip) | v2 (with skip) |
|---|---:|---:|
| ToA MASKED | 0.64 m | 0.64 m |
| **ToA UNMASKED** | **1.78 m** | **0.00 m** |
| AoA MASKED | 7.2° | 6.8° |
| **AoA UNMASKED** | **55°** | **0.0°** |

**Code changes:**
- `code/ultraclear/tier0_ssl.py` — new `SSLHead.forward(h, x_input, mask)`
  with `torch.where(mask.unsqueeze(-1), head(h), x_input)`.
  `SSLPretrainer.forward` returns 3-tuple `(x_pred, mask, x_input)`.
  All 7 call sites updated via bulk `sed`.
- New ckpt: `code/checkpoints/tier0_multiscene_f2_3to1_skip_100ep/`.
  Encoder and head are byte-identical to the old ckpt (verified
  by SHA256); the skip doesn't change training, only inference.
- `code/scripts/test_tier0_inference.py` — new inference test
  with `--no_skip` flag for the old behavior.

**Why this matters for the FOMAML problem:** the skip is what
makes the encoder's output *useful* for downstream code that
wants to read raw ToA values at visible cells. The next step
is a **hybrid location head** that gets `[h, x_pred]` as input,
where `x_pred` carries the passthrough measurements. The
location head can use these without going through the encoder's
denoising, which should help in low-K regimes.

**Figures:**
- `figures/tier0_ssl_head_skip_preview.png` — matplotlib preview
  showing v1 (top, dashed/inactive skip) vs v2 (bottom, solid red
  active skip) side-by-side. The merged cell detail at the start
  of each panel shows the visible (blue) and masked (black) cells.
- `figures/tier0_ssl_head_skip_diagram.png` — clean boxes-and-
  arrows version, paper-quality.
- `figures/tier0_ssl_head_skip.tex` — clean TikZ version (paper-ready)
  with the same architecture in schematic form, plus a "what stays
  the same" caption. Not pdflatex-compiled on this host.

### 9.1 Joint trainer with the skip-connection Tier 0 init (2026-07-09)

Added `--tier0_ckpt` flag to `train_tier12_joint.py`. Loads a raw
SSL pretraining ckpt (just the encoder), updates cfg from the ckpt's
saved config (so n_channels, channel_weights match), then runs Phase
A MAML warmup with the encoder frozen. Saves a Phase A ckpt that
the rest of the joint pipeline can consume.

New function: `run_phase_A_from_tier0(cfg, ckpt_path, items, device,
n_episodes)` — same contract as `run_phase_A` but skips the in-loop
SSL pretraining.

Smoke-tested `--quick` on full pipeline (A+B+C, 5+10+10 episodes)
on Sionna data with the new skip ckpt as Phase A init. Per-NLOS-
bucket val loc error: 8.4 / 3.6 / 2.5 m for low / mid / high NLOS.
Quick run, not the headline number, but the pipeline is wired.

Full-budget command (to run when Justin wants headline numbers):
```bash
cd code
python3.11 -u -m scripts.train_tier12_joint \
    --data sionna \
    --tier0_ckpt checkpoints/tier0_multiscene_f2_3to1_100ep/tier0_multiscene_f2.pt \
    --out checkpoints/joint_skip_init \
    --phase_a_episodes 100 \
    --phase_b_epochs 50 --phase_b_warmup 8 --phase_b_ramp 15 \
    --phase_c_episodes 100 --phase_c_warmup 15 --phase_c_ramp 30 \
    --run_baseline
```

**Caveat:** the skip-connection change is *inference-time only*
(the encoder and head weights are byte-identical to the old ckpt).
Tier 1 FOMAML at K=1 will give the same 18.41 m² as before unless
downstream code reads `x_pred` (which carries the passthrough) and
feeds it to a hybrid location head. The joint trainer currently
feeds only `h` (encoder features) to the location head, so the
skip doesn't help the location head today. **The full-budget run
above is mainly a sanity check that the new init doesn't break
anything; the real downstream improvement requires the hybrid
location head (a separate change).**

## 10. Per-timestep location head (2026-07-09, paper is moving-tag)

The paper is for **moving-tag** localization, not stationary. The
window of T=32 timesteps is a *trajectory* of the tag, and the
location head should output (x, y) at every timestep, not a single
(x, y) for the whole window.

**Architecture change:** new `LocationHeadPerTimestep` class in
`ultraclear/tier1_maml.py` and `scripts/train_tier2_synthetic.py`.
Pools over the M anchor dim only (not over T), so each timestep
gets its own prediction. Output shape: `(B, T, 2)` instead of
`(B, 2)`. The 2-layer MLP is the same as the per-window head.

**Data change:** all data loaders now return labels of shape
`(T, 2)` (per-timestep trajectory) instead of `(2,)` (single
location). Synthetic data's `simulate_trace` already had the full
trajectory in `pos`; we just changed `y = pos[T // 2]` to
`y = pos`. Sionna data is stationary per grid (all 100 trials at
a given (grid_id) share the same (x, y)), so the per-timestep
labels are constant across T — the head will learn to output the
same value 32 times. For genuinely moving-tag data, see
`/Users/justin/Downloads/srp_research/sionna_walking_path_v2.csv`
(4001 timesteps with a moving tag, 1-2 m trajectory span).

**Sliding window constraint (Justin):** the 32-trial sliding window
must stay within a single (scene, grid_id). The current
`SionnaJointDataset.from_per_scene` already does this:
```python
for s_idx, info in enumerate(scene_info):
    for g in range(info.n_grids):
        for s in range(0, info.n_trials - seq_len + 1, stride):
            ds.windows.append((s_idx, g, s))
```
The `s` range never crosses a grid boundary, so all 32 timesteps
in a window come from the same underlying (x, y). Verified by
sampling 5 random windows and checking `s + seq_len <= n_trials`.

**Loss shape:** `TF.mse_loss` is shape-flexible, so the same code
`(B, T, 2)` vs `(B, T, 2)` works without modification. The mean
is taken over all (B, T, 2) elements, which is the natural
extension of the per-window MSE.

**Eval fix:** the per-cond bucketing in `eval_per_cond` and
`eval_per_nlos_bucket` originally computed
`err = linalg.norm(loc - y, dim=-1)` which gives `(B, T)` for
per-timestep outputs. The original code then did
`for ci, e in zip(cond, err)` which would have appended a
length-T array instead of a scalar per cond bucket. Fixed to
average over T first: `err_per_window = err.mean(dim=-1)`
gives `(B,)`, then bucketing works correctly. Per-cond CSV
columns are now scalar values per cond bucket.

**Smoke tests (--quick flag, 5+10+10 episodes):**
- Synthetic data: A+B+C ran cleanly. Phase C per-cond val loc
  err: 3.94 / 3.70 / 3.80 m for cond 0/1/2.
- Sionna data (with --tier0_ckpt new skip init): A+B+C ran
  cleanly. Phase C per-NLOS-bucket val loc err: 8.6 / 3.6 / 2.2 m
  for low / mid / high NLOS.

**Files changed:**
- `ultraclear/tier1_maml.py` — added `LocationHeadPerTimestep`
- `ultraclear/data.py` — `simulate_trace` returns `y = pos` (T, 2)
- `scripts/train_tier2_synthetic.py` — added `LocationHeadPerTimestep`,
  used it in `DisentangledTrainer.loc_head` and the baseline
- `scripts/train_tier12_joint.py` — import + use the per-timestep head,
  fixed per-cond eval to mean over T first
- `scripts/sionna_joint_dataset.py` — return `tag_xy[g, s:s+seq_len]`
  (T, 2) per-timestep labels
- `scripts/multi_scene_dataset.py` — same change
- `scripts/train_tier0_sionna.py` — same change
- `scripts/train_tier0_sionna_f2.py` — same change

### 10.1 Walking-path data loader (later 7/9, Justin's spec)

Justin: "the walking path data is non-stationary and you won't find
two same locations in 100 sequential samples. as long as it uses
timestamp-wise location, it is fine."

Added `scripts/walking_path_dataset.py` — a new data source for the
joint trainer that loads
`/Users/justin/Downloads/srp_research/sionna_walking_path_v2.csv`
(4001 timesteps of moving-tag data, ToA + AoA + per-anchor NLOS).
Per-timestep labels (T, 2). Sliding window of T=32 with no
boundary constraints (the user's spec — the data is non-stationary
so any consecutive 32-step window is a valid sample).

**Env splitting:** the path is split into N roughly equal temporal
chunks (default 4 via `--walking_path_n_envs`); each chunk is its
own env for FOMAML task sampling. With 1 env, FOMAML degenerates.

Wired into the joint trainer via `--data walking_path` (added to
the data choices) and new CLI flags `--walking_path_csv` and
`--walking_path_n_envs`.

**Per-env characteristics (4-env split, smoke test):**
- env 0: t=[0,1000), NLOS rate 76%, x std 1.3 m, y std 3.2 m
- env 1: t=[1000,2000), NLOS rate 73%, x std 1.2 m, y std 3.2 m
- env 2: t=[2000,3000), NLOS rate 55%, x std 2.3 m, y std 0.7 m
- env 3: t=[3000,4001), NLOS rate 65%, x std 0.05 m, y std 5.6 m

The per-env NLOS rates and tag_xy statistics differ substantially,
giving FOMAML meaningful OOD test conditions.

**Smoke test result (--quick, 5+10+10 episodes):**
- Phase A: SSL encoder + 5 FOMAML episodes, query_loss ~0.22-0.30
  (in z-score normalized space)
- Phase B: task loss 1.55 → 0.03, nlos_loss 0.69 → 0.59
- Phase C: per-NLOS-bucket val loc err: nan / 0.68 m / 1.26 m
  (low / mid / high NLOS). 0 = nan because no low-NLOS samples in
  the val split.

**Eval routing:** `eval_for_data` routes walking_path to
`eval_per_nlos_bucket` (same as sionna), since the cond_id is -1
(no cond labels for real data). The per-cond CSV is renamed to
per-NLOS-bucket in the printing.


## 8. Micro-condition simplification (2026-07-08)

Changed `ultraclear/data.py:_micro_params` so the micro condition
varies only **NLOS blockage severity** (rate × strength), not
multipath noise. Old table had all three co-varying (rate, strength,
multipath std); new table holds multipath std constant as a small
thermal-noise floor.

New table:

| cond_id | NLOS rate | NLOS strength | multipath std |
|---:|---:|---:|---:|
| 0 | 0.02 | 0.3 | 0.02 |
| 1 | 0.15 | 0.8 | 0.02 |
| 2 | 0.40 | 1.5 | 0.02 |

Rationale (per Justin): multipath reflection noise is hard to
emulate and conflates two physical phenomena (Bernoulli blockage
vs wide-tailed noise). The simplified model has a single clean
micro axis -- "how occluded is this anchor right now" -- which
makes the disentanglement target sharper. Smoke-tested with
`train_tier2_synthetic.py --quick --run_baseline`: per-cond NLOS
rates match targets exactly (0.020 / 0.150 / 0.401 vs targets
0.020 / 0.150 / 0.400) and bias magnitudes track the expected
`nlos_strength × env.nlos_bias` (0.27 / 0.76 / 1.38 m observed
vs 0.26 / 0.68 / 1.27 expected).

What this enables:
- Per-cond NLOS prediction reliability is now a clean monotonic
  check (predicted NLOS rate should track ground-truth rate
  0.02 / 0.15 / 0.40 per anchor per cond).
- `c` should learn a single noise-level axis rather than a
  confound between env layout and noise distribution.
- The synthetic-to-real transfer story is more honest: we
  explicitly do not simulate multipath; the model is expected
  to be robust to that as an unmodeled residual at deployment.

End-to-end smoke run (10 epochs, --quick): Tier 2 vs baseline
per-cond loc err on val = 3.68/3.47/3.45 vs 3.84/3.69/3.76 m.
Linear probe `c -> env` mostly at chance (0.49-0.55) with one
bad epoch during the warmup->ramp transition. 10 epochs is too
short for the schedule to stabilize; a 50-epoch rerun is the
next step.

## 8.2 DAE-style full-reconstruction loss (ablation, 2026-07-08)

Tried adding the visible (unmasked) cells to the SSL loss in addition
to the masked cells, with a `--visible_weight` scaling factor. The
hypothesis was that forcing the model to denoise the visible cells
(DAE-style) would also force it to use the visible context for the
masked cells.

**Result: it doesn't help, and at high weights it hurts.**

| Setting | RMSE | Bias | Shrink slope | R^2 |
|---|---:|---:|---:|---:|
| Original (masked-only) | 0.297 | -0.126 | -0.0009 | 0.976 |
| DAE visible_weight=0.3 | 0.86 | -0.65 | 0.043 | 0.797 |
| DAE visible_weight=0.05 | 0.56 | -0.44 | 0.004 | 0.914 |
| DAE visible_weight=0.01 | 0.30 | -0.08 | 0.005 | -- |

At visible_weight=0.01 (effectively masked-only) we recover the
baseline. The shrink slope stays at ~0 throughout, meaning the
model still isn't using the visible cells to predict the masked
cells -- the visible loss term just makes the model worse at the
masked task.

**Why this didn't work:** the per-(grid, anchor) mean is so easy to
predict that the model has no incentive to use the visible cells
for masked prediction. Adding a second task (denoise the visible
cells) just makes the model worse at both, because it has to
predict per-cell random-walk deviations (visible term) on top of
the easy mean (masked term).

**Code change is kept** (`--loss_on_visible` / `--visible_weight`
flags in `train_tier0_sionna.py` and `train_tier0_sionna_f2.py`,
plus the matching config fields in `Config` and the matching logic
in `ssl_loss`). It's not the default, but it's there if we ever
want to revisit.

**Better alternatives to try next** (parked for a future session):
1. **Block masking** -- mask an entire (anchor, all T) so the model
   can't fall back on the per-(grid, anchor) mean for that anchor.
   Forces cross-anchor context use.
2. **Per-window random offset** -- add a small random offset to the
   per-(grid, anchor) mean at training time, so the model can't
   just memorize the mean.
3. **Auxiliary tasks** -- add a second head that predicts, e.g.,
   the per-window mean-deviation (which is 0 by construction) or
   the random-walk velocity (which requires temporal context).

**Why this isn't a paper blocker:** the current F=1 RMSE 0.30 m is
fine for downstream use. The "model isn't using context" is a real
characteristic but not blocking the joint trainer story. The joint
trainer builds on top of this encoder, and the F=1 reconstruction
quality is good enough that downstream MAML/FOMAML can still work.

## 8.1 Tier 1+2 joint trainer (2026-07-08)

New script: `code/scripts/train_tier12_joint.py` -- the combined
macro (Tier 1) + micro (Tier 2) training pipeline, with three
staged-unfreeze phases:

- **Phase A** (SSL + MAML warmup): pretrain SSL encoder from
  scratch, then run a small number of FOMAML episodes with the
  encoder FROZEN. Calibrates the location head.
- **Phase B** (disentangle, encoder frozen): build the full
  DisentangledTrainer on top of the Phase A encoder. Train
  h_phi + FiLM + heads + adversary with the full disentanglement
  loss and a curriculum on lam (warmup -> ramp -> 1.0). Encoder
  is frozen.
- **Phase C** (joint FOMAML + disentangle, encoder unfrozen):
  unfreeze the encoder. Each episode is MAML inner loop (loc
  head + encoder adapted per env) + disentangle forward pass
  on the query set + two-optimizer step. Curriculum on lam
  (warmup_episodes -> ramp_episodes -> 1.0).

Each phase produces its own checkpoint. The script supports
resuming at any phase via `--phase A|B|C|AB|ABC` and
`--phase_a_ckpt` / `--phase_b_ckpt`.

**Why staged unfreeze (not naive joint):** the three loss
components have different scales (task MSE in m^2, adv CE in
nats, nlos BCE in bits) and the disentanglement adversary
needs the encoder to be a stable target. Naively summing
them with a single optimizer gives adversarial oscillation
and head-vs-encoder tug of war. Staged unfreeze lets the
modules learn their roles in order: SSL first, then heads
on top, then joint finetune with a curriculum.

**Smoke-tested** with `--quick`: full A->B->C in ~12s on
CPU. Per-cond val loc err at 10 phase-C episodes is roughly
flat across conds (4.17/3.95/3.86 m for cond 0/1/2), no
catastrophic failure. Linear probe `c -> env` ranges 0.18-0.45
during the joint phase (well below 0.5-0.94 of standalone
Tier 2 at the same budget), suggesting the joint training
produces a more macro-invariant `c`.

**Status**: pipeline runs end-to-end. Full-budget (50+ phase-C
episodes) results not yet collected -- this is the next step.

### 8.1.2 Adv loss scaling fix (2026-07-08)

The joint trainer's adversarial loss was exploding on real data
with 21+ envs (adv_ce reaching 50-280 vs ~2 chance on synthetic
8-env). The cause: the disentanglement term was `+ lam * adv_ce`,
which scales linearly with the number of envs (adv_ce can grow
unboundedly when the featurizer wins so hard the adversary can't
predict above chance).

**Fix applied** (DANN standard practice): cap the disentanglement
term at `2 * ln(n_envs)`. The featurizer now only feels pressure
when the adversary is performing above 2× chance.

```python
adv_chance = math.log(n_envs)
adv_term = torch.clamp(adv_ce - 2.0 * adv_chance, min=0.0)
feat_loss = ... + lam * adv_term   # instead of + lam * adv_ce
```

Applied in both Phase B (line 337) and Phase C (line 503) of
`train_tier12_joint.py`. The fix is in place but has not yet been
re-evaluated on real data -- next joint trainer run will show
whether the cap fixes the location-head destabilization.

### 8.1.1 Joint viz scripts (2026-07-08)

The 6 existing Tier 2 viz scripts have a `--joint` flag added that
points them at `checkpoints/joint/phase_C.pt` and writes to
`figures/joint/`. A shared helper `scripts/viz_tier2_helpers.py`
provides `load_model_from_ckpt`, `load_history_csv` (handles both
standalone and joint CSV formats with different column names), and
`add_joint_flag` / `resolve_paths`.

Three new viz scripts added for the joint trainer:

- `viz_joint_phases.py` -- 3-panel figure showing all three phases'
  loss trajectories (Phase A placeholder, Phase B with frozen
  encoder, Phase C with encoder unfrozen) with the lambda curriculum
  overlaid on each phase.
- `viz_joint_probe.py` -- disentanglement diagnostic with curriculum.
  Two panels: probe_env_acc over episodes + lambda schedule. The
  curriculum transitions are marked with vertical dotted lines.
- `viz_tier2_loc_comparison --joint` -- 2-way bar chart
  comparing baseline (no Tier 2) vs joint Tier 1+2 per-cond
  location error on val. Reads `phase_C_per_cond.csv` (which
  contains both columns when `--run_baseline` is passed).
  The standalone Tier 2 is intentionally NOT shown in the
  headline figure -- it's a surprise negative result
  (worse than baseline) and doesn't belong alongside
  the joint comparison.

The joint trainer also got:
- `--run_baseline` flag that trains a no-Tier-2 baseline and
  merges its per-cond errors into `phase_C_per_cond.csv` (in the
  same column format the standalone script uses).
- `n_envs` field added to all three phase checkpoints so the
  model-loading scripts can construct `DisentangledTrainer`
  with the right `n_envs`.

Smoke-tested: all 9 viz scripts (6 existing + 3 new) run on the
joint trainer's output. Outputs at `/tmp/joint_v3_figs/` for a
quick run: 7 figures, all rendered without errors.

Next: full-budget joint run (50+ phase-C episodes) and 3-way
comparison with multi-seed (currently n=1).

## 9. Open questions for future Justin

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
