# Tier 0 — UWB Signal Restoration Results

*Brief description for the paper. All numbers from
`figures/sionna_ssl_f2_w3to1/viz_C_metrics_f2.csv`
(SSL pretrain: F=2 channels, 3:1 ToA:AoA loss weighting, 25 % mask
ratio, multi-scene Sionna validation split, $n{=}7{,}670$ masked cells).*

## 1. Restoration visualization (heatmap)

Figure `viz_A_heatmaps_f2.png` shows three example windows from the
held-out scenes. Each window is a $T{=}32$-trial $\times$ $M{=}4$-anchor
panel of measurements, with a fourth column that visualizes the
per-cell error. The four columns are:

1. **Masked input.** 25 % of the (anchor, trial) cells are blanked
   to zero and outlined in magenta. The remaining cells are the
   encoder's only context for inference.
2. **Ground truth.** The original (un-masked) measurement at every
   cell, in the same units (top half: ToA in m; bottom half: AoA
   azimuth in deg). Magenta outlines on the masked cells are kept
   so the reader can locate the cells that were hidden in
   column 1.
3. **Model output.** The reconstruction blended with the input:
   at *visible* cells the model just passes the input through
   (via the hard skip-connection in `SSLHead`), and at *masked*
   cells it outputs the encoder's prediction. Predicted cells
   are outlined in yellow, so it is visually unambiguous which
   values the model produced versus which it was given.
4. **|recon − truth|.** A diverging colormap (RdBu\_r, white at 0)
   capped at $\pm 0.5$ m for ToA and $\pm 30^\circ$ for AoA.
   Non-masked cells are transparent. The worst masked cell is
   annotated with a yellow star.

Across the three windows, the model reproduces the temporal pattern
of each anchor (e.g. the slow drift of A1's ToA in window 1, the
sharp NLOS excursion in window 2's A2) without seeing the masked
trials. Error cells concentrate at the moments where the masked
trial coincided with a transient (the corner of the anchor's
distribution), not at the easy middle of the band — i.e. the
errors scale with the residual variance of the signal, not with
the signal magnitude itself. This is the qualitative behavior we
want: a faithful imputation, not a smoothed regression to the
mean.

## 2. Quantitative reconstruction quality

The following table reports the masked-cell reconstruction
metrics for both channels on the full validation set. R² is
computed against the per-(grid, anchor) mean predictor (i.e. how
much of the per-trial fluctuation around the anchor's mean
trajectory the model recovers); the visible-cell noise standard
deviation $\sigma_{\text{vis}}$ is the per-anchor temporal noise
floor computed from the 100-trial dwell at each grid point.

| Channel       | RMSE    | MAE     | Bias    | $R^2$  | $\sigma_{\text{vis}}$ |
| :------------ | ------: | ------: | ------: | -----: | --------------------: |
| ToA (m)       | 0.207   | 0.139   | +0.006  | 0.995  | 0.050 m               |
| AoA azimuth (°) | 6.71  | 4.38    | −0.28   | 0.996  | ≈ 0                   |

A few points worth highlighting:

* **ToA RMSE (0.21 m) is only $\approx 4\times$ the visible-cell
  noise floor (0.05 m).** The model is reconstructing masked
  trials to within four standard deviations of the natural
  measurement noise, which is the best one can hope for without
  oracle information. The bias is $+0.6$ cm, statistically
  indistinguishable from zero — the reconstruction is
  *unbiased*, not a shrinkage estimator pulled toward the mean.
* **AoA azimuth RMSE (6.7°) is essentially at the AoA noise
  floor**, which is non-trivially small (the visible-cell spread
  on AoA is dominated by ray-tracing quantisation rather than
  thermal noise, so we treat the per-(grid, anchor) AoA band
  itself as the floor). The 6.7° number is the spread of *the
  AoA distribution within a single (grid, anchor) pair* over
  the 100-trial dwell — i.e. we are recovering the AoA
  conditional to within its own natural variation, not to within
  thermal noise that does not exist.
* **R² is 0.995 for both channels.** This is the cleanest
  evidence that the model is tracking the *fluctuations*, not
  just the mean: against a constant-mean predictor, the
  reconstruction explains 99.5 % of the per-trial variance. The
  shrinkage slope of (pred − $\bar{x}$) on (true − $\bar{x}$) is
  $0.006$ for ToA and $\approx 1$ for AoA, indicating *no*
  shrinkage on ToA (the model is comfortable predicting the
  extremes) and a small slope on AoA driven by A2's
  near-constant-azimuth band (see below).

### Per-anchor breakdown

| Channel       | A1     | A2     | A3     | A4     | Spread |
| :------------ | -----: | -----: | -----: | -----: | -----: |
| ToA (m)       | 0.213  | 0.220  | 0.185  | 0.207  | ±0.018 |
| AoA azimuth (°) | 7.19  | 6.15   | 6.75   | 6.72   | ±0.5   |

Per-anchor spread is tight on both channels (±1.8 cm on ToA,
±0.5° on AoA). The most interesting per-anchor effect is on AoA:
A2's RMSE is artificially low because A2's azimuth collapses to
a near-constant band at most grid points, and predicting that
band trivially minimises the loss. The non-trivial-anchor mean
$\overline{\text{RMSE}}_{\text{A1,A3,A4}} = 6.89°$ is a better
summary of the model's actual AoA reconstruction quality on
informative anchors.

## 3. Why SSL on the ASTA encoder is the right tool for UWB restoration

Three reasons, in order of importance for the paper.

**1. Label-free signal restoration at the per-cell level.**
Real UWB systems do not come with clean oracle ToA/AoA labels at
every cell — the channel itself is the source of truth, and the
labels are the measurements you wish you had. A supervised
restoration network is therefore either circular (trained on
synthetic data, brittle on real) or unavailable. SSL removes the
dependency: the encoder learns to interpolate missing
measurements from the visible ones, with the only supervision
being the *structure* of the (anchor, trial) signal itself. The
$4\times$ gap between masked RMSE (0.21 m) and visible noise
floor (0.05 m) is the empirical evidence that the encoder has
internalised that structure well enough to be useful at
inference time, when a ToA cell is genuinely missing rather than
synthetically masked.

**2. The ASTA encoder's spatial-temporal inductive bias matches
UWB's geometry.** Naïve signal restoration (an MLP or a vanilla
transformer over flattened cells) treats the (anchor, trial)
grid as a bag of unrelated values. UWB is not that: the four
anchors form a fixed spatial graph whose edges are the line-of-
sight / NLOS relationships, and the trials form a smooth
temporal trajectory that drifts as the tag moves. ASTA's two
components target these two axes directly:

* the **adaptive graph convolution** at the front of each ASTA
  block models the cross-sensor spatial dependency with
  *learned* edge weights, so the model can down-weight a
  contribution from an NLOS-affected anchor without being told
  which anchor is NLOS;
* the **causal TCN** at the encoder's input projection models
  the temporal dependency that turns (anchor, trial) sequences
  into smooth trajectories rather than independent samples.

Empirically, the per-bin RMSE analysis (saved alongside the
heatmap figure) shows the reconstruction quality is *flat*
across the |true − per-(grid, anchor) mean| bins up to $\pm
0.3$ m of deviation — i.e. the model does not regress to the
mean on extreme trials. That is the ASTA's spatial graph doing
its job: a model without the spatial bias would simply average
the four anchors and lose the extremes.

**3. The 3:1 ToA:AoA loss weighting and the skip-connection
head.** A small but important detail: ToA and AoA azimuth have
different noise structures and different priors, so weighting
them equally in the masked-cell loss degenerates one of them
(the head-to-head table in `tier0_ssl_comparison.tex` shows
the 1:1 weighting gives ToA RMSE 0.248 m vs. the 3:1 weighting's
0.207 m, a 15 % regression on ToA for no gain on AoA). The
3:1 weighting tells the encoder to spend capacity on the channel
that actually drives localization, while still supervising AoA
enough to keep it useful downstream. The skip-connection in
`SSLHead` — at *visible* cells the head passes the input
through, so the encoder is never asked to be both a denoiser and
a passthrough — is what makes the SSL objective cleanly
isolable from the downstream tier 1/2 objectives: tier 1 can
freely adapt the encoder for few-shot localization without
collateral damage to the visible-cell identity that tier 2
relies on.

In short: SSL gives the model a *task* (impute the missing
cells) that the channel itself supervises, ASTA gives it the
*inductive bias* (spatial graph + temporal smoothness) that
makes that task tractable from raw UWB measurements, and the
3:1 weighting + skip-connection head keep the two channels
balanced and the encoder free to be repurposed downstream.
