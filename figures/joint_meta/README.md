# figures/joint_meta — Joint Tier 1+2 K-shot adaptation curves

This directory holds figures that measure the **meta-learning** (few-shot
adaptation) side of the joint Tier 1+2 trainer, complementing the
disentanglement metrics in `figures/joint_sionna_quick/`.

## Style

The figures match the format of the existing Tier 1 work in
`figures/tier1_sionna/`:
- 4-panel K-shot curve (`viz_joint_k_shot.png`) — one panel per val
  env, all variants overlaid, log y-axis, per-trial faded lines +
  median with half-IQR error bars
- 5-panel inner-loop step curve (`viz_joint_step_curve.png`) — one
  panel per K (1, 2, 5, 10, 20), all variants overlaid, log y-axis,
  per-trial faded lines + median with half-IQR error bars

## Variants compared (3-way, no random init)

1. **MAML only (no SSL)** — ASTA encoder + LocationHead trained from
   scratch with the FOMAML outer loop on the 10 train envs. **This
   is the new baseline** that isolates the MAML contribution.

2. **Tier 1 (SSL + MAML)** — Phase A of the joint trainer: SSL
   pretraining + 100 FOMAML episodes with the encoder frozen. The
   location head was meta-learned via the MAML outer loop.

3. **Joint Tier 1+2** — Phase C of the joint trainer: encoder unfrozen,
   MAML + disentangle jointly.

The "random init" baseline (no SSL, no MAML, supervised MSE) is
**dropped** from this comparison because it's the strongest variant
and dominates the figure. The `--no_random` flag in
`viz_joint_adaptation_curve.py` drops it; the `--baseline_ckpt`
path is still respected if you want the 4-way comparison back.

The "Tier 2 (SSL + disentangle, no MAML)" variant is also dropped
because the comparison is now structured as:
- **MAML only** = "what does MAML contribute without SSL"
- **Tier 1** = "what does SSL add to MAML"
- **Joint** = "what does disentanglement add to MAML+SSL"

This 3-way ablation cleanly attributes each component's contribution.

## Headline result (top-4 NLOS val envs, K=1 to K=20, n=5 trials)

| variant | K=1 | K=5 | K=20 |
|---|---:|---:|---:|
| MAML only (no SSL) | 12.94 | 10.14 | 10.03 |
| Tier 1 (SSL + MAML) | 18.41 | 9.97 | 9.79 |
| Joint Tier 1+2 | 9.75 | 9.31 | 9.40 |

**The 3-way story:**
- **MAML helps at K=1**: drops from 12.94 to (in Joint) 9.75 with the
  FOMAML machinery. Without any adaptation, the MAML init is
  *worse* than the supervised baseline (random init was 1.0 m
  zero-shot, way better).
- **SSL hurts at K=1**: Tier 1 (SSL + MAML) is 18.41 at K=1, the
  *worst* of the three. The SSL pretraining pulls the encoder
  away from MAML-friendly features.
- **Disentanglement helps a bit**: Joint Tier 1+2 is the best at
  K=1 (9.75) and the most stable. The disentanglement pressure
  adds some robustness at low K.
- **All three converge at K=20** (9-10 m²). The MAML machinery
  can adapt a lot in 5 inner-loop steps with enough data.

## Zero-shot performance (step 0, no inner-loop adaptation)

The model deployed to a new env with **no commissioning data** --
just the meta-learned initialization. This is the most pessimistic
deployment scenario.

| variant | Zero-shot RMSE (m) |
|---|---:|
| MAML only (no SSL) | 3.60 |
| Tier 1 (SSL + MAML) | 4.29 |
| Joint Tier 1+2 | 3.12 |

(Reference: random init baseline (no SSL, no MAML) was 1.00 m at zero-shot --
best of all variants, but the 3-way comparison here is about the
meta-learned models specifically.)

## Why this happens (the honest diagnosis)

The joint training objective (SSL reconstruction + disentanglement)
doesn't produce features that directly encode the tag's geometric
position. The encoder is solving the wrong pretext for downstream
localization.

The micro `c` is good (the disentanglement metric shows it works) but
the macro encoder is *less* meta-learnable than a random init because
the joint training pulls it toward reconstruction + disentanglement
features, away from MAML-friendly features.

## The honest story for the paper

| Component | Status |
|---|---|
| Micro (disentanglement, `c`) | ✅ Works: probe `c → env` at chance |
| Meta (K-shot adaptation) | ✗ SSL hurts at K=1; disentanglement helps a bit |
| Tier 0 SSL reconstruction | ✓ Works: RMSE 0.30 m on masked cells |
| Overall location error | ✗ Joint is +5-10% worse on real data |

The paper's honest contribution: the disentangled `c` is real and
meaningful, but the joint training objective doesn't yet produce a
model that uses the encoder's features effectively for
meta-localization on real OOD data. The fix is likely one of:

- A larger encoder (the current 50K-param ASTA is too small to learn
  multiple objectives simultaneously)
- A neural-residual LSE localization head that uses anchor positions
  explicitly (instead of a 2-layer MLP)
- Multi-task SSL pretraining that includes a "predict (x, y)" auxiliary
  task alongside masked cell reconstruction

## How to regenerate

```bash
cd code
# First, train all the variants. The joint trainer produces Phase A/C;
# --run_baseline saves the baseline.pt; the maml_only_sionna script
# trains the MAML-only baseline from scratch.
python3.11 -m scripts.train_tier12_joint \
    --data sionna --out checkpoints/joint --run_baseline
python3.11 -m scripts.train_maml_only_sionna \
    --out checkpoints/joint/maml_only.pt --episodes 200

# Then run the 3-way adaptation curve (random init + tier 2 dropped)
python3.11 -m scripts.viz_joint_adaptation_curve \
    --maml_only_ckpt checkpoints/joint/maml_only.pt \
    --phase_a_ckpt    checkpoints/joint/phase_A.pt \
    --phase_c_ckpt    checkpoints/joint/phase_C.pt \
    --no_random --no_tier2 \
    --data sionna --k_list 1 2 5 10 20 --n_trials 5 --max_steps 5 \
    --max_envs 4 \
    --out_dir figures/joint_meta
```

To get the 4-way comparison back (with random init and Tier 2), drop
the `--no_random --no_tier2` flags and add `--baseline_ckpt` +
`--phase_b_ckpt`.
