# figures/joint_full — Full-budget joint Tier 1+2 visualization outputs

This directory holds figures produced by a full-budget run of the
joint Tier 1+2 training pipeline (Phase A: 50 SSL + 100 MAML,
Phase B: 50 disentangle epochs, Phase C: 100 episodes).

## Scripts that write here

- `viz_joint_phases` → `phases.png`
- `viz_joint_probe` → `probe.png`
- `viz_tier2_c_space --joint` → `c_space.png`
- `viz_tier2_film --joint` → `film.png`
- `viz_tier2_nlos_reliability --joint` → `nlos.png`
- `viz_tier2_adv_accuracy --joint` → `adv.png`
- `viz_tier2_loss_curves --joint` → `loss.png`
- `viz_tier2_loc_comparison --joint` → `loc.png`  (the headline 2-way comparison)

## What each figure shows

| File | What it shows |
|---|---|
| `phases.png` | All three training phases' loss trajectories in one figure. Phase A is a placeholder (MAML losses live in `phase_A.pt`, not a CSV); Phase B is disentangle with frozen encoder; Phase C is joint FOMAML + disentangle with encoder unfrozen. Lambda curriculum overlaid on each phase. |
| `probe.png` | Disentanglement diagnostic. Probe env accuracy on c, with the lambda curriculum on a side panel. Vertical dotted lines mark curriculum transitions. |
| `c_space.png` | t-SNE of c, colored by env_id (left, should look like a blob) and by micro condition (right, should show 3 clusters). |
| `film.png` | FiLM γ/β distributions + per-cond modulation profiles. Sanity check that the FiLM is doing something (γ ≠ 1.0, β ≠ 0.0). |
| `nlos.png` | Per-cond NLOS head accuracy and calibration. Predicted NLOS rate should track the ground-truth rate 0.02/0.15/0.40 per cond. |
| `adv.png` | Linear-probe env accuracy over Phase C episodes, chance line, disentanglement region shaded. The headline disentanglement figure. |
| `loss.png` | Six-panel loss curves for the joint trainer. Includes MAML meta_loss, disent task MSE, feat_loss, adv CE, cond CE, NLOS BCE, probe_env_acc, and the lambda schedule. |
| `loc.png` | **Headline figure.** Two-way per-cond location error: baseline vs joint Tier 1+2 on val envs. The baseline is the no-Tier-2 ASTA+LocationHead trained with the same data and seed (produced by `--run_baseline` in the joint trainer, 50 epochs to match the standalone Tier 2 budget). |

## Headline results (full-budget run, 2026-07-08)

See `loc.png` for the headline 2-way chart. The headline numbers
get printed to stdout when you run `viz_tier2_loc_comparison --joint`.

**Disentanglement metric** (probe c → env, val, 2 envs, chance=0.5):
- min=0.10  max=0.55  mean=0.248  fraction ≤ chance+0.05: 99%
- 99% of Phase C episodes have probe at-or-below chance — the
  disentanglement is structurally working.

## How to regenerate

```bash
cd code
python3.11 -m scripts.train_tier12_joint --out checkpoints/joint --run_baseline \
    --phase_a_episodes 100 --phase_a_ssl_epochs 50 \
    --phase_b_epochs 50 --phase_b_warmup 8 --phase_b_ramp 15 \
    --phase_c_episodes 100 --phase_c_warmup 15 --phase_c_ramp 30
python3.11 -m scripts.viz_joint_phases
python3.11 -m scripts.viz_joint_probe
python3.11 -m scripts.viz_tier2_c_space --joint
python3.11 -m scripts.viz_tier2_film --joint
python3.11 -m scripts.viz_tier2_nlos_reliability --joint
python3.11 -m scripts.viz_tier2_adv_accuracy --joint
python3.11 -m scripts.viz_tier2_loss_curves --joint
python3.11 -m scripts.viz_tier2_loc_comparison --joint
```
