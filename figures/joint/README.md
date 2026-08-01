# figures/joint — Joint Tier 1+2 visualization outputs

This directory holds figures produced by the joint Tier 1+2 training
pipeline (`code/scripts/train_tier12_joint.py`) and the joint-aware
viz scripts.

## Scripts that write here

By default, all of the following write to `figures/joint/` when run
with the `--joint` flag (or when the joint checkpoint is the default
in the script's argparser):

- `viz_tier2_c_space --joint` → `c_space.png`
- `viz_tier2_film --joint` → `film.png`
- `viz_tier2_nlos_reliability --joint` → `nlos.png`
- `viz_tier2_adv_accuracy --joint` → `adv.png`
- `viz_tier2_loss_curves --joint` → `loss.png`
- `viz_tier2_loc_comparison --joint` → `loc.png`
- `viz_tier2_loc_comparison --compare_all` → `loc_3way.png`
- `viz_joint_phases` → `all_phases.png` (renamed `phases.png` by default)
- `viz_joint_probe` → `probe_with_curriculum.png` (renamed `probe.png`)

## What each figure shows

| File | What it shows |
|---|---|
| `phases.png` | All three training phases' loss trajectories in one figure. Phase A is a placeholder (MAML losses live in `phase_A.pt`, not a CSV); Phase B is disentangle with frozen encoder; Phase C is joint FOMAML + disentangle with encoder unfrozen. Lambda curriculum overlaid on each phase. |
| `probe.png` | Disentanglement diagnostic on the joint trainer. Probe env accuracy on c, with the lambda curriculum on a side panel. Vertical dotted lines mark curriculum transitions. |
| `c_space.png` | t-SNE of c, colored by env_id (left, should look like a blob) and by micro condition (right, should show 3 clusters). |
| `film.png` | FiLM γ/β distributions + per-cond modulation profiles. Sanity check that the FiLM is doing something (γ ≠ 1.0, β ≠ 0.0). |
| `nlos.png` | Per-cond NLOS head accuracy and calibration. Predicted NLOS rate should track the ground-truth rate 0.02/0.15/0.40 per cond. |
| `adv.png` | Linear-probe env accuracy over training, chance line, disentanglement region shaded. The headline Tier 2 disentanglement figure. |
| `loss.png` | Six-panel loss curves for the joint trainer. Includes MAML meta_loss, disent task MSE, feat_loss, adv CE, cond CE, NLOS BCE, probe_env_acc, and the lambda schedule. |
| `loc.png` | Two-way per-cond location error: baseline vs joint Tier 1+2 on val envs. |
| `loc_3way.png` | Three-way per-cond location error: baseline vs standalone Tier 2 vs joint Tier 1+2. Requires both `tier2_per_cond.csv` and `phase_C_per_cond.csv`. |

## Current contents (smoke-test run, 2026-07-08)

All 8 PNGs above are from a 10-episode quick run (`--quick` flag).
The numbers are not yet meaningful — the joint trainer needs 50+
phase-C episodes to settle the curriculum. These figures are here
to confirm that the pipeline runs end-to-end and that all viz
scripts accept the joint checkpoint format.

For a real run:
```bash
cd code
python3.11 -m scripts.train_tier12_joint --out checkpoints/joint --run_baseline
python3.11 -m scripts.viz_joint_phases
python3.11 -m scripts.viz_joint_probe
python3.11 -m scripts.viz_tier2_c_space --joint
python3.11 -m scripts.viz_tier2_film --joint
python3.11 -m scripts.viz_tier2_nlos_reliability --joint
python3.11 -m scripts.viz_tier2_adv_accuracy --joint
python3.11 -m scripts.viz_tier2_loss_curves --joint
python3.11 -m scripts.viz_tier2_loc_comparison --joint
```

The 3-way comparison needs a standalone Tier 2 per-cond CSV too:
```bash
python3.11 -m scripts.train_tier2_synthetic --run_baseline --out checkpoints/tier2_synthetic
python3.11 -m scripts.viz_tier2_loc_comparison --compare_all
```
