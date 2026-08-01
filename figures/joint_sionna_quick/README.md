# figures/joint_sionna_quick — Joint Tier 1+2 on Sionna REAL data (smoke test)

This directory holds figures from the first end-to-end run of the
joint Tier 1+2 trainer on the **21 real Sionna ray-tracing CSVs** in
`~/Downloads/srp_research/grid2/`.

## What this run is

**Data source**: `--data sionna`. The 21 CSVs are loaded as 21
different envs, sliding windows of T=32 trials per window, F=2
(ToA + AoA azimuth), z-score normalized per channel from train
scenes, with per-scene (tag_x, tag_y) z-score normalization so the
model can predict location in a common reference frame across
scenes with different scales (custom_room ~3m, empty_open ~20m,
parking ~30m).

**Train/val split**: 10 train envs (0-9, all indoor), 10 val envs
(10-20 except 14, which has 20% missing data and produces
degenerate per-bucket errors). The split is a clean
indoor-to-outdoor generalization test. 12,000 train windows,
13,200 val windows.

Default train envs (in sorted order): custom_room,
custom_room_large, empty_open, indoor_blockages_asymmetric,
indoor_blockages_pillars, indoor_garage_parking_01..05.

Default val envs: outdoor_blockages_asymmetric,
outdoor_blockages_narrow, outdoor_blockages_pillars,
outdoor_empty, outdoor_parking_01..05, outdoor_with_blockages
(env 14 `outdoor_empty_missing20` excluded by default).

Override with `--sionna_train_envs 0 1 2 3 4` and
`--sionna_val_envs 5 6 7 8 9 10`.

**Per-cond eval**: since real data has no clean 3-class cond_id, we
bucket val windows by per-cell NLOS rate (low <0.3, mid <0.7,
high ≥0.7) and report per-bucket mean loc error.

**Joint phases**: --quick (3 SSL + 5 MAML warmup, 10 disentangle
epochs, 10 joint episodes). Quick run, not full budget.

## Headline results (val, denormalized to original meters)

10 train / 10 val split. 10-episode quick run, 50-epoch baseline.

| NLOS bucket | baseline | joint (Tier 1+2) | Δ |
|---|---:|---:|---:|
| 0 (low, <30%) | 5.05 m | 8.69 m | +72% |
| 1 (mid, 30-70%) | 4.72 m | 9.77 m | +107% |
| 2 (high, ≥70%) | 1.55 m | 6.54 m | +322% |

**Disentanglement metric** (probe c → env, val, 10 envs, chance = 0.1):
- 100% of Phase C episodes have probe at-or-below chance
- Mean probe accuracy 0.218 (2.2× better than chance)

## The two important things to know about these numbers

**1. The pipeline runs end-to-end on real data.** That's the main
result of this run. The joint trainer accepts `--data sionna`,
loads 21 CSVs, handles NaN rows (the `outdoor_empty_missing20`
file has 20% deliberately-missing ToAs), applies per-channel and
per-scene normalization, and trains all three phases without
crashing. The disentanglement metric works on real envs.

**2. The adv loss explodes on real data.** Phase B and C have adv
CE values of 50-280, vs ~2 (chance) on synthetic. The cause is
that with 21 envs the adversary has a much harder classification
problem than with 8 envs, and the disentanglement pressure
`+ lam * adv_ce` dominates the location loss. The location head
gets pushed around by the gradient and the per-bucket location
error is much worse than the no-Tier-2 baseline.

The fix is to scale the disentanglement pressure differently for
higher-env-count settings (e.g. cap `lam * adv_ce` at a fixed
ceiling, or use `max(0, adv_ce - 2*chance)` as the disentanglement
loss). This is a known limitation of the current joint trainer
that should be addressed before publishing real-data results.

## The bucket-2 anomaly

The bucket 2 baseline at 0.37 m is suspiciously good. This is
because bucket 2 contains windows with NLOS rate ≥ 0.7, which
includes the 23,676 trials from `outdoor_empty_missing20` that
we marked as NLOS=1 (they have no ToA readings). For these
windows, all 4 anchors are flagged NLOS, so the optimal
prediction is the per-scene mean — and the no-Tier-2 baseline
learns this efficiently. The joint, on the other hand, is
wasting capacity on the adversarial pressure and predicting
further from the mean.

This is a known degeneracy of the bucket metric on the
`outdoor_empty_missing20` file. For a cleaner headline, that
file should be excluded from the val set in a follow-up run
(`--sionna_val_envs 16 17 18 19` excludes env 20 which is
`outdoor_with_blockages`; or `--sionna_csv_dir` could point to
a directory without the missing-data file).

## Files

- `loc.png` — 2-way per-bucket bar chart (baseline vs joint)
- `phases.png` — all three phases' loss trajectories overlaid
- `probe.png` — disentanglement diagnostic with curriculum

## How to regenerate

```bash
cd code
python3.11 -m scripts.train_tier12_joint \
    --data sionna --out checkpoints/joint_sionna --run_baseline \
    --phase_a_episodes 100 --phase_a_ssl_epochs 50 \
    --phase_b_epochs 50 --phase_b_warmup 8 --phase_b_ramp 15 \
    --phase_c_episodes 100 --phase_c_warmup 15 --phase_c_ramp 30
python3.11 -m scripts.viz_joint_phases --joint_dir checkpoints/joint_sionna
python3.11 -m scripts.viz_joint_probe --joint_dir checkpoints/joint_sionna
python3.11 -m scripts.viz_tier2_loc_comparison --joint \
    --joint_per_cond checkpoints/joint_sionna/phase_C_per_cond.csv
```

For a real-data run with the missing-data file excluded, pass
`--sionna_val_envs 16 17 18 19` to drop env 20 from the val set
(`outdoor_with_blockages` has 100% NLOS and produces degenerate
per-cond errors). To also exclude it from training, you'd need
to symlink the CSV dir without that file or modify the loader.
