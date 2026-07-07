# Trained model checkpoints

Small (.pt) artifacts from training runs. Each file contains:
- `model_state`: state dict for `SSLPretrainer` (encoder + head)
- `train_losses`, `val_losses`: per-epoch MSE lists
- `cfg`: the full `Config` used for training (mask_ratio, ssl_epochs, etc.)
- `csv`: path to the dataset the model was trained on

## Files

| File | Dataset | F | Mask ratio | Channel weights | Epochs | Final val MSE | Final val RMSE (ToA / AoA) |
|------|---------|--:|-----------:|------------------|-------:|--------------:|--------------------------:|
| `tier0_empty_open_25pct_100ep.pt` | `code/data/empty_open.csv` (Sionna, LOS only) | 1 (ToA) | 0.25 | (1.0,) | 100 | 0.033 | 0.181 m / — |
| `tier0_empty_open_15pct_100ep.pt` | `code/data/empty_open.csv` (Sionna, LOS only) | 1 (ToA) | 0.15 | (1.0,) | 100 | 0.050 | 0.224 m / — |
| `tier0_empty_open_f2_25pct_100ep.pt` | `code/data/empty_open.csv` (Sionna, LOS only) | 2 (ToA + AoA) | 0.25 | (1.0, 1.0)  | 100 | 0.0036 (z-norm) | 0.247 m / 6.32° |
| `tier0_empty_open_f2_25pct_3to1_100ep.pt` | `code/data/empty_open.csv` (Sionna, LOS only) | 2 (ToA + AoA) | 0.25 | (3.0, 1.0)  | 100 | 0.0090 (z-norm) | 0.211 m / 6.45° |
| `tier1_multi_scene_random_100ep.pt` | 6 scenes (`empty_open`, `custom_room_large`, `outdoor_blockages_pillars`, `indoor_blockages_pillars`, `outdoor_parking_01`, `indoor_garage_parking_01`) | 2 | 0.25 (Tier 0) / FOMAML | (3.0, 1.0) (Tier 0) | 100 FOMAML episodes | varies | varies (see Tier 1 few-shot table) |
| `tier1_multi_scene_tier0init_100ep.pt` | same 6 scenes | 2 | 0.25 (Tier 0) / FOMAML | (3.0, 1.0) (Tier 0) | 100 FOMAML episodes | varies | varies (see Tier 1 few-shot table) |

The Tier 1 checkpoints (FOMAML on multi-scene Sionna) take the
encoder from the corresponding Tier 0 checkpoint as initialization,
then run FOMAML outer-loop for 100 episodes over 4 train scenes.
The two variants differ in encoder init: **random** vs
**Tier 0 pretrained** (transferred from
`tier0_empty_open_f2_25pct_3to1_100ep.pt`). Few-shot evaluation is
on the 2 held-out scenes (outdoor_parking_01, indoor_garage_parking_01).
See `figures/tier1_sionna/compare_tier1_few_shot.{png,csv}` for the
K-shot curve.

The 3:1 weighting (ToA upweighted 3× over AoA) reduces ToA RMSE by
~15% (0.247 → 0.211 m) at the cost of a small (0.13°) increase in AoA
RMSE — a worthwhile trade because ToA is the geometrically meaningful
channel that flows into the downstream location head.

The F=2 checkpoints also store `norm_mean` and `norm_std` (per-channel
z-score stats computed from the training grids) so the viz scripts
can denormalize predictions back to original units. The mask is JOINT
across channels: if (t, m) is masked, both ToA and AoA are zeroed at
that cell, forcing the model to use cross-anchor context.

## Loading a checkpoint

```python
import torch
from ultraclear.config import Config
from ultraclear.tier0_ssl import SSLPretrainer

ckpt = torch.load("tier0_empty_open_25pct_100ep.pt", weights_only=False)
cfg = Config(**{k: v for k, v in ckpt["cfg"].items() if k in Config.__dataclass_fields__})
model = SSLPretrainer(cfg)
model.load_state_dict(ckpt["model_state"])
# use model.encoder for downstream tiers (Tier 1 FOMAML, Tier 3 heads)
```

## Reproducing the runs

```bash
cd code
# 25% mask, 100 epochs
python3.11 -m scripts.train_tier0_sionna \
    --csv data/empty_open.csv \
    --epochs 100 --mask_ratio 0.25 \
    --out ../checkpoints/tier0_empty_open_25pct_100ep.pt

# 15% mask, 100 epochs
python3.11 -m scripts.train_tier0_sionna \
    --csv data/empty_open.csv \
    --epochs 100 --mask_ratio 0.15 \
    --out ../checkpoints/tier0_empty_open_15pct_100ep.pt
```

## What these checkpoints are good for

- Loading the pretrained ASTA encoder for downstream tiers (FOMAML, location head, etc.)
- Regenerating any of the figures in `figures/sionna_ssl/` via the `viz_tier0_*` and `compare_mask_ratios` scripts
- Comparing different mask ratios as a hyperparameter (the 25% run has a lower training loss, but the per-masked-cell RMSE is essentially identical to 15%)

## Known limitation (see also MEMORY / paper)

The SSL encoder converges to a "predict approximately the per-(window, anchor)-mean" solution
rather than one that tracks true deviations. Per-masked-cell RMSE on val is ~0.30 m for both
runs, vs. a trivial "average visible same-anchor cells" baseline of ~0.05 m. A harder mask
pattern (mask an entire anchor at all T trials) is the next experiment.
