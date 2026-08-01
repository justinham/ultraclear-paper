"""Eval baseline_w01_50ep.pt: error reduction per cell.
Compare |measured - GT| vs |restored - GT| for the actual held-out val (envs 10..20\{14}).
"""
import sys, time
import numpy as np
import torch
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path("/Users/justin/ultraclear-paper/code")))

from ultraclear.config import Config
from ultraclear.tier3_heads import DownstreamModel
from scripts.sionna_joint_dataset import build_sionna_joint_dataset
from scripts.train_tier12_joint import TensorUWB

CKPT = "checkpoints/baseline_w01_50ep.pt"
CSV_DIR = "/Users/justin/Downloads/srp_research/grid2"
DEVICE = "cpu"

TRAIN_ENVS = set(range(10))
VAL_ENVS = set(range(10, 21)) - {14}

ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
cfg = Config(**{k: v for k, v in ck['cfg'].items() if k in Config.__dataclass_fields__})
cfg.device = DEVICE

csv_paths = sorted(str(p) for p in Path(CSV_DIR).glob("*_A1A4swapped.csv"))
ds, scene_info = build_sionna_joint_dataset(
    csv_paths=csv_paths, n_channels=cfg.n_channels, seq_len=cfg.seq_len, stride=cfg.seq_len,
)
ds.compute_norm_stats(TRAIN_ENVS)
# Manually iterate like sionna_to_items_with_restore but actually use VAL_ENVS
val_indices = []
for i in range(len(ds)):
    x, y, env, _c, nlos, restore = ds.getitem_with_restore(i)
    if int(env.item()) in VAL_ENVS:
        val_indices.append(i)
print(f"val_indices: {len(val_indices)}")

# Build a per-cell GT lookup
print("Loading raw CSVs for ground truth...")
gt_by_env = {}
measured_by_env = {}
flag_by_env = {}
for info in ds.scene_info:
    name = info.name
    csv_path = f"{CSV_DIR}/sionna_uwb_grid_{name}_A1A4swapped.csv"
    df = pd.read_csv(csv_path)
    n_grids = 400
    n_trials = 100
    gt = np.stack([df[f"A{m}_gt_dist_m"].values for m in range(1, 5)], axis=1).reshape(n_grids, n_trials, 4).astype(np.float32)
    meas = np.stack([df[f"A{m}_measured_dist_m"].values for m in range(1, 5)], axis=1).reshape(n_grids, n_trials, 4).astype(np.float32)
    flag = np.stack([df[f"A{m}_los_nlos"].values for m in range(1, 5)], axis=1).reshape(n_grids, n_trials, 4).astype(np.float32)
    nan_mask = np.isnan(meas).any(axis=2)
    flag[nan_mask] = 0.0
    meas[nan_mask] = 0.0
    gt[nan_mask] = 0.0
    gt_by_env[info.env_id] = gt
    measured_by_env[info.env_id] = meas
    flag_by_env[info.env_id] = flag
print(f"  loaded {len(gt_by_env)} envs")

# Subset of val items into a TensorUWB-like list
# Easiest: use a Subset of the dataset
from torch.utils.data import Subset
val_ds_full = SionnaJointWithRaw(ds) if False else None
# Instead: build a custom dataset wrapping the indices
class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        return self.base.getitem_with_restore(self.indices[i])

val_subset = IndexedSubset(ds, val_indices)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)

model = DownstreamModel(cfg).to(DEVICE)
model.load_state_dict(ck['model_state'])
model.eval()

norm_mean = torch.from_numpy(ds.norm_mean).to(DEVICE)
norm_std = torch.from_numpy(ds.norm_std).to(DEVICE)

per_env = {}
t0 = time.time()
with torch.no_grad():
    for batch_idx, (X, y, env, _c, nlos, restore) in enumerate(val_loader):
        X = X.to(DEVICE); env = env.to(DEVICE)
        out = model(X)
        restored = out['restored']  # (B, T, M, F)
        item_start = batch_idx * val_loader.batch_size
        for b in range(X.size(0)):
            item_i = val_indices[item_start + b]
            s_idx, grid, s = ds.windows[item_i]
            eid = ds.scene_info[s_idx].env_id
            gt_w = gt_by_env[eid][grid, s:s+cfg.seq_len, :]
            meas_w = measured_by_env[eid][grid, s:s+cfg.seq_len, :]
            flag_w = flag_by_env[eid][grid, s:s+cfg.seq_len, :]
            rest_w = restored[b, :, :, 0].cpu().numpy() * norm_std[0].item() + norm_mean[0].item()
            valid = meas_w > 0.01
            meas_err = np.abs(meas_w[valid] - gt_w[valid])
            rest_err = np.abs(rest_w[valid] - gt_w[valid])
            los = (flag_w[valid] == 1.0)
            nlos = ~los
            per_env.setdefault(eid, {"los_meas": [], "los_rest": [], "nlos_meas": [], "nlos_rest": []})
            per_env[eid]["los_meas"].append(meas_err[los])
            per_env[eid]["los_rest"].append(rest_err[los])
            per_env[eid]["nlos_meas"].append(meas_err[nlos])
            per_env[eid]["nlos_rest"].append(rest_err[nlos])

print(f"eval took {time.time()-t0:.1f}s")

# Aggregate
print("\n" + "="*100)
print("ERROR REDUCTION: per-cell |measured - GT| vs |restored - GT|")
print("="*100)
print(f"  {'env':>3}  {'name':<35}  {'cls':<5}  {'n_cells':>8}  {'meas_MAE':>10}  {'rest_MAE':>10}  {'Δ (m)':>10}  {'%impr':>8}")
totals = {"los_meas": [], "los_rest": [], "nlos_meas": [], "nlos_rest": []}
for eid in sorted(per_env.keys()):
    name = next((s.name for s in ds.scene_info if s.env_id == eid), "?")
    for cls, mk, rk in [("LOS", "los_meas", "los_rest"), ("NLOS", "nlos_meas", "nlos_rest")]:
        m = np.concatenate(per_env[eid][mk])
        r = np.concatenate(per_env[eid][rk])
        if m.size == 0:
            print(f"  {eid:>3}  {name:<35}  {cls:<5}  (empty)")
            continue
        mae_m = m.mean()
        mae_r = r.mean()
        delta = mae_m - mae_r
        pct = 100.0 * delta / mae_m if mae_m > 1e-9 else 0.0
        print(f"  {eid:>3}  {name:<35}  {cls:<5}  {m.size:>8}  {mae_m:>10.4f}  {mae_r:>10.4f}  {delta:>+10.4f}  {pct:>+7.1f}%")
        totals[mk].append(m)
        totals[rk].append(r)

print()
print("="*100)
print("OVERALL (MAE view)")
print("="*100)
for label, mk, rk in [
    ("ALL",  "los", "los"),
    ("LOS",  "los", "los"),
    ("NLOS", "nlos", "nlos"),
]:
    pass
all_meas_los = np.concatenate(totals["los_meas"])
all_rest_los = np.concatenate(totals["los_rest"])
all_meas_nlos = np.concatenate(totals["nlos_meas"])
all_rest_nlos = np.concatenate(totals["nlos_rest"])
for label, m, r in [
    ("ALL",  np.concatenate([all_meas_los, all_meas_nlos]), np.concatenate([all_rest_los, all_rest_nlos])),
    ("LOS",  all_meas_los, all_rest_los),
    ("NLOS", all_meas_nlos, all_rest_nlos),
]:
    mae_m = m.mean()
    mae_r = r.mean()
    delta = mae_m - mae_r
    pct = 100.0 * delta / mae_m if mae_m > 1e-9 else 0.0
    print(f"  {label:<5}  n={m.size:>8}  meas MAE={mae_m:.4f}  rest MAE={mae_r:.4f}  Δ={delta:+.4f}  improvement={pct:+.1f}%")

print()
print("="*100)
print("OVERALL (RMSE view)")
print("="*100)
for label, m, r in [
    ("ALL",  np.concatenate([all_meas_los, all_meas_nlos]), np.concatenate([all_rest_los, all_rest_nlos])),
    ("LOS",  all_meas_los, all_rest_los),
    ("NLOS", all_meas_nlos, all_rest_nlos),
]:
    rmse_m = float(np.sqrt((m**2).mean()))
    rmse_r = float(np.sqrt((r**2).mean()))
    delta = rmse_m - rmse_r
    pct = 100.0 * delta / rmse_m if rmse_m > 1e-9 else 0.0
    print(f"  {label:<5}  n={m.size:>8}  meas RMSE={rmse_m:.4f}  rest RMSE={rmse_r:.4f}  Δ={delta:+.4f}  improvement={pct:+.1f}%")
