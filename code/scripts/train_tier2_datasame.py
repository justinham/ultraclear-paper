"""Tier 2 (heads-only) on the 10 data_same scenarios, leave-scenes-out.

All 10 files share one bit-identical 400-point tag grid, so the only thing that
varies between them is the propagation channel. That makes leave-SCENES-out the
right generalization test: train heads on 8 scenes, test on 2 unseen scenes.
Splits on locations instead would leak, because every location exists in every
scene.

Heads: NLOS + Location only (the restoration head cost 40% localization
accuracy in the earlier grid2 experiments and is left out).

Reports:
  - held-out-scene error (the headline)
  - seen-scene error, for the generalization gap
  - error stratified by NLOS level of the query window
  - NLOS head precision / recall / F1 (uses the corrected `flag < 0.5` target)

Usage:
    python3.11 -u -m scripts.train_tier2_datasame --quick
    python3.11 -u -m scripts.train_tier2_datasame --epochs 300
    python3.11 -u -m scripts.train_tier2_datasame --ckpt <path>
"""
from __future__ import annotations
import argparse
import csv as csvmod
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ultraclear.config import Config
from ultraclear.tier3_heads import DownstreamModel
from scripts.train_tier2_empty_space import load_encoder_from_ssl

DATA_DIR = Path('/Users/justin/Downloads/srp_research/data_same')
DEFAULT_CKPT = Path('/Users/justin/ultraclear-paper/code/checkpoints/'
                    'tier0_crossscene_deep_500ep/tier0_crossscene_contrast.pt')
ANCHORS_XY = np.array([[-0.524, -0.893],
                       [-0.550,  0.345],
                       [ 0.519, -0.894],
                       [ 0.537,  0.345]])
CENTROID = ANCHORS_XY.mean(0)


def discover(data_dir=DATA_DIR):
    hits = sorted(glob.glob(str(Path(data_dir) /
                               'sionna_uwb_grid_*_A1A4swapped.csv')))
    return [(Path(h).name.replace('sionna_uwb_grid_', '')
             .replace('_A1A4swapped.csv', ''), Path(h)) for h in hits]


class TenSceneDataset(Dataset):
    """Windows across all scenes, indexed by (scene, grid, start)."""

    def __init__(self, scene_csvs, seq_len=32, stride=32):
        self.seq_len = seq_len
        self.scene_names = [n for n, _ in scene_csvs]
        self.scenes, self.windows = [], []
        for s_idx, (name, path) in enumerate(scene_csvs):
            df = pd.read_csv(path)
            per_grid = []
            for gid in sorted(df.grid_id.unique()):
                sub = df[df.grid_id == gid].sort_values('trial_id')
                if len(sub) < seq_len:
                    continue
                toa = np.stack([sub[f"A{a}_measured_dist_m"].values
                                for a in (1, 2, 3, 4)], axis=1
                               ).astype(np.float32)[:, :, None]
                # NLOS target: flag==1 means LOS, so NLOS is flag < 0.5
                nlos = np.stack([(sub[f"A{a}_los_nlos"].to_numpy(float) < 0.5
                                  ).astype(np.float32)
                                 for a in (1, 2, 3, 4)], axis=1)
                per_grid.append({
                    'toa': toa, 'nlos': nlos, 'n': len(sub),
                    'gt': np.array([float(sub.tag_x.iloc[0]),
                                    float(sub.tag_y.iloc[0])], np.float32),
                })
            self.scenes.append({'name': name, 'grids': per_grid})
            for gi, g in enumerate(per_grid):
                for st in range(0, g['n'] - seq_len + 1, stride):
                    self.windows.append((s_idx, gi, st))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        si, gi, st = self.windows[i]
        g = self.scenes[si]['grids'][gi]
        sl = slice(st, st + self.seq_len)
        return (torch.from_numpy(g['toa'][sl]),
                torch.from_numpy(g['nlos'][sl]),
                torch.from_numpy(g['gt']),
                torch.tensor(si))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(DEFAULT_CKPT))
    ap.add_argument('--data_dir', default=str(DATA_DIR))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lambda_loc', type=float, default=1.0)
    ap.add_argument('--lambda_nlos', type=float, default=0.5)
    ap.add_argument('--n_held_out', type=int, default=2,
                    help='scenes held out for testing')
    ap.add_argument('--held_scenes', type=str, default=None,
                    help='comma-separated scene names to hold out, e.g. '
                         '"indoor_garage_parking_03,outdoor_parking_04". '
                         'Overrides the random split.')
    ap.add_argument('--split_mode', type=str, default='random',
                    choices=['random', 'stratified'],
                    help='random = seeded permutation of all scenes '
                         '(legacy); stratified = hold out one indoor and '
                         'one outdoor scene, balanced across the pool')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = Config()
    cfg.device = args.device
    cfg.seed = args.seed
    torch.manual_seed(cfg.seed)
    np.random.seed(args.seed)
    epochs = 3 if args.quick else args.epochs

    scene_csvs = discover(args.data_dir)
    print(f"ckpt:  {Path(args.ckpt).parent.name}")
    print(f"data:  {args.data_dir}  ({len(scene_csvs)} scenes)")
    print(f"device={cfg.device} epochs={epochs} seed={cfg.seed}")

    ds = TenSceneDataset(scene_csvs, seq_len=cfg.seq_len)
    print(f"windows: {len(ds)}   locations/scene: {len(ds.scenes[0]['grids'])}")

    # ---- leave-scenes-out split ----
    # Three ways to choose the held-out scenes:
    #   1. --held_scenes  : explicit names (used by the rotation sweep)
    #   2. --split_mode stratified : one indoor + one outdoor, drawn with
    #      the seeded RNG so the sweep is reproducible
    #   3. --split_mode random (legacy) : seeded permutation of all
    #      scenes, first n_held_out taken
    scene_names_all = [s['name'] for s in ds.scenes]
    indoor_idx = [i for i, n in enumerate(scene_names_all)
                  if n.startswith('indoor')]
    outdoor_idx = [i for i, n in enumerate(scene_names_all)
                   if n.startswith('outdoor')]

    if args.held_scenes:
        want = [s.strip() for s in args.held_scenes.split(',') if s.strip()]
        missing = [w for w in want if w not in scene_names_all]
        if missing:
            raise SystemExit(f'unknown scene name(s): {missing}\n'
                             f'available: {scene_names_all}')
        held = sorted(scene_names_all.index(w) for w in want)
        print(f"\nsplit: EXPLICIT held-out scenes {want}")
    elif args.split_mode == 'stratified':
        rng = np.random.RandomState(cfg.seed)
        held = sorted([int(rng.choice(indoor_idx)),
                       int(rng.choice(outdoor_idx))])
        print(f"\nsplit: STRATIFIED (1 indoor + 1 outdoor, seed {cfg.seed})")
    else:
        rng = np.random.RandomState(cfg.seed)
        order = rng.permutation(len(ds.scenes))
        held = sorted(order[:args.n_held_out].tolist())
        print(f"\nsplit: RANDOM permutation of all scenes "
              f"(seed {cfg.seed}, n_held_out={args.n_held_out})")

    seen = [i for i in range(len(ds.scenes)) if i not in held]
    print(f"TRAIN scenes ({len(seen)}): "
          f"{[ds.scenes[i]['name'] for i in seen]}")
    print(f"HELD-OUT scenes ({len(held)}): "
          f"{[ds.scenes[i]['name'] for i in held]}")

    tr_idx = [k for k, (s, _, _) in enumerate(ds.windows) if s not in held]
    print(f"train windows: {len(tr_idx)}")

    # ---- norm stats from train scenes ----
    chunks = []
    for si in seen:
        for g in ds.scenes[si]['grids']:
            chunks.append(g['toa'].reshape(-1, cfg.n_channels))
    cat = np.concatenate(chunks, 0)
    mu = np.nanmean(cat, 0).astype(np.float32)
    sd_ = (np.nanstd(cat, 0) + 1e-6).astype(np.float32)
    print(f"norm mean={mu} std={sd_}")

    mu_t = torch.from_numpy(mu)
    sd_t = torch.from_numpy(sd_)

    def collate(b):
        return (torch.stack([x[0] for x in b]),
                torch.stack([x[1] for x in b]),
                torch.stack([x[2] for x in b]),
                torch.stack([x[3] for x in b]))

    loader = DataLoader(torch.utils.data.Subset(ds, tr_idx),
                        batch_size=cfg.ssl_batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0)

    def nz(X):
        return torch.nan_to_num((X - mu_t) / sd_t, nan=0.0,
                                posinf=0.0, neginf=0.0)

    # ---- model: frozen encoder + NLOS + Location heads ----
    encoder, _ = load_encoder_from_ssl(Path(args.ckpt), cfg)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    model = DownstreamModel(cfg)
    model.encoder = encoder
    print(f"\nencoder FROZEN from {Path(args.ckpt).parent.name}")
    print("heads: NLOS + Location (no restoration)")

    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith('encoder.') and p.requires_grad]
    print(f"training {sum(p.numel() for p in head_params):,} head params")
    opt = torch.optim.Adam(head_params, lr=args.lr)

    bce = torch.nn.BCEWithLogitsLoss()
    sl1 = torch.nn.SmoothL1Loss()

    t0 = time.time()
    hist = []
    for ep in range(epochs):
        model.train()
        model.encoder.eval()
        tot, n = 0.0, 0
        for X, NL, Y, _S in loader:
            X, NL, Y = nz(X), NL, Y
            out = model(X)
            loss = args.lambda_loc * sl1(out['loc'], Y) + \
                   args.lambda_nlos * bce(out['nlos_logits'], NL)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * X.size(0)
            n += X.size(0)
        hist.append((ep + 1, tot / max(1, n)))
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            print(f"  ep {ep+1:4d}/{epochs}  train={tot/max(1,n):.4f}  "
                  f"({time.time()-t0:.1f}s)")
    print(f"trained in {time.time()-t0:.1f}s")

    # ---- eval: held-out AND seen scenes, stratified by NLOS ----
    model.eval()

    def evaluate(scene_ids, label):
        idx = [k for k, (s, _, _) in enumerate(ds.windows)
               if s in scene_ids]
        dl = DataLoader(torch.utils.data.Subset(ds, idx),
                        batch_size=cfg.ssl_batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
        P, G, S, NLQ, LQ = [], [], [], [], []
        with torch.no_grad():
            for X, NL, Y, SC in dl:
                X = nz(X)
                out = model(X)
                P.append(out['loc'].numpy())
                G.append(Y.numpy())
                S.append(SC.numpy())
                NLQ.append(out['nlos_logits'].numpy())
                LQ.append(NL.numpy())
        P = np.concatenate(P); G = np.concatenate(G)
        S = np.concatenate(S)
        NLQ = np.concatenate(NLQ); LQ = np.concatenate(LQ)
        E = np.sqrt(((P - G) ** 2).sum(-1))
        print(f"\n=== {label} ===")
        print(f"  windows: {len(E)}")
        print(f"  rmse={np.sqrt((E**2).mean()):.4f}  "
              f"med={np.median(E):.4f}  "
              f"p90={np.percentile(E,90):.4f}  "
              f"p95={np.percentile(E,95):.4f}  "
              f"max={E.max():.4f}")
        # per-scene
        for si in scene_ids:
            m = S == si
            if m.sum() == 0:
                continue
            e = E[m]
            print(f"    [{ds.scenes[si]['name']:26s}] rmse={np.sqrt((e**2).mean()):.4f}"
                  f"  med={np.median(e):.4f}  p90={np.percentile(e,90):.4f}"
                  f"  max={e.max():.4f}")
        # by range
        R = np.linalg.norm(G - CENTROID, axis=1)
        print("  by range from anchor centroid:")
        for lo, hi in ((0, 5), (5, 10), (10, 14), (14, 25)):
            m = (R >= lo) & (R < hi)
            if m.sum() < 5:
                continue
            e = E[m]
            print(f"    {lo:2d}-{hi:2d} m  n={m.sum():5d}  "
                  f"med={np.median(e):.4f}  p90={np.percentile(e,90):.4f}")
        # by NLOS in query. LQ is ALREADY the NLOS indicator (1 = NLOS),
        # so use it directly -- applying `< 0.5` again would invert it.
        q_nlos = LQ.mean(axis=(1, 2))
        print("  by NLOS fraction in query window:")
        for lo, hi, lab in ((0, .01, 'LOS only'), (.01, .3, 'some'),
                            (.3, 1.01, 'heavy')):
            m = (q_nlos >= lo) & (q_nlos < hi)
            if m.sum() < 5:
                continue
            e = E[m]
            print(f"    {lab:9s} n={m.sum():5d}  "
                  f"med={np.median(e):.4f}  p90={np.percentile(e,90):.4f}")
        # NLOS head quality. Same care: LQ is already the NLOS indicator.
        yhat = (NLQ > 0).astype(np.float32)
        ytrue = LQ.astype(np.float32)
        tp = float((yhat * ytrue).sum()); fp = float((yhat * (1 - ytrue)).sum())
        fn = float(((1 - yhat) * ytrue).sum()); tn = float(((1 - yhat) * (1 - ytrue)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"  NLOS head: precision={prec:.4f} recall={rec:.4f} F1={f1:.4f}"
              f"  (base rate NLOS={ytrue.mean()*100:.2f}%)")
        return dict(E=E, P=P, G=G, S=S, prec=prec, rec=rec, f1=f1)

    held_r = evaluate(set(held), "HELD-OUT SCENES")
    seen_r = evaluate(set(seen), "TRAIN (seen) SCENES")

    gap = (np.sqrt((held_r['E'] ** 2).mean()) -
           np.sqrt((seen_r['E'] ** 2).mean()))
    print(f"\n=== generalization gap (rmse) ===")
    print(f"  seen {np.sqrt((seen_r['E']**2).mean()):.4f}  ->  "
          f"held-out {np.sqrt((held_r['E']**2).mean()):.4f}   "
          f"({gap:+.4f} m)")

    # ---- save ----
    out_dir = Path(args.out or
                   f"checkpoints/tier2_datasame_{Path(args.ckpt).parent.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'history': hist,
        'cfg': cfg.__dict__,
        'norm_mean': mu, 'norm_std': sd_,
        'ssl_ckpt': str(args.ckpt),
        'train_scenes': [ds.scenes[i]['name'] for i in seen],
        'held_out_scenes': [ds.scenes[i]['name'] for i in held],
        'held': dict(rmse=float(np.sqrt((held_r['E'] ** 2).mean())),
                     median=float(np.median(held_r['E'])),
                     p90=float(np.percentile(held_r['E'], 90)),
                     max=float(held_r['E'].max()),
                     nlos_f1=held_r['f1']),
        'seen': dict(rmse=float(np.sqrt((seen_r['E'] ** 2).mean())),
                     median=float(np.median(seen_r['E'])),
                     p90=float(np.percentile(seen_r['E'], 90)),
                     max=float(seen_r['E'].max()),
                     nlos_f1=seen_r['f1']),
    }, out_dir / 'tier2_datasame.pt')
    print(f"\nsaved -> {out_dir / 'tier2_datasame.pt'}")

    with open(out_dir / 'history.csv', 'w', newline='') as f:
        w = csvmod.writer(f)
        w.writerow(['epoch', 'train_loss'])
        for e, l in hist:
            w.writerow([e, f'{l:.6f}'])


if __name__ == '__main__':
    main()