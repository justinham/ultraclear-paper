"""Two-head downstream model on the frozen cross-scene SSL encoder.

    frozen encoder  ->  NLOS head      (per-anchor NLOS probability)
                     ->  Location head  (2D position)

Clean-up vs `DownstreamModel`:
  - `restore_head` is REMOVED entirely, not merely untrained
  - `loc_head` reads the ENCODER FEATURES directly, instead of consuming the
    restoration head's output. Previously the location head took the middle
    timestep of `restored`, so once the restoration head stopped training the
    position head was reading random-init output.

Heads share the frozen trunk but have independent parameters and independent
losses:
    L = lambda_loc * SmoothL1(loc) + lambda_nlos * BCE(nlos)

Usage:
    python3.11 -u -m scripts.train_tier2_datasame_twohead --quick
    python3.11 -u -m scripts.train_tier2_datasame_twohead --epochs 300
"""
from __future__ import annotations
import argparse
import csv as csvmod
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ultraclear.config import Config
from ultraclear.asta import ASTAEncoder
from ultraclear.tier3_heads import NLOSHead
from scripts.train_tier2_datasame import TenSceneDataset, discover, DATA_DIR
from scripts.train_tier2_empty_space import load_encoder_from_ssl

DEFAULT_CKPT = Path('/Users/justin/ultraclear-paper/code/checkpoints/'
                    'tier0_crossscene_deep_500ep/tier0_crossscene_contrast.pt')
ANCHORS_XY = np.array([[-0.524, -0.893],
                       [-0.550,  0.345],
                       [ 0.519, -0.894],
                       [ 0.537,  0.345]])
CENTROID = ANCHORS_XY.mean(0)


class LocationHeadFromEncoder(nn.Module):
    """2D position straight from pooled encoder features.

    h: (B, T, M, d) -> mean-pool over (T, M) -> MLP -> (B, 2).

    Unlike `LocationHeadFromRestore`, this does not depend on any other head,
    so it trains from step 1 instead of reading a randomly-initialised
    reconstruction head.
    """

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = h.mean(dim=(1, 2))            # (B, d)
        return self.net(self.norm(z))      # (B, 2)


class TwoHeadModel(nn.Module):
    """Frozen encoder + NLOS head + Location head. No restoration head."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = ASTAEncoder(cfg)
        self.nlos_head = NLOSHead(cfg.d_model)
        self.loc_head = LocationHeadFromEncoder(cfg.d_model)

    def forward(self, x):
        h = self.encoder(x)                # (B, T, M, d)
        return {
            'nlos_logits': self.nlos_head(h),   # (B, T, M)
            'loc': self.loc_head(h),            # (B, 2)
        }


def build(ckpt, cfg):
    encoder, _ = load_encoder_from_ssl(Path(ckpt), cfg)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    m = TwoHeadModel(cfg)
    m.encoder = encoder
    return m


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
    ap.add_argument('--n_held_out', type=int, default=2)
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    epochs = 3 if args.quick else args.epochs

    print(f"ckpt:  {Path(args.ckpt).parent.name}")
    print(f"data:  {args.data_dir}")
    print(f"epochs={epochs} seed={cfg.seed} lr={args.lr}")
    print(f"loss: lambda_loc={args.lambda_loc} lambda_nlos={args.lambda_nlos}")

    scene_csvs = discover(args.data_dir)
    ds = TenSceneDataset(scene_csvs, seq_len=cfg.seq_len)
    print(f"{len(ds.scenes)} scenes, {len(ds)} windows")

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
        print(f"split: EXPLICIT held-out scenes {want}")
    elif args.split_mode == 'stratified':
        rng = np.random.RandomState(cfg.seed)
        held = sorted([int(rng.choice(indoor_idx)),
                       int(rng.choice(outdoor_idx))])
        print(f"split: STRATIFIED (1 indoor + 1 outdoor, seed {cfg.seed})")
    else:
        rng = np.random.RandomState(cfg.seed)
        order = rng.permutation(len(ds.scenes))
        held = sorted(order[:args.n_held_out].tolist())
        print(f"split: RANDOM permutation of all scenes "
              f"(seed {cfg.seed}, n_held_out={args.n_held_out})")

    seen = [i for i in range(len(ds.scenes)) if i not in held]
    print(f"TRAIN  ({len(seen)}): {[ds.scenes[i]['name'] for i in seen]}")
    print(f"HELD   ({len(held)}): {[ds.scenes[i]['name'] for i in held]}")

    tr_idx = [k for k, (s, _, _) in enumerate(ds.windows) if s not in held]

    chunks = []
    for si in seen:
        for g in ds.scenes[si]['grids']:
            chunks.append(g['toa'].reshape(-1, cfg.n_channels))
    cat = np.concatenate(chunks, 0)
    mu = np.nanmean(cat, 0).astype(np.float32)
    sd_ = (np.nanstd(cat, 0) + 1e-6).astype(np.float32)
    mu_t, sd_t = torch.from_numpy(mu), torch.from_numpy(sd_)
    print(f"norm mean={mu} std={sd_}")

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

    model = build(args.ckpt, cfg)
    groups = {}
    for k, v in model.state_dict().items():
        g = k.split('.')[0]
        groups[g] = groups.get(g, 0) + v.numel()
    print("\n=== model ===")
    for mod, n in groups.items():
        tag = 'FROZEN' if mod == 'encoder' else 'trained'
        print(f"  {mod:12s} {n:>8,}  ({tag})")
    head_params = [p for n_, p in model.named_parameters()
                   if not n_.startswith('encoder.') and p.requires_grad]
    n_head = sum(p.numel() for p in head_params)
    print(f"  {'TRAINED':12s} {n_head:>8,}")
    assert 'restore_head' not in groups, "restore_head must be absent"

    opt = torch.optim.Adam(head_params, lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    sl1 = nn.SmoothL1Loss()

    t0 = time.time()
    hist = []
    for ep in range(epochs):
        model.train()
        model.encoder.eval()
        tot, n = 0.0, 0
        for X, NL, Y, _S in loader:
            X, NL, Y = nz(X), NL, Y
            out = model(X)
            loss = (args.lambda_loc * sl1(out['loc'], Y)
                    + args.lambda_nlos * bce(out['nlos_logits'], NL))
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

    model.eval()

    def evaluate(scene_ids, label):
        idx = [k for k, (s, _, _) in enumerate(ds.windows) if s in scene_ids]
        dl = DataLoader(torch.utils.data.Subset(ds, idx),
                        batch_size=cfg.ssl_batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
        P, G, S, NLQ, LQ = [], [], [], [], []
        with torch.no_grad():
            for X, NL, Y, SC in dl:
                out = model(nz(X))
                P.append(out['loc'].numpy()); G.append(Y.numpy())
                S.append(SC.numpy())
                NLQ.append(out['nlos_logits'].numpy()); LQ.append(NL.numpy())
        P = np.concatenate(P); G = np.concatenate(G)
        S = np.concatenate(S); NLQ = np.concatenate(NLQ); LQ = np.concatenate(LQ)
        E = np.sqrt(((P - G) ** 2).sum(-1))
        R = np.linalg.norm(G - CENTROID, axis=1)
        print(f"\n=== {label} ===  n={len(E)}")
        print(f"  rmse={np.sqrt((E**2).mean()):.4f}  med={np.median(E):.4f}"
              f"  p90={np.percentile(E,90):.4f}  p95={np.percentile(E,95):.4f}"
              f"  max={E.max():.4f}")
        for si in scene_ids:
            m = S == si
            e = E[m]
            print(f"    [{ds.scenes[si]['name']:26s}] "
                  f"rmse={np.sqrt((e**2).mean()):.4f} med={np.median(e):.4f} "
                  f"p90={np.percentile(e,90):.4f} max={e.max():.4f}")
        print("  by range:")
        for lo, hi in ((0, 5), (5, 10), (10, 14), (14, 25)):
            m = (R >= lo) & (R < hi)
            if m.sum() < 5:
                continue
            e = E[m]
            print(f"    {lo:2d}-{hi:2d} m n={m.sum():5d} med={np.median(e):.4f}"
                  f" p90={np.percentile(e,90):.4f}")
        print("  by NLOS in query:")
        q = LQ.mean(axis=(1, 2))
        for lo, hi, lab in ((0, .01, 'LOS only'), (.01, .3, 'some'),
                            (.3, 1.01, 'heavy')):
            m = (q >= lo) & (q < hi)
            if m.sum() < 5:
                continue
            e = E[m]
            print(f"    {lab:9s} n={m.sum():5d} med={np.median(e):.4f}"
                  f" p90={np.percentile(e,90):.4f}")
        yhat = (NLQ > 0).astype(np.float32)
        ytrue = LQ.astype(np.float32)          # already NLOS indicator
        tp = float((yhat * ytrue).sum()); fp = float((yhat * (1 - ytrue)).sum())
        fn = float(((1 - yhat) * ytrue).sum())
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        print(f"  NLOS head: P={pr:.4f} R={rc:.4f} F1={f1:.4f}"
              f"  (base NLOS={ytrue.mean()*100:.2f}%)")
        return E, f1

    E_h, f1_h = evaluate(set(held), "HELD-OUT SCENES")
    E_s, f1_s = evaluate(set(seen), "TRAIN (seen) SCENES")
    r_s = float(np.sqrt((E_s ** 2).mean())); r_h = float(np.sqrt((E_h ** 2).mean()))
    print(f"\ngap: seen {r_s:.4f} -> held {r_h:.4f}  ({r_h - r_s:+.4f} m)")

    out_dir = Path(args.out or
                   f"checkpoints/tier2_datasame_twohead_{Path(args.ckpt).parent.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'history': hist, 'cfg': cfg.__dict__,
        'norm_mean': mu, 'norm_std': sd_,
        'ssl_ckpt': str(args.ckpt),
        'heads': ['nlos', 'location'],
        'train_scenes': [ds.scenes[i]['name'] for i in seen],
        'held_out_scenes': [ds.scenes[i]['name'] for i in held],
        'held': dict(rmse=r_h, median=float(np.median(E_h)),
                     p90=float(np.percentile(E_h, 90)),
                     max=float(E_h.max()), nlos_f1=f1_h),
        'seen': dict(rmse=r_s, median=float(np.median(E_s)),
                     p90=float(np.percentile(E_s, 90)),
                     max=float(E_s.max()), nlos_f1=f1_s),
        'n_trained_params': n_head,
    }, out_dir / 'tier2_twohead.pt')
    print(f"\nsaved -> {out_dir / 'tier2_twohead.pt'}")
    with open(out_dir / 'history.csv', 'w', newline='') as f:
        w = csvmod.writer(f)
        w.writerow(['epoch', 'train_loss'])
        for e, l in hist:
            w.writerow([e, f'{l:.6f}'])


if __name__ == '__main__':
    main()