"""Cross-validation sweep over the 10 data_same scenes.

Rotates the held-out pair through all 5 indoor x 5 outdoor combinations
that keep exactly one indoor and one outdoor scene out:

    (indoor_01, outdoor_01) ... (indoor_05, outdoor_05)   5 folds
    (indoor_01, outdoor_02) ... (indoor_05, outdoor_05)  ... etc.

Actually: all 5 x 5 = 25 combinations are enumerated, but each fold holds
out ONE indoor and ONE outdoor scene, so each scene appears in the test
set exactly 5 times across the 25 folds. Because the training set is the
complement (8 scenes), every fold trains on a different 8-scene subset.

With --folds 5 the script picks 5 stratified folds (one per outdoor
scene, rotating the indoor scene), which is the practical version: it
gives 5 independent benchmarks where every scene is held out exactly
once.

For each fold it trains the Tier-2 heads on the 8 training scenes with
the Tier-0 encoder FROZEN, then evaluates on the 2 held-out scenes and
on the 8 training scenes, recording:
  - localization: RMSE, p50, p90, max (metres)
  - identification: NLOS F1, precision, recall

Results are written to a CSV and a JSON summary.

Usage:
    python3.11 -u -m scripts.crossval_tier2_datasame --folds 5 --epochs 300
    python3.11 -u -m scripts.crossval_tier2_datasame --folds 25 --epochs 300
    python3.11 -u -m scripts.crossval_tier2_datasame --quick
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path('/Users/justin/ultraclear-paper/code')
PAPER = Path('/Users/justin/ultraclear-paper')
# All cross-validation artefacts land in one place so the paper folder is
# self-contained: figures, tables, raw results and logs.
RESULTS = PAPER / 'results' / 'tier2_crossval'
TABLES = PAPER / 'tables' / 'tier2_crossval'
FIGS = PAPER / 'figures' / 'tier2_crossval'
LOGS = PAPER / 'logs' / 'tier2_crossval'
for _d in (RESULTS, TABLES, FIGS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)


def scene_pool():
    """Return (indoor_names, outdoor_names) from the data dir."""
    import glob
    data_dir = Path('/Users/justin/Downloads/srp_research/data_same')
    hits = sorted(glob.glob(str(data_dir / 'sionna_uwb_grid_*_A1A4swapped.csv')))
    names = [Path(h).name.replace('sionna_uwb_grid_', '')
             .replace('_A1A4swapped.csv', '') for h in hits]
    indoor = [n for n in names if n.startswith('indoor')]
    outdoor = [n for n in names if n.startswith('outdoor')]
    return indoor, outdoor


def build_folds(indoor, outdoor, n_folds):
    """Build the fold list.

    n_folds <= 5  : one fold per outdoor scene, rotating the indoor one.
                    Every scene appears exactly once in the test set.
    n_folds == 25 : all 5x5 combinations.
    """
    folds = []
    if n_folds >= 25:
        for i in indoor:
            for o in outdoor:
                folds.append((i, o))
        return folds
    # rotate: fold k holds out indoor[k % 5] and outdoor[k % 5]
    for k in range(n_folds):
        folds.append((indoor[k % len(indoor)], outdoor[k % len(outdoor)]))
    return folds


def run_fold(train_script, held, out_dir, epochs, lr, seed, quick, extra_args):
    """Run one training job, streaming its output to a per-fold log file.

    Streaming (rather than capture_output) matters: the train script
    prints a line per epoch, and buffering it all until the fold ends
    makes a 10-minute fold look like a hang.
    """
    cmd = [sys.executable, '-u', '-m', train_script,
           '--held_scenes', ','.join(held),
           '--out', str(out_dir),
           '--epochs', str(epochs),
           '--lr', str(lr),
           '--seed', str(seed)]
    if quick:
        cmd.append('--quick')
    cmd.extend(extra_args)
    t0 = time.time()
    print(f"\n{'=' * 72}\nFOLD: hold out {held}\n$ {' '.join(cmd)}\n{'=' * 72}",
          flush=True)

    log_path = LOGS / f"{out_dir.name}.log"
    with open(log_path, 'w') as lf:
        r = subprocess.run(cmd, cwd=str(CODE), stdout=lf,
                           stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f'--- tail of {log_path} ---')
        print(log_path.read_text()[-4000:])
        raise SystemExit(f'fold {held} FAILED (rc={r.returncode})')
    tail = log_path.read_text()
    print(tail[-2000:], flush=True)
    print(f"(full log: {log_path})", flush=True)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_script', default='scripts.train_tier2_datasame')
    ap.add_argument('--folds', type=int, default=5,
                    help='5 = one per outdoor scene (each scene held out '
                         'once); 25 = all indoor x outdoor combinations')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--tag', default='cv5')
    ap.add_argument('--discard_ckpts', action='store_true',
                    help='delete the per-fold checkpoints when the sweep '
                         'finishes. DEFAULT is to KEEP them -- they are '
                         'only ~350 KB each and they are the record of '
                         'every benchmark.')
    ap.add_argument('--extra', nargs='*', default=[],
                    help='extra args passed through to the train script')
    args = ap.parse_args()

    indoor, outdoor = scene_pool()
    print(f"indoor scenes ({len(indoor)}): {indoor}")
    print(f"outdoor scenes ({len(outdoor)}): {outdoor}")

    folds = build_folds(indoor, outdoor, args.folds)
    print(f"\n{len(folds)} folds:")
    for k, (i, o) in enumerate(folds):
        print(f"  fold {k}: hold out {i} + {o}")

    rows = []
    for k, (i, o) in enumerate(folds):
        held = [i, o]
        tag = re.sub(r'[^a-z0-9]+', '_', f"{i}_{o}".lower())
        out_dir = CODE / 'checkpoints' / f'tier2_cv_{args.tag}_fold{k}_{tag}'
        dt = run_fold(args.train_script, held, out_dir, args.epochs,
                      args.lr, args.seed, args.quick, args.extra)

        import torch
        ckpt_name = ('tier2_twohead.pt' if 'twohead' in args.train_script
                     else 'tier2_datasame.pt')
        ck = torch.load(out_dir / ckpt_name,
                        map_location='cpu', weights_only=False)
        row = {
            'fold': k,
            'held_indoor': i,
            'held_outdoor': o,
            'train_scenes': ' | '.join(ck['train_scenes']),
            'train_rmse': round(ck['seen']['rmse'], 4),
            'train_p50': round(ck['seen']['median'], 4),
            'train_p90': round(ck['seen']['p90'], 4),
            'train_max': round(ck['seen']['max'], 4),
            'train_f1': round(ck['seen']['nlos_f1'], 4),
            'held_rmse': round(ck['held']['rmse'], 4),
            'held_p50': round(ck['held']['median'], 4),
            'held_p90': round(ck['held']['p90'], 4),
            'held_max': round(ck['held']['max'], 4),
            'held_f1': round(ck['held']['nlos_f1'], 4),
            'epochs': len(ck['history']),
            'train_seconds': round(dt, 1),
        }
        rows.append(row)
        print(f"\n>>> fold {k}: HELD p50={row['held_p50']:.3f} "
              f"p90={row['held_p90']:.3f} rmse={row['held_rmse']:.3f} "
              f"F1={row['held_f1']:.4f}  |  "
              f"TRAIN p50={row['train_p50']:.3f} "
              f"p90={row['train_p90']:.3f} F1={row['train_f1']:.4f}")

        if args.discard_ckpts:
            for f in ('tier2_datasame.pt', 'tier2_twohead.pt', 'history.csv'):
                p = out_dir / f
                if p.exists():
                    p.unlink()
            # keep the directory empty rather than littering checkpoints
            try:
                out_dir.rmdir()
            except OSError:
                pass
        else:
            print(f"  kept: {out_dir}")

    # ---- summary ----
    csv_path = RESULTS / f'tier2_crossval_{args.tag}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                'min': float(v.min()), 'max': float(v.max())}

    summary = {
        'tag': args.tag,
        'n_folds': len(rows),
        'epochs': args.epochs,
        'seed': args.seed,
        'train_script': args.train_script,
        'folds': rows,
        'held_out': {k: agg(f'held_{k}') for k in ('rmse', 'p50', 'p90', 'max', 'f1')},
        'train': {k: agg(f'train_{k}') for k in ('rmse', 'p50', 'p90', 'max', 'f1')},
    }
    json_path = RESULTS / f'tier2_crossval_{args.tag}.json'
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 72}\nSUMMARY over {len(rows)} folds\n{'=' * 72}")
    for split in ('train', 'held_out'):
        print(f"-- {split} --")
        for k in ('rmse', 'p50', 'p90', 'max', 'f1'):
            a = summary[split][k]
            print(f"   {k:5s} mean {a['mean']:8.4f}  sd {a['std']:7.4f}  "
                  f"range [{a['min']:.4f}, {a['max']:.4f}]")
    print(f"\nsaved -> {csv_path}")
    print(f"saved -> {json_path}")


if __name__ == '__main__':
    main()
