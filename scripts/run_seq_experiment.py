"""Run the confirmed sequence-retrieval protocol: grid, seed replicates, and shuffle sensitivity (validation only).

Stage 1: 8 configurations (pool/sasrec x C in {50, 100} x dim in {32, 64}), seed 42.
Stage 2: per encoder, the configuration with the best tuning-half Recall@100 (filtered) at its checkpoint
         reruns with seeds 1-4, each with its own tuning-half early stopping.
Stage 3: membership-preserving within-timestamp shuffle: selected SASRec with seeds 42 and 1-4, selected pool
         with seed 42.
Two runs in parallel. Finished runs (with summary.json) are skipped, so the script can resume.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def launch(configs, out):
    pending = [c for c in configs if not (out/name(c)/'summary.json').exists()]; active = []
    while pending or active:
        while pending and len(active) < 2:
            c = pending.pop(0)
            cmd = [sys.executable, str(ROOT/'scripts/train_seq.py'), '--encoder', c['encoder'], '--C', str(c['C']), '--dim', str(c['dim']),
                   '--seed', str(c['seed']), '--out', str(out)] + (['--shuffle'] if c.get('shuffle') else [])
            log = open(out/f'{name(c)}.stdout', 'w')
            active.append((c, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
        time.sleep(20)
        for item in list(active):
            if item[1].poll() is not None:
                active.remove(item)
                if item[1].returncode: raise RuntimeError(f'{name(item[0])} failed; see {name(item[0])}.stdout')
                print(f'finished {name(item[0])}', flush=True)

def name(c):
    return f"{c['encoder']}_C{c['C']}_d{c['dim']}_s{c['seed']}"+('_shuffle' if c.get('shuffle') else '')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True); args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    grid = [dict(encoder=e, C=C, dim=d, seed=42) for e in ['pool', 'sasrec'] for C in [50, 100] for d in [32, 64]]
    launch(grid, out)
    selected = {}
    for e in ['pool', 'sasrec']:
        runs = [json.loads((out/name(c)/'summary.json').read_text()) | {'config': c} for c in grid if c['encoder']==e]
        best = max(runs, key=lambda r: r['best_tune_recall_at_100_filtered'])
        selected[e] = {k: best['config'][k] for k in ['encoder', 'C', 'dim']}
    (out/'selection.json').write_text(json.dumps({'selected': selected, 'rule': 'best tuning-half Recall@100 (filtered) at the checkpoint, seed 42'}, indent=2))
    launch([dict(selected[e], seed=s) for e in ['pool', 'sasrec'] for s in [1, 2, 3, 4]], out)
    launch([dict(selected['sasrec'], seed=s, shuffle=True) for s in [42, 1, 2, 3, 4]]+[dict(selected['pool'], seed=42, shuffle=True)], out)
    print(json.dumps(selected), flush=True)

if __name__ == '__main__':
    main()
