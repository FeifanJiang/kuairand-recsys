"""Embedding-prior screen for the selected sequence configurations (validation only).

Fixed configurations from runs_20261020: pool C=100 d=64 and SASRec C=50 d=64. Stage 1: embed_l2 in {1, 10, 100},
seed 42, same budget and early stopping; embed_l2 = 0 is the earlier run (control). Per encoder, the level with the
best tuning-half Recall@100 (filtered) is selected among {0, 1, 10, 100}. Stage 2 runs only when the selected level is
nonzero and beats embed_l2 = 0 on the tuning half by more than the earlier report-half seed SD (pool 0.108 pp,
SASRec 0.148 pp): seeds 1-4 at that level, compared with the five embed_l2 = 0 seeds on the report half.
Two runs in parallel (results are not bitwise reproducible under concurrent load). Resumable.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT/'reports/seq_retrieval/runs_20261020'
CONFIGS = {'pool': dict(encoder='pool', C=100, dim=64), 'sasrec': dict(encoder='sasrec', C=50, dim=64)}
LEVELS = [1, 10, 100]; THRESHOLD = {'pool': .00108, 'sasrec': .00148}

def name(c, l2, seed): return f"{c['encoder']}_C{c['C']}_d{c['dim']}"+(f'_l2-{l2:g}' if l2 else '')+f'_s{seed}'

def launch(jobs, out):
    pending = [j for j in jobs if not (out/name(*j)/'summary.json').exists()]; active = []
    while pending or active:
        while pending and len(active) < 2:
            c, l2, seed = pending.pop(0)
            cmd = [sys.executable, str(ROOT/'scripts/train_seq.py'), '--encoder', c['encoder'], '--C', str(c['C']), '--dim', str(c['dim']),
                   '--seed', str(seed), '--embed-l2', str(l2), '--out', str(out)]
            active.append(((c, l2, seed), subprocess.Popen(cmd, stdout=open(out/f'{name(c, l2, seed)}.stdout', 'w'), stderr=subprocess.STDOUT)))
        time.sleep(20)
        for item in list(active):
            if item[1].poll() is not None:
                active.remove(item)
                if item[1].returncode: raise RuntimeError(f'{name(*item[0])} failed')
                print(f'finished {name(*item[0])}', flush=True)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True); args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    launch([(CONFIGS[e], l2, 42) for e in CONFIGS for l2 in LEVELS], out)
    decision = {}
    for e, c in CONFIGS.items():
        tune = {0: json.loads((CONTROL/name(c, 0, 42)/'summary.json').read_text())['best_tune_recall_at_100_filtered']}
        for l2 in LEVELS: tune[l2] = json.loads((out/name(c, l2, 42)/'summary.json').read_text())['best_tune_recall_at_100_filtered']
        best = max(tune, key=tune.get); gain = tune[best]-tune[0]
        decision[e] = {'tuning_half_recall_at_100_by_l2': tune, 'selected_l2': best, 'gain_over_l2_0': gain,
                       'threshold': THRESHOLD[e], 'confirm': bool(best!=0 and gain > THRESHOLD[e])}
    (out/'screening_decision.json').write_text(json.dumps(decision, indent=2))
    confirm = [(CONFIGS[e], d['selected_l2'], s) for e, d in decision.items() if d['confirm'] for s in [1, 2, 3, 4]]
    if confirm: launch(confirm, out)
    print(json.dumps(decision, indent=2), flush=True)

if __name__ == '__main__':
    main()
