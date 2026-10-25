"""Run the pre-specified BPR-MF and iALS grids on validation, then seed replicates of each winner.

iALS: k in {16, 32, 64} x l2 in {1, 10, 100} x alpha in {3, 10, 30}.
BPR: k in {16, 32, 64} x (occurrence l2 in {1e-4, 1e-3, 1e-2} or Gaussian l2 in {1, 10, 30}).
Selection: validation macro Recall@100 per model; winners rerun with seeds 1-4. A winner on a
grid edge is reported as such. Test data are not read.
"""
from pathlib import Path
from datetime import datetime, timezone
import itertools
import json
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRIDS = {
    'ials': [{'dim': k, 'l2': l, 'alpha': a} for k, l, a in itertools.product([16, 32, 64], [1, 10, 100], [3, 10, 30])],
    'bpr': [{'dim': k, 'l2': l, 'penalty': p} for k in [16, 32, 64]
            for p, ls in [('occurrence', [1e-4, 1e-3, 1e-2]), ('gaussian', [1, 10, 30])] for l in ls],
}
SEEDS = [1, 2, 3, 4]

def run(model, setting, seed):
    command = [sys.executable, str(ROOT/'scripts/train_mf.py'), '--model', model, '--seed', str(seed)]
    for key, value in setting.items(): command += [f'--{key}', str(value)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    run_id = result.stdout.splitlines()[0].removeprefix('Run: ').split(';')[0]
    folder = ROOT/'reports/mf'/run_id
    metrics = pd.read_csv(folder/'metrics.csv').set_index('k')
    row = {'model': model, **setting, 'seed': seed, 'run_id': run_id,
           **{f'macro_recall_at_{k}': metrics.loc[k, 'macro_recall'] for k in [20, 50, 100]},
           'delta_vs_popularity_at_100': metrics.loc[100, 'paired_macro_delta'],
           **json.loads((folder/'diagnostics.json').read_text())}
    print(f'{model} {setting} seed={seed}: Recall@100={row["macro_recall_at_100"]:.4%}', flush=True)
    return row

def edges(model, best):
    grid = pd.DataFrame(GRIDS[model]); hits = []
    for key in grid.columns:
        values = sorted(grid[key].unique()) if key!='penalty' else None
        if values is None: continue
        pool = sorted(grid.loc[grid.penalty.eq(best['penalty']), key].unique()) if 'penalty' in grid and key=='l2' else values
        if best[key] in (pool[0], pool[-1]): hits.append(f'{key}={best[key]:g}')
    return hits

def main():
    out = ROOT/'reports/mf'/f'grid_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
    out.mkdir(parents=True)
    rows = []; selection = {}
    for model in ['ials', 'bpr']:
        for setting in GRIDS[model]:
            rows.append(run(model, setting, 42)); pd.DataFrame(rows).to_csv(out/'grid_results.csv', index=False)
        grid = pd.DataFrame([r for r in rows if r['model']==model])
        best = grid.loc[grid.macro_recall_at_100.idxmax()]
        setting = {k: (int(best[k]) if k=='dim' else best[k]) for k in GRIDS[model][0]}
        replicates = [best.to_dict()]+[run(model, setting, seed) for seed in SEEDS]
        pd.DataFrame(replicates).to_csv(out/f'replicates_{model}.csv', index=False)
        recall = pd.DataFrame(replicates).macro_recall_at_100
        selection[model] = {'setting': setting, 'selected_run_id': best.run_id, 'grid_edges_hit': edges(model, best),
            'seeds': [42]+SEEDS, 'recall_at_100_mean': float(recall.mean()), 'recall_at_100_sd': float(recall.std(ddof=1))}
        (out/'selection.json').write_text(json.dumps(selection, indent=2, default=float))
    (ROOT/'reports/mf/latest_grid.txt').write_text(out.name+'\n')
    print(json.dumps(selection, indent=2, default=float))

if __name__ == '__main__':
    main()
