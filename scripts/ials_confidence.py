"""Confidence weighting for implicit ALS: binary, frequency, and time-decay (train and validation only).

Preference p_uv = 1 for any training long view. Confidence on edges:
  binary:    c = 1 + alpha                                  (current iALS)
  frequency: c = 1 + alpha log(1 + n_uv)                    (n_uv = training long-view events)
  time:      c = 1 + alpha log(1 + sum_t exp(-(T - t)/tau)) (T = training cutoff, tau in days)
Non-edges keep preference 0 and confidence 1. k=32, 15 iterations, exact Gram-matrix solves.
Grids: binary and frequency alpha in {10, 30, 100, 300} x l2 in {10, 30, 100, 300} (16 each);
time tau in {2, 7} days x alpha in {10, 30, 100} x l2 in {30, 100, 300} (18). Validation users are
split 50/50 (seed 2028): the tuning half selects each scheme by Recall@100; final configurations
rerun with seeds 1-4 and are compared on the report half. Binary alpha=30, l2=100, seed 42 must
reproduce the saved train_mf.py run exactly.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import itertools
import json
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch
from retrieval_eval import load_targets, exact_top100

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
DIM, ITERATIONS = 32, 15
GRIDS = {'binary': [{'alpha': a, 'l2': l} for a, l in itertools.product([10, 30, 100, 300], [10, 30, 100, 300])],
         'frequency': [{'alpha': a, 'l2': l} for a, l in itertools.product([10, 30, 100, 300], [10, 30, 100, 300])],
         'time': [{'tau': t, 'alpha': a, 'l2': l} for t, a, l in itertools.product([2, 7], [10, 30, 100], [30, 100, 300])]}

def edges(taus=(2, 7)):
    catalog = np.sort(pd.read_csv(DATA/'candidate_videos.csv').video_id.to_numpy(np.int64))
    users = np.sort(pd.read_csv(DATA/'training_users.csv').user_id.to_numpy(np.int64))
    frames = []; max_train = 0
    for chunk in pd.read_csv(DATA/'train.csv.gz', usecols=['user_id','video_id','long_view','time_ms'], chunksize=250000):
        max_train = max(max_train, int(chunk.time_ms.max()))
        frames.append(chunk.loc[chunk.long_view.eq(1), ['user_id','video_id','time_ms']])
    events = pd.concat(frames, ignore_index=True)
    events['key'] = np.searchsorted(users, events.user_id.to_numpy())*len(catalog)+np.searchsorted(catalog, events.video_id.to_numpy())
    age_days = (max_train-events.time_ms.to_numpy())/86_400_000
    stats = pd.DataFrame({'key': events.key, 'count': 1, **{f'decay_{t:g}': np.exp(-age_days/t) for t in taus}}).groupby('key').sum().sort_index()
    return users, catalog, stats.index.to_numpy(np.int64), stats, max_train

def confidence_extra(stats, scheme, setting):
    """Returns c_uv - 1 for every edge, aligned with the sorted pair keys."""
    if scheme=='binary': return np.full(len(stats), float(setting['alpha']))
    signal = stats['count'] if scheme=='frequency' else stats[f'decay_{setting["tau"]:g}']
    return setting['alpha']*np.log1p(signal.to_numpy())

def ials(users, catalog, pair_keys, extra, l2, seed):
    """train_mf.train_ials with a per-edge confidence; identical operations when extra is constant."""
    torch.manual_seed(seed)
    n_items = len(catalog); k = DIM; u = torch.from_numpy(pair_keys//n_items); i = torch.from_numpy(pair_keys%n_items)
    a_edge = torch.from_numpy(extra).float()
    X = torch.randn(len(users), k)*.01; Y = torch.zeros(n_items, k)
    active = torch.unique(i); Y[active] = torch.randn(len(active), k)*.01
    eye = torch.eye(k)*l2
    by_item = torch.argsort(i, stable=True); i_sorted = i[by_item]; u_by_item = u[by_item]; a_by_item = a_edge[by_item]
    bounds = torch.searchsorted(i_sorted, active); bounds = torch.cat([bounds, torch.tensor([len(i_sorted)])])
    def solve_rows(F_other, rows, cols, weights, row_ids):
        G = F_other.T@F_other
        A = G.expand(len(row_ids), k, k).clone(); rhs = torch.zeros(len(row_ids), k)
        local = torch.searchsorted(row_ids, rows)
        for s in range(0, len(rows), 20000):
            f = F_other[cols[s:s+20000]]; r = local[s:s+20000]; a = weights[s:s+20000][:,None]
            A.index_add_(0, r, a[:,:,None]*f[:,:,None]*f[:,None,:]); rhs.index_add_(0, r, (1+a)*f)
        return torch.linalg.solve(A+eye, rhs)
    for it in range(ITERATIONS):
        X = solve_rows(Y, u, i, a_edge, torch.arange(len(users)))
        for c in range(0, len(active), 20000):
            ids = active[c:c+20000]; lo, hi = int(bounds[c]), int(bounds[min(c+20000, len(active))])
            Y[ids] = solve_rows(X, i_sorted[lo:hi], u_by_item[lo:hi], a_by_item[lo:hi], ids)
    return X, Y

def run(scheme, setting, seed, out, cache={}):
    if 'data' not in cache:
        users, catalog, pair_keys, stats, max_train = edges()
        groups, _, _, _ = load_targets(users, catalog, max_train)
        cache['data'] = (users, catalog, pair_keys, stats, groups)
    users, catalog, pair_keys, stats, groups = cache['data']
    name = f'{scheme}_'+'_'.join(f'{k}{v:g}' for k, v in setting.items())+f'_s{seed}'
    if (out/f'per_user_{name}.csv').exists(): return name  # resume: finished runs are kept, not refit
    started = time.perf_counter()
    X, Y = ials(users, catalog, pair_keys, confidence_extra(stats, scheme, setting), setting['l2'], seed)
    eval_users = groups.index.to_numpy(); rows = torch.from_numpy(np.searchsorted(users, eval_users))
    with torch.no_grad():
        ids, _ = exact_top100(lambda a, b: X[rows]@Y[a:b].T, len(eval_users), len(catalog), lambda m: None)
    recs = catalog[ids]
    recall = pd.DataFrame({f'recall_at_{k}': [len(t.intersection(r[:k]))/len(t) for t, r in zip(groups, recs)] for k in [20, 50, 100]}, index=eval_users)
    recall.to_csv(out/f'per_user_{name}.csv'); np.savez_compressed(out/f'top100_{name}.npz', user_ids=eval_users, video_ids=recs)
    print(f'{name}: Recall@100={recall.recall_at_100.mean():.4%}; {time.perf_counter()-started:.0f}s', flush=True)
    return name

def worker(jobs_file, out):
    torch.set_num_threads(4)
    for job in json.loads(Path(jobs_file).read_text()): run(job['scheme'], job['setting'], job['seed'], out)

def launch(jobs, out, tag):
    halves = [jobs[0::2], jobs[1::2]]; processes = []
    for n, half in enumerate(halves):
        path = out/f'jobs_{tag}_{n}.json'; path.write_text(json.dumps(half))
        processes.append(subprocess.Popen([sys.executable, __file__, '--worker', str(path), '--out', str(out)],
                                          stdout=open(out/f'log_{tag}_{n}.txt', 'w'), stderr=subprocess.STDOUT))
    if any(p.wait() for p in processes): raise RuntimeError(f'{tag} jobs failed')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--worker'); parser.add_argument('--out')
    parser.add_argument('--resume', help='existing run folder; completed configurations are reused'); args = parser.parse_args()
    if args.worker: return worker(args.worker, Path(args.out))
    out = Path(args.resume) if args.resume else ROOT/'reports/ials_confidence'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume: (out/'protocol.json').write_text(json.dumps({'preference': 'p_uv = 1 for any training long view', 'grids': GRIDS, 'dim': DIM,
        'iterations': ITERATIONS, 'split': 'validation users 50/50 with seed 2028; tuning half selects by Recall@100 per scheme',
        'final': 'selected configuration per scheme rerun with seeds 1-4; compared on the report half with paired bootstrap',
        'regression': 'binary alpha=30, l2=100, seed 42 equals the saved train_mf.py top-100', 'edge_rule': 'report edge winners; no automatic extension',
        'test_accessed': False}, indent=2))
    launch([{'scheme': s, 'setting': g, 'seed': 42} for s in GRIDS for g in GRIDS[s]], out, 'grid')
    users, catalog, pair_keys, stats, max_train = edges(); groups, _, _, _ = load_targets(users, catalog, max_train)
    order = np.random.default_rng(2028).permutation(len(groups)); tune = groups.index[np.sort(order[:len(groups)//2])]
    report = groups.index[np.sort(order[len(groups)//2:])]
    name = lambda s, g, seed: f'{s}_'+'_'.join(f'{k}{v:g}' for k, v in g.items())+f'_s{seed}'
    rows = []
    for s in GRIDS:
        for g in GRIDS[s]:
            r = pd.read_csv(out/f'per_user_{name(s, g, 42)}.csv', index_col=0)
            rows.append({'scheme': s, **g, **{f'tune_recall_at_{k}': r.loc[tune, f'recall_at_{k}'].mean() for k in [20, 50, 100]},
                         **{f'report_recall_at_{k}': r.loc[report, f'recall_at_{k}'].mean() for k in [20, 50, 100]}})
    grid = pd.DataFrame(rows); grid.to_csv(out/'grid.csv', index=False)
    saved_mf = ROOT/'reports/mf'/(ROOT/'reports/mf/latest_grid.txt').read_text().strip()
    reference = np.load(ROOT/'artifacts/mf'/json.loads((saved_mf/'selection.json').read_text())['ials']['selected_run_id']/'validation_top100.npz')
    mine = np.load(out/f'top100_{name("binary", {"alpha": 30, "l2": 100}, 42)}.npz')
    selected = {}
    for s in GRIDS:
        part = grid.loc[grid.scheme.eq(s)]; best = part.loc[part.tune_recall_at_100.idxmax()]
        setting = {k: best[k].item() for k in GRIDS[s][0]}
        edge = [k for k in setting if setting[k] in (min(x[k] for x in GRIDS[s]), max(x[k] for x in GRIDS[s]))]
        selected[s] = {'setting': setting, 'on_grid_edge': edge}
    launch([{'scheme': s, 'setting': selected[s]['setting'], 'seed': seed} for s in GRIDS for seed in [1, 2, 3, 4]], out, 'seeds')
    seeds = [42, 1, 2, 3, 4]
    per = {s: {k: pd.concat([pd.read_csv(out/f'per_user_{name(s, selected[s]["setting"], seed)}.csv', index_col=0)[f'recall_at_{k}'] for seed in seeds], axis=1)
               for k in [20, 50, 100]} for s in GRIDS}
    train = pd.read_csv(DATA/'train.csv.gz', usecols=['user_id','video_id','long_view'])
    activity = train.loc[train.long_view.eq(1)].drop_duplicates(['user_id','video_id']).groupby('user_id').size()
    tertile = pd.qcut(activity.reindex(groups.index), 3, labels=['light', 'medium', 'heavy'])
    rng = np.random.default_rng(2026); comparisons = []; summary = []
    for s in GRIDS:
        for k in [20, 50, 100]:
            seed_means = per[s][k].loc[report].mean(axis=0)
            summary.append({'scheme': s, 'k': k, 'report_recall_mean': float(seed_means.mean()), 'report_recall_sd': float(seed_means.std(ddof=1))})
    for s in ['frequency', 'time']:
        for k in [20, 50, 100]:
            for group in ['all', 'light', 'medium', 'heavy']:
                users_g = report if group=='all' else report[tertile.loc[report].eq(group).to_numpy()]
                diff = (per[s][k].loc[users_g].mean(axis=1)-per['binary'][k].loc[users_g].mean(axis=1)).to_numpy()
                draws = rng.choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
                comparisons.append({'comparison': f'{s} - binary', 'k': k, 'users': group, 'n': len(diff), 'difference': float(diff.mean()),
                                    'ci95_low': float(np.quantile(draws, .025)), 'ci95_high': float(np.quantile(draws, .975))})
    pd.DataFrame(summary).to_csv(out/'final_summary.csv', index=False); pd.DataFrame(comparisons).to_csv(out/'comparisons.csv', index=False)
    result = {'regression_matches_saved_ials': bool(np.array_equal(mine['video_ids'], reference['video_ids'])), 'selected': selected,
              'tune_users': len(tune), 'report_users': len(report), 'repeated_long_view_pairs': int((stats['count']>1).sum())}
    (out/'selection.json').write_text(json.dumps(result, indent=2, default=float))
    (ROOT/'reports/ials_confidence/latest_run.txt').write_text(out.name+'\n')
    pd.set_option('display.width', 250)
    print(json.dumps(result, indent=2, default=float)); print(pd.DataFrame(summary).to_string(index=False)); print(pd.DataFrame(comparisons).round(6).to_string(index=False))

if __name__ == '__main__':
    main()
