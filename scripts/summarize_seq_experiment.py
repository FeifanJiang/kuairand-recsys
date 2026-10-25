"""Summarize the confirmed sequence-retrieval experiment (validation only; run after run_seq_experiment.py).

Baselines are recomputed on the same users, targets, filter, and split: frozen time-decay iALS from its saved
factors (five seeds) and time-decayed popularity (tau 0.5 d); their report-half means must match step 19.
Comparisons on the report half, seed-averaged per-user recall, paired user bootstrap (2000, seed 2026):
sasrec - pool, sasrec - ials_time, sasrec - time_popularity, pool - ials_time (filtered primary, unfiltered
secondary; K = 20/50/100). Also: the grid, chosen epochs, shuffle sensitivity, degree groups, fixed-zero slots.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from retrieval_eval import load_targets, exact_top100
from seq_retrieval import load_sequences

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
FOLLOWUP = 'reports/ials_time_followup/20261019T192838Z'
SEEDS = [42, 1, 2, 3, 4]; GROUPS = ['0', '1-2', '3-19', '>=20']

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--runs', required=True); args = parser.parse_args()
    runs = Path(args.runs); torch.set_num_threads(8)
    data = load_sequences(DATA); catalog, users = data['catalog'], data['users']; n = len(catalog)
    groups, _, _, _ = load_targets(users, catalog, data['T'])
    order = np.random.default_rng(2028).permutation(len(groups))
    split = pd.Series('report', index=groups.index); split.iloc[order[:len(groups)//2]] = 'tune'
    report = split.index[split.eq('report')]; eval_users = groups.index.to_numpy()
    targets_idx = [np.searchsorted(catalog, np.fromiter(t, np.int64)) for t in groups]
    degree = data['degree']; group_of = np.select([degree==0, degree<=2, degree<=19], [0, 1, 2], 3)
    seen = {u: data['trainable'][x-1] for u, (x, _) in data['sequences'].items()}
    rows = torch.from_numpy(np.searchsorted(users, eval_users)); pr, pi = [], []
    for r, u in enumerate(eval_users): pr += [r]*len(seen.get(u, ())); pi += list(seen.get(u, ()))
    pr, pi = torch.tensor(pr), torch.tensor(pi); o = torch.argsort(pi, stable=True); pr, pi = pr[o], pi[o]
    def recall_frame(ids):
        return pd.DataFrame({f'recall_at_{k}': [len(np.intersect1d(t, r[:k]))/len(t) for t, r in zip(targets_idx, ids)] for k in [20, 50, 100]}, index=eval_users)
    per = {}; lists = {}
    # Frozen time-decay iALS from saved factors.
    sel = json.loads((ROOT/FOLLOWUP/'stage1_extension/selection.json').read_text())['time']['setting']
    art = ROOT/'artifacts/ials_time_followup'/Path(FOLLOWUP).name
    for seed in SEEDS:
        f = torch.load(art/(f'time_'+'_'.join(f'{k}{v:g}' for k, v in sel.items())+f'_s{seed}.pt'), weights_only=False)
        for filtered in [False, True]:
            def block(a, b, X=f['X'], Y=f['Y']):
                s = X[rows]@Y[a:b].T
                if filtered:
                    lo, hi = torch.searchsorted(pi, a), torch.searchsorted(pi, b); s[pr[lo:hi], pi[lo:hi]-a] = float('-inf')
                return s
            with torch.no_grad(): ids, _ = exact_top100(block, len(rows), n, lambda m: None)
            per.setdefault(('ials_time', filtered), []).append(recall_frame(ids)); lists.setdefault(('ials_time', filtered), []).append(ids)
    # Time-decayed popularity (tau = 0.5 days), ties by ascending video_id.
    ev = pd.read_csv(DATA/'train.csv.gz', usecols=['video_id','long_view','time_ms']).query('long_view==1')
    decay = np.bincount(np.searchsorted(catalog, ev.video_id.to_numpy()), weights=np.exp(-(data['T']-ev.time_ms.to_numpy())/86_400_000/.5), minlength=n)
    order_tp = np.lexsort((catalog, -decay))
    for filtered in [False, True]:
        seen_sets = {u: set(seen.get(u, ()).tolist()) if u in seen else set() for u in eval_users}
        ids = np.array([[i for i in order_tp[:100+len(seen_sets[u])] if not (filtered and i in seen_sets[u])][:100] for u in eval_users])
        per[('time_popularity', filtered)] = [recall_frame(ids)]; lists[('time_popularity', filtered)] = [ids]
    tp = ROOT/'reports/time_popularity'/(ROOT/'reports/time_popularity/latest_run.txt').read_text().strip(); tsum = pd.read_csv(tp/'summary.csv')
    regression = {}
    for model, saved in [('ials_time', 'ials_time'), ('time_popularity', 'time_popularity_tau0.5')]:
        for filtered in [False, True]:
            for k in [20, 50, 100]:
                expected = float(tsum.query('method==@saved and seen_filter==@filtered and k==@k').report_recall.iloc[0])
                got = float(np.mean([f.loc[report, f'recall_at_{k}'].mean() for f in per[(model, filtered)]]))
                regression[f'{model}_{"filtered" if filtered else "unfiltered"}_k{k}'] = abs(expected-got) < 1e-12
    # Sequence models.
    selection = json.loads((runs/'selection.json').read_text())['selected']
    name = lambda c, s, sh=False: f"{c['encoder']}_C{c['C']}_d{c['dim']}_s{s}"+('_shuffle' if sh else '')
    epochs = {}
    for enc in ['pool', 'sasrec']:
        for seed in SEEDS:
            d = runs/name(selection[enc], seed); summ = json.loads((d/'summary.json').read_text()); epochs.setdefault(enc, []).append(summ['best_epoch'])
            for filtered in [False, True]:
                per.setdefault((enc, filtered), []).append(pd.read_csv(d/f'per_user_{"filtered" if filtered else "unfiltered"}.csv', index_col=0))
            top = np.load(ROOT/'artifacts/seq_retrieval'/runs.name/name(selection[enc], seed)/'top100.npz')
            for filtered in [False, True]: lists.setdefault((enc, filtered), []).append(np.searchsorted(catalog, top['filtered' if filtered else 'unfiltered']))
    grid = []
    for enc in ['pool', 'sasrec']:
        for C in [50, 100]:
            for dim in [32, 64]:
                c = dict(encoder=enc, C=C, dim=dim); d = runs/name(c, 42); summ = json.loads((d/'summary.json').read_text())
                pu = pd.read_csv(d/'per_user_filtered.csv', index_col=0)
                grid.append({'encoder': enc, 'C': C, 'dim': dim, 'best_epoch': summ['best_epoch'], 'epochs_trained': summ['epochs_trained'],
                             'tune_recall_at_100_filtered': summ['best_tune_recall_at_100_filtered'], 'report_recall_at_100_filtered': float(pu.loc[report, 'recall_at_100'].mean()),
                             'selected': selection[enc]==c})
    pd.DataFrame(grid).to_csv(runs/'grid_summary.csv', index=False)
    rng = np.random.default_rng(2026)
    def mean_user(model, filtered, k): return pd.concat([f[f'recall_at_{k}'] for f in per[(model, filtered)]], axis=1).mean(axis=1).loc[report]
    comps = []
    for a, b in [('sasrec', 'pool'), ('sasrec', 'ials_time'), ('sasrec', 'time_popularity'), ('pool', 'ials_time')]:
        for filtered in [True, False]:
            for k in [20, 50, 100]:
                diff = (mean_user(a, filtered, k)-mean_user(b, filtered, k)).to_numpy(); draws = rng.choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
                comps.append({'comparison': f'{a} - {b}', 'filtered': filtered, 'k': k, 'difference': float(diff.mean()),
                              'ci95_low': float(np.quantile(draws, .025)), 'ci95_high': float(np.quantile(draws, .975))})
    pd.DataFrame(comps).to_csv(runs/'comparisons.csv', index=False)
    table = []
    for model in ['time_popularity', 'ials_time', 'pool', 'sasrec']:
        for filtered in [True, False]:
            for k in [20, 50, 100]:
                vals = [f.loc[report, f'recall_at_{k}'].mean() for f in per[(model, filtered)]]
                table.append({'model': model, 'filtered': filtered, 'k': k, 'report_recall': float(np.mean(vals)), 'seed_sd': float(np.std(vals, ddof=1)) if len(vals)>1 else 0.})
    pd.DataFrame(table).to_csv(runs/'report_recall.csv', index=False)
    # Degree groups and fixed-zero videos on the report half (filtered lists, seed average).
    rep_pos = np.flatnonzero(split.to_numpy()=='report'); deg_rows = []
    for model in ['time_popularity', 'ials_time', 'pool', 'sasrec']:
        for ids in lists[(model, True)]:
            ids = ids[rep_pos]; hits = np.array([np.isin(r, targets_idx[i]) for r, i in zip(ids, rep_pos)]); counts = np.array([len(targets_idx[i]) for i in rep_pos])
            row = {'model': model, 'zero_degree_slot_share': float((degree[ids]==0).mean()), 'zero_degree_hits': int((hits & (degree[ids]==0)).sum())}
            for gi, g in enumerate(GROUPS):
                m = group_of[ids]==gi; row[f'slot_share_{g}'] = float(m.mean()); row[f'recall_contribution_{g}'] = float(((hits & m).sum(1)/counts).mean())
            deg_rows.append(row)
    pd.DataFrame(deg_rows).groupby('model', sort=False).mean().to_csv(runs/'degree_groups_report_half.csv')
    # Shuffle sensitivity.
    shuffle = {}
    for enc, seeds in [('sasrec', SEEDS), ('pool', [42])]:
        for filtered in [True, False]:
            for k in [20, 50, 100]:
                ordered = pd.concat([pd.read_csv(runs/name(selection[enc], s)/f'per_user_{"filtered" if filtered else "unfiltered"}.csv', index_col=0)[f'recall_at_{k}'] for s in seeds], axis=1).mean(axis=1).loc[report]
                shuffled = pd.concat([pd.read_csv(runs/name(selection[enc], s, True)/f'per_user_{"filtered" if filtered else "unfiltered"}.csv', index_col=0)[f'recall_at_{k}'] for s in seeds], axis=1).mean(axis=1).loc[report]
                diff = (shuffled-ordered).to_numpy(); draws = rng.choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
                shuffle[f'{enc}_{"filtered" if filtered else "unfiltered"}_k{k}'] = {'ordered': float(ordered.mean()), 'shuffled': float(shuffled.mean()), 'difference': float(diff.mean()),
                                                                                    'ci95': np.quantile(draws, [.025, .975]).tolist()}
    out = {'selection': selection, 'chosen_epochs': epochs, 'baseline_regression_vs_step19': regression, 'baseline_regression_all_passed': all(regression.values()),
           'shuffle_sensitivity': shuffle, 'evidence_status': 'report half inspected repeatedly; development evidence, not fresh confirmation', 'test_accessed': False}
    (runs/'experiment_summary.json').write_text(json.dumps(out, indent=2))
    pd.set_option('display.width', 250)
    print(json.dumps(out, indent=2)); print(pd.DataFrame(grid).to_string(index=False)); print(pd.DataFrame(table).query('k==100').to_string(index=False)); print(pd.DataFrame(comps).to_string(index=False))

if __name__ == '__main__':
    main()
