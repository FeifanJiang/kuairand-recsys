"""Non-personalized time baselines versus time-decay iALS (train and validation only).

Separates global popularity shifts from personalized recency. Baselines: popularity (training
long-view events), time-decayed popularity score_v = sum over v's long views of exp(-(T - t)/tau)
with tau in {0.5, 1, 2, 7} days chosen on the tuning half, and freshness (newest upload first,
ties by training long views). Ties end by ascending video_id. Compared on the report half with the
frozen five-seed binary and time-decay iALS fits from ials_time_followup.py, without and with the
seen filter (the user's training long-view videos); target denominators never change.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import torch
from retrieval_eval import load_targets, exact_top100
from ials_confidence import edges

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
META = ROOT/'data/raw/KuaiRand-1K/data/video_features_basic_1k.csv'
FOLLOWUP = ROOT/'reports/ials_time_followup/20261019T192838Z'
TAUS = [.5, 1, 2, 7]; SEEDS = [42, 1, 2, 3, 4]

def main():
    torch.set_num_threads(8)
    out = ROOT/'reports/time_popularity'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); out.mkdir(parents=True)
    users, catalog, pair_keys, stats, max_train = edges((2, 7)); n_items = len(catalog)
    groups, _, _, _ = load_targets(users, catalog, max_train)
    order = np.random.default_rng(2028).permutation(len(groups))
    split = pd.Series('report', index=groups.index); split.iloc[order[:len(groups)//2]] = 'tune'
    tune, report = split.index[split.eq('tune')], split.index[split.eq('report')]
    events = pd.read_csv(DATA/'train.csv.gz', usecols=['video_id','long_view','time_ms']).query('long_view==1')
    age = (max_train-events.time_ms.to_numpy())/86_400_000; index = np.searchsorted(catalog, events.video_id.to_numpy())
    counts = np.bincount(index, minlength=n_items).astype(float)
    meta = pd.read_csv(META, usecols=['video_id','upload_dt']).set_index('video_id').upload_dt.reindex(catalog)
    upload = pd.to_datetime(meta, errors='coerce').astype('int64').to_numpy().astype(float); upload[pd.isna(meta).to_numpy()] = -np.inf
    orders = {'popularity': np.lexsort((catalog, -counts)), 'freshness': np.lexsort((catalog, -counts, -upload))}
    for tau in TAUS:
        orders[f'time_popularity_tau{tau:g}'] = np.lexsort((catalog, -np.bincount(index, weights=np.exp(-age/tau), minlength=n_items)))
    seen = pd.Series(pair_keys%n_items, index=users[pair_keys//n_items]).groupby(level=0).agg(set)
    def global_lists(order, filtered):
        lists = []
        for u in groups.index:
            s = seen.get(u, set()) if filtered else set()
            picked = [catalog[i] for i in order[:100+len(s)] if i not in s][:100]
            lists.append(picked)
        return lists
    def recall(lists): return pd.DataFrame({f'recall_at_{k}': [len(t.intersection(r[:k]))/len(t) for t, r in zip(groups, lists)] for k in [20, 50, 100]}, index=groups.index)
    per = {}
    for name, order in orders.items():
        for filtered in [False, True]: per[(name, filtered)] = [recall(global_lists(order, filtered))]
    tau_rows = [{'tau_days': tau, 'tune_recall_at_100': per[(f'time_popularity_tau{tau:g}', False)][0].loc[tune, 'recall_at_100'].mean(),
                 'report_recall_at_100': per[(f'time_popularity_tau{tau:g}', False)][0].loc[report, 'recall_at_100'].mean()} for tau in TAUS]
    tau_table = pd.DataFrame(tau_rows); tau_table.to_csv(out/'tau_selection.csv', index=False)
    best_tau = float(tau_table.loc[tau_table.tune_recall_at_100.idxmax(), 'tau_days'])
    # Frozen iALS fits: unfiltered and seen-filtered full-catalog re-ranking from saved factors.
    selection = json.loads((FOLLOWUP/'stage1_extension/selection.json').read_text())
    art = ROOT/'artifacts/ials_time_followup'/FOLLOWUP.name
    rows = torch.from_numpy(np.searchsorted(users, groups.index.to_numpy())); row_of = {u: r for r, u in enumerate(groups.index)}
    pairs = [(row_of[u], i) for u, items in seen.items() if u in row_of for i in items]
    pr = torch.tensor([p[0] for p in pairs]); pi = torch.tensor([p[1] for p in pairs]); o = torch.argsort(pi); pr, pi = pr[o], pi[o]
    fname = lambda s, g, seed: f'{s}_'+'_'.join(f'{k}{v:g}' for k, v in g.items())+f'_s{seed}'
    time_lists = {}
    for scheme in ['binary', 'time']:
        for filtered in [False, True]:
            per[(f'ials_{scheme}', filtered)] = []
            for seed in SEEDS:
                f = torch.load(art/f'{fname(scheme, selection[scheme]["setting"], seed)}.pt', weights_only=False)
                def block(a, b, X=f['X'], Y=f['Y']):
                    s = X[rows]@Y[a:b].T
                    if filtered:
                        lo, hi = torch.searchsorted(pi, a), torch.searchsorted(pi, b); s[pr[lo:hi], pi[lo:hi]-a] = float('-inf')
                    return s
                with torch.no_grad(): ids, _ = exact_top100(block, len(rows), n_items, lambda m: None)
                per[(f'ials_{scheme}', filtered)].append(recall(catalog[ids]))
                if scheme=='time' and not filtered: time_lists[seed] = catalog[ids]
    methods = ['popularity', 'freshness', f'time_popularity_tau{best_tau:g}', 'ials_binary', 'ials_time']
    rng = np.random.default_rng(2026); summary = []; comparisons = []
    for filtered in [False, True]:
        for m in methods:
            for k in [20, 50, 100]:
                values = [p.loc[report, f'recall_at_{k}'].mean() for p in per[(m, filtered)]]
                summary.append({'method': m, 'seen_filter': filtered, 'k': k, 'report_recall': float(np.mean(values)),
                                'seed_sd': float(np.std(values, ddof=1)) if len(values)>1 else 0.})
        for a, b in [('ials_time', f'time_popularity_tau{best_tau:g}'), ('ials_time', 'ials_binary'), (f'time_popularity_tau{best_tau:g}', 'popularity'),
                     (f'time_popularity_tau{best_tau:g}', 'ials_binary')]:
            for k in [20, 50, 100]:
                x = pd.concat([p[f'recall_at_{k}'] for p in per[(a, filtered)]], axis=1).mean(axis=1).loc[report]
                y = pd.concat([p[f'recall_at_{k}'] for p in per[(b, filtered)]], axis=1).mean(axis=1).loc[report]
                diff = (x-y).to_numpy(); draws = rng.choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
                comparisons.append({'seen_filter': filtered, 'comparison': f'{a} - {b}', 'k': k, 'difference': float(diff.mean()),
                                    'ci95_low': float(np.quantile(draws, .025)), 'ci95_high': float(np.quantile(draws, .975))})
    pd.DataFrame(summary).to_csv(out/'summary.csv', index=False); pd.DataFrame(comparisons).to_csv(out/'comparisons.csv', index=False)
    global_top = set(catalog[orders[f'time_popularity_tau{best_tau:g}'][:100]])
    overlap = float(np.mean([np.mean([len(set(r) & global_top)/100 for r in time_lists[s]]) for s in SEEDS]))
    hits_in_global = []
    for s in SEEDS:
        hit = [v for t, r in zip(groups, time_lists[s]) for v in r if v in t]
        hits_in_global.append(np.mean([v in global_top for v in hit]))
    info = {'selected_tau_days': best_tau, 'time_ials_top100_overlap_with_time_popularity_top100': overlap,
            'share_of_time_ials_hits_in_time_popularity_top100': float(np.mean(hits_in_global)),
            'ials_configurations': selection, 'test_accessed': False}
    (out/'info.json').write_text(json.dumps(info, indent=2))
    (ROOT/'reports/time_popularity/latest_run.txt').write_text(out.name+'\n')
    pd.set_option('display.width', 250)
    print(tau_table.to_string(index=False)); print(json.dumps(info, indent=2, default=float))
    print(pd.DataFrame(summary).query('k==100').round(5).to_string(index=False)); print(pd.DataFrame(comparisons).round(5).to_string(index=False))

if __name__ == '__main__':
    main()
