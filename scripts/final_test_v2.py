"""Frozen final evaluation (reports/test_phase_v2/protocol.json).

--phase validation: train on standard train, evaluate on validation, and check every model against
the saved validation results (regression check of this code path; no test data are read).
--phase test: train on standard train + validation (data/processed/test_phase), evaluate once on the
standard test split and the random_test supplement, exactly as the protocol specifies.
Models: popularity, freshness, time_popularity (tau 0.5 d), bpr, ials_binary, ials_time; seeded models
use seeds 42, 1, 2, 3, 4. Settings: unfiltered and filtered (the user's train + validation long views,
frozen at T, excluded from the full catalog before top-K; denominators unchanged).
"""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import argparse
import json
import numpy as np
import pandas as pd
import torch
from retrieval_eval import exact_top100
from ials_confidence import confidence_extra, ials
from train_mf import train_bpr
from train_ranker import per_user_auc

ROOT = Path(__file__).resolve().parents[1]
META = ROOT/'data/raw/KuaiRand-1K/data/video_features_basic_1k.csv'
PROTOCOL = json.loads((ROOT/'reports/test_phase_v2/protocol.json').read_text())
M = PROTOCOL['models']; SEEDS = M['ials_time']['seeds']; TAU = M['ials_time']['tau_days']
SEEDED = ['bpr', 'ials_binary', 'ials_time']; MODELS = ['popularity', 'freshness', 'time_popularity']+SEEDED

def load(data, split):
    catalog = np.sort(pd.read_csv(data/'candidate_videos.csv').video_id.to_numpy(np.int64))
    users = np.sort(pd.read_csv(data/'training_users.csv').user_id.to_numpy(np.int64))
    frames = []; T = 0
    for chunk in pd.read_csv(data/'train.csv.gz', usecols=['user_id','video_id','long_view','time_ms'], chunksize=250000):
        T = max(T, int(chunk.time_ms.max())); frames.append(chunk.loc[chunk.long_view.eq(1), ['user_id','video_id','time_ms']])
    events = pd.concat(frames, ignore_index=True)
    events['key'] = np.searchsorted(users, events.user_id.to_numpy())*len(catalog)+np.searchsorted(catalog, events.video_id.to_numpy())
    age_days = (T-events.time_ms.to_numpy())/86_400_000
    stats = pd.DataFrame({'key': events.key, 'count': 1, f'decay_{TAU:g}': np.exp(-age_days/TAU)}).groupby('key').sum().sort_index()
    targets = []; split_min = 10**18
    for chunk in pd.read_csv(data/f'{split}.csv.gz', usecols=['user_id','video_id','long_view','time_ms'], chunksize=250000):
        split_min = min(split_min, int(chunk.time_ms.min())); targets.append(chunk.loc[chunk.long_view.eq(1), ['user_id','video_id']])
    assert T < split_min
    target = pd.concat(targets).drop_duplicates()
    eligible = target.loc[target.user_id.isin(users) & target.video_id.isin(catalog)]
    groups = eligible.groupby('user_id').video_id.agg(set)
    coverage = {'all_target_pairs': len(target), 'eligible_target_pairs': len(eligible), 'coverage': len(eligible)/len(target), 'users': len(groups)}
    return users, catalog, stats.index.to_numpy(np.int64), stats, events, T, groups, coverage

def global_scores(catalog, events, T):
    """Higher is better; ties resolved by ascending video_id through the order index."""
    index = np.searchsorted(catalog, events.video_id.to_numpy()); n = len(catalog)
    counts = np.bincount(index, minlength=n).astype(float)
    decay = np.bincount(index, weights=np.exp(-(T-events.time_ms.to_numpy())/86_400_000/TAU), minlength=n)
    meta = pd.read_csv(META, usecols=['video_id','upload_dt']).set_index('video_id').upload_dt.reindex(catalog)
    upload = pd.to_datetime(meta, errors='coerce').astype('int64').to_numpy().astype(float); upload[pd.isna(meta).to_numpy()] = -np.inf
    orders = {'popularity': np.lexsort((catalog, -counts)), 'freshness': np.lexsort((catalog, -counts, -upload)),
              'time_popularity': np.lexsort((catalog, -decay))}
    raw = {'popularity': counts, 'time_popularity': decay, 'freshness': None}
    return orders, raw

def fit(model, seed, users, catalog, pair_keys, stats):
    if model=='bpr':
        c = M['bpr']; torch.manual_seed(seed); rng = np.random.default_rng(seed)
        args = SimpleNamespace(dim=c['dim'], l2=float(c['l2']), penalty='gaussian', negatives=c['negatives'], epochs=c['epochs'],
                               batch_size=c['batch_size'], item_lr=c['item_learning_rate'], user_lr=c['user_learning_rate'])
        score, _, _ = train_bpr(args, users, catalog, pair_keys, rng, lambda m: None)
        return lambda rows, a, b: score(rows, a, b).detach()
    c = M[model]; scheme = 'binary' if model=='ials_binary' else 'time'
    setting = {'alpha': c['alpha']} if scheme=='binary' else {'tau': c['tau_days'], 'alpha': c['alpha']}
    X, Y = ials(users, catalog, pair_keys, confidence_extra(stats, scheme, setting), float(c['l2']), seed)
    return lambda rows, a, b: X[rows]@Y[a:b].T

def run(phase):
    torch.set_num_threads(4)
    data = ROOT/('data/processed/test_phase' if phase=='test' else 'data/processed'); split = 'test' if phase=='test' else 'validation'
    out = ROOT/'reports/test_phase_v2'/('results' if phase=='test' else 'validation_regression')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    users, catalog, pair_keys, stats, events, T, groups, coverage = load(data, split); n = len(catalog)
    eval_users = groups.index.to_numpy(); rows = torch.from_numpy(np.searchsorted(users, eval_users)); row_of = {u: r for r, u in enumerate(eval_users)}
    seen = pd.Series(pair_keys%n, index=users[pair_keys//n]).groupby(level=0).agg(set)
    pr, pi = [], []
    for u, items in seen.items():
        if u in row_of: pr += [row_of[u]]*len(items); pi += list(items)
    pr, pi = torch.tensor(pr), torch.tensor(pi); o = torch.argsort(pi, stable=True); pr, pi = pr[o], pi[o]
    target_index = [set(np.searchsorted(catalog, np.fromiter(t, np.int64)).tolist()) for t in groups]
    unreachable = sum(len(t & seen.get(u, set())) for u, t in zip(eval_users, target_index))
    def recall(ids):
        return pd.DataFrame({f'recall_at_{k}': [len(t.intersection(r[:k]))/len(t) for t, r in zip(target_index, ids)] for k in [20, 50, 100]}, index=eval_users)
    def model_topk(block, filtered):
        def masked(a, b):
            s = block(rows, a, b).clone()
            if filtered:
                lo, hi = torch.searchsorted(pi, a), torch.searchsorted(pi, b); s[pr[lo:hi], pi[lo:hi]-a] = float('-inf')
            return s
        with torch.no_grad(): ids, _ = exact_top100(masked, len(rows), n, lambda m: None)
        return ids
    def global_topk(order, filtered):
        return [[i for i in order[:100+len(seen.get(u, ()))] if not (filtered and i in seen.get(u, ()))][:100] for u in eval_users]
    orders, raw = global_scores(catalog, events, T)
    per = {}; lists = {}; random_scores = {}
    random = None
    if phase=='test':
        random = pd.read_csv(data/'random_test.csv.gz', usecols=['user_id','video_id','long_view','time_ms'])
        assert int(random.time_ms.min()) > T
        random = random.loc[random.user_id.isin(users) & random.video_id.isin(catalog)].reset_index(drop=True)
        ru = torch.from_numpy(np.searchsorted(users, random.user_id.to_numpy())); rv = torch.from_numpy(np.searchsorted(catalog, random.video_id.to_numpy()))
    for model in MODELS:
        for seed in (SEEDS if model in SEEDED else [None]):
            if model in SEEDED: block = fit(model, seed, users, catalog, pair_keys, stats)
            for filtered in [False, True]:
                ids = model_topk(block, filtered) if model in SEEDED else global_topk(orders[model], filtered)
                per.setdefault((model, filtered), []).append(recall(ids))
                if not filtered: lists[(model, seed)] = catalog[np.asarray(ids)]
            if random is not None and model!='freshness':
                score = pair_scores(block, ru, rv) if model in SEEDED else raw[model][rv.numpy()]
                random_scores.setdefault(model, []).append(per_user_auc(random.user_id.to_numpy(), random.long_view.to_numpy(), score))
            print(f'{phase}: {model} seed {seed} done', flush=True)
    for (model, filtered), frames in per.items():
        pd.concat(frames, axis=1, keys=range(len(frames))).to_csv(out/f'per_user_{model}_{"filtered" if filtered else "unfiltered"}.csv')
    np.savez_compressed(out/'unfiltered_top100.npz', **{f'{m}_s{s}': v for (m, s), v in lists.items()})
    return out, per, lists, random_scores, coverage, unreachable, groups, random

def pair_scores(block, ru, rv):
    """Score observed (user, video) pairs: one call per distinct video, over the users exposed to it."""
    scores = np.empty(len(ru)); videos = rv.numpy()
    with torch.no_grad():
        for v in np.unique(videos):
            idx = np.flatnonzero(videos==v)
            scores[idx] = block(ru[torch.from_numpy(idx)], int(v), int(v)+1)[:, 0].numpy()
    return scores

def summarize(phase, out, per, random_scores, coverage, unreachable, groups, random):
    rng = np.random.default_rng(2026); users = groups.index
    def mean_user(model, filtered, k): return pd.concat([f[f'recall_at_{k}'] for f in per[(model, filtered)]], axis=1).mean(axis=1)
    table = []
    for model in MODELS:
        for filtered in [False, True]:
            for k in [20, 50, 100]:
                seed_means = [f[f'recall_at_{k}'].mean() for f in per[(model, filtered)]]
                table.append({'model': model, 'filtered': filtered, 'k': k, 'recall': float(np.mean(seed_means)),
                              'seed_sd': float(np.std(seed_means, ddof=1)) if len(seed_means)>1 else 0.})
    comps = []
    pairs = [('primary', 'ials_time', 'time_popularity', True), ('key_secondary', 'ials_time', 'ials_binary', True),
             ('secondary', 'ials_time', 'time_popularity', False), ('secondary', 'ials_time', 'ials_binary', False),
             ('secondary', 'bpr', 'ials_binary', True), ('secondary', 'bpr', 'popularity', True),
             ('secondary', 'bpr', 'ials_binary', False), ('secondary', 'bpr', 'popularity', False)]
    for role, a, b, filtered in pairs:
        for k in [20, 50, 100]:
            diff = (mean_user(a, filtered, k)-mean_user(b, filtered, k)).to_numpy(); draws = rng.choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
            comps.append({'role': role if k==100 else 'secondary', 'comparison': f'{a} - {b}', 'filtered': filtered, 'k': k,
                          'difference': float(diff.mean()), 'ci95_low': float(np.quantile(draws, .025)), 'ci95_high': float(np.quantile(draws, .975))})
    pd.DataFrame(table).to_csv(out/'recall.csv', index=False); pd.DataFrame(comps).to_csv(out/'comparisons.csv', index=False)
    summary = {'phase': phase, 'coverage': coverage, 'filtered_unreachable_targets': int(unreachable),
               'filtered_unreachable_share': unreachable/coverage['eligible_target_pairs']}
    if random is not None:
        means = {m: pd.concat(v, axis=1).mean(axis=1) for m, v in random_scores.items()}; rows = []
        for other in [m for m in means if m!='ials_time']:
            common = means['ials_time'].index.intersection(means[other].index)
            diff = (means['ials_time']-means[other]).loc[common].to_numpy(); draws = rng.choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
            rows.append({'comparison': f'ials_time - {other}', 'users': len(common), 'ials_time_auc': float(means['ials_time'].loc[common].mean()),
                         'other_auc': float(means[other].loc[common].mean()), 'difference': float(diff.mean()),
                         'ci95_low': float(np.quantile(draws, .025)), 'ci95_high': float(np.quantile(draws, .975))})
        pd.DataFrame(rows).to_csv(out/'random_test_auc.csv', index=False)
        summary['random_test'] = {'eligible_exposures': len(random), 'long_view_rate': float(random.long_view.mean()),
                                  'per_seed_auc': {m: [float(a.mean()) for a in v] for m, v in random_scores.items()}}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    return pd.DataFrame(table), pd.DataFrame(comps), summary

def regression_check(out, per, lists, groups):
    """Validation phase: every model must reproduce the saved validation results."""
    users = groups.index; order = np.random.default_rng(2028).permutation(len(users)); report = users[np.sort(order[len(users)//2:])]
    checks = {}
    fu = ROOT/'reports/ials_time_followup/20261019T192838Z'; sel = json.loads((fu/'stage1_extension/selection.json').read_text())
    name = lambda s, g, seed: f'{s}_'+'_'.join(f'{k}{v:g}' for k, v in g.items())+f'_s{seed}'
    for model, scheme in [('ials_binary', 'binary'), ('ials_time', 'time')]:
        for i, seed in enumerate(SEEDS):
            saved = pd.read_csv(fu/'final_fits'/f'per_user_{name(scheme, sel[scheme]["setting"], seed)}.csv', index_col=0)
            # Saved values went through CSV text; compare with the same tolerance as the other checks.
            checks[f'{model}_s{seed}_unfiltered_per_user'] = bool(np.allclose(saved.loc[users].to_numpy(), per[(model, False)][i].loc[users].to_numpy(), rtol=0, atol=1e-12))
    mf = ROOT/'reports/mf'/(ROOT/'reports/mf/latest_grid.txt').read_text().strip()
    bpr_runs = pd.read_csv(mf/'replicates_bpr.csv').fillna({'seed': 42}).astype({'seed': int}).set_index('seed').run_id
    for seed in SEEDS:
        saved = np.load(ROOT/'artifacts/mf'/bpr_runs.loc[seed]/'validation_top100.npz')['video_ids']
        checks[f'bpr_s{seed}_unfiltered_top100'] = bool(np.array_equal(saved, lists[('bpr', seed)]))
    pop = pd.read_csv(ROOT/'reports/popularity/per_user_metrics.csv').set_index('user_id')
    checks['popularity_unfiltered_per_user'] = bool(np.allclose(pop.loc[users, 'recall_at_100'].to_numpy(), per[('popularity', False)][0].loc[users, 'recall_at_100'].to_numpy(), rtol=0, atol=1e-15))
    tp = ROOT/'reports/time_popularity'/(ROOT/'reports/time_popularity/latest_run.txt').read_text().strip()
    tsum = pd.read_csv(tp/'summary.csv')
    for model, saved_name in [('time_popularity', 'time_popularity_tau0.5'), ('freshness', 'freshness'), ('ials_time', 'ials_time'), ('ials_binary', 'ials_binary'), ('popularity', 'popularity')]:
        for filtered in [False, True]:
            for k in [20, 50, 100]:
                expected = float(tsum.query('method==@saved_name and seen_filter==@filtered and k==@k').report_recall.iloc[0])
                got = float(np.mean([f.loc[report, f'recall_at_{k}'].mean() for f in per[(model, filtered)]]))
                checks[f'{model}_{"filtered" if filtered else "unfiltered"}_report_recall_at_{k}'] = bool(abs(expected-got) < 1e-12)
    s3 = pd.read_csv(fu/'stage3_seen_filter/results.csv')
    for k in [20, 50, 100]:
        expected = float(s3.query('model=="bpr" and k==@k').report_filtered.iloc[0])
        got = float(np.mean([f.loc[report, f'recall_at_{k}'].mean() for f in per[('bpr', True)]]))
        checks[f'bpr_filtered_report_recall_at_{k}'] = bool(abs(expected-got) < 1e-12)
    (out/'regression_checks.json').write_text(json.dumps({'all_passed': all(checks.values()), 'checks': checks}, indent=2))
    return checks

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['validation', 'test'], required=True); args = parser.parse_args()
    out, per, lists, random_scores, coverage, unreachable, groups, random = run(args.phase)
    table, comps, summary = summarize(args.phase, out, per, random_scores, coverage, unreachable, groups, random)
    if args.phase=='validation':
        checks = regression_check(out, per, lists, groups)
        print('regression all passed:', all(checks.values())); print({k: v for k, v in checks.items() if not v})
    pd.set_option('display.width', 250)
    print(json.dumps(summary, indent=2)); print(table.query('k==100').to_string(index=False)); print(comps.query('k==100').to_string(index=False))

if __name__ == '__main__':
    main()
