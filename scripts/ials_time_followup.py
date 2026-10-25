"""Follow-up to the time-aware iALS run (train and validation only; test untouched).

Stage 1, limited extension then freeze: time decay adds tau in {0.5, 1} days x alpha {10, 30, 100} x
l2 {30, 100, 300}; binary iALS adds l2 in {1000, 3000} x alpha {10, 30, 100, 300}. Earlier seed-42
grid results are reused. Selection uses the tuning half only (seed-2028 split); a changed selection
reruns seeds 1-4.
Stage 2, random-exposure check: frozen binary and time configurations, five seeds, per-user AUC on
random_validation exposures of known users and in-catalog videos; no tuning.
Stage 3, seen-video filter: seen = the user's training long-view videos, frozen at the cutoff. Every
model (popularity, LSM, BPR, binary iALS, time iALS) is refit deterministically from its seeds and
re-ranked over the full catalog with seen videos excluded. Unfiltered regenerated lists must equal
the saved ones. Same target denominators before and after; filtered-out targets are reported.
"""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import argparse
import itertools
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from retrieval_eval import load_targets, exact_top100
from ials_confidence import edges, confidence_extra, ials
from lsm_lr_branches import Trainer
from train_mf import train_bpr
from train_ranker import per_user_auc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
PREVIOUS = ROOT/'reports/ials_confidence/20261019T073602Z'
TAUS = (0.5, 1, 2, 7)
NEW = {'time': [{'tau': t, 'alpha': a, 'l2': l} for t, a, l in itertools.product([.5, 1], [10, 30, 100], [30, 100, 300])],
       'binary': [{'alpha': a, 'l2': l} for a, l in itertools.product([10, 30, 100, 300], [1000, 3000])]}
SEEDS = [42, 1, 2, 3, 4]
name = lambda s, g, seed: f'{s}_'+'_'.join(f'{k}{v:g}' for k, v in g.items())+f'_s{seed}'

def shared():
    users, catalog, pair_keys, stats, max_train = edges(TAUS)
    groups, _, _, _ = load_targets(users, catalog, max_train)
    order = np.random.default_rng(2028).permutation(len(groups))
    split = pd.Series('report', index=groups.index); split.iloc[order[:len(groups)//2]] = 'tune'
    return users, catalog, pair_keys, stats, max_train, groups, split

def recall_frame(groups, recs):
    return pd.DataFrame({f'recall_at_{k}': [len(t.intersection(r[:k]))/len(t) for t, r in zip(groups, recs)] for k in [20, 50, 100]}, index=groups.index)

def fit_ials(scheme, setting, seed, d, art=None):
    users, catalog, pair_keys, stats, groups = d['users'], d['catalog'], d['pair_keys'], d['stats'], d['groups']
    X, Y = ials(users, catalog, pair_keys, confidence_extra(stats, scheme, setting), setting['l2'], seed)
    if art is not None: torch.save({'X': X, 'Y': Y, 'scheme': scheme, 'setting': setting, 'seed': seed}, art/f'{name(scheme, setting, seed)}.pt')
    return X, Y

def worker(jobs, out, art):
    torch.set_num_threads(4)
    users, catalog, pair_keys, stats, max_train, groups, split = shared()
    d = {'users': users, 'catalog': catalog, 'pair_keys': pair_keys, 'stats': stats, 'groups': groups}
    rows = torch.from_numpy(np.searchsorted(users, groups.index.to_numpy()))
    for job in json.loads(Path(jobs).read_text()):
        n = name(job['scheme'], job['setting'], job['seed'])
        if (out/f'per_user_{n}.csv').exists(): continue
        X, Y = fit_ials(job['scheme'], job['setting'], job['seed'], d, art if job.get('save') else None)
        with torch.no_grad(): ids, _ = exact_top100(lambda a, b: X[rows]@Y[a:b].T, len(rows), len(catalog), lambda m: None)
        recall_frame(groups, catalog[ids]).to_csv(out/f'per_user_{n}.csv'); print(n, flush=True)

def launch(jobs, out, art, tag):
    procs = []
    for k, half in enumerate([jobs[0::2], jobs[1::2]]):
        path = out/f'jobs_{tag}_{k}.json'; path.write_text(json.dumps(half))
        procs.append(subprocess.Popen([sys.executable, __file__, '--worker', str(path), '--out', str(out), '--art', str(art)],
                                      stdout=open(out/f'log_{tag}_{k}.txt', 'w'), stderr=subprocess.STDOUT))
    if any(p.wait() for p in procs): raise RuntimeError(f'{tag} failed')

def stage1(root, art, split):
    out = root/'stage1_extension'; out.mkdir(exist_ok=True)
    launch([{'scheme': s, 'setting': g, 'seed': 42} for s in NEW for g in NEW[s]], out, art, 'grid')
    tune = split.index[split.eq('tune')]; report = split.index[split.eq('report')]; rows = []
    old = pd.read_csv(PREVIOUS/'grid.csv')
    for s in ['binary', 'time']:
        keys = ['tau', 'alpha', 'l2'] if s=='time' else ['alpha', 'l2']
        for _, r in old.loc[old.scheme.eq(s)].iterrows():
            g = {k: float(r[k]) for k in keys}
            rows.append({'scheme': s, **g, 'stage': 'previous', 'tune_recall_at_100': r.tune_recall_at_100, 'report_recall_at_100': r.report_recall_at_100})
        for g in NEW[s]:
            pu = pd.read_csv(out/f'per_user_{name(s, g, 42)}.csv', index_col=0)
            rows.append({'scheme': s, **g, 'stage': 'extension', 'tune_recall_at_100': pu.loc[tune, 'recall_at_100'].mean(),
                         'report_recall_at_100': pu.loc[report, 'recall_at_100'].mean()})
    grid = pd.DataFrame(rows); grid.to_csv(out/'grid.csv', index=False)
    selected = {}
    for s in ['binary', 'time']:
        part = grid.loc[grid.scheme.eq(s)]; best = part.loc[part.tune_recall_at_100.idxmax()]
        keys = ['tau', 'alpha', 'l2'] if s=='time' else ['alpha', 'l2']
        setting = {k: float(best[k]) for k in keys}
        selected[s] = {'setting': setting, 'stage': best.stage,
                       'on_grid_edge': [k for k in keys if setting[k] in (part[k].min(), part[k].max())]}
    (out/'selection.json').write_text(json.dumps(selected, indent=2)); return selected

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--worker'); parser.add_argument('--out'); parser.add_argument('--art')
    parser.add_argument('--resume', help='run stamp to continue; finished fits are reused'); args = parser.parse_args()
    if args.worker: return worker(args.worker, Path(args.out), Path(args.art))
    torch.set_num_threads(4)  # regenerated LSM and BPR fits must match the 4-thread runs that saved the reference lists
    stamp = args.resume or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = ROOT/'reports/ials_time_followup'/stamp; art = ROOT/'artifacts/ials_time_followup'/stamp
    root.mkdir(parents=True, exist_ok=True); art.mkdir(parents=True, exist_ok=True)
    if not args.resume: (root/'protocol.json').write_text(json.dumps({'previous_run': str(PREVIOUS.relative_to(ROOT)), 'extension': NEW, 'seeds': SEEDS,
        'selection': 'tuning half only (seed 2028); frozen before stages 2 and 3',
        'random_check': 'per-user AUC on random_validation for known users and in-catalog videos; seed-averaged per user; paired bootstrap; no tuning',
        'seen_filter': 'seen = training long-view videos of the user, frozen at cutoff; full-catalog re-ranking; same targets; all models',
        'budget_note': 'time decay searched 36 configurations in total, binary 24', 'test_accessed': False}, indent=2))
    users, catalog, pair_keys, stats, max_train, groups, split = shared()
    selected = stage1(root, art, split)
    # Final five-seed fits of the frozen configurations, saving factors for stages 2 and 3.
    final = root/'final_fits'; final.mkdir(exist_ok=True)
    launch([{'scheme': s, 'setting': selected[s]['setting'], 'seed': seed, 'save': True} for s in ['binary', 'time'] for seed in SEEDS], final, art, 'final')
    report = split.index[split.eq('report')]
    # Stage 2: random-exposure validation.
    s2 = root/'stage2_random'; s2.mkdir(exist_ok=True)
    random = pd.read_csv(DATA/'random_validation.csv.gz', usecols=['user_id','video_id','long_view','time_ms'])
    assert int(random.time_ms.min())>max_train
    random = random.loc[random.user_id.isin(users) & random.video_id.isin(catalog)].reset_index(drop=True)
    ru = torch.from_numpy(np.searchsorted(users, random.user_id.to_numpy())); rv = torch.from_numpy(np.searchsorted(catalog, random.video_id.to_numpy()))
    aucs = {s: [] for s in ['binary', 'time']}
    for s in aucs:
        for seed in SEEDS:
            f = torch.load(art/f'{name(s, selected[s]["setting"], seed)}.pt', weights_only=False)
            score = (f['X'][ru]*f['Y'][rv]).sum(-1).numpy()
            aucs[s].append(per_user_auc(random.user_id.to_numpy(), random.long_view.to_numpy(), score))
    mean = {s: pd.concat(v, axis=1).mean(axis=1) for s, v in aucs.items()}
    common = mean['binary'].index.intersection(mean['time'].index); diff = (mean['time']-mean['binary']).loc[common].to_numpy()
    draws = np.random.default_rng(2026).choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
    stage2 = {'exposures': len(random), 'long_view_rate': float(random.long_view.mean()), 'users_with_both_outcomes': len(common),
              'binary_per_user_auc': float(mean['binary'].loc[common].mean()), 'time_per_user_auc': float(mean['time'].loc[common].mean()),
              'per_seed_binary': [float(a.mean()) for a in aucs['binary']], 'per_seed_time': [float(a.mean()) for a in aucs['time']],
              'difference': float(diff.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist()}
    (s2/'results.json').write_text(json.dumps(stage2, indent=2)); print(json.dumps(stage2, indent=2), flush=True)
    # Stage 3: seen-video filter for every model, full-catalog re-ranking.
    s3 = root/'stage3_seen_filter'; s3.mkdir(exist_ok=True); n_items = len(catalog)
    eval_users = groups.index.to_numpy(); rows = torch.from_numpy(np.searchsorted(users, eval_users))
    row_of_user = {u: r for r, u in enumerate(np.searchsorted(users, eval_users))}
    seen_u, seen_i = pair_keys//n_items, pair_keys%n_items
    keep = np.isin(seen_u, list(row_of_user)); seen_r = np.array([row_of_user[u] for u in seen_u[keep]]); seen_i = seen_i[keep]
    order = np.argsort(seen_i, kind='stable'); seen_r, seen_i = torch.from_numpy(seen_r[order]), torch.from_numpy(seen_i[order])
    def topk(score_block, filtered):
        def block(a, b):
            s = score_block(a, b).clone()
            if filtered:
                lo, hi = torch.searchsorted(seen_i, a), torch.searchsorted(seen_i, b)
                s[seen_r[lo:hi], seen_i[lo:hi]-a] = float('-inf')
            return s
        ids, _ = exact_top100(block, len(rows), n_items, lambda m: None); return catalog[ids]
    target_pairs = [(u, v) for u, t in groups.items() for v in t]
    seen_set = set(pair_keys.tolist()); uidx = dict(zip(users, range(len(users))))
    unreachable = sum((uidx[u]*n_items+np.searchsorted(catalog, v)) in seen_set for u, v in target_pairs)
    events = pd.read_csv(DATA/'train.csv.gz', usecols=['video_id','long_view']).query('long_view==1').video_id.value_counts().reindex(catalog, fill_value=0)
    pop_order = np.lexsort((catalog, -events.to_numpy())); pop_score = torch.empty(n_items); pop_score[torch.from_numpy(pop_order)] = -torch.arange(n_items, dtype=torch.float32)
    lsm_grid = ROOT/'reports/lsm'/(ROOT/'reports/lsm/latest_grid.txt').read_text().strip()
    mf_grid = ROOT/'reports/mf'/(ROOT/'reports/mf/latest_grid.txt').read_text().strip()
    saved = {'lsm': (pd.read_csv(lsm_grid/'replicates.csv'), ROOT/'artifacts/lsm'), 'bpr': (pd.read_csv(mf_grid/'replicates_bpr.csv'), ROOT/'artifacts/mf')}
    def lsm_block(seed):
        args = SimpleNamespace(negatives=4, batch_size=2048, dim=16, l2=10., item_lr=.02, user_lr=.01)
        tr = Trainer(users, catalog, pair_keys, args); model, sp, de = tr.build(seed); rng = np.random.default_rng(seed)
        for _ in range(10): tr.epoch(model, sp, de, rng)
        return lambda a, b: model.score_block(rows, a, b).detach()
    def bpr_block(seed):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        args = SimpleNamespace(dim=16, l2=10., penalty='gaussian', negatives=4, epochs=10, batch_size=2048, item_lr=.02, user_lr=.01)
        score, _, _ = train_bpr(args, users, catalog, pair_keys, rng, lambda m: None)
        return lambda a, b: score(rows, a, b).detach()
    def ials_block(s, seed):
        f = torch.load(art/f'{name(s, selected[s]["setting"], seed)}.pt', weights_only=False); return lambda a, b: f['X'][rows]@f['Y'][a:b].T
    results = []; checks = {}; per_user = {}
    for model in ['popularity', 'lsm', 'bpr', 'ials_binary', 'ials_time']:
        for seed in ([42] if model=='popularity' else SEEDS):
            block = {'popularity': lambda: (lambda a, b: pop_score[a:b][None,:].expand(len(rows), -1)), 'lsm': lambda: lsm_block(seed),
                     'bpr': lambda: bpr_block(seed), 'ials_binary': lambda: ials_block('binary', seed), 'ials_time': lambda: ials_block('time', seed)}[model]()
            with torch.no_grad():
                unfiltered = topk(block, False); filtered = topk(block, True)
            if model in saved:
                table, folder = saved[model]; run_id = table.fillna({'seed': 42}).astype({'seed': int}).set_index('seed').loc[seed, 'run_id']
                checks[f'{model}_s{seed}'] = bool(np.array_equal(np.load(folder/run_id/'validation_top100.npz')['video_ids'], unfiltered))
            for kind, recs in [('unfiltered', unfiltered), ('filtered', filtered)]:
                pu = recall_frame(groups, recs); per_user.setdefault((model, kind), []).append(pu)
            print(f'stage 3 {model} seed {seed} done', flush=True)
    rng = np.random.default_rng(2026)
    for model in ['popularity', 'lsm', 'bpr', 'ials_binary', 'ials_time']:
        for k in [20, 50, 100]:
            before = pd.concat([p[f'recall_at_{k}'] for p in per_user[(model, 'unfiltered')]], axis=1).mean(axis=1).loc[report]
            after = pd.concat([p[f'recall_at_{k}'] for p in per_user[(model, 'filtered')]], axis=1).mean(axis=1).loc[report]
            diff = (after-before).to_numpy(); draws = rng.choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
            results.append({'model': model, 'k': k, 'report_unfiltered': float(before.mean()), 'report_filtered': float(after.mean()),
                            'difference': float(diff.mean()), 'ci95_low': float(np.quantile(draws, .025)), 'ci95_high': float(np.quantile(draws, .975))})
    pd.DataFrame(results).to_csv(s3/'results.csv', index=False)
    (s3/'checks.json').write_text(json.dumps({'regenerated_unfiltered_equals_saved': checks, 'target_pairs': len(target_pairs),
        'targets_unreachable_after_filter': int(unreachable), 'unreachable_share': unreachable/len(target_pairs)}, indent=2))
    (ROOT/'reports/ials_time_followup/latest_run.txt').write_text(stamp+'\n')
    print(json.dumps(selected, indent=2)); print(pd.DataFrame(results).round(6).to_string(index=False)); print(json.load(open(s3/'checks.json')))

if __name__ == '__main__':
    main()
