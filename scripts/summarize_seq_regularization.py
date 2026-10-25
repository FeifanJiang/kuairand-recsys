"""Summarize the embedding-prior experiment (validation only).

Report half, seed-averaged per-user recall, paired user bootstrap (2000, seed 2026): pool embed_l2=100 vs 0,
SASRec embed_l2=10 vs 0, each regularized model vs frozen time-decay iALS and time-decayed popularity, and
SASRec vs pooling under regularization. Row norms are computed from every best checkpoint (the embed_l2 = 0
runs did not log norms during training), with degree-group means; curves give peak-to-final declines.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from retrieval_eval import load_targets, exact_top100
from seq_retrieval import load_sequences

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
CONTROL = ROOT/'reports/seq_retrieval/runs_20261020'; REG = ROOT/'reports/seq_retrieval/regularization/runs'
FOLLOWUP = 'reports/ials_time_followup/20261019T192838Z'; SEEDS = [42, 1, 2, 3, 4]
MODELS = {'pool_l2_0': (CONTROL, 'pool_C100_d64_s{}'), 'pool_l2_100': (REG, 'pool_C100_d64_l2-100_s{}'),
          'sasrec_l2_0': (CONTROL, 'sasrec_C50_d64_s{}'), 'sasrec_l2_10': (REG, 'sasrec_C50_d64_l2-10_s{}')}

def main():
    torch.set_num_threads(8)
    data = load_sequences(DATA); catalog, users = data['catalog'], data['users']; n = len(catalog)
    groups, _, _, _ = load_targets(users, catalog, data['T'])
    order = np.random.default_rng(2028).permutation(len(groups))
    split = pd.Series('report', index=groups.index); split.iloc[order[:len(groups)//2]] = 'tune'
    report = split.index[split.eq('report')]; eval_users = groups.index.to_numpy()
    targets_idx = [np.searchsorted(catalog, np.fromiter(t, np.int64)) for t in groups]
    seen = {u: data['trainable'][x-1] for u, (x, _) in data['sequences'].items()}
    def recall_frame(ids):
        return pd.DataFrame({f'recall_at_{k}': [len(np.intersect1d(t, r[:k]))/len(t) for t, r in zip(targets_idx, ids)] for k in [20, 50, 100]}, index=eval_users)
    per = {}
    rows = torch.from_numpy(np.searchsorted(users, eval_users)); pr, pi = [], []
    for r, u in enumerate(eval_users): pr += [r]*len(seen.get(u, ())); pi += list(seen.get(u, ()))
    pr, pi = torch.tensor(pr), torch.tensor(pi); o = torch.argsort(pi, stable=True); pr, pi = pr[o], pi[o]
    sel = json.loads((ROOT/FOLLOWUP/'stage1_extension/selection.json').read_text())['time']['setting']
    art = ROOT/'artifacts/ials_time_followup'/Path(FOLLOWUP).name
    for seed in SEEDS:
        f = torch.load(art/('time_'+'_'.join(f'{k}{v:g}' for k, v in sel.items())+f'_s{seed}.pt'), weights_only=False)
        for filtered in [False, True]:
            def block(a, b, X=f['X'], Y=f['Y']):
                s = X[rows]@Y[a:b].T
                if filtered:
                    lo, hi = torch.searchsorted(pi, a), torch.searchsorted(pi, b); s[pr[lo:hi], pi[lo:hi]-a] = float('-inf')
                return s
            with torch.no_grad(): ids, _ = exact_top100(block, len(rows), n, lambda m: None)
            per.setdefault(('ials_time', filtered), []).append(recall_frame(ids))
    ev = pd.read_csv(DATA/'train.csv.gz', usecols=['video_id','long_view','time_ms']).query('long_view==1')
    decay = np.bincount(np.searchsorted(catalog, ev.video_id.to_numpy()), weights=np.exp(-(data['T']-ev.time_ms.to_numpy())/86_400_000/.5), minlength=n)
    order_tp = np.lexsort((catalog, -decay))
    for filtered in [False, True]:
        sets = {u: set(seen[u].tolist()) if u in seen else set() for u in eval_users}
        ids = np.array([[i for i in order_tp[:100+len(sets[u])] if not (filtered and i in sets[u])][:100] for u in eval_users])
        per[('time_popularity', filtered)] = [recall_frame(ids)]
    epochs, curves = {}, {}
    for m, (folder, pattern) in MODELS.items():
        for seed in SEEDS:
            d = folder/pattern.format(seed); s = json.loads((d/'summary.json').read_text()); epochs.setdefault(m, []).append(s['best_epoch'])
            c = pd.read_csv(d/'curve.csv'); e = c.dropna(subset=['tune_recall_at_100_filtered'])
            curves.setdefault(m, []).append({'peak_tune': float(e.tune_recall_at_100_filtered.max()), 'final_tune': float(e.tune_recall_at_100_filtered.iloc[-1]),
                                             'loss_at_best': float(c.loc[c.epoch.eq(s['best_epoch']), 'train_loss'].iloc[0]), 'final_loss': float(c.train_loss.iloc[-1])})
            for filtered in [False, True]:
                per.setdefault((m, filtered), []).append(pd.read_csv(d/f'per_user_{"filtered" if filtered else "unfiltered"}.csv', index_col=0))
    deg = torch.from_numpy(data['degree'][data['trainable']]); norms = []
    for m, (folder, pattern) in MODELS.items():
        for seed in SEEDS:
            sd = torch.load(ROOT/'artifacts/seq_retrieval'/folder.name/pattern.format(seed)/'best_model.pt')
            row = {'model': m, 'seed': seed}
            for table in ['inp', 'out']:
                r = sd[f'{table}.weight'][1:].norm(dim=1)
                row.update({f'{table}_p50': float(r.median()), f'{table}_p99': float(r.quantile(.99)), f'{table}_mean_deg_le2': float(r[deg<=2].mean()),
                            f'{table}_mean_deg_ge20': float(r[deg>=20].mean())})
            norms.append(row)
    norms = pd.DataFrame(norms); norms.to_csv(REG/'norms_best_checkpoints.csv', index=False)
    table = []
    for m in ['time_popularity', 'ials_time']+list(MODELS):
        for filtered in [True, False]:
            for k in [20, 50, 100]:
                v = [f.loc[report, f'recall_at_{k}'].mean() for f in per[(m, filtered)]]
                table.append({'model': m, 'filtered': filtered, 'k': k, 'report_recall': float(np.mean(v)), 'seed_sd': float(np.std(v, ddof=1)) if len(v)>1 else 0.})
    pd.DataFrame(table).to_csv(REG/'report_recall.csv', index=False)
    rng = np.random.default_rng(2026); comps = []
    mean_user = lambda m, f, k: pd.concat([x[f'recall_at_{k}'] for x in per[(m, f)]], axis=1).mean(axis=1).loc[report]
    for a, b in [('pool_l2_100', 'pool_l2_0'), ('sasrec_l2_10', 'sasrec_l2_0'), ('pool_l2_100', 'ials_time'), ('sasrec_l2_10', 'ials_time'),
                 ('pool_l2_100', 'time_popularity'), ('sasrec_l2_10', 'time_popularity'), ('sasrec_l2_10', 'pool_l2_100')]:
        for f in [True, False]:
            for k in [20, 50, 100]:
                diff = (mean_user(a, f, k)-mean_user(b, f, k)).to_numpy(); draws = rng.choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
                comps.append({'comparison': f'{a} - {b}', 'filtered': f, 'k': k, 'difference': float(diff.mean()),
                              'ci95_low': float(np.quantile(draws, .025)), 'ci95_high': float(np.quantile(draws, .975))})
    pd.DataFrame(comps).to_csv(REG/'comparisons.csv', index=False)
    curve_summary = {m: {k: float(np.mean([c[k] for c in v])) for k in v[0]} for m, v in curves.items()}
    out = {'chosen_epochs': epochs, 'curves_mean_over_seeds': curve_summary, 'norms_mean_over_seeds': norms.groupby('model').mean(numeric_only=True).drop(columns='seed').to_dict('index'),
           'evidence_status': 'report half inspected repeatedly; development evidence', 'test_accessed': False}
    (REG/'experiment_summary.json').write_text(json.dumps(out, indent=2))
    pd.set_option('display.width', 250)
    print(json.dumps(out, indent=2)); print(pd.DataFrame(table).query('k==100').round(5).to_string(index=False)); print(pd.DataFrame(comps).round(5).to_string(index=False))

if __name__ == '__main__':
    main()
