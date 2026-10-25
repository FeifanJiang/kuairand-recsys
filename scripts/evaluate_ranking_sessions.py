"""Post-hoc within-context AUC for the latest ranking suite (added after seeing suite results).

Per-user AUC pools exposures across scenarios and hours, so request context (tab, hour)
earns credit it cannot earn when ranking candidates inside one request. Here AUC is
computed within (user, date, hour, tab) groups, where context covariates are constant,
then averaged per user. Uses saved validation logits; no retraining, no test data.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
COMPARISONS = [('covariates','degree'), ('degree_covariates','covariates'), ('degree_covariates','degree'),
               ('full','degree_covariates')]

def group_auc(groups, y, s):
    frame = pd.DataFrame({'g':groups,'y':y,'s':s})
    frame['r'] = frame.groupby('g').s.rank()
    g = frame.assign(rp=frame.r*frame.y).groupby('g').agg(n=('y','size'),n1=('y','sum'),rank_pos=('rp','sum'))
    g = g.loc[g.n1.gt(0) & g.n1.lt(g.n)]
    return ((g.rank_pos-g.n1*(g.n1+1)/2)/(g.n1*(g.n-g.n1))).rename('auc'), g.n

def main():
    suite = ROOT/'reports/ranking'/(ROOT/'reports/ranking/latest_suite.txt').read_text().strip()
    runs = pd.read_csv(suite/'runs.csv')
    rows, comparisons = [], []
    rng = np.random.default_rng(2026)
    for split in ['validation','random_validation']:
        frame = pd.read_csv(DATA/f'{split}.csv.gz', usecols=['user_id','time_ms','tab','long_view'])
        local = pd.to_datetime(frame.time_ms, unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
        key = pd.DataFrame({'u':frame.user_id,'d':local.dt.strftime('%Y%m%d'),'h':local.dt.hour,'t':frame.tab})
        groups = pd.factorize(pd.MultiIndex.from_frame(key))[0]
        group_user = pd.Series(frame.user_id.to_numpy(), index=groups).groupby(level=0).first()
        y = frame.long_view.to_numpy()
        per_user = {}
        for r in runs.itertuples():
            logit = np.load(ROOT/'artifacts/ranking'/r.run_id/'validation_logits.npz')[split]
            auc, n = group_auc(groups, y, logit)
            user_auc = auc.groupby(group_user.loc[auc.index].to_numpy()).mean()
            per_user.setdefault(r.variant, []).append(user_auc)
            rows.append({'split':split,'variant':r.variant,'seed':r.seed,'groups':len(auc),
                         'exposures_in_groups':int(n.sum()),'users':len(user_auc),'session_gauc':float(user_auc.mean())})
        per_user = {v: pd.concat(s, axis=1).mean(axis=1) for v, s in per_user.items()}
        for a, b in COMPARISONS:
            common = per_user[a].index.intersection(per_user[b].index)
            diff = (per_user[a].loc[common]-per_user[b].loc[common]).to_numpy()
            draws = rng.choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
            comparisons.append({'split':split,'comparison':f'{a} - {b}','users':len(diff),'session_gauc_difference':float(diff.mean()),
                                'ci95_low':float(np.quantile(draws,.025)),'ci95_high':float(np.quantile(draws,.975))})
    results = pd.DataFrame(rows); results.to_csv(suite/'session_metrics.csv', index=False)
    summary = results.groupby(['split','variant'], sort=False).agg(groups=('groups','first'),users=('users','first'),
        exposures_in_groups=('exposures_in_groups','first'),session_gauc_mean=('session_gauc','mean'),session_gauc_std=('session_gauc','std'))
    summary.reset_index().to_csv(suite/'session_summary.csv', index=False)
    pd.DataFrame(comparisons).to_csv(suite/'session_comparisons.csv', index=False)
    (suite/'session_protocol.json').write_text(json.dumps({'added_after_seeing_results':True,
        'group':'(user, Asia/Shanghai date, hour, tab); groups need both outcomes',
        'aggregation':'mean over groups per user, then mean over users; per-user values averaged over seeds for comparisons',
        'bootstrap':'paired users, 2000 resamples, seed 2026','test_accessed':False}, indent=2))
    print(summary.round(4).to_string()); print(pd.DataFrame(comparisons).round(5).to_string(index=False))

if __name__ == '__main__':
    main()
