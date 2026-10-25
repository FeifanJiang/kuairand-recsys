"""Video-feature and personalization logistic models for random-exposure scoring (validation only).

Protocol: reports/feature_model/protocol_confirmed.json.
Training rows: standard training exposures; every statistic for a row at time t uses only training events
with time < t (same-timestamp events excluded), computed by grouping on (key, t), taking cumulative sums
within the key, and subtracting the row's own timestamp group.
Evaluation rows (standard validation, random validation): history frozen at the training cutoff T; decayed
counts are aged to the exposure time and hours since the last long view use the exposure time.
L1: video attributes and video/author/tag statistics. L2: L1 plus user, user x author, user x tag, and
duration-preference features. Logistic regression, mean log loss + l2 ||beta||^2, full-batch L-BFGS.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from train_ranker import per_user_auc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
META = ROOT/'data/raw/KuaiRand-1K/data/video_features_basic_1k.csv'
DAY = 86_400_000; TAUS = [.5, 2.]; PRIOR = 20.; GRID = [1e-4, 1e-3, 1e-2]
FOLLOWUP = ROOT/'artifacts/ials_time_followup/20261019T192838Z'; IALS = 'time_tau0.5_alpha30_l2100_s{}.pt'

def load_meta():
    m = pd.read_csv(META, usecols=['video_id','author_id','tag','upload_dt','video_type','upload_type','music_type'])
    m['tag1'] = m.tag.astype(str).str.split(',').str[0].where(m.tag.notna(), 'missing')
    m['upload_day'] = pd.to_datetime(m.upload_dt, errors='coerce')
    m['music_type'] = m.music_type.astype(str).where(m.music_type.notna(), 'missing')
    return m.drop(columns=['tag', 'upload_dt']).set_index('video_id')

def load_frame(name, meta):
    f = pd.read_csv(DATA/f'{name}.csv.gz', usecols=['user_id','video_id','time_ms','long_view','duration_ms','tab'])
    f = f.rename(columns={'time_ms': 't', 'long_view': 'lv'}).join(meta, on='video_id')
    f['const'] = 0; f['lv_logdur'] = f.lv*np.log1p(f.duration_ms/1000); f['lv_short'] = f.lv*(f.duration_ms <= 18000)
    return f

KEYS = {'g': ['const'], 'v': ['video_id'], 'a': ['author_id'], 'tg': ['tag1'], 'u': ['user_id'],
        'ua': ['user_id', 'author_id'], 'ut': ['user_id', 'tag1']}
DECAY = {'v', 'a'}; SUMS = {'u': ['lv_logdur', 'lv_short']}

def group_table(ev, key, T0):
    cols = KEYS[key]; extra = SUMS.get(key, [])
    agg = {'n': ('lv', 'size'), 'l': ('lv', 'sum'), **{c: (c, 'sum') for c in extra}}
    g = ev.groupby(cols+['t'], sort=True).agg(**agg).reset_index()
    if key in DECAY:
        for tau in TAUS: g[f'w{tau:g}'] = g.l*np.exp((g.t-T0)/(tau*DAY))
        g['lastlv'] = np.where(g.l > 0, g.t, np.nan)
    return g

def prequential(ev, key, T0):
    """Statistics strictly before each row's timestamp, aligned to ev rows."""
    cols = KEYS[key]; g = group_table(ev, key, T0); grp = g.groupby(cols, sort=False)
    out = g[cols+['t']].copy()
    for c in ['n', 'l']+SUMS.get(key, []): out[f'{key}_{c}'] = grp[c].cumsum()-g[c]
    if key in DECAY:
        for tau in TAUS: out[f'{key}_d{tau:g}'] = (grp[f'w{tau:g}'].cumsum()-g[f'w{tau:g}'])*np.exp(-(g.t-T0)/(tau*DAY))
        g['prev'] = grp.lastlv.shift(); out[f'{key}_lastlv'] = g.groupby(cols, sort=False).prev.ffill()
    return ev[cols+['t']].merge(out, on=cols+['t'], how='left').drop(columns=cols+['t']).set_index(ev.index)

def frozen(ev, key, T0):
    """Final statistics at the cutoff, one row per key."""
    cols = KEYS[key]; g = group_table(ev, key, T0)
    agg = {f'{key}_n': ('n', 'sum'), f'{key}_l': ('l', 'sum'), **{f'{key}_{c}': (c, 'sum') for c in SUMS.get(key, [])}}
    if key in DECAY: agg.update({f'{key}_E{tau:g}': (f'w{tau:g}', 'sum') for tau in TAUS}); agg[f'{key}_lastlv'] = ('lastlv', 'max')
    return g.groupby(cols).agg(**agg).reset_index()

def query(frame, tables, T0):
    """Frozen-at-T statistics for evaluation rows; unseen keys get zero counts and missing last-long-view."""
    out = pd.DataFrame(index=frame.index)
    for key, tab in tables.items():
        cols = KEYS[key]; m = frame[cols].merge(tab, on=cols, how='left').set_index(frame.index)
        for c in tab.columns:
            if c in cols: continue
            if c.endswith('lastlv'): out[c] = m[c]
            elif '_E' in c:
                tau = float(c.split('_E')[1]); out[c.replace('_E', '_d')] = m[c].fillna(0)*np.exp(-(frame.t-T0)/(tau*DAY))
            else: out[c] = m[c].fillna(0)
    return out

def assemble(f, s, level, levels=None):
    """Feature matrix from rows f and statistics s (identical code for training and evaluation rows)."""
    p0 = np.where(s.g_n > 0, s.g_l/np.maximum(s.g_n, 1), .25)
    smooth = lambda l, n: (l+PRIOR*p0)/(n+PRIOR)
    X = {'log_duration': np.log1p(f.duration_ms/1000), 'short_le_18s': (f.duration_ms <= 18000).astype(float), 'zero_duration': (f.duration_ms == 0).astype(float)}
    day = pd.to_datetime(f.t, unit='ms', utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None).dt.normalize()
    age = (day-f.upload_day).dt.days
    X['log_video_age'] = np.log1p(age.clip(lower=0).fillna(0)); X['video_age_missing'] = age.isna().astype(float)
    X['is_ad'] = (f.video_type == 'AD').astype(float)
    for k in ['v', 'a']:
        X[f'{k}_log_exposures'] = np.log1p(s[f'{k}_n']); X[f'{k}_log_long_views'] = np.log1p(s[f'{k}_l']); X[f'{k}_rate'] = smooth(s[f'{k}_l'], s[f'{k}_n'])
        for tau in TAUS: X[f'{k}_log_decay{tau:g}'] = np.log1p(s[f'{k}_d{tau:g}'])
        hrs = (f.t-s[f'{k}_lastlv'])/3_600_000
        X[f'{k}_log_hours_since_long_view'] = np.log1p(hrs.fillna(0).clip(lower=0)); X[f'{k}_no_long_view_yet'] = hrs.isna().astype(float)
    X['tag_log_exposures'] = np.log1p(s.tg_n); X['tag_rate'] = smooth(s.tg_l, s.tg_n)
    if level == 'L2':
        X['u_log_exposures'] = np.log1p(s.u_n); X['u_rate'] = smooth(s.u_l, s.u_n)
        X['ua_log_exposures'] = np.log1p(s.ua_n); X['ua_log_long_views'] = np.log1p(s.ua_l); X['ua_rate'] = smooth(s.ua_l, s.ua_n)
        X['ut_log_long_views'] = np.log1p(s.ut_l); X['ut_share_of_user_long_views'] = s.ut_l/np.maximum(s.u_l, 1); X['ut_rate'] = smooth(s.ut_l, s.ut_n)
        mean_dur = s.u_lv_logdur/np.maximum(s.u_l, 1)
        X['duration_gap_to_user_mean'] = np.where(s.u_l > 0, np.abs(X['log_duration']-mean_dur), 0.); X['user_has_no_long_view'] = (s.u_l == 0).astype(float)
        X['user_short_long_view_share'] = s.u_lv_short/np.maximum(s.u_l, 1)
    X = pd.DataFrame(X, index=f.index).astype(float)
    if levels is None:
        levels = {'upload_type': [v for v, c in f.upload_type.value_counts().items() if c >= 1000],
                  'music_type': sorted(f.music_type.unique().tolist()), 'tag1': f.tag1.value_counts().index[:50].tolist()}
    for col, lv in levels.items():
        v = f[col].astype(str).where(f[col].notna(), 'missing')
        v = v.where(v.isin(lv) | (v == 'missing'), 'other')
        for level_value in lv+['other', 'missing']: X[f'{col}={level_value}'] = (v == level_value).astype(float)
    return X, levels

def fit_lr(X, y, l2):
    w = torch.zeros(X.shape[1], dtype=torch.float32, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    with torch.no_grad(): b.fill_(float(np.log(y.mean()/(1-y.mean()))))
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, history_size=20, line_search_fn='strong_wolfe', tolerance_grad=1e-7, tolerance_change=1e-12)
    def closure():
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(X@w+b, y)+l2*w.pow(2).sum(); loss.backward(); return loss
    opt.step(closure); final = float(closure()); grad = float(torch.cat([w.grad, b.grad]).norm())
    return w.detach(), b.detach(), final, grad

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True); parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(8); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = load_meta(); train = load_frame('train', meta)
    if args.smoke: train = train.loc[train.user_id < 60].reset_index(drop=True)
    T0, T = int(train.t.min()), int(train.t.max())
    pre = pd.concat([prequential(train, k, T0) for k in KEYS], axis=1)
    tables = {k: frozen(train, k, T0) for k in KEYS}
    users = np.sort(pd.read_csv(DATA/'training_users.csv').user_id.to_numpy()); catalog = np.sort(pd.read_csv(DATA/'candidate_videos.csv').video_id.to_numpy())
    val = load_frame('validation', meta); rnd = load_frame('random_validation', meta)
    assert int(val.t.min()) > T and int(rnd.t.min()) > T
    rnd = rnd.loc[rnd.user_id.isin(users)].reset_index(drop=True); rnd['in_catalog'] = rnd.video_id.isin(catalog)
    order = np.random.default_rng(2028); groups_users = np.sort(val.loc[val.lv.eq(1) & val.user_id.isin(users) & val.video_id.isin(catalog), 'user_id'].unique())
    perm = order.permutation(len(groups_users)); tune_users = set(groups_users[perm[:len(groups_users)//2]]); report_users = set(groups_users[perm[len(groups_users)//2:]])
    val_tune = val.loc[val.user_id.isin(tune_users)].reset_index(drop=True); val_report = val.loc[val.user_id.isin(report_users)].reset_index(drop=True)
    s_tune, s_report, s_rnd = query(val_tune, tables, T0), query(val_report, tables, T0), query(rnd, tables, T0)
    results = {}; preds = {}; deg = train.loc[train.lv.eq(1)].drop_duplicates(['user_id', 'video_id']).video_id.value_counts()
    for level in ['L1', 'L2']:
        Xtr, levels = assemble(train, pre, level); mean, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        prep = lambda X: torch.tensor(((X[Xtr.columns]-mean)/sd).to_numpy(np.float32))
        Xt = prep(Xtr); y = torch.tensor(train.lv.to_numpy(np.float32))
        Xv = prep(assemble(val_tune, s_tune, level, levels)[0]); yv = torch.tensor(val_tune.lv.to_numpy(np.float32))
        fits = []
        for l2 in GRID:
            w, b, obj, grad = fit_lr(Xt, y, l2)
            with torch.no_grad(): ll = float(F.binary_cross_entropy_with_logits(Xv@w+b, yv))
            fits.append({'l2': l2, 'train_objective': obj, 'gradient_norm': grad, 'tune_log_loss': ll, 'w': w, 'b': b})
            print(f'{level} l2={l2:g}: train objective {obj:.6f}, grad {grad:.2e}, tune log loss {ll:.6f}', flush=True)
        best = min(fits, key=lambda r: r['tune_log_loss'])
        coef = pd.DataFrame({'feature': list(Xtr.columns), 'coefficient_standardized': best['w'].numpy(), 'train_mean': mean.to_numpy(), 'train_sd': sd.to_numpy()})
        coef.to_csv(out/f'coefficients_{level}.csv', index=False)
        torch.save({'w': best['w'], 'b': best['b'], 'columns': list(Xtr.columns), 'mean': mean, 'sd': sd, 'levels': levels, 'l2': best['l2']}, out/f'model_{level}.pt')
        with torch.no_grad():
            preds[(level, 'random')] = (prep(assemble(rnd, s_rnd, level, levels)[0])@best['w']+best['b']).numpy()
            preds[(level, 'report')] = (prep(assemble(val_report, s_report, level, levels)[0])@best['w']+best['b']).numpy()
        results[level] = {'grid': [{k: v for k, v in r.items() if k not in ('w', 'b')} for r in fits], 'selected_l2': best['l2'], 'features': len(Xtr.columns)}
        del Xt, Xtr
    # iALS reference scores (frozen at T), five seeds; videos outside its catalog score 0.
    ials_rnd, ials_rep = [], []
    for seed in [42, 1, 2, 3, 4]:
        fct = torch.load(FOLLOWUP/IALS.format(seed), weights_only=False)
        def score(frame):
            u = torch.from_numpy(np.searchsorted(users, frame.user_id.to_numpy())); vi = np.searchsorted(catalog, frame.video_id.to_numpy())
            inside = (vi < len(catalog)) & (catalog[np.minimum(vi, len(catalog)-1)] == frame.video_id.to_numpy())
            s = np.zeros(len(frame)); idx = np.flatnonzero(inside)
            s[idx] = (fct['X'][u[idx]]*fct['Y'][torch.from_numpy(vi[idx])]).sum(-1).numpy(); return s
        ials_rnd.append(score(rnd)); ials_rep.append(score(val_report))
    rng = np.random.default_rng(2026)
    def user_auc(frame, s): return per_user_auc(frame.user_id.to_numpy(), frame.lv.to_numpy(), s)
    def paired(a, b):
        common = a.index.intersection(b.index); d = (a.loc[common]-b.loc[common]).to_numpy()
        draws = rng.choice(d, size=(2000, len(d)), replace=True).mean(axis=1)
        return {'users': len(d), 'a': float(a.loc[common].mean()), 'b': float(b.loc[common].mean()), 'difference': float(d.mean()),
                'ci95': np.quantile(draws, [.025, .975]).tolist()}
    A = rnd.in_catalog.to_numpy(); rA = rnd.loc[A]
    aucA = {'L1': user_auc(rA, preds[('L1', 'random')][A]), 'L2': user_auc(rA, preds[('L2', 'random')][A]),
            'iALS': pd.concat([user_auc(rA, s[A]) for s in ials_rnd], axis=1).mean(axis=1)}
    aucB = {'L1': user_auc(rnd, preds[('L1', 'random')]), 'L2': user_auc(rnd, preds[('L2', 'random')])}
    comparisons = {'scope_A_primary_L2_minus_iALS': paired(aucA['L2'], aucA['iALS']), 'scope_A_key_ablation_L2_minus_L1': paired(aucA['L2'], aucA['L1']),
                   'scope_A_L1_minus_iALS': paired(aucA['L1'], aucA['iALS']), 'scope_B_L2_minus_L1': paired(aucB['L2'], aucB['L1'])}
    # Pooled AUC by video group with user-cluster bootstrap.
    rnd['degree'] = rnd.video_id.map(deg).fillna(0)
    group = np.select([~rnd.in_catalog, rnd.degree.eq(0), rnd.degree.le(2)], ['outside_catalog', 'degree_0', 'degree_1_2'], 'degree_ge_3')
    from train_ranker import auc as pooled_auc
    ials_mean = np.mean(ials_rnd, axis=0); pooled = {}
    for g in ['outside_catalog', 'degree_0', 'degree_1_2', 'degree_ge_3']:
        m = group == g; sub = rnd.loc[m]; uniq = sub.user_id.unique(); row = {'exposures': int(m.sum()), 'positives': int(sub.lv.sum()), 'users': len(uniq)}
        scores = {'L1': preds[('L1', 'random')][m], 'L2': preds[('L2', 'random')][m]}
        if g != 'outside_catalog': scores['iALS_seed_mean_score'] = ials_mean[m]
        for name, s in scores.items(): row[f'{name}_auc'] = pooled_auc(sub.lv.to_numpy(), s)
        pairs = [('L2', 'L1')]+([('L2', 'iALS_seed_mean_score'), ('L1', 'iALS_seed_mean_score')] if g != 'outside_catalog' else [])
        uid = sub.user_id.to_numpy(); yv = sub.lv.to_numpy(); index = {u: np.flatnonzero(uid == u) for u in uniq}
        for a, b in pairs:
            draws = []
            for _ in range(2000):
                pick = np.concatenate([index[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
                if yv[pick].min() == yv[pick].max(): continue
                draws.append(pooled_auc(yv[pick], scores[a][pick])-pooled_auc(yv[pick], scores[b][pick]))
            row[f'{a}_minus_{b}'] = {'difference': row[f'{a}_auc']-row[f'{b}_auc'], 'ci95': np.quantile(draws, [.025, .975]).tolist()}
        pooled[g] = row
    from evaluate_ranking_sessions import group_auc
    local = pd.to_datetime(val_report.t, unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
    sessions = pd.factorize(pd.MultiIndex.from_arrays([val_report.user_id, local.dt.strftime('%Y%m%d'), local.dt.hour, val_report.tab]))[0]
    owner = pd.Series(val_report.user_id.to_numpy(), index=sessions).groupby(level=0).first()
    def session_gauc(s):
        a, _ = group_auc(sessions, val_report.lv.to_numpy(), s); return float(a.groupby(owner.loc[a.index].to_numpy()).mean().mean())
    context = {'L1': session_gauc(preds[('L1', 'report')]), 'L2': session_gauc(preds[('L2', 'report')]),
               'iALS_seed_mean_score_outside_catalog_0': session_gauc(np.mean(ials_rep, axis=0))}
    summary = {'protocol': 'reports/feature_model/protocol_confirmed.json', 'smoke': args.smoke, 'fits': results,
               'scope_A': {'exposures': int(A.sum()), 'auc_users': {k: len(v) for k, v in aucA.items()}, 'mean_user_auc': {k: float(v.mean()) for k, v in aucA.items()}},
               'scope_B': {'exposures': len(rnd), 'auc_users': {k: len(v) for k, v in aucB.items()}, 'mean_user_auc': {k: float(v.mean()) for k, v in aucB.items()}},
               'comparisons': comparisons, 'pooled_auc_by_video_group': pooled, 'standard_validation_report_session_gauc': context,
               'finished_utc': datetime.now(timezone.utc).isoformat(), 'test_accessed': False}
    (out/'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    np.savez_compressed(out/'random_validation_scores.npz', user_id=rnd.user_id.to_numpy(), video_id=rnd.video_id.to_numpy(), lv=rnd.lv.to_numpy(),
                        L1=preds[('L1', 'random')], L2=preds[('L2', 'random')], iALS=np.array(ials_rnd))
    print(json.dumps(summary, indent=2, default=float))

if __name__ == '__main__':
    main()
