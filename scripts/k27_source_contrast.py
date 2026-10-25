"""Extension analysis (reports/expansion/protocol_extension_source.json): same week, same features, same rows and fitting
procedure; only the label source differs (recommended vs random exposures).

extract   stream development users' week-1 standard (recommended) exposures from the 27K tar, keep the rows of the
          adaptation users (main population with both outcomes in week-1 random exposures), apply the 1K validity rules,
          restrict to scope A (video with pre-cutoff exposure), and sample per (user, Shanghai date) cell exactly the
          number of rows the random refit set has in that cell (uniform keys, smallest kept; cells with fewer logged rows
          contribute all of them and the shortfall is recorded, never filled from elsewhere)
fit       M2 form (LR-L2 standardized columns, lambda grid, selection on report-group week-1 random AUC, refit on all
          target users) for: logged-week1, random-week1 (= the published M2 refit), and the pool variant (both sets
          restricted to random-pool videos, cell-matched to the logged pool rows); test on development week 2.
Writes run_dev/ext_source.json; refuses to run the test twice.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import build_event_store as bes
import feature_model as fm
import feature_matrix as fx
import k27_adapt as ka
import k27_eval as ke
import k27_eval_rows as er

ROOT = Path(__file__).resolve().parents[1]; ART = ROOT/'artifacts/k27_experiment'; RUN = ROOT/'reports/expansion/run_dev'; FEAT = ROOT/'data/k27/features_dev_8m'
TAR = ROOT/'data/raw/k27_download/KuaiRand-27K.tar.gz'; OUTD = ROOT/'data/k27/source_contrast'; FIRST, LAST = 20220422, 20220428; GRID = [1e-3, 1e-2, 1e-1]; TIE = 5e-4

def cells(frame):
    return frame.user_id.to_numpy().astype(np.int64)*100 + (bes.shanghai_date(frame.time_ms.to_numpy())-20220400)

def adaptation_rows():
    """The random refit rows (development, week 1, main population, both outcomes, scope A) with their cells."""
    e = pd.read_pickle(ART/'eval/rows.pkl'); main_m = (e.user_has_long_view & e.scope_A).to_numpy(); both = e.user_id.isin(ka.both_outcome_users(e, main_m)).to_numpy()
    r = e.loc[main_m & both].reset_index(drop=True); r = r.rename(columns={'t': 'time_ms', 'lv': 'long_view'}); r['cell'] = cells(r); return r

def stage_extract():
    OUTD.mkdir(parents=True, exist_ok=True); rnd = adaptation_rows(); quota = rnd.groupby('cell').size(); users = np.sort(rnd.user_id.unique())
    exposed = np.load(ROOT/'artifacts/serving_bundle/v1/stats_v.npz')['keys'] if (ROOT/'artifacts/serving_bundle/v1/stats_v.npz').exists() else np.sort(pd.read_pickle(FEAT/'frozen_v.pkl').video_id.to_numpy())
    rng = np.random.default_rng(20261023); kept = None; counts = {'rows_read': 0, 'rows_target_users_window_valid': 0, 'rows_scope_A': 0}; avail = pd.Series(dtype=np.int64)
    for name, fh in bes.stream_members(TAR, r'log_standard_4_22_to_5_08_27k_part\d\.csv$'):
        for chunk in pd.read_csv(fh, chunksize=bes.CHUNK):
            counts['rows_read'] += len(chunk); chunk = chunk.loc[chunk.user_id.isin(users) & chunk.is_rand.eq(0)]
            if len(chunk) == 0: continue
            invalid = chunk.isna().any(axis=1).to_numpy().copy()
            for c in bes.BINARY: invalid |= ~chunk[c].isin([0, 1]).to_numpy()
            invalid |= (chunk[bes.NONNEG] < 0).any(axis=1).to_numpy(); invalid |= ~chunk.tab.between(0, 14).to_numpy(); invalid |= ~chunk.date.between(20220408, 20220508).to_numpy()
            chunk = chunk.loc[~invalid]; d = bes.shanghai_date(chunk.time_ms.to_numpy()); chunk = chunk.loc[(d >= FIRST) & (d <= LAST)]
            counts['rows_target_users_window_valid'] += len(chunk); v = chunk.video_id.to_numpy(np.int64); i = np.minimum(np.searchsorted(exposed, v), len(exposed)-1); chunk = chunk.loc[exposed[i] == v]
            counts['rows_scope_A'] += len(chunk)
            if len(chunk) == 0: continue
            chunk = chunk[list(bes.COLUMNS)].copy(); chunk['cell'] = cells(chunk); chunk = chunk.loc[chunk.cell.isin(quota.index)]; chunk['key'] = rng.random(len(chunk))
            avail = avail.add(chunk.groupby('cell').size(), fill_value=0)
            pool = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True); pool = pool.sort_values(['cell', 'key'], kind='stable')
            rank = pool.groupby('cell').cumcount(); kept = pool.loc[rank.to_numpy() < quota.reindex(pool.cell).to_numpy()].reset_index(drop=True)
        print(name, counts, 'kept', 0 if kept is None else len(kept), flush=True)
    kept = kept.drop_duplicates(subset=list(bes.COLUMNS)).drop(columns=['key']).reset_index(drop=True); got = kept.groupby('cell').size().reindex(quota.index, fill_value=0)
    short = (quota-got).clip(lower=0); info = {**counts, 'quota_rows': int(quota.sum()), 'quota_cells': int(len(quota)), 'kept_rows': int(len(kept)), 'cells_short': int((short > 0).sum()), 'shortfall_rows': int(short.sum()),
                                             'cells_with_zero_logged_rows': int((got == 0).sum()), 'rule': 'cells with fewer logged rows than quota contribute all of them; nothing is filled from other cells or users'}
    kept.to_pickle(OUTD/'logged_week1_sample.pkl'); (OUTD/'logged_week1_sample.json').write_text(json.dumps(info, indent=2)); print(json.dumps(info, indent=2)); return info

def features(raw, lr, tables, T0, vocab):
    f = er.attach_video(raw.reset_index(drop=True)); s = er.history(f, tables, T0, tag_codes={t: i for i, t in enumerate(vocab)})
    X, _ = fx.standardized(f, s, 'L2', lr['levels'], stats=(lr['columns'], np.asarray(lr['mean'], float), np.asarray(lr['sd'], float))); return f, X.astype(np.float64)

def fit_m2(X, y, tune_mask, all_mask, e_sel, X_sel_fn):
    """Grid on the tune-user rows, selection on report-group week-1 random AUC, refit on all rows."""
    rows = []
    for l2 in GRID:
        w, b, obj, g = ka.fit_ridge(X[tune_mask], y[tune_mask], l2); rows.append({'l2': l2, 'objective': obj, 'grad_norm': g, 'train_rows': int(tune_mask.sum()), 'report_week1_auc': float(ke.user_auc(e_sel, X_sel_fn()@w+b, np.ones(len(e_sel), bool)).mean()), 'w': w, 'b': b})
    best = max(r['report_week1_auc'] for r in rows); chosen = sorted([r for r in rows if r['report_week1_auc'] >= best-TIE], key=lambda r: -r['l2'])[0]
    w, b, obj, g = ka.fit_ridge(X[all_mask], y[all_mask], chosen['l2']); return {'grid': [{k: v for k, v in r.items() if k not in ('w', 'b')} for r in rows], 'selected_l2': chosen['l2'], 'refit_rows': int(all_mask.sum()), 'w': w, 'b': b}

def stage_fit():
    dest = RUN/'ext_source.json'; assert not dest.exists(), 'source contrast already run'
    lr, ials, hyb, _ = ka.frozen_models(); T0 = json.loads((FEAT/'meta.json').read_text())['T0']; vocab = json.loads((er.VMETA/'meta.json').read_text())['vocab']['tag1']
    tables = {k: pd.read_pickle(FEAT/f'frozen_{k}.pkl') for k in fm.KEYS}; g = pd.read_csv(ROOT/'data/splits/development_tune_report.csv').set_index('pure_user_id').group
    rnd = adaptation_rows(); logged = pd.read_pickle(OUTD/'logged_week1_sample.pkl'); info = json.loads((OUTD/'logged_week1_sample.json').read_text())
    pool_ids = pd.read_csv(ROOT/'data/raw/pure_download/KuaiRand-Pure/data/video_features_basic_pure.csv', usecols=['video_id']).video_id.to_numpy()
    lp = pd.read_pickle(ROOT/'data/k27/dev_week1_standard_pool.pkl'); lp = lp.loc[lp.user_id.isin(rnd.user_id.unique())].copy(); lp['cell'] = cells(lp)
    # pool variant: random rows cell-matched to the logged pool rows
    quota_p = lp.groupby('cell').size(); rp = rnd.loc[rnd.cell.isin(quota_p.index)].copy(); rp['key'] = np.random.default_rng(5).random(len(rp)); rp = rp.sort_values(['cell', 'key'], kind='stable')
    rp = rp.loc[rp.groupby('cell').cumcount().to_numpy() < quota_p.reindex(rp.cell).to_numpy()].drop(columns=['key']); got_p = rp.groupby('cell').size().reindex(quota_p.index, fill_value=0)
    e = pd.read_pickle(ART/'eval/rows.pkl'); se = pd.read_pickle(ART/'eval/history.pkl'); sel_m = (e.group.eq('report') & e.user_has_long_view & e.scope_A).to_numpy()
    sel_m = sel_m & e.user_id.isin(ka.both_outcome_users(e, (e.user_has_long_view & e.scope_A).to_numpy())).to_numpy(); e_sel, s_sel = e.loc[sel_m].reset_index(drop=True), se.loc[sel_m].reset_index(drop=True)
    X_sel, _ = fx.standardized(e_sel, s_sel, 'L2', lr['levels'], stats=(lr['columns'], np.asarray(lr['mean'], float), np.asarray(lr['sd'], float))); X_sel = X_sel.astype(np.float64)
    f2 = pd.read_pickle(ART/'eval_week2/rows.pkl'); s2 = pd.read_pickle(ART/'eval_week2/history.pkl'); X2, _ = fx.standardized(f2, s2, 'L2', lr['levels'], stats=(lr['columns'], np.asarray(lr['mean'], float), np.asarray(lr['sd'], float))); X2 = X2.astype(np.float64)
    main2 = f2.user_has_long_view.to_numpy(); A2 = f2.scope_A.to_numpy(); results, models = {}, {}
    sets = {'logged_week1': logged, 'random_week1': rnd[list(bes.COLUMNS)+['cell']], 'logged_week1_pool': lp[list(bes.COLUMNS)+['cell']], 'random_week1_pool': rp[list(bes.COLUMNS)+['cell']]}
    for name, raw in sets.items():
        f, X = features(raw, lr, tables, T0, vocab); y = raw.long_view.to_numpy(float); tune = g.reindex(raw.user_id).eq('tune').to_numpy()
        m = fit_m2(X, y, tune, np.ones(len(raw), bool), e_sel, lambda: X_sel); models[name] = m
        results[name] = {'rows': int(len(raw)), 'users': int(raw.user_id.nunique()), 'positive_rate': float(y.mean()), 'selected_l2': m['selected_l2'], 'grid': m['grid'], 'refit_rows': m['refit_rows'], 'scope_A_share': float(f.video_id.isin(tables['v'].video_id).mean())}
    scores = {name: [X2@m['w']+m['b']] for name, m in models.items()}; inp2 = ka.inputs(f2, s2, lr, ials, hyb); scores['LR-L2_6M_logged'] = [inp2['logit']]; scores['iALS_tuned'] = inp2['ials_raw']
    A = joblib.load(ka.ADAPT/'adapted_models.joblib'); m1 = (np.array(A['m1_stats'][0]), np.array(A['m1_stats'][1])); scores['M2_published'] = [ka.design('M2_L2_refit', inp2, 0, m1)@A['final']['M2_L2_refit']['w']+A['final']['M2_L2_refit']['b']]
    aucA = {k: ke.user_auc(f2, v, main2 & A2) for k, v in scores.items()}; C = lambda a, b: ke.paired(aucA[a], aucA[b])
    out = {'protocol': 'reports/expansion/protocol_extension_source.json', 'evidence_level': 'extension analysis (development week 2 already viewed for earlier models)', 'extraction': info,
           'pool_variant': {'quota_cells': int(len(quota_p)), 'quota_rows': int(quota_p.sum()), 'random_pool_rows_kept': int(len(rp)), 'random_cells_short': int(((quota_p-got_p).clip(lower=0) > 0).sum()), 'random_shortfall_rows': int((quota_p-got_p).clip(lower=0).sum())},
           'training_sets': results, 'week2': {'users': int(len(aucA['logged_week1'])), 'exposures': int((main2 & A2).sum()), 'mean_user_auc': {k: float(v.mean()) for k, v in aucA.items()},
                     'random_week1_equals_published_M2': bool(abs(float(aucA['random_week1'].mean())-float(aucA['M2_published'].mean())) < 1e-6),
                     'contrasts': {'random_minus_logged_week1': C('random_week1', 'logged_week1'), 'logged_week1_minus_LRL2_6M': C('logged_week1', 'LR-L2_6M_logged'), 'random_week1_minus_LRL2_6M': C('random_week1', 'LR-L2_6M_logged'),
                                   'logged_week1_minus_iALS_tuned': C('logged_week1', 'iALS_tuned'), 'random_week1_minus_iALS_tuned': C('random_week1', 'iALS_tuned'),
                                   'pool_random_minus_logged': C('random_week1_pool', 'logged_week1_pool'), 'pool_logged_minus_LRL2_6M': C('logged_week1_pool', 'LR-L2_6M_logged'), 'pool_random_minus_LRL2_6M': C('random_week1_pool', 'LR-L2_6M_logged')}},
           'reading': 'under identical features, cell-matched rows and procedure, the random-minus-logged difference is attributed to the label source (exposure mechanism and the label distribution it induces); controlled comparison, not a causal identification',
           'finished_utc': datetime.now(timezone.utc).isoformat()}
    joblib.dump({k: {kk: vv for kk, vv in m.items() if kk != 'grid'} for k, m in models.items()}, ka.ADAPT/'extension_source_models.joblib'); dest.write_text(json.dumps(out, indent=2, default=float)); print(json.dumps(out, indent=2, default=float))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['extract', 'fit']); a = p.parse_args(); {'extract': stage_extract, 'fit': stage_fit}[a.stage]()
