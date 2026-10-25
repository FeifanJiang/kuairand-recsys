"""Random-label adaptation stage (reports/expansion/protocol_adaptation.json, part B).

rows   development users' week-2 random exposures (04-29..05-08) with frozen-at-cutoff history, saved for the test;
       no metric is computed here
fit    12 ridge logistic fits on tune-group week-1 rows (forms M1-M4 x lambda grid), selection on report-group week-1
       per-user AUC (iALS seed 0 inputs), refit of each form's selected lambda on all development week-1 rows ->
       frozen adapted models (artifacts/k27_experiment/adapt); writes run_dev/selection_adapt.json
test   week-2 evaluation of the frozen adapted models and the frozen baselines; runs once (run_dev/adapt_week2.json)
All inputs are frozen at the training cutoff; z-scores use the hybrid's tune-group week-1 reference constants.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import build_event_store as bes
import feature_model as fm
import feature_matrix as fx
import k27_eval_rows as er
import k27_eval as ke
import k27_deep as kd
import k27_hybrid as kh
from train_ranker import auc as pooled_auc

ROOT = Path(__file__).resolve().parents[1]; ART = ROOT/'artifacts/k27_experiment'; FEAT = ROOT/'data/k27/features_dev_8m'
RUN = ROOT/'reports/expansion/run_dev'; ADAPT = ART/'adapt'
RANDOM_LOG = ROOT/'data/raw/pure_download/KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv'
FORMS = {'M1_blend': [1e-4, 1e-3, 1e-2], 'M2_L2_refit': [1e-3, 1e-2, 1e-1], 'M3_L2_iALS_refit': [1e-3, 1e-2, 1e-1], 'M4_L2_iALS_prior': [1e-3, 1e-2, 1e-1]}
ORDER = list(FORMS); TIE = 5e-4; M1_INPUTS = ['z_LR', 'z_iALS', 'has_factors', 'log1p_v_long_views', 'log1p_v_exposures']

def frozen_models():
    sel = json.loads((RUN/'selection_final.json').read_text()); lr = torch.load(ART/f"{sel['LR-L2']['name']}.pt", weights_only=False)
    base = sel['iALS_tuned']['name'].removeprefix('iALS_').rsplit('_s', 1)[0]; ials = [torch.load(ART/'ials'/f'{base}_s{k}.pt', weights_only=False) for k in range(5)]
    hyb = json.loads((RUN/'hybrid.json').read_text()); trees = [joblib.load(ART/f"{sel['Tree-L2']['name'].rsplit('_s', 1)[0]}_s{k}.joblib") for k in range(3)]
    return lr, ials, hyb, trees

def inputs(f, s, lr, ials, hyb):
    X, _ = fx.standardized(f, s, 'L2', lr['levels'], stats=(lr['columns'], np.asarray(lr['mean'], float), np.asarray(lr['sd'], float)))
    with torch.no_grad(): logit = (torch.from_numpy(X)@lr['w'].float()+lr['b'].float()).numpy().astype(np.float64)
    c_lr = hyb['standardization']['LR-L2_mean_sd']; c_i = hyb['standardization']['iALS_mean_sd_per_seed']; raw, z_i, flag = [], [], None
    for k, fct in enumerate(ials):
        sc, fl = kd.ials_inputs(f, fct); raw.append(sc); z_i.append((sc-c_i[k][0])/c_i[k][1]); flag = fl if flag is None else flag
    return {'X': X.astype(np.float64), 'logit': logit, 'z_lr': (logit-c_lr[0])/c_lr[1], 'z_i': z_i, 'ials_raw': raw, 'flag': flag.astype(np.float64),
            'log_v_l': np.log1p(s.v_l.to_numpy(float)), 'log_v_n': np.log1p(s.v_n.to_numpy(float)), 'c_lr': c_lr, 'c_i': c_i}

def design(form, inp, k, m1_stats=None):
    if form == 'M1_blend':
        D = np.stack([inp['z_lr'], inp['z_i'][k], inp['flag'], inp['log_v_l'], inp['log_v_n']], 1)
        return D if m1_stats is None else (D-m1_stats[0])/m1_stats[1]
    if form == 'M2_L2_refit': return inp['X']
    return np.concatenate([inp['X'], np.stack([inp['z_i'][k], inp['flag']], 1)], 1)

def fit_ridge(D, y, l2, w0=None):
    """Mean log loss + l2 * ||w - w0||^2 (w0 = 0 unless given), intercept free; full-batch L-BFGS in float64."""
    X = torch.tensor(D, dtype=torch.float64); yt = torch.tensor(y, dtype=torch.float64)
    w0 = torch.zeros(X.shape[1], dtype=torch.float64) if w0 is None else torch.tensor(w0, dtype=torch.float64)
    w = w0.clone().requires_grad_(True); b = torch.tensor([float(np.log(y.mean()/(1-y.mean())))], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=1000, history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-10, tolerance_change=1e-15)
    def closure():
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(X@w+b, yt)+l2*(w-w0).pow(2).sum(); loss.backward(); return loss
    opt.step(closure); obj = float(closure()); g = float(torch.cat([w.grad, b.grad]).norm())
    return w.detach().numpy(), float(b), obj, g

def both_outcome_users(e, mask):
    g = e.loc[mask].groupby('user_id').lv.agg(['min', 'max']); return g.index[(g['min'] == 0) & (g['max'] == 1)]

def stage_rows():
    part = pd.read_csv(ROOT/'data/splits/development_tune_report.csv'); users = np.sort(part.pure_user_id.to_numpy())
    raw, counts = er.random_rows(RANDOM_LOG, users, 20220429, 20220508); f = er.attach_video(raw)
    T0 = json.loads((FEAT/'meta.json').read_text())['T0']; vocab = json.loads((er.VMETA/'meta.json').read_text())['vocab']['tag1']
    tables = {k: pd.read_pickle(FEAT/f'frozen_{k}.pkl') for k in fm.KEYS}; s = er.history(f, tables, T0, tag_codes={t: i for i, t in enumerate(vocab)})
    f['group'] = f.user_id.map(part.set_index('pure_user_id').group); f['scope_A'] = s.v_n.gt(0).to_numpy()
    f['user_has_long_view'] = f.user_id.isin(tables['u'].loc[tables['u'].u_l.gt(0), 'user_id']).to_numpy()
    assert bes.shanghai_date(f.t.to_numpy()).min() >= 20220429 and bes.shanghai_date(f.t.to_numpy()).max() <= 20220508
    d = ART/'eval_week2'; d.mkdir(parents=True, exist_ok=False); f.to_pickle(d/'rows.pkl'); s.to_pickle(d/'history.pkl')
    info = {'cleaning': counts, 'rows': len(f), 'users': int(f.user_id.nunique()), 'scope_A_rows': int(f.scope_A.sum()), 'note': 'no label aggregate computed at this stage'}
    (d/'info.json').write_text(json.dumps(info, indent=2)); return info

def stage_fit():
    lr, ials, hyb, _ = frozen_models(); e = pd.read_pickle(ART/'eval/rows.pkl'); se = pd.read_pickle(ART/'eval/history.pkl')
    main_m = (e.user_has_long_view & e.scope_A).to_numpy(); both = e.user_id.isin(both_outcome_users(e, main_m)).to_numpy()
    tr = e.group.eq('tune').to_numpy() & main_m & both; sel_m = e.group.eq('report').to_numpy() & main_m & both; alld = main_m & both
    inp = inputs(e, se, lr, ials, hyb); y = e.lv.to_numpy(float)
    D1 = design('M1_blend', inp, 0); m1_stats = (D1[tr].mean(0), D1[tr].std(0, ddof=1)); m1_stats[1][m1_stats[1] == 0] = 1
    w0 = np.concatenate([lr['w'].numpy().astype(np.float64), [0., 0.]]); fits, sel = {}, {}
    for form, grid in FORMS.items():
        rows = []
        for l2 in grid:
            D = design(form, inp, 0, m1_stats); w, b, obj, g = fit_ridge(D[tr], y[tr], l2, w0 if form == 'M4_L2_iALS_prior' else None)
            rows.append({'form': form, 'l2': l2, 'objective': obj, 'grad_norm': g, 'train_rows': int(tr.sum()), 'report_week1_auc': float(ke.user_auc(e, D@w+b, sel_m).mean()), 'w': w, 'b': b})
        best = max(r['report_week1_auc'] for r in rows); chosen = sorted([r for r in rows if r['report_week1_auc'] >= best-TIE], key=lambda r: -r['l2'])[0]
        fits[form] = rows; sel[form] = {'l2': chosen['l2'], 'report_week1_auc': chosen['report_week1_auc']}
    best = max(v['report_week1_auc'] for v in sel.values()); best_form = [f_ for f_ in ORDER if sel[f_]['report_week1_auc'] >= best-TIE][0]
    final = {}
    for form in FORMS:
        D = design(form, inp, 0, m1_stats); w, b, obj, g = fit_ridge(D[alld], y[alld], sel[form]['l2'], w0 if form == 'M4_L2_iALS_prior' else None)
        final[form] = {'w': w, 'b': b, 'l2': sel[form]['l2'], 'objective': obj, 'grad_norm': g, 'train_rows': int(alld.sum())}
    ADAPT.mkdir(parents=True, exist_ok=True)
    joblib.dump({'final': final, 'tune_fits': fits, 'm1_stats': [m1_stats[0].tolist(), m1_stats[1].tolist()], 'w0_M4': w0.tolist(), 'forms': FORMS, 'm1_inputs': M1_INPUTS}, ADAPT/'adapted_models.joblib')
    base = json.loads((RUN/'final.json').read_text())['result']['scope_A']['mean_user_auc']; hy = json.loads((RUN/'hybrid.json').read_text())['report_scope_A']['mean_user_auc']['hybrid']
    out = {'selection': sel, 'best_adapted_method': best_form, 'tie_rule': 'per form larger lambda within 0.0005; across forms simpler (M1 < M2 < M3 < M4) within 0.0005',
           'grid_results_week1_report_group': [{k: v for k, v in r.items() if k not in ('w', 'b')} for rows in fits.values() for r in rows],
           'week1_report_group_baselines_seed_avg': {'LR-L2': base['LR-L2'], 'iALS_tuned': base['iALS_tuned'], 'hybrid': hy, 'note': 'adapted-model week-1 figures above use iALS seed 0 and were trained on the tune group; selection evidence, reported separately'},
           'rows': {'tune_train': int(tr.sum()), 'report_select': int(sel_m.sum()), 'all_dev_refit': int(alld.sum())}, 'm1_inputs': M1_INPUTS, 'm1_stats': [m1_stats[0].tolist(), m1_stats[1].tolist()],
           'final_models': {k: {kk: vv for kk, vv in v.items() if kk not in ('w', 'b')} for k, v in final.items()}, 'coefficients_M1_final': dict(zip(M1_INPUTS, final['M1_blend']['w'].tolist())),
           'fitted_utc': datetime.now(timezone.utc).isoformat()}
    (RUN/'selection_adapt.json').write_text(json.dumps(out, indent=2, default=float)); return out

def stage_test():
    dest = RUN/'adapt_week2.json'; assert not dest.exists(), 'week-2 test already run'
    sel = json.loads((RUN/'selection_adapt.json').read_text()); A = joblib.load(ADAPT/'adapted_models.joblib'); m1 = (np.array(A['m1_stats'][0]), np.array(A['m1_stats'][1]))
    lr, ials, hyb, trees = frozen_models(); f = pd.read_pickle(ART/'eval_week2/rows.pkl'); s = pd.read_pickle(ART/'eval_week2/history.pkl'); inp = inputs(f, s, lr, ials, hyb)
    scores = {'LR-L2': [inp['logit']], 'iALS_tuned': inp['ials_raw'], 'hybrid': [kh.blend(hyb['selected_w'], inp['logit'], inp['ials_raw'][k], inp['c_lr'], inp['c_i'][k]) for k in range(5)]}
    scores['Tree-L2'] = [t['model'].decision_function(fx.tree_matrix(f, s, t['levels'], level='L2')[0]) for t in trees]
    for form in FORMS:
        w, b = A['final'][form]['w'], A['final'][form]['b']
        scores[form] = [design(form, inp, 0, m1)@w+b] if form == 'M2_L2_refit' else [design(form, inp, k, m1)@w+b for k in range(5)]
    scores['best_adapted'] = scores[sel['best_adapted_method']]
    for k, v in scores.items(): assert all(np.isfinite(x).all() for x in v), f'non-finite {k}'
    main_m = f.user_has_long_view.to_numpy(); A_m = f.scope_A.to_numpy()
    aucA = {k: ke.user_auc(f, v, main_m & A_m) for k, v in scores.items()}; aucB = {k: ke.user_auc(f, v, main_m) for k, v in scores.items()}
    C = lambda x, y_, d=aucA: ke.paired(d[x], d[y_])
    out = {'window': '2022-04-29..05-08 (development users; first and only evaluation)', 'best_adapted_method': sel['best_adapted_method'],
           'scope_A': {'exposures': int((main_m & A_m).sum()), 'users': int(len(aucA['LR-L2'])), 'mean_user_auc': {k: float(v.mean()) for k, v in aucA.items()}},
           'primary': {'best_adapted_minus_iALS_tuned': C('best_adapted', 'iALS_tuned')},
           'key_secondary': {'best_adapted_minus_hybrid': C('best_adapted', 'hybrid'), 'M3_minus_M2': C('M3_L2_iALS_refit', 'M2_L2_refit'), 'M4_minus_M3': C('M4_L2_iALS_prior', 'M3_L2_iALS_refit'),
                             'LRL2_minus_iALS_tuned': C('LR-L2', 'iALS_tuned'), 'hybrid_minus_iALS_tuned': C('hybrid', 'iALS_tuned')},
           'other': {f'{form}_minus_iALS_tuned': C(form, 'iALS_tuned') for form in FORMS} | {'M1_minus_hybrid': C('M1_blend', 'hybrid'), 'TreeL2_minus_LRL2': C('Tree-L2', 'LR-L2'), 'M2_minus_LRL2': C('M2_L2_refit', 'LR-L2')},
           'scope_B': {'exposures': int(main_m.sum()), 'mean_user_auc': {k: float(v.mean()) for k, v in aucB.items()}, 'best_adapted_minus_iALS_tuned': C('best_adapted', 'iALS_tuned', aucB), 'best_adapted_minus_hybrid': C('best_adapted', 'hybrid', aucB)},
           'seed_means_scope_A': {k: [float(ke.user_auc(f, r, main_m & A_m).mean()) for r in v] for k, v in scores.items() if len(v) > 1 and k != 'best_adapted'}}
    deg = s.v_l.to_numpy(); grp = np.select([deg == 0, deg <= 2], ['degree_0', 'degree_1_2'], 'degree_ge_3'); rng = np.random.default_rng(2026)
    mean_score = {k: np.mean(v, axis=0) for k, v in scores.items()}; pooled = {}; users = aucA['LR-L2'].index
    for g in ['degree_0', 'degree_1_2', 'degree_ge_3']:
        mm = main_m & A_m & (grp == g) & f.user_id.isin(users).to_numpy(); yv = f.lv.to_numpy()[mm]; uid = f.user_id.to_numpy()[mm]; uniq = np.unique(uid)
        row = {'exposures': int(mm.sum()), 'users': len(uniq), 'auc_seed_mean_score': {k: pooled_auc(yv, v[mm]) for k, v in mean_score.items() if k != 'best_adapted'}}
        index = {u: np.flatnonzero(uid == u) for u in uniq}; picks = [np.concatenate([index[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(2000)]
        d = [pooled_auc(yv[q], mean_score['best_adapted'][mm][q])-pooled_auc(yv[q], mean_score['iALS_tuned'][mm][q]) for q in picks if 0 < yv[q].sum() < len(q)]
        row['best_adapted_minus_iALS_tuned'] = {'difference': pooled_auc(yv, mean_score['best_adapted'][mm])-row['auc_seed_mean_score']['iALS_tuned'], 'ci95': np.quantile(d, [.025, .975]).tolist()}
        pooled[g] = row
    out['pooled_auc_by_video_degree'] = pooled
    cs = ~main_m; ca = {k: ke.user_auc(f, v, cs) for k, v in scores.items() if k != 'best_adapted'}
    out['cold_start'] = {'users_with_random': int(f.loc[cs, 'user_id'].nunique()), 'users_with_both_outcomes': int(len(ca['LR-L2'])), 'mean_user_auc_all_exposures': {k: float(v.mean()) if len(v) else None for k, v in ca.items()}}
    np.savez_compressed(ART/'eval_week2/scores.npz', **{k: np.array(v) for k, v in scores.items()}); pd.DataFrame(aucA).to_csv(RUN/'adapt_week2_per_user_auc_scope_A.csv')
    out['finished_utc'] = datetime.now(timezone.utc).isoformat(); dest.write_text(json.dumps(out, indent=2, default=float)); return out

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['rows', 'fit', 'test']); a = p.parse_args(); torch.set_num_threads(8)
    r = {'rows': stage_rows, 'fit': stage_fit, 'test': stage_test}[a.stage](); print(json.dumps(r, indent=2, default=float))
