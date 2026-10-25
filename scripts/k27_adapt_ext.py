"""Extension analysis (reports/expansion/protocol_extension_cf.json): adapt the collaborative representation with the same
random-exposure labels as M3.

C1: 32 element-wise products u_k * v_k of the frozen tuned iALS factors (seed 0) plus the has-factors flag.
C2: LR-L2 standardized columns plus the same 33 collaborative inputs.
Interaction columns are standardized with mean/sd of the tune-group week-1 training rows (fixed). Fitting, selection
(report-group week-1 per-user AUC, ties -> larger lambda) and the refit on all development week-1 rows follow
k27_adapt.py; the test is development week 2 (already viewed for earlier models: extension evidence). Collaborative
inputs use seed 0 only, so C1/C2 have a single run; baselines are reported both seed-averaged (as published) and seed 0.
Writes run_dev/ext_cf.json; refuses to run twice.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import k27_adapt as ka
import k27_eval as ke
import k27_hybrid as kh

ROOT = Path(__file__).resolve().parents[1]; ART = ROOT/'artifacts/k27_experiment'; RUN = ROOT/'reports/expansion/run_dev'
FORMS = {'C1_cf_interactions': [1e-4, 1e-3, 1e-2], 'C2_L2_plus_cf': [1e-3, 1e-2, 1e-1]}; TIE = 5e-4

def interactions(f, fct):
    """x_u * y_v (32 columns) for rows with valid factors, zeros otherwise; plus the has-factors flag."""
    def locate(keys, ids):
        i = np.minimum(np.searchsorted(keys, ids), len(keys)-1); return i, keys[i] == ids
    ui, uok = locate(np.asarray(fct['users']), f.user_id.to_numpy()); vi, vok = locate(np.asarray(fct['catalog']), f.video_id.to_numpy())
    X, Y = fct['X'].numpy(), fct['Y'].numpy(); ok = uok & vok & (np.linalg.norm(X[ui], axis=1) > 0) & (np.linalg.norm(Y[vi], axis=1) > 0)
    inter = np.where(ok[:, None], X[ui]*Y[vi], 0.).astype(np.float64); return inter, ok.astype(np.float64)

def design(form, X_l2, inter_std, flag):
    cf = np.concatenate([inter_std, flag[:, None]], 1)
    return cf if form == 'C1_cf_interactions' else np.concatenate([X_l2, cf], 1)

def main():
    dest = RUN/'ext_cf.json'; assert not dest.exists(), 'extension already run'
    lr, ials, hyb, _ = ka.frozen_models(); fct0 = ials[0]
    e = pd.read_pickle(ART/'eval/rows.pkl'); se = pd.read_pickle(ART/'eval/history.pkl'); main_m = (e.user_has_long_view & e.scope_A).to_numpy()
    both = e.user_id.isin(ka.both_outcome_users(e, main_m)).to_numpy(); tr = e.group.eq('tune').to_numpy() & main_m & both; sel_m = e.group.eq('report').to_numpy() & main_m & both; alld = main_m & both
    inp = ka.inputs(e, se, lr, ials, hyb); inter, flag = interactions(e, fct0); y = e.lv.to_numpy(float)
    mean, sd = inter[tr].mean(0), inter[tr].std(0, ddof=1); sd[sd == 0] = 1; inter_std = (inter-mean)/sd
    fits, sel = {}, {}
    for form, grid in FORMS.items():
        D = design(form, inp['X'], inter_std, flag); rows = []
        for l2 in grid:
            w, b, obj, g = ka.fit_ridge(D[tr], y[tr], l2); rows.append({'form': form, 'l2': l2, 'objective': obj, 'grad_norm': g, 'train_rows': int(tr.sum()), 'report_week1_auc': float(ke.user_auc(e, D@w+b, sel_m).mean()), 'w': w, 'b': b})
        best = max(r['report_week1_auc'] for r in rows); chosen = sorted([r for r in rows if r['report_week1_auc'] >= best-TIE], key=lambda r: -r['l2'])[0]
        fits[form] = rows; sel[form] = {'l2': chosen['l2'], 'report_week1_auc': chosen['report_week1_auc']}
    final = {}
    for form in FORMS:
        D = design(form, inp['X'], inter_std, flag); w, b, obj, g = ka.fit_ridge(D[alld], y[alld], sel[form]['l2']); final[form] = {'w': w, 'b': b, 'l2': sel[form]['l2'], 'objective': obj, 'train_rows': int(alld.sum())}
    joblib.dump({'final': final, 'interaction_mean': mean, 'interaction_sd': sd, 'forms': FORMS}, ka.ADAPT/'extension_cf_models.joblib')
    # ---- week-2 test (development users; extension evidence) ----
    f2 = pd.read_pickle(ART/'eval_week2/rows.pkl'); s2 = pd.read_pickle(ART/'eval_week2/history.pkl'); inp2 = ka.inputs(f2, s2, lr, ials, hyb); inter2, flag2 = interactions(f2, fct0); inter2_std = (inter2-mean)/sd
    A = joblib.load(ka.ADAPT/'adapted_models.joblib'); m1 = (np.array(A['m1_stats'][0]), np.array(A['m1_stats'][1]))
    scores = {'LR-L2': [inp2['logit']], 'iALS_tuned': inp2['ials_raw'], 'iALS_tuned_seed0': [inp2['ials_raw'][0]],
              'hybrid': [kh.blend(hyb['selected_w'], inp2['logit'], inp2['ials_raw'][k], inp2['c_lr'], inp2['c_i'][k]) for k in range(5)],
              'M2_L2_refit': [ka.design('M2_L2_refit', inp2, 0, m1)@A['final']['M2_L2_refit']['w']+A['final']['M2_L2_refit']['b']],
              'M3_L2_iALS_refit': [ka.design('M3_L2_iALS_refit', inp2, k, m1)@A['final']['M3_L2_iALS_refit']['w']+A['final']['M3_L2_iALS_refit']['b'] for k in range(5)],
              'M3_seed0': [ka.design('M3_L2_iALS_refit', inp2, 0, m1)@A['final']['M3_L2_iALS_refit']['w']+A['final']['M3_L2_iALS_refit']['b']]}
    for form in FORMS: scores[form] = [design(form, inp2['X'], inter2_std, flag2)@final[form]['w']+final[form]['b']]
    for k, v in scores.items(): assert all(np.isfinite(x).all() for x in v), k
    main2 = f2.user_has_long_view.to_numpy(); A2 = f2.scope_A.to_numpy(); aucA = {k: ke.user_auc(f2, v, main2 & A2) for k, v in scores.items()}; aucB = {k: ke.user_auc(f2, v, main2) for k, v in scores.items()}
    C = lambda a, b, d=aucA: ke.paired(d[a], d[b])
    ref = json.loads((RUN/'adapt_week2.json').read_text())['scope_A']['mean_user_auc']
    out = {'protocol': 'reports/expansion/protocol_extension_cf.json', 'evidence_level': 'extension analysis: development week 2 was already viewed for M1-M4; collaborative inputs are seed 0 only',
           'selection_week1_report_group': sel, 'grid_results_week1': [{k: v for k, v in r.items() if k not in ('w', 'b')} for rows in fits.values() for r in rows],
           'interaction_valid_share': {'week1_training_rows': float(flag[tr].mean()), 'week2_rows': float(flag2[main2 & A2].mean())}, 'refit_rows': int(alld.sum()),
           'week2': {'users': int(len(aucA['LR-L2'])), 'exposures': int((main2 & A2).sum()), 'mean_user_auc': {k: float(v.mean()) for k, v in aucA.items()},
                     'consistency_with_adapt_week2_json': {k: abs(float(aucA[k].mean())-ref[k]) < 1e-9 for k in ['LR-L2', 'iALS_tuned', 'hybrid', 'M2_L2_refit', 'M3_L2_iALS_refit']},
                     'contrasts': {'C1_minus_iALS_tuned': C('C1_cf_interactions', 'iALS_tuned'), 'C1_minus_iALS_tuned_seed0': C('C1_cf_interactions', 'iALS_tuned_seed0'), 'C2_minus_C1': C('C2_L2_plus_cf', 'C1_cf_interactions'),
                                   'C2_minus_M3': C('C2_L2_plus_cf', 'M3_L2_iALS_refit'), 'C2_minus_M3_seed0': C('C2_L2_plus_cf', 'M3_seed0'), 'C1_minus_M2': C('C1_cf_interactions', 'M2_L2_refit'),
                                   'C2_minus_iALS_tuned': C('C2_L2_plus_cf', 'iALS_tuned'), 'C2_minus_M2': C('C2_L2_plus_cf', 'M2_L2_refit'), 'M3_minus_iALS_tuned': C('M3_L2_iALS_refit', 'iALS_tuned')},
                     'scope_B': {'C1_minus_iALS_tuned': C('C1_cf_interactions', 'iALS_tuned', aucB), 'C2_minus_M3': C('C2_L2_plus_cf', 'M3_L2_iALS_refit', aucB)}},
           'coefficients_C1_final': {'interactions_abs_mean': float(np.abs(final['C1_cf_interactions']['w'][:32]).mean()), 'flag': float(final['C1_cf_interactions']['w'][32]), 'interactions': final['C1_cf_interactions']['w'][:32].tolist()},
           'finished_utc': datetime.now(timezone.utc).isoformat()}
    np.savez_compressed(ART/'eval_week2/scores_ext_cf.npz', **{k: np.array(v) for k, v in scores.items()}); dest.write_text(json.dumps(out, indent=2, default=float)); print(json.dumps(out, indent=2, default=float))

if __name__ == '__main__':
    main()
