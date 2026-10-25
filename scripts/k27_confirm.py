"""One-shot evaluation of the frozen 27K models on a user partition and window (protocol_confirmation.json).

--partition development   dry run: rebuilds development rows from the raw random log and must reproduce the saved
                          development results (week 1: run_dev/final.json, hybrid.json and deep_report.json on the
                          report group; week 2: run_dev/adapt_week2.json on all development users)
--partition confirmation  unseals the confirmation users' labels of that window; runs once per window and refuses to
                          run again
--window week1 | week2    2022-04-22..04-28 | 2022-04-29..05-08 (Asia/Shanghai date of time_ms)
Models, hyperparameters, seeds, blend weight, standardization constants and adapted coefficients are all frozen
artifacts of the development runs; nothing is selected here.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import feature_model as fm
import feature_matrix as fx
import k27_eval_rows as er
import k27_eval as ke
import k27_hybrid as kh
import k27_deep as kd
import k27_adapt as ka
from train_ranker import auc as pooled_auc

ROOT = Path(__file__).resolve().parents[1]; ART = ROOT/'artifacts/k27_experiment'; FEAT = ROOT/'data/k27/features_dev_8m'
DEV = ROOT/'reports/expansion/run_dev'; OUT = ROOT/'reports/expansion/run_confirmation'
RANDOM_LOG = ROOT/'data/raw/pure_download/KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv'
WINDOWS = {'week1': (20220422, 20220428), 'week2': (20220429, 20220508)}
FEATURE_MODELS = ['LR-L1', 'Tree-L1', 'LR-L2', 'Tree-L2']
DEEP_MODELS = ['LR-L2all', 'LR-L2+iALS', 'Deep-L2', 'Deep-L2+iALS', 'Deep-L2(late)']
ADAPT_MODELS = ['M1_blend', 'M2_L2_refit', 'M3_L2_iALS_refit', 'M4_L2_iALS_prior']

def evaluation_rows(partition, window):
    p = pd.read_csv(ROOT/'data/splits/expansion_users.csv'); users = np.sort(p.pure_user_id[p.partition.eq(partition)].to_numpy())
    if partition == 'development' and window == 'week1':                   # week-1 dry run: the report group, as final.json
        g = pd.read_csv(ROOT/'data/splits/development_tune_report.csv'); users = np.sort(g.pure_user_id[g.group.eq('report')].to_numpy())
    raw, counts = er.random_rows(RANDOM_LOG, users, *WINDOWS[window]); f = er.attach_video(raw)
    T0 = json.loads((FEAT/'meta.json').read_text())['T0']; vocab = json.loads((er.VMETA/'meta.json').read_text())['vocab']['tag1']
    tables = {k: pd.read_pickle(FEAT/f'frozen_{k}.pkl') for k in ['g', 'v', 'a', 'tg']}
    udir = FEAT if partition == 'development' else ROOT/'data/k27/features_confirmation'
    tables.update({k: pd.read_pickle(udir/f'frozen_{k}.pkl') for k in ['u', 'ua', 'ut']})
    s = er.history(f, tables, T0, tag_codes={t: i for i, t in enumerate(vocab)})
    f['scope_A'] = s.v_n.gt(0).to_numpy(); f['user_has_long_view'] = f.user_id.isin(tables['u'].loc[tables['u'].u_l.gt(0), 'user_id']).to_numpy()
    return f, s, {'users_in_partition': int(len(users)), 'cleaning': counts, 'rows': len(f), 'window': WINDOWS[window]}

def all_scores(f, s, sel):
    scores = {}
    for lvl in ['L1', 'L2']:
        scores[f'LR-{lvl}'] = [ke.score_lr(f, s, torch.load(ART/f"{sel[f'LR-{lvl}']['name']}.pt", weights_only=False))]
        base = sel[f'Tree-{lvl}']['name'].rsplit('_s', 1)[0]; runs = []
        for seed in [0, 1, 2]:
            t = joblib.load(ART/f'{base}_s{seed}.joblib'); Xe, _, _ = fx.tree_matrix(f, s, t['levels'], level=lvl); runs.append(t['model'].decision_function(Xe))
        scores[f'Tree-{lvl}'] = runs
    for key in ['iALS_tuned', 'iALS_reference']:
        base = sel[key]['name'].removeprefix('iALS_').rsplit('_s', 1)[0]; runs = []
        for seed in range(5):
            fct = torch.load(ART/'ials'/f'{base}_s{seed}.pt', weights_only=False); runs.append(ke.score_ials(f, fct['X'], fct['Y'], fct['users'], fct['catalog']))
        scores[key] = runs
    h = json.loads((DEV/'hybrid.json').read_text()); c_lr = h['standardization']['LR-L2_mean_sd']; c_i = h['standardization']['iALS_mean_sd_per_seed']
    scores['hybrid'] = [kh.blend(h['selected_w'], scores['LR-L2'][0], scores['iALS_tuned'][k], c_lr, c_i[k]) for k in range(5)]
    dsel = json.loads((DEV/'selection_deep.json').read_text()); full = torch.load(kd.FULL_IALS, weights_only=False)      # deep-ranker stage (protocol_deep.json), all frozen
    scores['LR-L2all'] = [ke.score_lr(f, s, torch.load(ART/f"{dsel['LR-L2all']['name']}.pt", weights_only=False))]
    scores['LR-L2+iALS'] = [kd.score_lr_fusion(torch.load(kd.MODELS/f"{dsel['LR-L2+iALS_late']['name']}.pt", weights_only=False), f, s, full)]
    for key, model, fam in [('Deep-L2', 'deep', 'deep'), ('Deep-L2+iALS', 'deep_ials', 'deep_ials'), ('Deep-L2(late)', 'deep_late', 'deep_ials')]:
        cfg = dsel[fam]['config_index']
        scores[key] = [kd.score_frame(torch.load(kd.MODELS/f'{model}_c{cfg}_s{sd}.pt', weights_only=False), f, s, full if model == 'deep_ials' else None) for sd in range(3)]
    scores['best_new'] = scores[dsel['best_new_method']]
    A = joblib.load(ka.ADAPT/'adapted_models.joblib'); asel = json.loads((DEV/'selection_adapt.json').read_text()); m1 = (np.array(A['m1_stats'][0]), np.array(A['m1_stats'][1]))
    lr_m, ials_m, hyb_m, _ = ka.frozen_models(); inp = ka.inputs(f, s, lr_m, ials_m, hyb_m)                        # adaptation stage (protocol_adaptation.json), all frozen
    for form in ADAPT_MODELS:
        w, b = A['final'][form]['w'], A['final'][form]['b']
        scores[form] = [ka.design(form, inp, 0, m1)@w+b] if form == 'M2_L2_refit' else [ka.design(form, inp, k, m1)@w+b for k in range(5)]
    scores['best_adapted'] = scores[asel['best_adapted_method']]
    for k, v in scores.items(): assert all(np.isfinite(x).all() for x in v), f'non-finite {k}'
    return scores

def evaluate(f, s, scores):
    main = f.user_has_long_view.to_numpy(); A = f.scope_A.to_numpy()
    aucA = {k: ke.user_auc(f, v, main & A) for k, v in scores.items()}; aucB = {k: ke.user_auc(f, v, main) for k, v in scores.items()}
    C = lambda a, b, d=aucA: ke.paired(d[a], d[b]); dsel = json.loads((DEV/'selection_deep.json').read_text()); asel = json.loads((DEV/'selection_adapt.json').read_text())
    out = {'scope_A': {'exposures': int((main & A).sum()), 'users': {k: len(v) for k, v in aucA.items()}, 'mean_user_auc': {k: float(v.mean()) for k, v in aucA.items()}},
           'primary_preregistered_stage1': {'LRL2_minus_iALS_tuned': C('LR-L2', 'iALS_tuned')},
           'new_method_question': {'best_new_method': dsel['best_new_method'], 'best_new_minus_iALS_tuned': C('best_new', 'iALS_tuned'),
                                   'hybrid_minus_iALS_tuned': C('hybrid', 'iALS_tuned'), 'hybrid_minus_LRL2': C('hybrid', 'LR-L2')},
           'adaptation': {'best_adapted_method': asel['best_adapted_method'], 'best_adapted_minus_iALS_tuned': C('best_adapted', 'iALS_tuned'), 'best_adapted_minus_hybrid': C('best_adapted', 'hybrid'),
                          'M3_minus_M2': C('M3_L2_iALS_refit', 'M2_L2_refit'), 'M4_minus_M3': C('M4_L2_iALS_prior', 'M3_L2_iALS_refit'), 'M2_minus_LRL2': C('M2_L2_refit', 'LR-L2'),
                          **{f'{form}_minus_iALS_tuned': C(form, 'iALS_tuned') for form in ADAPT_MODELS}},
           'secondary': {'TreeL2_minus_LRL2': C('Tree-L2', 'LR-L2'), 'TreeL2_minus_TreeL1': C('Tree-L2', 'Tree-L1'), 'LRL2_minus_LRL1': C('LR-L2', 'LR-L1'),
                         'LRL2_minus_iALS_reference': C('LR-L2', 'iALS_reference'), 'iALS_tuned_minus_reference': C('iALS_tuned', 'iALS_reference')},
           'deep_stage_secondary': {'DeepL2_minus_LRL2all': C('Deep-L2', 'LR-L2all'), 'DeepL2_minus_iALS_tuned': C('Deep-L2', 'iALS_tuned'), 'DeepL2iALS_minus_iALS_tuned': C('Deep-L2+iALS', 'iALS_tuned'),
                                    'DeepL2iALS_minus_DeepL2late': C('Deep-L2+iALS', 'Deep-L2(late)'), 'DeepL2iALS_minus_LRL2iALS': C('Deep-L2+iALS', 'LR-L2+iALS'), 'LRL2iALS_minus_iALS_tuned': C('LR-L2+iALS', 'iALS_tuned'),
                                    'LRL2all_minus_LRL2dev': C('LR-L2all', 'LR-L2')},
           'scope_B': {'exposures': int(main.sum()), 'mean_user_auc': {k: float(v.mean()) for k, v in aucB.items()},
                       'LRL2_minus_iALS_tuned': C('LR-L2', 'iALS_tuned', aucB), 'hybrid_minus_iALS_tuned': C('hybrid', 'iALS_tuned', aucB), 'best_adapted_minus_iALS_tuned': C('best_adapted', 'iALS_tuned', aucB),
                       'best_adapted_minus_hybrid': C('best_adapted', 'hybrid', aucB), 'TreeL2_minus_LRL2': C('Tree-L2', 'LR-L2', aucB), 'LRL2_minus_LRL1': C('LR-L2', 'LR-L1', aucB)},
           'seed_variation_scope_A': {k: [float(ke.user_auc(f, r, main & A).mean()) for r in v] for k, v in scores.items() if len(v) > 1 and k not in ('best_new', 'best_adapted')}}
    deg = s.v_l.to_numpy(); grp = np.select([deg == 0, deg <= 2], ['degree_0', 'degree_1_2'], 'degree_ge_3'); rng = np.random.default_rng(2026)
    mean_score = {k: np.mean(v, axis=0) for k, v in scores.items()}; pooled = {}; main_users = aucA['LR-L1'].index
    for g in ['degree_0', 'degree_1_2', 'degree_ge_3']:
        m = main & A & (grp == g) & f.user_id.isin(main_users).to_numpy(); yv = f.lv.to_numpy()[m]; uid = f.user_id.to_numpy()[m]; uniq = np.unique(uid)
        row = {'exposures': int(m.sum()), 'positives': int(yv.sum()), 'users': len(uniq), 'auc_seed_mean_score': {k: pooled_auc(yv, v[m]) for k, v in mean_score.items() if k not in ('best_new', 'best_adapted')}}
        index = {u: np.flatnonzero(uid == u) for u in uniq}; picks = [np.concatenate([index[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(2000)]
        for a, b in [('LR-L2', 'iALS_tuned'), ('hybrid', 'iALS_tuned'), ('best_adapted', 'iALS_tuned'), ('Tree-L2', 'LR-L2'), ('LR-L2', 'LR-L1')]:
            d = [pooled_auc(yv[q], mean_score[a][m][q])-pooled_auc(yv[q], mean_score[b][m][q]) for q in picks if 0 < yv[q].sum() < len(q)]
            row[f'{a}_minus_{b}'] = {'difference': pooled_auc(yv, mean_score[a][m])-pooled_auc(yv, mean_score[b][m]), 'ci95': np.quantile(d, [.025, .975]).tolist()}
        pooled[g] = row
    out['pooled_auc_by_video_degree'] = pooled
    cs = ~main; ca = {k: ke.user_auc(f, v, cs) for k, v in scores.items() if k not in ('best_new', 'best_adapted')}
    out['cold_start'] = {'users_with_random': int(f.loc[cs, 'user_id'].nunique()), 'users_with_both_outcomes': int(len(ca['LR-L1'])),
                         'mean_user_auc_all_exposures': {k: float(v.mean()) if len(v) else None for k, v in ca.items()}, 'note': 'descriptive only; iALS scores these users 0 -> AUC 0.5 (ties half)'}
    return out, aucA

def dry_run_checks(out, window):
    tol = 1e-9; checks = {}; m = out['scope_A']['mean_user_auc']
    if window == 'week1':
        ref = json.loads((DEV/'final.json').read_text())['result']; hyb = json.loads((DEV/'hybrid.json').read_text())['report_scope_A']; dr = json.loads((DEV/'deep_report.json').read_text())
        checks.update({k: abs(m[k]-ref['scope_A']['mean_user_auc'][k]) < tol for k in FEATURE_MODELS+['iALS_tuned', 'iALS_reference']})
        checks['hybrid'] = abs(m['hybrid']-hyb['mean_user_auc']['hybrid']) < tol
        checks['primary_difference'] = abs(out['primary_preregistered_stage1']['LRL2_minus_iALS_tuned']['difference']-ref['primary']['main_L2_minus_iALS_tuned']['difference']) < tol
        checks['hybrid_difference'] = abs(out['new_method_question']['hybrid_minus_iALS_tuned']['difference']-hyb['hybrid_minus_iALS_tuned']['difference']) < tol
        for k in DEEP_MODELS: checks[f'deep_{k}'] = abs(m[k]-dr['scope_A']['mean_user_auc'][k]) < tol
        checks['deep_difference'] = abs(out['deep_stage_secondary']['DeepL2iALS_minus_iALS_tuned']['difference']-dr['contrasts_scope_A']['DeepL2iALS_minus_iALS_tuned']['difference']) < tol
        checks['users'] = out['scope_A']['users']['LR-L2'] == ref['scope_A']['users']['LR-L2']
    else:
        ref = json.loads((DEV/'adapt_week2.json').read_text())
        checks.update({k: abs(m[k]-ref['scope_A']['mean_user_auc'][k]) < tol for k in ['LR-L2', 'Tree-L2', 'iALS_tuned', 'hybrid']+ADAPT_MODELS})
        checks['adaptation_primary_difference'] = abs(out['adaptation']['best_adapted_minus_iALS_tuned']['difference']-ref['primary']['best_adapted_minus_iALS_tuned']['difference']) < tol
        checks['users'] = out['scope_A']['users']['LR-L2'] == ref['scope_A']['users']
    return {'all_passed': all(checks.values()), **checks}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--partition', choices=['development', 'confirmation'], required=True); p.add_argument('--window', choices=list(WINDOWS), default='week1'); a = p.parse_args()
    torch.set_num_threads(8); OUT.mkdir(parents=True, exist_ok=True); suffix = '' if a.window == 'week1' else '_week2'
    dest = OUT/(f'development_dry_run{suffix}.json' if a.partition == 'development' else f'confirmation{suffix}.json')
    assert not dest.exists(), f'{dest.name} already exists: this evaluation runs once'
    if a.partition == 'confirmation':
        d = json.loads((OUT/f'development_dry_run{suffix}.json').read_text())['checks']; assert d['all_passed'], 'the development dry run of this window must pass first'
        assert json.loads((ROOT/'reports/expansion/protocol_confirmation.json').read_text())['status'].startswith('CONFIRMED'), 'protocol not confirmed'
    sel = json.loads((DEV/'selection_final.json').read_text()); f, s, info = evaluation_rows(a.partition, a.window)
    scores = all_scores(f, s, sel); out, aucA = evaluate(f, s, scores)
    rec = {'partition': a.partition, 'window': a.window, 'info': info, 'result': out, 'finished_utc': datetime.now(timezone.utc).isoformat()}
    if a.partition == 'development': rec['checks'] = dry_run_checks(out, a.window)
    pd.DataFrame(aucA).to_csv(OUT/f'{a.partition}{suffix}_per_user_auc_scope_A.csv'); np.savez_compressed(ART/f'{a.partition}{suffix}_confirm_scores.npz', **{k: np.array(v) for k, v in scores.items()})
    dest.write_text(json.dumps(rec, indent=2, default=float)); print(json.dumps(rec, indent=2, default=float))

if __name__ == '__main__':
    main()
