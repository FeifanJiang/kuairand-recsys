"""27K development comparison (reports/expansion/protocol_confirmed.json): LR-L1, LR-L2, Tree-L1, Tree-L2, iALS.

Stages (one process each, run serially by k27_run_all.py):
  eval      development random exposures 04-22..04-28, frozen-at-cutoff history, flags (group, scope A, history)
  lr        one level: fit the l2 grid on the 6M training rows, save models, score tune rows
  tree      one level, grid index and seed: fit HistGradientBoosting with early stopping, save model, score tune rows
  ials      one setting and seed: fit (k27_ials), score tune rows
  final     score report rows for every frozen model and seed, then every protocol contrast
Tune-stage outputs never contain report-group scores; selection reads tune AUCs only (k27_select.py).
"""
import argparse
import itertools
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import feature_model as fm
import feature_matrix as fx
import k27_eval_rows as er
import k27_eval as ke
from k27_eval_rows import training_frame

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT/'data/k27/features_dev_8m'; FEAT_ALL = ROOT/'data/k27/features_all_8m'; ART = ROOT/'artifacts/k27_experiment'; OUT = ROOT/'reports/expansion/run_dev'
ROWS = 6_000_000; LR_GRID = [1e-4, 1e-3, 1e-2]
TREE_GRID = [dict(max_leaf_nodes=a, min_samples_leaf=b, l2_regularization=c) for a, b, c in itertools.product([15, 63], [100, 1000], [0.0, 1.0])]
TREE_FIXED = dict(learning_rate=.1, max_iter=2000, early_stopping=True, validation_fraction=.1, scoring='loss', n_iter_no_change=20, tol=1e-7, max_bins=255)

def load_eval():
    e = pd.read_pickle(ART/'eval/rows.pkl'); s = pd.read_pickle(ART/'eval/history.pkl'); return e, s

def tune_mask(e): return (e.group.eq('tune') & e.scope_A & e.user_has_long_view).to_numpy()

def tune_auc(e, scores):
    return float(ke.user_auc(e, scores, tune_mask(e)).mean())

def stage_eval():
    part = pd.read_csv(ROOT/'data/splits/development_tune_report.csv'); dev = part.pure_user_id.to_numpy()
    raw, counts = er.random_rows(ROOT/'data/raw/pure_download/KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv', dev, 20220422, 20220428)
    f = er.attach_video(raw); vocab = json.loads((er.VMETA/'meta.json').read_text())['vocab']['tag1']
    tables = {k: pd.read_pickle(FEAT/f'frozen_{k}.pkl') for k in fm.KEYS}; T0 = json.loads((FEAT/'meta.json').read_text())['T0']
    s = er.history(f, tables, T0, tag_codes={t: i for i, t in enumerate(vocab)})
    f['group'] = f.user_id.map(part.set_index('pure_user_id').group); f['scope_A'] = s.v_n.gt(0).to_numpy()
    f['user_has_long_view'] = f.user_id.isin(tables['u'].loc[tables['u'].u_l.gt(0), 'user_id']).to_numpy()
    d = ART/'eval'; d.mkdir(parents=True, exist_ok=False); f.to_pickle(d/'rows.pkl'); s.to_pickle(d/'history.pkl')
    info = {'cleaning': counts, 'rows': len(f), 'scope_A_rows': int(f.scope_A.sum()), 'by_group_rows': f.group.value_counts().to_dict(), 'T0': T0}
    (d/'info.json').write_text(json.dumps(info, indent=2)); return info

def stage_lr(level, feat='dev'):
    """feat='dev': the frozen 6M development sample; feat='all': the 8M all-users sample S_all (protocol_deep.json control)."""
    feat_dir, rows, prefix = (FEAT, ROWS, f'LR-{level}') if feat == 'dev' else (FEAT_ALL, 8_000_000, f'LR-{level}all')
    e, se = load_eval(); m = e.group.eq('tune').to_numpy(); et, st = e.loc[m], se.loc[m]
    f, s = training_frame(feat_dir, rows); levels = fx.training_levels(f); y = torch.from_numpy(f.lv.to_numpy(np.float32))
    X, (cols, mean, sd) = fx.standardized(f, s, level, levels); del s; Xt = torch.from_numpy(X); res = []
    for l2 in LR_GRID:
        w, b, obj, grad = fm.fit_lr(Xt, y, l2)
        model = {'w': w, 'b': b, 'columns': cols, 'mean': mean, 'sd': sd, 'levels': levels, 'level': level, 'l2': l2}
        name = f'{prefix}_l2{l2:g}'; torch.save(model, ART/f'{name}.pt'); sc = ke.score_lr(et, st, model)
        assert np.isfinite(sc).all() and np.isfinite(obj), f'non-finite {name}'
        np.save(ART/f'{name}_tune_scores.npy', sc); res.append({'name': name, 'l2': l2, 'train_objective': obj, 'grad_norm': grad, 'tune_auc': tune_auc(et, sc)})
    return res

def stage_cache_tree(level):
    """Write the 6M-row tree matrix once; tree jobs memory-map it (file-backed pages need no swap)."""
    f, s = training_frame(FEAT, ROWS); levels = fx.training_levels(f)
    X, names, is_cat = fx.tree_matrix(f, s, levels, level=level); d = ART/'tree_cache'; d.mkdir(exist_ok=True)
    np.save(d/f'X_{level}.npy', X); np.save(d/f'y_{level}.npy', f.lv.to_numpy(np.float32))
    (d/f'meta_{level}.json').write_text(json.dumps({'names': names, 'is_cat': is_cat.tolist(), 'levels': levels, 'rows': len(f)}))
    return {'rows': len(f), 'columns': len(names)}

def stage_tree(level, config, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    e, se = load_eval(); m = e.group.eq('tune').to_numpy(); et, st = e.loc[m], se.loc[m]
    d = ART/'tree_cache'
    if (d/f'X_{level}.npy').exists():
        X = np.load(d/f'X_{level}.npy', mmap_mode='r'); y = np.load(d/f'y_{level}.npy'); meta = json.loads((d/f'meta_{level}.json').read_text())
        names, is_cat, levels = meta['names'], np.array(meta['is_cat']), meta['levels']
    else:
        f, s = training_frame(FEAT, ROWS); levels = fx.training_levels(f); y = f.lv.to_numpy(np.float32)
        X, names, is_cat = fx.tree_matrix(f, s, levels, level=level); del s
    t0 = time.time(); model = HistGradientBoostingClassifier(**TREE_FIXED, **TREE_GRID[config], categorical_features=is_cat, random_state=seed).fit(X, y); fit_s = time.time()-t0
    del X; name = f'Tree-{level}_c{config}_s{seed}'; joblib.dump({'model': model, 'levels': levels, 'level': level, 'names': names}, ART/f'{name}.joblib')
    Xe, _, _ = fx.tree_matrix(et, st, levels, level=level); sc = model.decision_function(Xe); assert np.isfinite(sc).all(), f'non-finite {name}'; np.save(ART/f'{name}_tune_scores.npy', sc)
    return {'name': name, 'config': TREE_GRID[config], 'seed': seed, 'iterations': int(model.n_iter_), 'fit_seconds': fit_s, 'tune_auc': tune_auc(et, sc)}

def stage_ials(tau, alpha, l2, seed):
    import subprocess, sys
    subprocess.run([sys.executable, str(ROOT/'scripts/k27_ials.py'), '--tau', str(tau), '--alpha', str(alpha), '--l2', str(l2), '--seed', str(seed), '--out', str(ART/'ials')], check=True)
    name = f'tau{tau:g}_alpha{alpha:g}_l2{l2:g}_s{seed}'; fct = torch.load(ART/'ials'/f'{name}.pt', weights_only=False)
    e, _ = load_eval(); et = e.loc[e.group.eq('tune').to_numpy()]; sc = ke.score_ials(et, fct['X'], fct['Y'], fct['users'], fct['catalog'])
    assert bool(torch.isfinite(fct['X']).all() and torch.isfinite(fct['Y']).all()) and np.isfinite(sc).all(), f'non-finite {name}'
    np.save(ART/f'iALS_{name}_tune_scores.npy', sc); return {'name': f'iALS_{name}', 'tau': tau, 'alpha': alpha, 'l2': l2, 'seed': seed, 'tune_auc': tune_auc(et, sc)}

def stage_final(group='report'):
    """Evaluation of the frozen selection (selection_final.json) on one group: 'report' once, or 'tune' as a dry run."""
    from train_ranker import auc as pooled_auc
    sel = json.loads((OUT/'selection_final.json').read_text()); e, se = load_eval()
    rep = e.group.eq(group).to_numpy(); er_, sr = e.loc[rep].reset_index(drop=True), se.loc[rep].reset_index(drop=True)
    main = er_.user_has_long_view.to_numpy(); A = er_.scope_A.to_numpy(); scores = {}
    for lvl in ['L1', 'L2']:
        scores[f'LR-{lvl}'] = [ke.score_lr(er_, sr, torch.load(ART/f"{sel[f'LR-{lvl}']['name']}.pt", weights_only=False))]
        base = sel[f'Tree-{lvl}']['name'].rsplit('_s', 1)[0]; runs = []
        for seed in [0, 1, 2]:
            t = joblib.load(ART/f'{base}_s{seed}.joblib'); Xe, _, _ = fx.tree_matrix(er_, sr, t['levels'], level=lvl); runs.append(t['model'].decision_function(Xe))
        scores[f'Tree-{lvl}'] = runs
    for key in ['iALS_tuned', 'iALS_reference']:
        base = sel[key]['name'].removeprefix('iALS_').rsplit('_s', 1)[0]; runs = []
        for seed in range(5):
            fct = torch.load(ART/'ials'/f'{base}_s{seed}.pt', weights_only=False); runs.append(ke.score_ials(er_, fct['X'], fct['Y'], fct['users'], fct['catalog']))
        scores[key] = runs
    scores['main_L2'] = scores[sel['main_L2_method']]
    np.savez_compressed(ART/f'{group}_scores.npz', **{k: np.array(v) for k, v in scores.items()})
    aucA = {k: ke.user_auc(er_, v, main & A) for k, v in scores.items()}; aucB = {k: ke.user_auc(er_, v, main) for k, v in scores.items() if not k.startswith('iALS')}
    pd.DataFrame(aucA).to_csv(OUT/f'{group}_per_user_auc_scope_A.csv'); main_users = aucA['LR-L1'].index
    C = lambda a, b, d=aucA: ke.paired(d[a], d[b])
    out = {'selection': {k: sel[k]['name'] for k in ['LR-L1', 'LR-L2', 'Tree-L1', 'Tree-L2', 'iALS_tuned', 'iALS_reference']}, 'main_L2_method': sel['main_L2_method'],
           'scope_A': {'exposures': int((main & A).sum()), 'users': {k: len(v) for k, v in aucA.items()}, 'mean_user_auc': {k: float(v.mean()) for k, v in aucA.items()}},
           'primary': {'main_L2_minus_iALS_tuned': C('main_L2', 'iALS_tuned')},
           'key_secondary': {'nonlinearity_TreeL2_minus_LRL2': C('Tree-L2', 'LR-L2'), 'personalization_TreeL2_minus_TreeL1': C('Tree-L2', 'Tree-L1'),
                             'personalization_LRL2_minus_LRL1': C('LR-L2', 'LR-L1')},
           'other_secondary': {'main_L2_minus_iALS_reference': C('main_L2', 'iALS_reference'), 'iALS_tuned_minus_reference': C('iALS_tuned', 'iALS_reference'),
                               'scope_B': {'exposures': int(main.sum()), 'TreeL2_minus_LRL2': C('Tree-L2', 'LR-L2', aucB), 'TreeL2_minus_TreeL1': C('Tree-L2', 'Tree-L1', aucB),
                                           'LRL2_minus_LRL1': C('LR-L2', 'LR-L1', aucB)}}}
    out['seed_variation_scope_A'] = {k: [float(ke.user_auc(er_, r, main & A).mean()) for r in v] for k, v in scores.items() if len(v) > 1 and k != 'main_L2'}
    deg = sr.v_l.to_numpy(); grp = np.select([deg == 0, deg <= 2], ['degree_0', 'degree_1_2'], 'degree_ge_3'); rng = np.random.default_rng(2026); pooled = {}
    mean_score = {k: np.mean(v, axis=0) for k, v in scores.items()}
    for g in ['degree_0', 'degree_1_2', 'degree_ge_3']:
        m = main & A & (grp == g) & er_.user_id.isin(main_users).to_numpy(); sub = er_.loc[m]; yv = sub.lv.to_numpy(); uid = sub.user_id.to_numpy(); uniq = np.unique(uid)
        row = {'exposures': int(m.sum()), 'positives': int(yv.sum()), 'users': len(uniq), 'auc_seed_mean_score': {k: pooled_auc(yv, v[m]) for k, v in mean_score.items()}}
        index = {u: np.flatnonzero(uid == u) for u in uniq}; picks = [np.concatenate([index[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(2000)]
        for a, b in [('main_L2', 'iALS_tuned'), ('Tree-L2', 'LR-L2'), ('Tree-L2', 'Tree-L1'), ('LR-L2', 'LR-L1')]:
            d = [pooled_auc(yv[p], mean_score[a][m][p])-pooled_auc(yv[p], mean_score[b][m][p]) for p in picks if 0 < yv[p].sum() < len(p)]
            row[f'{a}_minus_{b}'] = {'difference': row['auc_seed_mean_score'][a]-row['auc_seed_mean_score'][b], 'ci95': np.quantile(d, [.025, .975]).tolist()}
        pooled[g] = row
    out['pooled_auc_by_video_degree'] = pooled
    cs = ~main; cold = {'users_with_random': int(er_.loc[cs, 'user_id'].nunique())}
    ca = {k: ke.user_auc(er_, v, cs) for k, v in scores.items() if k != 'main_L2'}
    cold['users_with_both_outcomes'] = int(len(ca['LR-L1'])); cold['mean_user_auc_all_exposures'] = {k: float(v.mean()) if len(v) else None for k, v in ca.items()}
    cold['note'] = 'descriptive only; iALS scores these users 0 -> AUC 0.5 (ties half)'; out['cold_start'] = cold; out['group'] = group
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument('stage'); p.add_argument('--level'); p.add_argument('--config', type=int); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tau', type=float); p.add_argument('--alpha', type=float); p.add_argument('--l2', type=float); p.add_argument('--feat', choices=['dev', 'all'], default='dev'); a = p.parse_args()
    torch.set_num_threads(8); ART.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    if a.stage == 'eval': r = stage_eval(); tag = 'eval'
    elif a.stage == 'lr': r = stage_lr(a.level, a.feat); tag = f'lr_{a.level}'+('all' if a.feat == 'all' else '')
    elif a.stage == 'tree': r = stage_tree(a.level, a.config, a.seed); tag = f'tree_{a.level}_c{a.config}_s{a.seed}'
    elif a.stage == 'ials': r = stage_ials(a.tau, a.alpha, a.l2, a.seed); tag = f'ials_tau{a.tau:g}_alpha{a.alpha:g}_l2{a.l2:g}_s{a.seed}'
    elif a.stage == 'cache_tree': r = stage_cache_tree(a.level); tag = f'cache_tree_{a.level}'
    elif a.stage == 'final':
        assert not (OUT/'final.json').exists(), 'report group already evaluated'; r = stage_final(); tag = 'final'
    elif a.stage == 'final_dry_run_tune': r = stage_final('tune'); tag = 'final_dry_run_tune'
    else: raise SystemExit(f'unknown stage {a.stage}')
    rec = {'stage': tag, 'seconds': time.time()-t0, 'result': r}
    (OUT/f'{tag}.json').write_text(json.dumps(rec, indent=2, default=float)); print(json.dumps(rec, indent=2, default=float))

if __name__ == '__main__':
    main()
