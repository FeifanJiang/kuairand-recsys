"""Group evaluation of the deep-ranker stage (protocol_deep.json 'evaluation').

Scores LR-L2all (selected l2), LR-L2+iALS_late (selected l2), Deep-L2, Deep-L2+iALS and the late-rows Deep-L2 control
(3 seeds each) on one development group; LR-L2 (dev 6M), Tree-L2, tuned iALS and the hybrid come from the saved
group scores. Per-user AUC per seed averaged per user; paired user bootstrap (2000, seed 2026).
  --group tune    smoke test of the whole path on the tune group (already used for selection)
  --group report  development feedback (the report group was viewed before); runs once
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import k27_eval as ke
import k27_deep as kd
import k27_hybrid as kh
from train_ranker import auc as pooled_auc

ROOT = Path(__file__).resolve().parents[1]; ART = ROOT/'artifacts/k27_experiment'; RUN = ROOT/'reports/expansion/run_dev'

def all_scores(f, s, group, sel):
    full = torch.load(kd.FULL_IALS, weights_only=False); z = np.load(ART/f'{group}_scores.npz')
    scores = {'LR-L2': [z['LR-L2'][0]], 'Tree-L2': list(z['Tree-L2']), 'iALS_tuned': list(z['iALS_tuned'])}
    hyb = json.loads((RUN/'hybrid.json').read_text()); c_lr = hyb['standardization']['LR-L2_mean_sd']; c_i = hyb['standardization']['iALS_mean_sd_per_seed']
    scores['hybrid'] = [kh.blend(hyb['selected_w'], z['LR-L2'][0], z['iALS_tuned'][k], c_lr, c_i[k]) for k in range(len(c_i))]
    scores['LR-L2all'] = [ke.score_lr(f, s, torch.load(ART/f"{sel['LR-L2all']['name']}.pt", weights_only=False))]
    scores['LR-L2+iALS'] = [kd.score_lr_fusion(torch.load(kd.MODELS/f"{sel['LR-L2+iALS_late']['name']}.pt", weights_only=False), f, s, full)]
    for key, model, fam in [('Deep-L2', 'deep', 'deep'), ('Deep-L2+iALS', 'deep_ials', 'deep_ials'), ('Deep-L2(late)', 'deep_late', 'deep_ials')]:
        cfg = sel[fam]['config_index']
        scores[key] = [kd.score_frame(torch.load(kd.MODELS/f'{model}_c{cfg}_s{sd}.pt', weights_only=False), f, s, full if model == 'deep_ials' else None) for sd in range(3)]
    for k, v in scores.items(): assert all(np.isfinite(x).all() for x in v), f'non-finite {k}'
    return scores

def main():
    p = argparse.ArgumentParser(); p.add_argument('--group', choices=['tune', 'report'], default='report'); a = p.parse_args(); group = a.group
    dest = RUN/('deep_report.json' if group == 'report' else 'deep_report_smoke_tune.json'); assert not dest.exists(), f'{dest.name} exists: this evaluation runs once'
    sel = json.loads((RUN/'selection_deep.json').read_text())
    e = pd.read_pickle(ART/'eval/rows.pkl'); se = pd.read_pickle(ART/'eval/history.pkl'); m = e.group.eq(group).to_numpy()
    f, s = e.loc[m].reset_index(drop=True), se.loc[m].reset_index(drop=True); scores = all_scores(f, s, group, sel)
    main_m = f.user_has_long_view.to_numpy(); A = f.scope_A.to_numpy()
    aucA = {k: ke.user_auc(f, v, main_m & A) for k, v in scores.items()}; aucB = {k: ke.user_auc(f, v, main_m) for k, v in scores.items()}
    C = lambda x, y, d=aucA: ke.paired(d[x], d[y])
    out = {'group': group, 'evidence_role': 'development feedback: the report group was viewed before this stage' if group == 'report' else 'smoke test on the tune group (used for selection)',
           'scope_A': {'exposures': int((main_m & A).sum()), 'users': int(len(aucA['LR-L2'])), 'mean_user_auc': {k: float(v.mean()) for k, v in aucA.items()}},
           'contrasts_scope_A': {'DeepL2_minus_LRL2all': C('Deep-L2', 'LR-L2all'), 'DeepL2_minus_iALS_tuned': C('Deep-L2', 'iALS_tuned'), 'DeepL2_minus_TreeL2': C('Deep-L2', 'Tree-L2'),
                                 'DeepL2iALS_minus_iALS_tuned': C('Deep-L2+iALS', 'iALS_tuned'), 'DeepL2iALS_minus_DeepL2late': C('Deep-L2+iALS', 'Deep-L2(late)'),
                                 'DeepL2iALS_minus_LRL2iALS': C('Deep-L2+iALS', 'LR-L2+iALS'), 'DeepL2iALS_minus_hybrid': C('Deep-L2+iALS', 'hybrid'),
                                 'LRL2iALS_minus_iALS_tuned': C('LR-L2+iALS', 'iALS_tuned'), 'LRL2all_minus_LRL2dev': C('LR-L2all', 'LR-L2'), 'DeepL2late_minus_LRL2all': C('Deep-L2(late)', 'LR-L2all')},
           'scope_B': {'exposures': int(main_m.sum()), 'mean_user_auc': {k: float(v.mean()) for k, v in aucB.items()},
                       'DeepL2_minus_LRL2all': C('Deep-L2', 'LR-L2all', aucB), 'DeepL2iALS_minus_iALS_tuned': C('Deep-L2+iALS', 'iALS_tuned', aucB)},
           'seed_means_scope_A': {k: [float(ke.user_auc(f, r, main_m & A).mean()) for r in v] for k, v in scores.items() if len(v) > 1},
           'selection': {k: v['name'] if isinstance(v, dict) and 'name' in v else v for k, v in sel.items() if not k.endswith('_grid')}}
    deg = s.v_l.to_numpy(); grp = np.select([deg == 0, deg <= 2], ['degree_0', 'degree_1_2'], 'degree_ge_3'); rng = np.random.default_rng(2026)
    mean_score = {k: np.mean(v, axis=0) for k, v in scores.items()}; pooled = {}; users = aucA['LR-L2'].index
    for g in ['degree_0', 'degree_1_2', 'degree_ge_3']:
        mm = main_m & A & (grp == g) & f.user_id.isin(users).to_numpy(); yv = f.lv.to_numpy()[mm]; uid = f.user_id.to_numpy()[mm]; uniq = np.unique(uid)
        row = {'exposures': int(mm.sum()), 'users': len(uniq), 'auc_seed_mean_score': {k: pooled_auc(yv, v[mm]) for k, v in mean_score.items()}}
        index = {u: np.flatnonzero(uid == u) for u in uniq}; picks = [np.concatenate([index[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(2000)]
        for x, y in [('Deep-L2+iALS', 'iALS_tuned'), ('Deep-L2', 'LR-L2all')]:
            d = [pooled_auc(yv[q], mean_score[x][mm][q])-pooled_auc(yv[q], mean_score[y][mm][q]) for q in picks if 0 < yv[q].sum() < len(q)]
            row[f'{x}_minus_{y}'] = {'difference': row['auc_seed_mean_score'][x]-row['auc_seed_mean_score'][y], 'ci95': np.quantile(d, [.025, .975]).tolist()}
        pooled[g] = row
    out['pooled_auc_by_video_degree'] = pooled
    np.savez_compressed(ART/f'{group}_deep_scores.npz', **{k: np.array(v) for k, v in scores.items()}); pd.DataFrame(aucA).to_csv(RUN/f'{group}_deep_per_user_auc_scope_A.csv')
    dest.write_text(json.dumps(out, indent=2, default=float)); print(json.dumps(out, indent=2, default=float))

if __name__ == '__main__':
    main()
