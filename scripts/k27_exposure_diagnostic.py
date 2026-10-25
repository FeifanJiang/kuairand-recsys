"""Exposure-mechanism diagnostic (protocol_adaptation.json, part A): same users, same week, same videos, two mechanisms.

Development users, week 1 (04-22..04-28). Set R: their random exposures (random-pool videos by construction);
set S: their recommended (standard-log) exposures restricted to random-pool videos (k27_week1_standard.py).
For every frozen score (tuned iALS seed 0 and 5-seed per-user average, LR-L2, hybrid) the per-user AUC is computed in
each set on scope-A rows; users with >= 1 pre-cutoff long view and both outcomes in BOTH sets form the paired
comparison (random minus recommended, paired user bootstrap 2000 / seed 2026). Descriptive only; no selection.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import feature_model as fm
import k27_eval_rows as er
import k27_eval as ke
import k27_adapt as ka
from train_ranker import auc as pooled_auc

ROOT = Path(__file__).resolve().parents[1]; ART = ROOT/'artifacts/k27_experiment'; FEAT = ROOT/'data/k27/features_dev_8m'

def scores_for(f, s, lr, ials, hyb):
    inp = ka.inputs(f, s, lr, ials, hyb)
    return {'iALS_seed0': [inp['ials_raw'][0]], 'iALS_5seeds': inp['ials_raw'], 'LR-L2': [inp['logit']],
            'hybrid': [ka.kh.blend(hyb['selected_w'], inp['logit'], inp['ials_raw'][k], inp['c_lr'], inp['c_i'][k]) for k in range(5)]}

def main():
    lr, ials, hyb, _ = ka.frozen_models()
    R = pd.read_pickle(ART/'eval/rows.pkl'); sR = pd.read_pickle(ART/'eval/history.pkl')
    std = pd.read_pickle(ROOT/'data/k27/dev_week1_standard_pool.pkl'); S = er.attach_video(std)
    T0 = json.loads((FEAT/'meta.json').read_text())['T0']; vocab = json.loads((er.VMETA/'meta.json').read_text())['vocab']['tag1']
    tables = {k: pd.read_pickle(FEAT/f'frozen_{k}.pkl') for k in fm.KEYS}; sS = er.history(S, tables, T0, tag_codes={t: i for i, t in enumerate(vocab)})
    has_lv = tables['u'].loc[tables['u'].u_l.gt(0), 'user_id']; S['user_has_long_view'] = S.user_id.isin(has_lv).to_numpy(); S['scope_A'] = sS.v_n.gt(0).to_numpy()
    mR = (R.user_has_long_view & R.scope_A).to_numpy(); mS = (S.user_has_long_view & S.scope_A).to_numpy()
    scR = scores_for(R, sR, lr, ials, hyb); scS = scores_for(S, sS, lr, ials, hyb)
    out = {'sets': {'random': {'rows': int(mR.sum()), 'users': int(R.loc[mR, 'user_id'].nunique()), 'positive_rate': float(R.loc[mR, 'lv'].mean())},
                    'recommended_pool': {'rows': int(mS.sum()), 'users': int(S.loc[mS, 'user_id'].nunique()), 'positive_rate': float(S.loc[mS, 'lv'].mean()), 'distinct_videos': int(S.loc[mS, 'video_id'].nunique())},
                    'random_distinct_videos': int(R.loc[mR, 'video_id'].nunique())}, 'per_score': {}}
    for name in scR:
        aR = ke.user_auc(R, scR[name], mR); aS = ke.user_auc(S, scS[name], mS); common = aR.index.intersection(aS.index)
        out['per_score'][name] = {'unpaired': {'random': {'users': int(len(aR)), 'mean_auc': float(aR.mean())}, 'recommended_pool': {'users': int(len(aS)), 'mean_auc': float(aS.mean())}},
                                  'paired_users': int(len(common)), 'paired_mean_auc': {'random': float(aR.loc[common].mean()), 'recommended_pool': float(aS.loc[common].mean())},
                                  'random_minus_recommended': ke.paired(aR, aS, users=common),
                                  'pooled_auc_seed_mean_score': {'random': pooled_auc(R.lv.to_numpy()[mR], np.mean(scR[name], axis=0)[mR]), 'recommended_pool': pooled_auc(S.lv.to_numpy()[mS], np.mean(scS[name], axis=0)[mS])}}
    common = ke.user_auc(R, scR['iALS_seed0'], mR).index.intersection(ke.user_auc(S, scS['iALS_seed0'], mS).index)
    out['paired_users_exposures'] = {'random_rows': int(R.loc[mR & R.user_id.isin(common).to_numpy()].shape[0]), 'recommended_rows': int(S.loc[mS & S.user_id.isin(common).to_numpy()].shape[0])}
    out['reading'] = 'same users, week, video pool and scores; only the exposure mechanism differs; descriptive, no selection; not a test of whether adaptation fixes learned fusion'
    (ROOT/'reports/expansion/exposure_diagnostic.json').write_text(json.dumps(out, indent=2, default=float)); print(json.dumps(out, indent=2, default=float))

if __name__ == '__main__':
    main()
