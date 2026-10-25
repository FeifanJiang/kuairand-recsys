"""The 27K training/evaluation code path reproduces the KuaiRand-1K feature-model run (reports/feature_model/run_20261021).

1. k27_eval_rows.random_rows on the raw 1K random log (known users, 04-22..04-28) gives the rows of
   data/processed/random_validation.csv.gz for those users.
2. The 27K video_meta attributes equal the 1K metadata for every 1K training and random-validation video.
3. Saved LR-L1 / LR-L2 models scored through k27_eval.score_lr (chunked features, saved statistics) equal
   the saved random-validation logits; saved iALS factors scored through score_ials equal the saved scores.
4. Refitting LR-L1 / LR-L2 on the 1K training rows with the chunked float32 matrix and the selected l2
   gives the saved coefficients.
5. k27_eval.user_auc on the reproduced scores gives the saved scope-A / scope-B mean per-user AUC.
Only 1K data already used for development is read (no test, no random test).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import feature_model as fm
import feature_matrix as fx
import k27_eval_rows as er
import k27_eval as ke

ROOT = Path(__file__).resolve().parents[1]; RUN = ROOT/'reports/feature_model/run_20261021'
COLS = ['user_id', 'video_id', 'time_ms', 'long_view', 'duration_ms', 'tab']

def main():
    out = {}; users = np.sort(pd.read_csv(fm.DATA/'training_users.csv').user_id.to_numpy())
    raw, counts = er.random_rows(ROOT/'data/raw/KuaiRand-1K/data/log_random_4_22_to_5_08_1k.csv', users, 20220422, 20220428)
    ref = pd.read_csv(fm.DATA/'random_validation.csv.gz'); ref = ref.loc[ref.user_id.isin(users)]
    srt = lambda d: d[COLS].astype('int64').sort_values(COLS).reset_index(drop=True)
    out['1_random_rows_equal'] = bool(srt(raw).equals(srt(ref))); out['1_rows'] = len(raw); out['1_cleaning_counts'] = counts
    meta = fm.load_meta(); train = fm.load_frame('train', meta); T0 = int(train.t.min())
    rnd = fm.load_frame('random_validation', meta); rnd = rnd.loc[rnd.user_id.isin(users)].reset_index(drop=True)
    f = er.attach_video(rnd.rename(columns={'t': 'time_ms', 'lv': 'long_view'})[COLS])
    vids = np.union1d(train.video_id.unique(), rnd.video_id.unique()); fv = er.attach_video(pd.DataFrame({'user_id': 0, 'video_id': vids, 'time_ms': 0, 'long_view': 0, 'duration_ms': 0, 'tab': 0}))
    m = meta.loc[vids]; ok = bool((fv.author_id.to_numpy() == m.author_id.to_numpy()).all())
    for c in ['tag1', 'video_type', 'upload_type', 'music_type']: ok &= bool((fv[c].astype(str).to_numpy() == m[c].astype(str).to_numpy()).all())
    ok &= bool(((fv.upload_day.to_numpy() == m.upload_day.to_numpy()) | (fv.upload_day.isna().to_numpy() & m.upload_day.isna().to_numpy())).all())
    out['2_video_attributes_equal'] = ok; out['2_videos'] = len(vids)
    tables = {k: fm.frozen(train, k, T0) for k in fm.KEYS}; s = er.history(f, tables, T0)
    saved = np.load(RUN/'random_validation_scores.npz'); out['3_row_order_matches_saved'] = bool((saved['user_id'] == f.user_id.to_numpy()).all() and (saved['video_id'] == f.video_id.to_numpy()).all())
    scores = {}
    for level in ['L1', 'L2']:
        model = torch.load(RUN/f'model_{level}.pt', weights_only=False); model['level'] = level
        scores[level] = ke.score_lr(f, s, model); out[f'3_{level}_max_abs_logit_diff'] = float(np.abs(scores[level]-saved[level]).max())
        out[f'3_{level}_logits_equal_1e-4'] = out[f'3_{level}_max_abs_logit_diff'] < 1e-4
    cat = np.sort(pd.read_csv(fm.DATA/'candidate_videos.csv').video_id.to_numpy()); runs = []
    for j, seed in enumerate([42, 1, 2, 3, 4]):
        fct = torch.load(fm.FOLLOWUP/fm.IALS.format(seed), weights_only=False); runs.append(ke.score_ials(f, fct['X'], fct['Y'], users, cat))
    out['3_ials_max_abs_diff'] = float(np.abs(np.array(runs)-saved['iALS']).max()); out['3_ials_equal_1e-5'] = out['3_ials_max_abs_diff'] < 1e-5
    pre = pd.concat([fm.prequential(train, k, T0) for k in fm.KEYS], axis=1)
    for level in ['L1', 'L2']:
        model = torch.load(RUN/f'model_{level}.pt', weights_only=False)
        X, (cols, mean, sd) = fx.standardized(train, pre, level, model['levels'])
        w, b, _, _ = fm.fit_lr(torch.from_numpy(X), torch.tensor(train.lv.to_numpy(np.float32)), model['l2']); del X
        out[f'4_{level}_columns_equal'] = cols == model['columns']
        out[f'4_{level}_max_abs_coef_diff'] = float(max(np.abs(w.numpy()-model['w'].numpy()).max(), abs(float(b-model['b']))))
        out[f'4_{level}_coefficients_equal_1e-3'] = out[f'4_{level}_max_abs_coef_diff'] < 1e-3
    summ = json.loads((RUN/'summary.json').read_text()); A = f.video_id.isin(cat).to_numpy()
    aucA = {'L1': ke.user_auc(f, saved['L1'], A), 'L2': ke.user_auc(f, saved['L2'], A), 'iALS': ke.user_auc(f, list(saved['iALS']), A)}
    out['5_scope_A_mean_auc_equal'] = all(abs(aucA[k].mean()-summ['scope_A']['mean_user_auc'][k]) < 1e-12 for k in aucA)
    out['5_scope_B_mean_auc_equal'] = all(abs(ke.user_auc(f, saved[k]).mean()-summ['scope_B']['mean_user_auc'][k]) < 1e-12 for k in ['L1', 'L2'])
    p = ke.paired(aucA['L2'], aucA['iALS']); q = summ['comparisons']['scope_A_primary_L2_minus_iALS']
    out['5_paired_L2_minus_iALS_equal'] = bool(abs(p['difference']-q['difference']) < 1e-12 and p['users'] == q['users'])
    passed = all(v for k, v in out.items() if k.split('_')[-1] in ('equal', 'saved', '1e-4', '1e-5', '1e-3'))
    (ROOT/'reports/expansion/pipeline_checks_1k.json').write_text(json.dumps({'all_passed': passed, **out}, indent=2, default=float)); print(json.dumps({'all_passed': passed, **out}, indent=2, default=float))

if __name__ == '__main__':
    main()
