"""One-parameter blend of LR-L2 and tuned iALS on the 27K development users (protocol_hybrid.json).

score_w = (1 - w) * z(LR-L2 logit) + w * z(iALS score). z standardizes each model with the mean and SD of
its scores on the tune-group selection rows (main population, scope A; scores only, no label); each iALS
seed gets its own constants. w is chosen on the tune group with iALS seed 0 (ties within 0.0005 -> larger
w, i.e. closer to the stronger single model). The report group is then evaluated once, only at the selected
w, with all five iALS seeds (per-user AUC per seed, averaged per user) and labelled development feedback,
because the report group was already viewed for the main comparison. Nothing is trained: every score
comes from the finished experiment's artifacts (tune_scores.npz, report_scores.npz).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import k27_eval as ke
from train_ranker import auc as pooled_auc

ROOT = Path(__file__).resolve().parents[1]; ART = ROOT/'artifacts/k27_experiment'; OUT = ROOT/'reports/expansion/run_dev'
W = [round(.1*i, 1) for i in range(11)]; TIE = 5e-4

def load(group):
    e = pd.read_pickle(ART/'eval/rows.pkl'); m = e.group.eq(group).to_numpy()
    rows = e.loc[m].reset_index(drop=True); hist = pd.read_pickle(ART/'eval/history.pkl').loc[m].reset_index(drop=True)
    z = np.load(ART/f'{group}_scores.npz'); return rows, hist, z['LR-L2'][0], z['iALS_tuned']

def blend(w, lr, ials, c_lr, c_ials):
    return (1-w)*(lr-c_lr[0])/c_lr[1] + w*(ials-c_ials[0])/c_ials[1]

def main():
    assert not (OUT/'hybrid.json').exists(), 'hybrid already evaluated on the report group'
    rows, hist, lr, ials = load('tune'); sel = (rows.user_has_long_view & rows.scope_A).to_numpy()
    c_lr = (float(lr[sel].mean()), float(lr[sel].std(ddof=1)))
    c_ials = [(float(s[sel].mean()), float(s[sel].std(ddof=1))) for s in ials]
    curve = {w: float(ke.user_auc(rows, blend(w, lr, ials[0], c_lr, c_ials[0]), sel).mean()) for w in W}
    best = max(curve.values()); w_sel = max(w for w in W if curve[w] >= best-TIE)
    out = {'standardization': {'LR-L2_mean_sd': c_lr, 'iALS_mean_sd_per_seed': c_ials, 'rows': int(sel.sum())},
           'tune_curve_seed0': curve, 'selected_w': w_sel, 'tie_rule': 'within 0.0005 -> larger w', 'evidence_role': 'development feedback: the report group was viewed before this experiment'}
    rows, hist, lr, ials = load('report'); main = rows.user_has_long_view.to_numpy(); A = rows.scope_A.to_numpy()
    hyb = [blend(w_sel, lr, ials[s], c_lr, c_ials[s]) for s in range(len(ials))]
    auc = {'hybrid': ke.user_auc(rows, hyb, main & A), 'iALS_tuned': ke.user_auc(rows, list(ials), main & A), 'LR-L2': ke.user_auc(rows, lr, main & A)}
    out['report_scope_A'] = {'users': int(len(auc['hybrid'])), 'exposures': int((main & A).sum()), 'mean_user_auc': {k: float(v.mean()) for k, v in auc.items()},
                             'hybrid_minus_iALS_tuned': ke.paired(auc['hybrid'], auc['iALS_tuned']), 'hybrid_minus_LRL2': ke.paired(auc['hybrid'], auc['LR-L2']),
                             'seed_means_hybrid': [float(ke.user_auc(rows, h, main & A).mean()) for h in hyb]}
    aucB = {'hybrid': ke.user_auc(rows, hyb, main), 'iALS_tuned': ke.user_auc(rows, list(ials), main), 'LR-L2': ke.user_auc(rows, lr, main)}
    out['report_scope_B'] = {'users': int(len(aucB['hybrid'])), 'hybrid_minus_iALS_tuned': ke.paired(aucB['hybrid'], aucB['iALS_tuned']), 'hybrid_minus_LRL2': ke.paired(aucB['hybrid'], aucB['LR-L2'])}
    deg = hist.v_l.to_numpy(); grp = np.select([deg == 0, deg <= 2], ['degree_0', 'degree_1_2'], 'degree_ge_3'); rng = np.random.default_rng(2026)
    mean_score = {'hybrid': np.mean(hyb, axis=0), 'iALS_tuned': np.mean(ials, axis=0), 'LR-L2': lr}; pooled = {}
    for g in ['degree_0', 'degree_1_2', 'degree_ge_3']:
        m = main & A & (grp == g) & rows.user_id.isin(auc['hybrid'].index).to_numpy(); yv = rows.lv.to_numpy()[m]; uid = rows.user_id.to_numpy()[m]; uniq = np.unique(uid)
        row = {'exposures': int(m.sum()), 'users': len(uniq), 'auc_seed_mean_score': {k: pooled_auc(yv, v[m]) for k, v in mean_score.items()}}
        index = {u: np.flatnonzero(uid == u) for u in uniq}; picks = [np.concatenate([index[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(2000)]
        d = [pooled_auc(yv[p], mean_score['hybrid'][m][p])-pooled_auc(yv[p], mean_score['iALS_tuned'][m][p]) for p in picks if 0 < yv[p].sum() < len(p)]
        row['hybrid_minus_iALS_tuned'] = {'difference': row['auc_seed_mean_score']['hybrid']-row['auc_seed_mean_score']['iALS_tuned'], 'ci95': np.quantile(d, [.025, .975]).tolist()}
        pooled[g] = row
    out['report_pooled_auc_by_video_degree'] = pooled
    (OUT/'hybrid.json').write_text(json.dumps(out, indent=2, default=float)); print(json.dumps(out, indent=2, default=float))

if __name__ == '__main__':
    main()
