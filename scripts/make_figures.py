"""README figure: per-user AUC of the frozen models on the held-out confirmation users, week 2 (the primary window).

Bars start at chance (0.5); whiskers are 95% user-bootstrap intervals of the mean (2000 resamples, seed 2026) computed
from the per-user AUC table written by k27_confirm.py. The pre-registered paired contrasts are in
reports/expansion/run_confirmation/confirmation_week2.json. Needs matplotlib (requirements-dev.txt).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; C = ROOT/'reports/expansion/run_confirmation'; OUT = ROOT/'docs/figures'
LABELS = [('LR-L1', 'Logistic regression, non-personalized features'), ('Tree-L1', 'Gradient-boosted trees, non-personalized'),
          ('LR-L2', 'Logistic regression, personalized features'), ('Tree-L2', 'Gradient-boosted trees, personalized'),
          ('Deep-L2', 'DLRM-style ranker (ID embeddings + features)'), ('iALS_reference', 'iALS, reference configuration'),
          ('iALS_tuned', 'iALS, tuned on random exposures'), ('hybrid', 'Blend of LR-L2 and iALS (one parameter)'),
          ('M2_L2_refit', 'LR-L2 re-fitted on 98K random-exposure labels (M2)'), ('M3_L2_iALS_refit', 'M2 + iALS score (M3)')]

def main():
    auc = pd.read_csv(C/'confirmation_week2_per_user_auc_scope_A.csv', index_col=0); rng = np.random.default_rng(2026); rows = []
    for key, label in LABELS:
        v = auc[key].to_numpy(); draws = rng.choice(v, (2000, len(v)), replace=True).mean(1)
        rows.append((label, float(v.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))))
    fig, ax = plt.subplots(figsize=(9, 4.6)); y = np.arange(len(rows))[::-1]
    for yi, (label, m, lo, hi) in zip(y, rows):
        trained_on_random = '(M2)' in label or '(M3)' in label
        ax.barh(yi, m-.5, left=.5, color='#E4572E' if '(M3)' in label else '#F3A683' if trained_on_random else '#4C78A8' if 'iALS' not in label and 'Blend' not in label else '#54A24B', height=.72)
        ax.errorbar(m, yi, xerr=[[m-lo], [hi-m]], fmt='none', ecolor='black', elinewidth=1, capsize=2)
        ax.text(hi+.0015, yi, f'{m:.3f}', va='center', fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9); ax.set_xlim(.56, .70); ax.set_xlabel('Mean per-user AUC on randomly exposed videos (95% user-bootstrap interval)')
    ax.set_title(f'Held-out confirmation users, week 2 (n = {len(auc):,})', fontsize=10)
    ax.axvline(auc['iALS_tuned'].mean(), color='#54A24B', lw=.8, ls='--'); ax.grid(axis='x', lw=.3); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    fig.text(.01, .01, 'blue: trained on logged exposures   green: collaborative filtering / blend   orange: re-fitted on random-exposure labels', fontsize=8, color='#444')
    OUT.mkdir(parents=True, exist_ok=True); fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(OUT/'confirmation_week2_auc.png', dpi=160); print(OUT/'confirmation_week2_auc.png')

if __name__ == '__main__':
    main()
