"""Brute-force spot check of the 27K chunked features on real data (no random-exposure label is read).

For 40 fixed-seed sampled training rows, counts and sums over the whole pre-cutoff store with time < row
time are recomputed by direct masking and compared with the saved prequential features (g, v, a, tg, u,
ua, ut counts and long views, u duration sums, v decay and last long view). For 20 videos, users and
authors the frozen tables are compared with direct totals.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import feature_model as fm
from k27_features import events

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(); p.add_argument('--features', required=True); a = p.parse_args(); d = Path(a.features)
    ev, _, _ = events(); T0 = int(ev['t'].min()); rows = pd.read_pickle(d/'rows.pkl')
    pre = {k: pd.read_pickle(d/f'pre_{k}.pkl') for k in fm.KEYS}; frozen = {k: pd.read_pickle(d/f'frozen_{k}.pkl') for k in ['v', 'u', 'a']}
    rng = np.random.default_rng(11); pick = rng.choice(len(rows), 40, replace=False); bad = []
    t, lv = ev['t'], ev['lv']
    for i in pick:
        r = rows.row.iloc[i]; before = t < t[r]
        masks = {'g': before, 'v': before & (ev['video_id'] == ev['video_id'][r]), 'a': before & (ev['author_id'] == ev['author_id'][r]),
                 'tg': before & (ev['tag1'] == ev['tag1'][r]), 'u': before & (ev['user_id'] == ev['user_id'][r])}
        masks['ua'] = masks['u'] & (ev['author_id'] == ev['author_id'][r]); masks['ut'] = masks['u'] & (ev['tag1'] == ev['tag1'][r])
        for k, m in masks.items():
            got = pre[k].loc[r]
            if got[f'{k}_n'] != m.sum() or got[f'{k}_l'] != lv[m].sum(): bad.append((int(r), k, 'n/l'))
        m = masks['u']
        if not np.isclose(pre['u'].loc[r, 'u_lv_logdur'], ev['lv_logdur'][m].sum()) or pre['u'].loc[r, 'u_lv_short'] != ev['lv_short'][m].sum(): bad.append((int(r), 'u', 'sums'))
        m = masks['v']; direct = (lv[m]*np.exp(-(t[r]-t[m])/(.5*fm.DAY))).sum()
        if not np.isclose(pre['v'].loc[r, 'v_d0.5'], direct, rtol=1e-9, atol=1e-12): bad.append((int(r), 'v', 'decay'))
        last = t[m & (lv == 1)].max() if (m & (lv == 1)).any() else np.nan
        if not (np.isnan(last) and np.isnan(pre['v'].loc[r, 'v_lastlv']) or last == pre['v'].loc[r, 'v_lastlv']): bad.append((int(r), 'v', 'lastlv'))
    for k, col in [('v', 'video_id'), ('u', 'user_id'), ('a', 'author_id')]:
        tab = frozen[k].set_index(col)
        for key in rng.choice(tab.index.to_numpy(), 20, replace=False):
            m = ev[col] == key
            if tab.loc[key, f'{k}_n'] != m.sum() or tab.loc[key, f'{k}_l'] != lv[m].sum(): bad.append((int(key), k, 'frozen'))
    out = {'rows_checked': 40, 'frozen_keys_checked': 60, 'mismatches': bad, 'all_passed': not bad}
    (ROOT/'reports/expansion/k27_feature_spotcheck.json').write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
