"""Leakage and consistency checks for feature_model.py on a hand-built event table (no real data).

1. Prequential counts exclude the row's own timestamp group and every later event.
2. Decayed long views equal a direct sum over strictly earlier long views.
3. Last-long-view times come from strictly earlier timestamps only.
4. Frozen statistics at the cutoff equal the totals, and aged decay equals a direct sum at query time.
5. Changing a later label, or a label at the same timestamp, leaves a row's training features unchanged.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import feature_model as fm

def main():
    DAY = fm.DAY; t = np.array([0, 0, 1, 1, 2, 3])*DAY//2
    ev = pd.DataFrame({'user_id': [1, 2, 1, 3, 1, 2], 'video_id': [7, 7, 7, 8, 7, 7], 'author_id': [5]*6, 'tag1': ['x']*6, 't': t,
                       'lv': [1, 0, 1, 1, 0, 1], 'duration_ms': [10000]*6})
    ev['const'] = 0; ev['lv_logdur'] = ev.lv*np.log1p(ev.duration_ms/1000); ev['lv_short'] = ev.lv*(ev.duration_ms <= 18000)
    T0 = int(ev.t.min()); out = {}
    pre = fm.prequential(ev, 'v', T0)
    expect_n = [sum(1 for j in range(6) if ev.video_id[j]==ev.video_id[i] and ev.t[j] < ev.t[i]) for i in range(6)]
    expect_l = [sum(ev.lv[j] for j in range(6) if ev.video_id[j]==ev.video_id[i] and ev.t[j] < ev.t[i]) for i in range(6)]
    out['1_counts_strictly_before'] = pre.v_n.tolist()==expect_n and pre.v_l.tolist()==expect_l
    for tau in fm.TAUS:
        direct = [sum(ev.lv[j]*np.exp(-(ev.t[i]-ev.t[j])/(tau*DAY)) for j in range(6) if ev.video_id[j]==ev.video_id[i] and ev.t[j] < ev.t[i]) for i in range(6)]
        out[f'2_decay_tau{tau:g}_matches_direct'] = bool(np.allclose(pre[f'v_d{tau:g}'].to_numpy(), direct))
    last = [max([ev.t[j] for j in range(6) if ev.video_id[j]==ev.video_id[i] and ev.t[j] < ev.t[i] and ev.lv[j]==1], default=np.nan) for i in range(6)]
    out['3_last_long_view_strictly_before'] = bool(np.allclose(pre.v_lastlv.to_numpy(), np.array(last, float), equal_nan=True))
    tab = fm.frozen(ev, 'v', T0); q = pd.DataFrame({'video_id': [7], 't': [5*DAY]})
    s = fm.query(q, {'v': tab}, T0)
    direct = sum(ev.lv[j]*np.exp(-(5*DAY-ev.t[j])/(.5*DAY)) for j in range(6) if ev.video_id[j]==7)
    out['4_frozen_totals_and_aged_decay'] = bool(s.v_n.iloc[0]==5 and s.v_l.iloc[0]==3 and np.isclose(s['v_d0.5'].iloc[0], direct))
    def flip(rows):
        c = ev.copy(); c.loc[rows, 'lv'] = 1-c.loc[rows, 'lv']
        c['lv_logdur'] = c.lv*np.log1p(c.duration_ms/1000); c['lv_short'] = c.lv*(c.duration_ms <= 18000)
        return pd.concat([fm.prequential(c, k, T0) for k in fm.KEYS], axis=1).fillna(-1).to_numpy(float)
    base = pd.concat([fm.prequential(ev, k, T0) for k in fm.KEYS], axis=1).fillna(-1).to_numpy(float)
    same_time = flip([1])                                       # row 1 shares row 0's timestamp
    out['5_row0_unchanged_when_same_timestamp_label_changes'] = bool(np.allclose(base[0], same_time[0]))
    later = flip([4, 5])                                        # rows 4 and 5 are later than rows 0-3
    out['5_rows0_3_unchanged_when_later_labels_change'] = bool(np.allclose(base[:4], later[:4]))
    out['5_sanity_row5_changes_when_earlier_label_changes'] = not bool(np.allclose(base[5], flip([4])[5]))
    passed = all(v for v in out.values())
    dest = Path(__file__).resolve().parents[1]/'reports/feature_model'; dest.mkdir(parents=True, exist_ok=True)
    (dest/'implementation_checks.json').write_text(json.dumps({'all_passed': passed, **out}, indent=2))
    print(json.dumps({'all_passed': passed, **out}, indent=2))

if __name__ == '__main__':
    main()
