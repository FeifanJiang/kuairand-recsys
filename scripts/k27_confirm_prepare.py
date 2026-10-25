"""Frozen-at-cutoff user statistics (keys u, ua, ut) for one user partition, from the pre-cutoff store.

Only pre-cutoff standard logs are read (allowed for confirmation users); no random-exposure row is
touched. Rows of the partition's users are selected first, then feature_model.frozen runs per user
bucket (chunked_features.key_blocks). With --partition development the tables must equal the ones
k27_features.py saved for the development users (features_dev_8m/frozen_{u,ua,ut}.pkl); that check runs
before the same code is trusted on the confirmation users.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import feature_model as fm
import chunked_features as cf
import build_event_store as bes

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT/'data/k27/standard_to_0421_clean'; VMETA = ROOT/'data/k27/video_meta'; FEAT = ROOT/'data/k27/features_dev_8m'
USER_KEYS = ['u', 'ua', 'ut']

def partition_users(name):
    p = pd.read_csv(ROOT/'data/splits/expansion_users.csv'); return np.sort(p.pure_user_id[p.partition.eq(name)].to_numpy())

def subset_events(users):
    s = bes.load(STORE); keep = np.flatnonzero(np.isin(np.asarray(s['user_id']), users))
    vid = np.load(VMETA/'video_id.npy'); v = np.asarray(s['video_id'])[keep]; pos = np.searchsorted(vid, v); assert (vid[pos] == v).all()
    lv = np.asarray(s['long_view'])[keep]; dur = np.asarray(s['duration_ms'])[keep]
    return {'user_id': np.asarray(s['user_id'])[keep], 'video_id': v, 't': np.asarray(s['time_ms'])[keep], 'lv': lv,
            'author_id': np.load(VMETA/'author_id.npy')[pos], 'tag1': np.load(VMETA/'tag1.npy')[pos],
            'const': np.zeros(len(lv), np.int8), 'lv_logdur': lv*np.log1p(dur/1000), 'lv_short': (lv*(dur <= 18000)).astype(np.int8)}

def frozen_user_tables(ev, T0, n_buckets=16):
    out = {}
    for key in USER_KEYS:
        parts = [fm.frozen(cf.frame(ev, idx), key, T0) for idx in cf.key_blocks(ev, key, n_buckets) if len(idx)]
        out[key] = pd.concat(parts, ignore_index=True)
    return out

def same_table(a, b, key):
    cols = fm.KEYS[key]
    if len(a) != len(b) or list(a.columns) != list(b.columns): return False
    oa = np.lexsort([a[c].to_numpy() for c in reversed(cols)]); ob = np.lexsort([b[c].to_numpy() for c in reversed(cols)])
    for c in a.columns:
        x, y = a[c].to_numpy()[oa], b[c].to_numpy()[ob]
        ok = np.array_equal(x, y) if c in cols else np.allclose(x.astype(float), y.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True)
        if not ok: return False
    return True

def main():
    p = argparse.ArgumentParser(); p.add_argument('--partition', choices=['development', 'confirmation'], required=True); a = p.parse_args()
    T0 = json.loads((FEAT/'meta.json').read_text())['T0']; users = partition_users(a.partition)
    ev = subset_events(users); tables = frozen_user_tables(ev, T0)
    info = {'partition': a.partition, 'users': int(len(users)), 'rows': int(len(ev['t'])), 'T0': T0, 'labels_read': 'standard-log long_view before the cutoff only; no random exposure'}
    if a.partition == 'development':
        info['equal_to_features_dev_8m'] = {k: same_table(tables[k], pd.read_pickle(FEAT/f'frozen_{k}.pkl'), k) for k in USER_KEYS}
        info['all_passed'] = all(info['equal_to_features_dev_8m'].values())
        (ROOT/'reports/expansion/confirm_prepare_checks.json').write_text(json.dumps(info, indent=2)); print(json.dumps(info, indent=2))
        if not info['all_passed']: raise SystemExit('development user tables differ from features_dev_8m')
    else:
        d = ROOT/'data/k27/features_confirmation'; d.mkdir(parents=True, exist_ok=False)
        for k in USER_KEYS: tables[k].to_pickle(d/f'frozen_{k}.pkl')
        (d/'meta.json').write_text(json.dumps(info, indent=2)); print(json.dumps(info, indent=2))

if __name__ == '__main__':
    main()
