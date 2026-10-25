"""Training-row sample and features for 27K users (no random-exposure label is read).

Sample: every standard exposure of the chosen users (--users development, the default, or all 26,904 users
with pre-cutoff exposures) in the pre-cutoff store gets one uniform draw from a fixed-seed generator in store order; rows ranked by that draw form nested equal-probability samples
(the first 1M of the ranking are inside the first 2M, and so on). The ranking prefix of length --max-rows is
saved, so any smaller sample is a prefix of it.
Features: chunked_features on the full pre-cutoff history of all users (user keys for development users only).
Each training row's statistics use events strictly earlier than the row (same-timestamp events excluded);
the frozen-at-cutoff tables for evaluation are saved alongside.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import feature_model as fm
import chunked_features as cf
import build_event_store as bes

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT/'data/k27/standard_to_0421_clean'; VMETA = ROOT/'data/k27/video_meta'
SEED = 20261023

def events():
    s = bes.load(STORE); vid = np.load(VMETA/'video_id.npy'); pos = np.searchsorted(vid, np.asarray(s['video_id']))
    assert (vid[pos] == np.asarray(s['video_id'])).all()
    lv = np.asarray(s['long_view']); dur = np.asarray(s['duration_ms'])     # lv is int8; pandas group sums upcast to int64
    ev = {'user_id': np.asarray(s['user_id']), 'video_id': np.asarray(s['video_id']), 't': np.asarray(s['time_ms']), 'lv': lv,
          'author_id': np.load(VMETA/'author_id.npy')[pos], 'tag1': np.load(VMETA/'tag1.npy')[pos],
          'const': np.zeros(len(lv), np.int8), 'lv_logdur': lv*np.log1p(dur/1000), 'lv_short': (lv*(dur <= 18000)).astype(np.int8)}
    return ev, s, pos

def sample(user_id, max_rows, users, seed):
    rows = np.flatnonzero(np.isin(user_id, users)) if users is not None else np.arange(len(user_id))
    draw = np.random.default_rng(seed).random(len(rows))                     # one draw per eligible row, store order
    return rows[np.argsort(draw, kind='stable')[:max_rows]], len(rows)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--max-rows', type=int, default=4_000_000); p.add_argument('--out', required=True)
    p.add_argument('--buckets', type=int, default=64); p.add_argument('--ranges', type=int, default=64)
    p.add_argument('--users', choices=['development', 'all'], default='development'); p.add_argument('--seed', type=int, default=SEED); a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    ev, s, pos = events(); T0 = int(ev['t'].min())
    part = pd.read_csv(ROOT/'data/splits/expansion_users.csv'); dev = np.sort(part.pure_user_id[part.partition.eq('development')].to_numpy())
    ranked, n_dev = sample(ev['user_id'], a.max_rows, dev if a.users == 'development' else None, a.seed); np.save(out/'sample_ranked_rows.npy', ranked)
    wanted = np.sort(ranked)
    for key in fm.KEYS:
        if key in cf.CARRY:
            pre, tab = cf.carry_prequential(ev, key, T0, a.ranges, wanted); pre = pre.reindex(wanted)
        else:
            users = dev if key in ('u', 'ua', 'ut') and a.users == 'development' else None; parts, tabs = [], []
            for idx in cf.key_blocks(ev, key, a.buckets, users):
                if len(idx) == 0: continue
                f = cf.frame(ev, idx); q = fm.prequential(f, key, T0)
                parts.append(q.loc[q.index.intersection(wanted)]); tabs.append(fm.frozen(f, key, T0))
            pre = pd.concat(parts).reindex(wanted); tab = pd.concat(tabs, ignore_index=True)
        pre.to_pickle(out/f'pre_{key}.pkl'); tab.to_pickle(out/f'frozen_{key}.pkl'); print(key, pre.shape, tab.shape, flush=True)
    rows = pd.DataFrame({'row': wanted, 'user_id': ev['user_id'][wanted], 'video_id': ev['video_id'][wanted], 't': ev['t'][wanted],
                         'lv': ev['lv'][wanted], 'duration_ms': np.asarray(s['duration_ms'])[wanted], 'tab': np.asarray(s['tab'])[wanted], 'video_pos': pos[wanted]})
    rows.to_pickle(out/'rows.pkl')
    meta = {'seed': a.seed, 'users': a.users, 'rule': 'uniform draw per eligible pre-cutoff standard exposure (store order); nested prefixes of the draw ranking',
            'eligible_rows': int(n_dev), 'max_rows': a.max_rows, 'T0': T0, 'buckets': a.buckets, 'ranges': a.ranges,
            'store_meta_sha256': hashlib.sha256((STORE/'meta.json').read_bytes()).hexdigest(), 'labels_read': 'standard-log long_view only; no random exposure'}
    (out/'meta.json').write_text(json.dumps(meta, indent=2)); print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
