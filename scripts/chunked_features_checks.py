"""Equivalence of chunked_features with feature_model on the full KuaiRand-1K training log.

1. Prequential features of a fixed random sample of rows equal feature_model.prequential on the whole log,
   for every key and several bucket / time-range counts; user keys are compared on a user subset only and
   must be absent for other users.
2. Frozen tables equal feature_model.frozen (user keys restricted to the same user subset).
3. Negative control: author statistics computed inside video buckets (incomplete author history) differ,
   so the comparison is able to detect a partition that cuts a key's history.
Only data/processed/train.csv.gz is read.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import feature_model as fm
import chunked_features as cf

ROOT = Path(__file__).resolve().parents[1]

def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(a.shape == b.shape and np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True))

def main():
    train = fm.load_frame('train', fm.load_meta()); T0 = int(train.t.min())
    ev = {c: train[c].to_numpy() for c in cf.COLUMNS}
    rng = np.random.default_rng(7); wanted = np.sort(rng.choice(len(train), 300_000, replace=False))
    users = np.sort(rng.choice(train.user_id.unique(), 300, replace=False))
    ref_pre = {k: fm.prequential(train, k, T0) for k in fm.KEYS}; ref_tab = {k: fm.frozen(train, k, T0) for k in fm.KEYS}
    sel = train.user_id.to_numpy()[wanted]; in_users = np.isin(sel, users); out = {'rows_compared': len(wanted), 'users_for_user_keys': len(users)}
    for n_buckets, n_ranges in [(16, 8), (5, 3), (64, 20)]:
        pre, tabs = cf.build(ev, T0, wanted, n_buckets, n_ranges, user_keys_for=users); tag = f'b{n_buckets}_r{n_ranges}'
        for k in fm.KEYS:
            ref = ref_pre[k].loc[wanted]; mine = pre[ref.columns]
            if k in ('u', 'ua', 'ut'):
                out[f'1_{tag}_{k}'] = same(mine[in_users], ref[in_users]) and bool(mine[~in_users].isna().all().all())
            else:
                out[f'1_{tag}_{k}'] = same(mine, ref)
            cols = fm.KEYS[k]; r = ref_tab[k]
            if k in ('u', 'ua', 'ut'): r = r[r.user_id.isin(users)]
            r = r.sort_values(cols).reset_index(drop=True); m = tabs[k].sort_values(cols).reset_index(drop=True)
            out[f'2_{tag}_{k}_frozen'] = bool(len(r) == len(m) and (r[cols].astype(str).values == m[cols].astype(str).values).all()) and same(m[r.columns.drop(cols)], r[r.columns.drop(cols)])
    v_bucket = np.flatnonzero(train.video_id.to_numpy() % 16 == 0)
    wrong = fm.prequential(train.iloc[v_bucket], 'a', T0)
    out['3_negative_control_author_in_video_bucket_differs'] = not same(wrong, ref_pre['a'].iloc[v_bucket])
    passed = all(v for k, v in out.items() if k[0].isdigit())
    dest = ROOT/'reports/expansion'; dest.mkdir(parents=True, exist_ok=True)
    (dest/'chunked_features_checks.json').write_text(json.dumps({'all_passed': passed, **out}, indent=2)); print(json.dumps({'all_passed': passed, **out}, indent=2))

if __name__ == '__main__':
    main()
