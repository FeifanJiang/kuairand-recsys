"""Checks on the 27K pre-cutoff event store (no random-exposure labels are read).

1. Restricted to Pure's videos, the 27K store equals the Pure store built by the same code (row multisets).
2. No exact duplicate rows across the two 4/08-4/21 parts; every user and video exists in the 27K tables;
   all timestamps fall on or before 2022-04-21 Asia/Shanghai.
3. Size summary: rows, users, videos, long views, and development-user exposures.
4. Rows of the 1K users (1K ids equal 27K ids) equal data/processed/train.csv.gz on the stored columns,
   which checks cleaning, de-duplication, the date rule and global video ids at once.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import build_event_store as bes

ROOT = Path(__file__).resolve().parents[1]

def table(s, mask=None):
    f = pd.DataFrame({c: np.asarray(s[c]) if mask is None else np.asarray(s[c])[mask] for c in bes.COLUMNS})
    return f.sort_values(list(bes.COLUMNS)).reset_index(drop=True)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--pure-store', required=True); a = p.parse_args()
    s = bes.load(ROOT/'data/k27/standard_to_0421_clean'); out = {}
    pool = pd.read_csv(ROOT/'data/raw/pure_download/KuaiRand-Pure/data/video_features_basic_pure.csv', usecols=['video_id']).video_id.to_numpy()
    v = np.asarray(s['video_id']); mine = table(s, np.isin(v, pool)); ref = table(bes.load(a.pure_store))
    out['1_pool_rows_27k'] = len(mine); out['1_pool_rows_pure'] = len(ref); out['1_pool_rows_identical'] = bool(len(mine) == len(ref) and mine.equals(ref))
    key = pd.DataFrame({c: np.asarray(s[c]) for c in ['user_id', 'video_id', 'time_ms']})
    out['2_duplicate_user_video_time'] = int(key.duplicated().sum()); del key
    users = pd.read_csv(ROOT/'data/raw/pure_download/KuaiRand-Pure/data/user_features_pure.csv', usecols=['user_id']).user_id.to_numpy()
    u = np.asarray(s['user_id']); vids = np.load(ROOT/'data/k27/video_meta/video_id.npy')
    out['2_all_users_in_user_table'] = bool(np.isin(np.unique(u), users).all()); out['2_all_videos_in_video_table'] = bool(np.isin(np.unique(v), vids).all())
    t = np.asarray(s['time_ms']); out['2_max_shanghai_date'] = int(bes.shanghai_date(t[[t.argmax()]])[0]); out['2_min_shanghai_date'] = int(bes.shanghai_date(t[[t.argmin()]])[0])
    part = pd.read_csv(ROOT/'data/splits/expansion_users.csv'); dev = part.pure_user_id[part.partition.eq('development')].to_numpy()
    used = pd.read_csv(ROOT/'data/raw/pure_download/user_map_1k_to_pure.csv').user_id_pure.to_numpy(); lv = np.asarray(s['long_view'])
    grp = np.where(np.isin(u, dev), 'development', np.where(np.isin(u, used), '1k_users', 'confirmation'))
    out['3_rows'] = int(len(u)); out['3_users'] = int(len(np.unique(u))); out['3_videos'] = int(len(np.unique(v))); out['3_long_views'] = int(lv.sum())
    out['3_by_group'] = {g: {'rows': int((grp == g).sum()), 'users': int(len(np.unique(u[grp == g]))), 'long_views': int(lv[grp == g].sum())} for g in ['development', 'confirmation', '1k_users']}
    k1 = pd.read_csv(ROOT/'data/processed/train.csv.gz', usecols=list(bes.COLUMNS)).astype({c: dt for c, dt in bes.COLUMNS.items()})
    k1 = k1[list(bes.COLUMNS)].sort_values(list(bes.COLUMNS)).reset_index(drop=True); mine1k = table(s, np.isin(u, used))
    out['4_1k_rows_27k'] = len(mine1k); out['4_1k_rows_processed_train'] = len(k1); out['4_1k_rows_identical'] = bool(len(mine1k) == len(k1) and mine1k.equals(k1))
    dest = ROOT/'reports/expansion/k27_store_checks.json'; dest.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
