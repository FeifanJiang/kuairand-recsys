"""Development users' standard (recommended) exposures of random-pool videos in week 1 (2022-04-22..04-28), streamed
from the 27K release tar for the exposure-mechanism diagnostic (k27_exposure_diagnostic.py).

Rows are kept when: standard log (is_rand == 0), user in the development partition, video in the random pool
(video_features_basic_pure), Asia/Shanghai date of time_ms in the window; the 1K cleaning rules apply (invalid
values dropped, full-record duplicates removed). Confirmation users' rows are never kept. Output:
data/k27/dev_week1_standard_pool.pkl with user_id, video_id, time_ms, long_view, duration_ms, tab.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import build_event_store as bes

ROOT = Path(__file__).resolve().parents[1]
TAR = ROOT/'data/raw/k27_download/KuaiRand-27K.tar.gz'; OUT = ROOT/'data/k27/dev_week1_standard_pool.pkl'
FIRST, LAST = 20220422, 20220428

def main():
    part = pd.read_csv(ROOT/'data/splits/expansion_users.csv'); dev = part.pure_user_id[part.partition.eq('development')].to_numpy()
    pool = pd.read_csv(ROOT/'data/raw/pure_download/KuaiRand-Pure/data/video_features_basic_pure.csv', usecols=['video_id']).video_id.to_numpy()
    parts = []; counts = {'rows_read': 0, 'rows_dev_pool': 0, 'invalid': 0}
    for name, fh in bes.stream_members(TAR, r'log_standard_4_22_to_5_08_27k_part\d\.csv$'):
        for chunk in pd.read_csv(fh, chunksize=bes.CHUNK):
            counts['rows_read'] += len(chunk)
            chunk = chunk.loc[chunk.user_id.isin(dev) & chunk.video_id.isin(pool) & chunk.is_rand.eq(0)]
            if len(chunk) == 0: continue
            invalid = chunk.isna().any(axis=1).to_numpy().copy()
            for c in bes.BINARY: invalid |= ~chunk[c].isin([0, 1]).to_numpy()
            invalid |= (chunk[bes.NONNEG] < 0).any(axis=1).to_numpy(); invalid |= ~chunk.tab.between(0, 14).to_numpy(); invalid |= ~chunk.date.between(20220408, 20220508).to_numpy()
            counts['rows_dev_pool'] += len(chunk); counts['invalid'] += int(invalid.sum()); parts.append(chunk.loc[~invalid])
        print(name, counts, flush=True)
    ev = pd.concat(parts, ignore_index=True); dup = ev.duplicated(); counts['full_record_duplicates'] = int(dup.sum()); ev = ev.loc[~dup]
    d = bes.shanghai_date(ev.time_ms.to_numpy()); ev = ev.loc[(d >= FIRST) & (d <= LAST), list(bes.COLUMNS)].reset_index(drop=True); counts['rows_in_window'] = len(ev)
    ev.to_pickle(OUT); (OUT.with_suffix('.json')).write_text(json.dumps(counts, indent=2)); print(json.dumps(counts, indent=2))

if __name__ == '__main__':
    main()
