"""Stream KuaiRand standard logs out of the release tar.gz into a compact column store (no extraction).

Members matching --members are read in order with tarfile stream mode and pandas chunks; only the columns
used by feature_model are kept. Cleaning follows prepare_data.py: rows with
invalid values (missing, non-binary flags, negative ids/times/durations, tab outside 0-14, publisher date
outside 2022-04-08..05-08, is_rand inconsistent with the file) are dropped and counted. Rows are kept when
they are standard exposures (is_rand == 0) whose Asia/Shanghai calendar date (from time_ms) is on or before
--last-date. A 64-bit hash of every raw column is stored so full-record duplicates (removed in prepare_data)
can be found; dedupe() writes the final store without them. Each column is appended to <out>/<column>.bin;
<out>/meta.json records dtypes, row count and per-member counts.
"""
import argparse
import io
import json
import re
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = {'user_id': np.int32, 'video_id': np.int32, 'time_ms': np.int64, 'long_view': np.int8,
           'duration_ms': np.int64, 'tab': np.int16}
BINARY = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate', 'long_view', 'is_profile_enter', 'is_rand']
NONNEG = ['user_id', 'video_id', 'time_ms', 'play_time_ms', 'duration_ms', 'profile_stay_time', 'comment_stay_time']
CHUNK = 2_000_000

class _Sequential(io.RawIOBase):
    """Read-only, non-seekable view of a tar stream member (tarfile's stream object lacks seekable())."""
    def __init__(self, f): self.f = f
    def readable(self): return True
    def readinto(self, b):
        data = self.f.read(len(b)); b[:len(data)] = data; return len(data)

def stream_members(tar_path, pattern):
    """Yield (member name, file object) for members whose name matches pattern, in archive order."""
    with tarfile.open(tar_path, mode='r|gz') as tar:
        for m in tar:
            if m.isfile() and re.search(pattern, m.name):
                yield m.name, io.BufferedReader(_Sequential(tar.extractfile(m)), 1 << 24)

def shanghai_date(time_ms):
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(time_ms), unit='ms', utc=True)).tz_convert('Asia/Shanghai')
    return (d.year*10000+d.month*100+d.day).to_numpy()

def build(tar_path, pattern, out, last_date):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    files = {c: open(out/f'{c}.bin', 'wb') for c in list(COLUMNS)+['row_hash']}; per_member = {}; total = 0; schema = None
    for name, fh in stream_members(tar_path, pattern):
        seen = kept = invalid_rows = 0; random_file = Path(name).name.startswith('log_random_')
        for chunk in pd.read_csv(fh, chunksize=CHUNK):
            sig = list(zip(chunk.columns, map(str, chunk.dtypes))); schema = schema or sig
            assert sig == schema, f'column types differ between chunks in {name}: row hashes would not be comparable'
            invalid = chunk.isna().any(axis=1).to_numpy().copy()
            for c in BINARY: invalid |= ~chunk[c].isin([0, 1]).to_numpy()
            invalid |= (chunk[NONNEG] < 0).any(axis=1).to_numpy()
            invalid |= ~chunk.tab.between(0, 14).to_numpy(); invalid |= ~chunk.date.between(20220408, 20220508).to_numpy()
            invalid |= chunk.is_rand.eq(1).to_numpy() != random_file
            keep = ~invalid & chunk.is_rand.eq(0).to_numpy() & (shanghai_date(chunk.time_ms) <= last_date)
            seen += len(chunk); kept += int(keep.sum()); invalid_rows += int(invalid.sum())
            for c, dt in COLUMNS.items():
                v = chunk[c].to_numpy()[keep]; assert (v.astype(dt) == v).all(), f'{c} overflows {dt}'
                files[c].write(v.astype(dt).tobytes())
            files['row_hash'].write(pd.util.hash_pandas_object(chunk.loc[keep], index=False).to_numpy(np.uint64).tobytes())
        per_member[name] = {'rows_read': seen, 'rows_kept': kept, 'invalid_rows': invalid_rows}; total += kept
        print(name, per_member[name], flush=True)
    for f in files.values(): f.close()
    meta = {'source': str(tar_path), 'members': per_member, 'rows': total, 'last_date': last_date,
            'dtypes': {c: np.dtype(dt).str for c, dt in COLUMNS.items()} | {'row_hash': '<u8'}}
    (out/'meta.json').write_text(json.dumps(meta, indent=2)); return meta

def dedupe(src, out):
    """Copy store src to out without full-record duplicates (first occurrence kept, as in prepare_data)."""
    s = load(src); meta = json.loads((Path(src)/'meta.json').read_text())
    key = pd.DataFrame({c: np.asarray(s[c]) for c in ['row_hash', 'user_id', 'video_id', 'time_ms']})
    keep = ~key.duplicated().to_numpy(); del key
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    for c in s:
        if c != 'row_hash': np.asarray(s[c])[keep].tofile(out/f'{c}.bin')
    meta.update({'rows_before_dedupe': meta['rows'], 'rows': int(keep.sum()), 'full_record_duplicates_removed': int((~keep).sum()), 'deduped_from': str(src)})
    meta['dtypes'].pop('row_hash'); (out/'meta.json').write_text(json.dumps(meta, indent=2)); return meta

def load(store):
    store = Path(store); meta = json.loads((store/'meta.json').read_text())
    return {c: np.memmap(store/f'{c}.bin', dtype=np.dtype(dt), mode='r', shape=(meta['rows'],)) for c, dt in meta['dtypes'].items()}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tar', required=True); p.add_argument('--members', required=True)
    p.add_argument('--out', required=True); p.add_argument('--last-date', type=int, default=20220421)
    a = p.parse_args(); raw = Path(a.out+'_with_duplicates')
    build(a.tar, a.members, raw, a.last_date); print(json.dumps(dedupe(raw, a.out), indent=2))
