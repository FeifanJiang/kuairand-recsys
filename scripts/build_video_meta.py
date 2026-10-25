"""Stream the KuaiRand video_features_basic table out of the release tar.gz into compact columns.

Columns follow feature_model.load_meta: author_id, tag1 (first listed tag, 'missing' when absent),
upload_day (days since 1970-01-01, -1 when unparseable), video_type, upload_type, music_type
(str of the raw value, 'missing' when absent). Strings are stored as int codes with vocabularies in meta.json.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_event_store import stream_members

CODED = ['tag1', 'video_type', 'upload_type', 'music_type']
CHUNK = 2_000_000

def transform(m):
    out = pd.DataFrame({'video_id': m.video_id, 'author_id': m.author_id})
    out['tag1'] = m.tag.astype(str).str.split(',').str[0].where(m.tag.notna(), 'missing')
    day = pd.to_datetime(m.upload_dt, errors='coerce')
    out['upload_day'] = np.where(day.notna(), (day - pd.Timestamp('1970-01-01')).dt.days.fillna(-1), -1).astype(np.int32)
    out['video_type'] = m.video_type.astype(str).where(m.video_type.notna(), 'missing')
    out['upload_type'] = m.upload_type.astype(str).where(m.upload_type.notna(), 'missing')
    out['music_type'] = m.music_type.astype(str).where(m.music_type.notna(), 'missing')
    return out

def build(tar, pattern, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False); vocab = {c: {} for c in CODED}; cols = {c: [] for c in ['video_id', 'author_id', 'upload_day']+CODED}
    members = stream_members(tar, pattern); name, fh = next(members)            # keep the generator alive: it owns the open tar
    for chunk in pd.read_csv(fh, usecols=['video_id','author_id','tag','upload_dt','video_type','upload_type','music_type'], chunksize=CHUNK):
        t = transform(chunk)
        for c in ['video_id', 'author_id', 'upload_day']: cols[c].append(t[c].to_numpy().astype(np.int64))
        for c in CODED:
            v = vocab[c]
            for s in pd.unique(t[c]): v.setdefault(s, len(v))
            cols[c].append(t[c].map(v).to_numpy().astype(np.int32))
    arrays = {c: np.concatenate(v) for c, v in cols.items()}
    assert len(np.unique(arrays['video_id'])) == len(arrays['video_id'])
    o = np.argsort(arrays['video_id'])
    for c, a in arrays.items():
        a = a[o]; dt = np.int64 if c == 'author_id' and a.max() > np.iinfo(np.int32).max else np.int32
        np.save(out/f'{c}.npy', a.astype(dt))
    meta = {'source': str(tar), 'member': name, 'rows': int(len(o)), 'vocab': {c: list(v) for c, v in vocab.items()}}
    (out/'meta.json').write_text(json.dumps(meta)); return {k: v for k, v in meta.items() if k != 'vocab'}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--tar', required=True); p.add_argument('--members', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); print(json.dumps(build(a.tar, a.members, a.out), indent=2))
