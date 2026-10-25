"""Evaluation and training rows.

Evaluation rows: random exposures of given users in a date window, cleaned with the prepare_data.py rules.

Rows are filtered to the requested users chunk by chunk before anything else is computed, so labels of
other users (the sealed confirmation users) are never aggregated. Cleaning: invalid values (missing,
non-binary flags, negative ids/times/durations, tab outside 0-14, publisher date outside 2022-04-08..05-08,
is_rand != 1 in a random log) are dropped, then full-record duplicates; the window uses the Asia/Shanghai
date of time_ms. Video attributes come from the compact video_meta store (same encoding as
feature_model.load_meta), and the history features are feature_model.query on frozen-at-cutoff tables. training_frame loads the first n rows of a
k27_features.py sample (nested prefix of its draw ranking) with their prequential statistics.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import feature_model as fm
import build_event_store as bes

ROOT = Path(__file__).resolve().parents[1]; VMETA = ROOT/'data/k27/video_meta'

def random_rows(path, users, first_date, last_date):
    users = np.asarray(users); parts = []; counts = {'rows_of_users': 0, 'invalid': 0}
    for chunk in pd.read_csv(path, chunksize=1_000_000):
        chunk = chunk.loc[chunk.user_id.isin(users)]                        # other users' rows are dropped first
        invalid = chunk.isna().any(axis=1).to_numpy().copy()
        for c in bes.BINARY: invalid |= ~chunk[c].isin([0, 1]).to_numpy()
        invalid |= (chunk[bes.NONNEG] < 0).any(axis=1).to_numpy()
        invalid |= ~chunk.tab.between(0, 14).to_numpy(); invalid |= ~chunk.date.between(20220408, 20220508).to_numpy()
        invalid |= chunk.is_rand.ne(1).to_numpy()
        counts['rows_of_users'] += len(chunk); counts['invalid'] += int(invalid.sum()); parts.append(chunk.loc[~invalid])
    ev = pd.concat(parts, ignore_index=True); dup = ev.duplicated(); counts['full_record_duplicates'] = int(dup.sum()); ev = ev.loc[~dup]
    d = bes.shanghai_date(ev.time_ms.to_numpy()); ev = ev.loc[(d >= first_date) & (d <= last_date)].reset_index(drop=True)
    counts['rows_in_window'] = len(ev); return ev, counts

def attach_video(ev):
    """Frame in the layout of feature_model.load_frame, attributes from video_meta."""
    vid = np.load(VMETA/'video_id.npy'); pos = np.searchsorted(vid, ev.video_id.to_numpy()); assert (vid[pos] == ev.video_id.to_numpy()).all()
    vocab = json.loads((VMETA/'meta.json').read_text())['vocab']
    f = pd.DataFrame({'user_id': ev.user_id.to_numpy(), 'video_id': ev.video_id.to_numpy(), 't': ev.time_ms.to_numpy(), 'lv': ev.long_view.to_numpy(),
                      'duration_ms': ev.duration_ms.to_numpy(), 'tab': ev.tab.to_numpy(), 'author_id': np.load(VMETA/'author_id.npy', mmap_mode='r')[pos]})
    for c in ['tag1', 'video_type', 'upload_type', 'music_type']: f[c] = np.array(vocab[c], dtype=object)[np.load(VMETA/f'{c}.npy', mmap_mode='r')[pos]]
    day = np.load(VMETA/'upload_day.npy', mmap_mode='r')[pos]
    f['upload_day'] = pd.to_datetime(np.where(day >= 0, day, 0), unit='D').where(day >= 0)
    f['const'] = 0; f['lv_logdur'] = f.lv*np.log1p(f.duration_ms/1000); f['lv_short'] = f.lv*(f.duration_ms <= 18000)
    return f

def training_frame(d, n):
    keep = np.sort(np.load(d/'sample_ranked_rows.npy')[:n])                  # nested prefix of the draw ranking
    rows = pd.read_pickle(d/'rows.pkl').set_index('row').loc[keep]
    s = pd.concat([pd.read_pickle(d/f'pre_{k}.pkl').loc[keep] for k in fm.KEYS], axis=1)
    vocab = json.loads((VMETA/'meta.json').read_text())['vocab']; pos = rows.video_pos.to_numpy()
    f = pd.DataFrame({'t': rows.t.to_numpy(), 'duration_ms': rows.duration_ms.to_numpy(), 'lv': rows.lv.to_numpy()}, index=rows.index)
    for c in ['tag1', 'video_type', 'upload_type', 'music_type']: f[c] = np.array(vocab[c], dtype=object)[np.load(VMETA/f'{c}.npy', mmap_mode='r')[pos]]
    day = np.load(VMETA/'upload_day.npy', mmap_mode='r')[pos]
    f['upload_day'] = pd.to_datetime(np.where(day >= 0, day, 0), unit='D').where(day >= 0)
    return f, s

def history(f, tables, T0, tag_codes=None):
    """feature_model.query; tag1 is matched through its integer code when the tables were built on codes."""
    q = f if tag_codes is None else f.assign(tag1=f.tag1.map(tag_codes))
    return fm.query(q, tables, T0)
