"""Audit KuaiRand-1K and create conservative temporal splits (Python + pandas).

Raw files are never modified. Full-month aggregate features are audited only.
Run from any directory: python prepare_data.py
"""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/KuaiRand-1K/data'
OUT = ROOT / 'data/processed'
REPORT = ROOT / 'reports'
CHUNK = 250_000
EXPECTED_MD5 = '6b0b9c8222d67fcd4c676218edca3f1f'
BINARY = ['is_click','is_like','is_follow','is_comment','is_forward','is_hate',
          'long_view','is_profile_enter','is_rand']


def log(message):
    print(message, flush=True)


def profile(path):
    """Scan every row; retain missingness, range, inferred types, and ID arrays."""
    rows, fields, ids = 0, {}, []
    for frame in pd.read_csv(path, chunksize=CHUNK):
        rows += len(frame)
        if not path.name.startswith('log_'):
            ids.append(frame.iloc[:, 0].to_numpy(copy=True))
        for col in frame:
            s = frame[col]
            rec = fields.setdefault(col, {'missing': 0, 'dtypes': set(), 'min': None, 'max': None})
            rec['missing'] += int(s.isna().sum())
            rec['dtypes'].add(str(s.dtype))
            if pd.api.types.is_numeric_dtype(s) and s.notna().any():
                lo, hi = float(s.min()), float(s.max())
                rec['min'] = lo if rec['min'] is None else min(lo, rec['min'])
                rec['max'] = hi if rec['max'] is None else max(hi, rec['max'])
    records = []
    for col, rec in fields.items():
        rec['dtypes'] = '|'.join(sorted(rec['dtypes']))
        records.append({'file': path.name, 'rows': rows, 'column': col,
                        **rec, 'missing_rate': rec['missing'] / rows})
    keys = np.concatenate(ids) if ids else None
    info = {'file': path.name, 'rows': rows, 'columns': len(fields),
            'bytes': path.stat().st_size}
    if keys is not None:
        info['missing_keys'] = int(pd.isna(keys).sum())
        info['duplicate_keys'] = int(pd.Series(keys).duplicated().sum())
    return records, info, keys


def save_csv(frame, path):
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=1) as stream:
        for start in range(0, len(frame), CHUNK):
            frame.iloc[start:start+CHUNK].to_csv(stream, index=False, header=start == 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'data/raw/KuaiRand-1K.tar.gz'
    digest = hashlib.md5()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_MD5:
        raise ValueError('Archive checksum does not match official KuaiRand-1K.')

    profiles, inventory, keysets = [], [], {}
    paths = sorted(RAW.glob('*.csv'))
    if len(paths) != 6:
        raise ValueError(f'Expected 6 CSV files, got {len(paths)}')
    for path in paths:
        log(f'Audit: {path.name}')
        columns, info, keys = profile(path)
        profiles.extend(columns)
        inventory.append(info)
        if keys is not None:
            if info['missing_keys'] or info['duplicate_keys']:
                raise ValueError(f'Non-unique/missing metadata keys: {path.name}')
            keysets[path.name] = keys
    pd.DataFrame(profiles).to_csv(REPORT / 'column_profile.csv', index=False)
    pd.DataFrame(inventory).to_csv(REPORT / 'file_inventory.csv', index=False)

    log('Load interaction logs')
    frames = []
    for path in paths:
        if not path.name.startswith('log_'):
            continue
        for frame in pd.read_csv(path, chunksize=CHUNK):
            # Validate before narrowing integer storage; retain any invalid values for quarantine.
            for col in frame:
                if pd.api.types.is_integer_dtype(frame[col]):
                    frame[col] = pd.to_numeric(frame[col], downcast='integer')
            frame['source_file'] = path.name
            frame['source_file'] = frame['source_file'].astype('category')
            frames.append(frame)
    events = pd.concat(frames, ignore_index=True)
    del frames
    events['source_file'] = events['source_file'].astype('category')
    event_cols = [c for c in events if c != 'source_file']
    raw_count = len(events)
    # Only full-record duplicates are removed, never repeated user-video pairs.
    dup = events.duplicated(subset=event_cols)
    duplicates = int(dup.sum())
    if duplicates:
        save_csv(events.loc[dup], OUT / 'exact_duplicates.csv.gz')
    events = events.loc[~dup].copy()
    invalid = events[event_cols].isna().any(axis=1)
    for col in BINARY:
        invalid |= ~events[col].isin([0, 1])
    invalid |= (events[['user_id','video_id','time_ms','play_time_ms','duration_ms',
                        'profile_stay_time','comment_stay_time']] < 0).any(axis=1)
    invalid |= ~events['tab'].between(0, 14)
    invalid |= ~events['date'].between(20220408, 20220508)
    source_random = events['source_file'].astype(str).str.startswith('log_random_')
    invalid |= source_random.ne(events['is_rand'].eq(1))
    invalid_count = int(invalid.sum())
    if invalid_count:
        save_csv(events.loc[invalid], OUT / 'quarantined_events.csv.gz')
    events = events.loc[~invalid].copy()
    del invalid, dup, source_random
    log(f'Clean events: {len(events):,}; exact duplicates: {duplicates:,}; quarantine: {invalid_count:,}')

    users = keysets['user_features_1k.csv']
    videos = keysets['video_features_basic_1k.csv']
    joins = {'missing_user_metadata_events': int((~events.user_id.isin(users)).sum()),
             'missing_video_metadata_events': int((~events.video_id.isin(videos)).sum())}
    # Timestamp remains UTC milliseconds. date is retained as publisher's calendar date.
    dates = pd.to_datetime(events.time_ms, unit='ms', utc=True)
    date_alignment = {}
    for zone in ['UTC', 'Asia/Shanghai']:
        local_dates = dates.dt.tz_convert(zone)
        mapped = local_dates.dt.year * 10000 + local_dates.dt.month * 100 + local_dates.dt.day
        date_alignment[zone] = float(mapped.eq(events.date).mean())
    # Use chronological timestamps for splitting; retain publisher date for audit.
    events['split_date'] = mapped.astype('int32')
    mismatch = events.split_date.ne(events.date)
    save_csv(events.loc[mismatch, ['user_id','video_id','time_ms','date','split_date','source_file']],
             REPORT/'date_mismatches.csv.gz')
    assert events.split_date.between(20220407,20220508).all(), 'Unexpected timestamp range'
    del dates, mapped, local_dates
    events.groupby(['date','is_rand'], observed=True).agg(
        events=('user_id','size'), users=('user_id','nunique'), videos=('video_id','nunique'),
        long_view_rate=('long_view','mean'), click_rate=('is_click','mean'),
        like_rate=('is_like','mean')).reset_index().to_csv(REPORT/'daily_summary.csv', index=False)
    events.groupby(['tab','is_rand'], observed=True).agg(
        events=('user_id','size'), long_view_rate=('long_view','mean'),
        click_rate=('is_click','mean'), like_rate=('is_like','mean')).reset_index().to_csv(
            REPORT/'scenario_summary.csv', index=False)
    user_counts = events.groupby('user_id').agg(events=('video_id','size'), positives=('long_view','sum'))
    user_counts.to_csv(REPORT/'user_activity.csv')
    item_counts = events.groupby('video_id').agg(exposures=('user_id','size'), positives=('long_view','sum'))
    quantiles = item_counts.quantile([0,.25,.5,.75,.9,.95,.99,1]).reset_index(names='quantile')
    quantiles.to_csv(REPORT/'item_activity_quantiles.csv', index=False)
    repeated_pairs = int(events.duplicated(['user_id','video_id']).sum())
    measures = events[['duration_ms','play_time_ms']].quantile([0,.25,.5,.75,.9,.95,.99,1])
    measures.to_csv(REPORT/'duration_quantiles.csv')
    anomalies = {'zero_duration_events': int(events.duration_ms.eq(0).sum()),
                 'watch_longer_than_video_events': int(events.play_time_ms.gt(events.duration_ms).sum())}
    del item_counts

    train_mask = events.is_rand.eq(0) & events.split_date.le(20220421)
    candidates = np.sort(events.loc[train_mask,'video_id'].unique())
    train_users = events.loc[train_mask,'user_id'].unique()
    pd.DataFrame({'video_id': candidates}).to_csv(OUT/'candidate_videos.csv', index=False)
    pd.DataFrame({'user_id': np.sort(train_users)}).to_csv(OUT/'training_users.csv', index=False)
    log(f'Candidate catalog: {len(candidates):,} videos')
    events['in_train_catalog'] = events.video_id.isin(candidates).astype('int8')
    events['known_train_user'] = events.user_id.isin(train_users).astype('int8')
    splits = [
        ('train', events.is_rand.eq(0) & events.split_date.le(20220421)),
        ('validation', events.is_rand.eq(0) & events.split_date.between(20220422,20220428)),
        ('test', events.is_rand.eq(0) & events.split_date.ge(20220429)),
        ('random_validation', events.is_rand.eq(1) & events.split_date.between(20220422,20220428)),
        ('random_test', events.is_rand.eq(1) & events.split_date.ge(20220429)),
    ]
    summaries = []
    assigned = 0
    for name, mask in splits:
        part = events.loc[mask].sort_values(['user_id','time_ms'], kind='stable')
        assigned += len(part)
        pos = part.long_view.eq(1)
        summaries.append({'split': name, 'events': len(part), 'users': int(part.user_id.nunique()),
            'videos': int(part.video_id.nunique()), 'positive_events': int(pos.sum()),
            'long_view_rate': float(part.long_view.mean()),
            'catalog_event_coverage': float(part.in_train_catalog.mean()),
            'catalog_positive_coverage': float(part.loc[pos,'in_train_catalog'].mean()),
            'known_user_coverage': float(part.known_train_user.mean()),
            'date_min': int(part.split_date.min()), 'date_max': int(part.split_date.max()),
            'time_ms_min': int(part.time_ms.min()), 'time_ms_max': int(part.time_ms.max())})
        log(f'Write {name}: {len(part):,} events')
        save_csv(part, OUT/f'{name}.csv.gz')
        del part
    assert assigned == len(events), 'Every retained event must be assigned once.'
    assert raw_count == len(events) + duplicates + invalid_count
    pd.DataFrame(summaries).to_csv(REPORT/'split_summary.csv', index=False)
    summary = {'archive_md5': digest.hexdigest(), 'raw_events': raw_count,
        'clean_events': len(events), 'exact_duplicates_removed': duplicates,
        'quarantined_events': invalid_count, 'users': int(events.user_id.nunique()),
        'videos': int(events.video_id.nunique()), 'repeated_pair_events_retained': repeated_pairs,
        'candidate_videos': len(candidates), 'date_alignment': date_alignment,
        'date_mismatch_events': int(mismatch.sum()),
        'metadata_join': joins, 'observations': anomalies, 'splits': summaries}
    (REPORT/'audit_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log('Finished. Reports and temporal splits are ready.')


if __name__ == '__main__':
    main()
