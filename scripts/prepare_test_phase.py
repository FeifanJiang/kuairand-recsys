"""Build the final test-phase inputs and freeze the test protocol before any test read.

Training = standard train + validation (2022-04-07 to 04-28). The catalog and known users
are redefined on that period. Test and random-test files are linked, not read, here.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
OUT = DATA/'test_phase'
REPORT = ROOT/'reports/test_phase'

PROTOCOL = {
    'training_period': 'standard train + validation, 2022-04-07 to 2022-04-28',
    'catalog': 'videos in standard train + validation logs',
    'known_users': 'users in standard train + validation logs',
    'models': {'popularity': 'long_view event count over training period, ties by ascending video_id',
               'lsm': {'dim': 16, 'l2': 10, 'penalty': 'gaussian', 'epochs': 10, 'negatives': 4,
                       'seeds': [42, 1, 2, 3, 4]}},
    'primary_metric': 'standard test macro Recall@100, LSM seed mean vs popularity; per-user recall averaged '
                      'over seeds, paired user bootstrap 2000 resamples seed 2026',
    'secondary_metrics': ['Recall@20 and Recall@50', 'catalog coverage of test positives',
                          'random_test AUC of retrieval scores for known users and in-catalog videos'],
    'commitment': 'No tuning after test access; results reported whatever they show',
}

def main():
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(DATA/f'{name}.csv.gz') for name in ['train', 'validation']]
    train = pd.concat(frames, ignore_index=True).sort_values(['user_id','time_ms'], kind='stable')
    assert train.is_rand.eq(0).all()
    train.to_csv(OUT/'train.csv.gz', index=False, compression='gzip')
    catalog = pd.DataFrame({'video_id': sorted(train.video_id.unique())})
    users = pd.DataFrame({'user_id': sorted(train.user_id.unique())})
    catalog.to_csv(OUT/'candidate_videos.csv', index=False); users.to_csv(OUT/'training_users.csv', index=False)
    for name in ['test', 'random_test']:
        link = OUT/f'{name}.csv.gz'
        if not link.exists(): link.symlink_to(DATA/f'{name}.csv.gz')
    split = pd.read_csv(ROOT/'reports/split_summary.csv').set_index('split')
    assert int(train.time_ms.max()) < int(split.loc['test','time_ms_min']) and int(train.time_ms.max()) < int(split.loc['random_test','time_ms_min'])
    protocol = dict(PROTOCOL, frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                    training_rows=len(train), training_positive_events=int(train.long_view.sum()),
                    catalog_size=len(catalog), known_users=len(users), training_max_time_ms=int(train.time_ms.max()),
                    test_accessed_when_frozen=False)
    (REPORT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps(protocol, indent=2))

if __name__ == '__main__':
    main()
