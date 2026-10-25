"""Read every saved split and check counts, labels, provenance and time boundaries."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
s = json.loads((ROOT/'reports/audit_summary.json').read_text())
result = []
for info in s['splits']:
    path = ROOT/'data/processed'/f"{info['split']}.csv.gz"
    count = pos = 0
    lo, hi = 99999999, 0
    tlo, thi = 10**18, 0
    for frame in pd.read_csv(path, usecols=['split_date','time_ms','long_view','is_rand'], chunksize=250_000):
        assert frame.is_rand.eq(int(info['split'].startswith('random_'))).all()
        assert frame.long_view.isin([0,1]).all()
        count += len(frame)
        pos += int(frame.long_view.sum())
        lo, hi = min(lo,int(frame.split_date.min())), max(hi,int(frame.split_date.max()))
        tlo, thi = min(tlo,int(frame.time_ms.min())), max(thi,int(frame.time_ms.max()))
    assert (count,pos,lo,hi,tlo,thi) == (
        info['events'],info['positive_events'],info['date_min'],info['date_max'],
        info['time_ms_min'],info['time_ms_max'])
    result.append({'file': path.name, 'rows': count, 'gzip_readable': True,
                   'counts_labels_dates_match': True})
    print(path.name, count, flush=True)
by_name = {row['split']: row for row in s['splits']}
assert by_name['train']['time_ms_max'] < by_name['validation']['time_ms_min']
assert by_name['validation']['time_ms_max'] < by_name['test']['time_ms_min']
assert by_name['random_validation']['time_ms_max'] < by_name['random_test']['time_ms_min']
assert sum(row['rows'] for row in result) == s['clean_events']
(ROOT/'reports/output_validation.json').write_text(json.dumps(
    {'strict_time_order':True, 'total_rows_match':True, 'files':result},indent=2))
print('All output checks passed.')
