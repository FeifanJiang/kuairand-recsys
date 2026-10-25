"""Fixed tune/report split inside the development users (no log or label is read).

All development users from data/splits/expansion_users.csv, sorted by Pure id, permuted with a fixed seed;
the first half tunes hyperparameters, the second half reports. Users without evaluation rows simply
contribute nothing; the split is not conditioned on labels or on having both outcomes.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261022

def main():
    p = pd.read_csv(ROOT/'data/splits/expansion_users.csv')
    dev = np.sort(p.loc[p.partition.eq('development'), 'pure_user_id'].to_numpy())
    rank = np.argsort(np.random.default_rng(SEED).permutation(len(dev)))
    out = pd.DataFrame({'pure_user_id': dev, 'group': np.where(rank < len(dev)//2, 'tune', 'report')})
    dest = ROOT/'data/splits/development_tune_report.csv'; out.to_csv(dest, index=False)
    meta = {'seed': SEED, 'source': 'development users of expansion_users.csv', 'labels_read': False,
            'counts': out.group.value_counts().to_dict(), 'csv_sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    (ROOT/'data/splits/development_tune_report_meta.json').write_text(json.dumps(meta, indent=2)); print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
