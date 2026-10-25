"""Fixed development/confirmation split of the KuaiRand users not used in the 1K project.

Universe: every user in KuaiRand-Pure user_features (the full user table) minus the 1,000 users mapped from
KuaiRand-1K. No log or label is read. Users are sorted by Pure id and permuted with a fixed seed; the first
half is development, the rest is confirmation. Confirmation users' random-exposure labels stay sealed.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PURE = ROOT/'data/raw/pure_download'
SEED = 20261021

def main():
    users = pd.read_csv(PURE/'KuaiRand-Pure/data/user_features_pure.csv', usecols=['user_id']).user_id
    assert users.is_unique
    used = pd.read_csv(PURE/'user_map_1k_to_pure.csv')
    assert len(used) == 1000 and used.user_id_pure.isin(users).all()
    unused = np.sort(users[~users.isin(used.user_id_pure)].to_numpy())
    order = np.random.default_rng(SEED).permutation(len(unused))
    part = np.where(np.argsort(order) < len(unused)//2, 'development', 'confirmation')   # rank in permutation
    out = pd.DataFrame({'pure_user_id': unused, 'partition': part})
    dest = ROOT/'data/splits/expansion_users.csv'; out.to_csv(dest, index=False)
    meta = {'seed': SEED, 'universe': 'Pure user_features minus 1K-mapped users', 'labels_read': False,
            'users_total_pure': int(len(users)), 'users_excluded_1k': int(len(used)), 'users_unused': int(len(unused)),
            'counts': out.partition.value_counts().to_dict(), 'csv_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
            'map_column_used': 'user_id_pure'}
    (ROOT/'data/splits/expansion_users_meta.json').write_text(json.dumps(meta, indent=2)); print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
