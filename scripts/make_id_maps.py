"""Regenerate the 1K -> Pure id maps read by split_expansion_users.py.

The original maps were built by matching id-free fingerprints of every 1K random exposure to KuaiRand-Pure
(reports/data_feasibility/pure_mapping_check.json); the mapping turned out to be the identity for users and for
random-pool videos, and the 27K release shares Pure's user table and random log byte for byte
(reports/expansion/k27_mapping_check.json). This script rebuilds the identity maps and asserts that every 1K user
and every video of the 1K random log exists in the Pure tables.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; K1 = ROOT/'data/raw/KuaiRand-1K/data'; PURE = ROOT/'data/raw/pure_download'

def main():
    users_1k = pd.read_csv(K1/'user_features_1k.csv', usecols=['user_id']).user_id.drop_duplicates().sort_values()
    videos_1k = pd.read_csv(K1/'log_random_4_22_to_5_08_1k.csv', usecols=['video_id']).video_id.drop_duplicates().sort_values()
    users_pure = set(pd.read_csv(PURE/'KuaiRand-Pure/data/user_features_pure.csv', usecols=['user_id']).user_id)
    videos_pure = set(pd.read_csv(PURE/'KuaiRand-Pure/data/video_features_basic_pure.csv', usecols=['video_id']).video_id)
    assert users_1k.isin(users_pure).all() and videos_1k.isin(videos_pure).all(), '1K ids missing from Pure: the identity mapping does not hold'
    pd.DataFrame({'user_id_1k': users_1k.to_numpy(), 'user_id_pure': users_1k.to_numpy()}).to_csv(PURE/'user_map_1k_to_pure.csv', index=False)
    pd.DataFrame({'video_id_1k': videos_1k.to_numpy(), 'video_id_pure': videos_1k.to_numpy()}).to_csv(PURE/'video_map_1k_to_pure.csv', index=False)
    print({'users': int(len(users_1k)), 'random_videos': int(len(videos_1k))})

if __name__ == '__main__':
    main()
