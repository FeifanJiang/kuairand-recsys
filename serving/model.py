"""Two-stage recommender over the serving bundle: iALS inner-product retrieval -> seen filter with backfill -> M3 scoring.

Everything is loaded once in Recommender.__init__; a request only searches the index and looks up its candidates.
Unknown users (not in the user table, or without a pre-cutoff long view) receive candidates from the time-decayed
popularity list. as_of must be after the training cutoff; the default comes from the bundle manifest.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # torch and faiss-cpu each ship an OpenMP runtime on macOS; without this the process aborts (OMP Error #15)
import json
import sys
import time
from pathlib import Path
import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'scripts')); sys.path.insert(0, str(ROOT/'serving'))
from features import FeatureStore  # noqa: E402

class Recommender:
    def __init__(self, bundle, threads=8, load_hnsw=True):
        t0 = time.time(); bundle = Path(bundle); faiss.omp_set_num_threads(threads); self.threads = threads
        self.store = FeatureStore(bundle); self.manifest = self.store.manifest; t1 = time.time()
        self.flat = faiss.read_index(str(bundle/'faiss_flat_ip.index')); t2 = time.time()
        self.hnsw = faiss.read_index(str(bundle/'faiss_hnsw_ip.index')) if load_hnsw else None; t3 = time.time()
        seen = np.load(bundle/'seen.npz'); self.seen_users, self.seen_offsets, self.seen_items = seen['users'], seen['offsets'], seen['items']
        pop = np.load(bundle/'time_pop.npz'); self.pop_ids, self.pop_scores = pop['video_id'].astype(np.int64), pop['score']
        self.T = self.manifest['cutoff']['T_ms']; self.as_of_default = self.manifest['as_of_default_ms']; self.n_items = len(self.store.catalog)
        self.load_seconds = {'features_and_models': t1-t0, 'flat_index': t2-t1, 'hnsw_index': t3-t2, 'seen_and_pop': time.time()-t3, 'total': time.time()-t0}

    # ---- helpers ----
    def seen_set(self, user_id):
        i = np.searchsorted(self.seen_users, user_id)
        if i < len(self.seen_users) and self.seen_users[i] == user_id: return self.seen_items[self.seen_offsets[i]:self.seen_offsets[i+1]].astype(np.int64)
        return np.empty(0, np.int64)

    def user_index(self, user_id):
        u = self.store.ials_users; i = np.searchsorted(u, user_id)
        return int(i) if i < len(u) and u[i] == user_id and self.store.user_valid[i] else None

    def check_as_of(self, as_of_ms):
        as_of_ms = self.as_of_default if as_of_ms is None else int(as_of_ms)
        if as_of_ms <= self.T: raise ValueError(f'as_of must be after the training cutoff ({self.T}); got {as_of_ms}')
        return as_of_ms

    # ---- retrieval ----
    def retrieve(self, user_id, n_candidates=200, index='flat', ef_search=128):
        seen = self.seen_set(user_id); ui = self.user_index(user_id)
        if ui is None:                                                        # unknown or zero-factor user: popularity candidates
            keep = ~np.isin(self.pop_ids, seen); ids = self.pop_ids[keep][:n_candidates]; sc = self.pop_scores[keep][:n_candidates]
            return {'video_ids': ids, 'retrieval_scores': sc.astype(np.float64), 'source': 'time_popularity', 'fetched': int(len(self.pop_ids)), 'rounds': 1}
        idx = self.flat if index == 'flat' else self.hnsw
        if idx is None: raise ValueError('hnsw index not loaded')
        if index == 'hnsw': idx.hnsw.efSearch = int(ef_search)
        x = self.store.X[ui][None, :]; k = min(n_candidates+len(seen)+50, self.n_items); rounds = 0
        while True:
            rounds += 1; D, I = idx.search(x, k); I, D = I[0], D[0]; ok = I >= 0; I, D = I[ok], D[ok]
            ids = self.store.catalog[I]; keep = ~np.isin(ids, seen); ids, D = ids[keep], D[keep]
            if len(ids) >= n_candidates or k >= self.n_items: break
            k = min(2*k, self.n_items)
        return {'video_ids': ids[:n_candidates], 'retrieval_scores': D[:n_candidates].astype(np.float64), 'source': f'ials_{index}', 'fetched': int(k), 'rounds': rounds}

    # ---- scoring ----
    def score(self, user_id, video_ids, as_of_ms=None, duration_ms=None):
        as_of_ms = self.check_as_of(as_of_ms); feats = self.store.build_features(user_id, video_ids, as_of_ms, duration_ms)
        return self.store.m3_scores(feats), feats

    def score_pairs(self, user_ids, video_ids, as_of_ms, duration_ms=None):
        """Row-wise scoring (one as_of per row) for the correctness check."""
        feats = self.store.build_features(np.asarray(user_ids), np.asarray(video_ids), np.asarray(as_of_ms), duration_ms); return self.store.m3_scores(feats), feats

    def recommend(self, user_id, as_of_ms=None, k=20, n_candidates=200, index='flat', ef_search=128):
        as_of_ms = self.check_as_of(as_of_ms); t0 = time.time(); r = self.retrieve(int(user_id), n_candidates, index, ef_search); t1 = time.time()
        if len(r['video_ids']) == 0: return {'user_id': int(user_id), 'as_of_ms': as_of_ms, 'items': [], 'candidates': 0, 'source': r['source']}
        scores, feats = self.score(int(user_id), r['video_ids'], as_of_ms); t2 = time.time(); order = np.argsort(-scores, kind='stable')[:k]
        items = [{'video_id': int(r['video_ids'][i]), 'score': float(scores[i]), 'retrieval_rank': int(i)+1, 'retrieval_score': float(r['retrieval_scores'][i])} for i in order]
        return {'user_id': int(user_id), 'as_of_ms': as_of_ms, 'user_known': self.user_index(int(user_id)) is not None, 'source': r['source'], 'candidates': int(len(r['video_ids'])),
                'seen_filtered': int(len(self.seen_set(int(user_id)))), 'fetched': r['fetched'], 'rounds': r['rounds'], 'items': items, 'bundle_version': self.manifest['bundle_version'],
                'timing_ms': {'retrieval': (t1-t0)*1000, 'features_and_scoring': (t2-t1)*1000, 'total': (time.time()-t0)*1000}}
