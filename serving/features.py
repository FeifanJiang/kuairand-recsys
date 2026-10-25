"""Feature construction for arbitrary (user, video) pairs at a request time, without labels.

FeatureStore.build_features(user_ids, video_ids, as_of_ms) returns the standardized LR-L2 design matrix (float32,
112 columns) plus the standardized iALS score and the has-factors flag, exactly as the offline functions produce
them for evaluation rows: video attributes and duration from the video table, frozen-at-cutoff statistics through
sorted-key lookups with the semantics of feature_model.query (missing keys -> zero counts and a missing last long
view; decayed counts aged to as_of), then feature_model.assemble and feature_matrix.standardized with the LR-L2
training statistics. The only inputs are ids and as_of; the service never reads an exposure row.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import feature_model as fm
import feature_matrix as fx

ROOT = Path(__file__).resolve().parents[1]; VMETA = ROOT/'data/k27/video_meta'
DECAY_TAUS = {'v': fm.TAUS, 'a': fm.TAUS}

def composite(a, b):
    return (np.asarray(a, np.int64) << 32) | np.asarray(b, np.int64)

class Lookup:
    """Sorted int64 keys -> columns; get() returns positions and a hit mask."""
    def __init__(self, path):
        z = np.load(path); self.keys = z['keys']; self.cols = {c: z[c] for c in z.files if c != 'keys'}
    def get(self, q):
        q = np.asarray(q, np.int64); i = np.minimum(np.searchsorted(self.keys, q), len(self.keys)-1); return i, self.keys[i] == q
    def column(self, name, i, hit, missing=0.):
        v = self.cols[name][i].astype(np.float64); return np.where(hit, v, missing)

class FeatureStore:
    def __init__(self, bundle):
        bundle = Path(bundle); self.manifest = json.loads((bundle/'manifest.json').read_text()); self.T0 = self.manifest['cutoff']['T0_ms']
        meta = json.loads((VMETA/'meta.json').read_text()); self.vocab = meta['vocab']
        self.video_ids = np.load(VMETA/'video_id.npy'); self.author = np.load(VMETA/'author_id.npy'); self.upload_day = np.load(VMETA/'upload_day.npy')
        self.cat = {c: np.load(VMETA/f'{c}.npy') for c in ['tag1', 'video_type', 'upload_type', 'music_type']}
        self.duration = np.load(bundle/'duration_ms.npy'); self.stats = {k: Lookup(bundle/f'stats_{k}.npz') for k in fm.KEYS}
        g = self.stats['g']; self.g_n, self.g_l = float(g.cols['g_n'][0]), float(g.cols['g_l'][0])
        models = joblib.load(bundle/'models.joblib'); self.lr = models['lr_l2']; self.m3 = models['m3']; self.const = models['constants']
        z = np.load(bundle/'ials.npz'); self.X, self.Y = z['X'], z['Y']; self.ials_users, self.catalog = z['users'], z['catalog']; self.user_valid, self.item_valid = z['user_valid'], z['item_valid']

    # ---- video table ----
    def video_positions(self, video_ids):
        v = np.asarray(video_ids, np.int64); i = np.minimum(np.searchsorted(self.video_ids, v), len(self.video_ids)-1); hit = self.video_ids[i] == v
        if not hit.all(): raise KeyError(f'unknown video ids: {v[~hit][:5].tolist()}')
        return i

    def frame(self, user_ids, video_ids, as_of_ms, duration_ms=None):
        """Rows in the layout of k27_eval_rows.attach_video (no label is involved)."""
        pos = self.video_positions(video_ids); n = len(pos); t = np.broadcast_to(np.asarray(as_of_ms, np.int64), (n,)).copy()
        dur = self.duration[pos].astype(np.float64) if duration_ms is None else np.asarray(duration_ms, np.float64); dur = np.where(dur < 0, 0., dur)
        f = pd.DataFrame({'user_id': np.broadcast_to(np.asarray(user_ids, np.int64), (n,)).copy(), 'video_id': np.asarray(video_ids, np.int64), 't': t, 'lv': 0, 'duration_ms': dur, 'tab': 0, 'author_id': self.author[pos]})
        for c in ['tag1', 'video_type', 'upload_type', 'music_type']: f[c] = np.array(self.vocab[c], dtype=object)[self.cat[c][pos]]
        day = self.upload_day[pos]; f['upload_day'] = pd.to_datetime(np.where(day >= 0, day, 0), unit='D').where(day >= 0)
        f['const'] = 0; f['lv_logdur'] = 0.; f['lv_short'] = 0; f['_tag_code'] = self.cat['tag1'][pos]
        return f

    # ---- frozen statistics with feature_model.query semantics ----
    def history(self, f):
        t = f.t.to_numpy(np.float64); s = pd.DataFrame(index=f.index); s['g_n'] = self.g_n; s['g_l'] = self.g_l
        for key, q in [('v', f.video_id.to_numpy()), ('a', f.author_id.to_numpy()), ('tg', f._tag_code.to_numpy()), ('u', f.user_id.to_numpy()),
                       ('ua', composite(f.user_id.to_numpy(), f.author_id.to_numpy())), ('ut', composite(f.user_id.to_numpy(), f._tag_code.to_numpy()))]:
            lk = self.stats[key]; i, hit = lk.get(q)
            for c in lk.cols:
                if c.endswith('lastlv'): s[c] = lk.column(c, i, hit, np.nan)
                elif '_E' in c:
                    tau = float(c.split('_E')[1]); s[c.replace('_E', '_d')] = lk.column(c, i, hit, 0.)*np.exp(-(t-self.T0)/(tau*fm.DAY))
                else: s[c] = lk.column(c, i, hit, 0.)
        return s

    # ---- iALS inputs ----
    def ials_inputs(self, user_ids, video_ids):
        u = np.asarray(user_ids, np.int64); v = np.asarray(video_ids, np.int64)
        ui = np.minimum(np.searchsorted(self.ials_users, u), len(self.ials_users)-1); uok = (self.ials_users[ui] == u) & self.user_valid[ui]
        vi = np.minimum(np.searchsorted(self.catalog, v), len(self.catalog)-1); vok = (self.catalog[vi] == v) & self.item_valid[vi]
        ok = uok & vok; score = np.where(ok, (self.X[ui]*self.Y[vi]).sum(1).astype(np.float64), 0.)
        c = self.const['ials_score_mean_sd_seed0']; return (score-c[0])/c[1], ok.astype(np.float64), score

    # ---- full design ----
    def build_features(self, user_ids, video_ids, as_of_ms, duration_ms=None):
        f = self.frame(user_ids, video_ids, as_of_ms, duration_ms); s = self.history(f)
        X, _ = fx.standardized(f, s, 'L2', self.lr['levels'], stats=(self.lr['columns'], self.lr['mean'], self.lr['sd']))
        z, flag, raw = self.ials_inputs(f.user_id.to_numpy(), f.video_id.to_numpy())
        return {'X': X, 'z_ials': z, 'has_factors': flag, 'ials_score': raw, 'frame': f, 'stats': s}

    def m3_scores(self, feats):
        D = np.concatenate([feats['X'].astype(np.float64), np.stack([feats['z_ials'], feats['has_factors']], 1)], 1)
        return D@self.m3['w']+self.m3['b']
