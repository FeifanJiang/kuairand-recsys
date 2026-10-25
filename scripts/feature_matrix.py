"""Memory-bounded feature matrices built from feature_model.assemble, chunk by chunk.

assemble is row-wise once the categorical levels are fixed, so calling it on row chunks with the levels of
the full training frame gives the same rows as one call on everything. Standardization uses float64 column
means and sample standard deviations (ddof = 1, zero replaced by one, as in feature_model.main) accumulated
over chunks; the scaled chunk is cast to float32 and written into one preallocated array. For trees the
numeric columns are kept unscaled and upload_type / music_type / tag1 become integer category codes (L1 or L2 numeric set).
"""
import numpy as np
import pandas as pd
import feature_model as fm

CATS = ['upload_type', 'music_type', 'tag1']

def training_levels(f):
    """The levels assemble derives from the training frame (levels=None branch)."""
    return {'upload_type': [v for v, c in f.upload_type.value_counts().items() if c >= 1000],
            'music_type': sorted(f.music_type.unique().tolist()), 'tag1': f.tag1.value_counts().index[:50].tolist()}

def _chunks(n, size):
    for i in range(0, n, size): yield slice(i, min(i+size, n))

def standardized(f, s, level, levels, chunk=500_000, stats=None):
    """float32 standardized matrix; stats=(columns, mean, sd) from training rows, computed here when None."""
    if stats is None:
        cols = None; n = 0; total = sq = None
        for sl in _chunks(len(f), chunk):                                   # pass 1: column sums (float64)
            X = fm.assemble(f.iloc[sl], s.iloc[sl], level, levels)[0]; cols = cols or list(X.columns)
            v = X[cols].to_numpy(np.float64); total = v.sum(0) if total is None else total+v.sum(0); n += len(v)
        mean = total/n
        for sl in _chunks(len(f), chunk):                                   # pass 2: centred squares
            v = fm.assemble(f.iloc[sl], s.iloc[sl], level, levels)[0][cols].to_numpy(np.float64)-mean
            sq = (v*v).sum(0) if sq is None else sq+(v*v).sum(0)
        sd = np.sqrt(sq/(n-1)); sd[sd == 0] = 1; stats = (cols, mean, sd)
    cols, mean, sd = stats; out = np.empty((len(f), len(cols)), np.float32)
    for sl in _chunks(len(f), chunk):
        out[sl] = ((fm.assemble(f.iloc[sl], s.iloc[sl], level, levels)[0][cols].to_numpy(np.float64)-mean)/sd).astype(np.float32)
    return out, stats

def tree_matrix(f, s, levels, chunk=500_000, level='L2'):
    """float32 numeric columns of `level` plus category codes; returns (array, column names, categorical mask)."""
    cats = {c: list(dict.fromkeys(levels[c]+["other", "missing"])) for c in CATS}; cols = None
    out = None
    for sl in _chunks(len(f), chunk):
        X = fm.assemble(f.iloc[sl], s.iloc[sl], level, levels)[0]; num = [c for c in X.columns if '=' not in c]; cols = cols or num
        block = np.empty((len(X), len(cols)+len(CATS)), np.float32); block[:, :len(cols)] = X[cols].to_numpy(np.float32)
        for j, c in enumerate(CATS):
            v = f[c].iloc[sl].astype(str).where(f[c].iloc[sl].notna(), 'missing'); v = v.where(v.isin(levels[c]) | (v == 'missing'), 'other')
            block[:, len(cols)+j] = pd.Categorical(v, categories=cats[c]).codes
        if out is None: out = np.empty((len(f), block.shape[1]), np.float32)
        out[sl] = block
    return out, cols+CATS, np.array([False]*len(cols)+[True]*len(CATS))
