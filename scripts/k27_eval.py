"""Scoring and evaluation helpers shared by the 27K experiment and its 1K consistency checks.

Per-user AUC is train_ranker.per_user_auc (average ranks, ties counted half). For stochastic models the
per-user AUC is computed per seed and averaged per user (repeated-run statistic, not an ensemble). Paired
comparisons resample users with replacement (2000 draws, seed 2026) on users present in both inputs.
"""
import numpy as np
import pandas as pd
import torch
import feature_matrix as fx
from train_ranker import per_user_auc

def score_lr(f, s, model, chunk=500_000):
    """Logits of a saved logistic model {'w','b','columns','mean','sd','levels','level'} on rows f with history s."""
    X, _ = fx.standardized(f, s, model['level'], model['levels'], chunk, stats=(model['columns'], np.asarray(model['mean'], float), np.asarray(model['sd'], float)))
    with torch.no_grad(): return (torch.from_numpy(X)@model['w'].float()+model['b'].float()).numpy().astype(np.float64)

def score_ials(f, X, Y, users, catalog):
    """x_u . y_v; users or videos without factors score 0."""
    def locate(keys, ids):
        i = np.searchsorted(keys, ids); i = np.minimum(i, len(keys)-1); return i, keys[i] == ids
    ui, uok = locate(np.asarray(users), f.user_id.to_numpy()); vi, vok = locate(np.asarray(catalog), f.video_id.to_numpy())
    s = np.zeros(len(f)); idx = np.flatnonzero(uok & vok)
    with torch.no_grad(): s[idx] = (X[torch.from_numpy(ui[idx])]*Y[torch.from_numpy(vi[idx])]).sum(-1).numpy()
    return s

def user_auc(f, scores, mask=None):
    """Per-user AUC Series; scores is one array or a list of per-seed arrays (averaged per user)."""
    m = np.ones(len(f), bool) if mask is None else np.asarray(mask)
    runs = scores if isinstance(scores, list) else [scores]
    return pd.concat([per_user_auc(f.user_id.to_numpy()[m], f.lv.to_numpy()[m], np.asarray(s)[m]) for s in runs], axis=1).mean(axis=1)

def paired(a, b, users=None, draws=2000, seed=2026):
    common = a.index.intersection(b.index) if users is None else pd.Index(users)
    d = (a.loc[common]-b.loc[common]).to_numpy(); rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(draws, len(d)), replace=True).mean(axis=1)
    return {'users': int(len(d)), 'a': float(a.loc[common].mean()), 'b': float(b.loc[common].mean()), 'difference': float(d.mean()),
            'ci95': np.quantile(boot, [.025, .975]).tolist()}
