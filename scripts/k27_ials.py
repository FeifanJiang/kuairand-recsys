"""Time-decay iALS on the 27K pre-cutoff long views (all users), reusing ials_confidence.ials unchanged.

Edges: every (user, video) pair with >= 1 cleaned pre-cutoff standard long view; confidence
c = 1 + alpha * log(1 + sum exp(-(T - t) / tau)) with T the last pre-cutoff timestamp (as 1K). Items are the
videos with a long view; a video with exposures but no long view would solve to the zero vector, so it is
scored 0 without being stored. Users are every user in the store. Factors are saved per (setting, seed);
no evaluation row is read here.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import build_event_store as bes
import ials_confidence as ic

ROOT = Path(__file__).resolve().parents[1]; STORE = ROOT/'data/k27/standard_to_0421_clean'

def edges(taus, before_date=None):
    """Long-view edges; with before_date only long views whose Asia/Shanghai date is <= before_date (early model), T = their last timestamp."""
    s = bes.load(STORE); lv = np.asarray(s['long_view']) == 1
    u = np.asarray(s['user_id'])[lv].astype(np.int64); v = np.asarray(s['video_id'])[lv].astype(np.int64); t = np.asarray(s['time_ms'])[lv]
    if before_date is not None:
        keep = bes.shanghai_date(t) <= before_date; u, v, t = u[keep], v[keep], t[keep]
    T = int(t.max())
    users = np.unique(np.asarray(s['user_id'])).astype(np.int64); catalog = np.unique(v)
    key = np.searchsorted(users, u)*len(catalog)+np.searchsorted(catalog, v); age = (T-t)/86_400_000
    stats = pd.DataFrame({'key': key, 'count': 1, **{f'decay_{x:g}': np.exp(-age/x) for x in taus}}).groupby('key').sum().sort_index()
    return users, catalog, stats.index.to_numpy(np.int64), stats, T

CACHE = ROOT/'artifacts/k27_experiment/ials_edges.npz'; TAUS = (0.5, 2.0)

def cached_edges(tau):
    """Edges from the cache written by --build-cache (same arrays as edges(), built once to save memory)."""
    z = np.load(CACHE); stats = pd.DataFrame({f'decay_{tau:g}': z[f'decay_{tau:g}']}, index=z['keys'])
    return z['users'], z['catalog'], z['keys'], stats, int(z['T'])

def build_cache():
    users, catalog, keys, stats, T = edges(TAUS)
    np.savez(CACHE, users=users, catalog=catalog, keys=keys, T=T, **{f'decay_{t:g}': stats[f'decay_{t:g}'].to_numpy() for t in TAUS})
    ok = {}
    for t in TAUS:                                                       # the cache must equal a single-tau build
        u1, c1, k1, s1, T1 = edges([t]); u2, c2, k2, s2, T2 = cached_edges(t)
        ok[f'tau{t:g}'] = bool(np.array_equal(u1, u2) and np.array_equal(c1, c2) and np.array_equal(k1, k2) and T1 == T2
                               and np.array_equal(s1[f'decay_{t:g}'].to_numpy(), s2[f'decay_{t:g}'].to_numpy()))
    print(json.dumps({'cache_identical_to_single_tau_build': ok})); assert all(ok.values())

def main():
    p = argparse.ArgumentParser(); p.add_argument('--tau', type=float, required=True); p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--l2', type=float, required=True); p.add_argument('--seed', type=int, default=0); p.add_argument('--out', required=True)
    p.add_argument('--before-date', type=int, help='early model: only long views with Shanghai date <= this'); a = p.parse_args()
    if a.tau in TAUS and a.before_date is None and not CACHE.exists(): raise SystemExit('run k27_ials.py --build-cache first')
    torch.set_num_threads(8); t0 = time.time()
    users, catalog, keys, stats, T = edges([a.tau], a.before_date) if a.before_date is not None else cached_edges(a.tau) if a.tau in TAUS else edges([a.tau]); t_edges = time.time()-t0
    extra = ic.confidence_extra(stats, 'time', {'tau': a.tau, 'alpha': a.alpha})
    X, Y = ic.ials(users, catalog, keys, extra, a.l2, a.seed); t_fit = time.time()-t0-t_edges
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); name = f'tau{a.tau:g}_alpha{a.alpha:g}_l2{a.l2:g}'+(f'_before{a.before_date}' if a.before_date is not None else '')+f'_s{a.seed}'
    torch.save({'X': X, 'Y': Y, 'users': users, 'catalog': catalog, 'T': T, 'setting': vars(a)}, out/f'{name}.pt')
    info = {'setting': name, 'users': len(users), 'items': len(catalog), 'edges': len(keys), 'seconds_edges': t_edges, 'seconds_fit': t_fit,
            'iterations': ic.ITERATIONS, 'dim': ic.DIM, 'before_date': a.before_date, 'T_shanghai_date': int(bes.shanghai_date(np.array([T]))[0]), 'finite': bool(torch.isfinite(X).all() and torch.isfinite(Y).all())}
    (out/f'{name}.json').write_text(json.dumps(info, indent=2)); print(json.dumps(info, indent=2))

if __name__ == '__main__':
    import sys
    build_cache() if sys.argv[1:] == ['--build-cache'] else main()
