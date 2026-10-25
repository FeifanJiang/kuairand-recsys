"""Out-of-core version of feature_model.prequential / feature_model.frozen for logs that do not fit in memory.

Every statistic is computed on a block that holds the complete history of the keys it covers:
- key-partitioned keys (v by video, a by author, u/ua/ut by user): rows are split by key modulo n_buckets,
  so every event of a given key lands in exactly one bucket; feature_model.prequential runs unchanged there.
- low-cardinality keys (g, tg): rows are split into time ranges whose boundaries never cut a timestamp;
  ranges are processed in time order and each key's running totals are carried into the next range
  (a cross-range time-increment table), so no key is computed from a partial history.
Within any block the rule of feature_model is kept: a row's features use events with strictly smaller
timestamps; the row's own timestamp group is added to the totals only after its features are produced.

Events are passed as a dict of 1-D arrays (numpy arrays or memmaps) with the columns feature_model expects;
`wanted` marks the rows whose prequential features are returned, and `user_keys_for` restricts the user
keys to a set of users (statistics of other users are never needed for their rows).
"""
import numpy as np
import pandas as pd
import feature_model as fm

PARTITION = {'v': 'video_id', 'a': 'author_id', 'u': 'user_id', 'ua': 'user_id', 'ut': 'user_id'}
CARRY = ('g', 'tg')
COLUMNS = ['user_id', 'video_id', 'author_id', 'tag1', 't', 'lv', 'const', 'lv_logdur', 'lv_short']

def frame(ev, idx):
    return pd.DataFrame({c: np.asarray(ev[c][idx]) for c in COLUMNS}, index=idx)

def key_blocks(ev, key, n_buckets, users=None):
    bucket = (np.asarray(ev[PARTITION[key]]) % n_buckets).astype(np.int16)
    if users is not None: bucket[~np.isin(np.asarray(ev['user_id']), users)] = -1
    order = np.argsort(bucket, kind='stable'); bounds = np.searchsorted(bucket[order], np.arange(n_buckets+1))
    for b in range(n_buckets):
        yield order[bounds[b]:bounds[b+1]]                          # ascending row indices within a bucket

def time_blocks(t, n_ranges):
    """Row index blocks ordered by time; a boundary never falls inside a timestamp group."""
    t = np.asarray(t); cuts = np.unique(np.quantile(t, np.linspace(0, 1, n_ranges+1)[1:-1]).astype(t.dtype))
    rng = np.searchsorted(cuts, t, side='right')                    # equal timestamps share a range
    for r in range(len(cuts)+1):
        yield np.flatnonzero(rng == r)

def carry_prequential(ev, key, T0, n_ranges, wanted):
    """Prequential statistics for a low-cardinality key, with running totals carried across time ranges."""
    cols = fm.KEYS[key]; outs, frozen_parts, carry = [], [], None
    for idx in time_blocks(ev['t'], n_ranges):
        f = frame(ev, idx); pre = fm.prequential(f, key, T0); tab = fm.frozen(f, key, T0)
        if carry is not None:
            add = f[cols].merge(carry, on=cols, how='left').set_index(f.index)
            for c in ['n', 'l']+fm.SUMS.get(key, []): pre[f'{key}_{c}'] += add[f'{key}_{c}'].fillna(0).to_numpy()
            if key in fm.DECAY:
                for tau in fm.TAUS:
                    pre[f'{key}_d{tau:g}'] += add[f'{key}_E{tau:g}'].fillna(0).to_numpy()*np.exp(-(f.t-T0)/(tau*fm.DAY))
                pre[f'{key}_lastlv'] = np.fmax(pre[f'{key}_lastlv'].to_numpy(), add[f'{key}_lastlv'].to_numpy())
        outs.append(pre.loc[pre.index.intersection(wanted)])
        carry = tab if carry is None else _merge_totals(carry, tab, key)
    return pd.concat(outs).sort_index(), carry

def _merge_totals(a, b, key):
    cols = fm.KEYS[key]; both = pd.concat([a, b])
    agg = {c: 'sum' for c in both.columns if c not in cols}
    if f'{key}_lastlv' in agg: agg[f'{key}_lastlv'] = 'max'
    return both.groupby(cols, as_index=False).agg(agg)

def build(ev, T0, wanted, n_buckets=16, n_ranges=8, user_keys_for=None):
    """Prequential features for rows `wanted` (sorted row indices) and frozen tables for every key."""
    wanted = np.asarray(wanted); pre, tables = [], {}
    for key in fm.KEYS:
        if key in CARRY:
            p, tab = carry_prequential(ev, key, T0, n_ranges, wanted); pre.append(p.reindex(wanted)); tables[key] = tab; continue
        users = user_keys_for if key in ('u', 'ua', 'ut') else None
        parts, tabs = [], []
        for idx in key_blocks(ev, key, n_buckets, users):
            if len(idx) == 0: continue
            f = frame(ev, idx); p = fm.prequential(f, key, T0)
            parts.append(p.loc[p.index.intersection(wanted)]); tabs.append(fm.frozen(f, key, T0))
        pre.append(pd.concat(parts).reindex(wanted)); tables[key] = pd.concat(tabs, ignore_index=True)
    return pd.concat(pre, axis=1), tables
