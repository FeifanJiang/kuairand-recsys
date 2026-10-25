"""Sequence retrieval models for KuaiRand: history mean pooling and SASRec with sampled softmax.

Task: from a user's earlier feed requests, predict videos long-viewed in later requests. time_ms marks a
feed request (several videos share one timestamp), so a target at time t may only use events with time < t,
and the order inside a timestamp group is an explicit arbitrary rule (ascending video_id), not watch order.
The representation for a target is read at q = last window position with time < t; causal attention over
positions <= q then sees only strictly earlier events, in every layer.
Windows: C context events plus a target region of K events; each event is a target at most once and a
target's history is the window events before q (between about C and C+K-1 events). Evaluation reads the
most recent C+K-1 events up to the cutoff T.
Trainable items are videos with at least one training long view. Other catalog videos have no rows in
either embedding table and a fixed score of 0, so sampling and optimizer updates cannot change them.
Loss: m negatives drawn uniformly with replacement from all N catalog videos, independently per target;
draws equal to the target contribute 0 and are not redrawn; with the original m in the factor N/m,
e^{s_p} + (N/m) sum_k 1[d_k != p] e^{s_{d_k}} is an unbiased estimate of the full softmax denominator.
Its logarithm, the loss, and the gradient are not unbiased in general.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]

def load_sequences(data_dir, tie_order='video_id'):
    """Per-user long-view sequences (training period only) and catalog maps."""
    catalog = np.sort(pd.read_csv(data_dir/'candidate_videos.csv').video_id.to_numpy(np.int64))
    users = np.sort(pd.read_csv(data_dir/'training_users.csv').user_id.to_numpy(np.int64))
    frames = []; T = 0
    for chunk in pd.read_csv(data_dir/'train.csv.gz', usecols=['user_id','video_id','long_view','time_ms'], chunksize=500000):
        T = max(T, int(chunk.time_ms.max())); frames.append(chunk.loc[chunk.long_view.eq(1), ['user_id','video_id','time_ms']])
    events = pd.concat(frames, ignore_index=True)
    # time_ms marks a feed request (median 6 exposures share one timestamp), so events within a timestamp
    # group have no observed watch order; they are ordered by ascending video_id (an arbitrary, fixed rule).
    # The sensitivity check permutes within timestamp groups after windows and evaluation histories are
    # fixed (shuffle_within_timestamps), so membership never changes.
    if tie_order!='video_id': raise ValueError(tie_order)
    events['tie'] = events.video_id.to_numpy()
    events = events.sort_values(['user_id', 'time_ms', 'tie'], kind='stable')
    catalog_index = np.searchsorted(catalog, events.video_id.to_numpy())
    degree = np.bincount(catalog_index, minlength=len(catalog))
    trainable = np.flatnonzero(degree > 0)                       # catalog indices with parameters
    item_of_catalog = np.full(len(catalog), -1, np.int64); item_of_catalog[trainable] = np.arange(len(trainable))
    events['item'] = item_of_catalog[catalog_index]+1            # 1..P; 0 is padding
    seqs = {u: (g.item.to_numpy(np.int64), g.time_ms.to_numpy(np.int64)) for u, g in events.groupby('user_id', sort=True)}
    return {'catalog': catalog, 'users': users, 'trainable': trainable, 'item_of_catalog': item_of_catalog,
            'degree': degree, 'sequences': seqs, 'T': T, 'events': len(events)}

def build_windows(seqs, C, K):
    """Windows of up to C context events followed by a target region of K events, regions tiling each user's
    sequence so every event is a target at most once. A target at time t reads query position q = last window
    position with time < t; its history is the window events up to q (all strictly earlier). Targets with no
    strictly earlier event in the window are dropped (only a user's first timestamp group when C >= 10, the
    largest tie group). C=0, K=L reproduces non-overlapping windows. Window length is C+K, left padded."""
    W = C+K; items, queries, targets, times, regions = [], [], [], [], []; first_group, boundary = 0, 0
    for x, t in seqs.values():
        for s0 in range(0, len(x), K):
            a0 = max(0, s0-C); xi, ti = x[a0:s0+K], t[a0:s0+K]
            q = np.searchsorted(ti, ti, side='left')-1
            region = np.arange(len(xi)) >= s0-a0
            keep = region & (q >= 0)
            lost = int((region & (q < 0)).sum())
            if s0==0: first_group += lost
            else: boundary += lost
            pad = W-len(xi)
            items.append(np.concatenate([np.zeros(pad, np.int64), xi]))
            queries.append(np.concatenate([np.full(pad, -1), np.where(keep, q+pad, -1)]))
            targets.append(np.concatenate([np.zeros(pad, np.int64), np.where(keep, xi, 0)]))
            times.append(np.concatenate([np.full(pad, -1, np.int64), ti])); regions.append(np.concatenate([np.zeros(pad, bool), region]))
    return {'items': torch.from_numpy(np.stack(items)), 'queries': torch.from_numpy(np.stack(queries)),
            'targets': torch.from_numpy(np.stack(targets)), 'times': torch.from_numpy(np.stack(times)),
            'regions': torch.from_numpy(np.stack(regions)), 'dropped_first_group': first_group, 'dropped_window_boundary': boundary}

def shuffle_within_timestamps(items, targets, times, regions, seed):
    """Sensitivity check: permute positions only among events with the same timestamp and the same role
    (context or target region) inside each fixed window, so window membership, target sets, query positions,
    and every target's history set are unchanged; only the arrangement inside timestamp groups moves."""
    rng = np.random.default_rng(seed); items, targets = items.clone().numpy(), targets.clone().numpy()
    t, r = times.numpy(), regions.numpy()
    for row in range(items.shape[0]):
        key = t[row]*2+r[row]                                  # timestamp and role
        valid = t[row] >= 0
        starts = np.flatnonzero(valid & np.r_[True, key[1:]!=key[:-1]])
        ends = np.r_[starts[1:], valid.sum()+np.argmax(valid)]
        for a, b in zip(starts, ends):
            if b-a > 1:
                perm = a+rng.permutation(b-a); items[row, a:b] = items[row, perm]; targets[row, a:b] = targets[row, perm]
    return torch.from_numpy(items), torch.from_numpy(targets)

def eval_histories(seqs, users, L, return_times=False):
    """Most recent L training long-view events per user (all are at or before T), left padded."""
    out = np.zeros((len(users), L), np.int64); tim = np.full((len(users), L), -1, np.int64)
    for r, u in enumerate(users):
        if u in seqs:
            x, t = seqs[u][0][-L:], seqs[u][1][-L:]; out[r, L-len(x):] = x; tim[r, L-len(t):] = t
    return (torch.from_numpy(out), torch.from_numpy(tim)) if return_times else torch.from_numpy(out)

class Block(nn.Module):
    """SASRec block as in the reference implementation: LayerNorm on queries, causal + padding masks,
    residual on the normalized query, LayerNorm, position-wise FFN with residual."""
    def __init__(self, d, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.ff1, self.ff2 = nn.Linear(d, d), nn.Linear(d, d); self.drop = nn.Dropout(dropout)
    def forward(self, x, pad):
        n = x.shape[1]; qn = self.norm1(x)
        scores = (self.q(qn)@self.k(x).transpose(1, 2))/math.sqrt(x.shape[-1])
        causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None] & ~pad[:, None, :]
        empty = ~allowed.any(-1, keepdim=True)                    # left-padded queries see no key
        # Finite logits for empty rows keep softmax and its backward pass free of NaN; their weights are zeroed.
        scores = scores.masked_fill(~allowed, float('-inf')).masked_fill(empty, 0.)
        attn = torch.softmax(scores, -1).masked_fill(empty, 0.)
        h = self.norm2(qn+self.drop(attn)@self.v(x))
        h = h+self.drop(self.ff2(self.drop(F.relu(self.ff1(h)))))
        return h*(~pad)[..., None]

class SeqRetriever(nn.Module):
    """encoder='pool': prefix mean of input embeddings; encoder='sasrec': transformer blocks.
    Both use the same untied input/output embeddings, final LayerNorm, and dot-product scoring."""
    def __init__(self, n_trainable, L, d, encoder, blocks=2, dropout=.2):
        super().__init__()
        self.encoder, self.L = encoder, L
        self.inp = nn.Embedding(n_trainable+1, d, padding_idx=0, sparse=True)
        self.out = nn.Embedding(n_trainable+1, d, padding_idx=0, sparse=True)   # row 0 unused
        nn.init.trunc_normal_(self.inp.weight, std=.02); nn.init.trunc_normal_(self.out.weight, std=.02)
        with torch.no_grad(): self.inp.weight[0] = 0; self.out.weight[0] = 0
        if encoder=='sasrec':
            self.pos = nn.Embedding(L, d); self.blocks = nn.ModuleList([Block(d, dropout) for _ in range(blocks)]); self.drop = nn.Dropout(dropout)
        self.final = nn.LayerNorm(d)
    def states(self, items):
        """Hidden state at every position, using only positions <= that position."""
        pad = items.eq(0); e = self.inp(items)
        if self.encoder=='pool':
            csum = e.cumsum(1); count = (~pad).cumsum(1).clamp(min=1)[..., None]
            return self.final(csum/count)*(~pad)[..., None]
        x = (e+self.pos.weight[None, -items.shape[1]:])*(~pad)[..., None]
        x = self.drop(x)
        for block in self.blocks: x = block(x, pad)
        return self.final(x)
    def output_scores(self, h, items):
        """Scores for trainable item ids (1..P); id 0 means a fixed-zero video and scores 0."""
        return torch.where(items.gt(0), (h[..., None, :]*self.out(items)).sum(-1), torch.zeros((), dtype=h.dtype))

def sampled_softmax_loss(model, items, queries, targets, n_catalog, item_of_catalog, m, generator, return_touched=False):
    """Uniform sampled softmax over the full catalog with logQ correction log(m / N)."""
    h_all = model.states(items)
    mask = queries.ge(0)
    rows, cols = mask.nonzero(as_tuple=True)
    h = h_all[rows, queries[rows, cols]]
    pos_item = targets[rows, cols]
    neg_catalog = torch.randint(0, n_catalog, (len(rows), m), generator=generator)
    neg_item = item_of_catalog[neg_catalog]+1                  # 0 for fixed-zero videos
    s_pos = model.output_scores(h, pos_item[:, None])[:, 0]
    s_neg = model.output_scores(h, neg_item)-math.log(m/n_catalog)
    s_neg = s_neg.masked_fill(neg_item.eq(pos_item[:, None]), float('-inf'))   # accidental hits
    logits = torch.cat([s_pos[:, None], s_neg], 1)
    loss = -torch.log_softmax(logits, 1)[:, 0].mean()
    if return_touched: return loss, len(rows), pos_item, neg_item
    return loss, len(rows)

def embedding_touch_counts(items, targets, n_trainable, m, n_catalog):
    """Per-epoch touches of each trainable row (index 1..P; 0 is padding): input rows are touched once per
    window occurrence (deterministic); output rows once per positive (deterministic) plus m T / N expected
    negative draws, where T is the number of targets per epoch."""
    c_in = torch.bincount(items[items.gt(0)], minlength=n_trainable+1).double()
    pos = targets[targets.gt(0)]; T = len(pos)
    c_out = torch.bincount(pos, minlength=n_trainable+1).double()+m*T/n_catalog
    return c_in, c_out, T

def embedding_penalty(model, items, pos_item, neg_item, c_in, c_out):
    """Sum over the batch's touches of |row|^2 / expected touches per epoch. Summed over one epoch this is
    exactly sum |input rows|^2 and, in expectation, sum |output rows|^2: a fixed Gaussian prior on both tables."""
    used_in = items[items.gt(0)]
    pen = (model.inp(used_in).pow(2).sum(-1)/c_in[used_in].float()).sum()
    used_out = torch.cat([pos_item, neg_item.reshape(-1)]); used_out = used_out[used_out.gt(0)]
    return pen+(model.out(used_out).pow(2).sum(-1)/c_out[used_out].float()).sum()

def catalog_scorer(model, histories, item_of_catalog):
    """Returns score_block(a, b) over catalog indices; user states are computed once.
    Fixed-zero videos score exactly 0."""
    with torch.no_grad(): h = model.states(histories)[:, -1]
    def score_block(start, stop):
        with torch.no_grad():
            ids = item_of_catalog[start:stop]+1; s = torch.zeros(len(h), stop-start); live = ids.gt(0)
            s[:, live] = h@model.out.weight[ids[live]].T
        return s
    return score_block

def stable_topk(score_block, n_users, n_items, k=100, block=65536, excluded=None):
    """Exact top-k with ties broken by ascending catalog index (stable descending sort)."""
    best_s = torch.full((n_users, 0), float('-inf')); best_i = torch.zeros((n_users, 0), dtype=torch.long)
    for a in range(0, n_items, block):
        b = min(a+block, n_items); s = score_block(a, b)
        if excluded is not None: s = excluded(s, a, b)
        idx = torch.arange(a, b).expand(n_users, -1)
        s_all = torch.cat([best_s, s], 1); i_all = torch.cat([best_i, idx], 1)
        order = torch.argsort(i_all, dim=1, stable=True)                     # index order first
        s_all, i_all = torch.gather(s_all, 1, order), torch.gather(i_all, 1, order)
        top = torch.argsort(s_all, dim=1, descending=True, stable=True)[:, :k]
        best_s, best_i = torch.gather(s_all, 1, top), torch.gather(i_all, 1, top)
    return best_i, best_s
