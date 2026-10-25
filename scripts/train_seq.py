"""Train one sequence-retrieval configuration under reports/seq_retrieval/protocol_draft.json (validation only).

Look-back windows (C context + K = C/2 target events), uniform sampled softmax with m negatives and factor
N/m, SparseAdam (embeddings) + Adam (other parameters) at lr 1e-3, batches of 4096 // K windows, 4 threads.
Every 2 epochs: tuning-half Recall@100 with the seen filter; stop after 3 evaluations without improvement;
keep the best checkpoint. Then evaluate all eligible users (unfiltered and seen-filtered), save per-user
recall, top-100 lists, fixed-zero and degree-group statistics, the model, maps, config, data md5, and a
copy of the scripts, and verify that the reloaded model reproduces the top-100 lists.
--shuffle runs the membership-preserving within-timestamp sensitivity check (seed 2029).
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import time
import numpy as np
import pandas as pd
import torch
from seq_retrieval import (SeqRetriever, sampled_softmax_loss, build_windows, load_sequences, eval_histories,
                           catalog_scorer, stable_topk, shuffle_within_timestamps, embedding_touch_counts, embedding_penalty)
from retrieval_eval import load_targets

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
GROUPS = ['0', '1-2', '3-19', '>=20']

def run_name(a):
    return f'{a.encoder}_C{a.C}_d{a.dim}'+(f'_l2-{a.embed_l2:g}' if a.embed_l2 else '')+f'_s{a.seed}'+('_shuffle' if a.shuffle else '')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', choices=['pool', 'sasrec'], required=True)
    parser.add_argument('--C', type=int, required=True); parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42); parser.add_argument('--negatives', type=int, default=256)
    parser.add_argument('--max-epochs', type=int, default=20); parser.add_argument('--eval-every', type=int, default=2)
    parser.add_argument('--patience', type=int, default=3); parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--shuffle', action='store_true'); parser.add_argument('--out', required=True)
    # Fixed Gaussian prior on both embedding tables, added to the objective (not decoupled weight decay):
    # each touched row is penalized by |row|^2 / expected touches per epoch, so one epoch accumulates
    # embed_l2 * sum |rows|^2 relative to the summed sampled-softmax loss.
    parser.add_argument('--embed-l2', type=float, default=0.)
    a = parser.parse_args(); K = a.C//2; W = a.C+K
    torch.set_num_threads(4); torch.manual_seed(a.seed)
    out = Path(a.out)/run_name(a); art = ROOT/'artifacts/seq_retrieval'/Path(a.out).name/run_name(a)
    out.mkdir(parents=True, exist_ok=False); art.mkdir(parents=True, exist_ok=True)
    def log(msg):
        print(msg, flush=True)
        with (out/'run.log').open('a') as f: f.write(msg+'\n')
    data = load_sequences(DATA); catalog, users = data['catalog'], data['users']
    ioc = torch.from_numpy(data['item_of_catalog']); n = len(catalog)
    w = build_windows(data['sequences'], a.C, K)
    items, targets = w['items'], w['targets']
    if a.shuffle: items, targets = shuffle_within_timestamps(items, targets, w['times'], w['regions'], 2029)
    queries = w['queries']
    groups, _, _, _ = load_targets(users, catalog, data['T'])
    order = np.random.default_rng(2028).permutation(len(groups))
    split = pd.Series('report', index=groups.index); split.iloc[order[:len(groups)//2]] = 'tune'
    eval_users = groups.index.to_numpy(); rows_all = np.searchsorted(users, eval_users)
    hist, hist_t = eval_histories(data['sequences'], users, W-1, return_times=True)
    if a.shuffle: hist, _ = shuffle_within_timestamps(hist, torch.zeros_like(hist), hist_t, torch.zeros_like(hist, dtype=torch.bool), 2029)
    targets_idx = [np.searchsorted(catalog, np.fromiter(t, np.int64)) for t in groups]
    degree = data['degree']; group_of = np.select([degree==0, degree<=2, degree<=19], [0, 1, 2], 3)
    seen = {}
    for u, (x, _) in data['sequences'].items(): seen[u] = data['trainable'][x-1]          # catalog indices
    def excluder(user_rows):
        pr, pi = [], []
        for r, u in enumerate(users[np.asarray(user_rows)]): pr += [r]*len(seen.get(u, ())); pi += list(seen.get(u, ()))
        pr, pi = torch.tensor(pr), torch.tensor(pi); o = torch.argsort(pi, stable=True); pr, pi = pr[o], pi[o]
        def f(s, lo_, hi_):
            lo, hi = torch.searchsorted(pi, lo_), torch.searchsorted(pi, hi_); s[pr[lo:hi], pi[lo:hi]-lo_] = float('-inf'); return s
        return f
    def topk(model, user_rows, filtered):
        model.eval()
        ids, scores = stable_topk(catalog_scorer(model, hist[user_rows], ioc), len(user_rows), n, k=100,
                                  excluded=excluder(user_rows) if filtered else None)
        model.train(); return ids.numpy(), scores.numpy()
    def recall(ids, user_index, k=100):
        return np.array([len(np.intersect1d(targets_idx[i], row[:k])) / len(targets_idx[i]) for i, row in zip(user_index, ids)])
    tune_index = np.flatnonzero(split.to_numpy()=='tune'); tune_rows = torch.from_numpy(rows_all[tune_index])
    model = SeqRetriever(len(data['trainable']), W, a.dim, a.encoder, dropout=a.dropout)
    sparse = torch.optim.SparseAdam([model.inp.weight, model.out.weight], lr=1e-3)
    dense_params = [p for name, p in model.named_parameters() if name not in ('inp.weight', 'out.weight')]
    dense = torch.optim.Adam(dense_params, lr=1e-3) if dense_params else None
    g = torch.Generator().manual_seed(a.seed); B = 4096//K
    if a.embed_l2: c_in, c_out, _ = embedding_touch_counts(items, targets, len(data['trainable']), a.negatives, n)
    degree_rows = torch.from_numpy(degree[data['trainable']])                # training degree per trainable row
    def norms():
        stats = {}
        for name, table in [('input', model.inp.weight), ('output', model.out.weight)]:
            r = table.detach()[1:].norm(dim=1)
            stats.update({f'{name}_norm_p50': float(r.median()), f'{name}_norm_p90': float(r.quantile(.9)), f'{name}_norm_p99': float(r.quantile(.99)),
                          f'{name}_norm_mean_degree_le2': float(r[degree_rows<=2].mean()), f'{name}_norm_mean_degree_ge20': float(r[degree_rows>=20].mean())})
        return stats
    config = {**vars(a), 'K': K, 'window': W, 'evaluation_history': W-1, 'batch_windows': B, 'threads': 4, 'lr': 1e-3,
              'trainable_items': len(data['trainable']), 'catalog': n, 'targets_per_epoch': int(queries.ge(0).sum()),
              'dropped_first_group': w['dropped_first_group'], 'tie_order': 'ascending video_id',
              'shuffle': 'membership-preserving within-timestamp shuffle, seed 2029' if a.shuffle else 'none',
              'data_md5': {f: hashlib.md5((DATA/f).read_bytes()).hexdigest() for f in ['train.csv.gz', 'validation.csv.gz', 'candidate_videos.csv', 'training_users.csv']},
              'started_utc': datetime.now(timezone.utc).isoformat(), 'test_accessed': False}
    (out/'config.json').write_text(json.dumps(config, indent=2))
    curve = []; best = (-1., 0); bad = 0; started = time.perf_counter()
    for epoch in range(1, a.max_epochs+1):
        perm = torch.randperm(len(items), generator=g); loss_sum = 0.; total_sum = 0.; count = 0
        for s0 in range(0, len(perm), B):
            idx = perm[s0:s0+B]
            if a.embed_l2:
                loss, k, pos_item, neg_item = sampled_softmax_loss(model, items[idx], queries[idx], targets[idx], n, ioc, a.negatives, g, return_touched=True)
                data_loss = loss; loss = loss+a.embed_l2*embedding_penalty(model, items[idx], pos_item, neg_item, c_in, c_out)/k
            else:
                loss, k = sampled_softmax_loss(model, items[idx], queries[idx], targets[idx], n, ioc, a.negatives, g); data_loss = loss
            if not torch.isfinite(loss): raise RuntimeError(f'non-finite loss at epoch {epoch}')
            sparse.zero_grad(); loss.backward(); sparse.step()
            if dense: dense.step(); dense.zero_grad()
            loss_sum += float(data_loss)*k; total_sum += float(loss)*k; count += k
        row = {'epoch': epoch, 'train_loss': loss_sum/count, 'elapsed_seconds': time.perf_counter()-started}
        if a.embed_l2: row['train_objective_with_penalty'] = total_sum/count
        if epoch % a.eval_every == 0: row.update(norms())
        if epoch % a.eval_every == 0:
            ids, _ = topk(model, tune_rows, True); row['tune_recall_at_100_filtered'] = float(recall(ids, tune_index).mean())
            if row['tune_recall_at_100_filtered'] > best[0]:
                best = (row['tune_recall_at_100_filtered'], epoch); bad = 0
                torch.save(model.state_dict(), art/'best_model.pt')
            else: bad += 1
        curve.append(row); pd.DataFrame(curve).to_csv(out/'curve.csv', index=False)
        log(json.dumps(row))
        if bad >= a.patience: log(f'early stop at epoch {epoch}'); break
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()): raise RuntimeError('non-finite parameters')
    model.load_state_dict(torch.load(art/'best_model.pt')); model.eval()
    all_rows = torch.from_numpy(rows_all); results = {}; lists = {}
    for filtered in [False, True]:
        ids, scores = topk(model, all_rows, filtered); lists['filtered' if filtered else 'unfiltered'] = ids
        per_user = pd.DataFrame({f'recall_at_{k}': recall(ids, range(len(eval_users)), k) for k in [20, 50, 100]}, index=eval_users)
        per_user.to_csv(out/f'per_user_{"filtered" if filtered else "unfiltered"}.csv')
        slots = group_of[ids]; hits = np.array([np.isin(row, t) for row, t in zip(ids, targets_idx)])
        stats = {'zero_degree_slot_share': float((degree[ids]==0).mean()), 'zero_degree_hits': int((hits & (degree[ids]==0)).sum()),
                 'zero_score_slot_share': float((scores==0).mean())}
        for gi, label in enumerate(GROUPS):
            stats[f'slot_share_{label}'] = float((slots==gi).mean()); stats[f'hits_{label}'] = int((hits & (slots==gi)).sum())
        results['filtered' if filtered else 'unfiltered'] = {**{f'recall_at_{k}': float(per_user[f'recall_at_{k}'].mean()) for k in [20, 50, 100]}, **stats}
    np.savez_compressed(art/'top100.npz', user_ids=eval_users, unfiltered=catalog[lists['unfiltered']], filtered=catalog[lists['filtered']])
    np.save(art/'trainable_catalog_index.npy', data['trainable'])
    reloaded = SeqRetriever(len(data['trainable']), W, a.dim, a.encoder, dropout=a.dropout); reloaded.load_state_dict(torch.load(art/'best_model.pt')); reloaded.eval()
    check_rows = all_rows[:16]
    r_ids, _ = stable_topk(catalog_scorer(reloaded, hist[check_rows], ioc), len(check_rows), n, k=100, excluded=excluder(check_rows))
    summary = {'run': run_name(a), 'best_epoch': best[1], 'best_tune_recall_at_100_filtered': best[0], 'epochs_trained': curve[-1]['epoch'],
               'results_all_eligible_users': results, 'reload_reproduces_filtered_top100_first_16_users': bool(np.array_equal(r_ids.numpy(), lists['filtered'][:16])),
               'parameters_finite': True, 'finished_utc': datetime.now(timezone.utc).isoformat()}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    code = out/'code'; code.mkdir(exist_ok=True)
    for f in ['seq_retrieval.py', 'train_seq.py', 'retrieval_eval.py']: shutil.copy(ROOT/'scripts'/f, code/f)
    summary['embed_l2'] = a.embed_l2; (out/'summary.json').write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary))

if __name__ == '__main__':
    main()
