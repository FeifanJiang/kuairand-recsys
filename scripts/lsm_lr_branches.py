"""Experiment 1: learning-rate decay and EMA for the original LSM, branched from one checkpoint.

Original featureless LSM (m=4, l2=10, k=16, Gaussian prior via expected-touch weights). Train
10 epochs (identical to the original 10-epoch run), save model, optimizer, and RNG states, then
run branches with the same steps and the same shuffling and negative-sampling sequence:
  constant: constant learning rates for 40 more epochs;
  decay: per-step cosine decay of both learning rates to 1% over the same 40 epochs;
  ema: exponential moving average (beta=0.9995 per step) of the constant branch's parameters.
Every epoch records validation Recall@20/50/100, training NLL and penalty, the exact prior,
the objective on a fixed diagnostic sample, and relative parameter updates. Primary comparison:
epoch 50 and the mean over epochs 41-50; no best-epoch selection. Train and validation only.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from retrieval_eval import load_targets, exact_top100, evaluate
from train_lsm import BipartiteLSM

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed'
BLOCKS = {'user_alpha': lambda m: m.user_alpha, 'z': lambda m: m.z.weight,
          'item_alpha': lambda m: m.item_alpha.weight, 'w': lambda m: m.w.weight}

def setup():
    catalog = np.sort(pd.read_csv(DATA/'candidate_videos.csv').video_id.to_numpy(np.int64))
    users = np.sort(pd.read_csv(DATA/'training_users.csv').user_id.to_numpy(np.int64))
    positives = []; max_train = 0
    for chunk in pd.read_csv(DATA/'train.csv.gz', usecols=['user_id','video_id','long_view','time_ms'], chunksize=250000):
        max_train = max(max_train, int(chunk.time_ms.max()))
        positives.append(chunk.loc[chunk.long_view.eq(1), ['user_id','video_id']].to_numpy(np.int64))
    positive = np.concatenate(positives)
    n_items = len(catalog)
    pair_keys = np.unique(np.searchsorted(users, positive[:,0])*n_items+np.searchsorted(catalog, positive[:,1]))
    return users, catalog, pair_keys, max_train

class Trainer:
    """Reproduces the train_lsm.py loop step for step."""
    def __init__(self, users, catalog, pair_keys, args):
        self.args = args; self.n_items = len(catalog); self.pair_keys = pair_keys
        self.u = pair_keys//self.n_items; self.i = pair_keys%self.n_items
        degree_u = np.bincount(self.u, minlength=len(users)); degree_v = np.bincount(self.i, minlength=self.n_items)
        self.degree_v = degree_v
        rate = np.where(degree_u>0, args.negatives*degree_u/np.maximum(self.n_items-degree_u,1), 1.)
        self.offset = torch.from_numpy(-np.log(rate)).float()
        self.user_alpha0 = np.log(np.maximum(degree_u,.5)/(self.n_items-degree_u))
        self.user_weight = torch.from_numpy(1/np.maximum(degree_u,1)).float()
        self.item_weight = torch.from_numpy(1/(degree_v+args.negatives*len(pair_keys)/self.n_items)).float()
    def build(self, seed):
        torch.manual_seed(seed)
        model = BipartiteLSM(self.user_alpha0, self.n_items, self.args.dim)
        sparse = torch.optim.SparseAdam([model.item_alpha.weight, model.w.weight], lr=self.args.item_lr)
        dense = torch.optim.Adam([model.user_alpha, model.z.weight], lr=self.args.user_lr)
        return model, sparse, dense
    def sample(self, rng, idx):
        ub = self.u[idx]; ib = self.i[idx]; n = self.n_items; keys_all = self.pair_keys
        neg = rng.integers(0, n, size=(len(idx), self.args.negatives), dtype=np.int64)
        for attempt in range(100):
            keys = ub[:,None]*n+neg; loc = np.searchsorted(keys_all, keys)
            bad = (loc<len(keys_all)) & (keys_all[np.minimum(loc,len(keys_all)-1)]==keys)
            if not bad.any(): break
            neg[bad] = rng.integers(0, n, size=int(bad.sum()))
        else: raise RuntimeError('Negative rejection did not converge')
        return torch.from_numpy(ub), torch.from_numpy(np.concatenate([ib[:,None], neg], axis=1))
    def loss(self, model, ut, items):
        theta, penalty = model.logits(ut, items, self.user_weight, self.item_weight)
        labels = torch.zeros_like(theta); labels[:,0] = 1
        nll = F.binary_cross_entropy_with_logits(theta+self.offset[ut][:,None], labels, reduction='sum')/len(ut)
        return nll, self.args.l2*penalty/len(ut)
    def epoch(self, model, sparse, dense, rng, schedule=None, ema=None):
        order = rng.permutation(len(self.u)); nll_sum = pen_sum = 0.
        for start in range(0, len(order), self.args.batch_size):
            idx = order[start:start+self.args.batch_size]
            ut, items = self.sample(rng, idx)
            if schedule: schedule()
            nll, pen = self.loss(model, ut, items); loss = nll+pen
            assert torch.isfinite(loss)
            sparse.zero_grad(set_to_none=True); dense.zero_grad(set_to_none=True)
            loss.backward(); sparse.step(); dense.step()
            if ema is not None:
                with torch.no_grad():
                    for name, get in BLOCKS.items(): ema[name].mul_(self.args.ema_beta).add_(get(model).detach(), alpha=1-self.args.ema_beta)
            nll_sum += nll.item()*len(idx); pen_sum += pen.item()*len(idx)
        return nll_sum/len(self.u), pen_sum/len(self.u)
    def exact_prior(self, model):
        """l2 * (sum alpha_v^2 + |W|^2 + |Z|^2) per edge: the fixed Gaussian prior the weights target."""
        with torch.no_grad():
            total = model.item_alpha.weight.pow(2).sum()+model.w.weight.pow(2).sum()+model.z.weight.pow(2).sum()
        return float(self.args.l2*total/len(self.u))

def swap_in(model, params):
    """Returns a copy of model carrying the given parameter blocks (used for EMA evaluation)."""
    clone = copy.deepcopy(model)
    with torch.no_grad():
        for name, get in BLOCKS.items(): get(clone).copy_(params[name])
    return clone

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--branch-epochs', type=int, default=40)
    parser.add_argument('--decay-floor', type=float, default=.01)
    parser.add_argument('--ema-beta', type=float, default=.9995)
    parser.add_argument('--dim', type=int, default=16); parser.add_argument('--l2', type=float, default=10.)
    parser.add_argument('--negatives', type=int, default=4); parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--item-lr', type=float, default=.02); parser.add_argument('--user-lr', type=float, default=.01)
    parser.add_argument('--diagnostic-edges', type=int, default=20000); parser.add_argument('--diagnostic-seed', type=int, default=2027)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'_s{args.seed}'
    out = ROOT/'reports/lsm_optimization'/run_id; art = ROOT/'artifacts/lsm_optimization'/run_id
    out.mkdir(parents=True); art.mkdir(parents=True)
    def log(msg):
        print(msg, flush=True)
        with (out/'run.log').open('a') as f: f.write(msg+'\n')
    users, catalog, pair_keys, max_train = setup()
    trainer = Trainer(users, catalog, pair_keys, args)
    steps_per_epoch = math.ceil(len(pair_keys)/args.batch_size)
    protocol = {'run_id': run_id, 'experiment': 'learning-rate decay and EMA branches', **vars(args),
        'model': 'original featureless LSM, Gaussian prior via expected-touch weights (train_lsm.py)',
        'training_budget': f'{args.warmup_epochs} shared epochs, then {args.branch_epochs} epochs per branch '
                           f'({args.branch_epochs*steps_per_epoch} steps); identical shuffling and negatives in every branch',
        'decay': f'per-step cosine from base rates to {args.decay_floor:g} x base over the branch, both optimizers',
        'ema': f'beta={args.ema_beta} per step on the constant branch, initialized at the branch checkpoint',
        'diagnostic_sample': f'{args.diagnostic_edges} training edges with {args.negatives} negatives each, seed {args.diagnostic_seed}, fixed for all branches and epochs',
        'selection': 'compare epoch-50 values and mean over the last 10 epochs; no best-epoch selection',
        'confirmation': 'seeds 1-4 only if a branch beats constant by > 0.05 pp Recall@100 at both epoch 50 and the last-10 mean',
        'test_accessed': False}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    groups, all_counts, eligible_pairs, baseline = load_targets(users, catalog, max_train)
    eval_users = groups.index.to_numpy(np.int64); eval_idx = torch.from_numpy(np.searchsorted(users, eval_users))
    drng = np.random.default_rng(args.diagnostic_seed)
    diag_idx = np.sort(drng.choice(len(pair_keys), size=args.diagnostic_edges, replace=False))
    diag_ut, diag_items = trainer.sample(drng, diag_idx)
    def record(model, branch, epoch, nll, pen, previous):
        with torch.no_grad():
            ids, _ = exact_top100(lambda a, b: model.score_block(eval_idx, a, b), len(eval_users), len(catalog), lambda m: None)
            dn, dp = trainer.loss(model, diag_ut, diag_items)
        _, metrics = evaluate(eval_users, catalog[ids], groups, all_counts, eligible_pairs, baseline)
        metrics = metrics.set_index('k'); slot = trainer.degree_v[ids.ravel()]
        row = {'branch': branch, 'epoch': epoch, **{f'recall_at_{k}': float(metrics.loc[k, 'macro_recall']) for k in [20, 50, 100]},
               'train_nll_per_edge': nll, 'train_penalty_per_edge': pen, 'exact_prior_per_edge': trainer.exact_prior(model),
               'diagnostic_nll_per_edge': float(dn), 'diagnostic_objective_per_edge': float(dn)+trainer.exact_prior(model),
               'slot_share_degree_at_most_2': float((slot<=2).mean())}
        current = {n: g(model).detach().clone() for n, g in BLOCKS.items()}
        for n in BLOCKS:
            row[f'update_{n}'] = float((current[n]-previous[n]).norm()/previous[n].norm().clamp(min=1e-12)) if previous else float('nan')
        return row, current
    rows = []
    rng = np.random.default_rng(args.seed)
    model, sparse, dense = trainer.build(args.seed)
    previous = {n: g(model).detach().clone() for n, g in BLOCKS.items()}
    started = time.perf_counter()
    for epoch in range(args.warmup_epochs):
        nll, pen = trainer.epoch(model, sparse, dense, rng)
        row, previous = record(model, 'shared', epoch+1, nll, pen, previous); rows.append(row)
        pd.DataFrame(rows).to_csv(out/'epoch_metrics.csv', index=False)
        log(f'shared epoch {epoch+1}: Recall@100={row["recall_at_100"]:.4%}; nll={nll:.5f}; elapsed={time.perf_counter()-started:.0f}s')
    checkpoint = {'model': model.state_dict(), 'sparse': sparse.state_dict(), 'dense': dense.state_dict(),
                  'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(), 'epoch': args.warmup_epochs,
                  'protocol': protocol}
    torch.save(checkpoint, art/f'checkpoint_epoch{args.warmup_epochs}.pt')
    branch_start = previous
    for branch in ['constant', 'decay']:
        model, sparse, dense = trainer.build(args.seed)
        # Optimizer.load_state_dict keeps references to the checkpoint's state tensors, so a deep copy keeps
        # one branch's Adam moments from leaking into the next. Runs before this fix (reports/lsm_optimization/
        # 20261018T234602Z_s42) started the decay branch from the constant branch's final Adam state.
        model.load_state_dict(checkpoint['model']); sparse.load_state_dict(copy.deepcopy(checkpoint['sparse']))
        dense.load_state_dict(copy.deepcopy(checkpoint['dense']))
        rng = np.random.default_rng(); rng.bit_generator.state = checkpoint['numpy_rng']; torch.set_rng_state(checkpoint['torch_rng'])
        total = args.branch_epochs*steps_per_epoch; step = [0]
        bases = [(g, g['lr']) for opt in (sparse, dense) for g in opt.param_groups]
        def cosine():
            fraction = step[0]/total
            for group, base in bases: group['lr'] = base*(args.decay_floor+(1-args.decay_floor)*.5*(1+math.cos(math.pi*fraction)))
            step[0] += 1
        ema = {n: g(model).detach().clone() for n, g in BLOCKS.items()} if branch=='constant' else None
        prev = branch_start; prev_ema = branch_start
        for epoch in range(args.warmup_epochs, args.warmup_epochs+args.branch_epochs):
            nll, pen = trainer.epoch(model, sparse, dense, rng, schedule=cosine if branch=='decay' else None, ema=ema)
            row, prev = record(model, branch, epoch+1, nll, pen, prev); row['learning_rate_item'] = sparse.param_groups[0]['lr']; rows.append(row)
            if ema is not None:
                row_ema, prev_ema = record(swap_in(model, ema), 'ema', epoch+1, nll, pen, prev_ema); rows.append(row_ema)
            pd.DataFrame(rows).to_csv(out/'epoch_metrics.csv', index=False)
            log(f'{branch} epoch {epoch+1}: Recall@100={row["recall_at_100"]:.4%}'+
                (f'; ema={row_ema["recall_at_100"]:.4%}' if ema is not None else '')+f'; elapsed={time.perf_counter()-started:.0f}s')
        torch.save({'model': model.state_dict(), 'ema': ema, 'protocol': protocol}, art/f'final_{branch}.pt')
    curves = pd.DataFrame(rows); last = args.warmup_epochs+args.branch_epochs
    summary = []
    for branch in ['constant', 'decay', 'ema']:
        c = curves.loc[curves.branch.eq(branch)]
        tail = c.loc[c.epoch.gt(last-10)]
        summary.append({'branch': branch, **{f'final_recall_at_{k}': float(c.loc[c.epoch.eq(last), f'recall_at_{k}'].iloc[0]) for k in [20, 50, 100]},
            **{f'last10_mean_recall_at_{k}': float(tail[f'recall_at_{k}'].mean()) for k in [20, 50, 100]},
            'last10_sd_recall_at_100': float(tail.recall_at_100.std(ddof=1)),
            'final_diagnostic_objective': float(c.diagnostic_objective_per_edge.iloc[-1])})
    summary = pd.DataFrame(summary); summary.to_csv(out/'summary.csv', index=False)
    base = summary.set_index('branch').loc['constant']
    confirm = {b: bool((summary.set_index('branch').loc[b, 'final_recall_at_100']-base.final_recall_at_100 > .0005) and
                       (summary.set_index('branch').loc[b, 'last10_mean_recall_at_100']-base.last10_mean_recall_at_100 > .0005))
               for b in ['decay', 'ema']}
    (out/'confirmation_rule.json').write_text(json.dumps(confirm, indent=2))
    log(summary.to_string(index=False)); log(json.dumps(confirm)); log('Finished.')

if __name__ == '__main__':
    main()
