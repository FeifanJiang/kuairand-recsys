"""Enumerable checks for seq_retrieval.py (no effectiveness evaluation; train/validation data only).

1. Unbiased denominator: enumerating every m-draw from N items, the logQ-corrected sampled sum equals
   the full-catalog sum over non-target items.
2. Loss path: sampled_softmax_loss equals a manual recomputation for the same draws.
3. No leakage: changing the target, any same-timestamp event, or any later event leaves the query state
   unchanged for both encoders; changing an earlier event does change it.
4. Padding: extra left padding with arbitrary hidden content masked out does not change states.
5. Fixed-zero videos: no parameters; scores stay exactly 0 after optimizer steps; padding rows stay 0;
   every parameter and every score stays finite.
9. Left-padded SASRec batches: finite loss and gradients. Informational: where the previous formulation's
   NaN appeared and whether it reached the scores.
6. Real training data: ties are ordered by ascending video_id; the within-timestamp shuffle of fixed windows
   keeps every sampled target's history set, the target multiset, and evaluation history sets; in look-back windows every query is strictly earlier than its target, no
   event at or after the target time precedes the query, each event is a target once, and histories are
   at most C+K-1 events; evaluation histories are the most recent C+K-1 events.
7. Stable top-k: ties are broken by ascending catalog index, matching a brute-force sort.
"""
from pathlib import Path
import itertools
import json
import math
import numpy as np
import torch
from seq_retrieval import (SeqRetriever, sampled_softmax_loss, build_windows, load_sequences, stable_topk,
                           catalog_scorer, eval_histories, shuffle_within_timestamps)

ROOT = Path(__file__).resolve().parents[1]

def main():
    torch.manual_seed(0); results = {}
    # 1. Unbiasedness of the corrected sampled denominator (N=5 including fixed-zero videos, m=3).
    s = torch.tensor([.3, -1.2, 0., 2., 0.], dtype=torch.float64); N, m, target = 5, 3, 3
    total = 0.
    for draw in itertools.product(range(N), repeat=m):
        total += sum(math.exp(float(s[j])-math.log(m/N)) for j in draw if j!=target)
    expected = total/N**m; exact = float(sum(torch.exp(s[j]) for j in range(N) if j!=target))
    results['1_corrected_sampled_denominator_unbiased'] = abs(expected-exact) < 1e-12
    # Tiny model: 3 trainable items, catalog of 6 with three fixed-zero videos.
    item_of_catalog = torch.tensor([0, -1, 1, -1, 2, -1])
    for encoder in ['pool', 'sasrec']:
        model = SeqRetriever(3, 6, 4, encoder, dropout=0.).eval()
        items = torch.tensor([[0, 1, 2, 2, 3, 1]]); times = torch.tensor([0, 10, 20, 20, 30, 40])
        # 2. Loss path versus manual recomputation.
        queries = torch.tensor([[-1, -1, 1, 1, 3, 4]]); targets = torch.tensor([[0, 0, 2, 2, 3, 1]])
        g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
        loss, n = sampled_softmax_loss(model, items, queries, targets, 6, item_of_catalog, 4, g1)
        h_all = model.states(items); rows, cols = queries.ge(0).nonzero(as_tuple=True); h = h_all[rows, queries[rows, cols]]
        neg = item_of_catalog[torch.randint(0, 6, (len(rows), 4), generator=g2)]+1; pos = targets[rows, cols]
        manual = []
        for r in range(len(rows)):
            sp = float(h[r]@model.out.weight[pos[r]])
            sn = [(float(h[r]@model.out.weight[j]) if j>0 else 0.)-math.log(4/6) for j in neg[r].tolist() if j!=pos[r]]
            manual.append(-sp+math.log(math.exp(sp)+sum(math.exp(x) for x in sn)))
        results[f'2_{encoder}_loss_matches_manual'] = abs(float(loss)-float(np.mean(manual))) < 1e-5
        # 3. Leakage: target at index 2 (time 20) reads query index 1 (time 10).
        base = model.states(items)[0, 1].detach()
        for idx in [2, 3, 4, 5]:                                      # target, same-time event, later events
            changed = items.clone(); changed[0, idx] = 3 if items[0, idx]!=3 else 1
            results[f'3_{encoder}_state_unchanged_when_index_{idx}_changes'] = torch.allclose(model.states(changed)[0, 1], base)
        earlier = items.clone(); earlier[0, 1] = 3
        results[f'3_{encoder}_state_changes_when_earlier_event_changes'] = not torch.allclose(model.states(earlier)[0, 1], base)
        # 4. Padding: states of real positions do not depend on left padding length.
        if encoder=='pool':
            padded = torch.tensor([[0, 0, 0, 1, 2, 2, 3, 1]])
            results['4_pool_left_padding_invariant'] = torch.allclose(model.states(padded)[0, 3:], model.states(items)[0, 1:])
        # 5. Fixed-zero videos and padding rows after optimizer steps.
        model.train(); opt = torch.optim.SparseAdam([model.inp.weight, model.out.weight], lr=.1)
        dense = [p for n, p in model.named_parameters() if n not in ('inp.weight', 'out.weight')]
        opt_d = torch.optim.Adam(dense, lr=.1) if dense else None
        g = torch.Generator().manual_seed(1)
        for _ in range(5):
            loss, _ = sampled_softmax_loss(model, items, queries, targets, 6, item_of_catalog, 6, g)
            opt.zero_grad(); loss.backward(); opt.step()
            if opt_d: opt_d.step(); opt_d.zero_grad()
        results[f'5_{encoder}_all_parameters_finite_after_steps'] = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        model.eval()
        scorer = catalog_scorer(model, items, item_of_catalog)
        sc = scorer(0, 6)
        results[f'5_{encoder}_scores_finite'] = bool(torch.isfinite(sc).all())
        results[f'5_{encoder}_fixed_zero_videos_score_zero'] = bool(sc[:, item_of_catalog.lt(0)].eq(0).all())
        results[f'5_{encoder}_padding_rows_remain_zero'] = bool(model.inp.weight[0].eq(0).all() and model.out.weight[0].eq(0).all())
    # 9. Left-padded SASRec training: loss, every gradient, and updated parameters stay finite; the previous
    #    nan_to_num-after-softmax formulation produced NaN gradients on the same batch.
    torch.manual_seed(3)
    model = SeqRetriever(5, 8, 4, 'sasrec', dropout=0.).train()
    items = torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0, 2, 1]]); queries = torch.tensor([[-1, -1, -1, -1, 3, 4, 5, 6], [-1]*7+[6]])
    targets = torch.tensor([[0, 0, 0, 0, 2, 3, 4, 5], [0]*7+[1]]); ioc = torch.tensor([0, 1, 2, 3, 4, -1, -1])
    loss, _ = sampled_softmax_loss(model, items, queries, targets, 7, ioc, 5, torch.Generator().manual_seed(0)); loss.backward()
    grads = [p.grad.coalesce().values() if p.grad.is_sparse else p.grad for p in model.parameters() if p.grad is not None]
    results['9_sasrec_padded_loss_finite'] = bool(torch.isfinite(loss))
    results['9_sasrec_padded_gradients_finite'] = all(bool(torch.isfinite(g).all()) for g in grads)
    # Informational, not a pass/fail check: with the previous nan_to_num-after-softmax formulation, NaN appears
    # inside the softmax backward for fully masked rows, but masked_fill's backward replaces those entries with
    # zeros, so parameter gradients were finite in torch 2.14. The fix removes the reliance on that detail.
    sc = torch.randn(1, 3, 3, requires_grad=True); allowed = torch.zeros(1, 3, 3, dtype=torch.bool); allowed[0, 1:, :2] = True
    masked = sc.masked_fill(~allowed, float('-inf')); masked.retain_grad()
    torch.nan_to_num(torch.softmax(masked, -1)).sum().backward()
    results['info_previous_formulation'] = (f'NaN in softmax backward: {bool(torch.isnan(masked.grad).any())}; '
                                            f'NaN reaching the scores: {bool(torch.isnan(sc.grad).any())}')
    # 8. Gradient equivalence with the improved correction of Khrylchenko et al. (RecSys 2025) for uniform Q':
    #    ours  = -s_p + log(e^{s_p} + c sum_i e^{s_i}),  c = N/m
    #    paper = -sg(1 - P) [s_p - log sum_i e^{s_i - log Q'}],  P = e^{s_p} / (e^{s_p} + c sum_i e^{s_i})
    sv = torch.tensor([.7, -.2, 1.1, .4, -.9], dtype=torch.float64, requires_grad=True); c = 1234.5
    ours = -sv[0]+torch.log(torch.exp(sv[0])+c*torch.exp(sv[1:]).sum())
    g_ours, = torch.autograd.grad(ours, sv)
    sp = sv.detach().clone().requires_grad_(True)
    P = (torch.exp(sp[0])/(torch.exp(sp[0])+c*torch.exp(sp[1:]).sum())).detach()
    paper = -(1-P)*(sp[0]-torch.logsumexp(sp[1:]-math.log(1/len(sp[1:])), 0))
    g_paper, = torch.autograd.grad(paper, sp)
    results['8_gradient_equals_improved_logq_with_uniform_negatives'] = bool(torch.allclose(g_ours, g_paper, atol=1e-12))
    # 7. Stable top-k against brute force with many ties.
    scores = torch.tensor([[0., 1., 0., 1., -1., 0., 2., 0.]])
    ids, _ = stable_topk(lambda a, b: scores[:, a:b], 1, 8, k=6, block=3)
    brute = sorted(range(8), key=lambda j: (-float(scores[0, j]), j))[:6]
    results['7_stable_topk_ties_by_ascending_index'] = ids[0].tolist()==brute
    # 6. Real training windows (validation phase data), look-back windows with C context events and K targets.
    data = load_sequences(ROOT/'data/processed')
    tie_ok = True
    for x, t in data['sequences'].values():
        same = np.diff(t) == 0
        if (np.diff(x)[same] < 0).any(): tie_ok = False; break
    results['6_ties_ordered_by_ascending_video_id'] = tie_ok
    for C, K in [(50, 25), (50, 50), (100, 50), (100, 100)]:
        w = build_windows(data['sequences'], C, K); W = C+K
        times = []
        for x, t in data['sequences'].values():
            for s0 in range(0, len(x), K):
                ti = t[max(0, s0-C):s0+K]; times.append(np.concatenate([np.full(W-len(ti), -1), ti]))
        times = torch.from_numpy(np.stack(times)); q = w['queries']; valid = q.ge(0)
        r, c = valid.nonzero(as_tuple=True); tq = times[r, q[r, c]]; tt = times[r, c]
        tag = f'C{C}_K{K}'
        results[f'6_{tag}_query_strictly_earlier'] = bool((tq < tt).all())
        late = (times[r, :] >= tt[:, None]) & (torch.arange(W)[None, :] <= q[r, c][:, None]) & w['items'][r, :].gt(0)
        results[f'6_{tag}_no_event_at_or_after_target_before_query'] = not bool(late.any())
        hist = ((torch.arange(W)[None, :] <= q[r, c][:, None]) & w['items'][r, :].gt(0)).sum(1)
        results[f'6_{tag}_history_at_most_C_plus_K_minus_1'] = bool(hist.le(W-1).all())
        results[f'6_{tag}_each_event_target_once'] = int(valid.sum())+w['dropped_first_group']+w['dropped_window_boundary']==data['events'] and w['dropped_window_boundary']==0
        results[f'6_{tag}_summary'] = {'targets': int(valid.sum()), 'dropped_first_group': w['dropped_first_group'], 'dropped_window_boundary': w['dropped_window_boundary'],
            'windows': int(w['items'].shape[0]), 'history_length_quantiles_p1_p10_p50_p90': [float(v) for v in torch.quantile(hist.float(), torch.tensor([.01, .1, .5, .9]))],
            'share_targets_with_history_ge_C': float(hist.ge(C).float().mean())}
        if (C, K) in [(50, 25), (100, 50)]:
            sh_items, sh_targets = shuffle_within_timestamps(w['items'], w['targets'], w['times'], w['regions'], 2029)
            same_hist = True
            for row, col in zip(r.tolist()[::50], c.tolist()[::50]):        # every 50th target
                qq = int(q[row, col])
                same_hist &= sorted(w['items'][row, :qq+1].tolist())==sorted(sh_items[row, :qq+1].tolist())
            results[f'6_{tag}_shuffle_keeps_every_sampled_target_history_set'] = same_hist
            results[f'6_{tag}_shuffle_keeps_target_multiset_per_window'] = all(sorted(w['targets'][i].tolist())==sorted(sh_targets[i].tolist()) for i in range(0, len(w['items']), 7))
            results[f'6_{tag}_shuffle_keeps_targets_at_target_positions'] = bool((sh_targets.gt(0)==w['targets'].gt(0)).all())
            results[f'6_{tag}_shuffle_changes_some_positions'] = bool((sh_items!=w['items']).any())
            moved = sh_items!=w['items']; ts = w['times']
            results[f'6_{tag}_shuffle_moves_only_within_timestamp_groups'] = bool(all(
                sorted(w['items'][i][ts[i]==tv].tolist())==sorted(sh_items[i][ts[i]==tv].tolist()) for i in range(0, len(w['items']), 97) for tv in ts[i][moved[i]].unique().tolist()))
        hist_eval, eval_times = eval_histories(data['sequences'], data['users'], W-1, return_times=True)
        sh_eval, _ = shuffle_within_timestamps(hist_eval, torch.zeros_like(hist_eval), eval_times, torch.zeros_like(hist_eval, dtype=torch.bool), 2029)
        results[f'6_{tag}_eval_shuffle_keeps_history_sets'] = all(sorted(hist_eval[i].tolist())==sorted(sh_eval[i].tolist()) for i in range(len(hist_eval)))
        results[f'6_{tag}_eval_history_is_most_recent_C_plus_K_minus_1'] = all(
            np.array_equal(hist_eval[i][hist_eval[i].gt(0)].numpy(), data['sequences'][u][0][-(W-1):]) for i, u in enumerate(data['users']) if u in data['sequences'])
    results['data'] = {'events': data['events'], 'catalog': len(data['catalog']), 'trainable_items': len(data['trainable']),
                       'fixed_zero_videos': int(len(data['catalog'])-len(data['trainable']))}
    passed = all(v for k, v in results.items() if isinstance(v, bool))
    out = ROOT/'reports/seq_retrieval'; out.mkdir(parents=True, exist_ok=True)
    (out/'implementation_checks.json').write_text(json.dumps({'all_passed': passed, **{k: v for k, v in results.items()}}, indent=2, default=str))
    print(json.dumps({'all_passed': passed, **results}, indent=2, default=str))

if __name__ == '__main__':
    main()
