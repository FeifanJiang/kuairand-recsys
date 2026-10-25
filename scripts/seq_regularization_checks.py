"""Checks for the embedding prior in seq_retrieval.py / train_seq.py (validation-phase data only).

1. Input table: summed over one epoch of batches, the normalized penalty equals sum |input rows|^2 exactly.
2. Output table: averaged over many epochs of uniform negative draws, the normalized penalty matches
   sum |output rows|^2 (expectation), within Monte Carlo error.
3. Real data, embed_l2 = 0: the first two epochs' training losses match the earlier runs to a relative 1e-5, for
   the selected pooling and SASRec configurations. Serial runs are bitwise reproducible, but concurrent runs
   (as in the earlier experiment) change float accumulation order (measured: ~1e-9 relative for pooling and
   ~1e-6 for SASRec after one epoch), so bitwise equality with those runs is not attainable.
"""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from seq_retrieval import SeqRetriever, sampled_softmax_loss, embedding_touch_counts, embedding_penalty

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT/'reports/seq_retrieval/runs_20261020'

def main():
    out = {}; torch.manual_seed(0)
    items = torch.tensor([[0, 1, 2, 3, 2], [0, 0, 4, 1, 5], [3, 3, 1, 2, 4], [0, 5, 5, 1, 2]])
    queries = torch.tensor([[-1, -1, 1, 2, 3], [-1, -1, -1, 2, 3], [-1, 1, 2, 3, 3], [-1, -1, 2, 3, 3]])
    targets = torch.where(queries.ge(0), items, torch.zeros_like(items))
    ioc = torch.tensor([0, 1, 2, 3, 4, -1, -1, -1]); n, m, P = 8, 6, 5
    model = SeqRetriever(P, 5, 3, 'pool')
    c_in, c_out, T = embedding_touch_counts(items, targets, P, m, n)
    g = torch.Generator().manual_seed(1); epoch_in = 0.
    for b in [[0, 1], [2, 3]]:
        idx = torch.tensor(b); _, _, pos, neg = sampled_softmax_loss(model, items[idx], queries[idx], targets[idx], n, ioc, m, g, return_touched=True)
        used = items[idx][items[idx].gt(0)]; epoch_in += float((model.inp(used).pow(2).sum(-1)/c_in[used].float()).sum())
    out['1_input_penalty_epoch_sum_exact'] = abs(epoch_in-float(model.inp.weight[1:].pow(2).sum())) < 1e-5
    totals = []
    for e in range(4000):
        tot = 0.
        for b in [[0, 1], [2, 3]]:
            idx = torch.tensor(b); _, _, pos, neg = sampled_softmax_loss(model, items[idx], queries[idx], targets[idx], n, ioc, m, g, return_touched=True)
            used = torch.cat([pos, neg.reshape(-1)]); used = used[used.gt(0)]
            tot += float((model.out(used).pow(2).sum(-1)/c_out[used].float()).sum())
        totals.append(tot)
    exact = float(model.out.weight[1:].pow(2).sum()); mean, se = np.mean(totals), np.std(totals)/np.sqrt(len(totals))
    out['2_output_penalty_epoch_expectation'] = {'monte_carlo_mean': mean, 'standard_error': se, 'target': exact, 'within_3_se': abs(mean-exact) < 3*se}
    # 3. Bitwise regression of the embed_l2 = 0 path on real data (two epochs, scratch output).
    scratch = Path(sys.argv[1])
    for enc, C, d in [('pool', 100, 64), ('sasrec', 50, 64)]:
        # No evaluation runs in two epochs, so the script stops at checkpoint loading after writing both curve rows;
        # only the curve is compared.
        subprocess.run([sys.executable, str(ROOT/'scripts/train_seq.py'), '--encoder', enc, '--C', str(C), '--dim', str(d), '--max-epochs', '2',
                        '--eval-every', '100', '--out', str(scratch)], capture_output=True)
        new = pd.read_csv(scratch/f'{enc}_C{C}_d{d}_s42/curve.csv').train_loss.to_numpy()
        assert len(new)==2, 'regression run did not complete two epochs'
        old = pd.read_csv(PREVIOUS/f'{enc}_C{C}_d{d}_s42/curve.csv').train_loss.to_numpy()[:2]
        rel = float(np.max(np.abs(new-old)/np.abs(old)))
        out[f'3_{enc}_l2_0_first_two_epoch_losses_match'] = rel < 1e-5; out[f'3_{enc}_max_relative_difference'] = rel
    passed = out['1_input_penalty_epoch_sum_exact'] and out['2_output_penalty_epoch_expectation']['within_3_se'] and all(v for k, v in out.items() if k.startswith('3_') and k.endswith('_match'))
    dest = ROOT/'reports/seq_retrieval/regularization'; dest.mkdir(parents=True, exist_ok=True)
    (dest/'implementation_checks.json').write_text(json.dumps({'all_passed': bool(passed), **out}, indent=2, default=float))
    print(json.dumps({'all_passed': bool(passed), **out}, indent=2, default=float))

if __name__ == '__main__':
    main()
