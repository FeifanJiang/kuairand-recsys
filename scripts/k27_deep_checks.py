"""Checks of k27_deep.py before any grid run (protocol_deep.json 'checks_before_running'); no evaluation label is used.

1. Cache consistency: for 200 random cached rows the raw ids, labels and dates equal rows.pkl / video_meta; every
   cached date is on or before the cutoff; the late-row count matches the date rule.
2. Evaluation-path consistency: the same 200 rows rebuilt through the evaluation frame builder
   (k27_eval_rows.attach_video) with their training-time prequential statistics give identical numeric inputs,
   categorical codes and id indices to the cache.
3. Holdout split: stratified, disjoint, fixed by SPLIT_SEED (two calls identical); evaluation rows all after the cutoff.
4. Penalty normalization: on a toy table with skewed touch counts, the per-batch penalties summed over one epoch equal
   n_batches * lambda * sum_rows ||e||^2 (relative error < 1e-5).
5. Determinism: two smoke runs (200k rows, 1 epoch, configuration 0, seed 0) give bitwise identical tune scores.
"""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import feature_model as fm
import build_event_store as bes
import k27_eval_rows as er
import k27_deep as kd

ROOT = Path(__file__).resolve().parents[1]

def main():
    out = {}; meta = json.loads((kd.CACHE/'meta.json').read_text()); rng = np.random.default_rng(3)
    num = np.load(kd.CACHE/'num.npy', mmap_mode='r'); cats = np.load(kd.CACHE/'cats.npy', mmap_mode='r'); ids = np.load(kd.CACHE/'ids_raw.npy', mmap_mode='r')
    y = np.load(kd.CACHE/'y.npy'); date = np.load(kd.CACHE/'date.npy'); row = np.load(kd.CACHE/'row.npy'); pick = np.sort(rng.choice(len(y), 200, replace=False))
    r = pd.read_pickle(kd.FEAT/'rows.pkl').set_index('row').loc[row[pick]]
    author = np.asarray(np.load(kd.VMETA/'author_id.npy', mmap_mode='r')[r.video_pos.to_numpy()])
    out['1_ids_labels_match_rows'] = bool((ids[pick, 0] == r.user_id.to_numpy()).all() and (ids[pick, 1] == r.video_id.to_numpy()).all() and (ids[pick, 2] == author).all() and (y[pick] == r.lv.to_numpy()).all())
    out['1_all_dates_on_or_before_cutoff'] = bool((date <= kd.CUTOFF_DATE).all()); out['1_late_rows_match'] = int((date >= kd.LATE_DATE).sum()) == meta['late_rows']
    raw = pd.DataFrame({'user_id': r.user_id.to_numpy(), 'video_id': r.video_id.to_numpy(), 'time_ms': r.t.to_numpy(), 'long_view': r.lv.to_numpy(), 'duration_ms': r.duration_ms.to_numpy(), 'tab': r.tab.to_numpy()})
    f = er.attach_video(raw); s = pd.concat([pd.read_pickle(kd.FEAT/f'pre_{k}.pkl').loc[row[pick]] for k in fm.KEYS], axis=1).reset_index(drop=True)
    voc = {'user': kd.vocabularies(np.asarray(ids[pick]), np.arange(200))['user'], 'video': np.unique(ids[pick, 1]), 'author': np.unique(ids[pick, 2])}   # every sampled id in the vocabulary
    model = {'levels': meta['levels'], 'mean': np.zeros(len(meta['numeric'])), 'sd': np.ones(len(meta['numeric'])), 'vocab': voc}
    dense, c2, i2 = kd.inputs_from_frame(f, s, model)
    out['2_numeric_identical'] = bool(np.array_equal(dense, np.asarray(num[pick]), equal_nan=True)); out['2_categorical_identical'] = bool(np.array_equal(c2, np.asarray(cats[pick]).astype(np.int64)))
    out['2_ids_identical'] = bool(np.array_equal(i2, kd.index_ids(np.asarray(ids[pick]), voc)))
    tr, ho = kd.split(y); tr2, ho2 = kd.split(y)
    out['3_split_disjoint_and_complete'] = bool(len(np.intersect1d(tr, ho)) == 0 and len(tr)+len(ho) == len(y)); out['3_split_fixed'] = bool(np.array_equal(ho, ho2))
    out['3_holdout_positive_rate_matches'] = bool(abs(float(y[ho].mean())-float(y.mean())) < 1e-3)
    e = pd.read_pickle(kd.ART/'eval/rows.pkl'); out['3_all_evaluation_rows_after_cutoff'] = bool(bes.shanghai_date(e.t.to_numpy()).min() > kd.CUTOFF_DATE)
    sizes = [50, 300, 1000]; ids_toy = np.stack([rng.choice(n, 3000, p=(w := rng.random(n)**3)/w.sum()) for n in sizes], 1)   # skewed touch counts, 3000 rows per table
    E_tables = [torch.randn(n, kd.D) for n in sizes]; counts = [torch.from_numpy(np.maximum(np.bincount(ids_toy[:, j], minlength=n), 1).astype(np.float32)) for j, n in enumerate(sizes)]
    lam = 1e-3; total = 0.; B = 256; n_batches = int(np.ceil(len(ids_toy)/B))
    for a in range(0, len(ids_toy), B):
        ib = torch.from_numpy(ids_toy[a:a+B]); total += float(kd.batch_penalty([E_tables[j][ib[:, j]] for j in range(3)], ib, counts, lam, n_batches))
    touched = sum(float(E_tables[j][np.unique(ids_toy[:, j])].pow(2).sum()) for j in range(3))
    out['4_penalty_epoch_total_rel_error'] = abs(total-n_batches*lam*touched)/(n_batches*lam*touched); out['4_penalty_normalization_ok'] = out['4_penalty_epoch_total_rel_error'] < 1e-5
    scores = []
    for k in range(2):
        subprocess.run([sys.executable, str(ROOT/'scripts/k27_deep.py'), 'train', '--model', 'deep', '--config', '0', '--seed', '0', '--smoke', '200000', '--max-epochs', '1',
                        '--out', str(ROOT/f'reports/expansion/run_dev/deep_smoke_run{k}.json')], check=True, cwd=ROOT/'scripts', capture_output=True)
        scores.append(np.load(kd.MODELS/'deep_c0_s0_smoke_tune_scores.npy'))
    out['5_two_smoke_runs_bitwise_identical'] = bool(np.array_equal(scores[0], scores[1]))
    out['5_smoke_epoch_seconds_200k_rows'] = json.loads((ROOT/'reports/expansion/run_dev/deep_smoke_run1.json').read_text())['result']['curve'][0]['seconds']
    passed = all(v for k, v in out.items() if isinstance(v, bool))
    (ROOT/'reports/expansion/deep_checks.json').write_text(json.dumps({'all_passed': passed, **out}, indent=2, default=float)); print(json.dumps({'all_passed': passed, **out}, indent=2, default=float))

if __name__ == '__main__':
    main()
