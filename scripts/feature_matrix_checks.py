"""feature_matrix equals the in-memory path of feature_model.main on KuaiRand-1K training rows.

On a fixed 1.5M-row subset of data/processed/train.csv.gz (with its own prequential features):
1. training_levels equals the levels assemble derives itself;
2. chunked standardized L1 / L2 matrices equal ((X - X.mean()) / X.std().replace(0, 1)).astype(float32);
3. tree_matrix numeric columns equal the L2 numeric columns cast to float32, and category codes decode to
   the same one-hot level as assemble's indicator columns.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import feature_model as fm
import feature_matrix as fx

ROOT = Path(__file__).resolve().parents[1]

def main():
    train = fm.load_frame('train', fm.load_meta()); T0 = int(train.t.min())
    pre = pd.concat([fm.prequential(train, k, T0) for k in fm.KEYS], axis=1)
    keep = np.sort(np.random.default_rng(5).choice(len(train), 1_500_000, replace=False))
    f, s = train.iloc[keep].reset_index(drop=True), pre.iloc[keep].reset_index(drop=True); del train, pre
    out = {}
    for level in ['L1', 'L2']:
        X, levels = fm.assemble(f, s, level); out['1_levels_equal'] = fx.training_levels(f) == levels
        ref = ((X-X.mean())/X.std().replace(0, 1)).to_numpy(np.float32)
        mine, (cols, _, _) = fx.standardized(f, s, level, levels, chunk=200_000)
        out[f'2_{level}_columns_equal'] = cols == list(X.columns)
        out[f'2_{level}_max_abs_diff'] = float(np.abs(mine-ref).max()); out[f'2_{level}_equal_within_1e-5'] = bool(np.allclose(mine, ref, rtol=0, atol=1e-5))
        if level == 'L2':
            T, names, is_cat = fx.tree_matrix(f, s, levels, chunk=200_000); num = [c for c in X.columns if '=' not in c]
            out['3_tree_numeric_identical'] = bool(np.array_equal(T[:, ~is_cat], X[num].to_numpy(np.float32), equal_nan=True))
            ok = True
            for j, c in enumerate(fx.CATS):
                cats = list(dict.fromkeys(levels[c]+['other', 'missing'])); dec = np.array(cats, dtype=object)[T[:, len(num)+j].astype(int)]
                onehot = np.array(cats, dtype=object)[np.argmax(np.stack([X[f'{c}={v}'].to_numpy() for v in cats], 1), 1)]
                ok &= bool((dec == onehot).all())
            out['3_tree_categories_match_onehot'] = ok
        del X, ref, mine
    passed = all(v for k, v in out.items() if 'max_abs' not in k)
    (ROOT/'reports/expansion/feature_matrix_checks.json').write_text(json.dumps({'all_passed': passed, **out}, indent=2)); print(json.dumps({'all_passed': passed, **out}, indent=2))

if __name__ == '__main__':
    main()
