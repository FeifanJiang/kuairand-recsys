"""DLRM-style deep ranker on the 27K all-users sample S_all (reports/expansion/protocol_deep.json).

Stages (one process each):
  cache        row-level inputs of S_all, built once: unscaled numeric L2 features (feature_matrix.tree_matrix,
               i.e. feature_model.assemble per chunk), categorical codes, raw ids (user, video, author), labels and
               Shanghai dates -> artifacts/k27_experiment/deep_cache
  cache-ials   fusion inputs: early-iALS scores and has-factors flags on the late training rows, plus the fixed
               scaling constants of the early and the full model on the same label-free reference pairs
  train        one configuration and seed of deep | deep_ials | deep_late: fit, score the tune-group rows, save
Embedding regularization: fixed Gaussian prior lambda_emb * sum_rows ||e||^2 on the user / video / author tables,
applied through the rows touched in each batch with expected-touch normalization (batch penalty = lambda_emb *
n_batches * sum_i ||e(id_i)||^2 / touches(id_i), so the per-epoch total equals n_batches * lambda_emb * sum ||e||^2); SparseAdam on the ID tables, Adam elsewhere. Early stopping on a fixed
stratified 10% holdout of the training rows (split seed 0) by log loss, patience 2, best epoch restored.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import build_event_store as bes
import feature_matrix as fx
import k27_eval as ke
from k27_eval_rows import training_frame

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT/'data/k27/features_all_8m'; ART = ROOT/'artifacts/k27_experiment'; CACHE = ART/'deep_cache'; MODELS = ART/'deep'
OUT = ROOT/'reports/expansion/run_dev'; STORE = ROOT/'data/k27/standard_to_0421_clean'; VMETA = ROOT/'data/k27/video_meta'
ROWS = 8_000_000; MIN_COUNT = 5; D = 32; CAT_DIMS = [4, 4, 8]; BATCH = 4096; MAX_EPOCHS = 10; PATIENCE = 2; HOLDOUT = .1; SPLIT_SEED = 0
GRID = [dict(lr=lr, dropout=p, emb_l2=l) for lr in [1e-3, 3e-4] for p in [0., .3] for l in [1e-6, 1e-5]]
LATE_DATE = 20220415; CUTOFF_DATE = 20220421
EARLY_IALS = ART/'ials/tau2_alpha30_l21000_before20220414_s0.pt'; FULL_IALS = ART/'ials/tau2_alpha30_l21000_s0.pt'
ID_NAMES = ['user', 'video', 'author']

class DLRM(nn.Module):
    def __init__(self, n_dense, cat_sizes, id_sizes, dropout):
        super().__init__()
        self.cats = nn.ModuleList([nn.Embedding(n, k) for n, k in zip(cat_sizes, CAT_DIMS)])
        self.ids = nn.ModuleList([nn.Embedding(n, D, sparse=True) for n in id_sizes])
        self.bottom = nn.Sequential(nn.Linear(n_dense+sum(CAT_DIMS), 128), nn.ReLU(), nn.Linear(128, D), nn.ReLU())
        self.top = nn.Sequential(nn.Linear(4*D+6, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        for e in list(self.cats)+list(self.ids): nn.init.normal_(e.weight, std=.01)

    def forward(self, dense, cats, ids):
        dense = self.bottom(torch.cat([dense]+[e(cats[:, j]) for j, e in enumerate(self.cats)], 1))
        E = [e(ids[:, j]) for j, e in enumerate(self.ids)]; vecs = [dense]+E
        dots = torch.stack([(vecs[i]*vecs[j]).sum(1) for i in range(4) for j in range(i+1, 4)], 1)
        return self.top(torch.cat([dense, dots]+E, 1)).squeeze(1), E

def standardize(num, mean, sd):
    return ((num-mean.astype(np.float32))/sd.astype(np.float32)).astype(np.float32)

def ials_inputs(f, fct):
    """iALS score x_u.y_v and a has-factors flag (user and video present with non-zero factors); 0 / 0 otherwise."""
    def locate(keys, ids):
        i = np.minimum(np.searchsorted(keys, ids), len(keys)-1); return i, keys[i] == ids
    ui, uok = locate(np.asarray(fct['users']), f.user_id.to_numpy()); vi, vok = locate(np.asarray(fct['catalog']), f.video_id.to_numpy())
    xn = (fct['X'].norm(dim=1) > 0).numpy(); yn = (fct['Y'].norm(dim=1) > 0).numpy()
    ok = uok & vok & xn[ui] & yn[vi]; sc = np.zeros(len(f)); idx = np.flatnonzero(ok)
    with torch.no_grad(): sc[idx] = (fct['X'][torch.from_numpy(ui[idx])]*fct['Y'][torch.from_numpy(vi[idx])]).sum(-1).numpy()
    return sc, ok.astype(np.float32)

def stage_cache():
    f, s = training_frame(FEAT, ROWS); levels = fx.training_levels(f)
    X, names, is_cat = fx.tree_matrix(f, s, levels, level='L2'); del s
    r = pd.read_pickle(FEAT/'rows.pkl').set_index('row').loc[f.index]
    date = bes.shanghai_date(r.t.to_numpy()); assert (date <= CUTOFF_DATE).all(), 'training row after the cutoff'
    author = np.asarray(np.load(VMETA/'author_id.npy', mmap_mode='r')[r.video_pos.to_numpy()])
    CACHE.mkdir(parents=True, exist_ok=False)
    np.save(CACHE/'num.npy', X[:, ~is_cat]); np.save(CACHE/'cats.npy', X[:, is_cat].astype(np.int16))
    np.save(CACHE/'ids_raw.npy', np.stack([r.user_id.to_numpy(), r.video_id.to_numpy(), author], 1).astype(np.int64))
    np.save(CACHE/'y.npy', r.lv.to_numpy(np.float32)); np.save(CACHE/'date.npy', date.astype(np.int32)); np.save(CACHE/'row.npy', f.index.to_numpy())
    meta = {'rows': len(f), 'numeric': [n for n, c in zip(names, is_cat) if not c], 'categorical': [n for n, c in zip(names, is_cat) if c],
            'cat_sizes': [len(dict.fromkeys(levels[c]+['other', 'missing'])) for c in fx.CATS], 'levels': levels, 'late_rows': int((date >= LATE_DATE).sum())}
    (CACHE/'meta.json').write_text(json.dumps(meta)); return {k: v for k, v in meta.items() if k != 'levels'}

def stage_cache_ials():
    ids = np.load(CACHE/'ids_raw.npy'); date = np.load(CACHE/'date.npy'); late = np.flatnonzero(date >= LATE_DATE)
    early = torch.load(EARLY_IALS, weights_only=False); full = torch.load(FULL_IALS, weights_only=False)
    early_date = int(bes.shanghai_date(np.array([early['T']]))[0]); assert early_date < LATE_DATE, 'early iALS data overlaps the late rows'
    sc, flag = ials_inputs(pd.DataFrame({'user_id': ids[late, 0], 'video_id': ids[late, 1]}), early)
    np.save(CACHE/'ials_early_late_score.npy', sc); np.save(CACHE/'ials_early_late_flag.npy', flag)
    e = pd.read_pickle(ART/'eval/rows.pkl'); ref = e.loc[(e.group.eq('tune') & e.user_has_long_view & e.scope_A).to_numpy(), ['user_id', 'video_id']]   # reference pairs, no label
    se, _ = ials_inputs(ref, early); sf, _ = ials_inputs(ref, full)
    const = {'reference_pairs': 'tune-group random exposures, main population, scope A (user, video) pairs; scores only', 'n_reference_pairs': int(len(ref)),
             'early': [float(se.mean()), float(se.std(ddof=1))], 'full': [float(sf.mean()), float(sf.std(ddof=1))], 'early_last_shanghai_date': early_date,
             'late_rows': int(len(late)), 'late_rows_flag_share': float(flag.mean()), 'reference_flag_share_early': float(ials_inputs(ref, early)[1].mean())}
    hyb = json.loads((OUT/'hybrid.json').read_text())['standardization']['iALS_mean_sd_per_seed'][0]
    assert np.allclose(const['full'], hyb, rtol=1e-9), 'full-model constants differ from hybrid.json'
    (CACHE/'ials_constants.json').write_text(json.dumps(const, indent=2)); return const

def load_rows(model):
    meta = json.loads((CACHE/'meta.json').read_text()); date = np.load(CACHE/'date.npy')
    sel = np.arange(len(date)) if model == 'deep' else np.flatnonzero(date >= LATE_DATE)
    num = np.asarray(np.load(CACHE/'num.npy', mmap_mode='r')[sel]); cats = np.asarray(np.load(CACHE/'cats.npy', mmap_mode='r')[sel])
    ids_raw = np.asarray(np.load(CACHE/'ids_raw.npy', mmap_mode='r')[sel]); y = np.load(CACHE/'y.npy')[sel]; extra = None
    if model == 'deep_ials':
        c = json.loads((CACHE/'ials_constants.json').read_text()); sc = np.load(CACHE/'ials_early_late_score.npy'); fl = np.load(CACHE/'ials_early_late_flag.npy')
        extra = np.stack([(sc-c['early'][0])/c['early'][1], fl], 1).astype(np.float32)
    return meta, num, cats, ids_raw, y, extra

def split(y):
    rng = np.random.default_rng(SPLIT_SEED); hold = np.zeros(len(y), bool)
    for lab in [0, 1]:
        idx = np.flatnonzero(y == lab); hold[rng.choice(idx, int(round(HOLDOUT*len(idx))), replace=False)] = True
    return np.flatnonzero(~hold), np.flatnonzero(hold)

def vocabularies(ids_raw, tr):
    voc = {'user': np.unique(np.asarray(bes.load(STORE)['user_id'])).astype(np.int64)}
    for j, name in [(1, 'video'), (2, 'author')]:
        v, c = np.unique(ids_raw[tr, j], return_counts=True); voc[name] = v[c >= MIN_COUNT]
    return voc

def index_ids(ids_raw, voc):
    cols = []
    for j, name in enumerate(ID_NAMES):
        k = voc[name]
        if len(k) == 0: cols.append(np.zeros(len(ids_raw), np.int64)); continue                      # empty vocabulary: everything is rare
        i = np.minimum(np.searchsorted(k, ids_raw[:, j]), len(k)-1); cols.append(np.where(k[i] == ids_raw[:, j], i+1, 0))   # 0 = unknown / rare
    return np.stack(cols, 1).astype(np.int64)

def batch_penalty(E, ids_batch, counts, lam, n_batches):
    """Expected-touch-normalized prior on the touched rows: the per-batch penalties of one epoch sum exactly to n_batches * lam * sum_rows ||e||^2."""
    return sum((E[j].pow(2).sum(1)/counts[j][ids_batch[:, j]]).sum() for j in range(len(E)))*(lam*n_batches)

def fit(cfg, seed, num, cats, ids, y, extra, tr, ho, meta, voc, log=print):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mean = num[tr].astype(np.float64).mean(0); sd = num[tr].astype(np.float64).std(0, ddof=1); sd[sd == 0] = 1
    dense = standardize(num, mean, sd)
    if extra is not None: dense = np.concatenate([dense, extra], 1)
    Xd, Xc, Xi, Y = torch.from_numpy(dense), torch.from_numpy(cats.astype(np.int64)), torch.from_numpy(ids), torch.from_numpy(y)
    counts = [torch.from_numpy(np.maximum(np.bincount(ids[tr, j], minlength=len(voc[n])+1), 1).astype(np.float32)) for j, n in enumerate(ID_NAMES)]
    net = DLRM(dense.shape[1], meta['cat_sizes'], [len(voc[n])+1 for n in ID_NAMES], cfg['dropout'])
    sparse = torch.optim.SparseAdam(list(net.ids.parameters()), lr=cfg['lr'])
    other = torch.optim.Adam([p for n, p in net.named_parameters() if not n.startswith('ids.')], lr=cfg['lr'])
    N = len(tr); n_batches = int(np.ceil(N/BATCH)); lam = cfg['emb_l2']; best = (np.inf, None, 0); wait = 0; curve = []
    def holdout_loss():
        net.eval(); tot = 0.
        with torch.no_grad():
            for a in range(0, len(ho), 65536):
                idx = torch.from_numpy(ho[a:a+65536]); tot += float(F.binary_cross_entropy_with_logits(net(Xd[idx], Xc[idx], Xi[idx])[0], Y[idx], reduction='sum'))
        return tot/len(ho)
    for epoch in range(1, MAX_EPOCHS+1):
        net.train(); perm = tr[rng.permutation(N)]; t0 = time.time(); tl = 0.
        for a in range(0, N, BATCH):
            idx = torch.from_numpy(perm[a:a+BATCH]); logit, E = net(Xd[idx], Xc[idx], Xi[idx])
            loss = F.binary_cross_entropy_with_logits(logit, Y[idx]); pen = batch_penalty(E, Xi[idx], counts, lam, n_batches)
            sparse.zero_grad(); other.zero_grad(); (loss+pen).backward(); sparse.step(); other.step(); tl += float(loss)*len(idx)
        assert np.isfinite(tl), 'non-finite training loss'
        hl = holdout_loss(); curve.append({'epoch': epoch, 'train_log_loss': tl/N, 'holdout_log_loss': hl, 'seconds': time.time()-t0}); log(json.dumps(curve[-1]))
        if hl < best[0]-1e-7: best = (hl, copy.deepcopy(net.state_dict()), epoch); wait = 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    net.load_state_dict(best[1]); net.eval()
    return net, mean, sd, {'curve': curve, 'best_epoch': best[2], 'best_holdout_log_loss': best[0], 'epochs_run': len(curve)}

def inputs_from_frame(f, s, model):
    X, names, is_cat = fx.tree_matrix(f, s, model['levels'], level='L2')
    dense = standardize(X[:, ~is_cat], model['mean'], model['sd']); cats = X[:, is_cat].astype(np.int64)
    ids = index_ids(np.stack([f.user_id.to_numpy(), f.video_id.to_numpy(), f.author_id.to_numpy()], 1).astype(np.int64), model['vocab'])
    return dense, cats, ids

def score_frame(model, f, s, full_ials=None):
    dense, cats, ids = inputs_from_frame(f, s, model)
    if model['model'] == 'deep_ials':
        sc, fl = ials_inputs(f, full_ials); c = model['ials_constants']['full']
        dense = np.concatenate([dense, np.stack([(sc-c[0])/c[1], fl], 1).astype(np.float32)], 1)
    net = DLRM(dense.shape[1], model['cat_sizes'], model['id_sizes'], model['cfg']['dropout']); net.load_state_dict(model['state_dict']); net.eval(); out = []
    with torch.no_grad():
        for a in range(0, len(f), 65536):
            out.append(net(torch.from_numpy(dense[a:a+65536]), torch.from_numpy(cats[a:a+65536]), torch.from_numpy(ids[a:a+65536]))[0].numpy())
    return np.concatenate(out).astype(np.float64)

def fusion_columns(f, s, X, full_ials, const_key, c):
    """Append the two iALS inputs (standardized score, has-factors flag) to a feature matrix of frame f."""
    sc, fl = ials_inputs(f, full_ials); k = c[const_key]
    return np.concatenate([X, np.stack([(sc-k[0])/k[1], fl], 1).astype(np.float32)], 1)

def score_lr_fusion(model, f, s, full_ials):
    X, _ = fx.standardized(f, s, 'L2', model['levels'], stats=(model['columns'][:-2], model['mean'], model['sd']))
    X = fusion_columns(f, s, X, full_ials, 'full', model['ials_constants'])
    with torch.no_grad(): return (torch.from_numpy(X)@model['w'].float()+model['b'].float()).numpy().astype(np.float64)

def stage_lr_fusion():
    """LR-L2 feature set (standardized, one-hot categoricals as in feature_model) + the two iALS inputs on the late rows of S_all; l2 grid; tune scores saved."""
    import feature_model as fm
    f, s = training_frame(FEAT, ROWS); date = np.load(CACHE/'date.npy'); row = np.load(CACHE/'row.npy'); assert np.array_equal(row, f.index.to_numpy()), 'cache order differs from the training frame'
    late = np.flatnonzero(date >= LATE_DATE); f, s = f.iloc[late], s.iloc[late]
    c = json.loads((CACHE/'ials_constants.json').read_text()); sc = np.load(CACHE/'ials_early_late_score.npy'); fl = np.load(CACHE/'ials_early_late_flag.npy')
    levels = fx.training_levels(f); X, (cols, mean, sd) = fx.standardized(f, s, 'L2', levels); del s
    X = np.concatenate([X, np.stack([(sc-c['early'][0])/c['early'][1], fl], 1).astype(np.float32)], 1)
    y = torch.from_numpy(f.lv.to_numpy(np.float32)); Xt = torch.from_numpy(X); full = torch.load(FULL_IALS, weights_only=False)
    e = pd.read_pickle(ART/'eval/rows.pkl'); se = pd.read_pickle(ART/'eval/history.pkl'); m = e.group.eq('tune').to_numpy(); et, st = e.loc[m], se.loc[m]
    MODELS.mkdir(parents=True, exist_ok=True); res = []
    for l2 in [1e-4, 1e-3, 1e-2]:
        w, b, obj, grad = fm.fit_lr(Xt, y, l2); name = f'LR-L2+iALS_late_l2{l2:g}'
        model = {'w': w, 'b': b, 'columns': cols+['ials_z', 'ials_flag'], 'mean': mean, 'sd': sd, 'levels': levels, 'level': 'L2', 'l2': l2, 'late_rows': int(len(late)), 'ials_constants': c, 'fusion': True}
        torch.save(model, MODELS/f'{name}.pt'); sct = score_lr_fusion(model, et, st, full); assert np.isfinite(sct).all() and np.isfinite(obj)
        np.save(MODELS/f'{name}_tune_scores.npy', sct)
        res.append({'name': name, 'l2': l2, 'train_objective': obj, 'grad_norm': grad, 'training_rows': int(len(late)), 'tune_auc': float(ke.user_auc(et, sct, (et.user_has_long_view & et.scope_A).to_numpy()).mean())})
    return res

def stage_train(model, cfg_idx, seed, smoke=None, max_epochs=None):
    global MAX_EPOCHS
    if max_epochs: MAX_EPOCHS = max_epochs
    cfg = GRID[cfg_idx]; meta, num, cats, ids_raw, y, extra = load_rows(model)
    if smoke:
        num, cats, ids_raw, y = num[:smoke], cats[:smoke], ids_raw[:smoke], y[:smoke]; extra = extra[:smoke] if extra is not None else None
    tr, ho = split(y); assert len(np.intersect1d(tr, ho)) == 0 and len(tr)+len(ho) == len(y), 'holdout overlaps training rows'
    voc = vocabularies(ids_raw, tr); ids = index_ids(ids_raw, voc); name = f'{model}_c{cfg_idx}_s{seed}'+('_smoke' if smoke else '')
    net, mean, sd, info = fit(cfg, seed, num, cats, ids, y, extra, tr, ho, meta, voc)
    saved = {'model': model, 'cfg': cfg, 'seed': seed, 'levels': meta['levels'], 'cat_sizes': meta['cat_sizes'], 'id_sizes': [len(voc[n])+1 for n in ID_NAMES], 'vocab': voc,
             'mean': mean, 'sd': sd, 'state_dict': net.state_dict(), 'training_rows': int(len(tr)), 'holdout_rows': int(len(ho)),
             'ials_constants': json.loads((CACHE/'ials_constants.json').read_text()) if model == 'deep_ials' else None}
    MODELS.mkdir(parents=True, exist_ok=True); torch.save(saved, MODELS/f'{name}.pt')
    e = pd.read_pickle(ART/'eval/rows.pkl'); se = pd.read_pickle(ART/'eval/history.pkl'); m = e.group.eq('tune').to_numpy(); et, st = e.loc[m], se.loc[m]
    assert bes.shanghai_date(et.t.to_numpy()).min() > CUTOFF_DATE, 'evaluation row before the cutoff'
    sc = score_frame(saved, et, st, torch.load(FULL_IALS, weights_only=False) if model == 'deep_ials' else None)
    assert np.isfinite(sc).all(), 'non-finite scores'; np.save(MODELS/f'{name}_tune_scores.npy', sc)
    tune_auc = float(ke.user_auc(et, sc, (et.user_has_long_view & et.scope_A).to_numpy()).mean())
    return {'name': name, 'model': model, 'config': cfg, 'config_index': cfg_idx, 'seed': seed, **info, 'vocab_sizes': {n: int(len(voc[n])) for n in ID_NAMES},
            'training_rows': int(len(tr)), 'holdout_rows': int(len(ho)), 'tune_auc': tune_auc}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['cache', 'cache-ials', 'train', 'lr-fusion']); p.add_argument('--model', choices=['deep', 'deep_ials', 'deep_late'])
    p.add_argument('--config', type=int); p.add_argument('--seed', type=int, default=0); p.add_argument('--smoke', type=int); p.add_argument('--max-epochs', type=int); p.add_argument('--out')
    a = p.parse_args(); torch.set_num_threads(8); t0 = time.time()
    r = stage_cache() if a.stage == 'cache' else stage_cache_ials() if a.stage == 'cache-ials' else stage_lr_fusion() if a.stage == 'lr-fusion' else stage_train(a.model, a.config, a.seed, a.smoke, a.max_epochs)
    rec = {'stage': a.stage, 'seconds': time.time()-t0, 'result': r}
    dest = Path(a.out) if a.out else OUT/(f'deep_{a.stage}.json' if a.stage != 'train' else f"{r['name']}.json")
    dest.write_text(json.dumps(rec, indent=2, default=float)); print(json.dumps({**rec, 'result': {k: v for k, v in r.items() if k != 'curve'} if isinstance(r, dict) else r}, indent=2, default=float))
