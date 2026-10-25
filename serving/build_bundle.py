"""Assemble the serving bundle from the frozen study artifacts (seed 0; nothing is retrained). Run from the repository root:

    python serving/build_bundle.py --version v1

Outputs under artifacts/serving_bundle/<version>/:
  duration_ms.npy                         video duration from video_features_basic_27k (milliseconds; -1 when missing), aligned with data/k27/video_meta/video_id.npy
  stats_<key>.npz                         sorted-key lookup arrays for the frozen statistics tables (keys g, v, a, tg, u, ua, ut), same semantics as feature_model.query
  seen.npz                                pre-cutoff long views per user (CSR), for the seen filter
  time_pop.npz                            top videos by time-decayed long-view popularity at the cutoff (tau 0.5 d), candidates for unknown users
  ials.npz                                X, Y (float32), user ids, catalog ids, valid-factor masks of the tuned iALS (seed 0)
  faiss_flat_ip.index, faiss_hnsw_ip.index   exact and HNSW inner-product indexes over Y (no normalisation)
  models.joblib                           LR-L2 (columns, mean, sd, levels, w, b), M3 (w, b, l2), standardisation constants of the iALS score and the LR logit
  manifest.json                           version, cutoff, default as_of, component hashes and sizes, build timings, library versions
"""
import argparse
import hashlib
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # torch and faiss-cpu each ship an OpenMP runtime on macOS; without this the process aborts (OMP Error #15)
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import faiss
import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'scripts'))
import build_event_store as bes  # noqa: E402
import feature_model as fm  # noqa: E402

ART = ROOT/'artifacts/k27_experiment'; FEAT_DEV = ROOT/'data/k27/features_dev_8m'; FEAT_ALL = ROOT/'data/k27/features_all_8m'
VMETA = ROOT/'data/k27/video_meta'; STORE = ROOT/'data/k27/standard_to_0421_clean'; TAR = ROOT/'data/raw/k27_download/KuaiRand-27K.tar.gz'
IALS = ART/'ials/tau2_alpha30_l21000_s0.pt'; LR = ART/'LR-L2_l20.0001.pt'; ADAPT = ART/'adapt/adapted_models.joblib'; HYBRID = ROOT/'reports/expansion/run_dev/hybrid.json'
TABLE_DIR = {'g': FEAT_DEV, 'v': FEAT_DEV, 'a': FEAT_DEV, 'tg': FEAT_DEV, 'u': FEAT_ALL, 'ua': FEAT_ALL, 'ut': FEAT_ALL}   # user keys from the all-users build
POP_TAU_DAYS = .5; POP_TOP = 2000; HNSW_M = 32; HNSW_EF_CONSTRUCTION = 200
AS_OF_DEFAULT = int(pd.Timestamp('2022-04-22 00:00', tz='Asia/Shanghai').value//1_000_000)          # first instant after the training window

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''): h.update(block)
    return h.hexdigest()

def composite(a, b):
    return (np.asarray(a, np.int64) << 32) | np.asarray(b, np.int64)

def build_duration(out):
    t0 = time.time(); vid = np.load(VMETA/'video_id.npy'); parts = []
    for name, fh in bes.stream_members(TAR, r'video_features_basic_27k\.csv$'):
        for chunk in pd.read_csv(fh, usecols=['video_id', 'video_duration'], chunksize=2_000_000): parts.append(chunk)
    m = pd.concat(parts).sort_values('video_id'); assert np.array_equal(m.video_id.to_numpy(), vid), 'video table differs from video_meta'
    dur = m.video_duration.to_numpy(np.float64); missing = ~np.isfinite(dur); dur = np.where(missing, -1, dur).astype(np.float32)
    np.save(out/'duration_ms.npy', dur)
    return {'videos': int(len(dur)), 'missing': int(missing.sum()), 'unit': 'milliseconds (same scale as the logged duration_ms, checked in serving/check_correctness.py)', 'seconds': time.time()-t0}

def build_tables(out):
    info = {}
    for key, cols in fm.KEYS.items():
        t0 = time.time(); tab = pd.read_pickle(TABLE_DIR[key]/f'frozen_{key}.pkl'); vals = {c: tab[c].to_numpy() for c in tab.columns if c not in cols}
        if len(cols) == 1: k = tab[cols[0]].to_numpy(np.int64)
        else: k = composite(tab[cols[0]].to_numpy(), tab[cols[1]].to_numpy())
        order = np.argsort(k, kind='stable'); assert len(np.unique(k)) == len(k), f'duplicate keys in {key}'
        np.savez(out/f'stats_{key}.npz', keys=k[order], **{c: (v[order].astype(np.float64) if v.dtype.kind == 'f' else v[order].astype(np.int32)) for c, v in vals.items()})   # float64: last-long-view timestamps need it
        info[key] = {'rows': int(len(k)), 'columns': list(vals), 'seconds': time.time()-t0}
    return info

def build_seen_and_pop(out):
    t0 = time.time(); s = bes.load(STORE); lv = np.asarray(s['long_view']) == 1
    u = np.asarray(s['user_id'])[lv].astype(np.int64); v = np.asarray(s['video_id'])[lv].astype(np.int64); t = np.asarray(s['time_ms'])[lv]
    T = int(np.asarray(s['time_ms']).max()); T0 = int(np.asarray(s['time_ms']).min())
    key = (u << 32) | v; uniq = np.unique(key); users = (uniq >> 32).astype(np.int64); items = (uniq & 0xFFFFFFFF).astype(np.int32)
    seen_users, start = np.unique(users, return_index=True); offsets = np.append(start, len(items)).astype(np.int64)
    np.savez(out/'seen.npz', users=seen_users, offsets=offsets, items=items)
    w = np.exp(-(T-t)/(POP_TAU_DAYS*86_400_000)); vid, inv = np.unique(v, return_inverse=True); score = np.bincount(inv, weights=w)
    top = np.argsort(-score, kind='stable')[:POP_TOP]; np.savez(out/'time_pop.npz', video_id=vid[top], score=score[top].astype(np.float32))
    return {'users_with_long_views': int(len(seen_users)), 'pairs': int(len(items)), 'top_videos': POP_TOP, 'tau_days': POP_TAU_DAYS, 'T0': T0, 'T': T, 'seconds': time.time()-t0}

def build_ials_and_indexes(out):
    t0 = time.time(); f = torch.load(IALS, weights_only=False); X = f['X'].float().numpy(); Y = np.ascontiguousarray(f['Y'].float().numpy())
    users = np.asarray(f['users'], np.int64); catalog = np.asarray(f['catalog'], np.int64)
    np.savez(out/'ials.npz', X=X, Y=Y, users=users, catalog=catalog, user_valid=np.linalg.norm(X, axis=1) > 0, item_valid=np.linalg.norm(Y, axis=1) > 0, T=int(f['T']))
    info = {'users': int(len(users)), 'items': int(len(catalog)), 'dim': int(Y.shape[1]), 'seconds_load': time.time()-t0}
    t1 = time.time(); flat = faiss.IndexFlatIP(Y.shape[1]); flat.add(Y); faiss.write_index(flat, str(out/'faiss_flat_ip.index')); info['seconds_flat_build'] = time.time()-t1
    t2 = time.time(); hnsw = faiss.IndexHNSWFlat(Y.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT); hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION; hnsw.add(Y)
    faiss.write_index(hnsw, str(out/'faiss_hnsw_ip.index')); info.update({'seconds_hnsw_build': time.time()-t2, 'hnsw_M': HNSW_M, 'hnsw_efConstruction': HNSW_EF_CONSTRUCTION, 'faiss_threads': faiss.omp_get_max_threads()})
    return info

def build_models(out):
    lr = torch.load(LR, weights_only=False); A = joblib.load(ADAPT); h = json.loads(HYBRID.read_text()); sel = json.loads((ROOT/'reports/expansion/run_dev/selection_adapt.json').read_text())
    m3 = A['final']['M3_L2_iALS_refit']
    models = {'lr_l2': {'columns': lr['columns'], 'mean': np.asarray(lr['mean'], np.float64), 'sd': np.asarray(lr['sd'], np.float64), 'levels': lr['levels'], 'w': lr['w'].numpy().astype(np.float64), 'b': float(lr['b']), 'l2': lr['l2'], 'name': 'LR-L2_l20.0001'},
              'm3': {'w': np.asarray(m3['w'], np.float64), 'b': float(m3['b']), 'l2': m3['l2'], 'train_rows': m3['train_rows'], 'design': 'LR-L2 standardized columns + [z_iALS, has_factors]', 'name': 'M3_L2_iALS_refit (refit on all development week-1 random rows)'},
              'constants': {'lr_logit_mean_sd': h['standardization']['LR-L2_mean_sd'], 'ials_score_mean_sd_seed0': h['standardization']['iALS_mean_sd_per_seed'][0]},
              'selection_adapt_best': sel['best_adapted_method']}
    assert len(models['m3']['w']) == len(models['lr_l2']['columns'])+2; joblib.dump(models, out/'models.joblib'); return {'lr_columns': len(lr['columns']), 'm3_coefficients': len(models['m3']['w'])}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--version', default='v1'); p.add_argument('--resume', action='store_true', help='keep finished steps (their _info_*.json exists)'); a = p.parse_args()
    out = ROOT/'artifacts/serving_bundle'/a.version; out.mkdir(parents=True, exist_ok=a.resume); t0 = time.time(); info = {}
    def step(name, fn, outputs):
        marker = out/f'_info_{name}.json'
        if a.resume and marker.exists() and all((out/o).exists() for o in outputs): info[name] = json.loads(marker.read_text()); print(name, 'kept', flush=True); return
        info[name] = fn(out); marker.write_text(json.dumps(info[name], default=float)); print(name, info[name], flush=True)
    step('duration', build_duration, ['duration_ms.npy']); step('tables', build_tables, [f'stats_{k}.npz' for k in fm.KEYS]); step('seen_pop', build_seen_and_pop, ['seen.npz', 'time_pop.npz'])
    step('ials', build_ials_and_indexes, ['ials.npz', 'faiss_flat_ip.index', 'faiss_hnsw_ip.index']); step('models', build_models, ['models.joblib'])
    components = {f.name: {'bytes': f.stat().st_size, 'sha256': sha256(f)} for f in sorted(out.iterdir()) if f.name != 'manifest.json' and not f.name.startswith('_info_')}
    sources = {str(f.relative_to(ROOT)): sha256(f) for f in [IALS, LR, ADAPT, HYBRID]}
    manifest = {'bundle_version': a.version, 'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': 0,
                'cutoff': {'T0_ms': info['seen_pop']['T0'], 'T_ms': info['seen_pop']['T'], 'last_training_date_shanghai': 20220421},
                'as_of_default_ms': AS_OF_DEFAULT, 'as_of_rule': 'request parameter; must be after T_ms; default = 2022-04-22 00:00 Asia/Shanghai; features are frozen at the cutoff and aged to as_of',
                'retrieval': {'model': 'tuned iALS tau 2 / alpha 30 / l2 1000, seed 0', 'metric': 'inner product, no normalisation', 'indexes': ['faiss_flat_ip (exact)', 'faiss_hnsw_ip']},
                'ranking': {'model': 'M3 = LR-L2 features + standardized iALS score + has-factors flag, refit on 98,021 development week-1 random-exposure labels'},
                'build_info': info, 'components': components, 'source_artifacts_sha256': sources,
                'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__, 'faiss': faiss.__version__},
                'build_seconds_total': time.time()-t0}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, default=float)); print(json.dumps({k: manifest[k] for k in ['bundle_version', 'cutoff', 'as_of_default_ms', 'build_seconds_total']}, indent=2))

if __name__ == '__main__':
    main()
