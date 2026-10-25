"""Correctness checks of the serving layer (serving/README.md, completion criteria 1-3). Run from the repository root.

1. Score equivalence: on 2,000 development week-1 random exposures (tune group) the service's M3 score with the logged
   duration must equal the offline score (k27_adapt.inputs / design); the intermediate statistics must equal
   feature_model.query's output and the design matrix must be identical.
2. Duration source: the same rows scored with the video-table duration; agreement share, difference distribution, score
   and within-user rank changes are reported (no requirement that outputs match).
3. Behaviour: backfill after the seen filter, unknown user, zero-factor user, as_of before the cutoff, unknown video.
4. Retrieval agreement: HNSW vs exact top-200 on 1,000 known users at several efSearch, with per-query latency.
Writes reports/expansion/serving/correctness.json.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # torch and faiss-cpu each ship an OpenMP runtime on macOS; without this the process aborts (OMP Error #15)
import json
import subprocess
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'scripts')); sys.path.insert(0, str(ROOT/'serving'))
ART = ROOT/'artifacts/k27_experiment'; OUT = ROOT/'reports/expansion/serving'; REF = OUT/'_offline_reference.joblib'
STAT_COLS = ['g_n', 'g_l', 'v_n', 'v_l', 'v_d0.5', 'v_d2', 'v_lastlv', 'a_n', 'a_l', 'a_d0.5', 'a_d2', 'a_lastlv', 'tg_n', 'tg_l', 'u_n', 'u_l', 'u_lv_logdur', 'u_lv_short', 'ua_n', 'ua_l', 'ut_n', 'ut_l']

def pair_flips(users, a, b):
    """Share of within-user pairs whose order differs between score vectors a and b."""
    flips = total = 0
    for u in np.unique(users):
        m = users == u
        if m.sum() < 2: continue
        da = np.sign(a[m][:, None]-a[m][None, :]); db = np.sign(b[m][:, None]-b[m][None, :]); iu = np.triu_indices(m.sum(), 1)
        flips += int((da[iu] != db[iu]).sum()); total += len(iu[0])
    return flips/total if total else float('nan')

def offline_reference():
    """Offline scores and intermediates for 2,000 sampled rows, computed in a torch-only process (torch ops and FAISS's
    OpenMP runtime segfault together on macOS, so the comparison runs in two processes)."""
    import torch; import k27_adapt as ka
    sel = json.loads((ka.RUN/'selection_final.json').read_text()); lr = torch.load(ART/f"{sel['LR-L2']['name']}.pt", weights_only=False)
    ials = [torch.load(ART/'ials'/f"{sel['iALS_tuned']['name'].removeprefix('iALS_').rsplit('_s', 1)[0]}_s0.pt", weights_only=False)]; hyb = json.loads((ka.RUN/'hybrid.json').read_text())
    A = joblib.load(ART/'adapt/adapted_models.joblib'); w, b = A['final']['M3_L2_iALS_refit']['w'], A['final']['M3_L2_iALS_refit']['b']
    e = pd.read_pickle(ART/'eval/rows.pkl'); se = pd.read_pickle(ART/'eval/history.pkl')
    m = (e.group.eq('tune') & e.user_has_long_view & e.scope_A).to_numpy(); idx = np.sort(np.random.default_rng(7).choice(np.flatnonzero(m), 2000, replace=False))
    f, s = e.iloc[idx].reset_index(drop=True), se.iloc[idx].reset_index(drop=True); inp = ka.inputs(f, s, lr, ials, hyb)
    joblib.dump({'f': f, 's': s, 'X': inp['X'].astype(np.float32), 'z': inp['z_i'][0], 'flag': inp['flag'], 'offline': ka.design('M3_L2_iALS_refit', inp, 0)@w+b}, REF)

def main(bundle=ROOT/'artifacts/serving_bundle/v1'):
    if not REF.exists(): subprocess.run([sys.executable, '-X', 'faulthandler', __file__, '--offline'], check=True, cwd=ROOT)
    ref = joblib.load(REF); f, s, offline = ref['f'], ref['s'], ref['offline']
    from model import Recommender
    rec = Recommender(bundle); out = {'bundle_version': rec.manifest['bundle_version'], 'load_seconds': rec.load_seconds, 'reference': 'offline scores computed in a separate torch process (k27_adapt.inputs / design), 2,000 tune-group week-1 rows, seed 7'}
    # 1. equivalence with the logged duration
    srv, feats = rec.score_pairs(f.user_id.to_numpy(), f.video_id.to_numpy(), f.t.to_numpy(), duration_ms=f.duration_ms.to_numpy())
    st = feats['stats']; stats_ok = {c: bool(np.allclose(st[c].to_numpy(float), s[c].to_numpy(float), rtol=1e-12, atol=1e-9, equal_nan=True)) for c in STAT_COLS}
    fr = feats['frame']; attr_ok = {'author_id': bool((fr.author_id.to_numpy() == f.author_id.to_numpy()).all()), 'tag1': bool((fr.tag1.astype(str).to_numpy() == f.tag1.astype(str).to_numpy()).all()),
                                   'upload_day': bool(((fr.upload_day == f.upload_day) | (fr.upload_day.isna() & f.upload_day.isna())).all()),
                                   **{c: bool((fr[c].astype(str).to_numpy() == f[c].astype(str).to_numpy()).all()) for c in ['video_type', 'upload_type', 'music_type']}}
    out['1_score_equivalence'] = {'rows': int(len(f)), 'max_abs_score_diff': float(np.abs(srv-offline).max()), 'design_identical': bool(np.array_equal(feats['X'], ref['X'])),
                                  'ials_z_max_abs_diff': float(np.abs(feats['z_ials']-ref['z']).max()), 'flag_identical': bool(np.array_equal(feats['has_factors'], ref['flag'])),
                                  'statistics_identical': stats_ok, 'attributes_identical': attr_ok}
    out['1_score_equivalence']['tolerance'] = '1e-6 absolute: the only difference is the float32 summation order of the iALS dot product (torch vs numpy); design matrix, flags, statistics and attributes must be identical'
    out['1_score_equivalence']['passed'] = out['1_score_equivalence']['max_abs_score_diff'] < 1e-6 and all(stats_ok.values()) and all(attr_ok.values()) and out['1_score_equivalence']['design_identical']
    # 2. video-table duration instead of the logged duration
    srv_meta, feats_meta = rec.score_pairs(f.user_id.to_numpy(), f.video_id.to_numpy(), f.t.to_numpy())
    d_meta = feats_meta['frame'].duration_ms.to_numpy(); d_log = f.duration_ms.to_numpy(float); raw_meta = rec.store.duration[rec.store.video_positions(f.video_id.to_numpy())]
    diff = d_meta-d_log; users = f.user_id.to_numpy()
    out['2_duration_source'] = {'rows': int(len(f)), 'table_missing_share': float((raw_meta < 0).mean()), 'equal_share': float((d_meta == d_log).mean()), 'within_1s_share': float((np.abs(diff) <= 1000).mean()),
                                'abs_diff_ms_quantiles_50_90_99_max': [float(x) for x in np.quantile(np.abs(diff), [.5, .9, .99, 1.])], 'ratio_median_table_over_logged': float(np.median(d_meta[d_log > 0]/d_log[d_log > 0])),
                                'score_abs_diff_quantiles_50_90_99_max': [float(x) for x in np.quantile(np.abs(srv_meta-offline), [.5, .9, .99, 1.])],
                                'within_user_pair_order_flips_share': pair_flips(users, offline, srv_meta), 'reading': 'inputs differ only where the video table and the logged duration disagree; outputs are not required to match'}
    # 3. behaviour
    cnt = np.diff(rec.seen_offsets); heavy = int(rec.seen_users[np.argmax(cnt)]); r = rec.recommend(heavy, index='flat'); seen = rec.seen_set(heavy)
    beh = {'heavy_user': {'user_id': heavy, 'seen': int(cnt.max()), 'candidates': r['candidates'], 'fetched': r['fetched'], 'rounds': r['rounds'], 'none_seen_in_candidates': bool(not np.isin([i['video_id'] for i in r['items']], seen).any()), 'top_k': len(r['items'])}}
    unknown = int(rec.store.ials_users.max())+1; r = rec.recommend(unknown); beh['unknown_user'] = {'source': r['source'], 'candidates': r['candidates'], 'all_scores_finite': bool(np.isfinite([i['score'] for i in r['items']]).all()), 'user_known': r['user_known']}
    sc, fe = rec.score(unknown, rec.pop_ids[:5]); beh['unknown_user']['has_factors_all_zero'] = bool((fe['has_factors'] == 0).all())
    zero = rec.store.ials_users[~rec.store.user_valid]; beh['zero_factor_users_in_table'] = int(len(zero))
    if len(zero): r = rec.recommend(int(zero[0])); beh['zero_factor_user'] = {'user_id': int(zero[0]), 'source': r['source'], 'candidates': r['candidates']}
    r = rec.recommend(heavy, index='hnsw', ef_search=128); beh['hnsw_path'] = {'candidates': r['candidates'], 'source': r['source'], 'rounds': r['rounds']}
    try: rec.recommend(heavy, as_of_ms=rec.T-1); beh['as_of_before_cutoff_rejected'] = False
    except ValueError: beh['as_of_before_cutoff_rejected'] = True
    try: rec.score(heavy, np.array([-7])); beh['unknown_video_rejected'] = False
    except KeyError: beh['unknown_video_rejected'] = True
    beh['passed'] = beh['heavy_user']['candidates'] == 200 and beh['heavy_user']['none_seen_in_candidates'] and beh['unknown_user']['source'] == 'time_popularity' and beh['unknown_user']['has_factors_all_zero'] and beh['as_of_before_cutoff_rejected'] and beh['unknown_video_rejected'] and beh['hnsw_path']['candidates'] == 200
    out['3_behaviour'] = beh
    # 4. retrieval agreement
    valid = np.flatnonzero(rec.store.user_valid); users_idx = np.random.default_rng(11).choice(valid, 1000, replace=False); X = rec.store.X[users_idx]
    t0 = time.time(); exact = np.stack([rec.flat.search(X[i:i+1], 200)[1][0] for i in range(len(X))]); t_flat = (time.time()-t0)/len(X)
    agree = {}
    for ef in [32, 128, 512]:
        rec.hnsw.hnsw.efSearch = ef; t0 = time.time(); got = np.stack([rec.hnsw.search(X[i:i+1], 200)[1][0] for i in range(len(X))]); t_h = (time.time()-t0)/len(X)
        ov = np.array([len(np.intersect1d(exact[i], got[i])) for i in range(len(X))])/200
        agree[f'ef_{ef}'] = {'top200_agreement_mean': float(ov.mean()), 'top200_agreement_p10': float(np.quantile(ov, .1)), 'latency_ms_per_query': t_h*1000}
    out['4_retrieval_agreement'] = {'users': int(len(X)), 'exact_flat_latency_ms_per_query': t_flat*1000, 'hnsw': agree, 'threads': rec.threads, 'note': 'single-query searches in a loop; inner product; no normalisation'}
    out['all_passed'] = out['1_score_equivalence']['passed'] and beh['passed']
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'correctness.json').write_text(json.dumps(out, indent=2, default=float)); print(json.dumps(out, indent=2, default=float))

if __name__ == '__main__':
    offline_reference() if sys.argv[1:] == ['--offline'] else main()
