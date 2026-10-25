# Serving demo

A two-stage recommendation service on the frozen study artifacts (seed 0, nothing retrained):

```
user_id, as_of  ->  iALS inner-product retrieval (FAISS)  ->  seen filter + backfill to 200 candidates
                ->  build_features(user_id, video_ids, as_of)  ->  M3 scoring  ->  top 20
```

## Design

| Item | Decision |
|---|---|
| Retrieval | tuned iALS factors (τ 2, α 30, λ 1000, seed 0); inner product, no normalisation; `IndexFlatIP` (exact, default) and `IndexHNSWFlat` (M 32, efConstruction 200, efSearch from the request) over 4.49M item vectors |
| Candidates | 200 after removing the user's pre-cutoff long views; the fetch size grows (k + seen + 50, then doubling) until 200 remain or the catalog is exhausted |
| Unknown user | not in the user table, or no pre-cutoff long view: candidates come from the time-decayed long-view popularity list at the cutoff (τ 0.5 d); iALS inputs are 0 with `has_factors = 0`; user statistics are zero counts |
| Features | `build_features(user_id, video_ids, as_of)`: video attributes and duration from the video table, frozen-at-cutoff statistics (video, author, tag, user, user × author, user × tag) through sorted-key lookups with the same semantics as `feature_model.query`, then `feature_model.assemble` and the LR-L2 standardisation |
| `as_of` | request parameter (ms); default 2022-04-22 00:00 Asia/Shanghai; requests before the cutoff are rejected (400); video age, hours since last long view and decayed counts are aged to `as_of` |
| Bundle | `artifacts/serving_bundle/<version>/` with `manifest.json`: version, cutoff, default `as_of`, SHA-256 and size of every component, hashes of the source artifacts, build timings, library versions; loaded once at startup |

## API

| Endpoint | Parameters |
|---|---|
| `GET /recommend` | `user_id`, `as_of`, `k=20`, `n_candidates=200`, `index=flat\|hnsw`, `ef_search=128` |
| `POST /score` | `{user_id, video_ids, as_of}` |
| `GET /health`, `GET /manifest` | — |

## Run

```sh
.venv/bin/pip install -r requirements-serving.txt
.venv/bin/python serving/build_bundle.py --version v1   # ~10 min, 9 GB peak; --resume keeps finished components
.venv/bin/python serving/check_correctness.py           # -> reports/expansion/serving/correctness.json
.venv/bin/python serving/benchmark.py                   # -> reports/expansion/serving/benchmark.json
KUAIRAND_BUNDLE=artifacts/serving_bundle/v1 .venv/bin/uvicorn serving.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Files: `build_bundle.py`, `features.py` (feature lookup and assembly), `model.py` (recommender), `app.py` (FastAPI),
`check_correctness.py`, `benchmark.py`. On macOS the scripts set `KMP_DUPLICATE_LIB_OK=TRUE` because torch and faiss-cpu
each ship an OpenMP runtime.

## Performance (`results/expansion/serving/benchmark.json`)

Apple M1, 8 cores, 16 GB; one uvicorn worker, 8 FAISS threads; client on the same machine; warm process; random known
users, default `as_of`, k = 20, 200 candidates; no per-request caching. Bundle: 14 components, 5.6 GB; index build 0.4 s
(exact) and 93 s (HNSW); server load 4.8 s; peak RSS 4.6 GB.

| Index | Concurrency | Requests | p50 ms | p95 ms | req/s | Retrieval p50 ms | Features + scoring p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact | 1 | 200 | 57 | 64 | 17.2 | 11.6 | 43 |
| exact | 4 | 400 | 164 | 189 | 23.8 | 23.5 | 122 |
| exact | 16 | 800 | 620 | 880 | 25.3 | 95.5 | 358 |
| HNSW (efSearch 128) | 1 | 200 | 44 | 47 | 22.6 | 0.8 | 41 |
| HNSW (efSearch 128) | 4 | 400 | 118 | 149 | 33.1 | 0.9 | 114 |
| HNSW (efSearch 128) | 16 | 800 | 473 | 705 | 33.3 | 11.4 | 335 |

Retrieval is cheap (exact inner product over 4.49M items: 12 ms; HNSW: under 1 ms); the request time is dominated by
feature assembly and scoring for 200 candidates (pandas-based `feature_model.assemble`, reused unchanged for
exactness). Throughput saturates at 25–33 req/s because the single worker is CPU-bound in Python; more workers or a
vectorised feature assembly are the next steps.

## Correctness (`results/expansion/serving/correctness.json`)

- **Service = offline.** On 2,000 development week-1 random exposures scored with `as_of` = exposure time, the design
  matrix, the has-factors flag, all 22 frozen statistics and the video attributes are identical to the offline
  pipeline; the M3 score differs by at most 7.4e-08 (float32 summation order, torch vs numpy).
- **Duration source.** The video table's `video_duration` equals the logged `duration_ms` on 100% of the rows (2.9% have
  no table value and a logged 0, handled by the `zero_duration` flag); no within-user pair changes order.
- **Behaviour.** Heaviest user (40,420 pre-cutoff long views): 200 candidates after the seen filter in one fetch; unknown
  users and the 81 users without a long view take the popularity path with finite scores; `as_of` before the cutoff
  and unknown video ids are rejected.
- **HNSW vs exact top-200** (1,000 users): efSearch 32 / 128 / 512 recover 65.5% / 69.2% / 72.4% of the exact top-200
  at 0.05 / 0.09 / 0.33 ms per query against 10.4 ms for the exact search. Graph indexes over raw inner products are
  weak for maximum-inner-product search, so the exact index is the default; a norm-padding transform would be the
  standard fix.
