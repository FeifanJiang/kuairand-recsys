# Reproduction

All commands are run from the repository root with the project virtual environment; scripts resolve data paths from
the repository root and import each other from `scripts/`. Scripts write to `reports/` and `artifacts/` (both ignored
in git); the published reference outputs live in `results/`, so a fresh run never collides with them. The runner
(`k27_run_all.py`) skips a job whose result JSON already exists in `reports/expansion/run_dev/`, and every one-shot
evaluation refuses to run when its output already exists; delete the corresponding file under `reports/` to rerun.
Costs are for an 8-core, 16 GB laptop (CPU only). The runner's swap and memory-pressure monitoring uses macOS commands
(`vm_stat`, `sysctl`, `/usr/bin/time -l`); on Linux it falls back to GNU `time -v` for peak memory and disables the
swap/pressure rules. Floating-point results are reproducible on the same
machine; across machines expect differences at the 1e-4 level in per-user AUC (float32 L-BFGS).

## 1. Environment

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt     # torch 2.14, pandas 3.0, scikit-learn 1.9
.venv/bin/pip install -r requirements-dev.txt                                   # matplotlib, only for scripts/make_figures.py
alias py=.venv/bin/python
```

## 2. Data

Download the three archives from https://kuairand.com or Zenodo record 10439422 and verify them (`md5sum` on Linux):

```sh
mkdir -p data/raw/pure_download data/raw/k27_download
md5 -q data/raw/KuaiRand-1K.tar.gz                   # 6b0b9c8222d67fcd4c676218edca3f1f
md5 -q data/raw/pure_download/KuaiRand-Pure.tar.gz   # 0820331067a3784d9691136f772b35a7
md5 -q data/raw/k27_download/KuaiRand-27K.tar.gz     # 3e3c799a24e2d23a4d2c757fbf9adf59
tar -xzf data/raw/KuaiRand-1K.tar.gz -C data/raw                                  # -> data/raw/KuaiRand-1K/data/*.csv
tar -xzf data/raw/pure_download/KuaiRand-Pure.tar.gz -C data/raw/pure_download    # -> data/raw/pure_download/KuaiRand-Pure/data/*.csv
```

The 27K archive is never extracted (it is streamed). `data/splits/` (tracked) holds the fixed user partitions and their
metadata; everything else under `data/` is generated below.

## 3. Stage 1: KuaiRand-1K (data preparation and retrieval)

```sh
py scripts/prepare_data.py            # audit, cleaning, temporal splits -> data/processed/, reports/*.csv, reports/audit_summary.json
py scripts/validate_outputs.py        # -> reports/output_validation.json
py scripts/run_mf_grid.py                                       # binary iALS / BPR grid -> reports/mf/grid_*/, artifacts/mf/
py scripts/ials_confidence.py                                   # confidence schemes -> reports/ials_confidence/
py scripts/ials_time_followup.py                                # time-decay iALS selection, seen filter -> reports/ials_time_followup/, artifacts/ials_time_followup/
py scripts/time_popularity_baseline.py                          # reads the frozen iALS fits above -> reports/time_popularity/
py scripts/run_seq_experiment.py --out reports/seq_retrieval/runs_20261020 && py scripts/summarize_seq_experiment.py --runs reports/seq_retrieval/runs_20261020
py scripts/run_seq_regularization.py --out reports/seq_retrieval/regularization/runs && py scripts/summarize_seq_regularization.py
py scripts/prepare_test_phase.py && py scripts/final_test_v2.py --phase validation && py scripts/final_test_v2.py --phase test   # held-out test period
py scripts/feature_model.py --out reports/feature_model/run_20261021                                                   # 1K feature models (L1, L2)
```

`ials_time_followup.py` also scores a latent space model in its last comparison block. That block needs
`py scripts/train_lsm.py --variant degree` (or `full`) runs and the grid selection file `reports/lsm/latest_grid.txt`, whose
grid scripts are not part of this repository; without them the block fails and only the iALS stages are reproduced.
Checks for this stage: `feature_model_checks.py` (no data), `seq_retrieval_checks.py`, `seq_regularization_checks.py`.

## 4. Stage 2: KuaiRand-27K (scoring randomly exposed videos)

```sh
py scripts/make_id_maps.py                   # identity 1K -> Pure id maps, asserted (data/raw/pure_download/*_map_1k_to_pure.csv)
py scripts/k27_mapping_check.py              # 27K user table and random log are byte-identical to Pure
py scripts/split_expansion_users.py && py scripts/split_development_users.py        # already tracked in data/splits/; rerunning reproduces them
py scripts/build_event_store.py --tar data/raw/pure_download/KuaiRand-Pure.tar.gz --members 'log_standard_.*\.csv$' --out data/k27/pure_store     # 1 min, for the store check
py scripts/build_event_store.py --tar data/raw/k27_download/KuaiRand-27K.tar.gz --members 'log_standard_.*\.csv$' --out data/k27/standard_to_0421_clean   # 10 min streaming, 3.4 GB
py scripts/build_video_meta.py --tar data/raw/k27_download/KuaiRand-27K.tar.gz --members 'video_features_basic_27k\.csv$' --out data/k27/video_meta        # 2 min
py scripts/k27_store_checks.py --pure-store data/k27/pure_store
py scripts/k27_features.py --max-rows 8000000 --out data/k27/features_dev_8m                                 # development users, 21 min, 9 GB peak
py scripts/k27_features.py --users all --seed 20261024 --max-rows 8000000 --out data/k27/features_all_8m     # all users (deep stage), 27 min, 9.5 GB
py scripts/k27_feature_spotcheck.py --features data/k27/features_dev_8m
py scripts/chunked_features_checks.py && py scripts/feature_matrix_checks.py && py scripts/k27_pipeline_checks_1k.py
py scripts/k27_ials.py --build-cache                                                                          # long-view edges, 80 s, 8 GB
py scripts/k27_run_all.py                                        # eval rows, LR x 2, 16 tree configs, 18 iALS configs; serial, ~8 h
py scripts/k27_select.py initial && py scripts/k27_run_all.py --jobs-file reports/expansion/run_dev/jobs_ials_extension.json
py scripts/k27_select.py final   && py scripts/k27_run_all.py --jobs-file reports/expansion/run_dev/jobs_seeds.json
py scripts/k27_experiment.py final_dry_run_tune && py scripts/k27_experiment.py final     # report group -> reports/expansion/run_dev/final.json
py scripts/k27_hybrid.py                                                                   # one-parameter blend -> hybrid.json
```

The runner (`k27_run_all.py`) runs one job per process, records wall time, peak memory and swap-outs in
`reports/expansion/run_dev/resource_log.jsonl`, and pauses on failure, non-finite values, sustained critical memory
pressure or a job slower than twice its baseline; rerunning resumes from the first unfinished job.

### Deep ranker (protocol_deep.json)

```sh
py scripts/k27_ials.py --tau 2 --alpha 30 --l2 1000 --seed 0 --before-date 20220414 --out artifacts/k27_experiment/ials   # early iALS, 6 min
py scripts/k27_deep.py cache && py scripts/k27_deep.py cache-ials && py scripts/k27_deep_checks.py
py scripts/k27_select.py deep-grid && py scripts/k27_run_all.py --jobs-file reports/expansion/run_dev/jobs_deep_grid.json   # ~45 min
py scripts/k27_select.py deep      && py scripts/k27_run_all.py --jobs-file reports/expansion/run_dev/jobs_deep_seeds.json  # ~10 min
py scripts/k27_deep_report.py --group tune && py scripts/k27_deep_report.py --group report                                 # -> deep_report.json
```

### Exposure diagnostic and random-label adaptation (protocol_adaptation.json)

```sh
py scripts/k27_week1_standard.py && py scripts/k27_exposure_diagnostic.py      # -> reports/expansion/exposure_diagnostic.json
py scripts/k27_adapt.py rows && py scripts/k27_adapt.py fit && py scripts/k27_adapt.py test   # week-2 test -> adapt_week2.json
```

### Confirmation on held-out users (protocol_confirmation.json)

```sh
py scripts/k27_confirm_prepare.py --partition development && py scripts/k27_confirm_prepare.py --partition confirmation
py scripts/k27_confirm.py --partition development --window week1 && py scripts/k27_confirm.py --partition development --window week2   # dry runs; compare with results/expansion/run_confirmation/development_dry_run*.json
py scripts/k27_confirm.py --partition confirmation --window week2 && py scripts/k27_confirm.py --partition confirmation --window week1
py scripts/make_figures.py                                                                                                     # docs/figures/
```

### Label-source contrast and collaborative adaptation variants (protocol_extension_source.json, protocol_extension_cf.json)

```sh
py scripts/k27_source_contrast.py extract && py scripts/k27_source_contrast.py fit   # same-week recommended vs random training labels -> run_dev/ext_source.json
py scripts/k27_adapt_ext.py                                                          # re-weighted iALS dimensions (C1), M3 + those interactions (C2) -> run_dev/ext_cf.json
```

## 5. Serving demo (serving/README.md)

```sh
.venv/bin/pip install -r requirements-serving.txt
py serving/build_bundle.py --version v1                       # duration table, lookup indexes, seen sets, popularity list, FAISS indexes, manifest (~10 min, 9 GB peak)
py serving/check_correctness.py                               # service == offline scores; duration-source report; behaviour tests; HNSW vs exact agreement
py serving/benchmark.py                                       # starts uvicorn, measures latency/throughput/peak memory -> reports/expansion/serving/benchmark.json
.venv/bin/uvicorn serving.app:app --host 127.0.0.1 --port 8000 --workers 1      # then GET /recommend?user_id=...&index=hnsw&ef_search=128
```

On macOS the serving scripts set `KMP_DUPLICATE_LIB_OK=TRUE` because torch and faiss-cpu each ship an OpenMP runtime.

## 6. What is tracked and what is generated

Tracked: code, documentation, the figure, `data/splits/`, and under `results/` the published reference outputs
(protocols, selection records, job lists, resource log, result JSON/CSV tables, check results). Generated locally and
ignored: `reports/` (what the scripts write), everything else under `data/`, and `artifacts/`. Not published even from
the reference runs: per-run logs and terminal output, code snapshots inside run folders, per-user tables, score dumps
(`.npz`, `.csv.gz`), superseded protocol drafts and the first development dry run.
