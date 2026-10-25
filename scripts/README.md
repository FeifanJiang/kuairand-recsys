# scripts/

All scripts are run from the repository root as `python scripts/<name>.py ...`; they import each other from this
directory, resolve data paths from the repository root and write to `reports/` and `artifacts/` (the published reference
outputs are under `results/`). Commands in order, with costs, are in
[docs/reproduction.md](../docs/reproduction.md).

| Purpose | Scripts |
|---|---|
| Data streaming and cleaning | `build_event_store.py` (27K logs from the tar, 1K cleaning rules, full-record de-duplication), `build_video_meta.py`, `prepare_data.py` (1K audit, cleaning, temporal splits), `validate_outputs.py`, `prepare_test_phase.py`, `make_id_maps.py`, `k27_week1_standard.py` |
| User partitions | `split_expansion_users.py` (development / confirmation, seed 20261021), `split_development_users.py` (tune / report) |
| Features | `feature_model.py` (feature definitions, prequential statistics, frozen tables, logistic fit), `chunked_features.py` (out-of-core statistics), `feature_matrix.py` (chunked float32 matrices), `k27_features.py` (sample + features for a 27K sample) |
| Models | `ials_confidence.py` (iALS solver and confidence schemes), `k27_ials.py` (27K iALS fits incl. the early model), `k27_experiment.py` (LR, HistGradientBoosting, iALS scoring stages), `k27_deep.py` (DLRM-style ranker), `k27_hybrid.py` (one-parameter blend), `train_mf.py` (binary iALS, BPR), `train_seq.py` / `seq_retrieval.py` (pooled-history and SASRec retrievers), `time_popularity_baseline.py` |
| Random-label adaptation | `k27_adapt.py` (rows / fit / test); extensions `k27_adapt_ext.py` (C1/C2 collaborative adaptation), `k27_source_contrast.py` (same-week recommended vs random training labels) |
| Orchestration and selection | `k27_run_all.py` (serial runner with resource rules), `k27_select.py` (selection from tune-group AUCs: initial / final / deep-grid / deep), `run_mf_grid.py`, `run_seq_experiment.py`, `run_seq_regularization.py`, `ials_time_followup.py` |
| Evaluation and confirmation | `k27_eval.py`, `k27_eval_rows.py`, `train_ranker.py` (per-user AUC), `retrieval_eval.py` (exact top-k retrieval metrics), `evaluate_ranking_sessions.py`, `k27_deep_report.py`, `k27_exposure_diagnostic.py`, `summarize_seq_experiment.py`, `summarize_seq_regularization.py`, `final_test_v2.py` (1K test period), `k27_confirm_prepare.py`, `k27_confirm.py` (held-out 27K users) |
| Implementation checks | `feature_model_checks.py` (synthetic, no data), `chunked_features_checks.py`, `feature_matrix_checks.py`, `k27_store_checks.py`, `k27_feature_spotcheck.py`, `k27_mapping_check.py`, `k27_pipeline_checks_1k.py`, `k27_deep_checks.py`, `seq_retrieval_checks.py`, `seq_regularization_checks.py` |
| Figures | `make_figures.py` (needs `requirements-dev.txt`) |
| Serving demo | `../serving/`: `build_bundle.py`, `features.py`, `model.py`, `app.py`, `check_correctness.py`, `benchmark.py` (needs `requirements-serving.txt`) |
| Latent space model | `train_lsm.py`, `lsm_lr_branches.py` (used in the retrieval comparison of `ials_time_followup.py`) |

One-shot evaluations (`k27_experiment.py final`, `k27_hybrid.py`, `k27_deep_report.py --group report`, `k27_adapt.py test`,
`k27_confirm.py --partition confirmation`, `final_test_v2.py`) refuse to run when their output already exists.
