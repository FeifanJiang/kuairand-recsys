# results/

Reference outputs of the runs reported in the README: result tables, protocols, selection records and checks. The
scripts write to `reports/` (git-ignored) with the same layout, so a fresh run can be compared file by file with these.
Per-run logs, code snapshots, per-user tables and score dumps are not published.

| What | Where |
|---|---|
| **Confirmation on held-out users** (week 2 primary, week 1) | `expansion/run_confirmation/confirmation_week2.json`, `confirmation.json`; development dry runs `development_dry_run*.json` |
| **Random-label adaptation** (selection on week 1, test on week 2) | `expansion/run_dev/selection_adapt.json`, `adapt_week2.json`; protocol `expansion/protocol_adaptation.json` |
| **Label-source contrast and collaborative adaptation variants** (development users, week 2) | `expansion/run_dev/ext_source.json`, `ext_cf.json`; protocols `protocol_extension_*.json` |
| **Serving demo** | `expansion/serving/` (bundle manifest, correctness checks, benchmark) |
| **Exposure-mechanism diagnostic** | `expansion/exposure_diagnostic.json`, `deep_fusion_diagnostics.json` |
| **Baseline comparison on 27K** (LR / trees / tuned iALS / blend, report group) | `expansion/run_dev/final.json`, `hybrid.json`, `selection_final.json`, per-job results `run_dev/*.json`, `resource_log.jsonl`; protocols `protocol_confirmed.json`, `protocol_hybrid.json` |
| **Deep ranker** | `expansion/run_dev/deep_report.json`, `selection_deep.json`; protocol `protocol_deep.json`; checks `deep_checks.json` |
| **Data verification and checks** | `expansion/k27_mapping_check.json`, `k27_store_checks.json`, `*checks*.json`, `k27_feature_spotcheck.json`, `training_size_frozen.json`, `resource_probe*/`; `data_feasibility/` (Pure vs 1K) |
| **1K feature models** | `feature_model/run_20261021/summary.json`, `protocol_confirmed.json` |
| **Retrieval on 1K** | test period `test_phase_v2/results/*/` (`recall.csv`, `comparisons.csv`); time-decay iALS selection `ials_time_followup/`; confidence schemes `ials_confidence/`; sequence retrievers `seq_retrieval/` (`report_recall.csv`, `comparisons.csv`, `experiment_summary.json`, per-run `summary.json` and `curve.csv`); popularity `time_popularity/`; matrix-factorization grid `mf/` |
| **1K data audit** (from `prepare_data.py`) | `audit_summary.json`, `output_validation.json`, `column_profile.csv`, `daily_summary.csv`, `split_summary.csv`, `scenario_summary.csv`, `item_activity_quantiles.csv`, `duration_quantiles.csv`, `file_inventory.csv` |

Evidence labels inside the files: *development* (tune and report groups), *development feedback* (report group viewed
again for a later frozen model), *test window* (development users' week 2) and *confirmation* (held-out users).
