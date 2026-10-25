# Method

Data, evaluation protocol, models, findings with pointers to the result files, and engineering checks. Numbers are from
`results/expansion/run_confirmation/` (confirmation users) unless a development file is named.

## Data

KuaiRand (https://kuairand.com; Gao et al., CIKM 2022) has three releases. The retrieval stage uses KuaiRand-1K; the
scoring stage uses KuaiRand-27K, with KuaiRand-Pure for ID verification. The archives are also on Zenodo record 10439422:

| Archive | MD5 | Size |
|---|---|---|
| KuaiRand-1K.tar.gz | `6b0b9c8222d67fcd4c676218edca3f1f` | 1.1 GB |
| KuaiRand-Pure.tar.gz | `0820331067a3784d9691136f772b35a7` | 47 MB |
| KuaiRand-27K.tar.gz | `3e3c799a24e2d23a4d2c757fbf9adf59` | 9.9 GB |

The 27K archive is read by streaming and never extracted. User ids and random-pool video ids are identical across
releases (`results/data_feasibility`, `results/expansion/k27_mapping_check.json`). After cleaning (invalid values
quarantined, full-record duplicates removed): 135,418,120 pre-cutoff logged exposures from 26,904 users on 15,298,279
videos, 35.8M long views; 1.14M random exposures of the 26,285 users outside the 1K release.

## Evaluation protocol

- **Temporal cutoff.** Training history ends 2022-04-21 (Asia/Shanghai date of `time_ms`); evaluation windows are
  04-22..04-28 (week 1) and 04-29..05-08 (week 2). Every training row's statistics use only events strictly earlier
  than the row (same-timestamp events excluded); evaluation features are frozen at the cutoff.
- **Target and metric.** KuaiRand logs items shown by a random policy to the same users. Scoring them with per-user AUC
  (ties counted half) measures within-user ranking free of the production policy's selection. Main population: users
  with at least one pre-cutoff long view and both outcomes in the window; scope A: videos with at least one pre-cutoff
  exposure. Scope B (all videos), cold-start users and pooled AUC by video degree are in the same JSON files.
- **User tiers.** The 26,285 users outside the 1K release were split 50/50 by seed into development and confirmation
  users before modelling; development users are split again into tune (hyperparameter selection) and report
  (development results). The 13,143 confirmation users were evaluated on week 2 (primary) and week 1 with frozen models,
  after a dry run of the same script on the development users reproduced the development results.
- **Protocols.** Each stage has a JSON protocol fixed before running (`results/expansion/protocol_*.json`): grids,
  selection and tie rules, the decision rule (+0.01 per-user AUC with a 95% interval excluding 0), expected outcomes
  and resource rules.
- **Statistics.** Stochastic models are evaluated per seed and averaged per user (a repeated-run statistic, not an
  ensemble); every contrast uses a paired user bootstrap (2,000 resamples, seed 2026) on identical users and exposures.

## Models

- **Features** (`scripts/feature_model.py`, `chunked_features.py`): prequential counts, long-view rates and recencies
  for video, author, tag, user, user × author and user × tag, plus video attributes and duration. L1 = non-personalized
  subset; L2 = with the user-interaction features. 6M training rows for the development models, 8M for the deep stage.
- **Logistic regression** (L-BFGS on float32 standardized features, λ grid) and **HistGradientBoosting** (8
  configurations, internal early stopping) on the same features.
- **DLRM-style ranker** (`k27_deep.py`): user / video / author ID embeddings with a Gaussian prior normalized by expected
  touches, concatenated with the features; 8 configurations, early stopping on a training holdout, CPU.
- **iALS** (`ials_confidence.py`, `k27_ials.py`): exact ALS solves on long-view edges with time-decayed confidence
  (τ, α, λ selected on tune users; 5 seeds). The reference configuration is the one selected for retrieval on
  KuaiRand-1K; the tuned configuration (τ 2, α 30, λ 1000) was selected on random exposures.
- **Blend** (`k27_hybrid.py`): one weight between the standardized LR-L2 logit and iALS score.
- **Adaptation on random-exposure labels** (`k27_adapt.py`): four ridge-logistic forms fitted on development users'
  week-1 random exposures (98,021 rows), selected on week 1, refit on all week-1 rows and tested on week 2 — M1: blend
  weight only; M2: the LR-L2 features; M3: features + iALS score; M4: M3 shrunk toward the logged-exposure model.

## Findings

1. **Tuned iALS beats the rankers trained on logged exposures.** Confirmation week 2: LR-L2 − iALS −0.016
   [−0.021, −0.011]; Tree-L2 −0.009 and Deep-L2 −0.013 relative to LR-L2 (the deep ranker overfits after one or two
   epochs on 8M rows; `run_dev/deep_report.json`). Personalized features add +0.043 over non-personalized ones.
2. **Fusion trained on logged exposures fails, and the exposure mechanism explains it.** Feeding the iALS score into
   the feature model with a temporal holdout gives it a coefficient near zero, linear or deep
   (`deep_fusion_diagnostics.json`). For the same development users, week and random-pool videos, the iALS score's
   per-user AUC is 0.60 under recommended exposure and 0.65 under random exposure, while LR-L2's is unchanged
   (`exposure_diagnostic.json`).
3. **Re-fitting on random-exposure labels.** M2 gains +0.042 over LR-L2 (same features, random instead of logged
   labels); the iALS input adds +0.003 (M3); shrinking toward the logged model adds nothing (M4); the blend weight alone
   gives +0.010 (M1). M3 beats the tuned iALS by +0.030 [+0.025, +0.034] on confirmation week 2 and +0.032 on week 1,
   matching the development result (+0.030, `run_dev/adapt_week2.json`).
4. **Label source, not recency** (`run_dev/ext_source.json`). Same development users, same week, rows matched on
   (user, date), same frozen features and procedure: training on week-1 random exposures beats training on week-1
   recommended exposures by +0.058 [+0.054, +0.062] (+0.046 restricted to the random-pool videos); the week-1 logged
   re-fit scores 0.013 below the 6M-row logged-exposure model.
5. **Collaborative adaptation variants** (`run_dev/ext_cf.json`). Re-weighting the 32 iALS latent dimensions on the
   same random labels scores −0.007 against the plain iALS score; adding those interactions to M3 gains +0.002.
6. **Retrieval on KuaiRand-1K** (`results/test_phase_v2/`, `results/seq_retrieval/`). Time-decayed-confidence iALS:
   Recall@100 1.79% vs 1.06% for time-decayed popularity on the held-out test period, ahead of binary iALS, BPR, a
   latent-space model and SASRec.

## Engineering checks

The chunked feature pipeline, the float32 matrix builder, the tar-streaming store, the video metadata store, the
deep-ranker inputs and the confirmation path each have a check script that reproduces a reference implementation
(`scripts/*_checks.py`, results in `results/expansion/*checks*.json`); `feature_model_checks.py` runs on a synthetic
table without data. The serving layer is checked against the offline scoring function (`serving/check_correctness.py`).

## Reproduce

Commands in order, with costs, are in [reproduction.md](reproduction.md); scripts are described in
[scripts/README.md](../scripts/README.md) and outputs in [results/README.md](../results/README.md).
