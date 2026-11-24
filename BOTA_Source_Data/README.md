# BOTA source data

Numerical data for Appendices D–G of *BOTA: Offline Trajectory Amortization for Short-Window User Deletion in Language Model Recommenders*. All CSV files use UTF-8 and a header row. Each link below opens a table and gives its unit of observation.

## Appendix D · MovieLens-1M FinalTest

| File | Rows | What a row represents |
|---|---:|---|
| [Binary predictions](01_movielens_finaltest/finaltest_binary.csv) | 72 | One of 12 methods for a request and seed, with supported users, binary fidelity, utility and residual intervals |
| [Ranking](01_movielens_finaltest/finaltest_ranking.csv) | 216 | One of three user groups per ranking endpoint, with user count, Recall and NDCG at 5, 10 and 20 |
| [Request composition](01_movielens_finaltest/request_composition.csv) | 2 | One frozen request, with frequency-group counts, exposure, training-frequency and Development-support ranges, and the fraction of deleted slots |

The two evaluation tasks cover 144 endpoints across requests L8 and L4M4 and seeds 41, 42 and 43. The ranking table has three rows per endpoint for all, requested and retained users. `endpoint_id` identifies the seed, request, method and evaluation task. BOTA0 is the source-only ablation, BOTA1 is the unit-response implementation control and BOTA64 is the calibrated response.

`D_U` is the mean absolute difference between Original and Exact-Masked prediction probabilities on the supported request examples. `E_U` is the same error for the evaluated method against Exact-Masked. Both use raw probability units. Multiply by 1,000 where the manuscript labels a column in thousandths. `local_residual` equals `E_U / D_U`. The binary table gives the 95% limits for this residual in `local_ci_lower` and `local_ci_upper`. The `p5_ci_*` columns give the limits for the error against Retain-Retrain-P5, normalized by the error of FullControl-P5 against Retain-Retrain-P5. Both intervals use 2,000 bootstrap draws of supported users with replacement. Each draw retains all examples for each sampled user and recomputes the ratio. Ranking rows report point estimates.

L8 and L4M4 share four users and cover twelve distinct users in total. The request-composition table gives aggregate statistics for each request. Each selected user occurred once in the 3,200-slot window and had at least ten Development examples. In the frozen selection rule, training-frequency ties use SHA-256 of the UTF-8 string `p3c1-tercile:{user}`. Within each eligible frequency group, candidates are ordered by the hexadecimal SHA-256 of `bota-short-pool:42:{role}:{user}`. L8 uses the first eight low-frequency users. L4M4 uses the first four low- and first four middle-frequency users.

## Appendix E · MovieLens-1M calibration

| File | Rows | What a row represents |
|---|---:|---|
| [Baseline search](02_movielens_calibration/baseline_search.csv) | 130 | A seed-42 candidate evaluated on L8 or L4M4, with hyperparameters, per-request utility, joint eligibility and selection |
| [Historical Fisher search](02_movielens_calibration/fisher_historical_search.csv) | 82 | A candidate run, a repeated control or a cross-seed check, identified by stage, phase and record type |
| [Fisher decisions by stage](02_movielens_calibration/fisher_stage_selections.csv) | 37 | A strength considered at a stage decision, with its mean error, eligibility and selection flag |
| [Frozen Dev-Select scores](02_movielens_calibration/fisher_frozen_dev_select.csv) | 12 | One of the two frozen strengths, 0 or 64, evaluated for a request and seed under the final scoring protocol |

The six-method baseline grid contains 65 configurations across two requests. Read `utility_eligible` for the request-level check and `candidate_eligible` for the joint decision. In the Fisher search, 30 grid rows cover 15 strengths on two requests, 16 rows record repeated controls and 36 record cross-seed checks. A run carried forward to a later decision appears once in the historical scores and may enter multiple stage decisions. Stage 14 selected 64. Use the 12 frozen scores to reproduce the appendix's final Dev-Select comparison. The historical search used the scoring protocol in effect at selection time.

## Appendix F · Sequential deletion and membership-inference diagnostics

| File | Rows | What a row represents |
|---|---:|---|
| [Sequential subsets](03_movielens_stress/sequential_subsets.csv) | 60 | A cumulative deletion subset of size 1, 2, 4 or 8 for a seed, request and ordering |
| [Attack scores](03_movielens_stress/membership_inference_scores.csv) | 180 | An attack score for one request group and model endpoint, including pooled AUC, macro-user AUC, membership advantage and their intervals |
| [Paired diagnostics](03_movielens_stress/membership_inference_paired.csv) | 15 | A primary-score paired comparison for a request group, with reference separation, gap improvement, intervals and interpretation flags |

Subset codes restart for each seed. Group by both `seed` and `subset_within_seed` to reproduce the count of 39 distinct evaluations. Keep all 60 rows to reproduce the order-specific summaries. The attack files cover six primary and nine stronger-request groups. Three model endpoints were scored with negative answer loss, confidence, negative entropy and absolute yes/no margin. Intervals use 2,000 user-level bootstrap resamples. The paired diagnostics use negative answer loss.

## Appendix G · Amazon Movies and TV

| File | Rows | What a row represents |
|---|---:|---|
| [Baseline calibration](04_amazon/baseline_calibration.csv) | 130 | One of 65 seed-42 candidates on K2 or K4, with request-level scores and the joint selection decision |
| [Fisher calibration](04_amazon/fisher_calibration.csv) | 40 | A strength evaluated on K2 or K4 during the seed-42 coarse/fine search or the seed-41/43 checks |
| [A1–A3 comparison](04_amazon/comparison.csv) | 60 | One of ten methods for a request and seed on the reused Development cohort, including all endpoint statuses |

For baseline calibration, `candidate_use` distinguishes a selected comparator, the diagnostic configuration for a method without an eligible candidate, and an unselected configuration. Hyperparameters are encoded as JSON within one column. Seed 42 selected Fisher strength 48. The historical cross-seed confirmation passed in seed 43 and failed in seed 41. Seeds 42 and 43 used the frozen setting for the separate A1–A3 comparison. Of the 60 comparison rows, 52 have scores, two IFRU rows record numerical failure and six SISA rows record unavailable weights. The latter eight retain their status and have blank score fields.

## Related materials

The [BOTA code and original result summaries](https://github.com/fuys24666/BOTA/tree/4aeae6f6a3e1e3a332d2e5f7575042dc381ed989) include the serving-latency table and code reproduction instructions. The frozen MovieLens Dev-Select and Amazon comparison CSVs in this package reproduce the corresponding public summaries. All scores here are archived experimental measurements. The tables report aggregate outputs and use study-level identifiers for methods, configurations, requests and subsets.
