# Manuscript results

These tables contain aggregate measurements for the binary recommendation and serving experiments. To recompute the headline comparisons from a checkout, run the following command from the repository root. It uses only the Python standard library.

```sh
python scripts/release/summarize_manuscript_results.py
```

| CSV | Rows | Unit of observation | Manuscript use |
|---|---:|---|---|
| [movielens_primary.csv](movielens_primary.csv) | 66 | One method, seed and request on MovieLens-1M FinalTest | Primary fidelity and utility, per-seed residuals, source-only ablation and two stopping-rule controls |
| [movielens_fisher.csv](movielens_fisher.csv) | 12 | One Fisher strength, seed and request on MovieLens-1M Dev-Select | Development confirmation of the calibrated response |
| [amazon_comparison.csv](amazon_comparison.csv) | 60 | One method, seed and request on the reused Amazon Movies and TV Development cohort | Auxiliary six-cell comparison, including unsuccessful or unavailable endpoints |
| [serving_latency.csv](serving_latency.csv) | 12 | One seed, request and resident bank size | Serving latency, offline construction cost and break-even request count |

## Accuracy and utility

The first three files share a schema. `D_U` is the mean absolute difference between Original and Exact-Masked prediction probabilities for the supported request examples. `E_U` measures the same error for the given method against Exact-Masked. Both are in **raw probability units**. Multiply MovieLens `E_U` by 1,000 to reproduce the main table's displayed units. `local_residual` is `E_U / D_U`. `toward_exact_rate` is the fraction of supported predictions closer to Exact-Masked than Original is; multiply by 100 for the percentage in the paper.

`local_samples` counts supported request examples. `utility_samples`, `overall_auc`, `overall_acc` and `overall_log_loss` refer to the full binary evaluation cohort. `calibration_status` describes configuration selection, `status` describes scoring, and `utility_eligible` records whether a scored endpoint passed the binary utility criteria.

- **MovieLens-1M.** Seeds 41, 42 and 43 share the same two requests (L8 and L4M4). Take the mean and sample standard deviation across seeds for each method and request. `BOTA64` is the calibrated response and `BOTA0` is source-only. Compute the Fisher error reduction as `1 - mean(E_U calibrated) / mean(E_U source-only)` within a split and request. The FinalTest CSV also contains the binary scores for FullControl-P5 and Retain-Retrain-P5 against Exact-Masked. The separately normalized stopping-rule sensitivity statistic uses a different reference.
- **Amazon.** Seeds 42 and 43 each have three K=2 requests (A1–A3). Within each seed, give each request equal weight. `BOTA` uses strength 48 and `BOTA-source-only` uses 0. The export retains each calibration decision, including scored PCGrad diagnostics without an eligible selected candidate. Blank metrics mark unavailable scores: six SISA weight sets were not retained and two IFRU solves failed. Exclude these from numeric comparisons without counting them as wins.

## Serving measurements

Each setting used 20 warm-ups and 1,000 timed requests. `mean_ms`, `p50_ms` and `p95_ms` measure composition, durable publication and reload validation with the impact bank already in memory. `bank_build_seconds` and `historical_exact_replay_seconds` use seconds. The amortization break-even count rounds up the bank construction time divided by the time saved per request relative to archived exact replay. The timed path excludes cold bank loading, network transfer and bank refresh.

## Provenance

The CSVs preserve metric precision from these files in the frozen experiment workspace. The evaluation exports select binary metrics and omit user identifiers and per-example predictions. The timing export retains one summary per setting.

| Export | Source under the experiment workspace |
|---|---|
| MovieLens FinalTest | `outputs/ali_stage25_historical_mask_audit_v1/summary/FinalTest/metrics.csv` |
| MovieLens Dev-Select | `outputs/ali_stage25_historical_mask_audit_v1/summary/Development/metrics.csv` |
| Amazon Development | `outputs/ali_stage27_amazon_receraser_rescore_v1/summary/amazon_all_methods.csv` (Stage 25 endpoints with the Stage 27 selected RecEraser scores) |
| Serving latency | `outputs/ali_stage17_lambda64_suite_v1/cells/latency_s{seed}/result.json` and the timing samples for each setting |

Acquire [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) and [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) from their providers. To rerun model scoring, see the [reproduction map](../docs/CURRENT_CODE_SNAPSHOT.md) and [external artifact requirements](../docs/ARTIFACTS.md).
