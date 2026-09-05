# Amazon Movies and TV new-user K2 result

This file records the aggregate, non-identifying output used for the paper's
dataset-extension experiment. It is not a replacement for the manifests and
per-sample evidence produced by a local reproduction run.

## Protocol

- Dataset: Amazon Reviews 2023, Movies and TV category.
- Seed: 42.
- Historical cohort: 256 users used to train the P5 recommendation Adapter.
- Adaptation cohort: 768 users disjoint from the historical cohort.
- Adaptation window: 3,200 examples, batch size 16, 200 AdamW steps.
- Request: one low-frequency and one middle-frequency user; two window
  exposures per user and four deleted interactions in total.
- Local evaluation: five Development examples per requested user, ten total.
- Selection used user frequency and deterministic hashes, not labels or model
  predictions.
- Evaluation is Development-only. No FinalTest or MIA data were accessed.
- Original-to-Exact absolute deletion effect: `D_U = 0.004311621189`.

## Aggregate output

| Method | AUC | ACC | LogLoss | MAE to Exact | Residual | Toward/Away |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0.797093 | 0.836719 | 0.370971 | 0.004312 | 1.000000 | 0/0 |
| Exact-Masked | 0.796956 | 0.837500 | 0.370728 | 0.000000 | 0.000000 | 10/0 |
| BOTA | 0.800265 | 0.840885 | 0.366186 | 0.002341 | 0.542999 | 8/2 |
| IFRU | 0.797104 | 0.836719 | 0.370965 | 0.004391 | 1.018470 | 0/10 |
| E2URec | 0.800625 | 0.835938 | 0.392706 | 0.009714 | 2.252940 | 1/9 |
| NegGrad | 0.651330 | 0.683073 | 0.701075 | 0.239738 | 55.602730 | 0/10 |
| PCGrad | 0.535671 | 0.379948 | 2.606139 | 0.299722 | 69.514906 | 1/9 |
| SISA | 0.625099 | 0.833854 | 0.451486 | 0.159386 | 36.966706 | 0/10 |
| RecEraser | 0.807195 | 0.843229 | 0.360221 | 0.013763 | 3.192056 | 0/10 |

The local residual is the method's mean absolute probability error to
Exact-Masked divided by the Original-to-Exact effect. BOTA was the only
approximate method below the Original reference for this request. RecEraser
obtained the strongest aggregate utility, illustrating that utility and local
counterfactual fidelity measure different properties.

The local reproduction command writes its corresponding report to
`outputs/amz_v4/k2_all_evaluation/amz_new_k2_all_eval_s42_v4/report.md`.
