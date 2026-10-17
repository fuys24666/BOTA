# Ali feedback Stage 0 artifact audit

- Preregistration: `ali-feedback-attack-sequential-scale-v1`
- Git HEAD: `530faf082fb4e2cb2679dfb64b0990cad10ac90c`
- Experiment root: `<EXPERIMENT_ROOT>`
- Scope: ML-1M Development only; FinalTest was not traversed.
- Result: **READY** (0 errors, 30 warnings)

## Artifact matrix

| Seed | Method | L8 artifacts | L4M4 artifacts | Status |
|---:|---|---:|---:|---|
| 41 | Original | 2 | 2 | present |
| 41 | Exact-Masked | 2 | 2 | present |
| 41 | BOTA | 2 | 2 | present |
| 41 | IFRU | 2 | 2 | present |
| 41 | E2URec | 2 | 2 | present |
| 41 | NegGrad | 2 | 2 | present |
| 41 | PCGrad | 2 | 2 | present |
| 41 | SISA | 4 | 4 | present |
| 41 | RecEraser | 4 | 4 | present |
| 41 | Retain-Retrain-P5 | 2 | 2 | present |
| 41 | FullControl-P5 | 2 | 2 | present |
| 42 | Original | 2 | 2 | present |
| 42 | Exact-Masked | 2 | 2 | present |
| 42 | BOTA | 2 | 2 | present |
| 42 | IFRU | 2 | 2 | present |
| 42 | E2URec | 2 | 2 | present |
| 42 | NegGrad | 2 | 2 | present |
| 42 | PCGrad | 2 | 2 | present |
| 42 | SISA | 4 | 4 | present |
| 42 | RecEraser | 4 | 4 | present |
| 42 | Retain-Retrain-P5 | 2 | 2 | present |
| 42 | FullControl-P5 | 2 | 2 | present |
| 43 | Original | 2 | 2 | present |
| 43 | Exact-Masked | 2 | 2 | present |
| 43 | BOTA | 2 | 2 | present |
| 43 | IFRU | 2 | 2 | present |
| 43 | E2URec | 2 | 2 | present |
| 43 | NegGrad | 2 | 2 | present |
| 43 | PCGrad | 2 | 2 | present |
| 43 | SISA | 4 | 4 | present |
| 43 | RecEraser | 4 | 4 | present |
| 43 | Retain-Retrain-P5 | 2 | 2 | present |
| 43 | FullControl-P5 | 2 | 2 | present |

## Findings

- **WARNING STALE_STAGING_PATH**: Original manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed41/core/models/Original-Short/bota_short_original_seed41_v3/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: Original manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed41/core/models/Original-Short/bota_short_original_seed41_v3/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: Exact-Masked manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed41/core/models/Retrain-Short/bota_short_exact_masked_seed41_v3/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: Exact-Masked manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed41/core/models/Retrain-Short/bota_short_exact_masked_seed41_v3/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: BOTA manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed41/core/models/BOTA-T2-Short/bota_short_bota_seed41_v3/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: BOTA manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed41/core/models/BOTA-T2-Short/bota_short_bota_seed41_v3/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: IFRU manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed41/core/models/IFRU-Short-LoRA/bota_short_ifru_seed41_v3/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: IFRU manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed41/core/models/IFRU-Short-LoRA/bota_short_ifru_seed41_v3/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: E2URec manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed41/e2urec/models/E2URec-Short-FixedAB/bota_short_e2urec_seed41_v1/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: E2URec manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed41/e2urec/models/E2URec-Short-FixedAB/bota_short_e2urec_seed41_v1/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: Original manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed42/core/models/Original-Short/bota_short_original_seed42_v3/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: Original manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed42/core/models/Original-Short/bota_short_original_seed42_v3/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: Exact-Masked manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed42/core/models/Retrain-Short/bota_short_exact_masked_seed42_v3/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: Exact-Masked manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed42/core/models/Retrain-Short/bota_short_exact_masked_seed42_v3/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: BOTA manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed42/core/models/BOTA-T2-Short/bota_short_bota_seed42_v3/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: BOTA manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed42/core/models/BOTA-T2-Short/bota_short_bota_seed42_v3/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: IFRU manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed42/core/models/IFRU-Short-LoRA/bota_short_ifru_seed42_v3/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: IFRU manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed42/core/models/IFRU-Short-LoRA/bota_short_ifru_seed42_v3/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: E2URec manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed42/e2urec/models/E2URec-Short-FixedAB/bota_short_e2urec_seed42_v1/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: E2URec manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed42/e2urec/models/E2URec-Short-FixedAB/bota_short_e2urec_seed42_v1/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: Original manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed43/core/models/Original-Short/bota_short_original_seed43_v3/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: Original manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed43/core/models/Original-Short/bota_short_original_seed43_v3/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: Exact-Masked manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed43/core/models/Retrain-Short/bota_short_exact_masked_seed43_v3/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: Exact-Masked manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed43/core/models/Retrain-Short/bota_short_exact_masked_seed43_v3/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: BOTA manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed43/core/models/BOTA-T2-Short/bota_short_bota_seed43_v3/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: BOTA manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed43/core/models/BOTA-T2-Short/bota_short_bota_seed43_v3/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: IFRU manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed43/core/models/IFRU-Short-LoRA/bota_short_ifru_seed43_v3/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: IFRU manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed43/core/models/IFRU-Short-LoRA/bota_short_ifru_seed43_v3/scenarios/L4M4/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: E2URec manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed43/e2urec/models/E2URec-Short-FixedAB/bota_short_e2urec_seed43_v1/scenarios/L8/scenario_manifest.json`)
- **WARNING STALE_STAGING_PATH**: E2URec manifest points to the pre-publication .work path; the final adapter exists and matches the recorded hash. (`<EXPERIMENT_ROOT>/seed43/e2urec/models/E2URec-Short-FixedAB/bota_short_e2urec_seed43_v1/scenarios/L4M4/scenario_manifest.json`)

## Matched-user Development support

| Seed | Scenario | Users | Development samples | Minimum/user | Maximum/user |
|---:|---|---:|---:|---:|---:|
| 41 | L8 | 8 | 372 | 13 | 113 |
| 41 | L4M4 | 8 | 335 | 13 | 114 |
| 42 | L8 | 8 | 372 | 13 | 113 |
| 42 | L4M4 | 8 | 335 | 13 | 114 |
| 43 | L8 | 8 | 372 | 13 | 113 |
| 43 | L4M4 | 8 | 335 | 13 | 114 |

## Interpretation

`READY` means the frozen Development artifacts needed to begin attack verification are present and internally bound to the registered requests. It does not mean that MIA, sequential deletion or scalable storage has already been demonstrated.
