# Stage 17 — MovieLens-1M fixed-lambda64 refresh

## Purpose and evidence boundaries

Stage 16 is complete and sealed by `docs/audits/ali_stage16_result_audit_v1.json`.
It already reevaluated the two primary requests in three seeds on the frozen
FinalTest cohort, including lambda=0/1/64, five calibrated approximate baselines,
Original, Exact-Masked and the two P5 references, with binary and ranking lanes.
Do not rerun Stage 16, reinterpret an old lambda1 stress test as lambda64, or tune
lambda in response to the new outcomes.

Amendment 16 fixes lambda64 for this diagnostic suite. These are **Development
diagnostics**, not an independent new holdout confirmation. Stage 17 reads the
sealed Stage 16 artifacts for provenance but never creates a FinalTest loader.
The original preregistrations, source modules and all historical runs are left
unchanged. Another dataset's calibration is a separate future protocol.

## Coverage / missing-experiment audit

| Evidence | Current status | Stage 17 action |
|---|---|---|
| Main binary fidelity, P5 target and sampled ranking | Stage 16 lambda64 complete | Preserve; no new FinalTest access |
| FS lambda0/1/64 ablation | Stage 14 Development and Stage 16 FinalTest complete | Reproduce adapters, report again on full Development |
| FM versus FS under the calibrated Fisher response | Not covered by lambda0/1/64 alone | Add FM0 and FM64; report FS0/FS64 as a 2x2 factor comparison, FS1 as historical control |
| Primary MIA | Old lambda1 not transferable | All primary endpoints, three seeds, two requests, four score attacks |
| Stronger requests and attacks | Old lambda1 not transferable | All nine frozen Stage 2 requests, seed42; only high_k4 eligible for formal attack interpretation, additionally gated by reference signal and utility |
| Cumulative deletion | Old lambda1 not transferable | All 20 Stage 5 cells per seed; reuse hash-bound Exact models; compute 39 distinct union endpoints and expand to all 60 cells |
| 1,200-step transport | Old lambda1 not transferable | Recompute FS1 and FS64 on the original Stage 3 six-epoch order; FS1 reproduction gate; both requests, three seeds |
| Storage bank and compression | Old lambda1 bank not transferable | Fresh 256-user FS64 banks, three seeds, eight 32-user chunks each; 12/32/64/128/256 nested screens, all 12 representations |
| Compression impact on behavior | Parameter screen alone insufficient | Evaluate all 12 representations at K=256 on both requests, all seeds; binary fidelity and sampled ranking, no format selection |
| SVD/effective rank | Stage 11 describes the old bank only | Recompute raw and row-normalized spectra, energy ranks, pairwise directions and query-specific projection errors from the new bank |
| Serving latency / offline cost | Old 12-user timing not a lambda64 measurement | Fresh dedicated FS64 12-user construction and 256-user construction; resident CPU compose/materialize/fsync/reload at both sizes |
| Cost break-even | Requires explicit assumptions | Cost-only estimate using new construction/serving timings and hash-frozen historical Stage 5 exact replay times; no fallback/refresh/network claim |
| Historical 600-step continuous relay | Different schedule from Stage 3 | Isolated old authority, seed42/L8, carry theta/m/v continuously, lambda64, fresh canonical and masked replay; parameter and Development behavior metrics |
| Historical position/frequency figure | Different source and requests | Isolated corrected P3-C5F authority, seed42, low/middle/high strata × four windows; fresh lambda64 parameter diagnostics and immutable masked replays |

There are **142 required result units**, plus cached model/prediction/bank units.
Scientific numerical failures remain explicit outcomes. They do not trigger
automatic clipping, a smaller lambda, a new request, or success-only reporting.
Infrastructure failures stop the run and preserve `.work` diagnostics.

### What this still cannot establish

- Score-based MIA is not a privacy certificate or a trained U-LiRA experiment.
  A weak/absent Original–Exact signal must be reported as inconclusive.
- Stage 5-style cumulative composition is from a common canonical trajectory.
  It is not deletion interleaved with new learning. The historical relay tests
  continuous transport of the same request, not arbitrary adaptive deletion.
- Full-Development binary queries and frozen Stage 15 DevSelect ranking queries
  differ. Report their coverage separately; never join by row number.
- The ranking candidate set is fixed sampled ranking, not full-catalogue ranking.
- Linear compression and SVD diagnostics do not prove impossibility for every
  method in this family. Population storage remains a projection, not a deployment.
- No calibrated reliability detector/fallback policy is implemented. Those are
  new research methods, not metric refreshes. Break-even assumes covered requests,
  a fixed reusable bank and no refresh/fallback cost; it is not a deployment claim.
- SISA had no eligible calibrated candidate in the existing selection authority;
  do not invent a newly tuned SISA baseline inside this frozen experiment.
- Historical relay/position are one-seed diagnostics under their original
  protocols, not additional independent replications of the v3 primary setting.

## Run from the existing pytorch environment

```powershell
cd C:\python\BOTA
.\scripts\bota_if\run_ali_stage17_lambda64_suite_v1.ps1 `
  -Mode Seal `
  -Workspace C:\python\E2URec-main

.\scripts\bota_if\run_ali_stage17_lambda64_suite_v1.ps1 `
  -Mode Full `
  -Phase All `
  -Seed All `
  -Workspace C:\python\E2URec-main
```

Seal is idempotent only for an identical source/input/protocol/runtime snapshot.
If the source has changed since sealing, do not overwrite the seal or delete old
results to bypass the check. Investigate/version the implementation correction.

Full is long and deliberately user-run. It performs substantially more inference
than Stage 16 because it refreshes stress tests and every compression candidate.
Run only one GPU process. The allocator is capped at 14 GiB; 10 GiB free memory
is required. Bank chunks bound memory and use short temporary paths on Windows.
Keep ample free disk space for banks, adapters, predictions and failed-stage logs.

The same Full command resumes verified completed units without changing the
protocol. Only an interrupted, unpublished unit is recomputed. Scientific
failure outcomes are reused as outcomes, not retried with a different setting.

For a first overnight batch, use `-Phase Primary -Seed All`. Then run `-Phase All`
to reuse Primary and continue. Individual phases:

```text
Primary, Stronger, Sequential, Horizon, Storage, StorageBehavior, Latency,
LegacyRelay, LegacyPosition
```

Stronger and the two legacy phases run only for seed42. `-Seed All` includes them.
An incomplete suite prints `PARTIAL`; that is not itself a runtime failure.

To check artifact completion without training/inference:

```powershell
.\scripts\bota_if\run_ali_stage17_lambda64_suite_v1.ps1 `
  -Mode Analyze -Workspace C:\python\E2URec-main
```

## Output and paper replacement checklist

Root: `C:\python\E2URec-main\outputs\ali_stage17_lambda64_suite_v1`

- `protocol/contract.json`: frozen source/input SHA-256, runtime, requests and rules.
- `cells/bank_*`, `cells/storage_bank_*`: fresh transport and bank evidence.
- `cells/eval_*`: fidelity, parameter diagnostics, separate P5 results when
  applicable, ranking and attacks; explicit user coverage and missing methods.
- `cells/storage_screen_s*/`: 120 rows per seed, spectra, pairwise cosines and
  compressed adapters. All twelve compression candidates remain visible.
- `cells/latency_s*/`: timing samples, bank-build timing and conditional cost estimate.
- `cells/legacy_*`: historical-protocol lambda64 replacements, not v3 pooled data.
- `cells/summary/behavior.csv`, `ranking.csv`, `parameter.csv`, `attacks.json`,
  `primary_ablation.csv`, `sequential_all_60_cells.json`,
  `historical_windows_relay.csv`, `all_results.json`, `report.md`.

After completion, compare lambda1 and lambda64 **within the same experimental
target and query population**. Replace the appropriate stress-test tables,
storage/spectrum figure, timing/cost discussion, FM/FS ablation and attack text.
Primary FinalTest results come from Stage 16, not the Development refresh.
Do not pre-commit to a positive outcome or preserve an old negative claim without
checking its lambda64 counterpart. Larger Fisher weight is empirical response
calibration; improvements alone do not validate the Fisher as the true Hessian.

## Validation (bounded; not formal scientific results)

```powershell
python -m pytest tests/test_ali_stage17_lambda64_suite_v1.py -q
$env:BOTA_STAGE17_GPU_SMOKE='1'
python -m pytest tests/test_ali_stage17_integration_v1.py -q -s
Remove-Item Env:BOTA_STAGE17_GPU_SMOKE
```

The real-model smoke checks only a few optimizer steps and four predictions per
lane, plus the historical runtime's canonical/T2 interface. It must not be quoted
as evidence that a 200/600/1,200-step experiment will pass.
