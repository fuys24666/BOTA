# Ali feedback experiments: Stage 0 Amendment 01

**Amendment ID:** `ali-stage0-amendment-01-mechanism-signal-equal-billing-v1`  
**Frozen:** 2026-09-11, before the amended experiments  
**Machine-readable authority:** `configs/ali_stage0_amendment01_v1.json`

## Relationship to the original preregistration

This document does not replace or edit the Stage 0 preregistration. The original
attack, sequential-request and scale contracts remain primary and authoritative.
Their registered L8 and L4M4 requests, thresholds, reporting obligations and
Development-only policy are unchanged. This amendment adds three experiments
that must be resolved before the expensive parent experiments are interpreted:

1. freeze the BOTA mechanism arm;
2. measure a separately labelled stronger-signal request family; and
3. add a fixed-horizon, equal-schedule comparison without conflating it with
   independently stopped retain-only retraining.

FinalTest remains forbidden during method development. Null and adverse results
must be reported.

## Evidence already known before this amendment

The amendment is not blind to all prior evidence. The manuscript already reports
the seed-42 FM/FS ablation, including the lower L4M4 behavioral residual for FM,
and the seed-42 empirical-Fisher improvement of 0.0179. The three-seed P5
sensitivity table is also already known. These observations are retrospective
context; they must never be described as prospectively preregistered findings.

The new evidential contribution is the frozen cross-seed decision rule and the
results not yet inspected under it.

## A1. Mechanism stabilization

Run `FM source-only`, `FS source-only` and `FS + empirical Fisher` on Development
for seeds 41, 42 and 43 and for both L8 and L4M4. All arms must share the same
canonical trajectory, request, fixed LoRA-B coordinate and evaluation rows.

Report every seed-by-scenario cell for local residual, absolute error `EU`, delta
cosine, norm ratio, relative L2 error, AUC, log loss, build time and artifact
size. Means and sample standard deviations summarize each scenario; three seeds
do not support a significance claim.

The empirical-Fisher correction is removed if, in both scenarios, source-only FS
is no worse than corrected FS by more than 0.02 in mean local residual or mean
relative L2 error and satisfies the existing utility guardrails. Otherwise the
corrected FS arm remains, but the correction is described as secondary and
non-uniform.

FM can replace the selected FS arm only if it has a lower mean local residual in
both scenarios, is no worse by more than 0.02 in mean relative L2 error in either
scenario and satisfies the utility guardrails. Otherwise FM remains diagnostic.
The selected mechanism arm is frozen before any new attack result is inspected.

## A2. Stronger-signal secondary analysis

The original L8 and L4M4 requests remain primary. A secondary outcome-blind
request family varies request size (`K=2,4,8`) and per-user exposure stratum
(`1`, `2--5`, and `>=8` slots). Eligible users must have at least ten Development
examples. Within each stratum, users are ordered deterministically by the SHA256
hash of their public association; model outcomes may not enter selection.

Every constructed request is reported. An absolute Original-to-Exact-Masked
probability effect `DU >= 0.001` is considered adequate for interpreting attack
movement in that secondary request. Passing this rule does not promote a request
to primary status, and failing it is itself reportable.

## A3. Fixed-horizon equal-schedule sensitivity

For every seed, take `T*` from the already existing FullControl-P5 best
Development checkpoint before generating amended endpoints. Then construct
Original-`T*`, Exact-Masked-`T*` and BOTA-`T*` with exactly the same initialization,
number of optimizer steps, ordered sample slots and full-batch denominator.
Exact-Masked-`T*` zeros the requested slots without compacting the retained data.

The matched target is Exact-Masked-`T*`. Retain-Retrain-P5 remains a separately
stopped converged sensitivity endpoint and is never used as the denominator of
this fixed-horizon residual. Failure limits BOTA to its validated 200-step
horizon; success still does not establish equivalence to independently
converged retain-only retraining.

## Execution order and stopping discipline

1. Mechanism stabilization and mechanism-arm freeze.
2. Stronger-signal secondary requests.
3. Fixed-horizon equal-schedule sensitivity.
4. Original Stage 0 attack verification.
5. Original Stage 0 sequential verification.
6. Original Stage 0 scale verification.

No amended arm, threshold, seed, request-selection rule, endpoint, metric or
decision rule may change after its affected result is inspected without a new
dated amendment.

## Validation command

From the repository root:

```powershell
python scripts/bota_if/validate_ali_stage0_amendment01_v1.py
```

The validator checks that the original preregistration and completed Stage 0
audit records still match their bound SHA256 hashes and that this amendment
retains the Development-only and no-suppression safeguards.
