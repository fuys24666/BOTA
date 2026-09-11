# Ali feedback Stage 0 Amendment 01 validation

- Amendment: `ali-stage0-amendment-01-mechanism-signal-equal-billing-v1`
- Validation date: 2026-09-11
- Status: **PASS**
- FinalTest accessed: **no**
- Experiment outputs read by the validator: **no**

## Verified safeguards

- The original Stage 0 preregistration and both completed artifact-audit records
  match the SHA256 hashes bound by Amendment 01.
- The primary attack, sequential and scale contracts remain unchanged.
- All amended experiments are Development-only and explicitly forbid FinalTest.
- Known seed-42 FM/FS and empirical-Fisher results are disclosed as retrospective
  context rather than represented as prospectively preregistered evidence.
- Stronger-signal and fixed-horizon experiments are labelled secondary and cannot
  replace L8 or L4M4 as primary scenarios.
- The mechanism arm must be frozen before any new attack result is inspected.
- All named arms, seeds, scenarios, metrics, null results and adverse results are
  reportable.

## Validation results

```text
Amendment 01 validation: PASS
Targeted test suite: 10 passed
```

The machine-readable hashes and check results are stored in
`docs/audits/ali_stage0_amendment01_validation_v1.json`.
