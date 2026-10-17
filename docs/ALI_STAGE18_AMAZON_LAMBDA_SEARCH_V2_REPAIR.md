# Stage 18 v2 — canonical-trajectory reproduction repair

## Diagnosis

v1 completed all200 transport steps without numerical candidate failures.
Its canonical endpoint matched formal Original exactly. It then failed the
historical lambda1 endpoint comparison, before any Development evaluation,
candidate selection or FinalTest access.

An isolated200-step legacy replay reproduced BOTH old Amazon BOTA adapters
tensor-exactly. Optimizer-state, RNG-state and canonical-trace hashes also
matched the old manifests. Fixed-A matrices matched across old/new endpoints.
However, the legacy replay's canonical B tensors differed from formal Original.

| Quantity | K2 | K4 |
|---|---:|---:|
| Actual legacy deletion-update L2 norm | 0.11767628 | 0.39064125 |
| Legacy canonical minus formal Original L2 norm | 5.48571279 | 5.48571279 |
| Legacy BOTA minus formal Original L2 norm | 5.48499020 | 5.48770295 |
| Old BOTA versus diagnostic replay max absolute difference | 0 | 0 |

The old `p2b.run_canonical_full` accumulates individual sample gradients
(`summed.add_(gradient / batch_size)`). Formal Original, Exact-Masked and
Stage18 use one backward pass through the averaged batch loss. Although these
expressions are mathematically equivalent, their floating-point accumulation
differs; here they produce different optimizer trajectories. The reproduced
legacy optimizer/RNG/trace hashes substantiate which trajectory the old run used.

Therefore v1's comparison mixed a canonical-trajectory offset with the actual
deletion update. It was an invalid same-trajectory reproduction requirement,
not a demonstrated Fisher failure and not a tokenizer issue.

Evidence (read-only after diagnosis):

`C:\python\E2URec-main\outputs\ali_stage18_reproduction_diagnostic\report.json`

`C:\python\E2URec-main\outputs\ali_stage18_reproduction_diagnostic\legacy_replay.pt`

## Repair and evidence boundaries

- Keep the formal batch-loss canonical convention used by Original/Exact-Masked.
- Keep exact seed42 canonical equality with Original as a hard gate.
- At EVERY transport step, independently advance `p2b.advance_full_transports`
  using the same formal canonical gradients and sources; require tensor-exact
  equality of lambda1 theta/m/v. This replaces the invalid historical-endpoint
  gate, not by relaxing its tolerance, but by comparing compatible trajectories.
- Retain historical endpoint differences as diagnostics, never call them
  reproduced by v2. Preflight requires the hash-bound successful legacy diagnosis.
- Search grid, refinement/selection/confirmation rules, data, controls and
  utility thresholds stay unchanged; Amendment17-repair01 documents the repair.
- v1 source/config/protocol and failed .work artifacts stay unchanged. v2 uses
  a separate output root; no unsealed v1 bank is promoted to a valid result.
- Do NOT describe subsequent historical-to-new Amazon gains as entirely due
  to lambda. Lambda0 versus the selected lambda WITHIN v2 is the controlled
  Fisher comparison. Historical Amazon comparisons need reevaluation under
  consistent canonical conventions before serving as updated paper evidence.

## Run

In the activated pytorch environment:

```powershell
cd C:\python\BOTA
.\scripts\bota_if\run_ali_stage18_amazon_lambda_search_v2.ps1 `
  -Mode Full `
  -Workspace C:\python\E2URec-main
```

Use v2, not the old v1 launcher. The first v2 bank must replay200steps because
the failed v1 run never persisted a complete bank or same-trajectory control.
Afterwards rerunning the same v2 command reuses sealed, verified units.

Outputs: `C:\python\E2URec-main\outputs\ali_stage18_amazon_lambda_search_v2`

`-Mode Preflight` checks inputs without training; `-Mode Smoke` is a two-step
non-scientific CUDA smoke test; `-Mode Analyze` inspects completed results.
Full training/calibration remains for the user to launch offline.

## Verification

- Legacy200-step replay: both requests reproduced tensor-exactly, with matching
  optimizer/RNG/trace hashes; no Development scoring.
- 59 unit/regression tests passed, including the new independent-control
  mismatch detector, frozen selection, no-reselection and v1 preservation rules.
- Real v2 smoke and preflight status are reported in the task handoff; neither
  constitutes a completed full lambda calibration or cross-seed confirmation.
