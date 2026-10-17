# Stage 13 — bounded Fisher-response sensitivity and calibration

This is Amendment 12, prompted by the completed Stage 12 lambda=2 boundary
winner. It supplements, never overwrites, prior protocols, code or artifacts.
Stage 12's protocol, selection, summary and six cells are explicitly hash-bound.
No FinalTest evaluation or authorization is included. Development and these
seeds have been evaluated before; do not describe this as an untouched holdout.

## Fixed search and stopping rules

1. Seed 42, L8/L4M4: coarse grid **2, 4, 8, 16**, with an additional lambda=1
   implementation control. Both 1 and 2 must reproduce frozen Stage 12 E_U,
   D_U, AUC and logloss within max(1e-8, 0.001*abs(old)); 1 also passes the
   inherited formal-adapter reproduction gate. A control failure blocks the
   cell and is not a negative scientific result.
2. Preserve the original selection rule: AUC damage <=0.005, logloss damage
   <=0.01 and no collapse in both requests; minimize their mean E_U, ties
   prefer proximity to 1 then candidate ID. Lambda scales the Fisher feedback
   inside full AdamW recursion, not the source or final endpoint.
3. Freeze the coarse decision. If 4 wins, fine grid is **3, 6**. If 8 wins,
   fine grid is **6, 12**. Only trigger this one fine round when the gain over
   frozen lambda=2 is **both >=1e-6 absolute E_U and >=1% relative**. These
   are engineering budget thresholds, NOT statistical significance. Failure
   of a neighboring candidate does not establish a smooth turning point.
4. No fine search for a boundary winner or sub-threshold gain. No lambda>16.
   A winner at 16 remains explicitly labeled an unresolved boundary optimum;
   cross-seed confirmation can still run, but no search expansion occurs.
5. Merge Stage 12's seven frozen candidates with 4/8/16 and at most two fine
   values. Rerun controls are reported separately, not duplicated in selection.
   Freeze the final selection before examining new confirmation predictions.
6. If the final winner is >2 and passes the same budget gate, compare it with
   **both lambda=1 and 2** on seeds 41/43 and both requests. Require all winner
   cells to satisfy utility, non-collapse and residual<1; require both controls
   to be eligible; mean E_U must improve over each control separately in each
   seed. Report individual regressions even if the mean passes. No reselection.
7. Otherwise report the raw screen winner and skip new confirmation, retaining
   Stage 12 as the existing reference. A failed confirmation is retained too.

All numerical/utility failures are retained. Only known numerical tangent
errors or nonfinite candidate states/predictions become ineligible rows. OOM,
IO, programming errors and failed controls abort and preserve diagnostics.
No automatic clipping, rescue, adaptive grid doubling or overwriting.

Each successful adapter also receives descriptive parameter-update norm,
cosine and relative L2 against Exact-Masked. These diagnostics are NOT inputs
to selection, proof of Fisher accuracy, or privacy/behavioral guarantees.

## Offline command

Run from `C:\python\BOTA` in the activated `pytorch` environment:

```powershell
.\scripts\bota_if\run_ali_stage13_lambda_search_v1.ps1 `
  -Mode Full `
  -Phase All `
  -Workspace C:\python\E2URec-main
```

`All` runs coarse, conditional fine, then conditional confirmation automatically.
Optional phases: `Coarse` stops after freezing the coarse decision; `Sweep`
stops after coarse/conditional fine and final selection; `Confirm` requires an
already frozen selection. Repeating phases reuses hash-verified completed cells.
Do not edit source/configuration during or between phases.

`Preflight` checks real inputs without loading models; `SyntheticDryRun` tests
selection on made-up numbers, never evidence. `Analyze` verifies saved bundles
and recomputes decisions without training. Runtime override if needed:

```powershell
$env:BOTA_PYTHON = 'C:\Users\Administrator\anaconda3\envs\pytorch\python.exe'
```

## Outputs and compute budget

Under `C:\python\E2URec-main\outputs\ali_stage13_lambda_search_v1`:

- `protocol/contract.json`: source/input hashes, runtime and parent bindings.
- `cells/coarse`, `cells/fine`, `cells/confirm`: separate atomic seed/request
  cells with adapters, metrics, predictions, reproduction and endpoint diagnostics.
- `coarse/decision.json`: fixed midpoint grid or explicit stopping reason.
- `selection/decision.json`: final frozen selection and budget-gate result.
- `summary/report.md`, `metrics.csv`, `summary.json`: all candidates, controls,
  failures, absolute/relative gains and confirmation results.
- `.work/<short-id>`: progress and preserved incomplete/failed diagnostics.

One shared 200-step canonical trajectory per phase/seed/request (not per
lambda). Coarse=400 physical optimizer steps; fine adds 400 if triggered;
confirmation adds 800 if triggered. Maximum **1,600** for an uninterrupted
complete run, plus inference. Failed attempts/restarts can add actual work.
GPU cap and minimum-free-memory policy inherit Stage 8. An OS-level output
lock blocks concurrent Stage 13 writers. Short staging paths avoid Windows
path-length trouble. Resume is per cell, not per optimizer step.

Real GPU execution is left to the user. Passing unit tests and input preflight
does not establish the scientific result or authorize publication claims.
