# Stage 12 — Development Fisher-response lambda extension

## Question and provenance

Ali noted that Stage 8 improved up to its largest searched lambda, 1. This
reviewer-motivated extension asks whether 1.5 or 2 improves the same short-window
matched-counterfactual target. Amendment 11 supplements, never edits, the frozen
Stage 8 protocol and its five candidate results. Prior Development exposure is
explicit: this is not a newly untouched confirmation dataset.

`lambda` multiplies empirical-Fisher feedback **inside** the full AdamW
parameter/moment recursion. Deletion source terms, window length, learning
rate, evaluation rows and utility guards remain fixed. It is neither an IF
damping parameter nor final endpoint interpolation. This stage does not address
privacy, long-horizon performance, new ranking metrics or storage compression.

## Frozen procedure

1. Validate Stage 8 source/adapter hashes, Stage 9 benchmark bindings, Stage 1
   references and the unchanged Dev-Fit/Dev-Select partition. No test loader is
   called. Freeze source hashes (including uncommitted files), input hashes,
   Git state and runtime metadata in a new protocol bundle.
2. Seed 42, both L8/L4M4: execute lambda=1/1.5/2 jointly on one shared canonical
   200-step trajectory per scenario. Thus 400 physical optimizer steps for the
   screen, not 200 steps per lambda. Inference is additional work.
3. Lambda=1 must reproduce the formal adapter within Stage 8's existing gate.
   On seed 42 its E_U, D_U, AUC and logloss must also match frozen Stage 8 within
   `max(1e-8, 0.001*abs(old))`. A mismatch blocks publication/selection; it is
   not counted as a negative scientific outcome.
4. Merge the five unchanged historical candidates with 1.5/2 (seven candidates,
   fourteen request cells). Rerun-1 remains a separately reported control.
   Eligibility: both scenarios satisfy AUC damage <=0.005 and logloss damage
   <=0.01, with no prediction collapse. Minimize equal-scenario mean E_U;
   exact ties prefer proximity to 1, then candidate ID. No candidate omission.
5. Freeze the selection bundle before confirmation. Only if lambda>1 wins,
   run that fixed winner and lambda=1 on seeds 41/43, both requests. This adds
   800 physical optimizer steps, at most 1,200 new steps overall. Previously
   evaluated seeds are called cross-seed confirmation, not untouched holdouts.
6. Confirmation passes only if the winner is utility-eligible/non-collapsed
   with residual<1 in all four cells and improves mean E_U separately in each
   seed. Control eligibility is also required. Individual cell regressions
   remain visible even if the aggregate rule passes.
7. A winner at 2 is still a search-boundary optimum. Do not extrapolate beyond
   it or expand the grid automatically. No result authorizes FinalTest.

Nonfinite candidate transport, adapter or predictions is an explicitly reported
ineligible numerical result (no clipping, rescue, or replacement setting).
Nonfinite lambda=1, IO, OOM, integrity and software failures abort the affected
cell and retain diagnostics. They are not misreported as completed negatives.

## Run offline

In the activated `pytorch` environment, from `C:\python\BOTA`:

```powershell
.\scripts\bota_if\run_ali_stage12_lambda_extension_v1.ps1 `
  -Mode Full `
  -Phase All `
  -Workspace C:\python\E2URec-main
```

`Full/All` performs the screen and conditional confirmation automatically.
To inspect the screen before confirmation, use `-Phase Sweep` and later
`-Phase Confirm`. Do not change code/configuration between phases.
`-Mode Preflight` checks inputs without loading a model. `SyntheticDryRun`
exercises selection only. `Analyze` verifies bundles and prints existing results.

If the activated environment is not picked up, set:

```powershell
$env:BOTA_PYTHON = 'C:\Users\Administrator\anaconda3\envs\pytorch\python.exe'
```

## Outputs and recovery

All new files live under
`C:\python\E2URec-main\outputs\ali_stage12_lambda_extension_v1`:

- `protocol/contract.json`: immutable protocol, source/input hashes and provenance.
- `cells/s42_L8`, etc.: atomic per-seed/scenario adapters, metrics, predictions,
  lambda=1 reproduction checks, actual step counts and runtime/memory observations.
- `selection/selection.json`: immutable seven-candidate screen and rerun controls.
- `summary/report.md`, `summary/metrics.csv`, `summary/summary.json`: full audit
  table, fixed selection, conditional confirmation and boundary flag.
- `.work/<short-id>/progress.json`: progress every ten optimizer steps;
  `FAILED.json` appears on aborted work, which is kept rather than deleted.

Rerun the same command after an interruption. Verified completed cells are reused;
only incomplete cells restart. This is **cell-level**, not optimizer-step resume.
No overwrite flag is provided. Frozen code/config/input changes require an explicit
new-version decision; do not delete valid results to bypass the integrity check.

Tests and synthetic checks do not constitute a completed GPU experiment. Even a
successful run supplies only Development evidence, not new FinalTest conclusions.
