# Stage 18 — Amazon Movies and TV Development calibration

## Scope and frozen decision

This stage searches only the FS empirical-Fisher response multiplier lambda.
It does not search learning rate, rank, source scaling, model architecture,
baseline hyperparameters, or ranking objectives. Lambda64 is a comparator,
not an assumed Amazon optimum. Amendment17 is separate from earlier sealed work.

1. Seed42: lambda=0,1,2,4,8,16,32,64,128,256. Use both frozen K2/K4 requests.
2. Choose the lowest equal-request mean absolute probability error E_U among
   candidates satisfying AUC damage <=0.005, logloss damage <=0.01, and no
   prediction collapse in BOTH requests. Exact ties prefer smaller lambda.
3. At most one midpoint round, only if the coarse winner is positive/internal
   and beats eligible lambda0 by >=1e-6 absolute AND >=1% relative mean E_U.
   Evaluate one arithmetic midpoint to each adjacent coarse value. Compare
   the new candidates together with all coarse candidates. Stop at boundaries
   or negligible improvement; do not expand beyond256 automatically.
4. Freeze the shared selected lambda, then evaluate seeds41/43 against 0,1,64.
   Never reselect from confirmation results. Confirmation requires utility gates,
   >=1e-6 absolute AND >=1% relative mean improvement over zero in each seed,
   and no request regression above1e-6. All failures remain reported.

## Data and interpretation

- Existing historical Amazon checkpoint and 200-step adaptation trajectory.
- 3840 Development examples; K2/K4 have only10/20 requested-user examples.
- K2 is contained in K4; these are not independent request samples.
- All Development data is used for evaluation/calibration only, not gradient
  training. No new DevFit subset is needed because this stage fits no auxiliary
  scorer or attack. The already-used cohort is NOT a new holdout.
- Seeds41/43 vary fixed-A initialization and runtime seed, but share historical
  pretraining, the cohort, request identities and batch order. This is conditional
  cross-seed confirmation, not three independent end-to-end pretraining runs.
- No FinalTest or NDCG-based selection. Formal evaluation and fair Amazon baseline
  calibration are separate future steps, not automatically authorized here.
- A boundary winner is the best tested candidate, not a global optimum.

## Integrity and runtime

The workspace benchmark YAML is the historical authority: its hash matches
the original prepared registry and model contracts. The repo copy later renamed
a runtime-config alias. The implementation checks semantic equality apart from
that alias, and verifies the historical source hashes. The old alias has a
Goodreads name, but model and data are explicitly selected from the Amazon
benchmark; no Goodreads dataset is used.

Seed42 canonical B tensors must exactly reproduce Original. Lambda1 must align
with the old BOTA update at cosine>=0.9999, relative L2 error<=0.01 and max error
<=0.001. This is a reproduction check, not a reason to emphasize lambda1 in the
eventual paper. Lambda0 is the primary no-Fisher ablation.

Full freezes source, config, data and reference hashes before computation.
Banks, Exact endpoints, per-model predictions, and selection decisions are
published atomically and verified on resume. Known numerical failures remain
candidate outcomes; CUDA/OOM/programming failures abort. No clipping rescue.
Do not edit source/config while Full is running or try to run two Full processes.
No raw/user data is copied into Git; outputs stay under E2URec-main/outputs.

## Commands (activated pytorch environment)

```powershell
cd C:\python\BOTA
.\scripts\bota_if\run_ali_stage18_amazon_lambda_search_v1.ps1 -Mode Preflight -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage18_amazon_lambda_search_v1.ps1 -Mode Full -Workspace C:\python\E2URec-main
```

Full defaults to all phases. Optional `-Phase Coarse` stops after the coarse
decision; `-Phase Sweep` stops after final seed42 selection. Running Full again
without Phase resumes and proceeds to confirmation. Reuse the identical command
after interruption. A current incomplete bank or prediction is recomputed;
completed verified units are retained. Failed .work diagnostics are not deleted.

`-Mode Smoke` runs only two training steps, four endpoint saves, and four-example
inference. Its outputs are explicitly non-scientific and never enter selection.
`-Mode Analyze` verifies the completed inventory and displays the result; it
does not launch missing experiments. Do not run Analyze while Full holds the lock.

Final outputs:

`C:\python\E2URec-main\outputs\ali_stage18_amazon_lambda_search_v1\summary\report.md`

`C:\python\E2URec-main\outputs\ali_stage18_amazon_lambda_search_v1\summary\result.json`

Maximum planned computation: four shared 200-step transport trajectories
(coarse, optional fine, two confirmations) plus four 200-step masked replays for
the confirmation seeds; at most40 candidate-request evaluations plus references.
Seed42 uses immutable old Exact endpoints. This is not40 separate full trainings.

## Implementation checks (2026-10-11)

- Real-asset preflight: PASS, 69 input files checked; request support10/20.
- Real CUDA smoke: PASS, two training steps with all10 coarse lambda values;
  endpoint save/reload plus four-example inference. This is not a full-run
  reproduction or evidence that any lambda is better.
- 47 tests passed: Stage18 selection/gating/refinement/confirmation/frozen
  decisions/tamper checks/orchestration/resume, existing Amazon tests and
  Stage14 regression tests.
- The full200-step historical reproduction gates and complete scientific
  evaluation are deliberately left to the user's offline Full run.
