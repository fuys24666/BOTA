# Stage 14 — bounded lambda extension and three-seed no-Fisher ablation

Amendment 13 supplements the completed Stage 13 boundary result without changing
historical protocols/code/results. It binds the Stage 13 protocol, decisions,
summary and six cells by manifest hashes. All runs use Development/Dev-Select;
no FinalTest access or authorization. Prior exposure to these seeds/data is
acknowledged. This is calibration and descriptive ablation, not proof of correct
curvature, privacy forgetting or statistical significance.

## Frozen decisions

1. Seed 42, both L8/L4M4: coarse grid **16,32,64,128**. Also run **0** for the
   no-Fisher ablation and **1** for implementation control. Full AdamW recursive
   state transport remains identical; lambda multiplies only empirical-Fisher
   feedback, never deletion sources or final endpoint interpolation.
2. Each phase reproduces 1 and 16 against Stage 13 E_U, D_U, AUC and logloss
   within max(1e-8,0.001*abs(old)); 1 also passes the original formal-adapter
   gate. Seed-42 zero must reproduce its frozen historical metrics too. Failure
   blocks that cell; it is not silently counted as a negative scientific result.
3. Same selection policy: AUC damage<=0.005, logloss damage<=0.01, no collapse
   in both requests. Minimize equal-scenario mean E_U, exact ties nearest 1
   then candidate ID. Reruns are controls, not duplicate selection candidates.
4. Continue only when the mean E_U gain over frozen seed-42 lambda=16 is both
   **>=1e-6 absolute and >=1% relative**. These are engineering budget thresholds.
   If 32 wins, fine grid is **24,48**; if 64 wins, fine grid is **48,96**. Freeze
   the coarse decision, then at most one midpoint round (with controls 1/16).
5. No fine scan for boundary winner or inadequate gain. Hard cap **128**;
   a winner there is an unresolved boundary optimum, never a global optimum.
   No automatic expansion. Merge unchanged Stage 13 rows with new candidates,
   then freeze the final decision before cross-seed execution.
6. **Always** run seeds 41/43 with 0/1/16 on both requests. If the final >16
   winner passes the continuation gate, include it in those same trajectories.
   Confirmation requires all winner cells utility-eligible/non-collapsed,
   residual<1, lambda=16 controls eligible, and mean E_U below 16 separately
   in each seed. Preserve individual regressions; no reselection on these seeds.
7. Regardless of search outcome, report paired zero-vs-16 on all six cells.
   If a new winner was tested, also report zero-vs-winner on all six, even if
   confirmation failed. Do not replace a failed winner with 16. Report utility,
   residual, Toward-Exact Rate, absolute/relative E_U differences, and all missing
   or failed outcomes. A zero denominator yields null relative gain; an aggregate
   is withheld if any of its six paired E_U values is unavailable (no dropping).

Lambda=0 removes feedback but retains source and AdamW moment transport.
Thus 16-vs-winner quantifies tuning gain, while 0-vs-target quantifies feedback
contribution within this matched implementation. They answer different questions.
The seed-42 values also participated in selection; do not call their aggregate
an independent validation estimate or claim significance without an actual test.

Known numerical failures retain ineligible candidate rows; cross-seed zero
failures retain failed ablations. OOM, software/IO/integrity errors and control
failures abort and preserve diagnostics. No rescue, clipping or overwrites.
Endpoint parameter norm, cosine and relative L2 to Exact-Masked are descriptive
only and never used to choose lambda.

## Offline run

From activated `(pytorch)` in `C:\python\BOTA`:

```powershell
.\scripts\bota_if\run_ali_stage14_lambda_search_v1.ps1 `
  -Mode Full `
  -Phase All `
  -Workspace C:\python\E2URec-main
```

`All` performs the full frozen workflow. `Coarse` pauses after coarse decision;
`Sweep` after coarse/conditional fine/final selection. `Confirm` requires final
selection already frozen and completes the mandatory ablation plus conditional
new-winner confirmation. Use the same code/config between phases.
`Preflight` validates actual input without loading models; `SyntheticDryRun`
only tests fake data. `Analyze` verifies artifacts and recomputes decisions and
ablations without model inference/training.

If needed, explicitly choose the runtime:

```powershell
$env:BOTA_PYTHON = 'C:\Users\Administrator\anaconda3\envs\pytorch\python.exe'
```

## Artifacts and recovery

Output root: `C:\python\E2URec-main\outputs\ali_stage14_lambda_search_v1`.

- `protocol/contract.json`: source/config/input hashes, parent bindings, Git/runtime.
- `cells/coarse`, `cells/fine`, `cells/confirm`: atomic seed/request adapters,
  predictions, metrics, control reproduction, endpoint diagnostics and actual steps.
- `coarse/decision.json`, `selection/decision.json`: immutable decisions.
- `summary/report.md`, `summary/summary.json`, `summary/metrics.csv`: all outcomes.
- `summary/fisher_ablation.json`: paired six-cell no-Fisher results and aggregates.
- `.work/<short-id>`: progress every ten steps; failed work retained with diagnostics.

Rerun the same command after interruption: verified completed cells are reused;
incomplete cells restart from scratch. Resume is per cell, not per optimizer step.
Never edit frozen files to bypass a check. An OS lock prevents concurrent Stage 14
writers. Stage 13 output remains unchanged. Do not run competing GPU experiments.

Uninterrupted optimizer budget: 400 coarse + optional 400 fine + mandatory 800
cross-seed/ablation = **1,200 or 1,600 physical steps**, plus inference. Source
trajectory is shared by lambdas within each cell. Coarse has six states, fine at
most four, confirmation three or four. Existing GPU memory guards are retained.
Stopped/failed attempts add actual work; no zero-online-cost or latency claim is
inferred from offline execution timing.
