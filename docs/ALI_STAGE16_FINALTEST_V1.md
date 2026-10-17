# Stage 16 — frozen lambda=64 joint main-experiment re-evaluation

## Scope

Amendment 15 authorizes one separately sealed FinalTest re-evaluation after
Stage 15 Development ranking is complete. Stage 15's 66 cells and summary are
independently checked and bound by `docs/audits/ali_stage15_result_audit_v1.json`.
All older preregistrations and experiment results remain unchanged.

Lambda is fixed at **64**, selected in Stage 14, not selected using NDCG.
There is no training, further search, request replacement, or best-seed selection.
This cohort has been evaluated previously: this is a frozen re-evaluation, not
a newly untouched holdout. The new access is explicitly recorded.

## Frozen matrix and outputs

Three seeds (41/42/43), two existing requests (L8/L4M4), 12 frozen endpoints:
Original, Exact-Masked, BOTA64, BOTA1, BOTA0, FullControl-P5,
Retain-Retrain-P5, IFRU, E2URec, NegGrad, PCGrad and RecEraser.
The extra lambda=0 control quantifies the response correction relative to
no Fisher feedback; lambda=1 retains the old published setting. Neither is a
candidate for post-test reselection. No baseline is retuned to ranking.

There are 72 endpoint combinations and two inference lanes: **144 saved cells**.

| Lane | Population and measures |
|---|---|
| binary | Original chronological 20,000 FinalTest prompts; requested-user absolute E_U and D_U, Exact-Masked residual with user-bootstrap CI, TER, JSD, RMS, AUC/accuracy/log loss and answer loss |
| P5 (from binary predictions) | Absolute discrepancy to Retain-Retrain-P5 and separately normalized P5 residual/CI using FullControl-P5; not substituted for Exact-Masked |
| ranking | Separate common-history candidate queries; sampled NDCG@5/10/20 and Recall@5/10/20, per user and user-macro all/requested/retained groups |

Requested-user support is reported separately for the original prompts and
ranking queries. Do not assume ranking coverage repairs missing binary support,
or mix these query sets and denominators. Three-seed means and sample SD are
descriptive; no new ranking acceptance threshold or privacy claim is introduced.
The unchanged binary utility guards are AUC damage <=0.005 and log-loss damage
<=0.01. The inherited combined comparison requires all six BOTA64 cells to
pass guards and residual <1, plus six-cell mean E_U below each of five baselines.
The stricter per-cell win count is also reported separately, not implied by that
aggregate decision. Failure is a result, never an instruction to tune lambda.

## Ranking protocol

The Stage 15 rule is transferred to the start of FinalTest, without scoring-based
selection. Use all FinalTest rows, not the Development Dev-Select subset.

- Catalogue: observed items before FinalTest, reconstructed from training and
  Development rows/histories and first FinalTest-row pre-event histories.
- Context: each user's common last ten interactions before their first
  FinalTest event. Later test interactions never enter a prompt.
- Positives: all distinct FinalTest items rated >=4, in the pre-test catalogue
  and not already interacted with before FinalTest.
- Negatives: 100 distinct unobserved catalogue items in the same frozen SHA256
  order as Stage 15. Exclude that user's observations through FinalTest end,
  including low ratings. This exclusion uses labels to define the offline
  evaluation pool; it is not information passed to the model.
- Fewer than 100 negatives: use all and report count. No positive: exclude from
  the macro average and list explicitly, never substitute zero NDCG. No
  negative: fail the protocol check without silently changing the candidate rule.
- Every method receives identical candidates, context and first-token
  P(Yes)/(P(Yes)+P(No)) scoring. Dummy targets are always `Yes.`, not relevance.
  Preserve the legacy no-attention-mask convention and batch size four.
  RecEraser combines token masses before normalization. Ties use ascending
  movie ID. Reject >512-token ranking prompts rather than truncate silently.
- Unobserved negatives have assumed zero relevance, not verified dislikes.
  This is **sampled-candidate NDCG**, not full-catalogue performance.

## Seal, admission and recovery

`Preflight` validates Development authorities, model hashes and Stage 15 seal;
it does not materialize a test loader or read old test metrics. `Seal` writes
the exact source/config/input/model contract before new test access. Unlike
Stage 10's clean-Git requirement, this amendment uses a source-hash inventory
and records the dirty Git snapshot, without committing unrelated work. Do not
edit experiment sources/configs or model files during the run.

`Full` requires `-ConfirmFinalTest`. The first access reserves a non-resettable
ledger, reconstructs the existing chronological cohort and saves candidate
hashes. Do not delete the ledger to rerun or overwrite an unfavorable result.
`Resume` requires the same confirmation and identical seal; it verifies/reuses
complete cells and restarts only incomplete cells. Failures remain in `.work`
and `access`. A code/protocol change after access needs a separate documented
amendment, not a silent reseal. Partial lane/seed execution is allowed only as
execution scheduling; every cell remains mandatory for the combined report.

Old Original/Exact/P5/baseline/lambda1 binary results are independently checked
against Stage 10 after inference. Differences in E_U, D_U, AUC, log loss or P5
residual exceeding max(1e-8, 0.001*abs(old)) produce a review-required status.
This preserves all scores; it does not suppress the report or authorize tuning.

## Commands (PowerShell, active pytorch environment)

```powershell
cd C:\python\BOTA
# Admission only; no FinalTest access.
.\scripts\bota_if\run_ali_stage16_finaltest_v1.ps1 -Mode Preflight
.\scripts\bota_if\run_ali_stage16_finaltest_v1.ps1 -Mode Seal

# Long run. Both lanes, all three seeds; zero optimizer updates.
.\scripts\bota_if\run_ali_stage16_finaltest_v1.ps1 `
  -Mode Full `
  -ConfirmFinalTest `
  -Workspace C:\python\E2URec-main

# Only after an interruption or partial scheduled execution, with unchanged seal:
.\scripts\bota_if\run_ali_stage16_finaltest_v1.ps1 `
  -Mode Resume `
  -ConfirmFinalTest `
  -Workspace C:\python\E2URec-main

# CPU recomputation from saved predictions; never reconstructs raw test rows:
.\scripts\bota_if\run_ali_stage16_finaltest_v1.ps1 -Mode Analyze
```

Optional scheduling: start `Full -Lane binary` and later `Resume -Lane ranking`;
or filter by `-Seed 41`, then `Resume -Seed All` to finish remaining cells.
These options do not select which results appear in the final report.

Outputs: `C:\python\E2URec-main\outputs\ali_stage16_finaltest_v1`.
`summary/report.md` is the readable report. `summary/summary.json` and CSVs
contain all cells, three-seed summaries, BOTA64-vs-0/1 absolute/relative gains,
and paired ranking deltas. `ranking_per_user.json` retains per-user metrics
and requested-user coverage. `cohort/candidates.json` records exclusions.

This evaluation does **not** update latency, attacks, long-horizon, sequential
deletion or storage experiments. Do not rewrite those results as lambda=64
results. Review the complete report before updating manuscript tables/claims.

## Local validation before sealing

Validation includes a synthetic end-to-end 144-cell aggregation and independent
summary regeneration; immutable-seal and confirmation guards; distinct P5 and
Exact-Masked denominators; missing-user reporting; ranking cutoff/exclusions;
and malformed prediction rejection. A bounded GPU smoke scores only four
existing Development candidates through BOTA0, BOTA64 and RecEraser and checks
agreement with the Stage 15 scorer. No FinalTest loader is allowed in that smoke.

Tests are `tests/test_ali_stage16_finaltest_v1.py` and the opt-in
`tests/test_ali_stage16_finaltest_integration_v1.py` (environment switch
`BOTA_STAGE16_GPU_SMOKE=1`). `SyntheticDryRun` requires no GPU or FinalTest.
