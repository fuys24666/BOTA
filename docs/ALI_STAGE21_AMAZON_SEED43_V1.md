# Stage 21: Amazon seed43 frozen-parameter follow-up

Requested after reviewing Stage20 seed42. This is exploratory seed sensitivity,
not a prospectively selected independent validation or fresh FinalTest.

## Frozen scope

- Same A1/A2/A3, one low- and one middle-frequency user each, same local samples,
  Dev-Fit/Dev-Select split, historical pretrained checkpoint and 200-step order.
- Run seed / LoRA-A seed 43. Baseline minibatch RNG and partition/training seeds 43.
- BOTA lambda48 and source-only lambda0. No retuning of any method.
- IFRU panel1024/damping0.1; E2URec forget0.4/student500/teacher1200;
  NegGrad forget0.05/25steps; PCGrad lr0.0005/25steps;
  SISA lr0.0001/3epochs; RecEraser lr0.002/1epoch.
- PCGrad/SISA remain ineligible diagnostics inherited from calibration. All 30
  cells are reported; utility failures do not trigger replacement or retuning.
- A seed43 Original from Stage18 is reused with its actual fixed-A tensors.
  Rebuilt BOTA canonical parameters must reproduce that endpoint tensor-exactly.
- Seed42 and historical Stage18/19 results are immutable, not relabelled or omitted.

## Run (PowerShell, pytorch environment)

```powershell
.\scripts\bota_if\run_ali_stage21_amazon_seed43_v1.ps1 -Mode Preflight -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage21_amazon_seed43_v1.ps1 -Mode Full -Workspace C:\python\E2URec-main
```

`-Mode Analyze` only reports completed results or lists missing units. Optional
`-Mode Smoke` uses a separate output root and one optimizer step, not formal metrics.
`-Method IFRU` (or another method) runs only that baseline plus common references;
the default `All` performs all methods and creates the complete summary.

## Resume and output

Repeat the same Full command after interruption. Hash-verified completed units
are reused. Interrupted gradient/IFRU/BOTA units restart from their own beginning;
partition runs resume from completed shards with RNG checkpoints. This is not
arbitrary-step resume. A file lock prevents concurrent writers. Temporary work is
retained for diagnostics. Generated partition-model tensors may be cleaned only
after predictions, hashes and manifests are preserved, as in Stage20.

Formal output: `outputs/ali_stage21_amazon_seed43_v1` under the workspace.
Final artifacts: `summary/report.md`, `comparison.csv`, `result.json`, manifests.
No changes to Stage20 source, configuration, output or manuscript.

## Runtime estimate

Stage20 final-comparison wall times (not its calibration scan): IFRU 18.5min,
E2URec 34.0min, NegGrad+PCGrad 2.0min, SISA 17.2min, RecEraser 9.3min.
Add references, BOTA bank and input verification: roughly 1.5-2h on the same GPU,
with up to 2.5h as a conservative allowance for different CG/shard work or contention.

## Repair 01: explicit IFRU numerical rejection

The first Full run completed common references/BOTA, then IFRU A1 raised
`false_residual_convergence`: the recursively updated CG residual passed its
threshold but the recomputed true residual failed the existing guard. This is
not a BOTA failure. The log did not preserve the two residual magnitudes, so the
size and precise numerical cause of the mismatch cannot be inferred from it.

The solver, damping, panel, iteration cap and tolerances remain unchanged. If
the explicit rejection recurs, its cell is published as NUMERICAL_SOLVER_FAILURE
with no endpoint/predictions and no valid performance score. Other requests and
methods continue. Unexpected exceptions still stop execution. Incomplete IFRU
means become NA; rejected cells never count as BOTA wins. This is failure
reporting, not a successful numerical solution or a tolerance relaxation.

`configs/ali_stage21_repair01_v1.json` binds the exact before/after hashes of the
two changed Stage21 files and the repair validator. An additive implementation
repair record is written under the run root; original protocol files and already
published results are not rewritten. Repeat the original Full command. Common
references and BOTA are reused; the unfinished IFRU A1 computation starts again.
