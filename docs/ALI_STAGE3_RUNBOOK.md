# Ali Stage 3: fixed-long-horizon equal-billing runbook

Stage 3 is a Development-only secondary reference-sensitivity experiment. It compares `Original-Tlong`, `Exact-Masked-Tlong`, and the frozen `BOTA-FS-EmpiricalFisher-Tlong` endpoint under the same initialization, fixed LoRA-A coordinate, ordered sample slots, full-batch denominator, and exactly 1,200 optimizer steps.

## Why Amendment 02 exists

Amendment 01 defined `T_star` using the best FullControl-P5 Development checkpoint. The pre-execution audit found `best_epoch=1` for seeds 41, 42, and 43, so the literal value was only 200 steps and would duplicate the short-horizon study. All three existing FullControl-P5 runs stopped after epoch 6 and recorded 1,200 optimizer steps. Amendment 02 therefore freezes `T_long=1,200` for every seed before Stage 3 endpoints or predictions are generated. Amendment 01 remains unchanged.

## Formal run

From `C:\python\BOTA` in the existing `pytorch` environment:

```powershell
.\scripts\bota_if\run_ali_stage3_equal_horizon_v1.ps1 `
  -Mode Full `
  -Seed All `
  -Workspace C:\python\E2URec-main
```

The launcher validates Amendment 02, runs seeds 41, 42, and 43 sequentially, and then atomically publishes the six-cell summary.

## Recovery and inspection

Completed seed runs are immutable. If the full command stops after a completed seed, rerun only the missing seeds, then summarize:

```powershell
.\scripts\bota_if\run_ali_stage3_equal_horizon_v1.ps1 -Mode Full -Seed 42 -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage3_equal_horizon_v1.ps1 -Mode Full -Seed 43 -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage3_equal_horizon_v1.ps1 -Mode Summarize -Workspace C:\python\E2URec-main
```

Outputs are written below `C:\python\E2URec-main\outputs\ali_stage3_equal_horizon_v1`. Formal execution requires a clean `C:\python\BOTA` Git worktree, exactly one CUDA GPU, and at least 10 GiB free GPU memory. FinalTest is never constructed or read. Retain-Retrain-P5 is hash-bound for lineage but is not loaded and is not used as the local-residual denominator.
