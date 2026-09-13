# Ali Stage 7 baseline Development sweep

Stage 7A runs one method at a time on seed 42. Every command uses the frozen
Dev-Fit/Dev-Select partition and forbids FinalTest. A formal run requires a
clean BOTA Git checkout so Amendment 06 and the implementation must be
committed first.

Set the repository Python explicitly if `python` is not the intended Conda
environment:

```powershell
$env:BOTA_PYTHON = "C:\Users\Administrator\anaconda3\envs\pytorch\python.exe"
```

Run lightweight validation first:

```powershell
.\scripts\bota_if\run_ali_stage7_baseline_sweep_v1.ps1 -Mode Preflight -Method NegGrad -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage7_baseline_sweep_v1.ps1 -Mode SyntheticDryRun -Method NegGrad -Workspace C:\python\E2URec-main
```

Recommended execution order, from shorter to longer:

```powershell
.\scripts\bota_if\run_ali_stage7_baseline_sweep_v1.ps1 -Mode Full -Method NegGrad -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage7_baseline_sweep_v1.ps1 -Mode Full -Method PCGrad -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage7_baseline_sweep_v1.ps1 -Mode Full -Method IFRU -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage7_baseline_sweep_v1.ps1 -Mode Full -Method E2URec -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage7_baseline_sweep_v1.ps1 -Mode Full -Method SISA -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage7_baseline_sweep_v1.ps1 -Mode Full -Method RecEraser -Workspace C:\python\E2URec-main
```

NegGrad and PCGrad should be run first because checkpoint reuse keeps them
comparatively short. IFRU repeatedly solves damped curvature systems. E2URec
trains one shared teacher and three student weight variants per scenario. SISA
and especially RecEraser are the longest because their multi-epoch shard
models must be trained and evaluated for every learning-rate candidate.

Each completed method publishes all candidate cells, a deterministic selection
decision and candidate-state hashes under
`outputs/ali_stage7_baseline_sweep_v1/runs`. Non-selected SISA and RecEraser
model tensors are removed only after their metrics and hashes have been
published; the selected ensemble remains available.
