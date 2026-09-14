# Ali Stage 8 BOTA response-strength sweep

Amendment 07 freezes this Development-only calibration after the six baseline
selections. The formal run evaluates five empirical-Fisher response strengths
jointly across L8 and L4M4. It never opens FinalTest.

Run from `C:\python\BOTA` in the `pytorch` environment:

```powershell
.\scripts\bota_if\run_ali_stage8_bota_response_sweep_v1.ps1 `
  -Mode Full `
  -Workspace C:\python\E2URec-main
```

The five response transports share each canonical trajectory. The run still
evaluates ten candidate/scenario adapters and can take substantial GPU time.
Do not change the grid or inspect FinalTest while it runs.

After completion:

```powershell
.\scripts\bota_if\run_ali_stage8_bota_response_sweep_v1.ps1 `
  -Mode Analyze `
  -Workspace C:\python\E2URec-main
```
