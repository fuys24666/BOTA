# Ali Stage 4A stronger-signal attack

Stage 4A is a seed-42 secondary analysis using the nine frozen Stage 2 requests
and their existing Original, Exact-Masked and FS_empirical_fisher BOTA adapters.
It performs no training and never accesses FinalTest.

All nine requests and all four registered attacks are reported. The DU adequacy
rule was frozen before attack execution. Only high_k4 passed that rule and only
high_k4 may receive a formal reference-signal and BOTA decision. The other eight
requests remain descriptive even if their attack values look favourable. Stage
4A cannot replace the negative parent Stage 4 result.

Run from `C:\python\BOTA` in the `pytorch` environment:

```powershell
.\scripts\bota_if\run_ali_stage4a_stronger_signal_attack_v1.ps1 `
  -Mode Full `
  -Workspace C:\python\E2URec-main
```
