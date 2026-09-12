# Ali Stage 5 sequential cumulative-union experiment

Stage 5 evaluates cumulative registered-user deletion at request sizes 1, 2, 4
and 8. Every endpoint is recomposed from the same canonical 200-step endpoint;
the previously published Adapter is never used as the next source. Each unique
user union receives a paired Exact-Masked replay with the unchanged schedule and
full batch denominator.

The primary matrix contains four L4M4 order families over seeds 41--43. L8 under
deterministic-hash order is a secondary homogeneous low-frequency sequence. All
60 cells are reported. MIA is retained as a descriptive diagnostic only because
the Stage 4 and Stage 4A reference-signal gates failed.

Run from `C:\python\BOTA` in the `pytorch` environment:

```powershell
.\scripts\bota_if\run_ali_stage5_sequential_v1.ps1 `
  -Mode Full `
  -Seed All `
  -Workspace C:\python\E2URec-main
```
