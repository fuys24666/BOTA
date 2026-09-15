# Ali Stage 9 cross-seed Development confirmation

Stage 9 evaluates the already frozen BOTA and baseline settings on held-out
model seeds 41 and 43. The existing seed-42 selection cells are imported only
for the three-seed report. FinalTest is never opened.

Run from `C:\python\BOTA` in the `pytorch` environment:

```powershell
.\scripts\bota_if\run_ali_stage9_crossseed_development_v1.ps1 `
  -Mode Full `
  -Seed All `
  -Method All `
  -Workspace C:\python\E2URec-main
```

The launcher is resumable: an already completed method/seed run is analyzed
instead of overwritten. After all twelve held-out runs, it automatically
creates the three-seed summary and applies the frozen stability rule.

To verify and summarize completed runs without training:

```powershell
.\scripts\bota_if\run_ali_stage9_crossseed_development_v1.ps1 `
  -Mode Analyze `
  -Seed All `
  -Method All `
  -Workspace C:\python\E2URec-main
```

If the final status is `STABLE_CONFIRMED`, stop there. Do not open FinalTest.
The next step is a new amendment that preregisters exactly one new FinalTest
evaluation before any test access. If the status is
`NOT_STABLE_NO_FINALTEST_AUTHORIZATION`, the protocol does not authorize that
evaluation.
