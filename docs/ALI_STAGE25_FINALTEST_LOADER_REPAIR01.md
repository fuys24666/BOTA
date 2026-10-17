# Stage25 FinalTest loader repair 01

The original Stage25 binary dataset factory used the Development-only
`JsonPromptDataset` even after explicit FinalTest authorization. Its independent
path guard correctly rejected the test file before inference.

This additive entry point preserves the original frozen source/configuration,
protocol hash, cells and Development summary. For authorized FinalTest binary
cohorts only, it uses Stage16's existing capability-gated `BinaryDataset`.
The item encoding and collation methods are inherited unchanged. The frozen
data digest is mandatory; unauthorized access is rejected before file reading.
Ranking and Development retain the original loader. Scoring, masks, cohorts,
model weights, lambda, utility thresholds and selection rules do not change.

The new entry point validates the parent plan and Development completion, then
publishes a hashed repair receipt at `repairs/finaltest_loader01` under the existing
execution lock. Subsequent resumes reject changes to this repair or its launcher.
No original artifact is overwritten. An existing execution.lock file is normal:
the OS file lock, not file existence, determines occupancy.

Run from C:\python\BOTA in the pytorch environment:

```powershell
.\scripts\bota_if\run_ali_stage25_loader_repair01.ps1 `
  -Mode Full `
  -Scope FinalTest `
  -ConfirmFinalTest `
  -Workspace C:\python\E2URec-main
```

Resume with the same command. The old launcher is intentionally unchanged and
still has the loader defect; use the repair launcher for FinalTest.

Verification on 2026-10-15: 36 tests passed (repair, Stage25 and Stage24 suites);
parent preflight passed; explicit authorized real-data loader smoke check loaded
20,000 frozen rows and collated first/last samples without model inference,
training or metric evaluation. This is not a full FinalTest completion claim.
