# Ali Stage 10 FinalTest re-evaluation

Run from `C:\python\BOTA` in the `pytorch` environment. This evaluates frozen
model artifacts; it does not train or update an optimizer.

Optional hash/CPU-coordinate preflight (does not open FinalTest):

```powershell
.\scripts\bota_if\run_ali_stage10_finaltest_v1.ps1 `
  -Mode Preflight `
  -Workspace C:\python\E2URec-main
```

Only after Amendment 09 and Stage 10 code are committed, perform the single
authorized evaluation:

```powershell
.\scripts\bota_if\run_ali_stage10_finaltest_v1.ps1 `
  -Mode Full `
  -Workspace C:\python\E2URec-main `
  -ConfirmFinalTest
```

This is one cohort-level run, not a separate manual run for each seed. All
three seeds and both scenarios are evaluated automatically. GPU inference
can take a substantial amount of time; keep the GPU dedicated to this run.

Do not rerun Full if it fails. Preserve the permanent access ledger and the
partial staging directory, then share the error for diagnosis. No automatic
retry/recovery is authorized. After completion, use Analyze without reopening
test data:

```powershell
.\scripts\bota_if\run_ali_stage10_finaltest_v1.ps1 `
  -Mode Analyze `
  -Workspace C:\python\E2URec-main
```

Output directory:
`C:\python\E2URec-main\outputs\ali_stage10_finaltest_v1\evaluations\ali_stage10_finaltest_v1`.

Historical test exposure must be disclosed: this is a post-tuning
re-evaluation of the previously used FinalTest cohort, not a fresh independent
holdout. Earlier long-horizon, attack and sequential limitations are unchanged.
