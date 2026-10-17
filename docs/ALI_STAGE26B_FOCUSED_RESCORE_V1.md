# Stage 26B — reduced frozen stress validation

User-approved compute reduction, defined before new scoring. The original Stage26
protocol and output directory remain unchanged. This profile inherits its exact
requests, model identities, Development rows, attack cohorts and audit tolerances.

## Kept

- Frozen BOTA lambda=64, Original, Exact-Masked. Compression also retains the
  corresponding uncompressed FS64 reference.
- Three seeds and all original groups: 6 primary, 9 stronger, 39 distinct
  sequential unions, 6 horizon, 72 compression, 1 historical relay (133 total).
- Full 20,000-example Development binary evaluation. Same reconstruction,
  binary utility metrics and bootstrap rules.
- All six primary + all nine stronger-request attack groups (three models).
  Only high_k4 permits formal stronger-request interpretation, subject to the
  original reference-signal and improvement gates.
- All 60 sequential order/size records, expanded from identical union caches.
- All twelve compression representations, not a success-selected subset.
- Existing parameter/SVD/storage/latency artifacts: verify and reuse, no rerun.

## Excluded prospectively to reduce compute

- All 261 ranking inference jobs. No stress-specific NDCG/Recall conclusion.
- Other baselines, FM and source-only rescoring: already validated main results
  remain separate; this run does not restore unvalidated FM behavioral claims.
- Sequential/horizon MIA. No new claims about their privacy behavior.
- No training, transport construction, retuning, bank reconstruction or FinalTest.

## Run

From `C:\python\BOTA` in the activated pytorch environment:

```powershell
.\scripts\bota_if\run_ali_stage26b_focused_rescore_v1.ps1 `
  -Mode Full `
  -Workspace C:\python\E2URec-main
```

Use the **26b_focused** script, not the old full Stage26 script. Full automatically
freezes this independent profile. Stop with Ctrl+C and rerun the same command to
resume verified completed units; the interrupted unit restarts. Hash or batch
invariance failure blocks execution and must not be bypassed. Missing weights
are explicit and never trigger automatic reconstruction.

Optional `-Mode Preflight`, `Freeze`, `Smoke`, `Analyze`; optional `-Family`
primary/stronger/sequential/horizon/compression/relay. The default Full runs all
remaining focused jobs. Smoke checks 64 examples in binary and attack lanes only,
without publishing full prediction caches.

## Output and time

Independent output root:
`C:\python\E2URec-main\outputs\ali_stage26b_focused_rescore_v1`

- `audit/`: scope, model inventory, unavailable weights, reused records and ETA.
- `predictions/`: masked inference and invariance checks; atomic resume units.
- `metrics/`: all group-specific results.
- `summary/All/`: behavior.csv, attacks.json, all_results.json,
  sequential_all_cells.json, REPORT.md. No ranking table is produced.

Expected workload: 204 binary + 45 attack jobs, zero ranking jobs. Same-machine
historical estimate: approximately 7.8h binary inference, plus attack, auditing,
and CPU bootstrap work; allow roughly **9–12h**, not a completion guarantee.
No complete long run is started during code preparation.
