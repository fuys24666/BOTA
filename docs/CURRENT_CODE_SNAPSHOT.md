# Reproducing the manuscript results

The November 24, 2026 manuscript tables were exported from the archived runs
of the October 17 Stage 27 implementation. The [result index](../results/README.md)
lists their units, cohorts and source files.

## Current result routes

| Evidence | Current route | Prerequisite |
|---|---|---|
| ML-1M main, source-only, P5 and ranking | Stage 25; loader-repair wrapper for FinalTest | Stage 16/17 endpoints and Stage 25 inventory/plan |
| Stronger requests, cumulative deletion, horizons, attacks, compression behavior | Stage 26B focused profile | Stage 17 artifacts and Stage 25 references |
| SVD, parameter-space errors, storage, independent timings | Frozen Stage 11/17 records where applicable | Original banks, protocols and measurements |
| Amazon six-cell comparison | Stage 25 plus Stage 27 selected RecEraser completion | Stage 20/21 selections, models, Stage 25 parent scores |

Stage 26B deliberately omits stress-specific ranking and several non-primary
arms; it does not restore those claims. The broader Stage 26 plan is separate.
Old Stage 16/17 behavior scores and the old single-request Amazon table are
not current headline results. Historical audits remain timestamped records.

## Safe entry-point checks (no training)

From the repository root in the configured environment:

```powershell
$ExperimentWorkspace = (Resolve-Path '..\E2URec-main').Path
.\scripts\bota_if\run_ali_stage25_historical_mask_audit_v1.ps1 `
  -Mode Preflight -Scope Development -Workspace $ExperimentWorkspace
.\scripts\bota_if\run_ali_stage26b_focused_rescore_v1.ps1 `
  -Mode Preflight -Workspace $ExperimentWorkspace
.\scripts\bota_if\run_ali_stage27_amazon_receraser_rescore_v1.ps1 `
  -Mode Preflight -Workspace $ExperimentWorkspace
```

The example assumes the experiment workspace is beside the BOTA checkout. If it
is elsewhere, set `$ExperimentWorkspace` to that directory. Preflight can write
inventory/audit reports in the workspace but does not train.
Read each runbook before `Full`. FinalTest uses
`run_ali_stage25_loader_repair01.ps1 -Scope FinalTest -ConfirmFinalTest` with
the appropriate mode, not a patch to the diagnostics loader's general test guard.
Do not initiate FinalTest merely to check installation.

Current scoring freezes lambda=64 (ML-1M) and lambda=48 (Amazon); lambda=0 is
the source-only ablation. Rescoring does not authorize retuning on FinalTest.
Amazon is auxiliary reused-Development evidence. Retain failure/ineligibility
records rather than selecting successful cells.

## Required external artifacts

- Dataset files and frozen preprocessing; T5 tokenizer/backbone.
- Initial recommendation checkpoints and fixed LoRA coordinates.
- Parent `outputs/.../protocol/` contracts and selected-configuration records.
- Frozen models/banks and caches explicitly reused by a stage.

See [ARTIFACTS.md](ARTIFACTS.md), the configs and individual runbooks. The
aggregate measurements in `results/` can be inspected directly. Rescoring the
archived endpoints requires the original experiment workspace.
A reproduction with different asset hashes needs its own declared protocol;
do not weaken hash guards to claim identity with the archived experiment.

## Source packaging and Git upload

`scripts/release/prepare_source_snapshot.py` inventories tracked and non-ignored
untracked files, rejects unexpected file classes, scans common credential
patterns and checks outgoing objects against cached `origin/main`. It does not
contact GitHub. This is not a complete privacy audit; metadata-selected request
identifiers and scientific summaries are intentional reproducibility inputs,
not raw data releases.

The ZIP preserves bytes. `.gitattributes` disables automatic newline conversion
to preserve source hashes. Previously tracked files may show line-ending-only
changes when staged; inspect the normal diff and
`git diff --cached --ignore-space-at-eol`.

The generated `git-paths.txt` is an explicit staging list. Follow the generated
upload guide: inspect, stage, commit, fetch, check divergence, then push. Do not
force push or rewrite history. The ZIP excludes `.git`. Confirm visibility on
GitHub before sharing. No staging, commits or push are performed by packaging.
