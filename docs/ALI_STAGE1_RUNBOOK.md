# Ali feedback experiments: Stage 1 mechanism stabilization

Stage 1 implements Amendment 01 without accessing FinalTest. It evaluates three
arms on Development for seeds 41, 42 and 43 and for L8 and L4M4:

- `FM_source_only`
- `FS_source_only`
- `FS_empirical_fisher`

The clean code repository is `C:\python\BOTA`. Existing models, data and new
outputs remain in the independent experiment workspace
`C:\python\E2URec-main`.

## Full launch

From PowerShell:

```powershell
cd C:\python\BOTA
$env:BOTA_PYTHON = "C:\Users\Administrator\anaconda3\envs\pytorch\python.exe"
.\scripts\bota_if\run_ali_stage1_mechanism_v1.ps1 `
  -Mode Full `
  -Seed All `
  -Scenario All `
  -Workspace C:\python\E2URec-main
```

The launcher first validates Amendment 01, then executes the six seed-scenario
runs serially. When all six complete, it automatically produces the frozen
cross-seed selection summary.

## Individual recovery

Completed runs are never overwritten. If a run fails, rerun only the missing
seed and scenario, for example:

```powershell
.\scripts\bota_if\run_ali_stage1_mechanism_v1.ps1 `
  -Mode Full -Seed 43 -Scenario L4M4 `
  -Workspace C:\python\E2URec-main
```

After all six runs exist, create the summary separately:

```powershell
.\scripts\bota_if\run_ali_stage1_mechanism_v1.ps1 `
  -Mode Summarize `
  -Workspace C:\python\E2URec-main
```

## Output locations

Per-run artifacts:

```text
C:\python\E2URec-main\outputs\ali_stage1_mechanism_v1\runs\
```

Frozen selection summary:

```text
C:\python\E2URec-main\outputs\ali_stage1_mechanism_v1\summaries\ali_stage1_mechanism_summary_v1\
```

The authoritative human-readable result is `summary.md`; `summary.json` contains
the complete machine-readable aggregates and decision record.

## Integrity behavior

- The BOTA code repository must be clean for a formal run.
- The experiment workspace may remain dirty; its Git state is recorded but is
  not used as the code authority.
- The FS+Fisher reconstruction must exactly reproduce the existing formal BOTA
  adapter or the run aborts before publication.
- All outputs are staged and atomically published only after integrity checks.
- The launcher and evaluator do not construct or access a FinalTest loader.
