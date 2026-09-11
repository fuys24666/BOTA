# Ali feedback experiments: Stage 2 stronger-signal analysis

Stage 2 implements Amendment 01 A2. It is a Development-only secondary
analysis and cannot replace the parent L8/L4M4 scenarios. With selection seed
42, it constructs all nine combinations of request size `K=2,4,8` and window
exposure stratum `single`, `moderate`, and `high`. Users require at least ten
Development examples and are ordered within each stratum by the SHA256 of a
public user association. Labels and model predictions are not used to select
users.

The only endpoints are Original, a separately replayed 200-step Exact-Masked
counterfactual, and the Stage 1 frozen `FS_empirical_fisher` BOTA arm. Every
request is reported, including weak, null, or adverse results. An absolute
Original-to-Exact probability effect `DU >= 0.001` only permits attack
interpretation in this secondary request; it does not promote the request to a
primary result. No attack is executed in Stage 2.

## Full launch

```powershell
cd C:\python\BOTA
$env:BOTA_PYTHON = "C:\Users\Administrator\anaconda3\envs\pytorch\python.exe"
.\scripts\bota_if\run_ali_stage2_stronger_signal_v1.ps1 `
  -Mode Full `
  -Workspace C:\python\E2URec-main
```

The run performs one shared 200-step canonical/BOTA-bank pass and nine paired
200-step Exact-Masked replays. It then evaluates all three endpoints on the
complete Development split. Completed outputs are never overwritten.

## Preflight and recovery

```powershell
.\scripts\bota_if\run_ali_stage2_stronger_signal_v1.ps1 `
  -Mode Preflight -Workspace C:\python\E2URec-main

.\scripts\bota_if\run_ali_stage2_stronger_signal_v1.ps1 `
  -Mode Analyze -Workspace C:\python\E2URec-main
```

The formal result is written to:

```text
C:\python\E2URec-main\outputs\ali_stage2_stronger_signal_v1\runs\
  ali_stage2_stronger_signal_seed42_v1\
```

`report.md` is the human-readable result; `metrics.json` and `metrics.csv`
contain all nine request cells. Raw user identifiers, prompts, labels, and
tokens are not persisted.
