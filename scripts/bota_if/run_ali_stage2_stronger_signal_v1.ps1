param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Analyze")]
  [string]$Mode="Preflight",
  [string]$Workspace="C:\python\E2URec-main",
  [string]$RunName="ali_stage2_stronger_signal_seed42_v1"
)
$ErrorActionPreference="Stop"
$Repo=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace=(Resolve-Path $Workspace).Path
$Python=if($env:BOTA_PYTHON){$env:BOTA_PYTHON}else{"python"}
$Config="configs/ali_stage2_stronger_signal_v1.yaml"

if($Mode -eq "Full") {
  & $Python scripts/bota_if/validate_ali_stage0_amendment01_v1.py --repo $Repo
  if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
}

& $Python -m src.bota_short_benchmark.ali_stage2_stronger_signal_v1 `
  --repo $Repo --workspace $Workspace --config $Config `
  --mode $Mode --run-name $RunName
exit $LASTEXITCODE
