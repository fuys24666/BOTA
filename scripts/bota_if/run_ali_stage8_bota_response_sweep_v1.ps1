param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Analyze")]
  [string]$Mode="Preflight",
  [string]$Workspace="C:\python\E2URec-main",
  [string]$RunName="ali_stage8_bota_response_seed42_v1"
)
$ErrorActionPreference="Stop"
$Repo=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace=(Resolve-Path $Workspace).Path
$Python=if($env:BOTA_PYTHON){$env:BOTA_PYTHON}else{"python"}
& $Python -m src.bota_short_benchmark.ali_stage8_bota_response_sweep_v1 `
  --repo $Repo --workspace $Workspace --mode $Mode --run-name $RunName
exit $LASTEXITCODE
