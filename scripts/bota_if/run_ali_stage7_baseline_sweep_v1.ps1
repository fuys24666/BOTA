param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Analyze")]
  [string]$Mode="Preflight",
  [ValidateSet("NegGrad","PCGrad","IFRU","E2URec","SISA","RecEraser")]
  [string]$Method="NegGrad",
  [string]$Workspace="C:\python\E2URec-main",
  [string]$RunName=""
)
$ErrorActionPreference="Stop"
$Repo=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace=(Resolve-Path $Workspace).Path
$Python=if($env:BOTA_PYTHON){$env:BOTA_PYTHON}else{"python"}
if(-not $RunName){$RunName="ali_stage7a_$($Method.ToLower())_seed42_v1"}
& $Python -m src.bota_short_benchmark.ali_stage7_baseline_sweep_v1 `
  --repo $Repo --workspace $Workspace --mode $Mode --method $Method --run-name $RunName
exit $LASTEXITCODE
