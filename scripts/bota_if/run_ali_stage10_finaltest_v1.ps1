param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Analyze")]
  [string]$Mode="Preflight",
  [string]$Workspace="C:\python\E2URec-main",
  [switch]$ConfirmFinalTest
)
$ErrorActionPreference="Stop"
$Repo=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace=(Resolve-Path $Workspace).Path
$Python=if($env:BOTA_PYTHON){$env:BOTA_PYTHON}else{"python"}
$Arguments=@("-m","src.bota_short_benchmark.ali_stage10_finaltest_v1","--repo",$Repo,"--workspace",$Workspace,"--mode",$Mode)
if($ConfirmFinalTest){$Arguments+="--confirm-final-test"}
& $Python @Arguments
exit $LASTEXITCODE
