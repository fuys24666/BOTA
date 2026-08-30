param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Analyze")]
  [string]$Mode="Preflight",
  [string]$RunName=""
)
$ErrorActionPreference="Stop"
$Root=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python=if($env:BOTA_PYTHON){$env:BOTA_PYTHON}else{"python"}
$Args=@("-m","src.bota_short_benchmark.fisher_ablation_v1","--root",$Root,"--config",(Join-Path $Root "configs/bota_short_fisher_ablation_v1.yaml"),"--mode",$Mode)
if($RunName){$Args+=@("--run-name",$RunName)}
& $Python @Args
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}

