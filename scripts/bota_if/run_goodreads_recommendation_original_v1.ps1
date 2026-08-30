param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Resume","Analyze")]
  [string]$Mode="Preflight",
  [string]$RunName=""
)
$ErrorActionPreference="Stop"
$Root=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$Arguments=@(
  "-m","src.bota_short_benchmark.goodreads_original",
  "--root",$Root,
  "--config","configs/bota_goodreads_recommendation_original_v1.yaml",
  "--mode",$Mode
)
if($RunName){$Arguments+=@("--run-name",$RunName)}
& $Python @Arguments
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
