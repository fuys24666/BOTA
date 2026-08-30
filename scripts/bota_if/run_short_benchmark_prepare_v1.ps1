param(
  [ValidateSet("Prepare","Preflight","SyntheticDryRun")][string]$Mode = "Preflight",
  [Parameter(Mandatory=$true)][string]$BenchmarkName
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
& $Python -m src.bota_short_benchmark.runner --root $ProjectRoot --config configs/bota_short_benchmark_v1.yaml --mode $Mode --method Original --benchmark-name $BenchmarkName
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
