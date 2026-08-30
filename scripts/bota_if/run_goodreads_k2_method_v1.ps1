param(
  [ValidateSet("Preflight", "SyntheticDryRun", "Full", "Analyze")][string]$Mode = "Preflight",
  [ValidateSet("Original", "Retrain", "IFRU", "SISA", "RecEraser", "BOTA")][Parameter(Mandatory = $true)][string]$Method,
  [string]$BenchmarkName = "goodreads_k2_short_seed42_v1",
  [string]$RunName = ""
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
& $Python -m src.bota_short_benchmark.runner --root $Root --config configs/bota_short_goodreads_k2_v1.yaml --mode $Mode --method $Method --benchmark-name $BenchmarkName --scenario K2 --run-name $RunName
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
