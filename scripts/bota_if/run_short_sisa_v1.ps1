param([ValidateSet("Preflight","SyntheticDryRun","Full","Analyze")][string]$Mode="Preflight",[Parameter(Mandatory=$true)][string]$BenchmarkName,[ValidateSet("All","L8","L4M4","L3M3H2")][string]$Scenario="All",[string]$RunName="")
$ErrorActionPreference="Stop";$Root=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path;$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
& $Python -m src.bota_short_benchmark.runner --root $Root --config configs/bota_short_benchmark_v1.yaml --mode $Mode --method SISA --benchmark-name $BenchmarkName --scenario $Scenario --run-name $RunName
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
