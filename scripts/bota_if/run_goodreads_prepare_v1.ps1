param([ValidateSet("Preflight","SyntheticDryRun","Prepare","Analyze")][string]$Mode="Preflight",[Parameter(Mandatory=$true)][string]$DatasetName)
$ErrorActionPreference="Stop";$Root=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path;$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
& $Python -m src.bota_short_benchmark.goodreads_prepare --root $Root --config configs/goodreads_comics_bota_v1.yaml --mode $Mode --dataset-name $DatasetName
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
