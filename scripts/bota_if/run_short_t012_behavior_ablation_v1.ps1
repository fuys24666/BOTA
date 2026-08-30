param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Analyze")][string]$Mode="Preflight",
  [string]$BenchmarkName="",
  [ValidateSet("L8","L4M4")][string]$Scenario="L8",
  [string]$OriginalRunName="",
  [string]$ExactMaskedRunName="",
  [Parameter(Mandatory=$true)][string]$RunName
)
$ErrorActionPreference="Stop"
$Root=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$Arguments=@(
  "-m", "src.bota_short_benchmark.t012_behavior_ablation",
  "--root", $Root,
  "--config", "configs/bota_short_t012_behavior_ablation_v1.yaml",
  "--mode", $Mode,
  "--run-name", $RunName
)
if($BenchmarkName){$Arguments += @("--benchmark-name", $BenchmarkName)}
if($Scenario){$Arguments += @("--scenario", $Scenario)}
if($OriginalRunName){$Arguments += @("--original-run-name", $OriginalRunName)}
if($ExactMaskedRunName){$Arguments += @("--exact-masked-run-name", $ExactMaskedRunName)}
& $Python @Arguments
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
