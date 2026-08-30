param(
  [ValidateSet("Preflight", "SyntheticDryRun", "Full", "Analyze")][string]$Mode = "Preflight",
  [string]$BenchmarkName = "goodreads_k2_short_seed42_v1",
  [string]$OriginalRunName = "",
  [string]$ExactMaskedRunName = "",
  [string]$BOTARunName = "",
  [string]$IFRURunName = "",
  [string]$SISARunName = "",
  [string]$RecEraserRunName = "",
  [string]$RunName = ""
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$Arguments = @("-m", "src.bota_short_benchmark.evaluation", "--root", $Root, "--config", "configs/bota_short_goodreads_k2_v1.yaml", "--mode", $Mode, "--benchmark-name", $BenchmarkName, "--run-name", $RunName)
if ($OriginalRunName) { $Arguments += @("--original-run-name", $OriginalRunName) }
if ($ExactMaskedRunName) { $Arguments += @("--retrain-run-name", $ExactMaskedRunName) }
if ($BOTARunName) { $Arguments += @("--bota-run-name", $BOTARunName) }
if ($IFRURunName) { $Arguments += @("--ifru-run-name", $IFRURunName) }
if ($SISARunName) { $Arguments += @("--sisa-run-name", $SISARunName) }
if ($RecEraserRunName) { $Arguments += @("--receraser-run-name", $RecEraserRunName) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
