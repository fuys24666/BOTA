param(
  [ValidateSet("Preflight", "SyntheticDryRun", "Full", "Analyze")][string]$Mode = "Preflight",
  [ValidateSet("FullControlP5", "RetainP5", "NegGrad", "PCGrad")][Parameter(Mandatory = $true)][string]$Method,
  [string]$BenchmarkName = "goodreads_k2_short_seed42_v1",
  [string]$SourceOriginalRunName = "",
  [string]$RunName = ""
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$Arguments = @("-m", "src.bota_short_benchmark.paper_v2", "--root", $Root, "--config", "configs/bota_short_paper_goodreads_k2_v2.yaml", "--mode", $Mode, "--method", $Method, "--benchmark-name", $BenchmarkName, "--scenario", "K2")
if ($SourceOriginalRunName) { $Arguments += @("--source-original-run-name", $SourceOriginalRunName) }
if ($RunName) { $Arguments += @("--run-name", $RunName) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
