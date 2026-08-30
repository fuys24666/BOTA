param(
  [ValidateSet("Preflight", "SyntheticDryRun", "BuildBank", "Full", "AnalyzeBank", "Analyze")][string]$Mode = "Preflight",
  [string]$BankName = "",
  [string]$RunName = ""
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$Arguments = @("-m", "src.bota_short_benchmark.online_latency_v1", "--root", $Root, "--config", "configs/bota_short_online_latency_v1.yaml", "--mode", $Mode)
if ($BankName) { $Arguments += @("--bank-name", $BankName) }
if ($RunName) { $Arguments += @("--run-name", $RunName) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
