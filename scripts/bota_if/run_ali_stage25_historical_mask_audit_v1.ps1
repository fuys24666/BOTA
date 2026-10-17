param(
  [ValidateSet("Preflight", "Freeze", "Full", "Analyze")][string]$Mode = "Preflight",
  [ValidateSet("Development", "FinalTest")][string]$Scope = "Development",
  [ValidateSet("All", "binary", "ranking")][string]$Lane = "All",
  [switch]$ConfirmFinalTest,
  [string]$Workspace = "C:\python\E2URec-main"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace = (Resolve-Path $Workspace).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$Stage25Args = @("-u", "-m", "src.bota_short_benchmark.ali_stage25_historical_mask_audit_v1", "--repo", $Repo, "--workspace", $Workspace, "--mode", $Mode, "--scope", $Scope, "--lane", $Lane)
if ($ConfirmFinalTest) { $Stage25Args += "--confirm-finaltest" }
Push-Location $Repo
try {
  & $Python @Stage25Args
  if ($LASTEXITCODE -ne 0) { throw "Stage25 failed. Completed cells can resume with the same command; do not bypass audit failures. Missing weights are reported, never retrained automatically." }
} finally { Pop-Location }
