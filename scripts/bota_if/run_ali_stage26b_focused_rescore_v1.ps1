param(
  [ValidateSet("Preflight", "Freeze", "Smoke", "Full", "Analyze")][string]$Mode = "Preflight",
  [ValidateSet("All", "primary", "stronger", "sequential", "horizon", "compression", "relay")][string]$Family = "All",
  [string]$Workspace = "C:\python\E2URec-main"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace = (Resolve-Path $Workspace).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$Stage26BArgs = @("-u", "-m", "src.bota_short_benchmark.ali_stage26b_focused_rescore_v1", "--repo", $Repo, "--workspace", $Workspace, "--mode", $Mode, "--family", $Family)
Push-Location $Repo
try {
  & $Python @Stage26BArgs
  if ($LASTEXITCODE -ne 0) { throw "Stage26B failed. Rerun the SAME focused command to resume completed units. Never bypass hash or invariance failures. No automatic retraining." }
} finally { Pop-Location }
