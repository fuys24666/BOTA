param(
  [ValidateSet("Preflight", "Freeze", "Smoke", "Full", "Analyze")][string]$Mode = "Preflight",
  [ValidateSet("All", "primary", "stronger", "sequential", "horizon", "compression", "relay")][string]$Family = "All",
  [string]$Workspace = "C:\python\E2URec-main"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace = (Resolve-Path $Workspace).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$Stage26Args = @("-u", "-m", "src.bota_short_benchmark.ali_stage26_stress_mask_rescore_v1", "--repo", $Repo, "--workspace", $Workspace, "--mode", $Mode, "--family", $Family)
Push-Location $Repo
try {
  & $Python @Stage26Args
  if ($LASTEXITCODE -ne 0) { throw "Stage26 failed. Completed model/cohort units resume with the same command. Do not bypass hash or invariance failures. No model is retrained automatically." }
} finally { Pop-Location }
