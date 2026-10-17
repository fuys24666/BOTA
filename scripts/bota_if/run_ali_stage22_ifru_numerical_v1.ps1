param(
  [ValidateSet("Preflight", "Freeze", "Smoke", "Full", "Analyze")]
  [string]$Mode = "Preflight",
  [ValidateSet("All", "A1", "A2", "A3")]
  [string]$Request = "All",
  [string]$Workspace = "C:\python\E2URec-main"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace = (Resolve-Path $Workspace).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
Push-Location $Repo
try {
  & $Python -u -m src.bota_short_benchmark.ali_stage22_ifru_numerical_v1 --repo $Repo --workspace $Workspace --mode $Mode --request $Request
  if ($LASTEXITCODE -ne 0) { throw "Stage22 $Mode/$Request failed. Rerun the same command: verified completed units are reused; the interrupted unit restarts. Historical Stage21 is untouched." }
} finally { Pop-Location }
