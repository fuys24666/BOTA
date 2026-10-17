param(
  [ValidateSet("Preflight", "Freeze", "Smoke", "Full", "Analyze")]
  [string]$Mode = "Preflight",
  [ValidateSet("All", "41", "42", "43")]
  [string]$Seed = "All",
  [string]$Workspace = "C:\python\E2URec-main"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace = (Resolve-Path $Workspace).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
Push-Location $Repo
try {
  & $Python -u -m src.bota_short_benchmark.ali_stage19_amazon_k2_requests_v1 --repo $Repo --workspace $Workspace --mode $Mode --seed $Seed
  if ($LASTEXITCODE -ne 0) { throw "Stage19 $Mode failed. Completed verified units are reusable; rerun the same command. Failed diagnostics remain in .work." }
} finally { Pop-Location }
