param(
  [ValidateSet("Preflight", "Smoke", "Full", "Analyze")]
  [string]$Mode = "Preflight",
  [ValidateSet("All", "Coarse", "Sweep")]
  [string]$Phase = "All",
  [string]$Workspace = "C:\python\E2URec-main"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace = (Resolve-Path $Workspace).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
Push-Location $Repo
try {
  & $Python -u -m src.bota_short_benchmark.ali_stage18_amazon_lambda_search_v2 --repo $Repo --workspace $Workspace --mode $Mode --phase $Phase
  if ($LASTEXITCODE -ne 0) { throw "Stage 18 $Mode/$Phase failed. Rerun the same command to reuse verified completed units; incomplete diagnostics remain in .work." }
} finally { Pop-Location }
