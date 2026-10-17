param(
  [ValidateSet("Preflight", "SyntheticDryRun", "Full", "Analyze")]
  [string]$Mode = "Preflight",
  [ValidateSet("All", "Coarse", "Sweep", "Confirm")]
  [string]$Phase = "All",
  [string]$Workspace = "C:\python\E2URec-main"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace = (Resolve-Path $Workspace).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
Push-Location $Repo
try {
  & $Python -u -m src.bota_short_benchmark.ali_stage13_lambda_search_v1 --repo $Repo --workspace $Workspace --mode $Mode --phase $Phase
  if ($LASTEXITCODE -ne 0) { throw "Stage 13 $Mode/$Phase failed (exit $LASTEXITCODE). Completed cells are reusable; incomplete diagnostics remain under .work." }
} finally { Pop-Location }
