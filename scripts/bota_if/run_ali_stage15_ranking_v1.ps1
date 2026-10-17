param(
  [ValidateSet("Preflight", "SyntheticDryRun", "Full", "Analyze")]
  [string]$Mode = "Preflight",
  [ValidateSet("All", "41", "42", "43")]
  [string]$Seed = "All",
  [ValidateSet("All", "L8", "L4M4")]
  [string]$Scenario = "All",
  [ValidateSet("All", "Original", "Exact-Masked", "BOTA64", "BOTA1", "FullControl-P5", "Retain-Retrain-P5", "IFRU", "E2URec", "NegGrad", "PCGrad", "RecEraser")]
  [string]$Method = "All",
  [string]$Workspace = "C:\python\E2URec-main"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace = (Resolve-Path $Workspace).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
Push-Location $Repo
try {
  & $Python -u -m src.bota_short_benchmark.ali_stage15_ranking_v1 --repo $Repo --workspace $Workspace --mode $Mode --seed $Seed --scenario $Scenario --method $Method
  if ($LASTEXITCODE -ne 0) { throw "Stage 15 $Mode failed (exit $LASTEXITCODE). Completed endpoint cells remain reusable; incomplete diagnostics are retained in .work." }
} finally { Pop-Location }
