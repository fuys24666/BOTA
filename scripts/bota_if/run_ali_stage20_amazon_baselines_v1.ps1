param(
  [ValidateSet("Preflight", "Freeze", "Smoke", "Full", "Analyze")]
  [string]$Mode = "Preflight",
  [ValidateSet("All", "Calibration", "Comparison")]
  [string]$Phase = "All",
  [ValidateSet("All", "IFRU", "E2URec", "NegGrad", "PCGrad", "SISA", "RecEraser")]
  [string]$Method = "All",
  [string]$Workspace = "C:\python\E2URec-main"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace = (Resolve-Path $Workspace).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
Push-Location $Repo
try {
  & $Python -u -m src.bota_short_benchmark.ali_stage20_amazon_baselines_v1 --repo $Repo --workspace $Workspace --mode $Mode --phase $Phase --method $Method
  if ($LASTEXITCODE -ne 0) { throw "Stage20 $Mode/$Phase/$Method failed. Rerun the same command to reuse verified completed units. Interrupted partition training resumes at a completed shard. Diagnostics remain in .work." }
} finally { Pop-Location }
