param(
  [ValidateSet("Preflight", "Freeze", "Smoke", "Full", "Analyze")]
  [string]$Mode = "Preflight",
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
  & $Python -u -m src.bota_short_benchmark.ali_stage21_amazon_seed43_v1 --repo $Repo --workspace $Workspace --mode $Mode --method $Method
  if ($LASTEXITCODE -ne 0) { throw "Stage21 $Mode/$Method failed. Rerun the same command to reuse verified completed units. Interrupted partition training resumes at completed shards; other incomplete units restart. No retuning." }
} finally { Pop-Location }
