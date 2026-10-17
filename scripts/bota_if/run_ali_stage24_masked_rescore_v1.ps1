param(
  [ValidateSet("Preflight", "Freeze", "Full", "Analyze")]
  [string]$Mode = "Preflight",
  [ValidateSet("All", "Amazon", "ML1M")]
  [string]$Dataset = "All",
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
  & $Python -u -m src.bota_short_benchmark.ali_stage24_masked_rescore_v1 --repo $Repo --workspace $Workspace --mode $Mode --dataset $Dataset --seed $Seed
  if ($LASTEXITCODE -ne 0) { throw "Stage24 failed. Completed units are reusable; interrupted units restart. Invariance failures stop scoring and must be diagnosed, not bypassed. Stage23 is unchanged." }
} finally { Pop-Location }
