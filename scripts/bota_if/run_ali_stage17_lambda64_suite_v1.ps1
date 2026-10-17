param(
 [ValidateSet('Preflight','Seal','Full','Analyze')][string]$Mode='Preflight',
 [ValidateSet('All','Primary','Stronger','Sequential','Horizon','Storage','StorageBehavior','Latency','LegacyRelay','LegacyPosition')][string]$Phase='All',
 [ValidateSet('All','41','42','43')][string]$Seed='All',
 [string]$Workspace='C:\python\E2URec-main'
)
$ErrorActionPreference='Stop'
$Repo=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Workspace=(Resolve-Path $Workspace).Path
$Python=if ($env:BOTA_PYTHON) {$env:BOTA_PYTHON} else {'python'}
Push-Location $Repo
try {
 & $Python -u -m src.bota_short_benchmark.ali_stage17_lambda64_suite_v1 --repo $Repo --workspace $Workspace --mode $Mode --phase $Phase --seed $Seed
 if ($LASTEXITCODE -ne 0) {throw "Stage17 failed. Keep all artifacts; rerun the identical Full command to resume after resolving infrastructure issues. Do not edit sealed code or relabel failed cells."}
} finally {Pop-Location}
