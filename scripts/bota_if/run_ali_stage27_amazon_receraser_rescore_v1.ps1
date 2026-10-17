param(
  [ValidateSet("Preflight", "Freeze", "Full", "Analyze")][string]$Mode = "Preflight",
  [ValidateSet("All", "42", "43")][string]$Seed = "All",
  [switch]$RebuildMissing,
  [string]$Workspace = "C:\python\E2URec-main"
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace = (Resolve-Path $Workspace).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$Stage27Args = @("-u", "-m", "src.bota_short_benchmark.ali_stage27_amazon_receraser_rescore_v1", "--repo", $Repo, "--workspace", $Workspace, "--mode", $Mode, "--seed", $Seed)
if ($RebuildMissing) { $Stage27Args += "--rebuild-missing" }
Push-Location $Repo
try {
  & $Python @Stage27Args
  if ($LASTEXITCODE -ne 0) { throw "Stage27 failed. No automatic reselection. Completed cells and shard checkpoints are retained. Exact model/aggregation mismatches require diagnosis; do not bypass them." }
} finally { Pop-Location }
