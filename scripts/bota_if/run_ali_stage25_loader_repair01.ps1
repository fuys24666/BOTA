param(
  [ValidateSet('Preflight', 'Full', 'Analyze')][string]$Mode = 'Preflight',
  [ValidateSet('Development', 'FinalTest')][string]$Scope = 'Development',
  [ValidateSet('All', 'binary', 'ranking')][string]$Lane = 'All',
  [switch]$ConfirmFinalTest,
  [string]$Workspace = 'C:\python\E2URec-main'
)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Workspace = (Resolve-Path $Workspace).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { 'python' }
$RepairArgs = @('-u', '-m', 'src.bota_short_benchmark.ali_stage25_finaltest_loader_repair01', '--repo', $Repo, '--workspace', $Workspace, '--mode', $Mode, '--scope', $Scope, '--lane', $Lane)
if ($ConfirmFinalTest) { $RepairArgs += '--confirm-finaltest' }
Push-Location $Repo
try {
  & $Python @RepairArgs
  if ($LASTEXITCODE -ne 0) { throw 'Stage25 loader repair failed; preserved completed cells can resume. Do not bypass audit failures.' }
} finally { Pop-Location }
