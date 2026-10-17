param(
  [ValidateSet("Preflight", "Seal", "SyntheticDryRun", "Full", "Resume", "Analyze")]
  [string]$Mode = "Preflight",
  [switch]$ConfirmFinalTest,
  [ValidateSet("All", "binary", "ranking")]
  [string]$Lane = "All",
  [ValidateSet("All", "41", "42", "43")]
  [string]$Seed = "All",
  [string]$Workspace = "C:\python\E2URec-main"
)
$ErrorActionPreference = "Stop"
if (($Mode -in @("Full", "Resume")) -and -not $ConfirmFinalTest) {
  throw "Full/Resume requires -ConfirmFinalTest. Preflight and Seal do not access FinalTest."
}
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace = (Resolve-Path $Workspace).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$RunArgs = @("-u", "-m", "src.bota_short_benchmark.ali_stage16_finaltest_v1", "--repo", $Repo, "--workspace", $Workspace, "--mode", $Mode, "--seed", $Seed, "--lane", $Lane)
if ($ConfirmFinalTest) { $RunArgs += "--confirm-final-test" }
Push-Location $Repo
try {
  & $Python @RunArgs
  if ($LASTEXITCODE -ne 0) { throw "Stage 16 $Mode failed (exit $LASTEXITCODE). Preserve the seal and access ledger. Use explicit Resume only with unchanged code/configuration; completed cells and diagnostics are retained." }
} finally { Pop-Location }
