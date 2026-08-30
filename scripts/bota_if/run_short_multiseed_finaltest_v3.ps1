param(
  [ValidateSet("Preflight","RecoveryPreflight","Full","RecoverZeroInference","Analyze")][string]$Mode="Preflight",
  [string]$DevelopmentRunSeed41="bota_short_development_seed41_v3",
  [string]$DevelopmentRunSeed42="bota_short_development_seed42_v3",
  [string]$DevelopmentRunSeed43="bota_short_development_seed43_v3",
  [string]$RunName="",
  [string]$FailedBinding="",
  [string]$FailedRunName="",
  [switch]$ConfirmFinalTest
)
$ErrorActionPreference="Stop"
$Root=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$Args=@("-m","src.bota_short_benchmark.multiseed_finaltest_v3","--root",$Root,"--mode",$Mode,"--development-run-seed41",$DevelopmentRunSeed41,"--development-run-seed42",$DevelopmentRunSeed42,"--development-run-seed43",$DevelopmentRunSeed43)
if($RunName) {$Args+=@("--run-name",$RunName)}
if($FailedBinding) {$Args+=@("--failed-binding",$FailedBinding)}
if($FailedRunName) {$Args+=@("--failed-run-name",$FailedRunName)}
if($ConfirmFinalTest) {$Args+="--confirm-final-test"}
& $Python @Args
exit $LASTEXITCODE
