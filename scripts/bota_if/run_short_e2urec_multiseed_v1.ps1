param(
  [ValidateSet("Preflight","Full","EvaluateDevelopment","SupplementalFinalTestPreflight","EvaluateSupplementalFinalTest","Analyze","AnalyzeSupplementalFinalTest")][string]$Mode="Preflight",
  [ValidateSet(0,41,42,43)][int]$Seed=0,
  [ValidateSet("All","L8","L4M4")][string]$Scenario="All",
  [string]$RunName="",
  [string]$DevelopmentRunName="bota_short_e2urec_multiseed_development_v1",
  [string]$PrimaryFinalTestRunName="bota_short_multiseed_finaltest_ml1m_seed41_43_v3_recovery1",
  [switch]$ConfirmSupplementalFinalTest
)
$ErrorActionPreference="Stop"
$Root=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$Arguments=@("-m","src.bota_short_benchmark.e2urec_short_v1","--root",$Root,"--mode",$Mode,"--scenario",$Scenario)
if($Seed -ne 0){$Arguments+=@("--seed",$Seed)}
if($RunName){$Arguments+=@("--run-name",$RunName)}
if($DevelopmentRunName){$Arguments+=@("--development-run-name",$DevelopmentRunName)}
if($PrimaryFinalTestRunName){$Arguments+=@("--primary-finaltest-run-name",$PrimaryFinalTestRunName)}
if($ConfirmSupplementalFinalTest){$Arguments+=@("--confirm-supplemental-final-test")}
& $Python @Arguments
exit $LASTEXITCODE
