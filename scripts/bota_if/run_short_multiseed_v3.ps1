param(
  [ValidateSet("Prepare","Preflight","SyntheticDryRun","Full","Analyze")][string]$Mode="Preflight",
  [ValidateSet(41,42,43)][int]$Seed=42,
  [ValidateSet("Original","ExactMasked","BOTA","IFRU","SISA","RecEraser","FullControlP5","RetainP5","NegGrad","PCGrad")][string]$Method="Original",
  [ValidateSet("All","L8","L4M4")][string]$Scenario="All",
  [string]$RunName=""
)
$ErrorActionPreference="Stop"
$Root=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$CoreConfig="configs/bota_short_benchmark_seed${Seed}_v3.yaml"
$PaperConfig="configs/bota_short_paper_seed${Seed}_v3.yaml"
$BenchmarkName="bota_short_i02_seed${Seed}_v3"
$OriginalRunName="bota_short_original_seed${Seed}_v3"

if($Mode -eq "Prepare") {
  & $Python -m src.bota_short_benchmark.runner --root $Root --config $CoreConfig --mode Prepare --method Original --benchmark-name $BenchmarkName --scenario $Scenario
  exit $LASTEXITCODE
}

$CoreMethods=@{
  Original="Original"; ExactMasked="Retrain"; BOTA="BOTA"; IFRU="IFRU"; SISA="SISA"; RecEraser="RecEraser"
}
if($CoreMethods.ContainsKey($Method)) {
  $Args=@("-m","src.bota_short_benchmark.runner","--root",$Root,"--config",$CoreConfig,"--mode",$Mode,"--method",$CoreMethods[$Method],"--benchmark-name",$BenchmarkName,"--scenario",$Scenario)
} else {
  $Args=@("-m","src.bota_short_benchmark.paper_v2","--root",$Root,"--config",$PaperConfig,"--mode",$Mode,"--method",$Method,"--benchmark-name",$BenchmarkName,"--scenario",$Scenario)
  if($Method -in @("NegGrad","PCGrad")) {$Args+=@("--source-original-run-name",$OriginalRunName)}
}
if($RunName) {$Args+=@("--run-name",$RunName)}
& $Python @Args
exit $LASTEXITCODE
