param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Analyze")][string]$Mode="Preflight",
  [ValidateSet(41,42,43)][int]$Seed=42,
  [ValidateSet("All","L8","L4M4")][string]$Scenario="All",
  [string]$RunName=""
)
$ErrorActionPreference="Stop"
$Root=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }
$Config="configs/bota_short_paper_seed${Seed}_v3.yaml"
$BenchmarkName="bota_short_i02_seed${Seed}_v3"
$Args=@("-m","src.bota_short_benchmark.paper_evaluation_v2","--root",$Root,"--config",$Config,"--mode",$Mode,"--benchmark-name",$BenchmarkName,"--scenario",$Scenario)
$Names=@{
  "--original-run-name"="bota_short_original_seed${Seed}_v3"
  "--exact-masked-run-name"="bota_short_exact_masked_seed${Seed}_v3"
  "--full-control-p5-run-name"="bota_short_full_control_p5_seed${Seed}_v3"
  "--retain-p5-run-name"="bota_short_retain_p5_seed${Seed}_v3"
  "--bota-run-name"="bota_short_bota_seed${Seed}_v3"
  "--ifru-run-name"="bota_short_ifru_seed${Seed}_v3"
  "--neggrad-run-name"="bota_short_neggrad_seed${Seed}_v3"
  "--pcgrad-run-name"="bota_short_pcgrad_seed${Seed}_v3"
  "--sisa-run-name"="bota_short_sisa_seed${Seed}_v3"
  "--receraser-run-name"="bota_short_receraser_seed${Seed}_v3"
}
foreach($Key in $Names.Keys) {$Args+=@($Key,$Names[$Key])}
if($RunName) {$Args+=@("--run-name",$RunName)}
& $Python @Args
exit $LASTEXITCODE
