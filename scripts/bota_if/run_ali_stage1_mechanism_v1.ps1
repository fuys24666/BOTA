param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Analyze","Summarize")]
  [string]$Mode="Preflight",
  [ValidateSet("All","41","42","43")]
  [string]$Seed="All",
  [ValidateSet("All","L8","L4M4")]
  [string]$Scenario="All",
  [string]$Workspace="C:\python\E2URec-main",
  [string]$RunName="",
  [string]$SummaryName="ali_stage1_mechanism_summary_v1"
)
$ErrorActionPreference="Stop"
$Repo=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace=(Resolve-Path $Workspace).Path
$Python=if($env:BOTA_PYTHON){$env:BOTA_PYTHON}else{"python"}
$Config="configs/ali_stage1_mechanism_v1.yaml"

if($Mode -eq "Summarize") {
  & $Python -m src.bota_short_benchmark.ali_stage1_mechanism_v1 `
    --repo $Repo --workspace $Workspace --config $Config `
    --mode Summarize --summary-name $SummaryName
  exit $LASTEXITCODE
}

if($Mode -eq "Full") {
  & $Python scripts/bota_if/validate_ali_stage0_amendment01_v1.py --repo $Repo
  if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
}

$Seeds=if($Seed -eq "All"){@(41,42,43)}else{@([int]$Seed)}
$Scenarios=if($Scenario -eq "All"){@("L8","L4M4")}else{@($Scenario)}
if($RunName -and ($Seeds.Count -ne 1 -or $Scenarios.Count -ne 1)) {
  throw "RunName may be supplied only for one seed and one scenario."
}

foreach($CurrentSeed in $Seeds) {
  foreach($CurrentScenario in $Scenarios) {
    $Arguments=@(
      "-m","src.bota_short_benchmark.ali_stage1_mechanism_v1",
      "--repo",$Repo,
      "--workspace",$Workspace,
      "--config",$Config,
      "--mode",$Mode,
      "--seed",$CurrentSeed,
      "--scenario",$CurrentScenario
    )
    if($RunName){$Arguments+=@("--run-name",$RunName)}
    & $Python @Arguments
    if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
  }
}

if($Mode -eq "Full" -and $Seed -eq "All" -and $Scenario -eq "All") {
  & $Python -m src.bota_short_benchmark.ali_stage1_mechanism_v1 `
    --repo $Repo --workspace $Workspace --config $Config `
    --mode Summarize --summary-name $SummaryName
  exit $LASTEXITCODE
}
