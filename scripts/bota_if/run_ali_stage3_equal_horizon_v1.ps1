param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Analyze","Summarize")]
  [string]$Mode="Preflight",
  [ValidateSet("All","41","42","43")]
  [string]$Seed="All",
  [string]$Workspace="C:\python\E2URec-main",
  [string]$RunName="",
  [string]$SummaryName="ali_stage3_equal_horizon_summary_v1"
)
$ErrorActionPreference="Stop"
$Repo=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace=(Resolve-Path $Workspace).Path
$Python=if($env:BOTA_PYTHON){$env:BOTA_PYTHON}else{"python"}
$Config="configs/ali_stage3_equal_horizon_v1.yaml"

& $Python scripts/bota_if/validate_ali_stage0_amendment02_v1.py --repo $Repo
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}

if($Mode -eq "Summarize") {
  & $Python -m src.bota_short_benchmark.ali_stage3_equal_horizon_v1 `
    --repo $Repo --workspace $Workspace --config $Config `
    --mode Summarize --summary-name $SummaryName
  exit $LASTEXITCODE
}

$Seeds=if($Seed -eq "All"){@(41,42,43)}else{@([int]$Seed)}
if($RunName -and $Seeds.Count -ne 1) {
  throw "-RunName may only be used with one explicit -Seed."
}
foreach($CurrentSeed in $Seeds) {
  $Arguments=@(
    "-m", "src.bota_short_benchmark.ali_stage3_equal_horizon_v1",
    "--repo", $Repo, "--workspace", $Workspace, "--config", $Config,
    "--mode", $Mode, "--seed", "$CurrentSeed"
  )
  if($RunName){$Arguments += @("--run-name", $RunName)}
  & $Python @Arguments
  if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
}

if($Mode -eq "Full" -and $Seed -eq "All") {
  & $Python -m src.bota_short_benchmark.ali_stage3_equal_horizon_v1 `
    --repo $Repo --workspace $Workspace --config $Config `
    --mode Summarize --summary-name $SummaryName
  exit $LASTEXITCODE
}
exit 0
