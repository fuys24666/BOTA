param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Analyze","Summary")]
  [string]$Mode="Preflight",
  [ValidateSet("All","41","42","43")]
  [string]$Seed="All",
  [string]$Workspace="C:\python\E2URec-main"
)
$ErrorActionPreference="Stop"
$Repo=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace=(Resolve-Path $Workspace).Path
$Python=if($env:BOTA_PYTHON){$env:BOTA_PYTHON}else{"python"}
$Seeds=if($Seed -eq "All"){@(41,42,43)}else{@([int]$Seed)}
if($Mode -eq "Summary"){
  & $Python -m src.bota_short_benchmark.ali_stage6_storage_screen_v1 --repo $Repo --workspace $Workspace --mode Summary
  exit $LASTEXITCODE
}
foreach($Current in $Seeds){
  $RunName="ali_stage6_storage_screen_seed${Current}_v1"
  & $Python -m src.bota_short_benchmark.ali_stage6_storage_screen_v1 --repo $Repo --workspace $Workspace --seed $Current --mode $Mode --run-name $RunName
  if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
}
if($Mode -eq "Full" -and $Seed -eq "All"){
  & $Python -m src.bota_short_benchmark.ali_stage6_storage_screen_v1 --repo $Repo --workspace $Workspace --mode Summary
}
exit $LASTEXITCODE
