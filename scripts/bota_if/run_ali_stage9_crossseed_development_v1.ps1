param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Analyze","Summarize")]
  [string]$Mode="Preflight",
  [ValidateSet("All","41","43")]
  [string]$Seed="All",
  [ValidateSet("All","BOTA","IFRU","E2URec","NegGrad","PCGrad","RecEraser")]
  [string]$Method="All",
  [string]$Workspace="C:\python\E2URec-main"
)
$ErrorActionPreference="Stop"
$Repo=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace=(Resolve-Path $Workspace).Path
$Python=if($env:BOTA_PYTHON){$env:BOTA_PYTHON}else{"python"}
$Module="src.bota_short_benchmark.ali_stage9_crossseed_development_v1"

if($Mode -eq "Summarize"){
  & $Python -m $Module --repo $Repo --workspace $Workspace --mode Summarize
  exit $LASTEXITCODE
}

$Seeds=if($Seed -eq "All"){@(41,43)}else{@([int]$Seed)}
$Methods=if($Method -eq "All"){@("BOTA","IFRU","E2URec","NegGrad","PCGrad","RecEraser")}else{@($Method)}
foreach($CurrentSeed in $Seeds){
  foreach($CurrentMethod in $Methods){
    $RunDir=Join-Path $Workspace "outputs\ali_stage9_crossseed_development_v1\seed$CurrentSeed\runs\ali_stage9_$($CurrentMethod.ToLower())_seed$($CurrentSeed)_v1"
    $EffectiveMode=$Mode
    if($Mode -eq "Full" -and (Test-Path $RunDir)){$EffectiveMode="Analyze"}
    & $Python -m $Module --repo $Repo --workspace $Workspace --mode $EffectiveMode --seed $CurrentSeed --method $CurrentMethod
    if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
  }
}
if($Mode -eq "Full" -or $Mode -eq "Analyze"){
  & $Python -m $Module --repo $Repo --workspace $Workspace --mode Summarize
  exit $LASTEXITCODE
}
