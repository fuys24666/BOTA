param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Analyze")][string]$Mode="Preflight",
  [Parameter(Mandatory=$true)][string]$BenchmarkName,
  [string]$OriginalRunName="",[string]$RetrainRunName="",[string]$IFRURunName="",
  [string]$SISARunName="",[string]$RecEraserRunName="",[string]$BOTARunName="",
  [string]$RunName="",[ValidateSet("Development")][string]$Split="Development"
)
$ErrorActionPreference="Stop";$Root=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path;$Python=if($env:BOTA_PYTHON){$env:BOTA_PYTHON}else{"python"}
$Arguments=@("-m","src.bota_short_benchmark.evaluation","--root",$Root,"--config","configs/bota_short_benchmark_v1.yaml","--mode",$Mode,"--benchmark-name",$BenchmarkName)
if($RunName){$Arguments+=@("--run-name",$RunName)}
if($OriginalRunName){$Arguments+=@("--original-run-name",$OriginalRunName)}
if($RetrainRunName){$Arguments+=@("--retrain-run-name",$RetrainRunName)}
if($IFRURunName){$Arguments+=@("--ifru-run-name",$IFRURunName)}
if($SISARunName){$Arguments+=@("--sisa-run-name",$SISARunName)}
if($RecEraserRunName){$Arguments+=@("--receraser-run-name",$RecEraserRunName)}
if($BOTARunName){$Arguments+=@("--bota-run-name",$BOTARunName)}
& $Python @Arguments
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
