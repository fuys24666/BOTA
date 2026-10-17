param(
  [ValidateSet("Preflight","SyntheticDryRun","Full","Analyze","Summary")]
  [string]$Mode="Preflight",
  [ValidateSet("All","41","42","43")][string]$Seed="All",
  [string]$Workspace="C:\python\E2URec-main",
  [ValidateSet("auto","cpu","cuda")][string]$Device="auto",
  [ValidateRange(1,65536)][int]$BlockSize=8192,
  [ValidateRange(1,256)][int]$Threads=8
)
$ErrorActionPreference="Stop"
$Repo=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Workspace=(Resolve-Path $Workspace).Path
$Python=if($env:BOTA_PYTHON){$env:BOTA_PYTHON}else{"python"}
$Seeds=if($Seed -eq "All"){@(41,42,43)}else{@([int]$Seed)}
Push-Location $Repo
try {
  if($Mode -in @("Summary","SyntheticDryRun")) {
    & $Python -m src.bota_short_benchmark.ali_stage11_svd_v1 --repo $Repo --workspace $Workspace --mode $Mode --threads $Threads
    if($LASTEXITCODE -ne 0){throw "Stage 11 $Mode failed (exit $LASTEXITCODE)"}
  } else {
    foreach($Current in $Seeds) {
      & $Python -m src.bota_short_benchmark.ali_stage11_svd_v1 --repo $Repo --workspace $Workspace --seed $Current --mode $Mode --device $Device --block-size $BlockSize --threads $Threads
      if($LASTEXITCODE -ne 0){throw "Stage 11 seed $Current failed (exit $LASTEXITCODE)"}
    }
    if($Mode -eq "Full" -and $Seed -eq "All") {
      & $Python -m src.bota_short_benchmark.ali_stage11_svd_v1 --repo $Repo --workspace $Workspace --mode Summary --threads $Threads
      if($LASTEXITCODE -ne 0){throw "Stage 11 summary failed (exit $LASTEXITCODE)"}
    }
  }
} finally { Pop-Location }
