param(
  [ValidateSet("Preflight", "Full", "Analyze")]
  [string]$Mode = "Full"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = if ($env:BOTA_PYTHON) { $env:BOTA_PYTHON } else { "python" }

$HistoricalDataset = "amazon_movies_tv_titles_seed42_v2"
$HistoricalRun = "amazon_movies_tv_titles_original_p5_seed42_v3"
$NewUserDataset = "amazon_movies_tv_newusers_seed42_v4"
$Benchmark = "amz_new_k2k4_s42_v4"
$EvaluationRun = "amz_new_k2_all_eval_s42_v4"
$PrepareHistoricalConfig = "configs/bota_amazon_movies_small_prepare_v1.yaml"
$HistoricalConfig = "configs/bota_amazon_movies_titles_original_p5_v3.yaml"
$PrepareNewUserConfig = "configs/bota_amazon_movies_newuser_prepare_v4.yaml"
$CoreConfig = "configs/bota_short_amazon_movies_newuser_k2k4_v4.yaml"
$PaperConfig = "configs/bota_short_paper_amazon_newuser_k2_v4.yaml"
$Runs = @{
  Original = "amz_new_orig_s42_v4"
  Retrain = "amz_new_exact_s42_v4"
  BOTA = "amz_new_bota_s42_v4"
  IFRU = "amz_new_ifru_s42_v4"
  SISA = "amz_new_sisa_k2_s42_v4"
  RecEraser = "amz_new_rec_k2_s42_v4"
  NegGrad = "amz_new_ng_k2_s42_v4"
  PCGrad = "amz_new_pc_k2_s42_v4"
  E2URec = "amz_new_e2u_k2_s42_v4"
}

function Invoke-BotaPython {
  param([string[]]$Arguments, [string]$Label)
  Write-Host "`n===== $Label =====" -ForegroundColor Cyan
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

if ($Mode -eq "Preflight") {
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_small_prepare", "--root", $Root, "--config", $PrepareHistoricalConfig, "--mode", "Preflight", "--dataset-name", $HistoricalDataset) "AMAZON RAW-DATA PREFLIGHT"
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_titles_original_p5", "--root", $Root, "--config", $HistoricalConfig, "--mode", "SyntheticDryRun", "--run-name", $HistoricalRun) "HISTORICAL P5 DRY RUN"
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_newuser_prepare", "--root", $Root, "--config", $PrepareNewUserConfig, "--mode", "SyntheticDryRun", "--dataset-name", $NewUserDataset) "NEW-USER PREPARATION DRY RUN"
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.runner", "--root", $Root, "--config", $CoreConfig, "--mode", "SyntheticDryRun", "--method", "BOTA", "--benchmark-name", $Benchmark, "--scenario", "K2") "BOTA K2 DRY RUN"
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_e2urec", "--root", $Root, "--config", $CoreConfig, "--mode", "SyntheticDryRun", "--benchmark-name", $Benchmark) "E2UREC DRY RUN"
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_k2_all_evaluation", "--root", $Root, "--mode", "SyntheticDryRun", "--benchmark-name", $Benchmark) "NINE-METHOD EVALUATION DRY RUN"
  exit 0
}

if ($Mode -eq "Analyze") {
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_small_prepare", "--root", $Root, "--config", $PrepareHistoricalConfig, "--mode", "Analyze", "--dataset-name", $HistoricalDataset) "HISTORICAL DATA AUDIT"
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_titles_original_p5", "--root", $Root, "--config", $HistoricalConfig, "--mode", "Analyze", "--run-name", $HistoricalRun) "HISTORICAL P5 AUDIT"
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_newuser_prepare", "--root", $Root, "--config", $PrepareNewUserConfig, "--mode", "Analyze", "--dataset-name", $NewUserDataset) "NEW-USER DATA AUDIT"
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_k2_all_evaluation", "--root", $Root, "--mode", "Analyze", "--run-name", $EvaluationRun) "NINE-METHOD RESULT AUDIT"
  exit 0
}

$Dirty = git -C $Root status --porcelain
if ($LASTEXITCODE -ne 0) { throw "BOTA must be initialized as a Git repository before a Full run" }
if ($Dirty) { throw "Commit the reproduction code before a Full run; the working tree must be clean" }

$HistoricalDataPath = Join-Path $Root "outputs/bota_amazon_movies_tv_v1/prepared/$HistoricalDataset"
if (Test-Path $HistoricalDataPath) {
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_small_prepare", "--root", $Root, "--config", $PrepareHistoricalConfig, "--mode", "Analyze", "--dataset-name", $HistoricalDataset) "HISTORICAL DATA AUDIT"
} else {
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_small_prepare", "--root", $Root, "--config", $PrepareHistoricalConfig, "--mode", "Prepare", "--dataset-name", $HistoricalDataset) "PREPARE HISTORICAL COHORT"
}

$HistoricalPath = Join-Path $Root "outputs/bota_amazon_movies_tv_v3/originals/$HistoricalRun"
$HistoricalWork = Join-Path $Root "outputs/bota_amazon_movies_tv_v3/originals/.work/$HistoricalRun"
$HistoricalMode = if (Test-Path $HistoricalPath) { "Analyze" } elseif (Test-Path (Join-Path $HistoricalWork "checkpoint.pt")) { "Resume" } else { "Full" }
Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_titles_original_p5", "--root", $Root, "--config", $HistoricalConfig, "--mode", $HistoricalMode, "--run-name", $HistoricalRun) "HISTORICAL P5 ($HistoricalMode)"

$NewUserPath = Join-Path $Root "outputs/bota_amazon_movies_tv_v4/prepared/$NewUserDataset"
if (Test-Path $NewUserPath) {
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_newuser_prepare", "--root", $Root, "--config", $PrepareNewUserConfig, "--mode", "Analyze", "--dataset-name", $NewUserDataset) "NEW-USER DATA AUDIT"
} else {
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_newuser_prepare", "--root", $Root, "--config", $PrepareNewUserConfig, "--mode", "Prepare", "--dataset-name", $NewUserDataset) "PREPARE DISJOINT NEW-USER COHORT"
}

$ProtocolPath = Join-Path $Root "outputs/amz_v4/k2k4/protocols/$Benchmark"
if (Test-Path $ProtocolPath) {
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.runner", "--root", $Root, "--config", $CoreConfig, "--mode", "Preflight", "--method", "Original", "--benchmark-name", $Benchmark, "--scenario", "All") "VERIFY REQUEST REGISTRY"
} else {
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.runner", "--root", $Root, "--config", $CoreConfig, "--mode", "Prepare", "--method", "Original", "--benchmark-name", $Benchmark, "--scenario", "All") "FREEZE K2 REQUEST REGISTRY"
}

$CoreIds = @{ Original = "Original-Short"; Retrain = "Retrain-Short"; BOTA = "BOTA-T2-Short"; IFRU = "IFRU-Short-LoRA" }
foreach ($Method in @("Original", "Retrain", "BOTA", "IFRU")) {
  $Path = Join-Path $Root "outputs/amz_v4/k2k4/models/$($CoreIds[$Method])/$($Runs[$Method])"
  $RunMode = if (Test-Path $Path) { "Analyze" } else { "Full" }
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.runner", "--root", $Root, "--config", $CoreConfig, "--mode", $RunMode, "--method", $Method, "--benchmark-name", $Benchmark, "--scenario", "All", "--run-name", $Runs[$Method]) "$Method CORE ENDPOINT ($RunMode)"
}

$PartitionIds = @{ SISA = "SISA-Short-T5"; RecEraser = "RecEraser-Adapter-Short" }
foreach ($Method in @("SISA", "RecEraser")) {
  $Path = Join-Path $Root "outputs/amz_v4/k2k4/models/$($PartitionIds[$Method])/$($Runs[$Method])"
  $RunMode = if (Test-Path $Path) { "Analyze" } else { "Full" }
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.runner", "--root", $Root, "--config", $CoreConfig, "--mode", $RunMode, "--method", $Method, "--benchmark-name", $Benchmark, "--scenario", "K2", "--run-name", $Runs[$Method]) "$Method ($RunMode)"
}

$PaperIds = @{ NegGrad = "NegGrad-Mixed-Short-BOnly"; PCGrad = "PCGrad-Short-BOnly" }
foreach ($Method in @("NegGrad", "PCGrad")) {
  $Path = Join-Path $Root "outputs/amz_v4/paper_k2/models/$($PaperIds[$Method])/$($Runs[$Method])"
  $RunMode = if (Test-Path $Path) { "Analyze" } else { "Full" }
  Invoke-BotaPython @("-m", "src.bota_short_benchmark.paper_v2", "--root", $Root, "--config", $PaperConfig, "--mode", $RunMode, "--method", $Method, "--benchmark-name", $Benchmark, "--scenario", "K2", "--run-name", $Runs[$Method], "--source-original-run-name", $Runs.Original) "$Method ($RunMode)"
}

$E2Path = Join-Path $Root "outputs/amz_v4/e2urec_k2/models/E2URec-Short-FixedAB/$($Runs.E2URec)"
$E2Mode = if (Test-Path $E2Path) { "Analyze" } else { "Full" }
Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_e2urec", "--root", $Root, "--config", $CoreConfig, "--mode", $E2Mode, "--benchmark-name", $Benchmark, "--original-run-name", $Runs.Original, "--run-name", $Runs.E2URec) "E2UREC ($E2Mode)"

$EvalPath = Join-Path $Root "outputs/amz_v4/k2_all_evaluation/$EvaluationRun"
$EvalMode = if (Test-Path $EvalPath) { "Analyze" } else { "Full" }
Invoke-BotaPython @("-m", "src.bota_short_benchmark.amazon_movies_k2_all_evaluation", "--root", $Root, "--config", $CoreConfig, "--paper-config", $PaperConfig, "--mode", $EvalMode, "--benchmark-name", $Benchmark, "--run-name", $EvaluationRun, "--original-run-name", $Runs.Original, "--exact-run-name", $Runs.Retrain, "--bota-run-name", $Runs.BOTA, "--ifru-run-name", $Runs.IFRU, "--e2urec-run-name", $Runs.E2URec, "--neggrad-run-name", $Runs.NegGrad, "--pcgrad-run-name", $Runs.PCGrad, "--sisa-run-name", $Runs.SISA, "--receraser-run-name", $Runs.RecEraser) "NINE-METHOD K2 EVALUATION ($EvalMode)"

Write-Host "`n===== AMAZON MOVIES AND TV NEW-USER K2 REPRODUCTION COMPLETED =====" -ForegroundColor Green
