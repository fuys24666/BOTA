# BOTA

Official reproduction code for **BOTA: Blockwise Optimizer-Aware Trajectory Amortization for Zero-Optimizer-Step Online Unlearning in Language-Model Recommenders**.

BOTA moves user-specific deletion reconstruction into the offline adaptation trajectory. It extracts slot-preserving deletion sources, propagates them through the full first-order AdamW state, stores terminal user impact vectors, and answers supported online deletion requests by vector composition and durable Adapter publication. The online path performs no training forward pass, backward pass, Hessian-vector product, iterative solve, or optimizer step.

This repository is a curated release of the formal experiments. It intentionally excludes private/local artifacts, intermediate failed runs, exploratory notebooks, model weights, datasets, and generated outputs.

## Main results reproduced by this code

The primary protocol uses a frozen T5-base backbone, fixed-A/trainable-B Q/V LoRA coordinates, and a matched 200-step slot-preserving counterfactual on MovieLens-1M.

| Scenario | BOTA local residual (3-seed FinalTest) | End-to-end online P95 |
|---|---:|---:|
| L8 | 0.4934 ± 0.2287 | 48.131 ms |
| L4M4 | 0.4804 ± 0.0995 | 48.503 ms |

The repository also contains the principal baselines, the AdamW-state ablation, the empirical-Fisher ablation, repeated online-latency measurement, and the single-seed Amazon Movies and TV new-user K2 dataset extension.

| Amazon new-user K2 method | MAE to Exact | Local residual | AUC | LogLoss |
|---|---:|---:|---:|---:|
| Original | 0.004312 | 1.000000 | 0.797093 | 0.370971 |
| Exact-Masked | 0.000000 | 0.000000 | 0.796956 | 0.370728 |
| BOTA | **0.002341** | **0.542999** | 0.800265 | 0.366186 |
| IFRU | 0.004391 | 1.018470 | 0.797104 | 0.370965 |
| E2URec | 0.009714 | 2.252940 | 0.800625 | 0.392706 |
| NegGrad | 0.239738 | 55.602730 | 0.651330 | 0.701075 |
| PCGrad | 0.299722 | 69.514906 | 0.535671 | 2.606139 |
| SISA | 0.159386 | 36.966706 | 0.625099 | 0.451486 |
| RecEraser | 0.013763 | 3.192056 | **0.807195** | **0.360221** |

These Amazon values are seed-42 Development results for one registered request. They support dataset extension but are not presented as multi-seed or FinalTest evidence. The frozen aggregate table and protocol summary are retained in [docs/AMAZON_K2_RESULTS.md](docs/AMAZON_K2_RESULTS.md).

## Repository layout

```text
configs/                 Frozen experiment configurations
scripts/bota_if/         PowerShell entry points for formal runs
data_preprocess/         Prompt-template utilities required by ML-1M loading
src/bota_short_benchmark Main benchmark, evaluation and latency code
src/bota_if/             BOTA trajectory transport kernels
src/paper_baselines/     SISA and RecEraser implementations
src/paper_ratio_suite/   IFRU implementation used in the comparison
src/paper_e2urec_fair_pair_v2/
                         E2URec fair-pair loss used by the adapted baseline
src/diagnostics/         Minimal shared T5/LoRA runtime dependencies
tests/                   Curated unit and synthetic tests
docs/                    Artifact layout and release notes
```

## What is not stored in Git

The following are deliberately excluded:

- MovieLens, Amazon Reviews 2023, or derived prompt data;
- T5-base weights and tokenizers;
- the approximately 892 MB ML-1M recommendation Original checkpoint;
- BOTA impact banks, Adapters, baseline checkpoints and evaluation outputs;
- FinalTest predictions or any user-level identifiers;
- the manuscript source, local research notes, failed-run ledgers and exploratory audits.

See [docs/ARTIFACTS.md](docs/ARTIFACTS.md) for the required local layout and
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) for the centralized
data, optimizer, baseline-budget and timing specification. The data preparation
follows the text-to-text recommendation setup of
[E2URec](https://github.com/justarter/E2URec); users must comply with the
original dataset and model licenses.

## Environment

The audited environment was:

- Windows 11 and PowerShell 7;
- Python 3.11;
- PyTorch 2.11.0 with CUDA 12.8;
- Transformers 4.57.6;
- one NVIDIA RTX 5070 Ti with 16 GB VRAM.

Other recent PyTorch/CUDA combinations may work but were not used for the reported numbers.

Create an environment and install a CUDA-compatible PyTorch build from the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```powershell
conda create -n bota python=3.11 -y
conda activate bota
# Install the appropriate PyTorch build first.
pip install -r requirements.txt
$env:BOTA_PYTHON = (Get-Command python).Source
```

The launchers use `BOTA_PYTHON` when it is set and otherwise fall back to `python`; no machine-specific Python path is embedded in the release.

## Quick code check

Synthetic mode checks argument routing without loading the real T5 model or dataset:

```powershell
./scripts/bota_if/run_short_bota_v1.ps1 `
  -Mode SyntheticDryRun `
  -BenchmarkName smoke `
  -Scenario L8
```

Run the curated tests with:

```powershell
pytest -q
```

## Full MovieLens-1M workflow

Full formal runs require the artifacts and paths listed in [docs/ARTIFACTS.md](docs/ARTIFACTS.md). The code records source hashes and requires a clean Git state for formal publication, so initialize and commit the repository before running `Full`:

```powershell
git init
git add .
git commit -m "Initial BOTA reproduction release"
```

### 1. Freeze the benchmark registry

```powershell
./scripts/bota_if/run_short_benchmark_prepare_v1.ps1 `
  -Mode Prepare `
  -BenchmarkName bota_short_i02_seed42_v1
```

### 2. Run the matched endpoints and BOTA

```powershell
./scripts/bota_if/run_short_original_v1.ps1 `
  -Mode Full -BenchmarkName bota_short_i02_seed42_v1 `
  -Scenario L8 -RunName bota_short_original_l8_seed42_v1

./scripts/bota_if/run_short_retrain_v1.ps1 `
  -Mode Full -BenchmarkName bota_short_i02_seed42_v1 `
  -Scenario L8 -RunName bota_short_exact_masked_l8_seed42_v1

./scripts/bota_if/run_short_bota_v1.ps1 `
  -Mode Full -BenchmarkName bota_short_i02_seed42_v1 `
  -Scenario L8 -RunName bota_short_bota_l8_seed42_v1
```

Replace `L8` with `L4M4` for the mixed low/middle-frequency request. The same launchers support `Analyze` after completion.

### 3. Run baselines

Formal entry points are provided for IFRU, E2URec, NegGrad, PCGrad, SISA, RecEraser, FullControl-P5 and Retain-Retrain-P5 in `scripts/bota_if/`. Each launcher exposes its parameters through PowerShell help, for example:

```powershell
Get-Help ./scripts/bota_if/run_short_ifru_v1.ps1 -Detailed
```

### 4. Three-seed protocol

```powershell
foreach ($Seed in 41, 42, 43) {
  ./scripts/bota_if/run_short_multiseed_v3.ps1 -Mode Prepare -Seed $Seed
  foreach ($Method in "Original", "ExactMasked", "BOTA", "IFRU", "SISA", "RecEraser", "NegGrad", "PCGrad") {
    ./scripts/bota_if/run_short_multiseed_v3.ps1 `
      -Mode Full -Seed $Seed -Method $Method -Scenario All
  }
}
```

FinalTest is intentionally a separate, explicit one-time step:

```powershell
./scripts/bota_if/run_short_multiseed_finaltest_v3.ps1 `
  -Mode Full `
  -RunName bota_short_multiseed_finaltest_v3 `
  -ConfirmFinalTest
```

Do not use FinalTest for model selection or hyperparameter tuning.

### 5. Reproduce the E2URec row

E2URec is adapted to the same registered L8/L4M4 requests and fixed-A,
trainable-B LoRA coordinate. It starts from each seed's completed
`Original-Short` model. Train all six seed/scenario conditions and aggregate
Development first:

```powershell
foreach ($Seed in 41, 42, 43) {
  ./scripts/bota_if/run_short_e2urec_multiseed_v1.ps1 `
    -Mode Full `
    -Seed $Seed `
    -Scenario All `
    -RunName "bota_short_e2urec_seed${Seed}_v1"
}

./scripts/bota_if/run_short_e2urec_multiseed_v1.ps1 `
  -Mode EvaluateDevelopment `
  -RunName bota_short_e2urec_multiseed_development_v1
```

The paper added E2URec after the primary FinalTest was complete. Its
baseline-only supplemental FinalTest reuses the frozen Original and
Exact-Masked predictions from that primary run and performs new inference only
for E2URec:

```powershell
$PrimaryFinal = "bota_short_multiseed_finaltest_ml1m_seed41_43_v3_recovery1"
$E2URecDev = "bota_short_e2urec_multiseed_development_v1"
$E2URecFinal = "bota_short_e2urec_supplemental_finaltest_v1"

./scripts/bota_if/run_short_e2urec_multiseed_v1.ps1 `
  -Mode SupplementalFinalTestPreflight `
  -DevelopmentRunName $E2URecDev `
  -PrimaryFinalTestRunName $PrimaryFinal

./scripts/bota_if/run_short_e2urec_multiseed_v1.ps1 `
  -Mode EvaluateSupplementalFinalTest `
  -RunName $E2URecFinal `
  -DevelopmentRunName $E2URecDev `
  -PrimaryFinalTestRunName $PrimaryFinal `
  -ConfirmSupplementalFinalTest

./scripts/bota_if/run_short_e2urec_multiseed_v1.ps1 `
  -Mode AnalyzeSupplementalFinalTest `
  -RunName $E2URecFinal
```

If your primary FinalTest has another run name, change only
`$PrimaryFinal`. The supplemental evaluator validates its manifest and
prediction hashes before reusing it.

## Online latency and Fisher ablation

The latency benchmark separates the one-time bank construction from 1,000 online repetitions:

```powershell
./scripts/bota_if/run_short_online_latency_v1.ps1 `
  -Mode BuildBank -BankName bota_short_latency_bank_seed42_v1

./scripts/bota_if/run_short_online_latency_v1.ps1 `
  -Mode Full `
  -BankName bota_short_latency_bank_seed42_v1 `
  -RunName bota_short_online_latency_seed42_v1
```

The source-only versus blockwise empirical-Fisher comparison is launched with:

```powershell
./scripts/bota_if/run_short_fisher_ablation_v1.ps1 `
  -Mode Full `
  -BenchmarkName bota_short_i02_seed42_v1 `
  -Scenario L8 `
  -RunName bota_short_fisher_ablation_l8_seed42_v1
```

## Amazon Movies and TV new-user K2 extension

The supplementary dataset-extension experiment uses the Movies and TV category of [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/). It first builds a 256-user historical cohort and trains its Development-selected P5 recommendation Adapter. A disjoint 768-user cohort is then used for the 200-step adaptation window. The registered K2 request contains one low-frequency and one middle-frequency user, each exposed twice in the window.

Place the official review and metadata files as described in [docs/ARTIFACTS.md](docs/ARTIFACTS.md), then check the complete pipeline without loading the dataset or model:

```powershell
./scripts/bota_if/run_amazon_movies_newuser_k2_v4.ps1 -Mode Preflight
```

After committing the reproduction code so that the Git worktree is clean, run the historical Adapter, the disjoint new-user preparation, all nine endpoints and the Development evaluation with:

```powershell
./scripts/bota_if/run_amazon_movies_newuser_k2_v4.ps1 -Mode Full
```

The final aggregate report is written to:

```text
outputs/amz_v4/k2_all_evaluation/amz_new_k2_all_eval_s42_v4/report.md
```

Completed artifacts can be checked without rerunning training:

```powershell
./scripts/bota_if/run_amazon_movies_newuser_k2_v4.ps1 -Mode Analyze
```

## Scientific scope

BOTA approximates a matched, short-window, slot-preserving masked trajectory under a fixed LoRA coordinate and optimizer protocol. It is not a certified-unlearning algorithm and does not claim equivalence to an arbitrary converged retraining endpoint. The empirical-Fisher term is a secondary response surrogate; the main mechanism is AdamW-aware source transport followed by zero-optimizer-step online composition.

## Citation

Please cite the BOTA paper once its public preprint record is available. The final BibTeX entry will be added here before archival release. The implementation also builds on the public E2URec text-to-text recommendation pipeline, which should be cited separately when its preprocessing or checkpoint lineage is used.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
