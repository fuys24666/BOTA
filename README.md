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

The repository also contains the principal baselines, the AdamW-state ablation, the empirical-Fisher ablation, repeated online-latency measurement, and the single-seed GoodReads K2 transfer experiment.

## Repository layout

```text
configs/                 Frozen experiment configurations
scripts/bota_if/         PowerShell entry points for formal runs
data_preprocess/         Prompt-template utilities required by ML-1M loading
src/bota_short_benchmark Main benchmark, evaluation and latency code
src/bota_if/             BOTA trajectory transport kernels
src/paper_baselines/     SISA and RecEraser implementations
src/paper_ratio_suite/   IFRU implementation used in the comparison
src/diagnostics/         Minimal shared T5/LoRA runtime dependencies
tests/                   Curated unit and synthetic tests
docs/                    Artifact layout and release notes
```

## What is not stored in Git

The following are deliberately excluded:

- MovieLens, GoodReads, or derived prompt data;
- T5-base weights and tokenizers;
- the approximately 892 MB ML-1M recommendation Original checkpoint;
- BOTA impact banks, Adapters, baseline checkpoints and evaluation outputs;
- FinalTest predictions or any user-level identifiers;
- the manuscript source, local research notes, failed-run ledgers and exploratory audits.

See [docs/ARTIFACTS.md](docs/ARTIFACTS.md) for the required local layout. The data preparation follows the text-to-text recommendation setup of [E2URec](https://github.com/justarter/E2URec); users must comply with the original dataset and model licenses.

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

Formal entry points are provided for IFRU, NegGrad, PCGrad, SISA, RecEraser, FullControl-P5 and Retain-Retrain-P5 in `scripts/bota_if/`. Each launcher exposes its parameters through PowerShell help, for example:

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

## GoodReads K2 transfer

GoodReads is included as a supplementary single-seed transfer experiment rather than a second primary benchmark. Prepare the public GoodReads comics data, train its recommendation Original, and then run the K2 protocol using the corresponding launchers in `scripts/bota_if/`. Exact paths and frozen authority fields are documented in [docs/ARTIFACTS.md](docs/ARTIFACTS.md).

## Scientific scope

BOTA approximates a matched, short-window, slot-preserving masked trajectory under a fixed LoRA coordinate and optimizer protocol. It is not a certified-unlearning algorithm and does not claim equivalence to an arbitrary converged retraining endpoint. The empirical-Fisher term is a secondary response surrogate; the main mechanism is AdamW-aware source transport followed by zero-optimizer-step online composition.

## Citation

Please cite the BOTA paper once its public preprint record is available. The final BibTeX entry will be added here before archival release. The implementation also builds on the public E2URec text-to-text recommendation pipeline, which should be cited separately when its preprocessing or checkpoint lineage is used.

## License

A source-code license has not yet been selected. Add an explicit `LICENSE` before making the GitHub repository public; without one, reuse rights are not granted automatically.
