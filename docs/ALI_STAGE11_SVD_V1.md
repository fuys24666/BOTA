# Stage 11 frozen-bank spectrum diagnosis

This is a post-hoc diagnosis of the known Stage 6 compression failure. Amendment 10 adds a new analysis; it does not modify the original preregistration, replace the failed Stage 6 decision, or authorize Stage 6B. No data loaders, model construction, optimizer steps, inference or FinalTest access are used.

## Run in the existing pytorch environment

```powershell
cd C:\python\BOTA
.\scripts\bota_if\run_ali_stage11_svd_v1.ps1 -Mode Preflight -Seed All -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage11_svd_v1.ps1 -Mode Full -Seed All -Workspace C:\python\E2URec-main
```

Full sequentially runs seeds 41, 42 and 43, then validates and summarizes all three. It verifies already completed seeds and skips their computation. Partial/failed jobs remain under the new output root's `.work` directory, never masquerade as completed runs and do not overwrite inputs. Restart with the same command. Resume requires identical analysis-code/configuration hashes and source hashes; modifying the implementation after a completed seed deliberately stops reuse.

The launcher uses the active `python`, or `$env:BOTA_PYTHON` if set. It works from another working directory. `-Device auto` uses CUDA when available and CPU otherwise. Use `-Device cpu` to avoid the GPU, `-BlockSize 4096` for smaller working blocks, and `-Threads 4` to reduce CPU contention. Existing banks total about 2.53 GiB on disk; only one seed is processed at a time. No model-sized GPU allocation is required; block buffers and the 256-by-256 Gram matrix dominate GPU memory. CPU memory also holds the memory-mapped bank and small adapter/query tensors. Exact runtime depends on float64 throughput and storage speed; progress is printed during Gram accumulation and query reconstruction.

Independent commands:

```powershell
.\scripts\bota_if\run_ali_stage11_svd_v1.ps1 -Mode SyntheticDryRun
.\scripts\bota_if\run_ali_stage11_svd_v1.ps1 -Mode Analyze -Seed All
.\scripts\bota_if\run_ali_stage11_svd_v1.ps1 -Mode Summary
python -m pytest tests/test_ali_stage11_svd_v1.py -q
```

## Analysis definitions

- Primary: uncentered raw X, rows = 256 users, columns = fixed LoRA-B coordinates. Accumulate G = X X^T in float64, in feature blocks; diagonalize only G. Do not allocate a d-by-d matrix or full right singular-vector matrix.
- Spectrum: sigma_i = sqrt(max(eigenvalue_i, 0)); energy_i = sigma_i squared / total squared singular values. Report 90/95/99% cumulative-energy ranks, stable rank = sum(sigma squared)/sigma_1 squared, entropy effective rank = exp(-sum(p log p)) with p = sigma/sum(sigma). These definitions are not interchangeable.
- Direction diagnostic: normalize each nonzero row to unit norm. This is a separate diagnostic, not a modified compression objective. Zero rows are excluded only from direction analysis and remain in primary bank statistics. Near-zero row counts are also disclosed. All pairwise cosines and individual row norms are exported.
- Rank grid: 4, 8, 16, 32, 64, 128, 192, 256. Report optimal raw-bank relative Frobenius error and, separately, direct blockwise reconstruction errors for the two fixed eight-user unions. Hash-based user mapping and reproduction of Stage 5 FP32 endpoints detect coordinate/membership mismatch.
- Query residual = norm(compressed query - Exact-Masked delta) / norm(Exact-Masked delta). Residual increase = compressed residual / uncompressed-bank residual - 1. It is not bank reconstruction error. Undefined zero-denominator ratios are null, not invented zeros.
- Compare new float64 query residual increases against the historical float32 Stage 6 grid where available. Do not force agreement: floating-point decomposition and near-tied eigenvalues at the cutoff may change the selected subspace. Investigate large discrepancies before interpreting them.
- Numerical rank uses Gram eigenvalues > largest eigenvalue times 1e-12. Materially negative eigenvalues below -1e-10 times spectral scale fail analysis; tiny roundoff negatives are recorded and clipped. A Gram-derived numerical rank is not a theorem about exact algebraic rank.

## Outputs

Root: `C:\python\E2URec-main\outputs\ali_stage11_svd_v1`.

Each `runs/seed41` (and seed42/43) contains `metrics.json`, `spectrum.csv`, `rank_errors.csv`, `row_norms.csv`, `pairwise_cosines.csv`, `report.md`, source/code/configuration provenance in `contract.json`, and a hashed manifest. Public row hashes, not raw user IDs, identify the bank. CSV files are plot-ready; this stage does not fabricate a paper figure or decide the scientific conclusion.

The `summary` folder contains per-seed spectrum summaries and all 48 seed-query-rank cells. No new compression representation is selected. Broad spectra constrain this tested shared linear representation, not arbitrary methods, nonlinear compression, privacy, or population-scale impossibility. A narrow global spectrum with poor request reconstruction instead calls for request-specific error analysis. Preserve all outcomes.

The code records Git state and exact source-file hashes and permits a dirty working tree for this read-only diagnostic. It does not commit or push anything automatically.
