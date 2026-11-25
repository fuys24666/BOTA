# BOTA

[![Tests](https://github.com/fuys24666/BOTA/actions/workflows/tests.yml/badge.svg)](https://github.com/fuys24666/BOTA/actions/workflows/tests.yml)

Research code for **BOTA: Offline Trajectory Amortization for Short-Window User Deletion in Language-Model Recommenders**.

[Read the preprint on Zenodo (v1)](https://doi.org/10.5281/zenodo.24519524)

BOTA moves request-specific reconstruction work into the offline adaptation trajectory. It propagates slot-preserving deletion sources through the parameter and moment states of AdamW, with a calibrated empirical-Fisher response, and stores terminal per-user impact vectors. Supported online requests compose these vectors and publish an Adapter with **zero optimizer steps**.

The [manuscript result tables](results/README.md) contain the per-seed MovieLens and Amazon comparisons and the per-setting serving measurements. They accompany the Stage 27 implementation and frozen protocols from October 17, 2026. See the [reproduction map](docs/CURRENT_CODE_SNAPSHOT.md) for the experimental inputs and entry points.

## Current evidence

### MovieLens-1M

Frozen T5-base/P5, fixed-A/trainable-B Q/V LoRA, matched 200-step masked replay; three seeds (41/42/43), two registered requests, and frozen Fisher response strength **lambda = 64**. The table uses attention-masked fixed-configuration FinalTest evaluation. Residual is normalized to Original = 1 and Exact-Masked = 0; lower is better.

| Request | Absolute local error E_U (x 10^-3) | Residual |
|---|---:|---:|
| L8 | 0.1008 +/- 0.0318 | 0.3207 +/- 0.1010 |
| L4M4 | 0.0608 +/- 0.0096 | 0.2264 +/- 0.0431 |

Values are mean +/- sample SD across seeds. Local evaluation covers 55 examples/four requested users for L8 and 104 examples/five requested users for L4M4. BOTA has lower absolute matched-counterfactual error than five eligible calibrated approximate baselines in all six seed-request cells. SISA has no eligible calibrated candidate and is not counted as a win.

Resident-bank online P95 is **42.583-43.396 ms** for 12 users and **42.448-44.117 ms** for 256 users across the six settings. Timing includes durable publication and reload validation, but excludes cold bank loading, bank refresh and fallback. Offline construction is accounted for separately.

### Amazon Movies and TV

The auxiliary comparison uses seeds **42 and 43**, three metadata-selected **K=2** requests (one low-frequency and one middle-frequency user each), frozen **lambda = 48**, and source-only **lambda = 0** as an ablation. It uses a reused Development cohort, **not an independent FinalTest replication**. Each request has ten local evaluation examples.

| Seed | BOTA mean E_U | Mean residual | E_U reduction vs source-only | Binary utility gate |
|---|---:|---:|---:|---:|
| 42 | 0.00819 | 0.7655 | 51.3% | 3/3 |
| 43 | 0.00564 | 0.9438 | 60.8% | 3/3 |

Means weight the three requests equally. The calibrated response improves source-only transport in all six cells, while improvement over Original occurs in three of six cells and depends on seed and request. E2URec, NegGrad, IFRU and RecEraser comparisons are retained, including utility failures. Two IFRU solves fail; PCGrad and SISA have no eligible calibrated candidate. Missing or failed endpoints are not counted as BOTA wins. Stage 27 completes the selected RecEraser evaluation without reselection.

## Scientific scope

The supported claim is **short-window matched-counterfactual reconstruction**, not certified privacy deletion or equivalence to unrestricted retain-only retraining. The calibrated Fisher response is a surrogate: improved reconstruction does not establish accurate recovery of the true Hessian. Request interactions, long-horizon transport, attack evidence and storage/fidelity trade-offs are evaluated separately. Low latency on a resident bank does not establish population-scale deployment.

ML-1M configurations are selected on Development and held fixed for the reported evaluation. FinalTest is a fixed, reused cohort, not a fresh independent replication. Do not interpret historical calibration, repeated scoring or auxiliary Development comparisons as independent test evidence.

## Code and experiment map

| Area | Implementation / entry points |
|---|---|
| AdamW-aware transport | `src/bota_if/`, `src/bota_short_benchmark/` |
| Baseline calibration | Stage 7 (ML-1M), Stage 20 (Amazon), Stage 21 (seed 43 transfer) |
| SVD and response calibration | Stages 11-14; Amazon Stage 18 v2 |
| Ranking protocol and primary endpoints | Stages 15-16; frozen lambda-64 suite in Stage 17 |
| Amazon requests and comparisons | Stages 19-21; numerical/pair diagnostics in Stages 22-23 |
| Masked, batch-invariance-checked scoring | Stages 24-25 and the Stage 25 FinalTest loader repair |
| Focused stress evaluation | Stage 26B; Stage 26 retains the broader, separate plan |
| Completed Amazon RecEraser scoring | Stage 27 |

Configuration files live in `configs/`; launchers in `scripts/bota_if/`; detailed protocols in `docs/`. Existing audits in `docs/audits/` are dependencies of frozen experiments and must not be discarded wholesale. Older modules can remain shared dependencies even when their historical result tables are superseded.

## Environment and tests

The experiment environment uses Python 3.11, PyTorch 2.11 with CUDA 12.8, Transformers 4.57.6 and an RTX 5070 Ti. Install the appropriate PyTorch build separately, then:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

GPU integration tests are opt-in; the default test run is not a full scientific reproduction. PowerShell entry points accept `-Workspace` and use `BOTA_PYTHON` when set, otherwise `python` from the active environment. Review each runbook before `Full`.

## External artifacts and reproduction

Datasets, tokenizers, pretrained models, recommendation checkpoints, impact banks and prediction caches are **not in Git**. See [ARTIFACTS.md](docs/ARTIFACTS.md) for original asset lineage and [CURRENT_CODE_SNAPSHOT.md](docs/CURRENT_CODE_SNAPSHOT.md) for current dependencies.

A clone alone cannot reproduce archived experiments: later stages require hash-bound parent protocols and frozen models from the external workspace. Missing models and hash mismatches must be diagnosed, not bypassed. A new independent reproduction must declare its own artifact lineage and protocol rather than claiming the archived experiment's identity.

Rerunning the same command resumes verified units where supported; the interrupted unit may restart. FinalTest requires explicit confirmation. Stage 27 reconstruction of missing selected models requires `-RebuildMissing`; it is not implicit. Uploading this snapshot does not launch experiments or replace results.

## Preparing a clean source snapshot

```powershell
python scripts/release/prepare_source_snapshot.py --output-dir ../BOTA_release_20261124
```

This creates a ZIP, SHA-256 manifest, Git path list and upload guide outside the repository. The archive includes the four aggregate CSVs listed in the [result index](results/README.md). Raw experiment directories, datasets, model weights and prediction caches are excluded. Review the file list before upload.

`.gitattributes` preserves file bytes because archived protocol checks include source hashes. Do not apply blanket line-ending conversion or formatting to frozen experimental files.

## Citation and license

Fu, Yusen, Ali Anaissi, and Ali Braytee (2026). *BOTA: Offline Trajectory Amortization for Short-Window User Deletion in Language-Model Recommenders*. Zenodo, version 1. [doi:10.5281/zenodo.24519524](https://doi.org/10.5281/zenodo.24519524)

Software metadata is in [CITATION.cff](CITATION.cff). If you use the upstream E2URec preprocessing or checkpoint lineage, cite that pipeline separately. See [THIRD_PARTY.md](THIRD_PARTY.md).

Code is licensed under [MIT](LICENSE). Data and pretrained model licenses remain those of their respective providers.
