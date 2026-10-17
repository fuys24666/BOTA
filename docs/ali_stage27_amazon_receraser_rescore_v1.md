# Stage27: six frozen Amazon RecEraser comparison endpoints

## Scope fixed before execution

- Amazon, seeds 42/43, the same metadata-selected A1/A2/A3 requests and 1,961 Dev-Select samples per endpoint.
- The Stage20-selected `receraser_lr0.002_e1`: four shards, one epoch, archived partition seed and training seed. Seed43 inherits the selection; no search.
- Only RecEraser is recovered and rescored. Stage25 mask-aware Original, Exact-Masked, BOTA lambda48, BOTA source-only and other available comparison scores are reused.
- No FinalTest, new requests, new lambda, retuning, or manuscript edits.
- PCGrad and SISA retain their historical NO_ELIGIBLE_CANDIDATE designation. IFRU solver failures remain unavailable. No missing score counts as a BOTA win.
- All six RecEraser cells are reported, including utility failures. Utility eligibility is recomputed, but is not a new model-selection rule.

## Why reconstruction may be necessary

Stage20/21 deliberately removed generated partition-model tensors after publishing predictions, configurations, aggregation weights and model hashes. RecEraser passed calibration; absent weights are not a calibration failure. Stored predictions cannot be converted into corrected mask-aware scores.

Preflight performs no training or inference. Full without `-RebuildMissing` refuses to train. Explicit `-RebuildMissing` authorizes only reconstruction of missing selected endpoints with the archived settings. It does not authorize tuning or replacement of failed solvers.

The unchanged historical partition engine is replayed, including its original Development aggregation fitting procedure. This is a reproduction attempt, not a newly chosen aggregation: the full models-directory hash (initial and final tensors), aggregation-file hash, partition record and optimizer budget must match the archive exactly before corrected scoring. A mismatch stops; there is no tolerance relaxation, silent refit acceptance or fallback to old predictions. Thus exact reproduction is a gate, not a promise that fresh training must reproduce on every runtime.

Only output/synthetic directories change in the copied YAML. Historical files are never overwritten. All reconstructed weights are retained.

## Scoring and validation

The Stage25 scorer supplies the encoder attention mask, mixes raw token probabilities with the verified frozen weights, then computes normalized Yes/No probabilities and token losses. It checks batch sizes 1/4/8, changed neighbor/order compositions, and probe-versus-full-cohort consistency with the frozen Stage25 tolerances. Stage25 metric formulas and utility thresholds remain unchanged.

## Running

```powershell
.\scripts\bota_if\run_ali_stage27_amazon_receraser_rescore_v1.ps1 -Mode Preflight -Workspace C:\python\E2URec-main

# Explicit permission to reconstruct missing archived models; not just inference:
.\scripts\bota_if\run_ali_stage27_amazon_receraser_rescore_v1.ps1 -Mode Full -Seed All -RebuildMissing -Workspace C:\python\E2URec-main
```

`-Seed 42` or `-Seed 43` can split execution. Summary requires all six cells. `-Mode Analyze` validates and summarizes completed cells without constructing/scoring models.

## Resume and evidence

Rerun the same Full command after a normal interruption. Completed scored cells are checksum-verified and reused; completed reconstructed endpoints are verified and reused. Interrupted training resumes at the last durable shard/RNG boundary; only the current incomplete shard is repeated. A published-but-uncommitted shard is preserved under `recovery/` before replay. Scoring resumes at endpoint granularity, so an interrupted endpoint repeats up to four shard forward passes. A memory-guard stop, missing RNG checkpoint, identity mismatch or batch-invariance failure requires inspection rather than automatic retry.

Outputs: `outputs/ali_stage27_amazon_receraser_rescore_v1/summary/REPORT.md`, `receraser_metrics.csv`, `amazon_all_methods.csv`; all weights under `models/`. Existing Stage20/21/25 results are untouched.

## Time budget (estimate, not a benchmark result)

Archived six-endpoint build-plus-old-scoring time sums to approximately 23.5 minutes, with 2,050 model optimizer steps. Stage25 Amazon single-model masked scoring took approximately 22 seconds per 1,961 samples; 24 shard evaluations are approximately nine minutes plus loading/checks. Reserve about **35–60 minutes** on the same RTX 5070 Ti with no competing GPU job. Pure corrected scoring with verified weights already present should take approximately **10–20 minutes**. Input hashing, interruptions, exact-reproduction failures and GPU contention can change these estimates.
