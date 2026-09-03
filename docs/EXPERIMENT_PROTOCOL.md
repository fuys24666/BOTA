# Frozen experiment protocol

This page centralizes the configuration needed to reproduce the manuscript
tables. Machine-readable values remain authoritative in `configs/` and the
run manifests.

## Data and splits

MovieLens-1M contains 1,000,209 ratings from 6,040 users over 3,706 rated
movies. Ratings are ordered by timestamp, user and movie. A target is emitted
after at least five prior interactions, using the most recent ten interactions
as history; ratings above 3 are positive. The 970,009 eligible examples are
split chronologically into 60,000 training, 20,000 Development and 20,000
FinalTest prompts. The training adaptation window is the first 3,200 indices
of a seed-42 permutation of the 60,000 training prompts, giving 200 batches of
16. FinalTest is reconstructed from the raw ratings and compared byte-for-byte
with the processed prompt file before evaluation.

L8 deletes eight low-frequency users. L4M4 deletes four low- and four
middle-frequency users. Frequency groups are defined from global training
frequency; request selection does not inspect labels or predictions.

## Shared model and optimizer

The backbone is T5-base. The primary short-window methods use Q/V LoRA with
rank 16, scaling 32, no dropout, a fixed orthonormal A matrix and zero-initialized
trainable B matrices. This coordinate contains 884,736 trainable values. The
fixed-A and stochastic seeds are 41, 42 and 43 in the three-seed experiment.
AdamW uses learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8, weight decay
0.01, no scheduler and no gradient clipping.

## Method budgets

| Method | Request-specific computation and budget |
|---|---|
| Original / Exact-Masked | 200 AdamW steps; canonical or slot-preserving masked batches |
| BOTA | one offline 200-step transported trajectory; zero online optimizer steps |
| IFRU | 200-step canonical authority; 512-example curvature panel, 12 power iterations, damping 0.01 times estimated maximum eigenvalue, at most 40 CG iterations, relative tolerance 1e-4 |
| NegGrad | 200 fixed-coordinate steps; forget weight 0.2 |
| PCGrad | 200 fixed-coordinate steps; deterministic symmetric two-task projection |
| E2URec | starts from Original-Short; augmented teacher 1,200 steps at 5e-5, then student 1,000 steps at 0.001 (one forget warm-up plus 999 joint steps), effective batch 16, microbatch 4, response scale 2, remembering/forgetting weights 0.6/0.4 |
| SISA | four full-T5 shards, four slices, one epoch per slice, learning rate 5e-4, batch 16 |
| RecEraser | four Q/V LoRA local models, rank 16, scaling 32, dropout 0.05; four balanced text-similarity partitions; learning rate 0.001; 200 attention-aggregation steps at learning rate 0.1 |

Original, Exact-Masked, BOTA, IFRU, NegGrad, PCGrad and the adapted E2URec
baseline use the shared fixed-A/trainable-B coordinate. SISA and RecEraser
retain their method-defining full-model/partitioned parameterizations; their
optimizer-step counts, trainable parameters and wall-clock costs are reported
rather than described as coordinate-matched.

## Evaluation and timing

All selection and ablation decisions use Development. The seed-41/42/43
combinations are frozen before a single primary FinalTest evaluation. E2URec
was subsequently evaluated through a baseline-only supplement that reuses
hashed Original and Exact-Masked predictions and performs no new inference for
other methods.

Training time starts immediately before model construction/loading and ends
after the completed artifact is durably moved into its run directory. BOTA
online latency starts before user-vector lookup and ends after composition,
Adapter materialization, durable publication and reload validation. It excludes
the one-time impact-bank build. The P50/P95 experiment uses 20 warm-up requests
and 1,000 measured requests per scenario.

## Entry points

- Primary three-seed runs: `scripts/bota_if/run_short_multiseed_v3.ps1`
- Primary FinalTest: `scripts/bota_if/run_short_multiseed_finaltest_v3.ps1`
- E2URec six-condition and supplemental FinalTest:
  `scripts/bota_if/run_short_e2urec_multiseed_v1.ps1`
- Repeated serving latency: `scripts/bota_if/run_short_online_latency_v1.ps1`

See `README.md` for complete commands and `ARTIFACTS.md` for required local
files.
