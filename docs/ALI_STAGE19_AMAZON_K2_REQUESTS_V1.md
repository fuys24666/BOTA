# Stage 19 — frozen new-request Amazon K2 validation

## What remains the same as the paper

The original Amazon extension reports one seed42 K2 request: two users,
four deleted slots in a200-step/3200-slot window, and10 local Development
examples. See the existing manuscript source
`C:\python\BOTA_PAPER\IJMLC_manuscript_v13\sections\08_appendices.tex`
and its `generated\amazon_results.tex`.

Stage19 keeps **K=2 for every request**, one lower-tercile and one middle-tercile
training-frequency user, exactly two window appearances per user, and the same
historical Amazon model, fixed-coordinate architecture and200-step setup.
It does NOT delete24 users at once. Twelve mutually user-disjoint pairs expand
request coverage while preserving the original four-slot perturbation size.

Original/Exact-Masked and BOTA use the corrected Stage18v2 canonical-gradient
convention. Historical paper BOTA endpoints are not used as the new BOTA
reference, because the Stage18 diagnostic established different numerical
canonical trajectories in that old implementation.

## Request registry

Outcome-blind preflight found61 eligible lower-tercile and23 eligible
middle-tercile users after excluding old request users458,481,558,767.
Hash ordering is fixed by Amendment18; pair the first12 from each role by rank.
All24 selected users are distinct, every pair has four masked slots and10 local
Development samples. Registry hash:

`4dd7c16aa8d6b475a0a4f37d3d9bed7ae8653bdf1d56a706d7cb7a2d9fa5cec7`

Selection uses training-frequency/window-exposure and Development-count sidecars
only, never labels, scores, residuals or anticipated improvements. The frequency
terciles retain historical tie-breaking; these are operational strata, not a
claim that every lower/middle user has a unique frequency value.

No unfavorable pair can be replaced. Insufficient pools cause a stop, not relaxed
criteria. Same12 pairs appear in seeds41/42/43.

## Fixed comparisons

- Primary: lambda48, frozen from Stage18, versus lambda0 (no Fisher).
- Reference target: matched Exact-Masked replay, separately computed for each
  request and seed. Original is the no-deletion comparator (its E_U is D_U).
- Secondary: fixed lambda64, testing the MovieLens setting without additional
  Amazon tuning. It cannot replace lambda48 after results are observed.
- Lambda1 is only an internal, nonselecting implementation control. At each step
  theta/m/v must be tensor-identical to the independent same-trajectory reference.
- Every bank shard canonical must be tensor-identical to that seed's sealed
  Stage18v2 Original. No legacy trajectory mixing.

All3840 Development examples are scored for overall utility. Each pair's10
local examples are used for E_U, D_U, residual and Toward-Exact Rate. Per-user
errors and complete predictions are retained. No FinalTest is loaded.

## Two evidence levels and stop rule

Amendment18 fixes the thresholds BEFORE new-request experiments:

1. **Mechanism gain:** primary lambda48 must pass utility/no-collapse criteria
   in all36 seed-request cells; improve mean E_U over lambda0 by >=1e-6 absolute
   AND >=1% relative in each seed; improve at least8 of12 requests in the
   three-seed mean; and have a positive lower95% paired request-bootstrap bound.
2. **Matched-counterfactual support:** only after(1), apply the same gain/win/
   bootstrap criteria versus Original. Better than no Fisher alone is not enough.

Bootstrap:5000 resamples of12 request identities, seed190042, preserving all
three seeds together within each request. These are conditional uncertainty
diagnostics; shared models/cohorts limit independent-population interpretation.
Missing or numerically failed primary cells cannot be discarded to pass.

Possible final labels:

- `L2_MATCHED_COUNTERFACTUAL_SUPPORT`: both gates pass within this narrow setup.
- `L1_FISHER_GAIN_ONLY`: only controlled Fisher improvement is supported.
- `NOT_SUPPORTED`: stop as boundary evidence; no automatic new parameter sweep.

These are prespecified operational evidence gates, not a privacy guarantee,
certified unlearning result, or proof of arbitrary request generalization.

## Scope caveats

The hypothesis was motivated by Stage18's mixed findings and K2 pattern. The old
K2 results are not counted as new confirmation. This is new-request validation
on **previously used Development data**, not a new holdout. Sample support is
still only five examples/user. Seeds vary fixed-A/runtime, not historical model
pretraining, data cohort, order, or user requests. No attack, storage, long-window,
baseline retuning, ranking expansion or manuscript rewrite is included.

## Run and resume

```powershell
cd C:\python\BOTA
.\scripts\bota_if\run_ali_stage19_amazon_k2_requests_v1.ps1 `
  -Mode Full `
  -Seed All `
  -Workspace C:\python\E2URec-main
```

Modes: `Preflight` (read-only checks), `Freeze` (seal registry and contracts),
`Smoke` (isolated two-step non-scientific QA), `Full`, `Analyze` (read-only).
Full freezes if necessary, then runs all seeds by default. Optional `-Seed 41`,
`42`, `43` runs only that seed. No final classification is published until all36
request cells exist. Rerun the same Full command after interruption: verified
banks, Exact endpoints, individual prediction jobs and request cells are reused.

Do not change source/config while running. Do not run two jobs concurrently.
The shared kernel prints `[Stage18] transport`; surrounding `[Stage19]` messages
identify the actual seed and shard. This does not restart the old parameter sweep.

Memory: each seed's24 users are split into three8-user shards; requests are not
split across shards. Total planned work: nine 200-step bank constructions plus
36 × 200-step masked replays = **9000 optimizer steps**. Inference:144 request-
endpoint panels plus three shared Originals =147 panels of3840 samples.
All three lambda endpoints share each bank trajectory, not separate training.

Output root:
`C:\python\E2URec-main\outputs\ali_stage19_amazon_k2_requests_v1`

Frozen registry and hashes: `protocol\result.json`.
Final table/report: `summary\report.md`; full results: `summary\result.json`.

## Pre-launch checks

Real-asset preflight passed. The eight-user CUDA smoke passed two transport
steps, tensor-exact same-trajectory control, serialization and10-example
inference. Its observed peak reservation was5.77GiB; this short check does not
predict full-run peak memory or validate final scientific results.
69 unit/regression tests passed, covering deterministic request selection, exclusions, no pool
relaxation, both gates, failure retention, no secondary-candidate rescue,
bootstrap determinism, incomplete/duplicate detection, sealed artifacts and
per-request resume. Full experiment remains for the user to run offline.
