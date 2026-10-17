# Stage 15 — frozen lambda=64 Development ranking

## Purpose and boundaries

Amendment 14 adds a supplementary ranking-utility evaluation. It does not
change any earlier preregistration, tune lambda, train models, authorize
FinalTest, or establish privacy forgetting. Lambda=64 and its Stage 14
selection, confirmation and checkpoint manifests are hash-bound before this
stage's inference. BOTA1 is the frozen old configuration, not a candidate.

Development and its seeds have been used previously. This is not an untouched
holdout. No processed FinalTest files/predictions are read. The established raw
lineage reconstruction reads the unsplit ratings source but materializes only
train and Development rows, not the final slice.

## Frozen candidate protocol

- Primary: sampled NDCG@10. Secondary: NDCG@5/20 and Recall@5/10/20.
- Reuse the exact existing Dev-Select index hash (10,088 rows); no new split.
- Catalogue: union of item IDs observed before Development, reconstructed from
  training rows/histories and each user's earliest Development row's history.
  Do not use future targets to enlarge it.
- For every Development user, include **all** distinct Dev-Select items with
  rating >=4 that belong to this catalogue and are absent from that user's
  pre-Development history/training interactions.
- Add 100 distinct unobserved catalogue items. Rank available IDs by the
  SHA256 of `[ali-stage15-ranking-v1, authoritative_user_id, movie_id]` and take
  the first 100. This deterministic sample is shared across all methods,
  scenarios and model seeds. No model scores inform sampling.
- Exclude **all** known user interactions through Development end from the
  negative pool, including nonselected Dev-Fit items and known low ratings.
  Do not consult FinalTest interactions to exclude future positives.
- Unobserved items receive assumed relevance zero: they are not verified
  dislikes. Other Development positives are not treated as sampled negatives.
- If fewer than 100 negatives exist, use all and report that count. A user
  with no eligible positive is excluded from macro averaging and explicitly
  listed in coverage, including requested users. Never impute NDCG=0 for an
  undefined positive-free query. An empty negative pool is a protocol error.
- Every candidate for a user uses the **same last ten pre-Development
  interactions**, not the target-specific histories from successive rows.
  Otherwise later history would expose held-out items and contexts would differ.
- Rank by `P(Yes)/(P(Yes)+P(No))` at the first decoder token. No relevance label
  is fed to the model: all decoder targets are the same dummy `Yes.` string,
  preserving the previous decoder shape rather than changing its numerical path.
  Ties use ascending movie ID. RecEraser mixes the frozen
  shard token probabilities before Yes/No normalization, matching its existing
  evaluator. Preserve the existing legacy input-padding/no-mask convention.
- Check every prompt fits 512 tokens; fail rather than silently truncate the
  target title/question. Cache tokenization in memory across model endpoints.

This is **sampled-candidate NDCG**, not a full-catalogue ranking result. Do not
compare it numerically to incompatible published candidate protocols. Sampling
can change method comparisons relative to full-catalogue evaluation; see
[Krichene and Rendle, KDD 2020](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/).

## Endpoints and reporting

Three seeds (41/42/43), L8 and L4M4, 11 frozen endpoints per cell:
Original, Exact-Masked, BOTA64, BOTA1, FullControl-P5, Retain-Retrain-P5,
IFRU, E2URec, NegGrad, PCGrad, RecEraser. SISA still has no eligible frozen
candidate and is not silently reintroduced or retrained. Baselines remain
calibrated to the existing counterfactual objective, not to NDCG.

Report per-user metrics and user-macro averages separately for all, requested,
and retained users, plus every requested user's coverage. Aggregate across
three seeds only when all three exist. Sample SD is descriptive, not a
confidence interval. Report per-cell NDCG@10 changes relative to Original and
Exact-Masked, including negative changes. There is no new ranking acceptance
threshold, hyperparameter selection, or automatic FinalTest permission.

The preflight on 2026-10-08 found 572 Development users, 484 eligible ranking
users and 53,809 candidate pairs per endpoint. User coverage is a data property,
not a model success result. Raw frozen candidates and all exclusions are saved
in `protocol/candidates.json` when Full first starts.

## Run in the existing pytorch environment

```powershell
cd C:\python\BOTA
.\scripts\bota_if\run_ali_stage15_ranking_v1.ps1 `
  -Mode Preflight `
  -Workspace C:\python\E2URec-main

.\scripts\bota_if\run_ali_stage15_ranking_v1.ps1 `
  -Mode Full `
  -Seed All `
  -Scenario All `
  -Method All `
  -Workspace C:\python\E2URec-main
```

For separate batches, restrict `-Seed 42`, `-Scenario L8` or `-Method BOTA64`.
This filters execution only, never changes the frozen 66-cell analysis matrix.
A subset run reports PARTIAL, not COMPLETED. Do not run two processes at once:
an OS-managed output lock prevents collisions. The `BOTA_PYTHON` environment
variable optionally selects an explicit Python executable.

Re-running the same Full command validates and skips completed endpoint cells.
An interrupted cell restarts that cell; it does not resume inside a shard/batch.
Incomplete diagnostics stay under `.work` and are never treated as results.
Do not modify code, config or input models during the run: hash drift blocks reuse.

```powershell
.\scripts\bota_if\run_ali_stage15_ranking_v1.ps1 `
  -Mode Analyze `
  -Workspace C:\python\E2URec-main
```

Analyze uses saved scores and independently recomputes metrics on CPU; it
does not load a model or recompute predictions. It reports missing cells, and
publishes a complete summary only after all 66 endpoint cells are available.

## Outputs

`C:\python\E2URec-main\outputs\ali_stage15_ranking_v1\`

- `protocol/contract.json`: immutable config, source, data and endpoint hashes.
- `protocol/candidates.json`: candidate IDs, relevance, common histories, coverage.
- `cells/s<seed>_<scenario>_<method>/scores.json`: all candidate scores in frozen order.
- `cells/.../metrics.json`: every user, macro group and requested-user coverage.
- `cells/.../identity.json`, `manifest.json`, `COMPLETED`: provenance and atomic completion.
- `summary/summary.json`: complete cells, seed aggregates and paired differences.
- `summary/report.md`: readable NDCG@10 / Recall@10 summary; all K values remain in JSON.

No earlier experimental result, manuscript, original preregistration or model
is overwritten. No Git commit or push is part of this stage.

## Verification

```powershell
python -m pytest tests/test_ali_stage15_ranking_v1.py -q
```

An optional, bounded integration test validates token lengths and five real
checkpoint interfaces. It compares first-token scoring with the previous
teacher-forced scorer on only four candidates per component; these are not
formal ranking results:

```powershell
$env:BOTA_STAGE15_GPU_SMOKE = '1'
python -m pytest tests/test_ali_stage15_ranking_integration_v1.py -q -s
Remove-Item Env:BOTA_STAGE15_GPU_SMOKE
```

If ranking results expose a problem, retain all results and review a separate
Development amendment; never change lambda or candidate rules within this run.
