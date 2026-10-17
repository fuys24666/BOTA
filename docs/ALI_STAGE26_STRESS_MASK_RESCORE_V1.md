# Stage 26: frozen stress rescoring and unaffected-artifact audit

## Scope

This is an evaluation repair, not a search or new training experiment.
MovieLens-1M primary BOTA remains lambda=64. Existing lambda=0 and matched
FM0/FM64 ablations are preserved. Historical lambda=1 diagnostics remain in
their archive and are not part of this new run. Original, Exact-Masked, P5 and
baseline weights are frozen. Amazon and FinalTest are not accessed.

Part A rescores the exact saved Stage17 endpoints with explicit encoder padding
masks, unchanged score/loss formulas, fixed batch size 4 and unchanged metrics:

- 6 primary seed/request groups, including FM behavior and attack diagnostics;
- all 9 stronger requests (only high_k4 has formal stronger-attack permission);
- 39 distinct sequential unions, expanded back to all 60 order/size cells;
- 6 independent 1,200-step horizon groups;
- 72 compression groups: all 12 representations, 3 seeds, 2 requests;
- 1 historical 600-step continuous relay, kept separate from the 1,200-step protocol.

Total: 133 evaluation groups. Stage17 full Development binary cohort (20,000),
Stage15 frozen sampled-ranking cohort (53,809), and original membership cohorts
are retained. Compression has no newly added attack experiment. The historical
relay retains its binary-only behavior scope. Stage17 float32 yes/no ranking
normalization is preserved; Stage15/25's special float64 normalization is not
silently substituted.

Part B verifies existing parameter metrics, raw/row-normalized SVD, storage sizes,
compression ratios, position diagnostics and timing manifests. These values do
not depend on prediction masking. It reports them as reused measurements, not
newly executed measurements, and never converts parameter fidelity into a
behavioral or privacy claim. No bank construction, optimizer/backward/HVP/CG
steps, compression fitting or GPU timing benchmark is called.

## Safeguards and resume

- Preflight hashes actual weights and inputs; absence and hash corruption differ.
- Missing weights are listed, never reconstructed automatically. A changed frozen
  artifact blocks execution. Restoring a previously missing file requires a
  separately reviewed inventory, not an unnoticed mid-run substitution.
- Protocol binds source code, runtime, GPU, data, candidate plan, model artifacts,
  requests, gates and original endpoint identities.
- Each model/cohort unit is atomically published with a checksum manifest.
- Identical units are reused across requests. Metrics still preserve every request.
- Ctrl+C leaves completed units reusable. The interrupted unit starts again;
  there is no within-unit minibatch resume. Failures remain in `.work`.
- Batch sizes 1/4/8 and different neighboring samples are checked. Audit failure
  blocks continuation, not counted as a scientific method failure.
- Missing comparators and numerical failures are not wins. Missing references
  prevent normalized reconstruction claims. No legacy predictions fill gaps.
- Metric bootstraps, attack signal thresholds and utility gates remain unchanged.

## Commands (activated pytorch environment, repository root)

```powershell
.\scripts\bota_if\run_ali_stage26_stress_mask_rescore_v1.ps1 -Mode Preflight -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage26_stress_mask_rescore_v1.ps1 -Mode Full -Workspace C:\python\E2URec-main
```

Full freezes the audited protocol automatically if not already frozen. The
same Full command resumes. No FinalTest confirmation flag is accepted.

Optional small real-model smoke: `-Mode Smoke` (3 lanes, 64 samples per lane,
no full cache entries). Optional partial family: `-Family sequential`, `horizon`,
`stronger`, `primary`, `compression` or `relay`. Full without Family completes
all remaining jobs. `-Mode Analyze` computes pending summaries from existing
prediction bundles without model inference; it reports incomplete coverage.

## Output

`C:\python\E2URec-main\outputs\ali_stage26_stress_mask_rescore_v1`

- `audit/endpoint_inventory.csv`: complete saved-weight/job inventory.
- `audit/missing_weights.json`: unavailable endpoints (empty when all retained).
- `audit/unaffected_records.json`: verified reuse pointers and parameter records.
- `audit/runtime_estimate.json`: observed-Stage25-based ETA, not a guarantee.
- `predictions/<id>`: mask-aware prediction, invariance probes and progress evidence.
- `metrics/<group>`: full binary/ranking/attack metrics per frozen group.
- `summary/All`: behavior.csv, ranking.csv, attacks.json, all_results.json,
  sequential_all_cells.json and REPORT.md.

The first inventory found all 762 unique units READY (264 binary, 261 ranking,
237 attack). Runtime is dominated by ranking inference and depends on GPU
contention, storage and bootstrap work; allow roughly 40–60 hours on the same
RTX 5070 Ti. Part B is checksum verification, normally minutes rather than a
new training run. No full run is started by code preparation.
