# Ali feedback experiments: Stage 0 preregistration

This record freezes the evidence plan before new attack, sequential-request or
storage-compression results are generated. The machine-readable authority is
`configs/ali_stage0_preregistration_v1.json`.

## Scope and evidential boundary

The primary study remains the three-seed ML-1M 200-step benchmark with the L8
and L4M4 registered requests. Development is the only permitted split during
method development. Existing FinalTest outputs are outside this audit and must
not be opened, regenerated or used to choose attacks, thresholds, compression
settings or sequential-request designs.

Exact-Masked is the primary matched-local counterfactual. Retain-Retrain-P5 is
a separately stopped retain-only sensitivity endpoint and must not be pooled
with the Exact-Masked residual. FullControl-P5 is its corresponding complete-data
anchor.

## Frozen questions

1. **Attack verification:** does BOTA move the matched-user attack result toward
   Exact-Masked rather than merely preserve aggregate AUC and log loss?
2. **Sequential verification:** does recomposition of a cumulative registered
   user union from the canonical endpoint remain close to an Exact-Masked replay
   of the same union?
3. **Scale verification:** can a measured representation reduce impact-bank
   storage by at least 16x without increasing local residual by more than 10%?

Null and adverse results are reportable outcomes. Attack results may not be used
to select checkpoints or retrospectively remove a method, seed or scenario.

## Primary attack contract

The primary attack uses negative answer loss under the matched-user protocol.
The primary statistic is pooled matched-user ROC-AUC, accompanied by macro
user-level ROC-AUC and a 2,000-resample authoritative-user bootstrap. Confidence,
negative entropy and absolute yes/no margin are mandatory secondary attacks.

Reference signal is considered adequate only if the absolute attack advantage
decreases from Original to Exact-Masked by at least 0.02 and the user-cluster
bootstrap confidence interval excludes zero in the expected direction. BOTA is
successful on the primary comparison only if its gap to Exact-Masked is smaller
than the Original-to-Exact-Masked gap and the bootstrapped improvement excludes
zero. Retain-Retrain-P5 remains a separately reported sensitivity comparison.

The utility guardrails, transferred from the earlier frozen evaluation
contracts, are an overall AUC decrease of at most 0.005 and a log-loss increase
of at most 0.01 relative to Original.

## Sequential contract

Every cumulative request is recomposed from the same canonical endpoint. Blind
addition to a previously published Adapter is not the primary algorithm. The
registered request sizes are 1, 2, 4 and 8 users, evaluated under low-first,
middle-first, alternating and deterministic hash order. Each step is compared
with a newly produced Exact-Masked replay of the same cumulative union.

If the two-user result is not closer to Exact-Masked than Original, or is
materially order-dependent, the manuscript retains one-shot-only scope.

## Scale contract

Bank sizes advance through 12, 32, 64, 128 and 256 users only after the previous
size completes. FP32, FP16, per-vector INT8, per-block INT8, top-q sparsity and
shared low-rank representations are all reportable. Shared ranks are fixed at
4, 8, 16, 32 and 64. Required measurements include storage, build time, peak GPU
memory, loading and composition time, local residual, utility and MIA.

User chunking may reduce peak memory but does not change the reported total
`O(KTd)` construction work. If effective rank grows approximately linearly with
the bank population or no representation reaches 16x compression within the
10% residual guardrail, the paper retains registered-set prototype scope.

## Artifact audit command

From the repository root:

```powershell
python scripts/bota_if/audit_ali_stage0_v1.py `
  --experiment-root C:\python\E2URec-main\outputs\bota_short_multiseed_v3
```

The command writes a JSON audit trail and a human-readable Markdown report under
`docs/audits/`. It does not traverse the existing supplemental FinalTest tree.

## Amendment rule

Runtime-only fixes, additional logging and machine-local path changes do not
alter the preregistration. Any change to a primary metric, threshold, attack,
seed, scenario, reference endpoint or FinalTest policy requires a dated
amendment recorded before the affected result is inspected.
