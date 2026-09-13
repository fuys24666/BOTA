# Ali Stage 0 Amendment 06

This reviewer-driven amendment freezes the Development-only baseline sweep
before any Stage 7 candidate is trained or evaluated. It does not revise the
original preregistration and cannot alter the adverse long-horizon, attack,
sequential-deletion or storage conclusions from Stages 3--6.

Stage 7A uses seed 42 and jointly selects one configuration per baseline across
L8 and L4M4. The six baselines are IFRU, E2URec, NegGrad, PCGrad, SISA and
RecEraser. The complete candidate grids are recorded in the amendment and must
be reported, including failed and utility-ineligible configurations.

Development is deterministically partitioned without using predictions. The
even SHA256 parity subset is Dev-Fit and is used only to fit RecEraser's learned
aggregation. The odd subset is Dev-Select and is the common model-selection
set for every baseline. This prevents RecEraser from fitting its aggregation
and selecting hyperparameters on the same examples.

Eligibility requires no more than 0.005 AUC damage and 0.01 log-loss damage
relative to Original in both request scenarios. Among eligible candidates, the
primary selector is the equal-scenario mean absolute requested-user error to
Exact-Masked. The normalized residual remains reportable but is not optimized
directly because its Original--Exact denominator is approximately 2e-4.

All candidate configurations, metrics and model-state hashes are permanent.
Non-selected model tensors may be discarded after verified evaluation to avoid
retaining many full T5 shard ensembles. The deterministically selected
evaluated candidate and its final artifacts are retained.

FinalTest access is forbidden. A selected Stage 7A configuration must be
atomically frozen before cross-seed Development confirmation. Any later
FinalTest rerun requires another dated amendment and must be described as a
post-review confirmatory rerun rather than the original untouched evaluation.
