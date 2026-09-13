# Ali Stage 5 result audit

Stage 5 completed on 13 September 2026 using Development only. All three runs,
60 registered cells and 39 unique-union endpoint sets passed manifest and hash
validation. FinalTest was neither constructed nor accessed.

The preregistered sequential decision failed. At K=2, the three-seed mean local
residual was 2.0675 for low-first, 0.2088 for middle-first and 0.6671 for both
alternating and deterministic-hash prefixes. The resulting spread was 1.8587,
far above the frozen maximum of 0.10. All 60 utility guardrails passed.

All four K=8 order families resolved to the same user-set hash within each seed.
This confirms that canonical recomposition is deterministic for a fixed union
and that the failure is not accumulated chain or reload drift. Instead, quality
depends strongly and non-monotonically on which user subset appears at an
intermediate request.

The result is frozen as negative and the manuscript retains one-shot-only
scope. Later storage experiments may evaluate registered one-shot unions but
cannot promote a general continuous-deletion claim.
