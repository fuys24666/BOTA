# Ali Stage 0 Amendment 05

This amendment freezes the storage screen before any Stage 6 result is
generated. It preserves the original 16x compression and 10% residual guardrail
and does not alter the negative Stage 4 attack or Stage 5 sequential decisions.

Stage 6A constructs nested 12, 32, 64, 128 and 256-user banks for seeds 41--43.
The twelve primary registered users appear first; remaining users are selected
by a deterministic outcome-independent hash among users exposed in the
adaptation window. No Development predictions, labels or example-count
threshold enter bank membership. This rule was fixed after preflight showed
that the initially drafted Development-count restriction left only 203 users
and therefore could not instantiate the already specified K=256 bank; no Stage
6 result had been generated. User chunking is permitted only as a memory
control and every repeated trajectory is counted.

All frozen FP32, FP16, INT8, top-q and shared-low-rank representations are
measured and screened in parameter space against the paired L8 and L4M4
Exact-Masked endpoints. Eligibility is determined at K=256 across every seed
and scenario. Only one deterministically selected candidate may proceed to
Stage 6B behavioral confirmation. If no candidate passes, the storage claim
fails without further tuning.
