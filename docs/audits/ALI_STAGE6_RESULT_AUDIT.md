# Ali Stage 6 result audit

Stage 6A completed on 13 September 2026 using parameter-space comparisons only.
All three 256-user banks, 360 fixed-grid cells and 4,800 physical optimizer-step
calls passed manifest and hash validation. Development predictions and FinalTest
were not accessed. FP32 bank recomposition reproduced the frozen Stage 5
artifacts to a maximum relative error of (8.67\times10^{-7}), within the fixed
(10^{-6}) tolerance.

The preregistered storage decision failed. FP16 preserved the parameter residual
almost exactly but compressed by only 2x. Per-block INT8 was also nearly
lossless, with a worst relative residual increase of 0.00194, but achieved only
3.996x measured compression. Thus the best high-fidelity result does not solve
the population-scale storage problem.

Every representation exceeding 16x failed the 10% fidelity guardrail. The least
damaging storage-eligible representation was 3.125% Top-q: it achieved 21.332x
compression but increased the parameter residual by as much as 2.325, or about
232%. Shared rank 8 achieved 31.989x compression but increased the residual by
as much as 3.465. Shared rank 16 measured 15.9949x rather than 16x and also had
large error; it cannot be rounded upward to pass either condition.

The result is frozen as negative with no selected candidate. Under Amendment 05,
Stage 6B behavioral and attack confirmation must not run. The defensible result
is a bounded diagnostic: ordinary quantization can reduce storage by about 4x
without materially worsening BOTA's existing approximation, whereas the fixed
aggressive compression schemes do not support the manuscript's deployment-scale
storage claim.
