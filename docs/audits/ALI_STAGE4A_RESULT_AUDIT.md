# Ali Stage 4A result audit

Stage 4A completed on 12 September 2026 using the nine frozen Stage 2 requests,
three fixed endpoints and four registered attacks. The manifest and all output
hashes validate. The run used Development only and did not construct or access
FinalTest.

The result is frozen as negative. Only `high_k4` passed the previously fixed
`DU >= 0.001` eligibility gate. Its Original-to-Exact-Masked reference signal
was -0.001713 with a 95% user-cluster bootstrap interval of
[-0.002709, 0.003000], far below the required 0.02 and not distinguishable from
zero. Its BOTA gap-improvement point estimate was 0.001629, but the 95% interval
[-0.000127, 0.002904] also crossed zero. Utility remained within the frozen
guardrails.

All eight ineligible requests remain descriptive. `high_k8` was the only one
whose BOTA gap-improvement interval excluded zero, but its DU of 0.000799 failed
the outcome-independent eligibility threshold and cannot be promoted to formal
evidence.

The frozen interpretation is that the current matched-user attacks do not
provide a valid Original-to-Exact-Masked forgetting signal, even in the sole
eligible stronger-signal request. Attack optimization therefore stops under
this protocol. The parent Stage 4 result remains authoritative, and subsequent
work proceeds to sequential deletion and storage scaling.
