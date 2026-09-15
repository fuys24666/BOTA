# Amendment 09: one post-confirmation FinalTest re-evaluation

Amendment 09 is frozen after the independently recomputed Stage 9 result
`STABLE_CONFIRMED`. It authorizes one inference-only evaluation of the frozen
60-cell endpoint inventory: three model seeds, L8/L4M4, four reference/control
endpoints, BOTA and five selected eligible baselines. SISA remains
`NO_ELIGIBLE_CANDIDATE` and is not replaced with a failed candidate.

The primary quantity is requested-user absolute probability error to the
matched 200-step Exact-Masked target. Retain-Retrain-P5 is reported alongside
this comparison as a converged-endpoint sensitivity using FullControl-P5 as
its control. Neither reference is used for further selection.

This same FinalTest cohort has previously been evaluated. The new result must
be labeled a post-Development-tuning re-evaluation, not an untouched one-shot
test or an independent new holdout. Historical exposure remains a limitation
even though the Stage 7/8 selection and Stage 9 confirmation used Development
only.

Preflight checks hashes and CPU adapter signatures without constructing test
examples or running GPU inference. Full requires an explicit confirmation
switch and a clean committed repository. Access is reserved permanently
before reconstructing FinalTest. A failure is recorded and is not
automatically retried, even under another run name. Any recovery requires a
separate evidence-based amendment.
