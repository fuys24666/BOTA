# Ali Stage 3 result audit

Stage 3 completed on 12 September 2026 before Stage 4 attack execution. All three seed runs and the six-cell summary passed the repository analyzer and hash checks. The experiment used Development only, never constructed FinalTest, and did not load or use Retain-Retrain-P5 as the local-residual denominator.

## Frozen result

| Scenario | DU, mean ± sample SD | EU, mean ± sample SD | Local residual | Delta cosine | Relative L2 | Point estimates toward Exact |
|---|---:|---:|---:|---:|---:|---:|
| L8 | 0.001037 ± 0.000037 | 0.001610 ± 0.000138 | 1.5516 ± 0.0844 | 0.5649 ± 0.1200 | 0.8375 ± 0.0796 | 0/3 |
| L4M4 | 0.001142 ± 0.000252 | 0.001230 ± 0.000489 | 1.0514 ± 0.1823 | 0.3926 ± 0.2679 | 0.8917 ± 0.1289 | 2/3 |

L8 moved away from `Exact-Masked-Tlong` in every seed. L4M4 produced point residuals below one for seeds 41 and 43, but both bootstrap confidence intervals crossed one; seed 42 moved away and its interval was entirely above one. The three-seed L4M4 mean was also above one.

The frozen interpretation is therefore negative: the experiment does not support reliable extrapolation of BOTA from the validated 200-step adaptation horizon to 1,200 steps. This finding narrows the method scope; it does not retroactively alter the preregistered 200-step primary experiments.

## Integrity and accounting

- All six seed-by-scenario cells are retained.
- Each seed executed 1,200 canonical steps and two 1,200-step Exact-Masked replays: 3,600 physical optimizer calls per seed and 10,800 in total.
- BOTA used zero online optimizer steps.
- No prediction collapse was observed.
- The machine-local source hashes and complete numerical record are frozen in `ali_stage3_result_audit_v1.json`.
- Stage 4 attack results may not be used to revise this interpretation or select another BOTA mechanism arm.
