# Ali Stage 8 BOTA response-strength audit

Frozen Development-only inventory. FinalTest was not accessed.

## Selection

Selected `bota_r1` (response scale 1.0), with equal-scenario mean $E_U=0.000115662$.

| Candidate | Response scale | Mean $E_U$ | Eligible | Selected |
|---|---:|---:|---|---|
| `bota_r0` | 0.00 | 0.000118615 | yes | no |
| `bota_r0.25` | 0.25 | 0.000117861 | yes | no |
| `bota_r0.5` | 0.50 | 0.000117122 | yes | no |
| `bota_r0.75` | 0.75 | 0.000116390 | yes | no |
| `bota_r1` | 1.00 | 0.000115662 | yes | yes |

## Formal scale-one reproduction

| Scenario | Cosine | Relative $L_2$ error | Maximum absolute error | Passed |
|---|---:|---:|---:|---|
| L8 | 0.999988194 | 0.004859979 | 0.000116501 | yes |
| L4M4 | 0.999988481 | 0.004801478 | 0.000119938 | yes |

The JSON companion freezes all per-scenario metrics, artifact hashes and source bindings.
