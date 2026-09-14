# Ali Stage 7A unified baseline candidate audit

Frozen Development-only inventory. FinalTest was not accessed.

## Selection summary

| Method | Status | Selected candidate | Mean $E_U$ | Eligible / total | Run commit |
|---|---|---|---:|---:|---|
| NegGrad | SELECTED | `neggrad_w0.05_s25` | 0.001857197 | 15 / 20 | `debb058a4242` |
| PCGrad | SELECTED | `pcgrad_lr0.0005_s25` | 0.003400070 | 6 / 12 | `debb058a4242` |
| IFRU | SELECTED | `ifru_p512_d0.1` | 0.000488145 | 6 / 6 | `debb058a4242` |
| E2URec | SELECTED | `e2urec_w0.2_s200` | 0.009446364 | 8 / 9 | `debb058a4242` |
| SISA | NO_ELIGIBLE_CANDIDATE | `--` | -- | 0 / 9 | `b593e45dbc00` |
| RecEraser | SELECTED | `receraser_lr0.0005_e3` | 0.008620742 | 7 / 9 | `b593e45dbc00` |

## Complete candidate inventory

| Method | Candidate | Mean $E_U$ | Eligible | Selected | Compute | Failure reasons |
|---|---|---:|---|---|---:|---|
| NegGrad | `neggrad_w0.05_s25` | 0.001857197 | yes | yes | 25 | -- |
| NegGrad | `neggrad_w0.05_s50` | 0.004748216 | yes | no | 50 | -- |
| NegGrad | `neggrad_w0.05_s100` | 0.007807112 | yes | no | 100 | -- |
| NegGrad | `neggrad_w0.05_s200` | 0.012650028 | yes | no | 200 | -- |
| NegGrad | `neggrad_w0.1_s25` | 0.002125367 | yes | no | 25 | -- |
| NegGrad | `neggrad_w0.1_s50` | 0.005429710 | yes | no | 50 | -- |
| NegGrad | `neggrad_w0.1_s100` | 0.009907809 | yes | no | 100 | -- |
| NegGrad | `neggrad_w0.1_s200` | 0.018958562 | yes | no | 200 | -- |
| NegGrad | `neggrad_w0.2_s25` | 0.002937183 | yes | no | 25 | -- |
| NegGrad | `neggrad_w0.2_s50` | 0.007104725 | yes | no | 50 | -- |
| NegGrad | `neggrad_w0.2_s100` | 0.015091194 | yes | no | 100 | -- |
| NegGrad | `neggrad_w0.2_s200` | 0.035493728 | no | no | 200 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| NegGrad | `neggrad_w0.4_s25` | 0.004523160 | yes | no | 25 | -- |
| NegGrad | `neggrad_w0.4_s50` | 0.010356700 | yes | no | 50 | -- |
| NegGrad | `neggrad_w0.4_s100` | 0.025734590 | no | no | 100 | L8:utility_or_collapse |
| NegGrad | `neggrad_w0.4_s200` | 0.043765587 | no | no | 200 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| NegGrad | `neggrad_w0.8_s25` | 0.006473873 | yes | no | 25 | -- |
| NegGrad | `neggrad_w0.8_s50` | 0.014607955 | yes | no | 50 | -- |
| NegGrad | `neggrad_w0.8_s100` | 0.038863036 | no | no | 100 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| NegGrad | `neggrad_w0.8_s200` | 0.052681930 | no | no | 200 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| PCGrad | `pcgrad_lr0.0005_s25` | 0.003400070 | yes | yes | 25 | -- |
| PCGrad | `pcgrad_lr0.0005_s50` | 0.007446321 | yes | no | 50 | -- |
| PCGrad | `pcgrad_lr0.0005_s100` | 0.017012181 | yes | no | 100 | -- |
| PCGrad | `pcgrad_lr0.0005_s200` | 0.037694211 | no | no | 200 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| PCGrad | `pcgrad_lr0.001_s25` | 0.006946476 | yes | no | 25 | -- |
| PCGrad | `pcgrad_lr0.001_s50` | 0.015610900 | yes | no | 50 | -- |
| PCGrad | `pcgrad_lr0.001_s100` | 0.040076284 | no | no | 100 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| PCGrad | `pcgrad_lr0.001_s200` | 0.052314052 | no | no | 200 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| PCGrad | `pcgrad_lr0.002_s25` | 0.014473655 | yes | no | 25 | -- |
| PCGrad | `pcgrad_lr0.002_s50` | 0.035789504 | no | no | 50 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| PCGrad | `pcgrad_lr0.002_s100` | 0.050443030 | no | no | 100 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| PCGrad | `pcgrad_lr0.002_s200` | 0.067887062 | no | no | 200 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| IFRU | `ifru_p512_d0.001` | 0.022149724 | yes | no | 53 | -- |
| IFRU | `ifru_p512_d0.01` | 0.003475169 | yes | no | 38 | -- |
| IFRU | `ifru_p512_d0.1` | 0.000488145 | yes | yes | 22 | -- |
| IFRU | `ifru_p1024_d0.001` | 0.028788734 | yes | no | 53 | -- |
| IFRU | `ifru_p1024_d0.01` | 0.004879318 | yes | no | 43 | -- |
| IFRU | `ifru_p1024_d0.1` | 0.000857232 | yes | no | 24 | -- |
| E2URec | `e2urec_w0.2_s200` | 0.009446364 | yes | yes | 1400 | -- |
| E2URec | `e2urec_w0.2_s500` | 0.011166604 | yes | no | 1700 | -- |
| E2URec | `e2urec_w0.2_s1000` | 0.014580465 | yes | no | 2200 | -- |
| E2URec | `e2urec_w0.4_s200` | 0.015853832 | yes | no | 1400 | -- |
| E2URec | `e2urec_w0.4_s500` | 0.016377692 | yes | no | 1700 | -- |
| E2URec | `e2urec_w0.4_s1000` | 0.018976603 | yes | no | 2200 | -- |
| E2URec | `e2urec_w0.6_s200` | 0.017888592 | yes | no | 1400 | -- |
| E2URec | `e2urec_w0.6_s500` | 0.019523687 | yes | no | 1700 | -- |
| E2URec | `e2urec_w0.6_s1000` | 0.020406547 | no | no | 2200 | L8:utility_or_collapse |
| SISA | `sisa_lr0.0001_e1` | 0.217382890 | no | no | 377 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| SISA | `sisa_lr0.0001_e3` | 0.154199398 | no | no | 1131 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| SISA | `sisa_lr0.0001_e5` | 0.158045714 | no | no | 1885 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| SISA | `sisa_lr0.0005_e1` | 0.208708140 | no | no | 377 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| SISA | `sisa_lr0.0005_e3` | 0.158892899 | no | no | 1131 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| SISA | `sisa_lr0.0005_e5` | 0.152798442 | no | no | 1885 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| SISA | `sisa_lr0.001_e1` | 0.227669039 | no | no | 377 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| SISA | `sisa_lr0.001_e3` | 0.183109267 | no | no | 1131 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| SISA | `sisa_lr0.001_e5` | 0.191620586 | no | no | 1885 | L8:utility_or_collapse; L4M4:utility_or_collapse |
| RecEraser | `receraser_lr0.0005_e1` | 0.013678613 | yes | no | 400 | -- |
| RecEraser | `receraser_lr0.0005_e3` | 0.008620742 | yes | yes | 1200 | -- |
| RecEraser | `receraser_lr0.0005_e5` | 0.012571913 | yes | no | 2000 | -- |
| RecEraser | `receraser_lr0.001_e1` | 0.012629145 | yes | no | 400 | -- |
| RecEraser | `receraser_lr0.001_e3` | 0.016496481 | yes | no | 1200 | -- |
| RecEraser | `receraser_lr0.001_e5` | 0.028253424 | yes | no | 2000 | -- |
| RecEraser | `receraser_lr0.002_e1` | 0.009806211 | yes | no | 400 | -- |
| RecEraser | `receraser_lr0.002_e3` | 0.032252828 | no | no | 1200 | L4M4:utility_or_collapse |
| RecEraser | `receraser_lr0.002_e5` | 0.054293786 | no | no | 2000 | L8:utility_or_collapse; L4M4:utility_or_collapse |

The JSON companion freezes per-scenario metrics, hyperparameters and hashes for every candidate.
