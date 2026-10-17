# Stage20: Amazon seed42 calibration and three-request comparison

This stage restores the historical Amazon exploratory baseline comparison.
BOTA uses the already selected Amazon response strength **48**, seed **42**,
and a source-only **0** ablation. There is no BOTA parameter reselection.
Original and Exact-Masked are references; neither is tuned.

Stage18/19 were already inspected before specifying this stage. The new pairs
were selected using metadata without model scores. This is an exploratory
single-seed evaluation on reused Development data, not a new independent
holdout or a replacement for the completed multi-seed sensitivity study.

## Frozen requests and split

Baseline calibration uses historical K2 and K4. One common configuration per
method is selected jointly across these two requests. Comparison uses:

| Request | Low-frequency user | Middle-frequency user | Masked slots | Local Development samples |
|---|---:|---:|---:|---:|
| A1 | 702 | 758 | 4 | 10 |
| A2 | 302 | 354 | 4 | 10 |
| A3 | 132 | 64 | 4 | 10 |

All six users are disjoint from all Stage18/19 requested users. They are the
first three remaining pairs in the existing Stage19 metadata hash order, with
two adaptation exposures and at least five Development samples per user.

All calibration and comparison requested users are excluded from Dev-Fit.
The other Development examples use a fixed hash parity partition. The current
metadata produces **1,879 Dev-Fit / 1,961 Dev-Select** examples. Every local
requested-user sample remains in Dev-Select. All methods use this identical
projection, including aggregate AUC/LogLoss. RecEraser attention fits Dev-Fit
only. These aggregate metrics therefore describe Dev-Select, not the full
3,840-example Development set, and must be labeled accordingly in the paper.

## Fixed search grids

| Baseline | Grid | Configurations |
|---|---|---:|
| IFRU | relative damping 0.001/0.01/0.1; curvature panel 512/1024; max CG 40 | 6 |
| E2URec | forget weight 0.2/0.4/0.6; student steps 200/500/1000; teacher 1200 steps | 9 |
| NegGrad | forget weight 0.05/0.1/0.2/0.4/0.8; steps 25/50/100/200 | 20 |
| PCGrad | learning rate 0.0005/0.001/0.002; steps 25/50/100/200 | 12 |
| SISA | learning rate 0.0001/0.0005/0.001; 1/3/5 epochs per slice; four shards/four slices | 9 |
| RecEraser | learning rate 0.0005/0.001/0.002; 1/3/5 epochs; four shards | 9 |

There are **65 candidates / 130 calibration cells**, followed by **30 comparison
cells** (ten method/reference/ablation rows times three requests).
Post-hoc fixed-coordinate methods start from the sealed formal Stage18v2
Original endpoint. SISA and RecEraser preserve their respective partitioned
training definitions and disclose the model-coordinate difference.

Selection minimizes equally weighted K2/K4 mean absolute counterfactual error
E_U, subject to both requests passing:

- AUC damage <= 0.005;
- LogLoss damage <= 0.01;
- finite predictions and no probability collapse.

Ties use compute count then candidate ID. All six decisions must be frozen
before any comparison endpoint is trained. No A1/A2/A3 score participates in
selection. The entire scan is retained.

If a baseline has no eligible candidate, it is reported as
`NO_ELIGIBLE_CANDIDATE`. Its lowest-error finite candidate is additionally
evaluated on A1/A2/A3 with `diagnostic_only=true`. Such a row must not be called
a successfully calibrated baseline. If all candidates are nonfinite, the
three explicit failed rows remain in the comparison and no successful-only
mean is calculated.

## Running offline

From the active `(pytorch)` PowerShell in `C:\python\BOTA`:

```powershell
.\scripts\bota_if\run_ali_stage20_amazon_baselines_v1.ps1 `
  -Mode Full `
  -Workspace C:\python\E2URec-main
```

`Full / All` freezes the protocol, calibrates all six baselines, seals all
selections, constructs BOTA for the new users, trains/evaluates frozen baseline
configurations, and publishes the complete summary. Rerun the same command
after interruption. Successfully sealed units are verified and reused.

Optional short checks:

```powershell
.\scripts\bota_if\run_ali_stage20_amazon_baselines_v1.ps1 -Mode Preflight
.\scripts\bota_if\run_ali_stage20_amazon_baselines_v1.ps1 -Mode Smoke
```

`Preflight` does not load a model. `Smoke` runs one canonical optimizer step,
eight inference samples, adapter save/reload and partition configuration
checks. Its outputs are isolated from formal comparison metrics.

Optional phase/method commands:

```powershell
# Calibrate one method at a time. Complete all six before Comparison.
.\scripts\bota_if\run_ali_stage20_amazon_baselines_v1.ps1 `
  -Mode Full -Phase Calibration -Method IFRU

# This requires all six frozen baseline selections.
.\scripts\bota_if\run_ali_stage20_amazon_baselines_v1.ps1 `
  -Mode Full -Phase Comparison -Method All

# Read the sealed summary or report the missing units.
.\scripts\bota_if\run_ali_stage20_amazon_baselines_v1.ps1 -Mode Analyze
```

Use the installed PyTorch environment. `$env:BOTA_PYTHON` can override the
Python executable if necessary. All paths may be overridden through
`-Workspace` as in earlier stages.

## Resources, checkpoints and retained artifacts

- One GPU, sequential execution. No simultaneous second Stage20 run.
- At least 32 GiB free disk is checked before freezing the protocol. Temporary
  SISA full-model slice checkpoints account for most peak disk usage.
- Gradient methods reuse checkpoint predictions within a training trajectory;
  a failed unfinished trajectory restarts, completed trajectories are reused.
- Partitioned training resumes after a completed shard. Stage20 stores CPU/CUDA
  RNG state alongside the shard boundary to reproduce subsequent stochastic
  training. An unfinished shard restarts.
- Partition candidate model tensors are removed after the unit is sealed.
  Predictions, configurations, manifests, tensor hashes, partition plans and
  timing records are retained. The selected configurations are retrained for
  the three comparison requests. This avoids accumulating hundreds of GiB.
- Cleanup is restricted to new Stage20 component directories. Historical
  artifacts and Stage18/19 results are preserved.
- Interrupted diagnostic files remain in `.work`. Completed model units and
  predictions are immutable and hash-verified.
- Actual initial and affected-shard optimizer counts are checked against the
  executable plan. The inherited partition engine's static budget is retained
  as an upper bound and is not substituted for measured optimizer calls.

Source/configuration hashes and input hashes bind the frozen protocol.
Changing implementation/configuration after formal execution has started is
rejected rather than silently mixing runs.

## Outputs

Under `C:\python\E2URec-main\outputs\ali_stage20_amazon_baselines_v1`:

- `protocol/result.json`: requests, split, input and source hashes;
- `selections/<method>/result.json`: all calibration rows and selected settings;
- `all_selections/result.json`: all six decisions frozen before comparison;
- `comparison/<method>/result.json`: three comparison rows per method;
- `summary/result.json`: complete metrics and request-level comparison counts;
- `summary/report.md`: readable results;
- `summary/calibration_candidates.csv`: all 130 scan cells;
- `summary/comparison.csv`: all 30 final comparison cells.

Standard deviations in the summary are **across three requests at seed42**,
not across random seeds. The mean residual is the mean of per-request ratios;
it should not be confused with the ratio of the two mean errors. No ranking,
attack, long-horizon or population-scale claim is added by this stage.
