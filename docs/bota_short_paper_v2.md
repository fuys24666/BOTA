# BOTA short-window paper study v2

This Development-only study keeps the frozen L8, L4M4 and L3M3H2 request registry. It never accesses FinalTest.

Two references are reported separately. `Original-Short-200step -> Exact-Masked-Reference-200step` measures local trajectory reconstruction. `FullControl-P5-Short -> Retain-Retrain-P5-Short` measures convergence under matched zero-Adapter initialization, optimizer, data order, Development validation and five consecutive non-improving complete epochs. These residuals are never pooled.

The P5 arms use fixed-A/QV-LoRA B-only coordinates, logical/effective batch 16 implemented as physical microbatch 4 with four-way exact weighted gradient accumulation, AdamW at 0.001, a maximum of 100 epochs, strict validation-loss improvement, and restoration of the best endpoint. Development inference also uses batch 4. FullControl trains on all 3,200 window rows. Each Retain arm trains on its scenario-specific compact 3,192 rows.

`NegGrad-Mixed-Short-BOnly` is a disclosed short-window adaptation of the E2URec mixed objective: `CE(Retain)-0.2*CE(Forget)`, initialized from the frozen Original-Short endpoint and run for 200 online steps. It is not the 3,000-step full-model `ng2` artifact. `PCGrad-Short-BOnly` uses the same endpoint and budget with deterministic symmetric two-task PCGrad over negative Forget CE and positive Retain CE. It is not the full-request `pc1` artifact.

All formal runs require clean Git, one CUDA device, immutable v1 request artifacts and atomically publish to `outputs/bota_short_paper_v2`.

## Independent scenario runs and timing

Every method entry point accepts `-Scenario L8` or `-Scenario L4M4` (the legacy default remains `All`). A single-scenario run publishes only that frozen request and must use a unique RunName. This prevents L8 and L4M4 elapsed times from being pooled.

Each scenario publishes `phase_timing.json` with the common schema `bota-short-phase-timing-v1`. The common fields are initialization, offline construction, online compute, Adapter publication, online total and end-to-end seconds. BOTA additionally separates its 200-step canonical trajectory/transport-bank construction from request-time vector composition. IFRU separates its shared endpoint/curvature-panel construction from request-time Forget gradient, lambda estimation, CG solve and candidate reconstruction. SISA and RecEraser currently expose only component-level wall time; their Adapter publication field is therefore null and explicitly marked as already included in online compute rather than silently estimated.

The v2 evaluation accepts the same `-Scenario` argument and publishes `efficiency.csv` alongside scientific metrics. It rejects a model run that does not contain the requested scenario.
