# Amendment 07: BOTA response-strength Development sweep

Amendment 07 was frozen after the complete Stage 7A baseline audit and before
any response-strength candidate was constructed or evaluated. It does not
modify the baseline selections.

The experiment keeps the formal 200-step BOTA trajectory, registered-user
source, fixed Q/V LoRA coordinate and full-state AdamW tangent unchanged. It
varies only the multiplier on the blockwise empirical-Fisher response term:
`0`, `0.25`, `0.5`, `0.75`, and `1`. Scale zero is FS source-only transport;
scale one must reproduce the frozen formal BOTA endpoint within the registered
numerical gate.

One scale is selected jointly for L8 and L4M4 using only the frozen Dev-Select
partition. The utility guardrails and primary mean-absolute counterfactual
error are identical to Stage 7A. FinalTest and all attack, long-horizon,
sequential and storage outcomes are excluded from selection.
