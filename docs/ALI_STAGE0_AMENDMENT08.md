# Amendment 08: cross-seed Development confirmation

Amendment 08 was frozen after the complete Stage 7 baseline audit and Stage 8
BOTA response-strength audit, and before any held-out-seed confirmation run.
It fixes `bota_r1` and the five eligible selected baseline configurations.
There is no further tuning by seed or request scenario.

Seed 42 is the selection seed. Seeds 41 and 43 are held-out confirmation
seeds. All three seeds are reported, but only the four held-out
seed-by-scenario cells enter the binary stability decision. BOTA must pass the
utility guardrails and have local residual below one in every held-out cell;
its unweighted held-out mean $E_U$ must also be lower than that of each frozen
eligible baseline.

Stage 7 found no eligible SISA candidate, so Stage 9 does not promote or rerun
a failed SISA setting. FinalTest is forbidden. A stable result only authorizes
creation of a separate, dated FinalTest preregistration; it does not authorize
opening FinalTest under Amendment 08.
