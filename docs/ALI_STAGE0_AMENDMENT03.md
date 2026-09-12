# Ali Stage 0 Amendment 03

This amendment freezes the secondary stronger-signal attack analysis before any
Stage 4A attack result is produced. It does not modify the completed parent
Stage 4 result, which remains negative and authoritative for L8/L4M4.

Stage 4A reuses the nine outcome-blind requests and the frozen
FS_empirical_fisher endpoints produced in Stage 2. Original, Exact-Masked and
BOTA are evaluated with the same matched-user attacks as Stage 4. All nine
requests and all four attacks are reported. The Stage 2 DU threshold was fixed
before attack execution; only high_k4 passed it and is eligible for formal
interpretation. Any success is secondary, seed-42 and high-frequency K=4 only.

If high_k4 lacks an adequate Original-to-Exact attack signal, attack optimization
under this protocol stops. FinalTest remains forbidden.
