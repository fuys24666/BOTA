# Ali Stage 4: matched-user MIA verification

Stage 4 is the preregistered Development-only attack experiment. It uses each
registered forget row as a member and Development rows from the same registered
user as nonmembers. FinalTest is forbidden.

The primary score is negative answer loss. Confidence, negative entropy and
absolute yes/no margin are mandatory secondary attacks. Every attack is reported
for all three seeds, both request scenarios and all ten endpoints. Retain-Retrain-P5
is a separately labelled sensitivity endpoint and is not the Exact-Masked
counterfactual denominator.

The paired authoritative-user bootstrap tests (1) whether Original has at least
0.02 more absolute attack advantage than Exact-Masked and (2) whether BOTA is
closer to Exact-Masked than Original. The primary cell also requires the frozen
Development utility guardrails.

Run from `C:\python\BOTA` in the `pytorch` environment:

```powershell
.\scripts\bota_if\run_ali_stage4_attack_v1.ps1 `
  -Mode Full `
  -Seed All `
  -Workspace C:\python\E2URec-main
```

The launcher executes all three seeds and then publishes the six-cell summary.
It refuses formal execution if the BOTA Git worktree is dirty or any frozen
config/evaluation manifest differs from its registered hash.

