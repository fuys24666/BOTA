# Ali Stage 6A storage and compression screen

Stage 6A builds a nested 256-user impact bank for each seed in 32-user chunks.
Every repeated 200-step trajectory is counted. It measures FP32, FP16,
per-vector and per-block INT8, three top-q levels and shared low-rank ranks
4--64 at bank sizes 12, 32, 64, 128 and 256.

The complete fixed grid is reported. Candidate selection occurs only at K=256
and requires at least 16x measured compression with no more than 10% parameter
residual increase relative to FP32 BOTA for L8 and L4M4 in every seed. Stage 6A
does not run Development inference and cannot by itself establish behavioral
fidelity. If a candidate exists, it is frozen for the later Stage 6B behavioral
confirmation.

```powershell
.\scripts\bota_if\run_ali_stage6_storage_screen_v1.ps1 `
  -Mode Full `
  -Seed All `
  -Workspace C:\python\E2URec-main
```
