# Ali feedback experiment records

The records are additive and must be read in this order:

1. `docs/ALI_STAGE0_PREREGISTRATION.md` and
   `configs/ali_stage0_preregistration_v1.json` freeze the primary attack,
   sequential and scale contracts.
2. `docs/audits/ALI_STAGE0_ARTIFACT_AUDIT.md` and
   `docs/audits/ali_stage0_artifact_audit_v1.json` record the read-only artifact
   audit completed before new experiments.
3. `docs/ALI_STAGE0_AMENDMENT01.md` and
   `configs/ali_stage0_amendment01_v1.json` add mechanism stabilization,
   stronger-signal secondary analysis and fixed-horizon equal-schedule
   sensitivity without replacing the original contracts.
4. `docs/audits/ALI_STAGE0_AMENDMENT01_VALIDATION.md` and
   `docs/audits/ali_stage0_amendment01_validation_v1.json` record the amendment
   validation and file hashes before amended experiments.

Later amendments must be appended here; historical records must not be edited
to make later decisions appear preregistered.
