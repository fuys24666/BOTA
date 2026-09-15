from src.bota_short_benchmark.ali_stage8_result_audit_v1 import RUN_NAME, SCHEMA


def test_stage8_audit_identity_is_frozen():
    assert SCHEMA == "bota-ali-stage8-result-audit-v1"
    assert RUN_NAME == "ali_stage8_bota_response_seed42_v1"
