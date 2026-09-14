from src.bota_short_benchmark.ali_stage7_result_audit_v1 import RUN_NAMES
from src.bota_short_benchmark.ali_stage7_baseline_sweep_v1 import METHODS


def test_every_stage7_method_has_one_frozen_run_name():
    assert tuple(RUN_NAMES) == METHODS
    assert len(set(RUN_NAMES.values())) == len(METHODS)
    assert RUN_NAMES["RecEraser"] == "ali_stage7a_receraser_seed42_v1"
