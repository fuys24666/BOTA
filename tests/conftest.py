from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(items):
    """Skip tests that require large, intentionally untracked paper artifacts."""
    required = {
        "test_predecessor_and_full_coordinate_are_frozen":
            ROOT / "outputs/bota_if_v1/p2a_user_sparse_module_oracle_audits",
        "test_goodreads_k2_registry_is_outcome_blind_and_exactly_two_exposures":
            ROOT / "outputs/bota_goodreads_v1/prepared/goodreads_comics_seed42_v1/manifest.json",
        "test_synthetic_cli_loads_no_real_model":
            ROOT / "data/ml-1m/raw_data/users.dat",
        "test_p1_runtime_resolves_strict_t5_base_config_at_second_level":
            ROOT / "outputs/ru1/i02s42v1/configs/base_t5.yaml",
    }
    for item in items:
        path = required.get(item.name)
        if path is not None and not path.exists():
            item.add_marker(pytest.mark.skip(reason=f"external paper artifact not installed: {path}"))
