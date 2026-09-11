from __future__ import annotations

from pathlib import Path

import yaml

from src.bota_short_benchmark.ali_stage2_stronger_signal_v1 import (
    REQUEST_SIZES,
    STRATA,
    build_request_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stage2_config_preserves_amendment_contract() -> None:
    value = yaml.safe_load(
        (ROOT / "configs/ali_stage2_stronger_signal_v1.yaml").read_text(encoding="utf-8")
    )
    assert value["test_access_policy"] == "forbidden"
    assert value["stage1_selection"]["selected_arm"] == "FS_empirical_fisher"
    assert value["protocol"]["selection_seed"] == 42
    assert value["protocol"]["request_sizes"] == [2, 4, 8]
    assert value["protocol"]["selection_uses_labels_or_predictions"] is False
    assert value["protocol"]["report_every_request"] is True
    assert value["evaluation"]["split"] == "Development"
    assert value["evaluation"]["final_test"] is False
    assert value["decision"] == {
        "adequate_effect_du_minimum": 0.001,
        "secondary_only": True,
        "may_replace_primary_scenarios": False,
    }


def test_request_registry_is_deterministic_nested_and_public() -> None:
    train_users: list[int] = []
    selected_users: list[int] = []
    next_user = 1
    for exposure in ([1] * 8 + [2] * 8 + [8] * 8):
        train_users.extend([next_user] * exposure)
        selected_users.append(next_user)
        next_user += 1
    while len(train_users) < 3200:
        train_users.append(next_user)
        next_user += 1
    development_users = [user for user in selected_users for _ in range(10)]
    protocol = {
        "exposure_strata": {
            "single": {"minimum": 1, "maximum": 1},
            "moderate": {"minimum": 2, "maximum": 5},
            "high": {"minimum": 8, "maximum": None},
        },
        "minimum_development_examples_per_user": 10,
        "ordering": "ascending_sha256_public_user_association_within_stratum",
    }
    left = build_request_registry(train_users, development_users, list(range(3200)), protocol)
    right = build_request_registry(train_users, development_users, list(range(3200)), protocol)
    assert left == right
    public, private, union_users = left
    assert len(private) == len(STRATA) * len(REQUEST_SIZES) == 9
    assert public["selection_uses_labels_or_predictions"] is False
    assert public["eligibility_uses_development_labels"] is False
    assert public["union_users"] == len(union_users) == 24
    assert all("users" not in request for request in public["requests"])
    for stratum in STRATA:
        rows = [request for request in private if request["stratum"] == stratum]
        assert [row["request_size"] for row in rows] == list(REQUEST_SIZES)
        assert rows[0]["users"] == rows[1]["users"][:2]
        assert rows[1]["users"] == rows[2]["users"][:4]


def test_launcher_runs_amendment_validation_before_formal_execution() -> None:
    text = (ROOT / "scripts/bota_if/run_ali_stage2_stronger_signal_v1.ps1").read_text(
        encoding="utf-8"
    )
    assert 'if($Mode -eq "Full")' in text
    assert "validate_ali_stage0_amendment01_v1.py" in text
    assert "ali_stage2_stronger_signal_v1" in text
