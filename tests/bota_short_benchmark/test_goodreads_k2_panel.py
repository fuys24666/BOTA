from pathlib import Path

from src.bota_short_benchmark.protocol import freeze_registry, load_config
from src.bota_short_benchmark.runner import allocator_fraction_for
from src.bota_short_benchmark.timing import SCENARIO_CHOICES

ROOT = Path(__file__).resolve().parents[2]


def test_goodreads_k2_protocol_is_frozen_and_uses_trained_original():
    config = load_config(ROOT / "configs/bota_short_goodreads_k2_v1.yaml")
    assert config["protocol"]["request_users"] == 2
    assert config["protocol"]["scenarios"] == [{"id": "K2", "composition": {"low": 2, "middle": 0, "high": 0}}]
    assert "recommendation_originals/goodreads_recommendation_original_seed42_v2/model" in config["source"]["original_checkpoint"]
    assert config["evaluation"]["final_test"] is False and "K2" in SCENARIO_CHOICES


def test_goodreads_k2_registry_is_outcome_blind_and_exactly_two_exposures():
    config = load_config(ROOT / "configs/bota_short_goodreads_k2_v1.yaml"); registry = freeze_registry(ROOT, config); scenario = registry["scenarios"][0]
    assert scenario["id"] == "K2" and len(scenario["users"]) == 2
    assert scenario["deleted_interactions"] == 2 and scenario["actual_window_ratio"] == 2 / 3200
    assert len(scenario["forget_train_indices"]) == 2 and len(set(scenario["users"])) == 2
    assert registry["selection_uses_labels_or_predictions"] is False


def test_goodreads_k2_wrappers_are_development_only():
    text = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in ("scripts/bota_if/run_goodreads_k2_prepare_v1.ps1", "scripts/bota_if/run_goodreads_k2_method_v1.ps1", "scripts/bota_if/run_goodreads_k2_evaluation_v1.ps1"))
    assert "FinalTest" not in text and "configs/bota_short_goodreads_k2_v1.yaml" in text


def test_allocator_fraction_respects_hard_reserved_cap_with_margin():
    fraction = allocator_fraction_for(15.92, .88, 14.0)
    assert fraction < .88
    assert fraction * 15.92 <= 13.9 + 1e-12
