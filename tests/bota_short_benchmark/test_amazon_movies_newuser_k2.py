from pathlib import Path

from src.bota_short_benchmark.amazon_movies_k2_all_evaluation import ORDER
from src.bota_short_benchmark.amazon_movies_newuser_prepare import load_config as load_prepare
from src.bota_short_benchmark.amazon_movies_titles_original_p5 import load_config as load_original
from src.bota_short_benchmark.protocol import load_config as load_benchmark


ROOT = Path(__file__).resolve().parents[2]


def test_amazon_cohorts_are_disjoint_and_development_only() -> None:
    config = load_prepare(ROOT / "configs/bota_amazon_movies_newuser_prepare_v4.yaml")
    assert config["protocol"]["historical_users"] == 256
    assert config["protocol"]["adaptation_users"] == 768
    assert config["scientific_scope"]["historical_and_adaptation_users_disjoint"] is True
    assert config["scientific_scope"]["final_test_created"] is False


def test_amazon_k2_request_is_low_middle_and_short_window() -> None:
    config = load_benchmark(ROOT / "configs/bota_short_amazon_movies_newuser_k2k4_v4.yaml")
    protocol = config["protocol"]
    assert protocol["optimizer_steps"] == 200
    assert protocol["interactions"] == 3200
    assert protocol["minimum_window_exposure_per_user"] == 2
    assert protocol["maximum_window_exposure_per_user"] == 5
    assert protocol["scenarios"][0] == {
        "id": "K2",
        "composition": {"low": 1, "middle": 1, "high": 0},
    }
    assert protocol["selection_uses_labels_or_predictions"] is False
    assert config["evaluation"]["final_test"] is False


def test_amazon_historical_original_and_nine_method_panel_are_frozen() -> None:
    original = load_original(ROOT / "configs/bota_amazon_movies_titles_original_p5_v3.yaml")
    assert original["source"]["train_samples"] == 4415
    assert original["source"]["development_samples"] == 1280
    assert original["scientific_scope"]["final_test_access"] is False
    assert ORDER == [
        "Original-Short", "Retrain-Short", "BOTA-T2-Short", "IFRU-Short-LoRA",
        "E2URec-Short-FixedAB", "NegGrad-Mixed-Short-BOnly",
        "PCGrad-Short-BOnly", "SISA-Short-T5", "RecEraser-Adapter-Short",
    ]


def test_amazon_launcher_is_portable() -> None:
    launcher = (ROOT / "scripts/bota_if/run_amazon_movies_newuser_k2_v4.ps1").read_text(encoding="utf-8")
    assert "C:\\Users\\" not in launcher
    assert "$env:BOTA_PYTHON" in launcher
    assert "FinalTest" not in launcher
