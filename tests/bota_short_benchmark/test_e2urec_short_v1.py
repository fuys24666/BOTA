from pathlib import Path

from src.bota_short_benchmark import e2urec_short_v1


ROOT = Path(__file__).resolve().parents[2]


def test_e2urec_frozen_budget_and_scope() -> None:
    assert e2urec_short_v1.SEEDS == (41, 42, 43)
    assert e2urec_short_v1.SCENARIOS == ("L8", "L4M4")
    assert e2urec_short_v1.TEACHER_STEPS == 1200
    assert e2urec_short_v1.STUDENT_STEPS == 1000
    assert e2urec_short_v1.EFFECTIVE_BATCH == 16
    assert e2urec_short_v1.MICROBATCH == 4


def test_e2urec_launcher_is_machine_independent() -> None:
    launcher = (ROOT / "scripts/bota_if/run_short_e2urec_multiseed_v1.ps1").read_text(
        encoding="utf-8"
    )
    assert "BOTA_PYTHON" in launcher
    assert "Administrator\\anaconda3" not in launcher
    assert "PrimaryFinalTestRunName" in launcher


def test_documented_finaltest_path_matches_evaluator() -> None:
    expected = "data/ml-1m/proc_data/data/test/test_10_simple.json"
    evaluator = (ROOT / "src/bota_short_benchmark/multiseed_finaltest_v3.py").read_text(
        encoding="utf-8"
    )
    artifacts = (ROOT / "docs/ARTIFACTS.md").read_text(encoding="utf-8")
    assert expected in evaluator
    assert expected in artifacts
