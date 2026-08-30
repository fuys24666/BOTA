from pathlib import Path

from src.bota_short_benchmark import paper_v2


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/bota_short_paper_goodreads_k2_v2.yaml"


def test_goodreads_k2_paper_panel_matches_ml1m_method_contract():
    config = paper_v2.load_config(CONFIG)
    assert config["source"] == {
        "v1_config": "configs/bota_short_goodreads_k2_v1.yaml",
        "v1_benchmark_name": "goodreads_k2_short_seed42_v1",
    }
    assert config["protocol"]["deleted_samples"] == 2
    assert config["protocol"]["patience"] == 5
    assert config["protocol"]["neggrad_steps"] == 200
    assert config["protocol"]["pcgrad_steps"] == 200
    assert config["evaluation"]["final_test"] is False


def test_paper_v2_manifest_uses_frozen_scenario_cardinality(tmp_path):
    scenario = {"id": "K2", "request_hash": "abc", "deleted_interactions": 2}
    paper_v2._scenario_manifest(tmp_path, "method", scenario, {"sha256": "def"}, {})
    import json
    value = json.loads((tmp_path / "scenario_manifest.json").read_text(encoding="utf-8"))
    assert value["deleted_interactions"] == 2 and value["test_accessed"] is False


def test_goodreads_k2_paper_wrappers_freeze_k2_and_never_name_finaltest():
    paths = (
        "scripts/bota_if/run_goodreads_k2_paper_method_v2.ps1",
        "scripts/bota_if/run_goodreads_k2_paper_evaluation_v2.ps1",
    )
    text = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in paths)
    assert '"K2"' in text
    assert "configs/bota_short_paper_goodreads_k2_v2.yaml" in text
    assert "FinalTest" not in text
