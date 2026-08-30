import json
from pathlib import Path

import numpy as np
import pytest

from src.bota_short_benchmark import t012_behavior_ablation as audit


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/bota_short_t012_behavior_ablation_v1.yaml"


def _prediction(values):
    return {"probability": values, "gold_label": [0, 1, 0, 1]}


def test_frozen_protocol_and_scope():
    config = audit.load_config(CONFIG)
    assert config["protocol"]["optimizer_steps"] == 200
    assert config["protocol"]["paired_reference_optimizer_steps"] == 200
    assert config["protocol"]["physical_optimizer_step_calls"] == 400
    assert config["protocol"]["authoritative_optimizer_steps_committed"] == 0
    assert config["protocol"]["variants"] == list(audit.VARIANTS)
    assert config["protocol"]["scenarios"] == ["L8", "L4M4"]
    assert config["evaluation"]["final_test"] is False


def test_windows_cuda_counter_uses_current_device_overload():
    source = (ROOT / "src/bota_short_benchmark/t012_behavior_ablation.py").read_text(encoding="utf-8")
    assert "torch.cuda.set_device(0)" in source
    assert "torch.cuda.reset_peak_memory_stats()" in source
    assert "torch.cuda.max_memory_reserved()" in source
    assert "reset_peak_memory_stats(device)" not in source


def test_canonical_transport_uses_formal_batch_backward():
    source = (ROOT / "src/bota_short_benchmark/t012_behavior_ablation.py").read_text(encoding="utf-8")
    body = source.split("def run_canonical_formal", 1)[1].split("def _predict_reference", 1)[0]
    assert "formal_loss = torch.sum(losses) / len(chosen)" in body
    assert "formal_loss.backward()" in body
    assert "parameter.grad =" not in body
    assert "retain_graph=True" in body


def test_powershell_wrapper_omits_empty_optional_arguments():
    source = (ROOT / "scripts/bota_if/run_short_t012_behavior_ablation_v1.ps1").read_text(encoding="utf-8")
    assert 'if($BenchmarkName){$Arguments += @("--benchmark-name", $BenchmarkName)}' in source
    assert 'if($OriginalRunName){$Arguments += @("--original-run-name", $OriginalRunName)}' in source
    assert 'if($ExactMaskedRunName){$Arguments += @("--exact-masked-run-name", $ExactMaskedRunName)}' in source
    assert "--benchmark-name $BenchmarkName" not in source


def test_config_rejects_budget_or_test_mutation(tmp_path):
    text = CONFIG.read_text(encoding="utf-8")
    for changed in (text.replace("optimizer_steps: 200", "optimizer_steps: 201"), text.replace("final_test: false", "final_test: true")):
        path = tmp_path / f"bad-{len(list(tmp_path.iterdir()))}.yaml"
        path.write_text(changed, encoding="utf-8")
        with pytest.raises(ValueError):
            audit.load_config(path)


def test_behavior_metrics_exact_and_original_boundaries():
    original = _prediction([.1, .3, .7, .9])
    exact = _prediction([.2, .4, .6, .8])
    users = [1, 1, 2, 2]
    exact_row = audit.behavior_metrics("exact", exact, exact, original, range(4), users, 20, 1)
    original_row = audit.behavior_metrics("original", original, exact, original, range(4), users, 20, 1)
    assert exact_row["local_residual"] == pytest.approx(0.0)
    assert exact_row["yes_no_jsd_to_exact"] == pytest.approx(0.0)
    assert original_row["local_residual"] == pytest.approx(1.0)
    assert np.isfinite(exact_row["ci_lower"])
    assert np.isfinite(exact_row["ci_upper"])


def test_coordinate_equality_is_exact():
    import torch
    left = {"A": {"x": torch.tensor([1.0])}}
    right = {"A": {"x": torch.tensor([1.0])}}
    assert audit._same_coordinate(left, right)
    right["A"]["x"][0] = 2
    assert not audit._same_coordinate(left, right)


def test_analyze_rejects_tampered_run(tmp_path):
    config = audit.load_config(CONFIG)
    config["output_root"] = "out"
    config_path = tmp_path / "config.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run = tmp_path / "out" / "runs" / "demo"
    run.mkdir(parents=True)
    (run / "COMPLETED").write_text(audit.MARKER + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        audit.analyze(tmp_path, config_path, "demo")


def test_synthetic_has_zero_steps_and_no_test(tmp_path):
    config = audit.load_config(CONFIG)
    config["output_root"] = "out"
    config_path = tmp_path / "config.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = audit.synthetic(tmp_path, config_path, "synthetic")
    payload = json.loads((Path(result["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
    assert payload["physical_optimizer_step_calls"] == 0
    assert payload["authoritative_optimizer_steps_committed"] == 0
    assert payload["test_accessed"] is False
