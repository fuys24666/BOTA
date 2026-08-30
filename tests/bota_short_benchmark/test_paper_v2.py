from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from src.bota_short_benchmark import paper_evaluation_v2 as evaluation
from src.bota_short_benchmark import paper_v2

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/bota_short_paper_v2.yaml"


def test_p5_is_frozen_and_complete_epoch_based():
    config = paper_v2.load_config(CONFIG)
    assert config["protocol"]["patience"] == 5
    assert config["protocol"]["max_epochs"] == 100
    assert config["protocol"]["batch_size"] == config["protocol"]["physical_microbatch"] * config["protocol"]["gradient_accumulation"] == 16
    assert config["protocol"]["validation_batch_size"] == 4
    state = (float("inf"), 0)
    for value in (1., .9, .91, .92, .93, .94):
        transition = paper_v2.early_stopping_transition(state[0], state[1], value)
        state = (transition["best"], transition["count"])
    assert not transition["stop"]
    transition = paper_v2.early_stopping_transition(state[0], state[1], .95)
    assert transition["stop"] and transition["count"] == 5


def test_improvement_resets_patience_and_equal_is_non_improving():
    first = paper_v2.early_stopping_transition(1., 4, .9)
    assert first["improved"] and first["count"] == 0 and not first["stop"]
    equal = paper_v2.early_stopping_transition(.9, 0, .9)
    assert not equal["improved"] and equal["count"] == 1
    with pytest.raises(ValueError): paper_v2.early_stopping_transition(1., 0, .9, patience=3)


def test_protocol_tampering_is_rejected(tmp_path):
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8")); value["protocol"]["patience"] = 3
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol changed"): paper_v2.load_config(path)


def test_pcgrad_nonconflict_is_mean_and_conflict_is_projected():
    merged, conflict = paper_v2._pcgrad([torch.tensor([1., 0.])], [torch.tensor([0., 1.])])
    assert not conflict and torch.equal(merged[0], torch.tensor([.5, .5]))
    left = torch.tensor([1., 0.]); right = torch.tensor([-1., 1.]); merged, conflict = paper_v2._pcgrad([left], [right])
    assert conflict and torch.isfinite(merged[0]).all()
    projected_left = left - torch.dot(left, right) / torch.dot(right, right) * right
    projected_right = right - torch.dot(left, right) / torch.dot(left, left) * left
    assert torch.allclose(merged[0], (projected_left + projected_right) / 2)


def test_short_neggrad_and_pcgrad_do_not_claim_ng2_pc1_identity():
    assert paper_v2.METHODS["NegGrad"] == "NegGrad-Mixed-Short-BOnly"
    assert paper_v2.METHODS["PCGrad"] == "PCGrad-Short-BOnly"
    doc = (ROOT / "docs/bota_short_paper_v2.md").read_text(encoding="utf-8")
    assert "not the 3,000-step full-model `ng2`" in doc
    assert "not the full-request `pc1`" in doc


def test_evaluation_freezes_two_reference_policy_and_all_models():
    assert evaluation.ORDER[:4] == ["Original-Short-200step", "Exact-Masked-Reference-200step", "FullControl-P5-Short", "Retain-Retrain-P5-Short"]
    assert len(evaluation.ORDER) == 10 and len(set(evaluation.ORDER)) == 10
    source = (ROOT / "src/bota_short_benchmark/paper_evaluation_v2.py").read_text(encoding="utf-8")
    assert '_residual(p, exact, original' in source
    assert '_residual(p, retain, control' in source


def test_residual_controls_are_not_interchanged():
    import numpy as np
    candidate = np.asarray([.2, .4]); reference = np.asarray([.3, .5]); control = np.asarray([.1, .1]); selected = np.asarray([0, 1])
    value = evaluation._residual(candidate, reference, control, selected)
    assert value["point"] == pytest.approx(.1 / .3)


def test_wrappers_are_development_only_and_forward_optional_names_safely():
    for path in (ROOT / "scripts/bota_if").glob("run_short_*_v2.ps1"):
        text = path.read_text(encoding="utf-8"); assert "FinalTest" not in text
    wrapper = (ROOT / "scripts/bota_if/run_short_paper_evaluation_v2.ps1").read_text(encoding="utf-8")
    assert "if($Pair[1])" in wrapper and "--exact-masked-run-name" in wrapper


def test_preflight_and_synthetic_paths_do_not_load_models():
    config = paper_v2.load_config(CONFIG)
    assert config["evaluation"]["final_test"] is False
    source = (ROOT / "src/bota_short_benchmark/paper_v2.py").read_text(encoding="utf-8")
    assert '"real_model_loaded": False' in source
    assert '"test_accessed": False' in source


def test_cuda_memory_accounting_uses_runtime_compatible_device_index():
    source = (ROOT / "src/bota_short_benchmark/paper_v2.py").read_text(encoding="utf-8")
    assert "torch.cuda.set_device(0)" in source
    assert "torch.cuda.reset_peak_memory_stats()" in source
    assert "torch.cuda.max_memory_reserved()" in source
    assert "reset_peak_memory_stats(device)" not in source


def test_weighted_microbatch_gradient_matches_logical_batch_mean():
    model = torch.nn.Linear(3, 1, bias=False); x = torch.arange(18, dtype=torch.float32).reshape(6, 3); y = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    full = torch.nn.functional.mse_loss(model(x), y); full.backward(); expected = model.weight.grad.detach().clone(); model.zero_grad(set_to_none=True)
    for start in range(0, 6, 2):
        loss = torch.nn.functional.mse_loss(model(x[start:start+2]), y[start:start+2]); (loss * (2/6)).backward()
    assert torch.allclose(model.weight.grad, expected, atol=1e-6, rtol=1e-6)
