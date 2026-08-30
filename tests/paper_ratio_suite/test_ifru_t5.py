from pathlib import Path

import pytest
import torch

from src.paper_ratio_suite.ifru_t5 import METHOD_ID, deterministic_panel, ifru_scale, load_frozen, make_sample_mean_ggn_operator, synthetic


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/paper_ifru_t5_2pct_v1.yaml"


def test_frozen_classic_ifru_formula_and_scale():
    config = load_frozen(CONFIG)
    assert config["method"]["update_scale"] == "k_over_n"
    assert ifru_scale(1258, 60000) == pytest.approx(0.020966666666666668)
    assert config["method"]["spillover_term"] == "zero"
    assert "retain_projection" in config["forbidden"]
    assert "trust_scale" in config["forbidden"]


def test_panel_is_deterministic_without_replacement():
    left = deterministic_panel(60000, 4096, 42); right = deterministic_panel(60000, 4096, 42)
    assert left == right and len(set(left)) == 4096
    assert min(left) >= 0 and max(left) < 60000


def test_synthetic_has_no_training_or_test_access():
    result = synthetic()
    assert result["method_id"] == METHOD_ID
    assert result["optimizer_constructed"] is False
    assert result["test_accessed"] is False
    assert result["retain_projection_used"] is False


def test_sample_mean_ggn_is_psd_on_tiny_model(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.1], dtype=torch.float64))
    class Model(torch.nn.Module):
        def forward(self, **batch):
            x = batch["input_ids"].to(torch.float64); logits = torch.stack((x @ parameter, -(x @ parameter)), dim=-1).unsqueeze(1)
            return type("Output", (), {"logits": logits})
    monkeypatch.setattr("src.paper_ratio_suite.ifru_t5.masked_forward", lambda model, batch: model(**batch))
    batch = {"input_ids": torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64), "target_ids": torch.tensor([[1], [0]])}
    counter = {}; operator = make_sample_mean_ggn_operator(Model(), [batch], [parameter], counter); vector = torch.tensor([0.4, -0.7], dtype=torch.float64); product = operator(vector)
    assert torch.isfinite(product).all()
    assert torch.dot(vector, product) >= -1e-12
    assert counter["operator_calls"] == 1

