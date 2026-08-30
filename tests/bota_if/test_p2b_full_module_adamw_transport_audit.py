from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
import torch
import yaml

from src.bota_if import p2b_full_module_adamw_transport_audit as audit

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/bota_if/p2b_full_module_adamw_transport_v1.yaml"


@pytest.fixture
def config(): return audit.load_frozen_config(CONFIG)


def test_predecessor_and_full_coordinate_are_frozen(config):
    predecessor = audit.validate_predecessor(ROOT, config)
    assert predecessor["classification"] == "user_specific_module_sparsity_insufficient"
    assert config["coordinate"]["transport_dimension"] == 884736
    assert config["coordinate"]["shared_low_rank_coordinate_used"] is False
    assert config["coordinate"]["module_truncation_used"] is False


@pytest.mark.parametrize("section,key,value", [
    ("coordinate", "transport_dimension", 576),
    ("coordinate", "module_truncation_used", True),
    ("schedule", "steps", 49),
    ("optimizer", "learning_rate", .002),
    ("transport", "state_dtype", "float64"),
    ("quantization", "fp16_relative_l2_maximum", .01),
    ("gates", "cosine_minimum", .7),
    ("runtime", "physical_optimizer_step_limit", 201),
    ("scientific_scope", "zero_authoritative_update", False),
])
def test_config_tamper_rejected(tmp_path, config, section, key, value):
    changed = copy.deepcopy(config); changed[section][key] = value; path = tmp_path / "config.yaml"; path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError): audit.load_frozen_config(path)


def test_full_fisher_product_matches_explicit_matrix():
    coefficients = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    vectors = torch.tensor([[1., 0., 2.], [0., 1., 3.]])
    explicit = coefficients.T @ coefficients / coefficients.shape[0]
    assert torch.allclose(audit.full_fisher_product(coefficients, vectors), vectors @ explicit)


def test_full_state_shape_preserves_users_and_parameter_shape():
    parameters = [torch.zeros(2, 3), torch.zeros(4)]
    states = audit.new_full_states(parameters, 3)
    assert set(states) == set(audit.VARIANTS)
    assert states["T2_AdamW_full_state"][0]["theta"].shape == (3, 2, 3)
    assert states["T1_AdamW_frozen_v"][1]["v"].shape == (3, 4)


def _tangent_inputs(dtype=torch.float32):
    zero = torch.zeros(4, dtype=dtype)
    return dict(theta=zero, gradient=zero, m=zero, v=zero, dtheta=zero, dm=zero, dv=zero, dgradient=torch.tensor([1., -2., 3., -4.], dtype=dtype), step=1, lr=.001, beta1=.9, beta2=.999, eps=1e-8, weight_decay=.01)


def test_frozen_v_zero_moment_branch_is_finite_and_skips_correction():
    args = _tangent_inputs(); theta, dm, dv = audit.stable_adamw_tangent_step(**args, full_v=False)
    expected_dm = .1 * args["dgradient"]
    expected_theta = -.001 * expected_dm / .1 / 1e-8
    assert torch.isfinite(theta).all() and torch.allclose(theta, expected_theta)
    assert torch.allclose(dm, expected_dm) and torch.count_nonzero(dv) == 0


def test_full_v_zero_root_zero_numerator_has_exact_zero_correction():
    args = _tangent_inputs(); full = audit.stable_adamw_tangent_step(**args, full_v=True); frozen = audit.stable_adamw_tangent_step(**args, full_v=False)
    assert all(torch.equal(left, right) for left, right in zip(full, frozen))


def test_true_zero_root_nonzero_numerator_is_rejected(monkeypatch):
    # This state cannot arise from a consistent first-order Adam trajectory, but
    # the explicit rejection protects against corrupted or mismatched state.
    args = _tangent_inputs(); args["m"] = torch.ones(4); args["dv"] = torch.ones(4)
    # beta2*dv makes v-tangent nonzero while the base second moment remains zero.
    with pytest.raises(RuntimeError, match="singular_adamw_tangent"):
        audit.stable_adamw_tangent_step(**args, full_v=True)


def test_stable_tangent_matches_float64_reference_away_from_zero():
    generator = torch.Generator().manual_seed(42); gradient = torch.randn(8, generator=generator).abs() + .2; m = torch.randn(8, generator=generator); v = torch.rand(8, generator=generator) + .1; perturbations = [torch.randn(8, generator=generator) * .01 for _ in range(4)]
    args = dict(theta=torch.randn(8, generator=generator), gradient=gradient, m=m, v=v, dtheta=perturbations[0], dm=perturbations[1], dv=perturbations[2], dgradient=perturbations[3], step=7, lr=.001, beta1=.9, beta2=.999, eps=1e-8, weight_decay=.01, full_v=True)
    actual = audit.stable_adamw_tangent_step(**args); reference = audit.p1.adamw_tangent_step(**{key: value.double() if torch.is_tensor(value) else value for key, value in args.items()})
    assert all(torch.allclose(left.double(), right, atol=2e-7, rtol=2e-5) for left, right in zip(actual, reference))


def test_batched_user_tangents_broadcast_base_adamw_state_exactly():
    generator = torch.Generator().manual_seed(7); shape = (2, 3); users = 3
    gradient = torch.rand(shape, generator=generator) + .2; m = torch.randn(shape, generator=generator); v = torch.rand(shape, generator=generator) + .1
    args = dict(theta=torch.randn(shape, generator=generator), gradient=gradient, m=m, v=v, dtheta=torch.randn((users, *shape), generator=generator) * .01, dm=torch.randn((users, *shape), generator=generator) * .01, dv=torch.randn((users, *shape), generator=generator) * .01, dgradient=torch.randn((users, *shape), generator=generator) * .01, step=5, lr=.001, beta1=.9, beta2=.999, eps=1e-8, weight_decay=.01, full_v=True)
    batched = audit.stable_adamw_tangent_step(**args)
    individual = []
    for user in range(users):
        row = {key: (value[user] if key in {"dtheta", "dm", "dv", "dgradient"} else value) for key, value in args.items()}
        individual.append(audit.stable_adamw_tangent_step(**row))
    for component in range(3):
        assert torch.allclose(batched[component], torch.stack([row[component] for row in individual]), atol=1e-7, rtol=1e-6)


def test_quantization_reports_fp16_and_bf16_without_persisting():
    config = {"quantization": {"fp16_relative_l2_maximum": .001, "bf16_relative_l2_maximum": .01, "cosine_minimum": .9999}}
    row = audit.quantization_report(torch.tensor([.1, -.2, .3]), config)
    assert row["float16"]["passed"] and row["bfloat16"]["passed"]
    assert row["float16"]["relative_l2_error"] < row["bfloat16"]["relative_l2_error"]


def test_fp16_overflow_is_structured_failure_not_nonfinite_json():
    config = {"quantization": {"fp16_relative_l2_maximum": .001, "bf16_relative_l2_maximum": .01, "cosine_minimum": .9999}}
    row = audit.quantization_report(torch.tensor([70000., 1.]), config)
    assert row["float16"]["status"] == "quantization_overflow"
    assert row["float16"]["quantized_nonfinite_count"] == 1
    assert row["float16"]["relative_l2_error"] is None and row["float16"]["passed"] is False
    assert row["bfloat16"]["status"] == "finite"


def test_nonfinite_transport_metrics_are_nullable_and_rejected():
    row = audit.vector_metrics(torch.tensor([1., 2.]), torch.tensor([float("inf"), 2.]))
    assert row["finite"] is False and row["candidate_nonfinite_count"] == 1
    assert row["cosine"] is row["relative_l2_error"] is None
    assert audit._finite(row)


def _user(t2=True, fp16=True, bf16=True):
    variants = {name: {"finite": t2, "base_gates_passed": name == "T2_AdamW_full_state" and t2, "prediction_quantization": {"float16": {"passed": fp16}, "bfloat16": {"passed": bf16}}} for name in audit.VARIANTS}
    return {"optimizer_aware_gate_passed": t2, "transport": {"variants": variants, "authority_quantization": {"float16": {"passed": fp16}, "bfloat16": {"passed": bf16, "reference_finite": True}}}}


def test_classification_separates_transport_and_quantization():
    classification, gates = audit.classify([_user()] * 3)
    assert classification == "full_module_transport_supported" and gates["all"]
    classification, gates = audit.classify([_user(), _user(t2=False), _user()])
    assert classification == "full_module_transport_numerically_unstable" and not gates["full_module_t2_all_users"]
    classification, gates = audit.classify([_user(), _user(fp16=False), _user()])
    assert classification == "full_module_transport_supported" and not gates["fp16_bank_all_users"] and not gates["all"]


def test_budget_is_200_and_transport_is_zero_extra_steps(config):
    row = audit.budget(config)
    assert row["physical_optimizer_step_calls"] == 200
    assert row["transport_extra_optimizer_steps"] == row["quantization_extra_optimizer_steps"] == 0
    assert row["authoritative_optimizer_steps_committed"] == 0


def test_synthetic_is_exact_and_zero_step():
    row = audit.synthetic()
    assert row["full_fisher_exact"] and row["fp16_passed"] and row["bf16_passed"]
    assert row["optimizer_steps"] == 0 and row["test_accessed"] is False


def test_execute_has_no_basis_builder_or_module_truncation_and_exact_step_budget():
    source = inspect.getsource(audit.execute)
    assert "build_transport_bases" not in source and "sparse_module_oracle" not in source
    assert source.count("p1.run_masked(") == 1
    assert "step_budget.calls != 200" in source
    assert '"authoritative_optimizer_steps_committed": 0' in source


def test_no_development_retrain_test_or_artifact_publication():
    source = inspect.getsource(audit.execute)
    assert "development.json" not in source and "validation_partition" not in source
    assert '"development_loaded": False' in source and '"retrain_loaded": False' in source
    assert '"model_artifact_published": False' in source and '"bank_artifact_published": False' in source
    assert "torch.save" not in source


def test_failure_publication_preserves_actual_step_count():
    source = inspect.getsource(audit.execute)
    assert '"status": "INTERRUPTED"' in source
    assert '"physical_optimizer_step_calls": step_budget.calls' in source
    assert "os.replace(stage, destination)" in source
