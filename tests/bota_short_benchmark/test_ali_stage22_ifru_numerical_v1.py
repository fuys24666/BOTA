from pathlib import Path
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from src.bota_short_benchmark import ali_stage22_numerics as n
from src.bota_short_benchmark import ali_stage22_ifru_numerical_v1 as s
from src.diagnostics.t5_lora_influence_feasibility_audit import conjugate_gradient

CG = dict(relative_tolerance=.0001, absolute_tolerance=1e-10, max_iterations=40,
          residual_explosion_factor=1000., pap_tolerance=1e-14)


@pytest.mark.parametrize("variant", ["legacy", "monitor", "restart5"])
def test_spd_against_direct_solve(variant):
    a = torch.tensor([[2., .2], [.2, 4.]], dtype=torch.float64)
    b = torch.tensor([1., 3.], dtype=torch.float64)
    out = n.diagnostic_cg(lambda x: a@x, b, .1, CG, variant)
    assert out["true_tolerance_satisfied"] and out["status"] == "converged"
    assert torch.allclose(out["solution"], torch.linalg.solve(a+.1*torch.eye(2), b), atol=1e-7)
    assert not out["solution_applied"] and not out["tolerance_relaxed"]


def test_legacy_exact_recurrence_matches_original():
    rng = torch.Generator().manual_seed(9)
    m = torch.randn(30, 30, generator=rng, dtype=torch.float64)
    a = m.T@m; b = torch.randn(30, generator=rng, dtype=torch.float64)
    expected = conjugate_gradient(lambda x: a@x, b, damping=.1, **CG, allow_truncated_solution=True)
    got = n.diagnostic_cg(lambda x: a@x, b, .1, CG)
    assert torch.equal(got["solution"], expected["solution"])
    assert got["operator_calls"] == expected["hvp_calls"]
    assert got["final_relative_residual"] == expected["final_relative_residual"]
    assert [r["recursive_residual_norm"] for r in got["history"]] == [r["residual_norm"] for r in expected["history"]]


def test_false_convergence_retains_real_residual_and_failure():
    # A deliberately nonlinear operator gives exact recursive convergence in
    # one step but a different A(x) at the smaller final solution magnitude.
    def nonlinear(v):
        return 2*v if n.norm(v) > .75 else 3*v
    b = torch.tensor([1.], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="false_residual_convergence"):
        conjugate_gradient(nonlinear, b, damping=0., **CG, allow_truncated_solution=True)
    out = n.diagnostic_cg(nonlinear, b, 0., CG)
    assert out["status"] == "false_residual_convergence"
    assert out["true_over_threshold"] > 1
    assert not out["historical_acceptance"] and not out["true_tolerance_satisfied"]
    assert len(out["history"]) == 1


def test_restart_is_bounded_and_does_not_relax_threshold():
    def nonlinear(v): return 2*v if n.norm(v) > .75 else 3*v
    b = torch.tensor([1.], dtype=torch.float64)
    out = n.diagnostic_cg(nonlinear, b, 0., CG, "restart5")
    assert out["restarts"] == [1]
    assert out["true_tolerance_satisfied"] and out["iterations"] <= 40
    assert out["target"] == .0001 and not out["solution_applied"]


def test_monitor_does_not_change_deterministic_recurrence():
    a = torch.diag(torch.arange(1., 12., dtype=torch.float64)); b = torch.ones(11, dtype=torch.float64)
    plain = n.diagnostic_cg(lambda v: a@v, b, .1, CG)
    mon = n.diagnostic_cg(lambda v: a@v, b, .1, CG, "monitor")
    assert torch.equal(plain["solution"], mon["solution"])
    assert all("true_relative_residual" in row for row in mon["history"])
    assert mon["operator_calls"] == plain["operator_calls"]+mon["iterations"]


def test_negative_curvature_recorded_without_success():
    out = n.diagnostic_cg(lambda v: -v, torch.ones(3), 0., CG)
    assert out["status"] == "significantly_negative_pAp"
    assert not out["historical_acceptance"] and not out["solution_applied"]


def test_truncation_not_misreported_as_true_convergence():
    a = torch.diag(torch.tensor([1., 10., 100.], dtype=torch.float64))
    out = n.diagnostic_cg(lambda v: a@v, torch.ones(3), .1, {**CG, "max_iterations": 1})
    assert out["status"] == "max_iterations_reached"
    assert out["historical_truncated_acceptance"] and not out["true_tolerance_satisfied"]


def test_zero_rhs():
    out = n.diagnostic_cg(lambda v: v, torch.zeros(2), .1, CG)
    assert out["iterations"] == 0 and out["true_tolerance_satisfied"]


def test_unexpected_oom_not_relabelled_as_solver_failure():
    def operator(v): raise RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        n.diagnostic_cg(operator, torch.ones(2), .1, CG)


def test_probes_detect_asymmetry_and_are_reproducible():
    a = torch.tensor([[1., 2.], [0., 3.]], dtype=torch.float64)
    g = torch.tensor([1., .4], dtype=torch.float64)
    out = n.operator_probes(lambda v: a@v, g, 42, [.0001, .01, 100.])
    assert out == n.operator_probes(lambda v: a@v, g, 42, [.0001, .01, 100.])
    assert out["symmetry"]["normalized_gap"] > 0
    assert out["repeatability"]["exact_equal"] and out["additivity"]["relative_l2"] < 1e-14


def test_bundle_resume_tamper_and_contract_checks(tmp_path):
    stage = s.stage_dir(tmp_path); dest = tmp_path/"unit"
    s.publish(stage, dest, {"contract": "abc", "status": "failure"})
    assert s.verify(dest, "abc")["status"] == "failure"
    with pytest.raises(ValueError, match="contract"): s.verify(dest, "def")
    with pytest.raises(FileExistsError): s.publish(s.stage_dir(tmp_path), dest, {})
    (dest/"result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed artifact"): s.verify(dest)


def test_summary_retains_all_failed_cells(tmp_path):
    c = {"requests": ["A1", "A2", "A3"], "variants": ["legacy", "monitor", "restart5"]}
    p = {"config": c}; contract = s.canonical_hash(p)
    for rid in c["requests"]:
        for unit in ["setup", "probes", *c["variants"]]:
            payload = {"contract": contract}
            if unit in c["variants"]:
                payload.update(request=rid, variant=unit, status="false_residual_convergence",
                               iterations=3, true_over_threshold=1.01)
            s.publish(s.stage_dir(tmp_path), tmp_path/"requests"/rid/unit, payload)
    out = s.analyze(tmp_path, p)
    assert len(out["rows"]) == 9 and out["status"] == "COMPLETED"
    assert all(row["status"] == "false_residual_convergence" for row in out["rows"])
    assert s.verify(tmp_path/"summary", contract)["selection_performed"] is False


def test_config_blocks_tolerance_changes(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    # Test file lives tests/bota_short_benchmark; repository is parents[2].
    c = s.load_config(repo)
    (tmp_path/"configs").mkdir()
    c["cg"]["relative_tolerance"] = .1
    (tmp_path/s.CONFIG).write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(ValueError, match="tolerances"): s.load_config(tmp_path)


def test_cpu_smoke_has_no_model_or_training():
    assert s.smoke() == {"status": "SMOKE_PASS", "model_loaded": False, "test_accessed": False, "optimizer_steps": 0}


def test_request_pipeline_resumes_verified_units_without_endpoint_updates(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    config = s.load_config(repo)
    request = {"id": "A1", "forget_train_indices": [1, 2], "users": [10, 20]}
    p = {"comparison": {"requests": [request]}, "registry": {"order": list(range(3200))}}
    protocol = {"config": config}
    historical = {"A1": {"status": "NUMERICAL_SOLVER_FAILURE"}}
    parameter = torch.nn.Parameter(torch.tensor([.2, .3], dtype=torch.float64))
    initial = parameter.detach().clone()
    loaded = []
    def fresh(*args):
        loaded.append(True)
        return SimpleNamespace(eval=lambda: None), ["B"], [parameter], {}, SimpleNamespace(pad_token_id=0), None, None, "cpu"
    monkeypatch.setattr(s.k, "fresh_endpoint", fresh)
    monkeypatch.setattr(s.k.runner, "materialize_resident_panel", lambda *a: ([{}], {"samples": 1024}))
    monkeypatch.setattr(s.k.runner, "make_sample_mean_ggn_operator", lambda *a: lambda v: 2*v)
    monkeypatch.setattr(s.k.runner, "estimate_lambda_max", lambda *a, **kw: {"lambda_max_hat": 2.})
    monkeypatch.setattr(s.k.runner, "sample_mean_gradients", lambda *a: ([torch.tensor([1., 2.])], {"loss": .3}))
    monkeypatch.setattr(s.k.s18, "release", lambda: None)
    s.run_request(tmp_path, tmp_path, protocol, p, historical, "A1")
    assert len(loaded) == 1 and torch.equal(parameter, initial)
    for variant in config["variants"]:
        unit = s.verify(tmp_path/"requests/A1"/variant, s.canonical_hash(protocol))
        assert unit["true_tolerance_satisfied"] and not unit["solution_applied"]
    assert s.verify(tmp_path/"requests/A1/legacy")["historical_status_reproduced"] is False
    # No model load or new writes when all units are complete.
    snapshots = {str(f): s.sha256_file(f) for f in tmp_path.rglob("*") if f.is_file()}
    s.run_request(tmp_path, tmp_path, protocol, p, historical, "A1")
    assert len(loaded) == 1
    assert snapshots == {str(f): s.sha256_file(f) for f in tmp_path.rglob("*") if f.is_file()}
