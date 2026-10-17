import copy
from pathlib import Path

import pytest
import torch

from src.bota_short_benchmark import ali_stage12_lambda_extension_v1 as s

ROOT = Path(__file__).resolve().parents[1]


def rows(scales=s.GRID, seed=42, optimum=1.5):
    return [{"candidate_id": s.cid(scale), "scenario": sid, "response_scale": scale,
             "E_U": abs(scale - optimum) * .1 + .1, "D_U": 1., "local_residual": .2,
             "utility_eligible": True, "prediction_collapse": False, "seed": seed,
             "status": "OK", "origin": "synthetic", "test_accessed": False}
            for scale in scales for sid in s.SCENARIOS]


def test_config_and_old_grid_unchanged():
    config = s.load_config(ROOT, ROOT / "configs/ali_stage12_lambda_extension_v1.json")
    assert config["protocol"]["final_test"] is False
    assert s.s8.SCALES == (0., .25, .5, .75, 1.)
    assert s.NEW_GRID == (1., 1.5, 2.)


def test_joint_selection_tie_and_boundary():
    assert s.select_rows(rows())["selected"]["response_scale"] == 1.5
    tied = rows()
    for row in tied:
        row["E_U"] = .1
    assert s.select_rows(tied)["selected"]["response_scale"] == 1
    assert s.select_rows(rows(optimum=2))["boundary_winner"] is True


@pytest.mark.parametrize("change", ("missing", "duplicate", "unknown"))
def test_bad_inventory_rejected(change):
    cells = rows()
    if change == "missing": cells.pop()
    elif change == "duplicate": cells.append(cells[0])
    else: cells[0]["scenario"] = "other"
    with pytest.raises(ValueError): s.select_rows(cells)


def test_failed_candidate_and_no_eligible():
    cells = rows()
    for row in cells:
        if row["response_scale"] == 1.5:
            row.update(status="NONFINITE_TRANSPORT", E_U=None)
    result = s.select_rows(cells)
    assert len(result["candidates"]) == 7
    assert result["selected"]["response_scale"] == 1
    assert result["candidates"][5]["equal_scenario_mean_EU"] is None
    for row in cells: row["utility_eligible"] = False
    assert s.select_rows(cells)["status"] == "NO_ELIGIBLE_CALIBRATION"


def test_one_bad_request_disqualifies_candidate():
    cells = rows()
    next(r for r in cells if r["response_scale"] == 1.5)["utility_eligible"] = False
    assert s.select_rows(cells)["selected"]["response_scale"] == 1


def test_reproduction_gate():
    old = {"E_U": .0001, "D_U": .0002, "overall_auc": .8, "overall_log_loss": .5}
    protocol = {"reproduction_atol": 1e-8, "reproduction_rtol": .001}
    assert s.check_reproduction(dict(old), old, protocol)
    new = dict(old, E_U=.00011)
    with pytest.raises(RuntimeError): s.check_reproduction(new, old, protocol)


def test_confirmation_is_per_seed_and_preserves_cell_regressions():
    cells = rows((1., 1.5), 41) + rows((1., 1.5), 43)
    assert s.confirmation_decision(cells, 1.5)["status"] == "CROSS_SEED_DEVELOPMENT_CONFIRMED"
    next(r for r in cells if r["seed"] == 41 and r["response_scale"] == 1.5)["E_U"] = .16
    result = s.confirmation_decision(cells, 1.5)
    assert result["status"] == "CROSS_SEED_DEVELOPMENT_CONFIRMED"
    assert len(result["cell_regressions"]) == 1
    for r in cells:
        if r["seed"] == 43 and r["response_scale"] == 1.5: r["E_U"] = .16
    assert s.confirmation_decision(cells, 1.5)["status"] == "NOT_STABLE"


def test_confirmation_failed_numerical_cell():
    cells = rows((1., 1.5), 41) + rows((1., 1.5), 43)
    next(r for r in cells if r["response_scale"] == 1.5).update(status="NONFINITE_PREDICTIONS", E_U=None)
    assert s.confirmation_decision(cells, 1.5)["status"] == "NOT_STABLE"


def test_atomic_bundle_detects_tampering(tmp_path):
    stage = s.staging(tmp_path)
    s.atomic_json(stage / "value.json", {"test_accessed": False})
    destination = tmp_path / "bundle"
    s.publish(stage, destination)
    s.verify_bundle(destination)
    s.atomic_json(destination / "value.json", {"test_accessed": True})
    with pytest.raises(ValueError): s.verify_bundle(destination)


def test_selection_freeze_refuses_replacement(tmp_path):
    result, _ = s.freeze_selection(tmp_path, rows(), [], "hash")
    assert result["selected"]["response_scale"] == 1.5
    s.freeze_selection(tmp_path, rows(), [], "hash")
    with pytest.raises(RuntimeError): s.freeze_selection(tmp_path, rows(optimum=2), [], "hash")


def test_sample_order_and_nonfinite_predictions():
    indices = [1, 2]
    pred = {"sample_order_sha256": s.canonical_hash(indices), "probability": [.2, .4], "answer_loss": [1., 2.], "gold_label": [0, 1]}
    assert s.validate_prediction(pred, indices)
    with pytest.raises(RuntimeError): s.validate_prediction(pred, [2, 1])
    pred["probability"][0] = float("nan")
    assert not s.validate_prediction(pred, indices)


def test_lambda_scales_feedback_not_source(monkeypatch):
    parameter = torch.tensor([1.], requires_grad=True)
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW([parameter])
    seen = []
    def tangent(*args, **kwargs):
        seen.append(args[7].clone())
        return args[4], args[5], args[6]
    monkeypatch.setattr(s.s8.p2b, "stable_adamw_tangent_step", tangent)
    monkeypatch.setattr(s.s8.p2b, "full_fisher_product", lambda rows, theta: torch.full_like(theta, 3.))
    engine = {"optimizer": {"betas": [.9, .99], "learning_rate": .01, "eps": 1e-8, "weight_decay": 0}}
    state = [{k: torch.zeros(1, 1) for k in ("theta", "m", "v")}]
    for scale in (0., 1., 2.):
        s.s8._advance_scaled(copy.deepcopy(state), [torch.ones(1, 1)], [torch.ones(1, 1) * 2], [parameter], optimizer, 1, engine, scale)
    assert [x.item() for x in seen] == [2., 5., 8.]


def toy_transport(monkeypatch):
    engine = {"schedule": {"batch_size": 2}, "optimizer": {
        "betas": [.9, .99], "learning_rate": .001, "eps": 1e-6, "weight_decay": .01}}
    data = torch.tensor([[.4, 1.], [1.3, .2]] * 200)
    monkeypatch.setattr(s.s8, "masked_batch", lambda dataset, chosen, pad: dataset[chosen])
    monkeypatch.setattr(s.s8, "move_batch", lambda batch, device: batch)
    monkeypatch.setattr(s.s8.p1, "_sample_losses", lambda model, batch: ((batch @ model).squeeze(-1) - .3).square())
    def arguments():
        parameter = torch.tensor([[.2], [.5]], requires_grad=True)
        return (parameter, data, list(range(400)), [1, 2] * 200, [1, 2], [parameter], ["p"],
                torch.device("cpu"), 0, engine, s.s8.StepBudget(200))
    return arguments


def test_200_step_cpu_control_matches_stage8_exactly(monkeypatch, tmp_path):
    arguments = toy_transport(monkeypatch)
    torch.manual_seed(7)
    original = s.s8._run_scaled(*arguments())
    torch.manual_seed(7)
    args = arguments()
    extended = s.run_transport(*args, s.NEW_GRID, tmp_path / "progress.json")
    assert args[-1].calls == 200
    assert extended[2] == {}
    assert torch.equal(original[0]["p"], extended[0]["p"])
    for name in ("theta", "m", "v"):
        assert torch.equal(original[1]["bota_r1"][0][name], extended[1]["bota_r1"][0][name])
    assert original[2:] == extended[3:]


@pytest.mark.parametrize("message,numerical", [
    ("nonfinite_stable_adamw_tangent", True), ("CUDA out of memory", False)])
def test_numerical_failure_is_reported_but_oom_aborts(monkeypatch, tmp_path, message, numerical):
    arguments = toy_transport(monkeypatch)
    real = s.s8._advance_scaled
    def advance(*args):
        if args[-1] == 2:
            raise RuntimeError(message)
        return real(*args)
    monkeypatch.setattr(s.s8, "_advance_scaled", advance)
    if numerical:
        result = s.run_transport(*arguments(), s.NEW_GRID, tmp_path / "progress.json")
        assert result[2]["bota_r2"]["first_failed_step"] == 1
        assert "bota_r2" not in result[1]
        assert "bota_r1.5" in result[1]
    else:
        with pytest.raises(RuntimeError, match="out of memory"):
            s.run_transport(*arguments(), s.NEW_GRID, tmp_path / "progress.json")


def test_pipeline_freezes_before_confirmation_and_resumes(monkeypatch, tmp_path):
    config_path = ROOT / "configs/ali_stage12_lambda_extension_v1.json"
    config = s.load_config(ROOT, config_path)
    root = tmp_path / config["output_root"]
    monkeypatch.setattr(s, "preflight", lambda *args: {})
    monkeypatch.setattr(s, "freeze_protocol", lambda *args: "contract")
    monkeypatch.setattr(s, "check_frozen", lambda *args: "contract")
    monkeypatch.setattr(s, "historical_rows", lambda *args: rows(s.s8.SCALES, optimum=1))
    calls = []
    def cell(repo, workspace, config, root, contract, seed, sid, scales, selection_hash=None):
        if seed != 42:
            assert selection_hash == s.sha256_file(root / "selection/selection.json")
        calls.append((seed, sid, scales))
        values = rows(scales, seed)
        for r in values:
            if r["response_scale"] > 1: r["E_U"] -= .06
        return {"rows": [r for r in values if r["scenario"] == sid]}
    monkeypatch.setattr(s, "execute_cell", cell)
    result = s.execute(ROOT, tmp_path, config_path, "All")
    assert result["status"] == "COMPLETED"
    assert result["selected"]["response_scale"] == 1.5
    assert len(calls) == 6
    assert result["confirmation"]["finaltest_authorized"] is False
    assert s.execute(ROOT, tmp_path, config_path, "All") == result
    s.verify_bundle(root / "summary")


def test_execution_lock_releases_after_exception(tmp_path):
    with pytest.raises(ValueError):
        with s.execution_lock(tmp_path):
            raise ValueError("interrupted")
    with s.execution_lock(tmp_path):
        assert (tmp_path / "execution.lock").is_file()


def test_no_improved_scale_skips_confirmation(monkeypatch, tmp_path):
    config_path = ROOT / "configs/ali_stage12_lambda_extension_v1.json"
    monkeypatch.setattr(s, "preflight", lambda *args: {})
    monkeypatch.setattr(s, "freeze_protocol", lambda *args: "contract")
    monkeypatch.setattr(s, "check_frozen", lambda *args: "contract")
    monkeypatch.setattr(s, "historical_rows", lambda *args: rows(s.s8.SCALES, optimum=1))
    def cell(repo, workspace, config, root, contract, seed, sid, scales, selection_hash=None):
        assert seed == 42
        return {"rows": [r for r in rows(scales, optimum=1) if r["scenario"] == sid]}
    monkeypatch.setattr(s, "execute_cell", cell)
    result = s.execute(ROOT, tmp_path, config_path, "All")
    assert result["selected"]["response_scale"] == 1
    assert result["confirmation"]["status"] == "NOT_TRIGGERED_NO_EXTENDED_WINNER"
