import copy
from pathlib import Path

import pytest
import torch

from src.bota_short_benchmark import ali_stage13_lambda_search_v1 as s

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ali_stage13_lambda_search_v1.json"


def rows(scales=s.COARSE_GRID, seed=42, scores=None):
    scores = scores or {2: .0001, 4: .00008, 8: .00009, 16: .00011}
    return [{"candidate_id": s.cid(scale), "response_scale": float(scale), "seed": seed,
             "scenario": sid, "E_U": scores.get(scale, .00012), "D_U": .001,
             "local_residual": scores.get(scale, .00012)/.001, "status": "OK",
             "utility_eligible": True, "prediction_collapse": False,
             "origin": "synthetic", "test_accessed": False}
            for scale in scales for sid in s.SCENARIOS]


def test_frozen_config_parent_and_old_grids_unchanged():
    config = s.load_config(ROOT, CONFIG)
    assert config["protocol"]["maximum_lambda"] == 16
    assert config["protocol"]["final_test"] is False
    assert s.s12.NEW_GRID == (1., 1.5, 2.)
    assert s.s8.SCALES == (0., .25, .5, .75, 1.)
    assert len(config["parent_manifests"]) == 9


@pytest.mark.parametrize("winner,grid", [(4, [3, 6]), (8, [6, 12])])
def test_exact_midpoint_rule(winner, grid):
    scores = {2: .0001, 4: .00009, 8: .00009, 16: .00011, winner: .00008}
    result = s.coarse_decision(rows(scores=scores), .0001)
    assert result["fine_grid"] == grid
    assert result["next_action"] == "ONE_PREDEFINED_FINE_ROUND"


def test_boundary_16_never_expands_or_fine_scans():
    result = s.coarse_decision(rows(scores={2: .0001, 4: .00009, 8: .00008, 16: .00007}), .0001)
    assert result["boundary_winner"] is True
    assert result["fine_grid"] == []
    assert result["next_action"] == "UPPER_BOUNDARY_STOP_NO_EXPANSION"


def test_lower_boundary_retains_reference():
    result = s.coarse_decision(rows(scores={2: .00008, 4: .00009, 8: .0001, 16: .00011}), .00008)
    assert result["fine_grid"] == []
    assert result["next_action"] == "LOWER_BOUNDARY_RETAIN_STAGE12"


def test_small_gain_stops_even_with_interior_best():
    result = s.coarse_decision(rows(scores={2: .0001, 4: .0000995, 8: .00012, 16: .00013}), .0001)
    assert result["selected"]["response_scale"] == 4
    assert result["fine_grid"] == []
    assert result["next_action"] == "GAIN_BELOW_BUDGET_GATE"


def test_budget_gate_requires_absolute_and_relative():
    assert not s.gain_over_two({"equal_scenario_mean_EU": .999998}, 1)["passes_budget_gate"]
    assert not s.gain_over_two({"equal_scenario_mean_EU": .0000095}, .00001)["passes_budget_gate"]
    assert s.gain_over_two({"equal_scenario_mean_EU": .00009}, .0001)["passes_budget_gate"]


def test_ineligible_numeric_and_utility_rows_are_not_dropped():
    values = rows()
    next(r for r in values if r["response_scale"] == 4).update(status="NONFINITE_TRANSPORT", E_U=None)
    next(r for r in values if r["response_scale"] == 8)["utility_eligible"] = False
    result = s.coarse_decision(values, .0001)
    assert len(result["candidates"]) == 4
    assert result["selected"]["response_scale"] == 2
    for row in values: row["utility_eligible"] = False
    assert s.coarse_decision(values, .0001)["selected"] is None


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unexpected"])
def test_grid_inventory(kind):
    values = rows()
    if kind == "missing": values.pop()
    elif kind == "duplicate": values.append(values[0])
    else: values[0]["candidate_id"] = "bota_r32"
    with pytest.raises(ValueError): s.coarse_decision(values, .0001)


def test_final_selection_merges_history_without_duplicate_control():
    old = rows(s.HISTORICAL_GRID, scores={2: .0001})
    new = rows((4, 8, 16, 3, 6), scores={3: .00006, 4: .00008, 6: .000075, 8: .00009, 16: .00011})
    result = s.final_decision(old+new, .0001)
    assert result["selected"]["response_scale"] == 3
    assert len(result["candidates"]) == 12
    assert result["confirmation_triggered"] is True
    with pytest.raises(ValueError): s.final_decision(old+new+rows((2,)), .0001)


def test_confirmation_requires_improving_both_controls_each_seed():
    values = rows((1, 2, 4), 41, {1: .00012, 2: .0001, 4: .00008}) + rows((1, 2, 4), 43, {1: .00012, 2: .0001, 4: .00008})
    assert s.confirmation_decision(values, 4)["status"] == "CROSS_SEED_DEVELOPMENT_CONFIRMED"
    for row in values:
        if row["seed"] == 43 and row["response_scale"] == 4: row["E_U"] = .00011
    result = s.confirmation_decision(values, 4)
    assert result["status"] == "NOT_STABLE"
    assert result["seeds"]["43"]["comparisons"]["bota_r1"]["passed"] is True
    assert result["seeds"]["43"]["comparisons"]["bota_r2"]["passed"] is False
    assert len(result["cell_regressions"]) == 2


def test_failed_candidate_cannot_confirm():
    values = rows((1, 2, 4), 41) + rows((1, 2, 4), 43)
    next(r for r in values if r["response_scale"] == 4).update(E_U=None, status="NONFINITE_TRANSPORT")
    assert s.confirmation_decision(values, 4)["status"] == "NOT_STABLE"


def test_frozen_decision_refuses_replacement(tmp_path):
    result = s.coarse_decision(rows(), .0001)
    digest = s.freeze_decision(tmp_path, "coarse", result)
    assert s.freeze_decision(tmp_path, "coarse", result) == digest
    result["fine_grid"] = [3, 5]
    with pytest.raises(RuntimeError): s.freeze_decision(tmp_path, "coarse", result)


def toy(monkeypatch):
    engine = {"schedule": {"batch_size": 2}, "optimizer": {
        "betas": [.9, .99], "learning_rate": .001, "eps": 1e-6, "weight_decay": .01}}
    data = torch.tensor([[.4, 1.], [1.3, .2]] * 200)
    monkeypatch.setattr(s.s8, "masked_batch", lambda dataset, chosen, pad: dataset[chosen])
    monkeypatch.setattr(s.s8, "move_batch", lambda batch, device: batch)
    monkeypatch.setattr(s.s8.p1, "_sample_losses", lambda model, batch: ((batch @ model).squeeze(-1) - .3).square())
    def arguments():
        parameter = torch.tensor([[.2], [.5]], requires_grad=True)
        return (parameter, data, list(range(400)), [1, 2]*200, [1, 2], [parameter], ["p"],
                torch.device("cpu"), 0, engine, s.s8.StepBudget(200))
    return arguments


def test_200_step_controls_match_stage12_exactly(monkeypatch, tmp_path):
    args = toy(monkeypatch)
    torch.manual_seed(7)
    prior = s.s12.run_transport(*args(), (1., 2.), tmp_path/"old.json")
    torch.manual_seed(7)
    current = s.run_transport(*args(), (1., *s.COARSE_GRID), tmp_path/"new.json")
    assert torch.equal(prior[0]["p"], current[0]["p"])
    for scale in (1, 2):
        for key in ("theta", "m", "v"):
            assert torch.equal(prior[1][s.cid(scale)][0][key], current[1][s.cid(scale)][0][key])
    assert prior[3:] == current[3:]


@pytest.mark.parametrize("scale,message,aborts", [(16, "nonfinite_stable_adamw_tangent", False),
    (2, "nonfinite_stable_adamw_tangent", True), (1, "nonfinite_stable_adamw_tangent", True),
    (16, "CUDA out of memory", True)])
def test_numeric_vs_control_and_system_failure(monkeypatch, tmp_path, scale, message, aborts):
    args = toy(monkeypatch)
    real = s.s8._advance_scaled
    def advance(*args):
        if args[-1] == scale: raise RuntimeError(message)
        return real(*args)
    monkeypatch.setattr(s.s8, "_advance_scaled", advance)
    if aborts:
        with pytest.raises(RuntimeError, match=message): s.run_transport(*args(), (1., 2., 16.), tmp_path/"progress.json")
    else:
        result = s.run_transport(*args(), (1., 2., 16.), tmp_path/"progress.json")
        assert result[2]["bota_r16"]["first_failed_step"] == 1


def test_completed_cell_resume_is_phase_and_decision_bound(tmp_path):
    identity = {"contract_sha256": "contract", "seed": 42, "scenario": "L8", "scales": [1., 2., 3., 6.], "selection_sha256": "decision", "phase": "fine"}
    stage = s.staging(tmp_path)
    payload = {"identity": identity, "rows": []}
    s.atomic_json(stage/"metrics.json", payload)
    s.publish(stage, tmp_path/"cells/fine/s42_L8")
    assert s.execute_cell(ROOT, tmp_path, {}, tmp_path, "contract", 42, "L8", (1., 2., 3., 6.), "decision", "fine") == payload
    with pytest.raises(ValueError): s.execute_cell(ROOT, tmp_path, {}, tmp_path, "contract", 42, "L8", (1., 2., 3., 6.), "changed", "fine")


@pytest.mark.parametrize("scales", [(2.,4.,8.,16.), (1.,2.,32.), (1.,2.,2.)])
def test_invalid_scales_rejected_before_loading_gpu(tmp_path, scales):
    with pytest.raises(ValueError): s.execute_cell(ROOT, tmp_path, {}, tmp_path, "h", 42, "L8", scales)


@pytest.mark.parametrize("case", ["fine", "upper", "tiny", "retain"])
def test_pipeline_branches_freeze_before_run_and_resume(monkeypatch, tmp_path, case):
    config = s.load_config(ROOT, CONFIG)
    root = tmp_path / config["output_root"]
    monkeypatch.setattr(s, "preflight", lambda *args: {})
    monkeypatch.setattr(s, "freeze_protocol", lambda *args: "contract")
    monkeypatch.setattr(s, "check_frozen", lambda *args: "contract")
    monkeypatch.setattr(s, "historical_rows", lambda *args: rows(s.HISTORICAL_GRID, scores={2: .0001}))
    calls = []
    def cell(repo, workspace, config, root, contract, seed, sid, scales, selection_hash=None, phase="coarse"):
        calls.append((phase, seed, sid, scales))
        if phase == "fine":
            assert selection_hash == s.sha256_file(root/"coarse/decision.json")
            assert scales == (1., 2., 3., 6.)
        if phase == "confirm":
            assert selection_hash == s.sha256_file(root/"selection/decision.json")
            assert scales == (1., 2., 3.) if case == "fine" else scales == (1., 2., 16.)
        scores = {1: .00012, 2: .0001, 4: .00008, 8: .00009, 16: .00011, 3: .00007, 6: .000075}
        if case == "upper": scores.update({4: .00009, 8: .00008, 16: .00007})
        if case == "tiny": scores.update({4: .0000995, 8: .00012, 16: .00013})
        if case == "retain": scores.update({4: .00011, 8: .00012, 16: .00013})
        return {"rows": [r for r in rows(scales, seed, scores) if r["scenario"] == sid]}
    monkeypatch.setattr(s, "execute_cell", cell)
    result = s.execute(ROOT, tmp_path, CONFIG, "All")
    count = {"fine": 8, "upper": 6, "tiny": 2, "retain": 2}[case]
    assert len(calls) == count
    assert result["status"] == "COMPLETED"
    assert result["confirmation"]["finaltest_authorized"] is False
    assert s.execute(ROOT, tmp_path, CONFIG, "All") == result
    s.verify_bundle(root/"summary")
    assert s.analyze(ROOT, tmp_path, config)["status"] == "COMPLETED_VERIFIED"
