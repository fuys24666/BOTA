from pathlib import Path

import pytest
import torch

from src.bota_short_benchmark import ali_stage14_lambda_search_v1 as s

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ali_stage14_lambda_search_v1.json"


def rows(scales=s.COARSE_GRID, seed=42, scores=None):
    scores = scores or {0: .00014, 1: .00012, 16: .0001, 32: .00008, 64: .00009, 128: .00011}
    return [{"candidate_id": s.cid(scale), "response_scale": float(scale), "seed": seed,
             "scenario": sid, "E_U": scores.get(scale, .00015), "D_U": .001,
             "local_residual": scores.get(scale, .00015)/.001, "status": "OK",
             "utility_eligible": True, "prediction_collapse": False,
             "toward_exact_rate": .7, "origin": "synthetic", "test_accessed": False}
            for scale in scales for sid in s.SCENARIOS]


def test_frozen_config_and_old_grids():
    config = s.load_config(ROOT, CONFIG)
    assert config["protocol"]["maximum_lambda"] == 128
    assert config["protocol"]["ablation_seeds"] == [41, 42, 43]
    assert config["protocol"]["final_test"] is False
    assert s.s13.COARSE_GRID == (2., 4., 8., 16.)
    assert len(config["parent_manifests"]) == 10


@pytest.mark.parametrize("winner,grid", [(32, [24, 48]), (64, [48, 96])])
def test_midpoint_rule(winner, grid):
    scores = {16: .0001, 32: .00009, 64: .00009, 128: .00011, winner: .00008}
    assert s.coarse_decision(rows(scores=scores), .0001)["fine_grid"] == grid


@pytest.mark.parametrize("scores,action,boundary", [
    ({16: .0001, 32: .00009, 64: .00008, 128: .00007}, "UPPER_BOUNDARY_STOP_NO_EXPANSION", True),
    ({16: .00008, 32: .00009, 64: .0001, 128: .00011}, "LOWER_BOUNDARY_RETAIN_STAGE13", False),
    ({16: .0001, 32: .0000995, 64: .00012, 128: .00013}, "GAIN_BELOW_BUDGET_GATE", False),
])
def test_fixed_stop_rules(scores, action, boundary):
    result = s.coarse_decision(rows(scores=scores), scores[16])
    assert result["fine_grid"] == []
    assert result["next_action"] == action
    assert result["boundary_winner"] is boundary


def test_budget_requires_both_thresholds():
    assert not s.gain_over_sixteen({"equal_scenario_mean_EU": .999998}, 1)["passes_budget_gate"]
    assert not s.gain_over_sixteen({"equal_scenario_mean_EU": .0000095}, .00001)["passes_budget_gate"]
    assert s.gain_over_sixteen({"equal_scenario_mean_EU": .00009}, .0001)["passes_budget_gate"]


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unexpected"])
def test_grid_inventory(kind):
    values = rows()
    if kind == "missing": values.pop()
    elif kind == "duplicate": values.append(values[0])
    else: values[0]["candidate_id"] = "bota_r256"
    with pytest.raises(ValueError): s.coarse_decision(values, .0001)


def test_failed_candidates_retained():
    values = rows()
    for r in values:
        if r["response_scale"] == 32: r.update(E_U=None, status="NUMERICAL_TRANSPORT_FAILURE")
    result = s.coarse_decision(values, .0001)
    assert len(result["candidates"]) == 4
    assert result["selected"]["response_scale"] == 64


def test_confirmation_against_16_not_zero():
    values = rows((0, 1, 16, 32), 41) + rows((0, 1, 16, 32), 43)
    for r in values:
        if r["response_scale"] == 0: r.update(E_U=None, status="NONFINITE_PREDICTIONS", utility_eligible=False)
    assert s.confirmation_decision(values, 32)["status"] == "CROSS_SEED_DEVELOPMENT_CONFIRMED"
    for r in values:
        if r["seed"] == 43 and r["response_scale"] == 32: r["E_U"] = .00011
    result = s.confirmation_decision(values, 32)
    assert result["status"] == "NOT_STABLE"
    assert len(result["cell_regressions"]) == 2


def ablation_values():
    return sum((rows((0, 1, 16, 32), seed) for seed in (41, 42, 43)), [])


def test_ablation_all_six_cells_and_both_targets():
    result = s.ablation_report(ablation_values(), 32)
    assert len(result["rows"]) == 12
    assert result["aggregates"]["bota_r32"]["improved_cells"] == 6
    assert result["aggregates"]["bota_r16"]["finite_cells"] == 6
    assert result["aggregates"]["bota_r32"]["relative_gain"] == pytest.approx(1-.00008/.00014)
    assert not result["statistical_significance_tested"]


def test_ablation_failure_not_dropped_or_replaced():
    values = ablation_values()
    next(r for r in values if r["response_scale"] == 32).update(E_U=None, status="NUMERICAL_TRANSPORT_FAILURE")
    result = s.ablation_report(values, 32)
    assert result["aggregates"]["bota_r32"]["finite_cells"] == 5
    assert result["aggregates"]["bota_r32"]["target_mean_EU"] is None
    assert result["aggregates"]["bota_r16"]["finite_cells"] == 6
    assert len(result["rows"]) == 12


def test_zero_denominator_and_worsening():
    values = ablation_values()
    for r in values:
        if r["response_scale"] == 0: r["E_U"] = 0.
    result = s.ablation_report(values, 32)
    assert result["aggregates"]["bota_r32"]["relative_gain"] is None
    assert result["aggregates"]["bota_r32"]["worsened_cells"] == 6
    assert all(r["relative_gain"] is None for r in result["rows"])


def test_ablation_missing_or_duplicate_rejected():
    values = ablation_values()
    with pytest.raises(ValueError): s.ablation_report(values[:-1], 32)
    with pytest.raises(ValueError): s.ablation_report(values+[values[0]], 32)


def toy(monkeypatch):
    engine = {"schedule": {"batch_size": 2}, "optimizer": {
        "betas": [.9, .99], "learning_rate": .001, "eps": 1e-6, "weight_decay": .01}}
    data = torch.tensor([[.4, 1.], [1.3, .2]] * 200)
    monkeypatch.setattr(s.s8, "masked_batch", lambda dataset, chosen, pad: dataset[chosen])
    monkeypatch.setattr(s.s8, "move_batch", lambda batch, device: batch)
    monkeypatch.setattr(s.s8.p1, "_sample_losses", lambda model, batch: ((batch @ model).squeeze(-1) - .3).square())
    def arguments():
        p = torch.tensor([[.2], [.5]], requires_grad=True)
        return (p, data, list(range(400)), [1, 2]*200, [1, 2], [p], ["p"], torch.device("cpu"), 0, engine, s.s8.StepBudget(200))
    return arguments


def test_200_step_recurrence_matches_parent_for_zero_and_controls(monkeypatch, tmp_path):
    args = toy(monkeypatch)
    torch.manual_seed(7)
    old = s.s13.run_transport(*args(), (0., 1., 16.), tmp_path/"old.json")
    torch.manual_seed(7)
    new = s.run_transport(*args(), (0., 1., *s.COARSE_GRID), tmp_path/"new.json")
    assert torch.equal(old[0]["p"], new[0]["p"])
    for scale in (0, 1, 16):
        for key in ("theta", "m", "v"):
            assert torch.equal(old[1][s.cid(scale)][0][key], new[1][s.cid(scale)][0][key])
    assert old[3:] == new[3:]


@pytest.mark.parametrize("scale,message,aborts", [(128, "nonfinite_stable_adamw_tangent", False),
    (0, "nonfinite_stable_adamw_tangent", False), (16, "nonfinite_stable_adamw_tangent", True),
    (1, "nonfinite_stable_adamw_tangent", True), (128, "CUDA out of memory", True)])
def test_failure_classification(monkeypatch, tmp_path, scale, message, aborts):
    args = toy(monkeypatch)
    real = s.s8._advance_scaled
    def advance(*args):
        if args[-1] == scale: raise RuntimeError(message)
        return real(*args)
    monkeypatch.setattr(s.s8, "_advance_scaled", advance)
    if aborts:
        with pytest.raises(RuntimeError, match=message): s.run_transport(*args(), (0., 1., 16., 128.), tmp_path/"progress.json")
    else:
        value = s.run_transport(*args(), (0., 1., 16., 128.), tmp_path/"progress.json")
        assert value[2][s.cid(scale)]["first_failed_step"] == 1


@pytest.mark.parametrize("scales", [(16.,32.,64.,128.), (1.,16.,256.), (1.,16.,-1.), (1.,16.,16.)])
def test_invalid_scales_rejected_before_loading_model(tmp_path, scales):
    with pytest.raises(ValueError): s.execute_cell(ROOT, tmp_path, {}, tmp_path, "h", 42, "L8", scales)


def test_resume_requires_same_phase_and_decision(tmp_path):
    identity = {"contract_sha256": "h", "seed": 41, "scenario": "L8", "scales": [0.,1.,16.,128.], "selection_sha256": "d", "phase": "confirm"}
    stage = s.staging(tmp_path)
    payload = {"identity": identity, "rows": []}
    s.atomic_json(stage/"metrics.json", payload)
    s.publish(stage, tmp_path/"cells/confirm/s41_L8")
    assert s.execute_cell(ROOT,tmp_path,{},tmp_path,"h",41,"L8",(0.,1.,16.,128.),"d","confirm") == payload
    with pytest.raises(ValueError): s.execute_cell(ROOT,tmp_path,{},tmp_path,"h",41,"L8",(0.,1.,16.,128.),"new","confirm")


@pytest.mark.parametrize("case", ["fine32", "fine64", "upper", "tiny", "retain", "failed_confirmation"])
def test_full_branches_always_complete_ablation(monkeypatch,tmp_path,case):
    cfg=s.load_config(ROOT,CONFIG);root=tmp_path/cfg["output_root"]
    monkeypatch.setattr(s,"preflight",lambda *args:{})
    def freeze(repo,workspace,path,config,root,checks):
        if not (root/"protocol").exists():
            stage=s.staging(root);s.atomic_json(stage/"contract.json",{});s.publish(stage,root/"protocol")
        return s.sha256_file(root/"protocol/contract.json")
    monkeypatch.setattr(s,"freeze_protocol",freeze)
    monkeypatch.setattr(s,"check_frozen",lambda *args:"unused")
    monkeypatch.setattr(s,"historical_rows",lambda *args:rows(s.HISTORICAL_GRID,scores={0:.00014,1:.00012,16:.0001}))
    calls=[]
    def cell(repo,workspace,config,root,contract,seed,sid,scales,selection_hash=None,phase="coarse"):
        calls.append((phase,seed,sid,scales))
        if phase=="fine":
            assert selection_hash==s.sha256_file(root/"coarse/decision.json")
            assert scales==(1.,16.,48.,96.) if case=="fine64" else scales==(1.,16.,24.,48.)
        if phase=="confirm":
            assert selection_hash==s.sha256_file(root/"selection/decision.json")
            assert {0.,1.,16.}.issubset(scales)
        scores={0:.00014,1:.00012,16:.0001,32:.00008,64:.00009,128:.00011,24:.00007,48:.000075,96:.000072}
        if case=="fine64":scores.update({32:.00009,64:.00008,48:.00007})
        if case=="upper":scores.update({32:.00009,64:.00008,128:.00007})
        if case=="tiny":scores.update({32:.0000995,64:.00012,128:.00013})
        if case=="retain":scores.update({32:.00011,64:.00012,128:.00013})
        if case=="failed_confirmation" and seed==43:scores[24]=.00013
        return {"rows":[r for r in rows(scales,seed,scores) if r["scenario"]==sid]}
    monkeypatch.setattr(s,"execute_cell",cell)
    result=s.execute(ROOT,tmp_path,CONFIG,"All")
    assert len(calls)==(8 if case in ("fine32","fine64","failed_confirmation") else 6)
    assert result["status"]=="COMPLETED"
    assert result["fisher_ablation"]["bota_r16"]["finite_cells"]==6
    if case in ("tiny","retain"):
        assert result["confirmation"]["status"]=="NOT_TRIGGERED_RETAIN_STAGE13_REFERENCE"
        assert len(result["fisher_ablation"])==1
        assert all(c[3]==(0.,1.,16.) for c in calls if c[0]=="confirm")
    if case=="failed_confirmation":
        assert result["confirmation"]["status"]=="NOT_STABLE"
        assert "bota_r24" in result["fisher_ablation"]
    assert s.execute(ROOT,tmp_path,CONFIG,"All")==result
    assert s.analyze(ROOT,tmp_path,cfg)["status"]=="COMPLETED_VERIFIED"
    s.verify_bundle(root/"summary")
