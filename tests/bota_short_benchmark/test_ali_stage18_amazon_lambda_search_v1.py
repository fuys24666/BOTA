from pathlib import Path

import pytest
import torch

from src.bota_short_benchmark import ali_stage18_amazon_lambda_search_v1 as s


def rows(scales, seed=42, best=4):
    return [{"seed": seed, "scenario": sid, "response_scale": x,
             "status": "OK", "E_U": .01 + abs(x-best)*.001,
             "utility_eligible": True, "prediction_collapse": False}
            for x in scales for sid in s.SCENARIOS]


def test_internal_winner_and_one_midpoint_round():
    d = s.choose(rows(s.GRID), s.GRID)
    assert d["selected"]["lambda"] == 4
    assert s.fine_grid(d) == [3, 6]
    with pytest.raises(ValueError):
        s.choose(rows(s.GRID)[:-1], s.GRID)
    with pytest.raises(ValueError):
        s.choose(rows(s.GRID) + rows([0]), s.GRID)


def test_boundary_and_negligible_gain_stop():
    assert s.fine_grid(s.choose(rows(s.GRID, best=256), s.GRID)) == []
    assert s.fine_grid(s.choose(rows(s.GRID, best=0), s.GRID)) == []
    r = rows(s.GRID)
    for x in r:
        x["E_U"] = .01 if x["response_scale"] != 4 else .01 - 1e-8
    assert s.fine_grid(s.choose(r, s.GRID)) == []


def test_gates_failure_ties_and_no_eligible():
    r = rows([0, 1, 2], best=1)
    for x in r:
        if x["response_scale"] == 1:
            x["status"] = "NUMERICAL_TRANSPORT_FAILURE"
            x["E_U"] = None
            x["utility_eligible"] = False
    assert s.choose(r, [0, 1, 2])["selected"]["lambda"] == 0
    for x in r:
        x["prediction_collapse"] = True
    assert s.choose(r, [0, 1, 2])["status"] == "NO_ELIGIBLE_CANDIDATE"


def test_confirmation_never_reselects():
    r = rows([0, 1, 4, 64], 41) + rows([0, 1, 4, 64], 43)
    assert s.confirmation(r, 4)["status"] == "CONFIRMED"
    for x in r:
        if x["seed"] == 43 and x["scenario"] == "K2" and x["response_scale"] == 4:
            x["E_U"] = .03
    d = s.confirmation(r, 4)
    assert d["status"] == "NOT_CONFIRMED"
    assert d["selected_lambda_unchanged"] == 4
    assert d["per_seed"]["43"]["regressed_requests"] == ["K2"]


def test_atomic_inventory_and_decision_immutability(tmp_path):
    original = {"selected": 4, "test_accessed": False}
    first = s.freeze_json(tmp_path, "selection", original)
    assert s.freeze_json(tmp_path, "selection", original) == first
    with pytest.raises(ValueError, match="refusing"):
        s.freeze_json(tmp_path, "selection", {"selected": 8})
    extra = tmp_path / "selection" / "extra.json"
    s.atomic_json(extra, {})
    with pytest.raises(ValueError, match="inventory"):
        s.verify(tmp_path / "selection")


def test_source_hash_tamper(tmp_path):
    p = tmp_path / "fake.json"
    s.atomic_json(p, {"a": 1})
    checks = {"source_sha256": {"fake.json": s.sha256_file(p)}}
    s.source_check(tmp_path, checks)
    s.atomic_json(p, {"a": 2})
    with pytest.raises(RuntimeError, match="changed"):
        s.source_check(tmp_path, checks)


def test_composition_and_dimension_guard():
    canonical = {"a": torch.tensor([[1., 2.]]), "b": torch.tensor([3.])}
    got = s.compose(canonical, ["a", "b"], torch.tensor([.1, .2, .3]))
    torch.testing.assert_close(got["a"], torch.tensor([[1.1, 2.2]]))
    torch.testing.assert_close(got["b"], torch.tensor([3.3]))
    with pytest.raises(ValueError, match="dimension"):
        s.compose(canonical, ["a", "b"], torch.ones(4))


def test_frozen_config_and_scope():
    repo = Path(__file__).resolve().parents[2]
    c = s.load_config(repo, repo / "configs/ali_stage18_amazon_lambda_search_v1.json")
    assert c["protocol"]["confirmation_changes_selection"] is False
    assert c["protocol"]["source_scaled"] is False
    assert c["test_access_policy"] == "forbidden"
    assert c["protocol"]["maximum_lambda"] == 256


def test_full_orchestration_and_verified_resume(tmp_path, monkeypatch):
    c = {"output_root": "outputs/test"}
    called = []
    checks = {"source_sha256": {}, "input_sha256": {}}
    monkeypatch.setattr(s, "audit", lambda *a: (checks, {}, {}, []))
    def fake_phase(root, workspace, b, reg, check, users, seed, scales, phase, contract):
        if seed != 42:
            s.verify(root / "selection")
            assert s.read(root / "selection/result.json")["selected"]["lambda"] == 4
        called.append((seed, phase))
        out = rows(scales, seed)
        for r in out:
            r["phase"] = phase
        s.freeze_json(root, f"metrics/{phase}_{seed}", {"rows": out})
        return out
    monkeypatch.setattr(s, "phase_rows", fake_phase)
    result = s.run(tmp_path, tmp_path, c, tmp_path / "config", "Coarse")
    assert result["status"] == "COARSE_COMPLETED"
    assert called == [(42, "coarse")]
    result = s.run(tmp_path, tmp_path, c, tmp_path / "config", "All")
    assert result["confirmation"]["status"] == "CONFIRMED"
    assert called[-4:] == [(42, "coarse"), (42, "fine"), (41, "confirm"), (43, "confirm")]
    before = s.sha256_file(tmp_path / "outputs/test/summary/result.json")
    again = s.run(tmp_path, tmp_path, c, tmp_path / "config", "All")
    assert again == result
    assert s.sha256_file(tmp_path / "outputs/test/summary/result.json") == before


def test_candidate_metric_preserves_utility_gate():
    indices = [0, 1, 2, 3]
    original = {"probability": [.1, .9, .2, .8], "gold_label": [0, 1, 0, 1]}
    exact = {**original, "probability": [.11, .89, .21, .79]}
    m = s.s8._metric("bota_r4", 4, "K2", exact, original, exact, indices, 0, 0, {"selection": s.SELECT})
    assert m["E_U"] == 0
    assert m["logloss_damage"] > .01
    assert m["utility_eligible"] is False
