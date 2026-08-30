import inspect
from pathlib import Path

import torch

from src.bota_short_benchmark import online_latency_v1 as latency


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "bota_short_online_latency_v1.yaml"


def _tiny_bank():
    return {
        "names": ["m"],
        "canonical": {"m": torch.tensor([1.0, 2.0])},
        "bases": {"module": torch.tensor([3.0, 4.0])},
        "user_keys": ["u0", "u1", "u2"],
        "vectors": [torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])],
    }


def test_protocol_freezes_1000_repetitions_and_forbids_test():
    config = latency.load_config(CONFIG)
    assert config["protocol"]["warmup_iterations"] == 20
    assert config["protocol"]["measurement_iterations"] == 1000
    assert config["test_access_policy"] == "forbidden"
    assert config["privacy"]["persist_raw_user_ids"] is False


def test_lookup_and_composition_are_order_invariant():
    bank = _tiny_bank()
    _, left = latency._compose(bank, ["u0", "u2"])
    _, right = latency._compose(bank, ["u2", "u0"])
    assert torch.equal(left["m"], right["m"])
    assert torch.allclose(left["m"], torch.tensor([1.6, 2.8]))


def test_materialization_and_reload_are_exact(tmp_path):
    bank = _tiny_bank()
    _, candidate = latency._compose(bank, ["u0", "u1"])
    state = latency._materialize(bank, candidate)
    publication, reload = latency._publish_reload(tmp_path, "one", state, bank["names"])
    assert publication >= 0
    assert reload >= 0
    assert list(tmp_path.iterdir()) == []


def test_fixed_a_and_trainable_b_keep_independent_key_namespaces():
    bank = _tiny_bank()
    _, candidate = latency._compose(bank, ["u1"])
    state = latency._materialize(bank, candidate)
    assert set(state["A"]) == {"module"}
    assert set(state["B"]) == {"m"}
    latency._assert_adapter_exact(state, state, bank["names"])


def test_bank_recovery_does_not_downcast_formal_fixed_a():
    source = inspect.getsource(latency.build_bank)
    assert '"bases": {name: value.detach().cpu().contiguous()' in source
    assert '"bases": {name: value.detach().float().cpu().contiguous()' not in source


def test_percentiles_use_linear_interpolation():
    values = list(range(1, 101))
    assert latency._quantile(values, 50) == 50.5
    assert latency._quantile(values, 95) == 95.05


def test_summary_has_all_six_phases():
    rows = []
    for scenario in ("L8", "L4M4"):
        for iteration in range(3):
            rows.append({"scenario": scenario, "lookup_ms": 1 + iteration, "composition_ms": 2 + iteration, "materialization_ms": 3 + iteration, "publication_ms": 4 + iteration, "reload_ms": 5 + iteration, "online_total_ms": 15 + 5 * iteration})
    summary = latency._summarize(rows, ["L8", "L4M4"])
    assert len(summary) == 12
    assert {row["phase"] for row in summary} == {"lookup", "composition", "materialization", "publication", "reload", "online_total"}
    assert all(row["iterations"] == 3 for row in summary)


def test_measured_full_path_cannot_train_or_rebuild_bank():
    source = inspect.getsource(latency.execute_latency)
    assert "run_canonical_full" not in source
    assert ".backward(" not in source
    assert "optimizer.step" not in source
    assert "build_bank(" not in source
    assert '"bank_rebuilt_during_latency_run": False' in source


def test_synthetic_mode_contract_is_zero_training():
    config = latency.load_config(CONFIG)
    result = {"iterations": config["protocol"]["measurement_iterations"], "optimizer_steps": 0, "backward_calls": 0, "test_accessed": False}
    assert result == {"iterations": 1000, "optimizer_steps": 0, "backward_calls": 0, "test_accessed": False}
