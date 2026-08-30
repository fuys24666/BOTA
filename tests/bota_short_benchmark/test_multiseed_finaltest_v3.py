from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from src.bota_short_benchmark import multiseed_finaltest_v3 as finaltest
from src.bota_short_benchmark import paper_v2, protocol as protocol_module, runner


ROOT = Path(__file__).resolve().parents[2]


def test_prepare_binds_request_and_run_seeds_without_name_error(tmp_path, monkeypatch):
    source = {}
    for name in ("development_json", "original_checkpoint", "base_config", "trajectory_config"):
        path = tmp_path / name
        path.write_text("[]" if name == "development_json" else "fixture\n", encoding="utf-8")
        source[name] = path.name
    train = tmp_path / "train_json"
    train.write_text("[" + ",".join("{}" for _ in range(3200)) + "]", encoding="utf-8")
    source["train_json"] = train.name
    source["dataset"] = "ml-1m"
    config = {"output_root": "outputs", "source": source, "protocol": {"seed": 42, "run_seed": 41}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    scenarios = [
        {"id": scenario, "forget_window_positions": list(range(8)), "deleted_interactions": 8}
        for scenario in ("L8", "L4M4")
    ]
    registry = {
        "order": list(range(3200)),
        "scenarios": scenarios,
        "public_scenarios": [{"id": row["id"]} for row in scenarios],
        "registry_sha256": "registry",
        "order_sha256": "order",
        "batch_order_sha256": "batch-order",
        "global_train_samples": 3200,
        "development_samples": 64,
    }
    monkeypatch.setattr(protocol_module, "load_config", lambda _: config)
    monkeypatch.setattr(protocol_module, "freeze_registry", lambda *_: registry)
    monkeypatch.setattr(protocol_module, "sisa_partition", lambda *_: {shard: {slice_id: [] for slice_id in range(4)} for shard in range(4)})
    monkeypatch.setattr(protocol_module, "deletion_plan", lambda *_: {})
    monkeypatch.setattr(protocol_module, "text_features", lambda *_: torch.zeros((3200, 32)))
    monkeypatch.setattr(protocol_module, "balanced_similarity_partition", lambda *_: [index % 4 for index in range(3200)])

    result = protocol_module.prepare(tmp_path, config_path, "seed41")
    prepared = Path(result["protocol_dir"])
    contract = json.loads((prepared / "contract.json").read_text(encoding="utf-8"))
    sisa = yaml.safe_load((prepared / "data" / "L8" / "sisa.yaml").read_text(encoding="utf-8"))
    receraser = yaml.safe_load((prepared / "data" / "L8" / "receraser.yaml").read_text(encoding="utf-8"))
    assert contract["request_seed"] == 42
    assert contract["run_seed"] == 41
    assert sisa["seed"] == receraser["seed"] == 41
    assert sisa["protocol"]["partition_seed"] == receraser["protocol"]["partition_seed"] == 42


@pytest.mark.parametrize("seed", [41, 42, 43])
def test_multiseed_configs_bind_training_request_and_coordinate_seed(seed):
    core = runner.load_config(ROOT / f"configs/bota_short_benchmark_seed{seed}_v3.yaml")
    paper = paper_v2.load_config(ROOT / f"configs/bota_short_paper_seed{seed}_v3.yaml")
    assert core["protocol"]["seed"] == 42
    assert core["protocol"]["run_seed"] == seed
    assert core["coordinate"]["fixed_a_seed"] == seed
    assert paper["protocol"]["seed"] == seed
    assert paper["source"]["v1_config"].endswith(f"seed{seed}_v3.yaml")


def test_finaltest_is_joint_across_three_seeds_and_two_registered_scenarios():
    assert finaltest.SEEDS == (41, 42, 43)
    assert finaltest.SCENARIOS == ("L8", "L4M4")
    assert set(finaltest.RUN_NAMES) == set(finaltest.dev_eval.ORDER)


def test_finaltest_refuses_access_without_explicit_confirmation(tmp_path):
    with pytest.raises(RuntimeError, match="ConfirmFinalTest"):
        finaltest.execute(tmp_path, "forbidden", {41: "a", 42: "b", 43: "c"}, False)


def test_finaltest_preflight_does_not_materialize_or_load_test():
    source = (ROOT / "src/bota_short_benchmark/multiseed_finaltest_v3.py").read_text(encoding="utf-8")
    preflight_body = source.split("def preflight", 1)[1].split("def _write_csv", 1)[0]
    assert "_final_lineage" not in preflight_body
    assert "_validate_final_file" not in preflight_body
    assert '"final_test_rows_materialized": 0' in preflight_body


def test_finaltest_has_one_shot_ledger_and_no_training_calls():
    source = (ROOT / "src/bota_short_benchmark/multiseed_finaltest_v3.py").read_text(encoding="utf-8")
    assert "reservation.mkdir()" in source
    assert "FINALTEST_ACCESS_FAILED_NO_RETRY" in source
    assert '"optimizer_steps": 0' in source
    assert '"backward_calls": 0' in source
    assert "optimizer.step(" not in source
    assert ".backward(" not in source


def test_authorized_finaltest_loader_bypasses_only_the_generic_path_guard(tmp_path):
    path = tmp_path / "test" / "final.json"
    path.parent.mkdir()
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(PermissionError, match="capability"):
        finaltest._AuthorizedFinalTestDataset(path, object(), object())
    dataset = finaltest._AuthorizedFinalTestDataset(path, object(), finaltest._FINALTEST_DATASET_CAPABILITY)
    assert len(dataset) == 0


def test_one_shot_binding_survives_git_head_changes_and_detects_legacy_ledger(tmp_path):
    authority = {"seeds": {"41": {"model_runs": {"BOTA": "frozen"}}}}
    binding = finaltest._combination_binding(authority)
    assert binding == finaltest._combination_binding(authority)
    ledger = tmp_path / "ledger"; ledger.mkdir()
    old_head = "a" * 40
    legacy = finaltest.canonical_hash({"schema": finaltest.SCHEMA, "git_head": old_head, "seeds": authority["seeds"], "scenarios": list(finaltest.SCENARIOS)})
    entry = ledger / legacy; entry.mkdir()
    (entry / "access_started.json").write_text(json.dumps({"git_head": old_head}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="already reserved/accessed"):
        finaltest._assert_combination_never_accessed(ledger, authority, binding)


def test_zero_inference_recovery_requires_exact_failed_ledger_and_is_one_shot(tmp_path, monkeypatch):
    authority = {"seeds": {"41": {"model_runs": {"BOTA": "frozen"}}}}
    old_head = "b" * 40
    failed_binding = finaltest.canonical_hash({"schema": finaltest.SCHEMA, "git_head": old_head, "seeds": authority["seeds"], "scenarios": list(finaltest.SCENARIOS)})
    failed = tmp_path / "outputs" / "bota_short_multiseed_finaltest_v3" / "access_ledger" / failed_binding
    failed.mkdir(parents=True)
    started = {"binding_sha256": failed_binding, "git_head": old_head, "status": "FINALTEST_ACCESS_STARTED", "test_accessed": True}
    failure = {"binding_sha256": failed_binding, "status": "FINALTEST_ACCESS_FAILED_NO_RETRY", "reason": "ValueError", "message": "test paths are forbidden in reconstructed diagnostics: test.json", "test_accessed": True}
    (failed / "access_started.json").write_text(json.dumps(started), encoding="utf-8")
    (failed / "access_failed.json").write_text(json.dumps(failure), encoding="utf-8")
    historical = "def execute():\n dataset = JsonPromptDataset(final_path, tokenizer)\n dev_eval._predict(x)\ndef analyze():\n pass\n"
    monkeypatch.setattr(finaltest, "_historical_finaltest_source", lambda *_: historical)
    evidence = finaltest._zero_inference_recovery_evidence(tmp_path, authority, failed_binding, "failed_run")
    assert evidence["prediction_calls_before_failure"] == 0
    assert evidence["metrics_rows_before_failure"] == 0
    assert evidence["model_or_algorithm_change_authorized"] is False
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: True if self.name == "zero_inference_recovery" else original_exists(self))
    with pytest.raises(RuntimeError, match="already reserved"):
        finaltest._zero_inference_recovery_evidence(tmp_path, authority, failed_binding, "failed_run")


def test_recovery_preflight_source_path_does_not_materialize_finaltest():
    source = (ROOT / "src/bota_short_benchmark/multiseed_finaltest_v3.py").read_text(encoding="utf-8")
    body = source.split("def recovery_preflight", 1)[1].split("def execute", 1)[0]
    assert "_final_lineage" not in body
    assert "_validate_final_file" not in body
    assert '"final_test_rows_materialized_during_preflight": 0' in body


def test_finaltest_uses_capability_loader_and_records_failure_boundary():
    source = (ROOT / "src/bota_short_benchmark/multiseed_finaltest_v3.py").read_text(encoding="utf-8")
    execute_body = source.split("\ndef execute(", 1)[1].split("\ndef analyze(", 1)[0]
    assert "JsonPromptDataset(final_path" not in execute_body
    assert "_AuthorizedFinalTestDataset(final_path" in execute_body
    assert '"prediction_calls": prediction_calls' in execute_body
    assert '"metrics_rows_written": metric_rows_written' in execute_body
    assert "Audited infrastructure recovery" in execute_body
    assert '"recovery": recovery_evidence' in execute_body


def test_short_ifru_keeps_frozen_cg_cap_and_records_truncated_fallback():
    source = (ROOT / "src/bota_short_benchmark/runner.py").read_text(encoding="utf-8")
    ifru = source.split("def _ifru", 1)[1].split("def _partitioned", 1)[0]
    assert "max_iterations=40" in ifru
    assert "allow_truncated_solution=True" in ifru


def test_multiseed_wrappers_expose_only_registered_scenarios():
    train = (ROOT / "scripts/bota_if/run_short_multiseed_v3.ps1").read_text(encoding="utf-8")
    final = (ROOT / "scripts/bota_if/run_short_multiseed_finaltest_v3.ps1").read_text(encoding="utf-8")
    assert 'ValidateSet(41,42,43)' in train
    assert 'ValidateSet("All","L8","L4M4")' in train
    assert "ConfirmFinalTest" in final
    assert '"RecoveryPreflight"' in final and '"RecoverZeroInference"' in final
    assert "FailedBinding" in final and "FailedRunName" in final
