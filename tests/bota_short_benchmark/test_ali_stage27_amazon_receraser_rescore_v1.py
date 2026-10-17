"""CPU-only safety and metric tests; no real model construction or inference."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from src.bota_short_benchmark import ali_stage27_amazon_receraser_rescore_v1 as s


def config(seed=42):
    return {"schema": "paper-receraser-adapter-v1", "seed": seed,
            "output_root": "old", "synthetic_root": "old/synthetic",
            "test_access_policy": "forbidden",
            "protocol": {"partition_seed": seed, "shards": 4, "epochs_per_slice": 1, "learning_rate": .002},
            "sources": {"train": {"path": "train", "sha256": "abc"}}}


def test_relocation_changes_only_paths():
    with TemporaryDirectory() as temp:
        workspace = Path(temp)
        old = config(43)
        new = s.relocated_config(old, workspace, workspace / "outputs/new", "s43_A1")
        assert {k for k in old if old[k] != new[k]} == {"output_root", "synthetic_root"}
        assert old["output_root"] == "old"
        s.validate_config(new, 43)


@pytest.mark.parametrize("field,value", [("learning_rate", .001), ("epochs_per_slice", 2), ("partition_seed", 41)])
def test_scope_cannot_retune(field, value):
    c = config()
    c["protocol"][field] = value
    with pytest.raises(ValueError):
        s.validate_config(c, 42)


def test_reconstruction_requires_explicit_opt_in():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        entry = {"old_run": str(root / "old"), "key": "s42_A1"}
        with patch.object(s.s20.partitioned, "execute") as execute:
            with pytest.raises(PermissionError):
                s.ensure_model(root, root, root, entry, False)
            execute.assert_not_called()
        assert not (root / "configs").exists()


def test_task_resume_and_tamper_detection():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        result = s.task(root, "cell", "contract", lambda _: {"status": "OK", "optimizer_updates": 0})
        with patch.object(s, "atomic_json", side_effect=AssertionError("must reuse")):
            assert s.task(root, "cell", "contract", lambda _: {}) == result
        with pytest.raises(ValueError, match="contract"):
            s.verify(root / "cell", "different")
        (root / "cell/result.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="changed artifact"):
            s.verify(root / "cell")


def test_summary_refuses_incomplete():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        plan = {"entries": [{"key": f"s{seed}_{r}"} for seed in (42, 43) for r in ("A1", "A2", "A3")]}
        result = s.summarize(root, plan)
        assert result["status"] == "INCOMPLETE" and len(result["pending"]) == 6


def test_exact_model_identity_gate():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        run = root / "new"
        unit = root / "old"
        (run / "models").mkdir(parents=True)
        unit.mkdir()
        s.atomic_text(run / "COMPLETED", "test\n")
        s.atomic_text(run / "models/weights", "frozen")
        s.atomic_json(run / "aggregation.json", {"weights": [.25] * 4})
        s.atomic_json(run / "partition.json", {"partition_seed": 43})
        s.atomic_json(unit / "partition.json", {"partition_seed": 43})
        s.atomic_json(run / "paper_model_manifest.json", {"optimizer_steps": 350})
        entry = {"unit": str(unit), "optimizer_steps": 350,
                 "expected_models_sha256": s.digest(run / "models"),
                 "expected_aggregation_sha256": s.digest(run / "aggregation.json")}
        with patch.object(s.s20.partitioned, "analyze_run"):
            assert s.model_match(run, entry)
            s.atomic_json(run / "aggregation.json", {"weights": [.1, .2, .3, .4]})
            with pytest.raises(ValueError, match="aggregation"):
                s.model_match(run, entry)
            s.atomic_text(run / "models/weights", "different")
            with pytest.raises(ValueError, match="tensor artifact"):
                s.model_match(run, entry)


def test_uncommitted_shard_preserved_not_deleted():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        run = root / "models/s42_A1/run"
        shard = run / "models/shard_00"
        shard.mkdir(parents=True)
        s.atomic_text(shard / "final_state.pt", "uncommitted")
        s.quarantine_uncommitted_shards(root, run)
        assert not shard.exists()
        assert len(list((root / "recovery").rglob("final_state.pt"))) == 1


def test_committed_shard_requires_rng():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        run = root / "models/s42_A1/run"
        s.atomic_json(run / "run_state.json", {"next_shard": 1})
        with pytest.raises(ValueError, match="RNG"):
            s.quarantine_uncommitted_shards(root, run)


def test_full_fails_before_training_without_optin():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        plan = {"entries": [{"seed": 42, "key": "s42_A1"}]}
        with patch.object(s, "preflight", return_value={"inventory": [{"status": "REBUILD_REQUIRED"}]}), \
                patch.object(s, "ensure_model") as builder:
            with pytest.raises(PermissionError):
                s.execute(root, root, root, plan, "All", False)
            builder.assert_not_called()


def test_summary_keeps_failed_utility_and_missing_solver(tmp_path):
    """Actual Stage25 formulas, not a new hand-coded comparison or success filter."""
    indices = [0, 1, 2, 3]
    pred = lambda p: {"probability": p, "answer_loss": [.2] * 4, "gold_label": [0, 1, 0, 1],
                      "sample_order_sha256": s.canonical_hash(indices)}
    original = pred([.1, .9, .2, .8])
    exact = pred([.15, .85, .25, .75])
    bad = pred([.9, .1, .8, .2])
    cohort = {"dataset": "Amazon", "scope": "Development", "lane": "binary", "indices": indices,
              "users": [5, 5, 6, 6], "selection": {"auc_damage_max": .005, "logloss_damage_max": .01}}
    jobs = [{"id": name, "method": name, "cohort": "Amazon_comparison_s43", "seed": 43,
             "request": "A1", "old": None, "provenance": {"users": [5, 6], "historical_eligible": True,
             "calibration_status": "SELECTED"}} for name in ["Original", "Exact-Masked", "RecEraser", "IFRU"]]
    results = {"Original": {"status": "OK", "prediction": original}, "Exact-Masked": {"status": "OK", "prediction": exact},
               "RecEraser": {"status": "OK", "prediction": bad}, "IFRU": {"status": "HISTORICAL_SOLVER_FAILURE", "prediction": None}}
    rows = s.s25.metric_rows(cohort, jobs, results)
    rec = next(r for r in rows if r["method"] == "RecEraser")
    assert not rec["utility_eligible"] and rec["historical_eligible"]
    assert rec["seed"] == 43 and len(rows) == 4
    assert next(r for r in rows if r["method"] == "IFRU")["status"] == "HISTORICAL_SOLVER_FAILURE"


def test_seed_selection_does_not_replace_requests():
    plan = {"entries": [{"seed": seed, "key": f"s{seed}_{r}"} for seed in (42, 43) for r in ("A1", "A2", "A3")]}
    assert [e["key"] for e in s.selected_entries(plan, "43")] == ["s43_A1", "s43_A2", "s43_A3"]
    assert len(s.selected_entries(plan, "All")) == 6


def test_metadata_only_repair_and_context_cleanup(tmp_path):
    engine = s.s20.partitioned
    writer = engine.atomic_json
    constructor = engine._new_local_model
    archived = {"trainable_parameters": 123, "total_parameters": 456}
    with patch.object(engine, "paper_model_manifest", side_effect=lambda **kwargs: kwargs) as manifest:
        with pytest.raises(RuntimeError, match="intentional"):
            with s.reconstruction_checkpoints(tmp_path / "run", archived):
                value = engine.paper_model_manifest(trainable_parameters=0, total_parameters=0)
                assert value == archived
                raise RuntimeError("intentional")
        assert engine.paper_model_manifest is manifest
    assert engine.atomic_json is writer
    assert engine._new_local_model is constructor
