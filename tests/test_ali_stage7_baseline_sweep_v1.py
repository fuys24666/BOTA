from pathlib import Path

import yaml

from src.bota_short_benchmark.ali_stage7_baseline_sweep_v1 import METHODS, SCENARIOS, _candidate_ids, _select
from src.paper_baselines.budget import partitioned_budget


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((ROOT / "configs/ali_stage7_baseline_sweep_v1.yaml").read_text(encoding="utf-8"))


def test_frozen_candidate_counts():
    config = _config()
    assert {method: len(_candidate_ids(config, method)) for method in METHODS} == {
        "NegGrad": 20,
        "PCGrad": 12,
        "IFRU": 6,
        "E2URec": 9,
        "SISA": 9,
        "RecEraser": 9,
    }


def test_selection_is_joint_and_lexicographic():
    config = _config(); rows = []
    for number, candidate in enumerate(_candidate_ids(config, "IFRU")):
        for scenario in SCENARIOS:
            rows.append({"candidate_id": candidate, "scenario": scenario, "E_U": .1 + number, "utility_eligible": number != 0, "prediction_collapse": False, "compute_count": 20-number})
    result = _select(rows, config, "IFRU")
    assert result["status"] == "SELECTED"
    assert result["selected"]["candidate_id"] == _candidate_ids(config, "IFRU")[1]
    assert result["selection_uses_finaltest"] is False


def test_no_eligible_candidate_is_not_silently_promoted():
    config = _config(); rows = []
    for candidate in _candidate_ids(config, "PCGrad"):
        for scenario in SCENARIOS:
            rows.append({"candidate_id": candidate, "scenario": scenario, "E_U": 0., "utility_eligible": False, "prediction_collapse": False, "compute_count": 1})
    result = _select(rows, config, "PCGrad")
    assert result["status"] == "NO_ELIGIBLE_CANDIDATE"
    assert result["selected"] is None


def test_partitioned_budget_counts_multiple_epochs():
    config={"protocol":{"epochs_per_slice":3,"effective_batch":16,"physical_microbatch":4,"shards":2},"budget":{"initial_partition_counts":[16,17],"post_delete_partition_counts":[15,16],"total_expected_optimizer_steps":15,"source_paper":"x","official_repository":"x","official_epoch_or_step_budget":"3 epochs","official_checkpoint_used_for_reported_result":"final","estimated_runtime":"bounded"},"data":{"train_samples":33}}
    result=partitioned_budget(config,"receraser")
    assert result["initial_training_optimizer_steps"]==9
    assert result["unlearning_incremental_optimizer_steps"]==6
    assert result["total_expected_optimizer_steps"]==15
