from pathlib import Path

import numpy as np

from src.bota_short_benchmark.ali_stage4_attack_v1 import CollatingConcatDataset, DISPLAY, SCENARIOS, SEEDS, _arrays, synthetic
from src.diagnostics.t5_zero_training_analysis import ATTACKS


def test_stage4_contract_surface() -> None:
    assert SEEDS == (41, 42, 43)
    assert SCENARIOS == ("L8", "L4M4")
    assert len(DISPLAY) == 10
    assert tuple(ATTACKS) == ("negative_answer_loss", "confidence", "negative_entropy", "absolute_yes_no_margin")


def test_attack_arrays_keep_authoritative_users() -> None:
    members = [{"user_id": 1, "answer_sequence_loss": 0.2}]
    nonmembers = [{"user_id": 1, "answer_sequence_loss": 0.8}]
    m, n, mu, nu = _arrays(members, nonmembers, ATTACKS["negative_answer_loss"])
    assert np.allclose(m, [-0.2]) and np.allclose(n, [-0.8])
    assert mu.tolist() == nu.tolist() == [1]


def test_synthetic_dry_run_is_user_clustered_and_test_free() -> None:
    result = synthetic()
    assert result["pooled_auc"] == 1.0
    assert result["bootstrap_unit"] == "matched_user"
    assert result["real_model_loaded"] is False
    assert result["test_accessed"] is False


def test_concat_dataset_preserves_first_dataset_collator() -> None:
    class Dataset:
        def __init__(self, values: list[int]) -> None: self.values = values
        def __len__(self) -> int: return len(self.values)
        def __getitem__(self, index: int) -> int: return self.values[index]
        def collate_fn(self, rows: list[int]) -> dict[str, list[int]]: return {"rows": rows}
    dataset = CollatingConcatDataset([Dataset([1]), Dataset([2, 3])])
    assert [dataset[index] for index in range(len(dataset))] == [1, 2, 3]
    assert dataset.collate_fn([1, 2]) == {"rows": [1, 2]}


def test_runbook_has_offline_command() -> None:
    text = Path("docs/ALI_STAGE4_RUNBOOK.md").read_text(encoding="utf-8")
    assert "-Mode Full" in text and "-Seed All" in text
    assert "FinalTest is forbidden" in text
