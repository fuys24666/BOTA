import csv
from copy import deepcopy
from pathlib import Path

import pytest

from src.bota_short_benchmark import ali_stage10_finaltest_v1 as stage10


def _rows():
    return [{"seed": seed, "scenario": sid, "method": method, **{field: .1 for field in stage10.SUMMARY_FIELDS}, "E_U": .1 if method == "BOTA" else .2, "local_exact_masked_residual": .5, "utility_eligible": True, "prediction_collapse": False} for seed in stage10.SEEDS for sid in stage10.SCENARIOS for method in stage10.METHODS]


def test_finaltest_capability_rejects_unauthorized_loader(tmp_path):
    with pytest.raises(PermissionError, match="capability"):
        stage10._AuthorizedDataset(tmp_path / "nonexistent.json", None, object())


def test_full_requires_confirmation_before_any_config_or_test_read(tmp_path):
    with pytest.raises(PermissionError, match="ConfirmFinalTest"):
        stage10.execute(tmp_path, tmp_path, tmp_path / "nonexistent.yaml", False)


def test_access_reservation_is_one_shot_even_after_failure(tmp_path):
    reservation = stage10._reserve(tmp_path / "ledger", "fixed-cohort")
    assert reservation.is_dir()
    with pytest.raises(RuntimeError, match="retry is forbidden"):
        stage10._reserve(tmp_path / "ledger", "fixed-cohort")


def test_cohort_binding_ignores_commit_run_name_and_other_audit_metadata():
    audit = {"endpoint_inventory": [{"sha256": "frozen"}], "shared_model_dependencies": {"base": "same"}, "run_name": "one", "git": "old"}
    modified = deepcopy(audit); modified.update(run_name="two", git="new")
    assert stage10._cohort_binding(audit) == stage10._cohort_binding(modified)
    modified["endpoint_inventory"][0]["sha256"] = "different"
    assert stage10._cohort_binding(audit) != stage10._cohort_binding(modified)


def test_finaltest_aggregate_reports_all_ten_methods_and_three_seeds():
    summary, decision = stage10.aggregate(_rows())
    assert len(summary) == 20
    assert all(row["seeds"] == 3 for row in summary)
    assert decision["status"] == "FINALTEST_LOCAL_COMPARISON_CONFIRMED"
    assert decision["historical_finaltest_exposure"] is True
    assert decision["retuning_or_retry_authorized"] is False


def test_finaltest_decision_includes_all_six_bota_cells():
    rows = _rows()
    next(row for row in rows if row["method"] == "BOTA" and row["seed"] == 42)["local_exact_masked_residual"] = 1.0
    _, decision = stage10.aggregate(rows)
    assert decision["status"] == "FINALTEST_LOCAL_COMPARISON_NOT_CONFIRMED"


def test_finaltest_aggregate_rejects_duplicates():
    rows = _rows(); rows[-1] = dict(rows[0])
    with pytest.raises(ValueError, match="duplicated"): stage10.aggregate(rows)


def test_csv_export_handles_heterogeneous_fields(tmp_path):
    path = tmp_path / "mixed.csv"
    stage10._write_csv(path, [{"a": 1}, {"a": 2, "support": [41, 43]}])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and rows[1]["support"] == "[41, 43]"


def test_artifact_path_rejects_workspace_escape(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        stage10._path(tmp_path, "../outside")
