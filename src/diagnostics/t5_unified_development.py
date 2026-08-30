from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch

from src.diagnostics.t5_joint_ablation import BRANCHES, development_metrics
from src.diagnostics.t5_reconstructed_official import sha256_file

CANONICAL_STEPS = (813, 850, 900, 1000, 1200)
ALL_STEPS = (812,) + CANONICAL_STEPS
SCHEMA = "t5-unified-development-comparison-v1"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _prediction(value: dict[str, Any]) -> dict[str, Any]:
    return value["prediction"] if "prediction" in value else value


def validate_prediction(
    prediction: dict[str, Any],
    reference: dict[str, Any],
    label: str,
) -> None:
    if prediction.get("test_accessed") is not False:
        raise ValueError(f"{label}: test_accessed must be false")
    if prediction.get("samples") != 20_000:
        raise ValueError(f"{label}: expected 20,000 development samples")
    if (
        prediction.get("sample_ids") != reference.get("sample_ids")
        or prediction.get("gold") != reference.get("gold")
        or prediction.get("sample_order_hash") != reference.get("sample_order_hash")
    ):
        raise ValueError(f"{label}: sample IDs/gold/order mismatch")


def _source_record(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "checkpoint_or_protocol": protocol,
        "test_accessed": False,
    }


def load_existing_predictions(
    project_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    canonical = (
        project_root
        / "outputs"
        / "t5_e2urec_diagnostics_v1"
        / "full_runs"
        / "t5_reconstructed_official_seed42_v2"
    )
    cache_path = canonical / "frozen_validation_predictions.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache["metadata"].get("test_accessed") is not False:
        raise ValueError("frozen reference cache is not test-free")
    predictions: dict[str, dict[str, Any]] = {}
    sources: dict[str, Any] = {
        "frozen_references": _source_record(cache_path, cache["metadata"]),
    }
    unavailable: list[dict[str, Any]] = []
    for role in ("original", "augmented", "retrain"):
        predictions[role] = cache["predictions"][role]
    reference = predictions["original"]
    for role in ("original", "augmented", "retrain"):
        validate_prediction(predictions[role], reference, role)

    canonical_contract = json.loads((canonical / "contract.json").read_text(encoding="utf-8"))
    if canonical_contract.get("test_accessed") is not False:
        raise ValueError("canonical protocol is not test-free")
    sources["canonical_contract"] = _source_record(
        canonical / "contract.json", canonical_contract
    )
    for step in ALL_STEPS:
        path = canonical / f"development_step_{step:05d}.json"
        if not path.is_file():
            unavailable.append(
                {
                    "model": f"j0_step{step}",
                    "status": "unavailable_missing_published_prediction",
                    "expected_path": str(path.resolve()),
                    "model_forward_performed": False,
                    "test_accessed": False,
                }
            )
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        prediction = _prediction(value)
        validate_prediction(prediction, reference, f"canonical step{step}")
        key = "step812" if step == 812 else f"j0_step{step}"
        predictions[key] = prediction
        sources[key] = _source_record(
            path,
            {
                "protocol": canonical_contract.get("protocol"),
                "protocol_sha256": canonical_contract.get("protocol_sha256"),
                "step": step,
            },
        )

    v2_root = project_root / "outputs" / "t5_e2urec_joint_ablation_v2" / "full_runs"
    for branch in (
        "j1_supervised_only_remember",
        "j2_kl_only_remember",
        "j3_forget_only_control",
    ):
        branch_root = v2_root / branch
        runs = [
            path
            for path in branch_root.iterdir()
            if path.is_dir()
            and json.loads((path / "run_state.json").read_text(encoding="utf-8")).get(
                "status"
            )
            == "COMPLETED"
        ]
        if len(runs) != 1:
            raise ValueError(f"{branch}: expected exactly one completed optimized-v2 run")
        run = runs[0]
        contract = json.loads((run / "contract.json").read_text(encoding="utf-8"))
        if (
            contract.get("branch") != branch
            or contract.get("formula") != BRANCHES[branch].formula
            or contract.get("test_used") is not False
        ):
            raise ValueError(f"{branch}: invalid contract")
        sources[f"{branch}_contract"] = _source_record(run / "contract.json", contract)
        short = branch.split("_", 1)[0]
        for step in CANONICAL_STEPS:
            path = run / f"development_step_{step:05d}.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            prediction = _prediction(value)
            validate_prediction(prediction, reference, f"{short} step{step}")
            key = f"{short}_step{step}"
            predictions[key] = prediction
            sources[key] = _source_record(
                path,
                {
                    "branch": branch,
                    "formula": contract["formula"],
                    "parent_step": contract["parent_step"],
                    "step": step,
                },
            )

    j4_root = (
        project_root
        / "outputs"
        / "t5_e2urec_joint_ablation_j4_v1"
        / "full_runs"
        / "j4_gradient_calibrated_supervised_remember"
        / "j4_gc_seed42_v2"
    )
    j4_state = json.loads((j4_root / "run_state.json").read_text(encoding="utf-8"))
    j4_contract = json.loads((j4_root / "contract.json").read_text(encoding="utf-8"))
    if (
        j4_state.get("status") != "COMPLETED"
        or j4_state.get("test_accessed") is not False
        or j4_contract.get("branch")
        != "j4_gradient_calibrated_supervised_remember"
        or j4_contract.get("lambda_sup") != 0.06251054027113696
        or j4_contract.get("test_used") is not False
    ):
        raise ValueError("J4 completed development contract is invalid")
    sources["j4_contract"] = _source_record(j4_root / "contract.json", j4_contract)
    for step in CANONICAL_STEPS:
        path = j4_root / f"development_step_{step:05d}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        prediction = _prediction(value)
        validate_prediction(prediction, reference, f"j4 step{step}")
        key = f"j4_step{step}"
        predictions[key] = prediction
        sources[key] = _source_record(
            path,
            {
                "branch": j4_contract["branch"],
                "formula": j4_contract["formula"],
                "lambda_sup": j4_contract["lambda_sup"],
                "parent_step": j4_contract["parent_step"],
                "step": step,
            },
        )
    return predictions, sources, unavailable


def metric_row(
    name: str,
    prediction: dict[str, Any],
    original: dict[str, Any],
    retrain: dict[str, Any],
) -> dict[str, Any]:
    value = development_metrics(prediction, original, retrain)
    return {
        "model": name,
        "auc": value["auc"],
        "accuracy": value["accuracy"],
        "log_loss": value["log_loss"],
        "probability_mean": value["probability"]["mean"],
        "probability_std": value["probability"]["std"],
        "positive_rate": value["positive_rate"],
        "mean_confidence": value["mean_confidence"],
        "l2_to_original": value["relative_original"]["l2_rms"],
        "l2_to_retrain": value["relative_retrain"]["l2_rms"],
        "jsd_to_original": value["relative_original"]["standard_jsd"],
        "jsd_to_retrain": value["relative_retrain"]["standard_jsd"],
        "legacy_symmetric_kl_to_original": value["relative_original"][
            "legacy_symmetric_kl"
        ],
        "legacy_symmetric_kl_to_retrain": value["relative_retrain"][
            "legacy_symmetric_kl"
        ],
        "agreement_original": value["relative_original"]["prediction_agreement"],
        "agreement_retrain": value["relative_retrain"]["prediction_agreement"],
        "retrain_direction_sign_agreement": value["retrain_direction"]["sign_agreement"],
        "retrain_direction_pearson": value["retrain_direction"]["pearson"],
        "retrain_direction_spearman": value["retrain_direction"]["spearman"],
        "mean_absolute_change": value["mean_absolute_change_from_original"],
        "samples": value["samples"],
        "sample_order_hash": value["sample_order_hash"],
        "test_accessed": False,
    }


def add_differences(rows: list[dict[str, Any]]) -> None:
    by_name = {row["model"]: row for row in rows}
    original = by_name["original"]
    step812 = by_name["step812"]
    for row in rows:
        if row.get("status") != "available":
            row["delta"] = {
                "utility_damage_vs_original": {
                    "auc_drop": None,
                    "accuracy_drop": None,
                    "log_loss_increase": None,
                },
                "retrain_proximity_improvement_vs_step812": None,
                "retrain_proximity_improvement_vs_j0": None,
                "retrain_proximity_improvement_vs_j2": None,
                "retrain_proximity_improvement_vs_j3": None,
            }
            continue
        row["delta"] = {
            "utility_damage_vs_original": {
                "auc_drop": original["auc"] - row["auc"],
                "accuracy_drop": original["accuracy"] - row["accuracy"],
                "log_loss_increase": row["log_loss"] - original["log_loss"],
            },
            "retrain_proximity_improvement_vs_step812": (
                step812["l2_to_retrain"] - row["l2_to_retrain"]
            ),
            "retrain_proximity_improvement_vs_j0": None,
            "retrain_proximity_improvement_vs_j2": None,
            "retrain_proximity_improvement_vs_j3": None,
        }
        parts = row["model"].rsplit("_step", 1)
        if len(parts) == 2 and parts[1].isdigit():
            step = parts[1]
            j0 = by_name.get(f"j0_step{step}")
            j2 = by_name.get(f"j2_step{step}")
            j3 = by_name.get(f"j3_step{step}")
            if j0 is not None and j0.get("status") == "available":
                row["delta"]["retrain_proximity_improvement_vs_j0"] = (
                    j0["l2_to_retrain"] - row["l2_to_retrain"]
                )
            if j2 is not None and j2.get("status") == "available":
                row["delta"]["retrain_proximity_improvement_vs_j2"] = (
                    j2["l2_to_retrain"] - row["l2_to_retrain"]
                )
            if j3 is not None and j3.get("status") == "available":
                row["delta"]["retrain_proximity_improvement_vs_j3"] = (
                    j3["l2_to_retrain"] - row["l2_to_retrain"]
                )


def _csv(rows: list[dict[str, Any]]) -> str:
    flattened = []
    for row in rows:
        value = {key: item for key, item in row.items() if key != "delta"}
        value.update(row["delta"]["utility_damage_vs_original"])
        value.update(
            {
                key: item
                for key, item in row["delta"].items()
                if key != "utility_damage_vs_original"
            }
        )
        flattened.append(value)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(flattened[0]))
    writer.writeheader()
    writer.writerows(flattened)
    return output.getvalue()


def build_unified_comparison(project_root: Path) -> dict[str, Any]:
    predictions, sources, unavailable = load_existing_predictions(project_root)
    rows = [
        {
            **metric_row(
                name, prediction, predictions["original"], predictions["retrain"]
            ),
            "status": "available",
        }
        for name, prediction in predictions.items()
        if name not in {"j1_step812", "j2_step812", "j3_step812"}
    ]
    for missing in unavailable:
        rows.append(
            {
                "model": missing["model"],
                "status": missing["status"],
                **{
                    key: None
                    for key in (
                        "auc",
                        "accuracy",
                        "log_loss",
                        "probability_mean",
                        "probability_std",
                        "positive_rate",
                        "mean_confidence",
                        "l2_to_original",
                        "l2_to_retrain",
                        "jsd_to_original",
                        "jsd_to_retrain",
                        "legacy_symmetric_kl_to_original",
                        "legacy_symmetric_kl_to_retrain",
                        "agreement_original",
                        "agreement_retrain",
                        "retrain_direction_sign_agreement",
                        "retrain_direction_pearson",
                        "retrain_direction_spearman",
                        "mean_absolute_change",
                        "samples",
                        "sample_order_hash",
                    )
                },
                "test_accessed": False,
            }
        )
    order = {"original": 0, "augmented": 1, "retrain": 2, "step812": 3}
    rows.sort(
        key=lambda row: (
            order.get(row["model"], 4),
            int(row["model"].rsplit("step", 1)[-1])
            if "step" in row["model"]
            else 0,
            row["model"],
        )
    )
    add_differences(rows)
    payload = {
        "schema": SCHEMA,
        "scope": "development_only",
        "checkpoint_selection_performed": False,
        "model_forward_performed": False,
        "rows": rows,
        "sources": sources,
        "unavailable": unavailable,
        "sample_order_hash": predictions["original"]["sample_order_hash"],
        "samples": 20_000,
        "test_loader_built": False,
        "test_accessed": False,
    }
    output = (
        project_root
        / "outputs"
        / "t5_e2urec_joint_ablation_v2"
        / "comparisons"
    )
    _atomic_text(
        output / "unified_development.json",
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    _atomic_text(output / "unified_development.csv", _csv(rows))
    return payload
