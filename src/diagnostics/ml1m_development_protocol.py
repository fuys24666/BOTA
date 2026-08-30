from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from data_preprocess.load_prompt_ml1m import get_template
from src.diagnostics.t5_reconstructed_official import sha256_file
from src.diagnostics.t5_unified_development import load_existing_predictions

SCHEMA = "t5-e2urec-ml1m-development-protocol-v1"
SEED = 42
FORGET_RATIO = 0.2
TAIL_WINDOW = 100_000
TRAIN_ROWS = 60_000
VALIDATION_ROWS = 20_000
RAW_FILES = ("users.dat", "ratings.dat", "movies.dat")
PROCESSED_FILES = {
    "validation": "data/ml-1m/proc_data/data/valid/valid_10_simple.json",
    "train": "data/ml-1m/proc_data/data/train/train_10_simple.json",
    "forget": "data/ml-1m/proc_data/data/train/forget_0.2_user_10_simple.json",
    "retain": "data/ml-1m/proc_data/data/train/retain_0.2_user_10_simple.json",
}

AGE = {
    1: "under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "above 56",
}
JOB = {
    0: "other or not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}


def reject_processed_test_path(path: Path) -> Path:
    resolved = path.resolve()
    if any("test" in part.lower() for part in resolved.parts):
        raise ValueError(f"processed test paths are forbidden: {resolved}")
    return resolved


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sample_order_hash(indices: Iterable[int]) -> str:
    return hashlib.sha256(
        np.asarray(list(indices), dtype=np.int64).tobytes()
    ).hexdigest()


@dataclass
class RawLineage:
    processed_split: str
    processed_row_index: int
    authoritative_user_id: int
    raw_source_file: str
    raw_source_row: int
    movie_id: int
    rating: int
    label: int
    timestamp: int
    history_ids: list[int]
    history_ratings: list[int]


def load_raw_metadata(raw_dir: Path) -> tuple[dict[int, tuple[str, str, str, str]], dict[int, str]]:
    users: dict[int, tuple[str, str, str, str]] = {}
    for line in (raw_dir / "users.dat").read_text(encoding="utf-8").splitlines():
        user, gender, age, job, zipcode = line.split("::")
        users[int(user)] = (
            "male" if gender == "M" else "female",
            AGE[int(age)],
            JOB[int(job)],
            zipcode,
        )
    movies: dict[int, str] = {}
    for line in (raw_dir / "movies.dat").read_text(
        encoding="ISO-8859-1"
    ).splitlines():
        movie, title, _genres = line.split("::", 2)
        movies[int(movie)] = title
    return users, movies


def reconstruct_authoritative_rows(raw_dir: Path) -> tuple[list[RawLineage], list[RawLineage], dict[str, Any]]:
    """Replay only train+validation locations; never materialize the final test slice."""
    users, movies = load_raw_metadata(raw_dir)
    ratings: list[tuple[int, int, int, int, int]] = []
    for raw_row, line in enumerate(
        (raw_dir / "ratings.dat").read_text(encoding="utf-8").splitlines()
    ):
        user, movie, rating, timestamp = line.split("::")
        user_id, movie_id = int(user), int(movie)
        if user_id in users and movie_id in movies:
            ratings.append(
                (int(timestamp), user_id, movie_id, int(rating), raw_row)
            )
    frame = pd.DataFrame(
        ratings,
        columns=["timestamp", "User ID", "Movie ID", "rating", "raw_source_row"],
    )
    frame.sort_values(
        ["timestamp", "User ID", "Movie ID"], kind="stable", inplace=True
    )
    frame.reset_index(drop=True, inplace=True)
    per_user = frame.groupby("User ID").size()
    filtered_count = len(frame) - sum(min(5, int(count)) for count in per_user)
    train_begin = filtered_count - TAIL_WINDOW
    validation_begin = filtered_count - 40_000
    validation_end = filtered_count - 20_000
    if train_begin < 0:
        raise ValueError("raw filtered data is smaller than formal tail window")
    histories = {user_id: ([], []) for user_id in users}
    train: list[RawLineage] = []
    validation: list[RawLineage] = []
    filtered_index = 0
    for row in frame.itertuples(index=False):
        timestamp, user_id, movie_id, rating, raw_row = map(int, row)
        history_ids, history_ratings = histories[user_id]
        if len(history_ids) >= 5:
            if train_begin <= filtered_index < validation_begin:
                train.append(
                    RawLineage(
                        "train",
                        len(train),
                        user_id,
                        "ratings.dat",
                        raw_row,
                        movie_id,
                        rating,
                        1 if rating > 3 else 0,
                        timestamp,
                        history_ids.copy(),
                        history_ratings.copy(),
                    )
                )
            elif validation_begin <= filtered_index < validation_end:
                validation.append(
                    RawLineage(
                        "validation",
                        len(validation),
                        user_id,
                        "ratings.dat",
                        raw_row,
                        movie_id,
                        rating,
                        1 if rating > 3 else 0,
                        timestamp,
                        history_ids.copy(),
                        history_ratings.copy(),
                    )
                )
            filtered_index += 1
        history_ids.append(movie_id)
        history_ratings.append(rating)
    if filtered_index != filtered_count:
        raise RuntimeError("raw replay filtered count mismatch")
    if len(train) != TRAIN_ROWS or len(validation) != VALIDATION_ROWS:
        raise RuntimeError("formal train/validation row count mismatch")
    return train, validation, {
        "raw_rating_rows": len(frame),
        "filtered_rows": filtered_count,
        "tail_window": TAIL_WINDOW,
        "train_filtered_slice": [train_begin, validation_begin],
        "validation_filtered_slice": [validation_begin, validation_end],
        "unmaterialized_final_slice": [validation_end, filtered_count],
        "raw_unsplit_source_read": True,
        "processed_test_split_read": False,
        "test_metrics_computed": False,
        "test_predictions_loaded": False,
    }


def render_record(
    row: RawLineage,
    users: dict[int, tuple[str, str, str, str]],
    movies: dict[int, str],
    history_k: int = 10,
) -> dict[str, str]:
    if row.authoritative_user_id not in users:
        raise ValueError("authoritative user ID missing from raw users")
    gender, age, job, _zipcode = users[row.authoritative_user_id]
    history = []
    for movie_id, rating in zip(
        row.history_ids[-history_k:], row.history_ratings[-history_k:]
    ):
        suffix = " stars)" if rating > 1 else " star)"
        history.append(f"{movies[movie_id]} ({rating}{suffix}")
    prompt_fields = {
        "User ID": row.authoritative_user_id,
        "Movie ID": movies[row.movie_id],
        "history ID": history,
        "Gender": gender,
        "Age": age,
        "Job": job,
        "history rating": row.history_ratings[-history_k:],
    }
    return {
        "input": get_template(prompt_fields, "simple"),
        "output": "Yes." if row.label == 1 else "No.",
    }


def compare_exact(
    replay: list[dict[str, str]], existing: list[dict[str, str]], label: str
) -> dict[str, Any]:
    if len(replay) != len(existing):
        raise ValueError(f"{label} row count mismatch: {len(replay)} != {len(existing)}")
    first = None
    mismatch = 0
    for index, (left, right) in enumerate(zip(replay, existing)):
        if left != right:
            mismatch += 1
            if first is None:
                first = {
                    "row_index": index,
                    "input_equal": left.get("input") == right.get("input"),
                    "output_equal": left.get("output") == right.get("output"),
                    "replay_record_sha256": _sha_value(left),
                    "existing_record_sha256": _sha_value(right),
                }
    report = {
        "split": label,
        "rows": len(replay),
        "exact_record_matches": len(replay) - mismatch,
        "mismatch_count": mismatch,
        "first_mismatch": first,
        "input_output_max_difference": (
            None if mismatch == 0 else "see first_mismatch hashes/equality"
        ),
        "record_order_preserved": mismatch == 0,
    }
    if mismatch:
        raise ValueError(f"{label} replay mismatch: {report}")
    return report


def formal_forget_users(train: list[RawLineage]) -> tuple[list[int], set[int]]:
    unique: list[int] = []
    seen: set[int] = set()
    for row in train:
        if row.authoritative_user_id not in seen:
            seen.add(row.authoritative_user_id)
            unique.append(row.authoritative_user_id)
    shuffled = unique.copy()
    random.seed(SEED)
    random.shuffle(shuffled)
    selected = shuffled[: int(FORGET_RATIO * len(shuffled))]
    return selected, set(selected)


def validate_train_partition(
    replay_train: list[dict[str, str]],
    train_rows: list[RawLineage],
    forget_users: set[int],
    existing_forget: list[dict[str, str]],
    existing_retain: list[dict[str, str]],
) -> dict[str, Any]:
    forget_indices = [
        index
        for index, row in enumerate(train_rows)
        if row.authoritative_user_id in forget_users
    ]
    forget_index_set = set(forget_indices)
    retain_indices = [
        index for index in range(len(train_rows)) if index not in forget_index_set
    ]
    replay_forget = [replay_train[index] for index in forget_indices]
    replay_retain = [replay_train[index] for index in retain_indices]
    compare_exact(replay_forget, existing_forget, "forget_train")
    compare_exact(replay_retain, existing_retain, "retain_train")
    forget_record_users = {
        train_rows[index].authoritative_user_id for index in forget_indices
    }
    retain_record_users = {
        train_rows[index].authoritative_user_id for index in retain_indices
    }
    if not forget_record_users <= forget_users:
        raise ValueError("forget train includes non-forget user")
    if retain_record_users & forget_users:
        raise ValueError("retain train includes forget user")
    if set(forget_indices) & set(retain_indices) or sorted(
        forget_indices + retain_indices
    ) != list(range(len(train_rows))):
        raise ValueError("forget/retain train partition is not exact")
    return {
        "forget_rows": len(forget_indices),
        "retain_rows": len(retain_indices),
        "forget_indices": forget_indices,
        "retain_indices": retain_indices,
        "forget_record_user_count": len(forget_record_users),
        "retain_record_user_count": len(retain_record_users),
        "partition_union_complete": True,
        "partition_mutually_exclusive": True,
    }


def sidecar_record(row: RawLineage, processed: dict[str, str]) -> dict[str, Any]:
    history_source = {
        "user_id": row.authoritative_user_id,
        "history_ids": row.history_ids,
        "history_ratings": row.history_ratings,
    }
    return {
        "processed_split": row.processed_split,
        "processed_row_index": row.processed_row_index,
        "authoritative_user_id": row.authoritative_user_id,
        "raw_source_file": row.raw_source_file,
        "raw_source_row": row.raw_source_row,
        "movie_id": row.movie_id,
        "rating": row.rating,
        "label": row.label,
        "timestamp": row.timestamp,
        "input_sha256": _sha_text(processed["input"]),
        "output_sha256": _sha_text(processed["output"]),
        "combined_record_sha256": _sha_value(processed),
        "preprocessing_seed": SEED,
        "template_version": "ml1m-simple-K10-original",
        "source_user_history_hash": _sha_value(history_source),
    }


def emit_sidecar(path: Path | None, records: list[dict[str, Any]]) -> str | None:
    if path is None:
        return None
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
        for record in records
    )
    _atomic_text(path, text)
    return sha256_file(path)


def split_validation(
    validation_rows: list[RawLineage], forget_users: set[int]
) -> dict[str, Any]:
    forget = [
        row.processed_row_index
        for row in validation_rows
        if row.authoritative_user_id in forget_users
    ]
    retain = [
        row.processed_row_index
        for row in validation_rows
        if row.authoritative_user_id not in forget_users
    ]
    if set(forget) & set(retain) or sorted(forget + retain) != list(
        range(len(validation_rows))
    ):
        raise ValueError("development split is not mutually exclusive and complete")

    def stats(indices: list[int]) -> dict[str, Any]:
        rows = [validation_rows[index] for index in indices]
        positives = sum(row.label for row in rows)
        return {
            "samples": len(rows),
            "users": len({row.authoritative_user_id for row in rows}),
            "positive_labels": positives,
            "negative_labels": len(rows) - positives,
            "has_both_classes": 0 < positives < len(rows),
            "sample_order_hash": sample_order_hash(indices),
        }

    return {
        "forget_validation_indices": forget,
        "retain_validation_indices": retain,
        "overall_validation_indices": list(range(len(validation_rows))),
        "stats": {
            "forget_user_validation": stats(forget),
            "retain_user_validation": stats(retain),
            "overall_validation": stats(list(range(len(validation_rows)))),
        },
        "mutually_exclusive": True,
        "union_equals_overall": True,
    }


def _safe_correlation(function, left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = function(left, right).statistic
    return float(value) if np.isfinite(value) else None


def subgroup_metric(
    name: str,
    prediction: dict[str, Any],
    original: dict[str, Any],
    retrain: dict[str, Any],
    indices: list[int],
    split_name: str,
) -> dict[str, Any]:
    probability = np.asarray(prediction["probabilities"], dtype=float)[indices]
    original_probability = np.asarray(original["probabilities"], dtype=float)[indices]
    retrain_probability = np.asarray(retrain["probabilities"], dtype=float)[indices]
    gold = np.asarray(prediction["gold"], dtype=int)[indices]
    predicted = probability >= 0.5
    epsilon = 1e-12

    def relative(reference: np.ndarray) -> dict[str, float]:
        p = np.clip(probability, epsilon, 1 - epsilon)
        q = np.clip(reference, epsilon, 1 - epsilon)
        middle = (p + q) / 2
        jsd = 0.5 * (
            p * np.log(p / middle)
            + (1 - p) * np.log((1 - p) / (1 - middle))
            + q * np.log(q / middle)
            + (1 - q) * np.log((1 - q) / (1 - middle))
        )
        legacy = p * np.log(p / q) + q * np.log(q / p)
        return {
            "l2_rms": float(np.sqrt(np.mean((probability - reference) ** 2))),
            "standard_jsd": float(np.mean(jsd)),
            "legacy_symmetric_kl": float(np.mean(legacy)),
            "prediction_agreement": float(
                np.mean(predicted == (reference >= 0.5))
            ),
        }

    change = probability - original_probability
    target = retrain_probability - original_probability
    both_classes = len(np.unique(gold)) == 2
    return {
        "model": name,
        "split": split_name,
        "samples": len(indices),
        "auc": float(roc_auc_score(gold, probability)) if both_classes else None,
        "auc_status": "finite" if both_classes else "not_applicable_single_class",
        "accuracy": float(accuracy_score(gold, predicted)),
        "log_loss": float(log_loss(gold, probability, labels=[0, 1])),
        "probability_mean": float(probability.mean()),
        "probability_std": float(probability.std()),
        "positive_rate": float(predicted.mean()),
        "mean_confidence": float(
            np.maximum(probability, 1 - probability).mean()
        ),
        "relative_original": relative(original_probability),
        "relative_retrain": relative(retrain_probability),
        "retrain_direction": {
            "sign_agreement": float(np.mean(np.sign(change) == np.sign(target))),
            "pearson": _safe_correlation(pearsonr, change, target),
            "spearman": _safe_correlation(spearmanr, change, target),
        },
        "mean_absolute_change": float(np.abs(change).mean()),
        "test_accessed": False,
    }


def build_subgroup_metrics(
    project_root: Path, split: dict[str, Any]
) -> dict[str, Any]:
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only subgroup evaluation refuses initialized CUDA")
    predictions, sources, unavailable = load_existing_predictions(project_root)
    split_indices = {
        "overall_validation": split["overall_validation_indices"],
        "forget_user_validation": split["forget_validation_indices"],
        "retain_user_validation": split["retain_validation_indices"],
    }
    rows = []
    for split_name, indices in split_indices.items():
        for name, prediction in predictions.items():
            if name in {"j1_step812", "j2_step812", "j3_step812"}:
                continue
            rows.append(
                subgroup_metric(
                    name,
                    prediction,
                    predictions["original"],
                    predictions["retrain"],
                    indices,
                    split_name,
                )
            )
    by_key = {(row["split"], row["model"]): row for row in rows}
    summary: dict[str, Any] = {}
    for split_name in split_indices:
        step812 = by_key[(split_name, "step812")]
        for model in (
            "j0_step1200",
            "j1_step1200",
            "j2_step1200",
            "j3_step1200",
            "j4_step1200",
        ):
            row = by_key[(split_name, model)]
            summary[f"{split_name}:{model}"] = {
                "forget_or_general_improvement_vs_step812_l2_retrain": (
                    step812["relative_retrain"]["l2_rms"]
                    - row["relative_retrain"]["l2_rms"]
                ),
                "utility_auc_change_vs_original": (
                    None
                    if row["auc"] is None
                    else row["auc"] - by_key[(split_name, "original")]["auc"]
                ),
                "utility_log_loss_change_vs_original": (
                    row["log_loss"]
                    - by_key[(split_name, "original")]["log_loss"]
                ),
                "step812_direction_sign_retention": _direction_retention(
                    predictions[model],
                    predictions["step812"],
                    predictions["original"],
                    split_indices[split_name],
                ),
            }
        j0 = by_key[(split_name, "j0_step1200")]
        j2 = by_key[(split_name, "j2_step1200")]
        j3 = by_key[(split_name, "j3_step1200")]
        j4 = by_key[(split_name, "j4_step1200")]
        summary[f"{split_name}:contrasts"] = {
            "j2_vs_j0_retrain_l2_improvement": (
                j0["relative_retrain"]["l2_rms"]
                - j2["relative_retrain"]["l2_rms"]
            ),
            "j3_vs_j2_retrain_l2_improvement": (
                j2["relative_retrain"]["l2_rms"]
                - j3["relative_retrain"]["l2_rms"]
            ),
            "j3_vs_j2_auc_change": (
                None
                if j3["auc"] is None or j2["auc"] is None
                else j3["auc"] - j2["auc"]
            ),
            "j3_vs_j2_log_loss_change": j3["log_loss"] - j2["log_loss"],
            "j4_vs_j0": _model_contrast(j4, j0),
            "j4_vs_j2": _model_contrast(j4, j2),
            "j4_vs_j3": _model_contrast(j4, j3),
        }
    return {
        "schema": SCHEMA,
        "scope": "development_only_subgroups",
        "runtime_device": "cpu",
        "models_loaded": False,
        "checkpoint_selection_performed": False,
        "rows": rows,
        "contrasts": summary,
        "prediction_sources": sources,
        "unavailable_predictions": unavailable,
        "test_loader_built": False,
        "test_accessed": False,
    }


def _model_contrast(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, float | None]:
    return {
        "retrain_l2_improvement": (
            reference["relative_retrain"]["l2_rms"]
            - candidate["relative_retrain"]["l2_rms"]
        ),
        "auc_change": (
            None
            if candidate["auc"] is None or reference["auc"] is None
            else candidate["auc"] - reference["auc"]
        ),
        "accuracy_change": candidate["accuracy"] - reference["accuracy"],
        "log_loss_change": candidate["log_loss"] - reference["log_loss"],
    }


def _direction_retention(
    prediction: dict[str, Any],
    step812: dict[str, Any],
    original: dict[str, Any],
    indices: list[int],
) -> float:
    current = np.asarray(prediction["probabilities"], dtype=float)[indices]
    warmup = np.asarray(step812["probabilities"], dtype=float)[indices]
    baseline = np.asarray(original["probabilities"], dtype=float)[indices]
    return float(np.mean(np.sign(current - baseline) == np.sign(warmup - baseline)))


def _metrics_csv(rows: list[dict[str, Any]]) -> str:
    flattened = []
    for row in rows:
        value = {
            key: item
            for key, item in row.items()
            if key not in {"relative_original", "relative_retrain", "retrain_direction"}
        }
        for prefix in ("relative_original", "relative_retrain", "retrain_direction"):
            value.update({f"{prefix}_{key}": item for key, item in row[prefix].items()})
        flattened.append(value)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(flattened[0]))
    writer.writeheader()
    writer.writerows(flattened)
    return output.getvalue()


def _git_head(project_root: Path) -> str:
    return subprocess.check_output(
        ["C:\\Program Files\\Git\\cmd\\git.exe", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
    ).strip()


def generate_protocol(
    project_root: Path,
    output_root: Path,
    sidecar_path: Path | None,
) -> dict[str, Any]:
    raw_dir = project_root / "data" / "ml-1m" / "raw_data"
    paths = {
        key: reject_processed_test_path(project_root / relative)
        for key, relative in PROCESSED_FILES.items()
    }
    input_hashes = {
        f"raw/{name}": sha256_file(raw_dir / name) for name in RAW_FILES
    }
    input_hashes.update(
        {f"processed/{key}": sha256_file(path) for key, path in paths.items()}
    )
    script_paths = (
        project_root / "data_preprocess" / "data2json.py",
        project_root / "data_preprocess" / "load_prompt_ml1m.py",
        project_root / "data_preprocess" / "ml-1m.ipynb",
        project_root / "data_preprocess" / "split_ml-1m.ipynb",
    )
    input_hashes.update(
        {f"preprocessor/{path.name}": sha256_file(path) for path in script_paths}
    )
    train_rows, validation_rows, replay_source = reconstruct_authoritative_rows(
        raw_dir
    )
    users, movies = load_raw_metadata(raw_dir)
    replay_train = [render_record(row, users, movies) for row in train_rows]
    replay_validation = [
        render_record(row, users, movies) for row in validation_rows
    ]
    existing = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in paths.items()
    }
    train_audit = compare_exact(replay_train, existing["train"], "train")
    validation_audit = compare_exact(
        replay_validation, existing["validation"], "validation"
    )
    forget_user_list, forget_users = formal_forget_users(train_rows)
    partition = validate_train_partition(
        replay_train,
        train_rows,
        forget_users,
        existing["forget"],
        existing["retain"],
    )
    validation_sidecar = [
        sidecar_record(row, processed)
        for row, processed in zip(validation_rows, replay_validation)
    ]
    sidecar_sha = emit_sidecar(sidecar_path, validation_sidecar)
    split = split_validation(validation_rows, forget_users)
    output_root.mkdir(parents=True, exist_ok=True)
    forget_manifest = {
        "schema": SCHEMA,
        "user_ids": forget_user_list,
        "user_count": len(forget_user_list),
        "source_train_user_count": len(
            {row.authoritative_user_id for row in train_rows}
        ),
        "generation": "pandas unique first-occurrence order; random.seed(42); random.shuffle; first int(0.2*N)",
        "seed": SEED,
        "ratio": FORGET_RATIO,
        "source_hashes": input_hashes,
        "test_accessed": False,
    }
    atomic_json(output_root / "forget_user_manifest.json", forget_manifest)
    atomic_json(
        output_root / "forget_validation_indices.json",
        {
            "indices": split["forget_validation_indices"],
            "stats": split["stats"]["forget_user_validation"],
            "test_accessed": False,
        },
    )
    atomic_json(
        output_root / "retain_validation_indices.json",
        {
            "indices": split["retain_validation_indices"],
            "stats": split["stats"]["retain_user_validation"],
            "test_accessed": False,
        },
    )
    current_validation_bytes = paths["validation"].read_bytes()
    replay_validation_bytes = json.dumps(replay_validation).encode("utf-8")
    audit = {
        "schema": SCHEMA,
        "raw_lineage": replay_source,
        "train": train_audit,
        "validation": {
            **validation_audit,
            "replay_sha256": hashlib.sha256(replay_validation_bytes).hexdigest(),
            "current_validation_sha256": hashlib.sha256(
                current_validation_bytes
            ).hexdigest(),
            "serialized_bytes_exact": replay_validation_bytes
            == current_validation_bytes,
            "sidecar_sha256": sidecar_sha,
            "unique_authoritative_user_per_row": all(
                isinstance(row.authoritative_user_id, int) for row in validation_rows
            ),
        },
        "train_partition": {
            key: value
            for key, value in partition.items()
            if key not in {"forget_indices", "retain_indices"}
        },
        "input_hashes": input_hashes,
        "existing_data_modified": False,
        "processed_test_split_read": False,
        "test_metrics_computed": False,
        "test_predictions_loaded": False,
    }
    atomic_json(output_root / "reconstruction_audit.json", audit)
    subgroup = build_subgroup_metrics(project_root, split)
    atomic_json(output_root / "subgroup_development_metrics.json", subgroup)
    _atomic_text(
        output_root / "subgroup_development_metrics.csv",
        _metrics_csv(subgroup["rows"]),
    )
    output_hashes = {
        path.name: sha256_file(path)
        for path in (
            sidecar_path,
            output_root / "forget_user_manifest.json",
            output_root / "forget_validation_indices.json",
            output_root / "retain_validation_indices.json",
            output_root / "reconstruction_audit.json",
            output_root / "subgroup_development_metrics.json",
            output_root / "subgroup_development_metrics.csv",
        )
        if path is not None
    }
    manifest = {
        "schema": SCHEMA,
        "preprocessing_commit": _git_head(project_root),
        "command": (
            "python -m src.diagnostics.ml1m_development_protocol "
            "--emit-provenance-sidecar validation_user_sidecar.jsonl"
        ),
        "parameters": {
            "seed": SEED,
            "forget_ratio": FORGET_RATIO,
            "history_k": 10,
            "template": "simple",
        },
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "sidecar_rows": len(validation_sidecar),
        "sidecar_users": len(
            {row.authoritative_user_id for row in validation_rows}
        ),
        "split_stats": split["stats"],
        "forget_user_count": len(forget_users),
        "raw_unsplit_source_read": True,
        "processed_test_split_read": False,
        "test_metrics_computed": False,
        "test_predictions_loaded": False,
        "test_accessed": False,
    }
    atomic_json(output_root / "development_protocol_manifest.json", manifest)
    after_hashes = {
        f"processed/{key}": sha256_file(path) for key, path in paths.items()
    }
    if any(
        after_hashes[key] != input_hashes[key] for key in after_hashes
    ):
        raise RuntimeError("existing processed data changed during protocol generation")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct authoritative ML-1M development user lineage"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/t5_e2urec_development_protocol_v1"),
    )
    parser.add_argument("--emit-provenance-sidecar", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (
        args.output_root
        if args.output_root.is_absolute()
        else (root / args.output_root)
    ).resolve()
    sidecar = args.emit_provenance_sidecar
    if sidecar is not None and not sidecar.is_absolute():
        sidecar = output / sidecar
    result = generate_protocol(root, output, sidecar)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
