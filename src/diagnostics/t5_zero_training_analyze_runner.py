from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil

from src.diagnostics.t5_reconstructed_official import sha256_file
from src.diagnostics.t5_zero_training_analysis import (
    ANALYSIS_SCHEMA,
    analyze_full_caches,
)
from src.diagnostics.t5_zero_training_audit import (
    CACHE_SCHEMA,
    EXPECTED_COUNTS,
    OUTPUT_NAME,
    SCHEMA,
    SPLITS,
    atomic_json,
    atomic_text,
    canonical_hash,
)
from src.diagnostics.t5_zero_training_decision import (
    DECISION_SCHEMA,
    PRIMARY_MIA_ATTACK,
    build_pareto_and_decision,
)


RUNNER_SCHEMA = "t5-e2urec-zero-training-analyze-runner-v1"
LOCK_SCHEMA = "t5-e2urec-zero-training-analysis-lock-v1"
FINAL_MANIFEST_SCHEMA = "t5-e2urec-zero-training-analysis-run-v1"
RUNNER_MODULE = "src.diagnostics.t5_zero_training_analyze_runner"
LEGACY_PARTIAL_FILES = ("metrics_and_mia.json", "metrics.csv", "manifest.json")


class AnalyzeLockError(RuntimeError):
    pass


class AnalyzeAlreadyRunning(AnalyzeLockError):
    pass


class AnalyzeRecoveryRequired(AnalyzeLockError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_run_name(run_name: str) -> str:
    if (
        not run_name
        or Path(run_name).name != run_name
        or "/" in run_name
        or "\\" in run_name
        or run_name in {".", ".."}
    ):
        raise ValueError("RunName must be one safe path component")
    return run_name


def _normalized_path(value: str | Path) -> str:
    return os.path.normcase(os.path.realpath(str(value)))


def _git_commit(project_root: Path) -> str:
    candidates = [
        shutil.which("git"),
        r"C:\Program Files\Git\cmd\git.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            completed = subprocess.run(
                [candidate, "rev-parse", "HEAD"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
    raise FileNotFoundError("git executable not found")


def _process_record() -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    created = process.create_time()
    return {
        "pid": process.pid,
        "process_creation_time_epoch": created,
        "process_creation_time_utc": datetime.fromtimestamp(
            created, timezone.utc
        ).isoformat(),
        "python_executable": str(Path(sys.executable).resolve()),
        "command_line": subprocess.list2cmdline([sys.executable, *sys.argv]),
    }


def inspect_lock_process(record: dict[str, Any]) -> dict[str, Any]:
    pid = int(record.get("pid", -1))
    result = {
        "pid_exists": False,
        "creation_time_matches": False,
        "command_matches": False,
        "run_name_matches": False,
        "python_matches": False,
        "active_analyze": False,
        "possible_pid_reuse": False,
    }
    try:
        process = psutil.Process(pid)
        actual_create = process.create_time()
        command = process.cmdline()
        executable = process.exe()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return result
    result["pid_exists"] = True
    expected_create = float(record.get("process_creation_time_epoch", -1.0))
    result["creation_time_matches"] = abs(actual_create - expected_create) < 0.01
    result["command_matches"] = any(RUNNER_MODULE in item for item in command)
    result["run_name_matches"] = str(record.get("run_name")) in command
    result["python_matches"] = _normalized_path(executable) == _normalized_path(
        str(record.get("python_executable", ""))
    )
    result["active_analyze"] = all(
        result[key]
        for key in (
            "pid_exists",
            "creation_time_matches",
            "command_matches",
            "run_name_matches",
            "python_matches",
        )
    )
    result["possible_pid_reuse"] = result["pid_exists"] and not result[
        "active_analyze"
    ]
    return result


def active_analyze_processes(run_name: str) -> list[dict[str, Any]]:
    modules = (
        RUNNER_MODULE,
        "src.diagnostics.t5_zero_training_analysis",
        "src.diagnostics.t5_zero_training_decision",
    )
    found = []
    for process in psutil.process_iter(
        ["pid", "create_time", "exe", "cmdline"]
    ):
        try:
            info = process.info
            command = [str(item) for item in (info.get("cmdline") or [])]
            if (
                int(info["pid"]) == os.getpid()
                or run_name not in command
                or not any(
                    any(module in item for item in command) for module in modules
                )
            ):
                continue
            created = float(info["create_time"])
            found.append(
                {
                    "pid": int(info["pid"]),
                    "process_creation_time_utc": datetime.fromtimestamp(
                        created, timezone.utc
                    ).isoformat(),
                    "python_executable": info.get("exe"),
                    "command_line": subprocess.list2cmdline(command),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return found


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def full_snapshot(
    project_root: Path, config_path: Path, run_name: str
) -> dict[str, Any]:
    full_run = project_root / "outputs" / OUTPUT_NAME / "full_runs" / run_name
    contract_path = full_run / "contract.json"
    state_path = full_run / "run_state.json"
    if not contract_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("Analyze requires Full contract and run_state")
    contract, state = _read_json(contract_path), _read_json(state_path)
    if (
        state.get("status") != "INFERENCE_COMPLETED"
        or state.get("test_accessed") is not False
        or state.get("optimizer_steps_executed") != 0
        or contract.get("schema") != SCHEMA
        or contract.get("cache_schema") != CACHE_SCHEMA
        or contract.get("run_name") != run_name
        or contract.get("test_accessed") is not False
    ):
        raise ValueError("Analyze requires a complete test-free Full contract")
    config_sha = sha256_file(config_path)
    if contract.get("config_sha256") != config_sha:
        raise ValueError("Full contract config SHA256 mismatch")
    models = list(state.get("models", []))
    if not models or "original" not in models or "retrain" not in models:
        raise ValueError("Full run model catalog is incomplete")
    inventory: list[dict[str, Any]] = []
    declared_content: list[dict[str, str]] = []
    for model in models:
        for split in SPLITS:
            data = full_run / "caches" / model / f"{split}.jsonl"
            manifest_path = full_run / "caches" / model / f"{split}.manifest.json"
            if not data.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"missing published cache {model}:{split}")
            manifest = _read_json(manifest_path)
            if (
                manifest.get("schema") != CACHE_SCHEMA
                or manifest.get("published") is not True
                or manifest.get("test_accessed") is not False
                or manifest.get("rows") != EXPECTED_COUNTS[split]
                or not isinstance(manifest.get("data_sha256"), str)
            ):
                raise ValueError(f"invalid cache manifest {model}:{split}")
            inventory.append(
                {
                    "model": model,
                    "split": split,
                    "manifest_sha256": sha256_file(manifest_path),
                    "declared_data_sha256": manifest["data_sha256"],
                    "data_size": data.stat().st_size,
                }
            )
            declared_content.append(
                {
                    "model": model,
                    "split": split,
                    "data_sha256": manifest["data_sha256"],
                }
            )
    return {
        "full_run": str(full_run.resolve()),
        "models": models,
        "bootstrap_seed": int(contract["bootstrap_seed"]),
        "bootstrap_resamples": int(contract["bootstrap_resamples"]),
        "primary_mia_score": contract["mia_primary_score"],
        "config_sha256": config_sha,
        "full_contract_sha256": sha256_file(contract_path),
        "full_run_state_sha256": sha256_file(state_path),
        "full_cache_inventory_sha256": canonical_hash(inventory),
        "full_cache_content_sha256": canonical_hash(declared_content),
        "analysis_schema": ANALYSIS_SCHEMA,
        "cache_schema": CACHE_SCHEMA,
        "test_accessed": False,
    }


def verify_full_cache_content(snapshot: dict[str, Any]) -> str:
    full_run = Path(snapshot["full_run"])
    actual = []
    for model in snapshot["models"]:
        for split in SPLITS:
            data = full_run / "caches" / model / f"{split}.jsonl"
            manifest = _read_json(
                full_run / "caches" / model / f"{split}.manifest.json"
            )
            actual_sha = sha256_file(data)
            if actual_sha != manifest.get("data_sha256"):
                raise AnalyzeRecoveryRequired(
                    f"cannot recover Analyze: cache content changed for "
                    f"{model}:{split}"
                )
            actual.append(
                {"model": model, "split": split, "data_sha256": actual_sha}
            )
    return canonical_hash(actual)


def _directory_sha256(path: Path) -> str:
    values = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            values.append((item.relative_to(path).as_posix(), sha256_file(item)))
    return canonical_hash(values)


def _control_root(project_root: Path) -> Path:
    return project_root / "outputs" / OUTPUT_NAME / "analysis_control"


def _final_output(project_root: Path, run_name: str) -> Path:
    return project_root / "outputs" / OUTPUT_NAME / "analysis_runs" / run_name


def validate_legacy_partial(path: Path, run_name: str) -> dict[str, Any]:
    if not path.is_dir():
        raise AnalyzeRecoveryRequired("legacy partial Analyze directory is missing")
    forbidden = (
        "pareto_and_decision.json",
        "pareto.csv",
        "report.md",
        "analysis_run_manifest.json",
        "ANALYSIS_COMPLETED",
    )
    if any((path / name).exists() for name in forbidden):
        raise AnalyzeRecoveryRequired(
            "legacy partial recovery requires metrics/MIA-only output"
        )
    for name in LEGACY_PARTIAL_FILES:
        if not (path / name).is_file():
            raise AnalyzeRecoveryRequired(
                f"legacy partial Analyze artifact is missing: {name}"
            )
    manifest = _read_json(path / "manifest.json")
    analysis = _read_json(path / "metrics_and_mia.json")
    metrics_sha = sha256_file(path / "metrics_and_mia.json")
    csv_sha = sha256_file(path / "metrics.csv")
    if (
        manifest.get("schema") != ANALYSIS_SCHEMA
        or manifest.get("test_accessed") is not False
        or manifest.get("metrics_and_mia_sha256") != metrics_sha
        or manifest.get("metrics_csv_sha256") != csv_sha
        or analysis.get("schema") != ANALYSIS_SCHEMA
        or analysis.get("run_name") != run_name
        or analysis.get("test_accessed") is not False
        or analysis.get("optimizer_steps_executed") != 0
        or analysis.get("model_loaded") is not False
    ):
        raise AnalyzeRecoveryRequired(
            "legacy partial Analyze safety fields or SHA256 are invalid"
        )
    return {
        "schema": ANALYSIS_SCHEMA,
        "run_name": run_name,
        "metrics_and_mia_sha256": metrics_sha,
        "metrics_csv_sha256": csv_sha,
        "manifest_sha256": sha256_file(path / "manifest.json"),
        "partial_directory_sha256": _directory_sha256(path),
        "test_accessed": False,
    }


def _archive_lock(
    control: Path, lock_dir: Path, *, reason: str
) -> tuple[Path, str]:
    parent_sha = _directory_sha256(lock_dir)
    history = control / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    reason_code = (
        "fail"
        if "failed" in reason
        else "stale"
        if "stale" in reason
        else "done"
    )
    destination = history / f"h-{stamp}-{reason_code}-{uuid.uuid4().hex[:8]}"
    os.replace(lock_dir, destination)
    return destination, parent_sha


def _existing_lock_message(record: dict[str, Any]) -> str:
    return (
        f"Analyze already running for RunName {record.get('run_name', 'unknown')}; "
        f"pid={record.get('pid', 'unknown')} "
        f"started={record.get('utc_started_time', 'unknown')} "
        f"command={record.get('command_line', 'metadata-pending')}"
    )


def acquire_analysis_lock(
    project_root: Path,
    run_name: str,
    snapshot: dict[str, Any],
    *,
    recover_failed: bool = False,
    recover_stale: bool = False,
    allow_existing_partial: bool = False,
    execution_mode: str = "full_analysis",
) -> tuple[Path, dict[str, Any]]:
    run_name = _safe_run_name(run_name)
    final_output = _final_output(project_root, run_name)
    if final_output.exists() and not allow_existing_partial:
        raise AnalyzeRecoveryRequired(
            f"Analyze output already exists for RunName {run_name}; refusing "
            "overwrite (verified legacy partials require explicit recovery)"
        )
    control = _control_root(project_root)
    control.mkdir(parents=True, exist_ok=True)
    lock_dir = control / f"{run_name}.lock"
    recovery: dict[str, Any] | None = None
    if lock_dir.exists():
        record_path = lock_dir / "record.json"
        if not record_path.is_file():
            if recover_stale:
                raise AnalyzeRecoveryRequired(
                    f"stale Analyze lock has incomplete metadata for RunName "
                    f"{run_name}; explicit recovery cannot bypass unverifiable "
                    "metadata"
                )
            raise AnalyzeAlreadyRunning(
                f"Analyze already running for RunName {run_name}; "
                "pid=metadata-pending started=metadata-pending "
                "command=metadata-pending"
            )
        old = _read_json(record_path)
        process_state = inspect_lock_process(old)
        if process_state["active_analyze"]:
            raise AnalyzeAlreadyRunning(_existing_lock_message(old))
        status = old.get("status")
        if status == "FAILED" and not recover_failed:
            raise AnalyzeRecoveryRequired(
                f"FAILED Analyze record exists for RunName {run_name}; "
                "use explicit failed-analysis recovery"
            )
        if status == "RUNNING" and not recover_stale:
            raise AnalyzeRecoveryRequired(
                f"stale RUNNING Analyze lock exists for RunName {run_name}; "
                "use explicit stale-lock recovery"
            )
        if status not in {"FAILED", "RUNNING"}:
            raise AnalyzeRecoveryRequired(
                f"Analyze lock status {status!r} cannot be recovered"
            )
        expected_switch = recover_failed if status == "FAILED" else recover_stale
        if not expected_switch:
            raise AnalyzeRecoveryRequired("matching explicit recovery switch required")
        for key in (
            "config_sha256",
            "full_contract_sha256",
            "full_run_state_sha256",
            "full_cache_inventory_sha256",
            "full_cache_content_sha256",
            "analysis_schema",
            "cache_schema",
        ):
            if old.get(key) != snapshot.get(key):
                raise AnalyzeRecoveryRequired(
                    f"cannot recover Analyze: {key} changed"
                )
        if verify_full_cache_content(snapshot) != snapshot[
            "full_cache_content_sha256"
        ]:
            raise AnalyzeRecoveryRequired(
                "cannot recover Analyze: Full cache content fingerprint changed"
            )
        archived, parent_sha = _archive_lock(
            control, lock_dir, reason=f"recovered_{status.lower()}"
        )
        recovery = {
            "reason": f"explicit_recovery_from_{status.lower()}",
            "parent_lock_sha256": parent_sha,
            "archived_lock": str(archived.resolve()),
            "prior_process_state": process_state,
            "parent_execution_mode": old.get("execution_mode", "full_analysis"),
        }
        if old.get("legacy_partial") is not None:
            recovery["parent_legacy_partial"] = old["legacy_partial"]
    try:
        lock_dir.mkdir()
    except FileExistsError:
        record_path = lock_dir / "record.json"
        record = _read_json(record_path) if record_path.is_file() else {
            "run_name": run_name
        }
        raise AnalyzeAlreadyRunning(_existing_lock_message(record)) from None
    process = _process_record()
    effective_execution_mode = (
        recovery.get("parent_execution_mode")
        if recovery is not None
        else execution_mode
    )
    record = {
        "schema": LOCK_SCHEMA,
        "runner_schema": RUNNER_SCHEMA,
        "run_name": run_name,
        **process,
        "utc_started_time": _utc_now(),
        "hostname": socket.gethostname(),
        "git_commit": _git_commit(project_root),
        **{
            key: snapshot[key]
            for key in (
                "config_sha256",
                "full_contract_sha256",
                "full_run_state_sha256",
                "full_cache_inventory_sha256",
                "full_cache_content_sha256",
                "analysis_schema",
                "cache_schema",
                "bootstrap_seed",
                "bootstrap_resamples",
                "primary_mia_score",
            )
        },
        "status": "RUNNING",
        "execution_mode": effective_execution_mode,
        "stage": "lock_acquired",
        "work_directory": str((lock_dir / "work").resolve()),
        "recovery": recovery,
        "optimizer_constructed": False,
        "optimizer_steps_executed": 0,
        "test_loader_built": False,
        "test_accessed": False,
    }
    atomic_json(lock_dir / "record.json", record)
    return lock_dir, record


class SparseProgress:
    def __init__(self, run_name: str, total: int, bootstrap: int) -> None:
        self.run_name = run_name
        self.total = total
        self.bootstrap = bootstrap
        self.started = time.monotonic()
        self.model_started: dict[str, float] = {}

    @staticmethod
    def _memory() -> tuple[float, float]:
        process = psutil.Process(os.getpid())
        info = process.memory_info()
        rss = info.rss / (1024**3)
        private_bytes = getattr(info, "private", None)
        if private_bytes is None:
            private_bytes = getattr(process.memory_full_info(), "uss", info.rss)
        return rss, private_bytes / (1024**3)

    def emit(self, value: str) -> None:
        print(value, flush=True)

    def start(self) -> None:
        self.emit(
            f"[analysis:start] run={self.run_name} models={self.total} "
            f"bootstrap={self.bootstrap}"
        )

    def lock_acquired(self, pid: int) -> None:
        self.emit(f"[analysis:lock:acquired] pid={pid}")

    def model(self, event: str, current: int, total: int, model: str) -> None:
        if event == "model_start":
            self.model_started[model] = time.monotonic()
            self.emit(
                f"[analysis:model:start] current={current}/{total} model={model} "
                "phase=metrics_mia"
            )
            return
        elapsed = time.monotonic() - self.started
        average = elapsed / max(current, 1)
        eta = average * max(total - current, 0)
        rss, private = self._memory()
        self.emit(
            f"[analysis:model:end] current={current}/{total} model={model} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s rss={rss:.2f}GB "
            f"private={private:.2f}GB"
        )


def _verify_staged_output(stage: Path) -> dict[str, str]:
    required = (
        "metrics_and_mia.json",
        "metrics.csv",
        "manifest.json",
        "pareto_and_decision.json",
        "pareto.csv",
        "report.md",
    )
    for name in required:
        if not (stage / name).is_file():
            raise ValueError(f"missing staged Analyze artifact {name}")
    analysis = _read_json(stage / "metrics_and_mia.json")
    decision = _read_json(stage / "pareto_and_decision.json")
    if (
        analysis.get("schema") != ANALYSIS_SCHEMA
        or decision.get("schema") != DECISION_SCHEMA
        or analysis.get("test_accessed") is not False
        or decision.get("test_accessed") is not False
        or analysis.get("optimizer_steps_executed") != 0
        or decision.get("optimizer_steps_executed") != 0
        or analysis.get("model_loaded") is not False
        or decision.get("model_loaded") is not False
    ):
        raise ValueError("invalid staged Analyze safety/provenance fields")
    return {name: sha256_file(stage / name) for name in required}


def run_analyze(
    project_root: Path,
    config_path: Path,
    run_name: str,
    *,
    recover_failed: bool = False,
    recover_stale: bool = False,
    recover_legacy_partial: bool = False,
    snapshot_function: Callable[[Path, Path, str], dict[str, Any]] = full_snapshot,
    analysis_function: Callable[..., dict[str, Any]] = analyze_full_caches,
    decision_function: Callable[..., dict[str, Any]] = build_pareto_and_decision,
) -> dict[str, Any]:
    project_root, config_path = project_root.resolve(), config_path.resolve()
    run_name = _safe_run_name(run_name)
    snapshot = snapshot_function(project_root, config_path, run_name)
    progress = SparseProgress(
        run_name, len(snapshot["models"]), snapshot["bootstrap_resamples"]
    )
    progress.start()
    legacy_processes = active_analyze_processes(run_name)
    if legacy_processes:
        active = legacy_processes[0]
        raise AnalyzeAlreadyRunning(
            f"Analyze already running for RunName {run_name}; "
            f"pid={active['pid']} "
            f"started={active['process_creation_time_utc']} "
            f"command={active['command_line']}"
        )
    lock_dir, record = acquire_analysis_lock(
        project_root,
        run_name,
        snapshot,
        recover_failed=recover_failed,
        recover_stale=recover_stale,
        allow_existing_partial=recover_legacy_partial,
        execution_mode=(
            "legacy_partial_decision_only"
            if recover_legacy_partial
            else "full_analysis"
        ),
    )
    progress.lock_acquired(record["pid"])
    stage = lock_dir / "work"
    try:
        stage.mkdir()
        legacy_provenance: dict[str, Any] | None = None
        if record["execution_mode"] == "legacy_partial_decision_only":
            final = _final_output(project_root, run_name)
            legacy_archive = lock_dir / "legacy_partial"
            if final.exists():
                legacy_provenance = validate_legacy_partial(final, run_name)
                os.replace(final, legacy_archive)
                archived_provenance = validate_legacy_partial(
                    legacy_archive, run_name
                )
                if archived_provenance != legacy_provenance:
                    raise ValueError("legacy partial changed during atomic archival")
            else:
                recovery = record.get("recovery") or {}
                if not recovery.get("archived_lock"):
                    raise AnalyzeRecoveryRequired(
                        "legacy partial archive is unavailable for recovery"
                    )
                parent = Path(str(recovery["archived_lock"]))
                candidate = parent / "legacy_partial"
                legacy_provenance = validate_legacy_partial(candidate, run_name)
                legacy_archive = candidate
            for name in LEGACY_PARTIAL_FILES:
                shutil.copy2(legacy_archive / name, stage / name)
            reused = validate_legacy_partial(stage, run_name)
            if reused != legacy_provenance:
                raise ValueError("reused legacy metrics/MIA SHA256 mismatch")
            record.update(
                {
                    "stage": "legacy_metrics_mia_reused",
                    "legacy_partial": {
                        **legacy_provenance,
                        "archive_path": str(legacy_archive.resolve()),
                        "archive_policy": (
                            "preserved under the completed or recovered control "
                            "history record"
                        ),
                        "metrics_mia_recomputed": False,
                    },
                }
            )
            atomic_json(lock_dir / "record.json", record)
            progress.emit(
                "[analysis:legacy-partial:reused] "
                f"metrics_sha256={legacy_provenance['metrics_and_mia_sha256']} "
                "metrics_mia_recomputed=false"
            )
        else:
            record["stage"] = "metrics_mia"
            atomic_json(lock_dir / "record.json", record)
            analysis_function(
                project_root,
                run_name,
                resamples=snapshot["bootstrap_resamples"],
                output_root=stage,
                progress=progress.model,
            )
        record["stage"] = "pareto_decision"
        atomic_json(lock_dir / "record.json", record)
        phase_started = time.monotonic()
        progress.emit("[analysis:phase:start] phase=pareto_decision")
        decision_function(
            project_root,
            run_name,
            resamples=snapshot["bootstrap_resamples"],
            analysis_root=stage,
        )
        progress.emit(
            "[analysis:phase:end] phase=pareto_decision "
            f"elapsed={time.monotonic() - phase_started:.1f}s"
        )
        output_hashes = _verify_staged_output(stage)
        completed = {
            "schema": FINAL_MANIFEST_SCHEMA,
            "status": "COMPLETED",
            "run_name": run_name,
            "utc_completed_time": _utc_now(),
            "git_commit": record["git_commit"],
            "full_contract_sha256": snapshot["full_contract_sha256"],
            "full_run_state_sha256": snapshot["full_run_state_sha256"],
            "full_cache_inventory_sha256": snapshot[
                "full_cache_inventory_sha256"
            ],
            "full_cache_content_sha256": snapshot[
                "full_cache_content_sha256"
            ],
            "artifacts": output_hashes,
            "execution_mode": record["execution_mode"],
            "legacy_partial_recovery": record.get("legacy_partial"),
            "optimizer_constructed": False,
            "optimizer_steps_executed": 0,
            "test_loader_built": False,
            "test_accessed": False,
        }
        atomic_json(stage / "analysis_run_manifest.json", completed)
        atomic_text(stage / "ANALYSIS_COMPLETED", json.dumps(completed, indent=2))
        final = _final_output(project_root, run_name)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError("refusing to overwrite existing Analyze output")
        os.replace(stage, final)
        record.update(
            {
                "status": "COMPLETED",
                "stage": "published",
                "utc_completed_time": completed["utc_completed_time"],
                "published_output": str(final.resolve()),
            }
        )
        try:
            atomic_json(lock_dir / "record.json", record)
            _archive_lock(_control_root(project_root), lock_dir, reason="completed")
        except OSError as error:
            print(
                f"[analysis:warning] completed control-record archival failed: "
                f"{error}",
                file=sys.stderr,
                flush=True,
            )
        progress.emit(
            f"[analysis:completed] output={final} "
            f"elapsed={time.monotonic() - progress.started:.1f}s"
        )
        return completed
    except BaseException as error:
        published_files = []
        if stage.exists():
            published_files = [
                item.relative_to(stage).as_posix()
                for item in sorted(stage.rglob("*"))
                if item.is_file()
            ]
        failure = {
            "schema": RUNNER_SCHEMA,
            "status": "FAILED",
            "run_name": run_name,
            "exception_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            "utc_failed_time": _utc_now(),
            "stage": record.get("stage"),
            "pid": record["pid"],
            "git_commit": record["git_commit"],
            "full_contract_sha256": snapshot["full_contract_sha256"],
            "staged_files": published_files,
            "published_files": [],
            "formal_output_published": False,
            "optimizer_constructed": False,
            "optimizer_steps_executed": 0,
            "test_loader_built": False,
            "test_accessed": False,
        }
        atomic_json(lock_dir / "ANALYSIS_FAILED.json", failure)
        record.update(
            {
                "status": "FAILED",
                "stage": record.get("stage"),
                "utc_failed_time": failure["utc_failed_time"],
                "failure_record_sha256": sha256_file(
                    lock_dir / "ANALYSIS_FAILED.json"
                ),
            }
        )
        atomic_json(lock_dir / "record.json", record)
        print(
            f"[analysis:failed] run={run_name} stage={record.get('stage')} "
            f"error={type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-instance T5 Analyze runner")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--recover-failed-analysis", action="store_true")
    parser.add_argument("--recover-stale-analysis-lock", action="store_true")
    parser.add_argument("--recover-legacy-partial-analysis", action="store_true")
    args = parser.parse_args()
    try:
        run_analyze(
            args.project_root,
            args.config,
            args.run_name,
            recover_failed=args.recover_failed_analysis,
            recover_stale=args.recover_stale_analysis_lock,
            recover_legacy_partial=args.recover_legacy_partial_analysis,
        )
    except AnalyzeLockError as error:
        print(str(error), file=sys.stderr, flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
