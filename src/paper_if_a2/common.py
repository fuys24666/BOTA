from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


ATOMIC_REPLACE_TIMEOUT_SECONDS = 5.0
ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def directory_hash(path: Path) -> str:
    return canonical_hash([(str(item.relative_to(path)).replace("\\", "/"), sha256_file(item)) for item in sorted(path.rglob("*")) if item.is_file()])


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text); handle.flush(); os.fsync(handle.fileno())
    deadline = time.monotonic() + ATOMIC_REPLACE_TIMEOUT_SECONDS; delay = ATOMIC_REPLACE_INITIAL_DELAY_SECONDS
    while True:
        try:
            os.replace(temporary, path); break
        except PermissionError:
            if time.monotonic() >= deadline: raise
            time.sleep(delay); delay = min(delay * 2, .25)


def atomic_json(path: Path, value: Any) -> None:
    ensure_finite(value); atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def safe_run_name(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value):
        raise ValueError("RunName must contain only letters, digits, '_' or '-'")
    return value


def safe_new_directory(root: Path, category: str, run_name: str) -> Path:
    output = (root / "outputs" / "paper_if_a2_v1" / category / safe_run_name(run_name)).resolve()
    expected = (root / "outputs" / "paper_if_a2_v1").resolve()
    if expected not in output.parents or output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); return output


def ensure_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value): raise ValueError("non-finite value")
    if isinstance(value, dict):
        for item in value.values(): ensure_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value: ensure_finite(item)


def git_snapshot(root: Path) -> dict[str, Any]:
    git = r"C:\Program Files\Git\cmd\git.exe"
    head = subprocess.run([git, "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run([git, "-C", str(root), "status", "--porcelain=v1"], capture_output=True, text=True, check=True).stdout
    return {"head": head, "clean": not status.strip(), "status_sha256": hashlib.sha256(status.encode()).hexdigest()}


def hardware_snapshot() -> dict[str, Any]:
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    parts = [item.strip() for item in query.stdout.strip().split(",")] if query.returncode == 0 else []
    return {"gpu_name": parts[0] if parts else None, "gpu_uuid": parts[1] if len(parts)>1 else None, "driver": parts[2] if len(parts)>2 else None, "gpu_memory_mib": float(parts[3]) if len(parts)>3 else None, "torch": torch.__version__, "cuda": torch.version.cuda, "python": __import__("sys").version.split()[0], "hostname_sha256": hashlib.sha256(socket.gethostname().encode()).hexdigest(), "single_gpu": torch.cuda.device_count() == 1}


def conflicting_processes() -> list[dict[str, Any]]:
    import psutil
    needles = ("t5_if_a2_efficiency_audit", "t5_e2urec_diagnostics", "t5_lora_influence_feasibility", "src.paper_if_a2", "normal_train.py", "unlearning_e2urec.py")
    current = os.getpid(); rows = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if process.info["pid"] != current and any(needle in command for needle in needles):
                rows.append({"pid": process.info["pid"], "command_sha256": hashlib.sha256(command.encode()).hexdigest()})
        except (psutil.AccessDenied, psutil.NoSuchProcess): pass
    return rows


def require_formal_preflight(root: Path) -> dict[str, Any]:
    git = git_snapshot(root); conflicts = conflicting_processes()
    if not git["clean"]: raise RuntimeError("formal paper run requires clean Git")
    if conflicts: raise RuntimeError("conflicting E2URec GPU process detected; stop it manually")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("formal paper run requires one CUDA GPU")
    return {"git": git, "hardware": hardware_snapshot(), "conflicts": conflicts}


@contextlib.contextmanager
def gpu_timer() -> Iterator[dict[str, float]]:
    if torch.cuda.is_available(): torch.cuda.synchronize()
    started = time.perf_counter(); box: dict[str, float] = {}
    try: yield box
    finally:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        box["seconds"] = time.perf_counter() - started
