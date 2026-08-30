from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch

from .common import atomic_json, atomic_text, canonical_hash, directory_hash, sha256_file


def write_contract(run_dir: Path, value: dict[str, Any]) -> str:
    atomic_json(run_dir / "contract.json", value); return sha256_file(run_dir / "contract.json")


def publish_manifest(run_dir: Path, files: list[str], extra: dict[str, Any]) -> dict[str, Any]:
    manifest = {"schema": "paper-if-a2-artifact-v1", "files": {name: (directory_hash(run_dir/name) if (run_dir/name).is_dir() else sha256_file(run_dir/name)) for name in files}, **extra}
    atomic_json(run_dir / "manifest.json", manifest); return manifest


def complete(run_dir: Path, marker: str) -> None:
    atomic_text(run_dir / "COMPLETED", marker + "\n")


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); stage = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with stage.open("wb") as handle: torch.save(value, handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(stage, path)


def publish_directory(stage: Path, destination: Path) -> None:
    if destination.exists(): raise FileExistsError(destination)
    os.replace(stage, destination)


def verify_completed(run_dir: Path, marker: str) -> dict[str, Any]:
    if (run_dir / "COMPLETED").read_text(encoding="utf-8") != marker + "\n": raise ValueError("invalid completion marker")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = run_dir / name; actual = directory_hash(path) if path.is_dir() else sha256_file(path)
        if actual != expected: raise ValueError(f"artifact SHA mismatch: {name}")
    return manifest


def save_resume(run_dir: Path, state: dict[str, Any]) -> None:
    target = run_dir / "resume"; stage = run_dir / f".resume.{uuid.uuid4().hex}.tmp"; stage.mkdir()
    atomic_torch_save(stage / "state.pt", state); atomic_json(stage / "manifest.json", {"state_sha256": sha256_file(stage/"state.pt"), "paper_retrain_resume_only": True})
    if target.exists(): shutil.rmtree(target)
    os.replace(stage, target)


def load_resume(run_dir: Path) -> dict[str, Any]:
    target = run_dir / "resume"; manifest = json.loads((target/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("paper_retrain_resume_only") is not True or sha256_file(target/"state.pt") != manifest.get("state_sha256"): raise ValueError("invalid Resume state")
    return torch.load(target/"state.pt", map_location="cpu", weights_only=False)


def remove_resume(run_dir: Path) -> None:
    target = run_dir / "resume"
    if target.exists(): shutil.rmtree(target)
    atomic_json(run_dir / "resume_deleted.json", {"resume_removed": True})
