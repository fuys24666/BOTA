from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from src.diagnostics.t5_reconstructed_official import sha256_file


GIT = Path("C:/Program Files/Git/cmd/git.exe")


def git_provenance(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    if not GIT.is_file():
        raise RuntimeError(f"Git executable is unavailable: {GIT}")

    def run(*arguments: str) -> str:
        try:
            return subprocess.run(
                [str(GIT), *arguments], cwd=root, check=True,
                capture_output=True, text=True, encoding="utf-8",
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"Git state cannot be verified: {arguments}") from error

    head = run("rev-parse", "--verify", "HEAD").strip()
    if len(head) != 40:
        raise RuntimeError("Git HEAD is not a full commit hash")
    porcelain = run("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    staged, tracked, untracked = [], [], []
    for line in porcelain:
        if len(line) < 4:
            raise RuntimeError(f"unparseable Git status record: {line!r}")
        code, path = line[:2], line[3:]
        if code == "??":
            untracked.append(path)
        else:
            if code[0] != " ":
                staged.append(path)
            if code[1] != " ":
                tracked.append(path)
    summary = {
        "tracked_modified": sorted(tracked),
        "staged_modified": sorted(staged),
        "untracked": sorted(untracked),
    }
    return {
        "git_commit": head,
        "git_working_tree_clean": not any(summary.values()),
        "git_status": summary,
        "git_status_sha256": hashlib.sha256(
            json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def require_clean_git(value: dict[str, Any], purpose: str) -> None:
    if value.get("git_working_tree_clean") is not True:
        raise RuntimeError(f"{purpose} requires a clean Git working tree")
    if not isinstance(value.get("git_commit"), str) or len(value["git_commit"]) != 40:
        raise RuntimeError(f"{purpose} requires a verified Git HEAD")


def implementation_provenance(project_root: Path, relative_paths: Iterable[str]) -> dict[str, Any]:
    root = project_root.resolve()
    files: dict[str, str] = {}
    for relative in relative_paths:
        path = (root / relative).resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"implementation path escapes repository: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        files[Path(relative).as_posix()] = sha256_file(path)
    return {
        "files": files,
        "canonical_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
