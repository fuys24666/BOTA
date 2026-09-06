from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = ("configs", "scripts", "src")
LOCAL_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:\\(?:Users|Program Files)\\|/home/[^/\s]+/)",
    re.IGNORECASE,
)


def test_release_sources_do_not_embed_machine_local_paths() -> None:
    offenders: list[str] = []
    for directory in SOURCE_DIRECTORIES:
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {
                ".py",
                ".ps1",
                ".yaml",
                ".yml",
            }:
                continue
            text = path.read_text(encoding="utf-8")
            if LOCAL_PATH_PATTERN.search(text):
                offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_citation_metadata_matches_public_repository() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0"
    assert citation["repository-code"] == "https://github.com/fuys24666/BOTA"
    assert citation["license"] == "MIT"
    assert citation["authors"][0]["orcid"] == "https://orcid.org/0009-0005-3403-3964"
