"""Shared scenario selection and phase-timing contracts for the short benchmark."""
from __future__ import annotations

import math
from typing import Any, Sequence


TIMING_SCHEMA = "bota-short-phase-timing-v1"
SCENARIO_CHOICES = ("All", "K2", "K4", "L8", "L4M4", "L3M3H2")


def select_scenarios(rows: Sequence[dict[str, Any]], scenario: str) -> list[dict[str, Any]]:
    if scenario not in SCENARIO_CHOICES:
        raise ValueError(f"unsupported scenario: {scenario}")
    selected = list(rows) if scenario == "All" else [row for row in rows if row.get("id") == scenario]
    if not selected or (scenario != "All" and len(selected) != 1):
        raise ValueError(f"scenario missing from frozen registry: {scenario}")
    return selected


def timing_record(
    *,
    scenario: str,
    initialization_seconds: float,
    offline_construction_seconds: float,
    online_compute_seconds: float,
    adapter_publication_seconds: float | None,
    end_to_end_seconds: float,
    details: dict[str, float] | None = None,
    publication_included_in_online_compute: bool = False,
) -> dict[str, Any]:
    values = {
        "initialization_seconds": initialization_seconds,
        "offline_construction_seconds": offline_construction_seconds,
        "online_compute_seconds": online_compute_seconds,
        "end_to_end_seconds": end_to_end_seconds,
    }
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("phase timing must be finite and nonnegative")
    if adapter_publication_seconds is not None and (
        not isinstance(adapter_publication_seconds, (int, float))
        or not math.isfinite(adapter_publication_seconds)
        or adapter_publication_seconds < 0
    ):
        raise ValueError("adapter publication timing must be finite and nonnegative")
    detail_values = dict(details or {})
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in detail_values.values()):
        raise ValueError("detailed phase timing must be finite and nonnegative")
    online_total = online_compute_seconds + (adapter_publication_seconds or 0.0)
    return {
        "schema": TIMING_SCHEMA,
        "scenario": scenario,
        **values,
        "adapter_publication_seconds": adapter_publication_seconds,
        "online_total_seconds": online_total,
        "publication_included_in_online_compute": bool(publication_included_in_online_compute),
        "details": detail_values,
    }
