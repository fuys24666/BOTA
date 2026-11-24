"""Recompute the headline summaries from the public, aggregate manuscript CSVs."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path.name} has no data rows")
    return rows


def matched_cells(rows, methods):
    """Index rows by (seed, request) and require one scored row per comparison method."""
    cells = defaultdict(dict)
    for row in rows:
        key = row["seed"], row["request"]
        name = row["method"]
        if name in cells[key]:
            raise ValueError(f"Duplicate {name} for seed {key[0]}, request {key[1]}")
        cells[key][name] = row

    for (seed, request), cell in cells.items():
        for name in methods:
            if name not in cell or cell[name]["status"] != "OK":
                raise ValueError(f"Missing scored {name} for seed {seed}, request {request}")
    return cells


def print_fidelity(cells, group_field, calibrated, source_only):
    groups = defaultdict(list)
    for cell in cells.values():
        groups[cell[calibrated][group_field]].append(cell)

    for label, members in groups.items():
        residual = mean(float(cell[calibrated]["local_residual"]) for cell in members)
        error = mean(float(cell[calibrated]["E_U"]) for cell in members)
        baseline = mean(float(cell[source_only]["E_U"]) for cell in members)
        if baseline == 0:
            raise ValueError(f"Zero source-only mean error for {group_field} {label}")
        reduction = 100 * (1 - error / baseline)
        print(f"  {group_field} {label}: mean residual {residual:.4f}, "
              f"mean error reduction vs source-only {reduction:.1f}%")


def main(results_dir):
    for filename in ["movielens_primary.csv", "movielens_fisher.csv"]:
        rows = read_rows(results_dir / filename)
        cells = matched_cells(rows, ("BOTA64", "BOTA0"))
        print(f"MovieLens-1M {rows[0]['split']}")
        print_fidelity(cells, "request", "BOTA64", "BOTA0")

    rows = read_rows(results_dir / "amazon_comparison.csv")
    cells = matched_cells(rows, ("BOTA", "BOTA-source-only", "Original"))
    print("Amazon Development")
    print_fidelity(cells, "seed", "BOTA", "BOTA-source-only")
    wins = sum(float(cell["BOTA"]["E_U"]) < float(cell["Original"]["E_U"])
               for cell in cells.values())
    counts = Counter(row["status"] for row in rows)
    print(f"  Lower error than Original: {wins}/{len(cells)} seed-request cells")
    print("  Outcome counts (all methods): "
          + ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))

    by_bank = defaultdict(list)
    for row in read_rows(results_dir / "serving_latency.csv"):
        by_bank[int(row["bank_users"])].append(float(row["p95_ms"]))
    print("Resident-bank P95 latency across seed-request cells")
    for users, values in sorted(by_bank.items()):
        print(f"  {users} users: {min(values):.3f}-{max(values):.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results",
                        help="Directory containing the four public CSVs (default: repository results/)")
    try:
        main(parser.parse_args().results_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Cannot summarize manuscript results: {exc}\n")
