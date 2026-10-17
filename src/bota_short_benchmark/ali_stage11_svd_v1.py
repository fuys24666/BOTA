"""Read-only, post-hoc spectrum diagnosis of frozen Stage 6 impact banks.

No model/dataset imports, inference, training, or FinalTest access. Float64
blockwise Gram accumulation avoids d-by-d matrices and retained right vectors.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import time
import uuid
from pathlib import Path

import torch

from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, directory_hash, git_snapshot, sha256_file

SCHEMA = "bota-ali-stage11-svd-v1"
SEEDS = (41, 42, 43)
RANKS = (4, 8, 16, 32, 64, 128, 192, 256)
FILES = ("metrics.json", "spectrum.csv", "rank_errors.csv", "row_norms.csv", "pairwise_cosines.csv", "report.md", "contract.json", "COMPLETED")
SOURCE_FILES = ("src/bota_short_benchmark/ali_stage11_svd_v1.py", "src/paper_if_a2/common.py", "configs/ali_stage11_svd_v1.json", "configs/ali_stage0_amendment10_v1.json", "scripts/bota_if/run_ali_stage11_svd_v1.ps1")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"Path escapes declared root: {relative}")
    return path


def check_hash(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Source hash mismatch: {path}")
    return actual


def load_config(repo):
    config = read_json(repo / "configs/ali_stage11_svd_v1.json")
    if (config["schema"] != SCHEMA or config["seeds"] != list(SEEDS)
            or config["bank_shape"] != [256, 884736] or config["ranks"] != list(RANKS)
            or config["test_access_policy"] != "forbidden"
            or config["output_root"] != "outputs/ali_stage11_svd_v1"):
        raise ValueError("Stage 11 protocol identity changed")
    for relative, key in (("docs/audits/ali_stage6_result_audit_v1.json", "stage6_audit_sha256"),
                          ("docs/audits/ali_stage5_result_audit_v1.json", "stage5_audit_sha256"),
                          ("configs/ali_stage0_preregistration_v1.json", "parent_preregistration_sha256")):
        check_hash(repo / relative, config[key])
    amendment = read_json(repo / config["amendment"])
    if amendment["ranks"] != list(RANKS) or amendment["test_access_policy"] != "forbidden":
        raise ValueError("Amendment 10 and configuration disagree")
    return config


def preflight(repo, workspace, seed):
    """Hash source bytes, but do not deserialize tensor banks or run analysis."""
    config = load_config(repo)
    if seed not in SEEDS:
        raise ValueError("Unsupported seed")
    source_hashes = {}
    runs, manifests = {}, {}
    for stage, label in ((5, "sequential"), (6, "storage_screen")):
        run = workspace / f"outputs/ali_stage{stage}_{label}_v1/runs/ali_stage{stage}_{label}_seed{seed}_v1"
        audit = read_json(repo / f"docs/audits/ali_stage{stage}_result_audit_v1.json")
        source_hashes[str(run / "manifest.json")] = check_hash(run / "manifest.json", audit["source_sha256"][f"seed{seed}_manifest"])
        manifest = read_json(run / "manifest.json")
        # Deliberately whitelist parameter-only files; never open predictions.
        required = ["COMPLETED", "request_registry.json"]
        if stage == 6:
            required += ["bank.pt", "contract.json", "metrics.json"]
        for name in required:
            path = run / name
            source_hashes[str(path)] = check_hash(path, manifest["files"][name])
        runs[stage], manifests[stage] = run, manifest
    contract = read_json(runs[6] / "contract.json")
    if contract["seed"] != seed or contract["bank_users"] != 256:
        raise ValueError("Stage 6 contract mismatch")
    registry = read_json(runs[6] / "request_registry.json")
    hashes = registry["user_hashes"]
    if len(hashes) != 256 or len(set(hashes)) != 256 or canonical_hash(hashes) != contract["bank_user_hashes_sha256"]:
        raise ValueError("Bank hash registry mismatch")
    request_cells = read_json(runs[5] / "request_registry.json")["cells"]
    queries = {}
    for sid, binding in contract["query_bindings"].items():
        cells = [row for row in request_cells if row["scenario"] == sid and row["request_size"] == 8]
        sets = {tuple(sorted(row["public_user_hashes"])) for row in cells}
        if len(sets) != 1 or {row["union_hash"] for row in cells} != {binding["union_hash"]}:
            raise ValueError("Stage 5 final union identity mismatch")
        members = next(iter(sets))
        if len(members) != 8 or len(set(members)) != 8:
            raise ValueError("Invalid eight-user query")
        entry = manifests[5]["endpoint_sha256"][binding["union_hash"]]
        if entry["directory_key"] != binding["directory_key"]:
            raise ValueError("Endpoint directory binding mismatch")
        paths = {}
        for method in ("bota", "exact_masked"):
            path = checked_path(runs[5], f"endpoints/{binding['directory_key']}/{method}/adapter")
            actual = directory_hash(path)
            if actual != entry[method]:
                raise ValueError(f"Endpoint changed: {path}")
            paths[method] = str(path / "adapter_model.pt")
            source_hashes[str(path)] = actual
        queries[sid] = {"rows": [hashes.index(member) for member in members], **paths}
    if set(queries) != {"L8", "L4M4"}:
        raise ValueError("Missing registered query")
    original = runs[5] / "endpoints/original/adapter"
    if directory_hash(original) != manifests[5]["original_adapter_sha256"]:
        raise ValueError("Original endpoint changed")
    source_hashes[str(original)] = manifests[5]["original_adapter_sha256"]
    return {"status": "READY", "seed": seed, "bank": str(runs[6] / "bank.pt"),
            "original": str(original / "adapter_model.pt"), "queries": queries,
            "user_hashes": hashes, "stage6_metrics": str(runs[6] / "metrics.json"),
            "source_hashes": source_hashes, "config": config,
            "code_hashes": {name: sha256_file(repo / name) for name in SOURCE_FILES},
            "test_accessed": False, "model_loaded": False}


def quantiles(values):
    if values.numel() == 0:
        return {key: None for key in ("min", "p25", "median", "p75", "p95", "max")}
    result = torch.quantile(values.double(), torch.tensor([0., .25, .5, .75, .95, 1.], dtype=torch.float64))
    return dict(zip(("min", "p25", "median", "p75", "p95", "max"), result.tolist()))


def accumulate_gram(x, block_size, device="cpu", progress=False):
    if x.ndim != 2 or block_size < 1:
        raise ValueError("Expected matrix and positive block size")
    gram = torch.zeros((x.shape[0], x.shape[0]), dtype=torch.float64, device=device)
    blocks = math.ceil(x.shape[1] / block_size)
    for n, start in enumerate(range(0, x.shape[1], block_size)):
        block = x[:, start:start+block_size].to(device=device, dtype=torch.float64)
        if not torch.isfinite(block).all():
            raise ValueError("Non-finite impact bank")
        gram.add_(block @ block.T)
        if progress and (n % 16 == 0 or n + 1 == blocks):
            print(f"[gram] block={n+1}/{blocks}", flush=True)
    return gram.cpu()


def spectrum(gram, negative_tolerance=1e-10, numerical_threshold=1e-12):
    if not torch.isfinite(gram).all():
        raise ValueError("Non-finite Gram matrix")
    values, u = torch.linalg.eigh((gram + gram.T) * .5)
    scale = float(values.abs().max()) if values.numel() else 0.
    if scale and float(values.min()) < -negative_tolerance * scale:
        raise ValueError("Materially negative Gram eigenvalue; inspect numerical stability")
    negative_min = float(values.min()) if values.numel() else 0.
    values = values.flip(0).clamp_min(0)
    u = u.flip(1)
    sigma = values.sqrt()
    total = float(values.sum())
    cumulative = values.cumsum(0) / total if total else torch.zeros_like(values)
    if total:
        p = sigma[sigma > 0] / sigma.sum()
        entropy_rank = float(torch.exp(-(p * p.log()).sum()))
        stable_rank = total / float(values[0])
        ranks = {str(level): min(len(values), int(torch.searchsorted(cumulative, level)) + 1) for level in (.9, .95, .99)}
    else:
        entropy_rank, stable_rank, ranks = 0., 0., {str(level): 0 for level in (.9, .95, .99)}
    return {"singular_values": sigma.tolist(), "eigenvalues": values.tolist(),
            "cumulative_energy": cumulative.tolist(), "entropy_effective_rank": entropy_rank,
            "effective_rank_definition": "exp(-sum(p log p)), p=sigma/sum(sigma)",
            "stable_rank": stable_rank, "energy_ranks": ranks,
            "numerical_rank": int((values > (float(values[0]) * numerical_threshold if len(values) else 0)).sum()),
            "numerical_rank_definition": "Gram eigenvalues > largest eigenvalue * threshold; float64 Gram diagnostic",
            "minimum_eigenvalue_before_clipping": negative_min, "total_energy": total}, u


def direction_diagnostics(gram):
    norms = gram.diag().clamp_min(0).sqrt()
    active = norms > 0
    ids = torch.where(active)[0]
    normal = gram[active][:, active] / (norms[active, None] * norms[None, active]) if len(ids) else gram[:0, :0]
    pairs = torch.triu_indices(len(ids), len(ids), offset=1)
    cos = normal[pairs[0], pairs[1]].clamp(-1, 1)
    report = {"zero_rows": int((~active).sum()), "near_zero_rows_relative_1e12": int((norms <= norms.max()*1e-12).sum()),
              "row_norms": quantiles(norms), "pairwise_cosines": quantiles(cos),
              "absolute_pairwise_cosines": quantiles(cos.abs()), "direction_rows": len(ids),
              "direction_policy": "Exclude exactly zero rows; retain and report near-zero rows."}
    rows = [{"row_i": int(ids[i]), "row_j": int(ids[j]), "cosine": float(c)} for i, j, c in zip(pairs[0], pairs[1], cos)]
    return report, normal, norms, rows


def query_errors(x, u, rows, exact, ranks, block_size):
    """Direct blocked residuals avoid cancellation in Gram quadratic forms."""
    b = torch.zeros(x.shape[0], dtype=torch.float64)
    b[rows] = 1
    coefficients = torch.stack([u[:, :r] @ (u[:, :r].T @ b) for r in ranks])
    errors = torch.zeros(len(ranks), dtype=torch.float64)
    distortion = torch.zeros_like(errors)
    base_error = exact_norm = query_norm = 0.
    for start in range(0, x.shape[1], block_size):
        block = x[:, start:start+block_size].double()
        q = b @ block
        target = exact[start:start+block_size].double()
        compressed = coefficients @ block
        errors += ((compressed - target) ** 2).sum(1)
        distortion += ((compressed - q) ** 2).sum(1)
        base_error += float(((q-target)**2).sum())
        exact_norm += float((target**2).sum())
        query_norm += float((q**2).sum())
    baseline = math.sqrt(base_error/exact_norm) if exact_norm else None
    result = []
    for i, rank in enumerate(ranks):
        residual = math.sqrt(float(errors[i])/exact_norm) if exact_norm else None
        result.append({"rank": rank, "parameter_residual": residual,
                       "fp32_parameter_residual": baseline,
                       "relative_residual_increase": residual/baseline-1 if baseline else None,
                       "query_compression_relative_l2": math.sqrt(float(distortion[i])/query_norm) if query_norm else None})
    return result


def flatten(state, names):
    if set(state["B"]) != set(names):
        raise ValueError("Adapter module coordinate mismatch")
    return torch.cat([state["B"][name].float().reshape(-1) for name in names])


def csv_text(rows):
    if not rows:
        return ""
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def validate_run(path, binding):
    manifest = read_json(path / "manifest.json")
    if manifest["schema"] != SCHEMA or set(manifest["files"]) != set(FILES):
        raise ValueError("Incomplete Stage 11 manifest")
    for name in FILES:
        check_hash(path / name, manifest["files"][name])
    if (path / "COMPLETED").read_text().strip() != SCHEMA:
        raise ValueError("Invalid completion marker")
    contract = read_json(path / "contract.json")
    for key in ("source_hashes", "code_hashes", "config", "seed"):
        if contract[key] != binding[key]:
            raise ValueError(f"Completed result has different {key}; preserve it and use a new version")
    return read_json(path / "metrics.json")


def execute(repo, workspace, seed, device, block_size):
    binding = preflight(repo, workspace, seed)
    config = binding["config"]
    root = workspace / config["output_root"]
    destination = root / "runs" / f"seed{seed}"
    if destination.exists():
        validate_run(destination, binding)
        return {"status": "VERIFIED_ALREADY_COMPLETED", "seed": seed, "run_dir": str(destination)}
    stage = root / ".work" / uuid.uuid4().hex[:12]
    stage.mkdir(parents=True)
    begun = time.perf_counter()
    atomic_json(stage / "contract.json", {**binding, "git": git_snapshot(repo), "device": device,
                                           "block_size": block_size, "torch_version": torch.__version__})
    try:
        bank = torch.load(binding["bank"], map_location="cpu", weights_only=True, mmap=True)
        x = bank["impacts"]
        if (list(x.shape) != config["bank_shape"] or bank["user_hashes"] != binding["user_hashes"]
                or bank["seed"] != seed or bank["selected_arm"] != "FS_empirical_fisher"):
            raise ValueError("Bank payload differs from frozen identity")
        print(f"[seed {seed}] bank={tuple(x.shape)} device={device}; no training/inference", flush=True)
        gram = accumulate_gram(x, block_size, device, progress=True)
        raw, u = spectrum(gram, config["negative_eigenvalue_relative_tolerance"], config["numerical_eigenvalue_relative_threshold"])
        directions, normal, norms, pairs = direction_diagnostics(gram)
        normalized, _ = spectrum(normal, config["negative_eigenvalue_relative_tolerance"], config["numerical_eigenvalue_relative_threshold"])
        original_state = torch.load(binding["original"], map_location="cpu", weights_only=True)
        names = list(original_state["B"])
        original = flatten(original_state, names)
        if original.numel() != x.shape[1]:
            raise ValueError("Bank and adapter dimension mismatch")
        stage6_rows = read_json(binding["stage6_metrics"])["rows"]
        rank_rows, checks = [], {}
        for sid, query in binding["queries"].items():
            print(f"[seed {seed}] query={sid}: rank reconstructions", flush=True)
            exact = flatten(torch.load(query["exact_masked"], map_location="cpu", weights_only=True), names)-original
            old = flatten(torch.load(query["bota"], map_location="cpu", weights_only=True), names)-original
            composed = x[query["rows"]].double().sum(0)
            denom = float(torch.linalg.vector_norm(old.double()))
            reproduction = float(torch.linalg.vector_norm(composed-old.double())) / max(denom, 1e-30)
            if reproduction > config["fp32_endpoint_reproduction_relative_tolerance"]:
                raise ValueError(f"Bank composition does not reproduce Stage 5: {sid}: {reproduction}")
            checks[sid] = reproduction
            rows = query_errors(x, u, query["rows"], exact, RANKS, block_size)
            for row in rows:
                r = row["rank"]
                energy = raw["cumulative_energy"][r-1]
                previous = [v for v in stage6_rows if v["bank_size"] == 256 and v["scenario"] == sid and v["representation"] == f"shared_low_rank_r{r}"]
                reported = previous[0]["relative_residual_increase"] if previous else None
                rank_rows.append({"seed": seed, "scenario": sid, **row, "retained_bank_energy": energy,
                                  "bank_relative_frobenius_error": math.sqrt(max(0., 1-energy)),
                                  "stage6_float32_residual_increase": reported,
                                  "stage11_minus_stage6_residual_increase": row["relative_residual_increase"]-reported if reported is not None and row["relative_residual_increase"] is not None else None})
        metric = {"schema": SCHEMA, "seed": seed, "raw_spectrum": raw, "normalized_spectrum": normalized,
                  "directions": directions, "query_rows": rank_rows, "endpoint_reproduction": checks,
                  "elapsed_seconds": time.perf_counter()-begun, "test_accessed": False, "model_loaded": False,
                  "optimizer_steps": 0, "stage6_decision_replaced": False}
        spectrum_rows = [{"seed": seed, "analysis": label, "index": i+1, "singular_value": s,
                          "energy_fraction": info["eigenvalues"][i]/info["total_energy"] if info["total_energy"] else 0.,
                          "cumulative_energy": info["cumulative_energy"][i]}
                         for label, info in (("raw_uncentered", raw), ("row_normalized", normalized)) for i, s in enumerate(info["singular_values"])]
        atomic_json(stage / "metrics.json", metric)
        atomic_text(stage / "spectrum.csv", csv_text(spectrum_rows))
        atomic_text(stage / "rank_errors.csv", csv_text(rank_rows))
        atomic_text(stage / "row_norms.csv", csv_text([{"row": i, "user_hash": h, "norm": float(norms[i])} for i, h in enumerate(binding["user_hashes"])]))
        atomic_text(stage / "pairwise_cosines.csv", csv_text(pairs))
        atomic_text(stage / "report.md", f"# Stage 11 seed {seed}\n\nPost-hoc parameter-space diagnosis only. No training, inference or FinalTest access.\n\nRaw entropy effective rank: {raw['entropy_effective_rank']:.4f}; stable rank: {raw['stable_rank']:.4f}.\n\nEnergy ranks: {raw['energy_ranks']}. Zero rows: {directions['zero_rows']}.\n\nDirection-normalized effective rank: {normalized['entropy_effective_rank']:.4f}.\n\nSee rank_errors.csv for query-specific reconstruction and comparison with Stage 6 float32 compression. The current spectrum uses float64 Gram accumulation. Near-tied cutoff eigenvalues can change the selected subspace.\n\nDo not infer universal orthogonality, population-scale impossibility, privacy guarantees or a new Stage 6 pass.\n")
        # Detect input mutation before publication, without reading any dataset.
        fresh = preflight(repo, workspace, seed)
        if fresh != binding:
            raise ValueError("Inputs or analysis code changed during execution")
        atomic_text(stage / "COMPLETED", SCHEMA+"\n")
        atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage/name) for name in FILES}})
        validate_run(stage, binding)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, destination)
        return {"status": "COMPLETED", "seed": seed, "run_dir": str(destination), "test_accessed": False}
    except Exception as error:
        # Preserve diagnostics in a short, unique staging path; never delete banks.
        atomic_json(stage / "FAILED.json", {"error": repr(error), "test_accessed": False})
        raise RuntimeError(f"Stage 11 failed before publication; diagnostic staging: {stage}") from error


def summarize(repo, workspace):
    config = load_config(repo)
    root = workspace / config["output_root"]
    metrics = [validate_run(root / "runs" / f"seed{s}", preflight(repo, workspace, s)) for s in SEEDS]
    rows = [{"seed": m["seed"], "raw_effective_rank": m["raw_spectrum"]["entropy_effective_rank"],
             "raw_stable_rank": m["raw_spectrum"]["stable_rank"],
             "rank90": m["raw_spectrum"]["energy_ranks"]["0.9"], "rank95": m["raw_spectrum"]["energy_ranks"]["0.95"],
             "rank99": m["raw_spectrum"]["energy_ranks"]["0.99"],
             "normalized_effective_rank": m["normalized_spectrum"]["entropy_effective_rank"],
             "zero_rows": m["directions"]["zero_rows"],
             "median_abs_cosine": m["directions"]["absolute_pairwise_cosines"]["median"]} for m in metrics]
    payload = {"schema": SCHEMA, "status": "DIAGNOSTICS_COMPLETE", "seeds": rows,
               "test_accessed": False, "stage6_decision_replaced": False,
               "source_manifests": {str(s): sha256_file(root/"runs"/f"seed{s}"/"manifest.json") for s in SEEDS}}
    target = root / "summary"
    if target.exists():
        manifest = read_json(target / "manifest.json")
        for name, digest in manifest["files"].items():
            check_hash(target/name, digest)
        if read_json(target/"summary.json") != payload:
            raise ValueError("Existing summary differs; preserve and investigate")
    else:
        stage = root / ".work" / uuid.uuid4().hex[:12]
        stage.mkdir(parents=True)
        atomic_json(stage / "summary.json", payload)
        atomic_text(stage / "summary.csv", csv_text(rows))
        atomic_text(stage / "rank_errors.csv", csv_text([r for m in metrics for r in m["query_rows"]]))
        atomic_text(stage / "report.md", "# Stage 11 three-seed diagnostic summary\n\nAll three seeds and both frozen requests are reported. See summary.csv and rank_errors.csv; per-seed spectrum.csv contains plot-ready values.\n\nThis is an explanatory follow-up to an observed failure, not an independent confirmation, new compression selection, privacy test, or population-scale feasibility claim. Stage 6 remains frozen.\n")
        atomic_json(stage/"manifest.json", {"schema": SCHEMA, "files": {n: sha256_file(stage/n) for n in ("summary.json", "summary.csv", "rank_errors.csv", "report.md")}})
        os.rename(stage, target)
    return {"status": "DIAGNOSTICS_COMPLETE", "summary_dir": str(target), "test_accessed": False}


def synthetic():
    x = torch.eye(8, dtype=torch.float64)
    result, _ = spectrum(accumulate_gram(x, 3))
    assert abs(result["entropy_effective_rank"]-8) < 1e-10
    return {"status": "SYNTHETIC_PASS", "effective_rank": result["entropy_effective_rank"], "test_accessed": False, "model_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workspace", type=Path, default=Path("C:/python/E2URec-main"))
    parser.add_argument("--mode", choices=("Preflight", "SyntheticDryRun", "Full", "Analyze", "Summary"), default="Preflight")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=41)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--block-size", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.block_size <= 65536 or args.threads < 1:
        parser.error("block-size must be 1..65536 and threads >= 1")
    torch.set_num_threads(args.threads)
    repo, workspace = args.repo.resolve(), args.workspace.resolve()
    if args.mode == "SyntheticDryRun":
        result = synthetic()
    elif args.mode == "Summary":
        result = summarize(repo, workspace)
    elif args.mode == "Full":
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        result = execute(repo, workspace, args.seed, device, args.block_size)
    else:
        binding = preflight(repo, workspace, args.seed)
        if args.mode == "Analyze":
            validate_run(workspace/binding["config"]["output_root"]/"runs"/f"seed{args.seed}", binding)
        result = {"status": "VALIDATED" if args.mode == "Analyze" else "READY", "seed": args.seed, "bank": binding["bank"], "source_files_verified": len(binding["source_hashes"]), "test_accessed": False, "bank_loaded": False}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
