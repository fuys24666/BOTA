"""Stage22: post-hoc, bounded numerical diagnosis of Stage21 IFRU.

No training, retuning, FinalTest access, adapter publication or rewriting of
historical results. Model work is opt-in via Full; Smoke is CPU synthetic only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import time
import uuid
from pathlib import Path

import torch

from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, sha256_file
from src.paper_if_a2.artifacts import atomic_torch_save
from . import ali_stage21_amazon_seed43_kernel_v1 as k
from . import ali_stage21_repair01 as repair
from . import ali_stage22_numerics as num

SCHEMA = "bota-ali-stage22-ifru-numerical-v1"
CONFIG = "configs/ali_stage22_ifru_numerical_v1.json"
OWN_FILES = [CONFIG, "src/bota_short_benchmark/ali_stage22_ifru_numerical_v1.py",
             "src/bota_short_benchmark/ali_stage22_numerics.py",
             "scripts/bota_if/run_ali_stage22_ifru_numerical_v1.ps1"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(repo):
    c = read(repo/CONFIG)
    expected = {"schema": SCHEMA, "seed": 43, "requests": ["A1", "A2", "A3"],
                "panel_samples": 1024, "panel_seed": 43, "batch_size": 8,
                "damping_ratio": .1, "power_iterations": 12, "power_tolerance": .0001,
                "variants": ["legacy", "monitor", "restart5"], "restart_interval": 5,
                "probe_scales": [.0001, .01, 100.], "probe_seed": 220043,
                "diagnostic_only": True, "selection_performed": False, "test_accessed": False,
                "final_test_forbidden": True, "apply_solutions": False, "optimizer_steps": 0,
                "output_root": "outputs/ali_stage22_ifru_numerical_v1",
                "parent_root": "outputs/ali_stage21_amazon_seed43_v1",
                "status": "SPECIFIED_BEFORE_STAGE22_MODEL_EXECUTION"}
    if any(c.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage22 diagnostic scope changed; use a new version")
    if c["cg"] != dict(relative_tolerance=.0001, absolute_tolerance=1e-10, max_iterations=40,
                       residual_explosion_factor=1000., pap_tolerance=1e-14):
        raise ValueError("historical CG tolerances/budget must be preserved")
    return c


def verify(dest, contract=None):
    m = read(dest/"manifest.json")
    if m["schema"] != SCHEMA or (dest/"COMPLETED").read_text(encoding="utf-8") != SCHEMA+"\n":
        raise ValueError(f"invalid Stage22 bundle: {dest}")
    actual = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if actual != set(m["files"]): raise ValueError(f"bundle inventory changed: {dest}")
    for rel, digest in m["files"].items():
        if sha256_file(dest/rel) != digest: raise ValueError(f"changed artifact: {dest/rel}")
    result = read(dest/"result.json")
    if contract is not None and result["contract"] != contract:
        raise ValueError(f"diagnostic contract changed: {dest}")
    return result


def stage_dir(root):
    path = root/".work"/uuid.uuid4().hex[:10]
    path.mkdir(parents=True)
    return path


def publish(stage, dest, result):
    if dest.exists(): raise FileExistsError(dest)
    atomic_json(stage/"result.json", result)
    atomic_text(stage/"COMPLETED", SCHEMA+"\n")
    atomic_json(stage/"manifest.json", {"schema": SCHEMA, "files": {
        p.relative_to(stage).as_posix(): sha256_file(p) for p in sorted(stage.rglob("*")) if p.is_file()}})
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.rename(stage, dest)


def audit(repo, workspace, c):
    """Read-only predecessor audit including Stage21's exact repair migration."""
    oldroot = workspace/c["parent_root"]
    for key, digest in c["parent_hashes"].items():
        k.verify(oldroot/key)
        if sha256_file(oldroot/key/"result.json") != digest: raise ValueError(f"changed predecessor: {key}")
    p = read(oldroot/"protocol/result.json")
    old = read(oldroot/"comparison/IFRU/result.json")
    if old["contract"] != canonical_hash(p): raise ValueError("Stage21 contract mismatch")
    checks = copy.deepcopy(p["checks"])
    r = read(repo/repair.RECORD)
    if set(r["migrations"]) != repair.ALLOWED or sha256_file(Path(repair.__file__)) != r["repair_validator_sha256"]:
        raise ValueError("unrecognised Stage21 repair")
    for name, binding in r["migrations"].items():
        if checks["source_sha256"][name] not in (binding["before"], binding["after"]):
            raise ValueError("Stage21 source predecessor changed")
        if sha256_file(repo/name) != binding["after"]: raise ValueError("Stage21 repair source changed")
        checks["source_sha256"][name] = binding["after"]
    # Unlike repair.check(), this never appends a record under Stage21.
    repair.parent.s19.check_inputs(repo, checks)
    rows = {x["request"]: x for x in old["rows"]}
    if set(rows) != set(c["requests"]): raise ValueError("historical requests changed")
    for rid, row in rows.items():
        if row["status"] != c["historical_status"][rid] or row["candidate_id"] != "ifru_p1024_d0.1":
            raise ValueError("historical outcomes/selection changed")
    if p["config"]["grids"]["IFRU"]["cg_max_iterations"] != 40:
        raise ValueError("inherited solver budget changed")
    requests = p["comparison"]["requests"]
    if [r["id"] for r in requests] != c["requests"] or len(p["registry"]["order"]) != 3200:
        raise ValueError("request/order mismatch")
    source_hashes = dict(checks["source_sha256"])
    source_hashes.update({name: sha256_file(repo/name) for name in OWN_FILES})
    # Cover dependencies even when a predecessor source inventory omitted one.
    for name in ("src/paper_ratio_suite/ifru_t5.py", "src/diagnostics/t5_lora_influence_feasibility_audit.py",
                 "src/diagnostics/t5_lora_influence_feasibility_audit_a2.py",
                 "src/if_a2_optimization/group_c_gpu_resident_hvp.py", "src/paper_if_a2/common.py",
                 "src/paper_if_a2/artifacts.py", repair.RECORD):
        source_hashes[name] = sha256_file(repo/name)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                   "float32_matmul_precision": torch.get_float32_matmul_precision()}
    protocol = {"schema": SCHEMA, "config": c, "parent_contract": canonical_hash(p),
                "source_sha256": source_hashes, "inherited_checks_sha256": canonical_hash(checks),
                "requests": requests, "environment": environment,
                "test_accessed": False, "diagnostic_only": True,
                "endpoint_evaluation": "not performed; no adapter is modified",
                "target_gap": {"solver": "damped sampled endpoint GGN inverse times forget sample-mean gradient",
                               "paper_reference": "200-step AdamW slot-preserving replay from fixed anchor",
                               "equivalence_assumed": False, "causal_attribution": "not established"}}
    return protocol, p, rows


def freeze(root, protocol):
    identity = canonical_hash(protocol)
    dest = root/"protocol"
    payload = {"contract": identity, "protocol": protocol}
    if dest.exists():
        if verify(dest) != payload: raise ValueError("frozen Stage22 source/input/environment changed")
    else: publish(stage_dir(root), dest, payload)
    return identity


def progress_writer(stage, request, unit):
    def write(payload):
        atomic_json(stage/"progress.json", {"request": request, "unit": unit, **payload})
        last = payload.get("last", {})
        print(f"[Stage22] {request}/{unit} iteration={payload.get('iterations', '-')} "
              f"recursive={last.get('recursive_relative_residual', '-')} true={last.get('true_relative_residual', '-')}", flush=True)
    return write


def run_request(root, workspace, protocol, p, historical, rid):
    c = protocol["config"]; contract = canonical_hash(protocol)
    units = ["setup", "probes", *c["variants"]]
    base = root/"requests"/rid
    if all((base/u).exists() for u in units):
        for u in units: verify(base/u, contract)
        print(f"[Stage22] reused all units for {rid}", flush=True)
        return
    request = next(r for r in p["comparison"]["requests"] if r["id"] == rid)
    model, names, params, bases, tok, data, users, device = k.fresh_endpoint(workspace, p)
    model.eval(); active_stage = None
    resident = operator = None
    try:
        panel = [p["registry"]["order"][i] for i in k.runner.deterministic_panel(3200, c["panel_samples"], c["panel_seed"])]
        resident, panel_meta = k.runner.materialize_resident_panel(data, panel, c["batch_size"], tok.pad_token_id, device)
        counter = {}; operator = k.runner.make_sample_mean_ggn_operator(model, resident, params, counter)
        anchor = k.runner.flatten_tensors([v.detach() for v in params])
        # Names + exact tensor bytes protect against a silently changed coordinate.
        import hashlib
        anchor_hash = hashlib.sha256(anchor.numpy().tobytes()).hexdigest()
        if (base/"setup").exists():
            setup = verify(base/"setup", contract)
            if setup["anchor_sha256"] != anchor_hash or setup["panel_sha256"] != canonical_hash(panel) or setup["names"] != names:
                raise ValueError("runtime endpoint/panel differs from saved diagnostics")
            vectors = torch.load(base/"setup/vectors.pt", map_location="cpu", weights_only=True)
            gradient = vectors["forget_gradient"]
            damping = setup["damping"]
        else:
            active_stage = stage_dir(root); started = time.perf_counter()
            estimate = k.runner.estimate_lambda_max(operator, anchor.numel(), seed=43,
                iterations=c["power_iterations"], convergence_tolerance=c["power_tolerance"], numerical_lower_bound=1e-14)
            parts, meta = k.runner.sample_mean_gradients(model, data, request["forget_train_indices"], params, device, 8, tok.pad_token_id)
            gradient = k.runner.flatten_tensors(parts)
            # Descriptive stationarity proxy on the SAME sampled panel, not a
            # full-objective convergence certificate and not a new solve target.
            panel_parts, panel_loss = k.runner.sample_mean_gradients(model, data, panel, params, device, 8, tok.pad_token_id)
            panel_gradient = k.runner.flatten_tensors(panel_parts)
            damping = estimate["lambda_max_hat"]*c["damping_ratio"]
            vectors = {"forget_gradient": gradient, "panel_gradient": panel_gradient}
            atomic_torch_save(active_stage/"vectors.pt", vectors)
            setup = {"contract": contract, "request": request, "names": names,
                     "anchor_sha256": anchor_hash, "parameter_norm": num.norm(anchor),
                     "parameter_dtypes": sorted({str(v.dtype) for v in params}),
                     "cg_vector_dtype": "torch.float64", "panel_sha256": canonical_hash(panel),
                     "panel": panel_meta, "estimate": estimate, "damping": damping,
                     "forget_gradient": {**meta, "norm": num.norm(gradient)},
                     "panel_gradient": {**panel_loss, "norm": num.norm(panel_gradient)},
                     "gradient_alignment": num.comparison(gradient, panel_gradient),
                     "historical": historical[rid], "operator_counter": dict(counter),
                     "wall_time_seconds": time.perf_counter()-started, "test_accessed": False}
            publish(active_stage, base/"setup", setup); active_stage = None
            del parts, panel_parts, panel_gradient
        # Historical-order replay is first: probes/monitoring cannot affect it.
        for variant in c["variants"]:
            dest = base/variant
            if dest.exists(): verify(dest, contract); continue
            active_stage = stage_dir(root)
            atomic_json(active_stage/"identity.json", {"contract": contract, "request": rid, "variant": variant})
            result = num.diagnostic_cg(operator, gradient, damping, c["cg"], variant,
                c["restart_interval"], progress_writer(active_stage, rid, variant))
            solution = result.pop("solution")
            atomic_torch_save(active_stage/"solution_diagnostic_only.pt", solution)
            result.update(contract=contract, request=rid, historical_status=historical[rid]["status"],
                          test_accessed=False, optimizer_steps=0, diagnostic_only=True,
                          solution_norm=num.norm(solution), setup_sha256=sha256_file(base/"setup/result.json"))
            if variant == "legacy":
                result["historical_status_reproduced"] = ((result["error"] == "false_residual_convergence")
                    if historical[rid]["status"] == "NUMERICAL_SOLVER_FAILURE" else result["historical_acceptance"])
                result["historical_final_relative_residual"] = historical[rid].get("cg_diagnostics", {}).get("final_relative_residual")
            else:
                legacy = verify(base/"legacy", contract)
                old_x = torch.load(base/"legacy/solution_diagnostic_only.pt", map_location="cpu", weights_only=True)
                result["vs_legacy_solution"] = num.comparison(solution, old_x)
                if variant == "monitor":
                    result["same_status_as_legacy"] = result["status"] == legacy["status"]
                del old_x
            publish(active_stage, dest, result); active_stage = None
            del solution
        if (base/"probes").exists(): verify(base/"probes", contract)
        else:
            active_stage = stage_dir(root)
            result = num.operator_probes(operator, gradient, c["probe_seed"], c["probe_scales"],
                                         progress_writer(active_stage, rid, "probes"))
            result.update(contract=contract, request=rid, test_accessed=False)
            publish(active_stage, base/"probes", result); active_stage = None
        if hashlib.sha256(k.runner.flatten_tensors([v.detach() for v in params]).numpy().tobytes()).hexdigest() != anchor_hash:
            raise RuntimeError("diagnostic modified model parameters")
    except BaseException as error:
        if active_stage is not None:
            atomic_json(active_stage/"FAILED.json", {"error": repr(error), "request": rid, "contract": contract,
                        "status": "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "EXECUTION_ERROR",
                        "test_accessed": False})
            print(f"[Stage22] incomplete diagnostics retained at {active_stage}", flush=True)
        raise
    finally:
        del operator, resident, model, params, bases, tok, data, users
        k.s18.release()


def analyze(root, protocol):
    c = protocol["config"]; contract = canonical_hash(protocol); missing = []; rows = []; probes = {}
    for rid in c["requests"]:
        base = root/"requests"/rid
        for unit in ["setup", "probes", *c["variants"]]:
            if not (base/unit).exists(): missing.append(f"{rid}/{unit}")
            else:
                value = verify(base/unit, contract)
                if unit in c["variants"]: rows.append(value)
                elif unit == "probes": probes[rid] = value
    if missing: return {"status": "INCOMPLETE", "missing": missing, "completed_solver_units": len(rows)}
    result = {"status": "COMPLETED", "contract": contract, "rows": rows, "probes": probes,
              "test_accessed": False, "selection_performed": False, "historical_results_unchanged": True,
              "scope": "Numerical diagnosis only; solutions NOT applied or scored; no privacy or BOTA-causality conclusion."}
    dest = root/"summary"
    if dest.exists(): verify(dest, contract); return result
    stage = stage_dir(root)
    fields = ["request", "variant", "historical_status", "status", "iterations", "operator_calls",
              "initial_residual", "target", "final_residual", "final_relative_residual", "true_over_threshold",
              "true_tolerance_satisfied", "historical_status_reproduced", "solution_norm", "wall_time_seconds"]
    with (stage/"solver_diagnostics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    lines = ["# Stage22 numerical diagnostic summary", "",
             "This is post-hoc diagnosis, not a replacement benchmark or a privacy test.", "",
             "| Request | Variant | Status | Iterations | True residual / threshold |",
             "|---|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['request']} | {row['variant']} | {row['status']} | {row['iterations']} | {row['true_over_threshold']:.6g} |")
    lines += ["", "Interpretation rules:", "",
              "- Legacy fails, restart passes: supports a numerical recurrence issue for that request, not correct unlearning.",
              "- Legacy differs from history: inspect environment/operator reproducibility before attribution.",
              "- Monitor differs from legacy: extra operator calls are not inert in this execution; investigate reproducibility.",
              "- Repeatability/additivity/symmetry/scaling probes are descriptive and directional, not certifications.",
              "- No FP64 model run was performed: CG/accumulation FP64 is distinct from model/HVP precision.",
              "- Sampled endpoint gradient norm does not establish convergence or target equivalence.",
              "- No inference about BOTA nonlinearity, user interaction or privacy follows from this stage alone."]
    atomic_text(stage/"REPORT.md", "\n".join(lines)+"\n")
    publish(stage, dest, result)
    return result


def smoke():
    g = torch.tensor([1., 2., -1.], dtype=torch.float64)
    matrix = torch.diag(torch.tensor([1., 3., 7.], dtype=torch.float64))
    cg = dict(relative_tolerance=.0001, absolute_tolerance=1e-10, max_iterations=40,
              residual_explosion_factor=1000., pap_tolerance=1e-14)
    for variant in ("legacy", "monitor", "restart5"):
        r = num.diagnostic_cg(lambda v: matrix@v, g, .1, cg, variant)
        assert r["true_tolerance_satisfied"] and not r["solution_applied"]
        assert torch.allclose(r["solution"], torch.linalg.solve(matrix+.1*torch.eye(3), g), atol=1e-7)
    return {"status": "SMOKE_PASS", "model_loaded": False, "test_accessed": False, "optimizer_steps": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path, default=Path("C:/python/E2URec-main"))
    parser.add_argument("--mode", choices=["Preflight", "Freeze", "Smoke", "Full", "Analyze"], default="Preflight")
    parser.add_argument("--request", choices=["All", "A1", "A2", "A3"], default="All")
    args = parser.parse_args(); repo = args.repo.resolve(); workspace = args.workspace.resolve()
    if args.mode == "Smoke": result = smoke()
    else:
        c = load_config(repo)
        print("[Stage22] read-only audit of predecessor, assets and sources; no model execution", flush=True)
        protocol, parent, historical = audit(repo, workspace, c)
        root = workspace/c["output_root"]
        if args.mode == "Preflight":
            result = {"status": "PREFLIGHT_PASS", "contract": canonical_hash(protocol), "requests": c["requests"],
                      "variants": c["variants"], "model_loaded": False, "test_accessed": False,
                      "optimizer_steps": 0, "output_root": str(root), "scope": c["interpretation"]}
        elif args.mode == "Analyze":
            if not (root/"protocol").exists(): result = {"status": "NOT_STARTED"}
            else:
                if verify(root/"protocol")["protocol"] != protocol: raise ValueError("frozen protocol changed")
                with k.lock(root): result = analyze(root, protocol)
        else:
            with k.lock(root):
                contract = freeze(root, protocol)
                if args.mode == "Freeze": result = {"status": "FROZEN", "contract": contract, "model_loaded": False}
                else:
                    for rid in (c["requests"] if args.request == "All" else [args.request]):
                        run_request(root, workspace, protocol, parent, historical, rid)
                    # Verify original inputs again before declaring completion.
                    after, _, _ = audit(repo, workspace, c)
                    if after != protocol: raise ValueError("source/input/environment changed during diagnostics")
                    result = analyze(root, protocol)
    if result.get("status") == "COMPLETED":
        print(json.dumps({"status": "COMPLETED", "solver_units": len(result["rows"]),
                          "report": str(root/"summary/REPORT.md"), "test_accessed": False}, indent=2))
    else: print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
