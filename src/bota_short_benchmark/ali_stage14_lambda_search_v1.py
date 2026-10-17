"""Stage 14: bounded coarse/fine Development Fisher-response calibration.

Frozen Stage 13 remains read-only. Coarse grid 16/32/64/128, at most one
prespecified midpoint round, confirmation to 16 and mandatory no-Fisher ablation.
No FinalTest access. Separate atomic phase/seed/request checkpoints.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import json
import math
import os
import platform
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, git_snapshot, sha256_file
from . import ali_stage8_bota_response_sweep_v1 as s8
from . import ali_stage9_crossseed_development_v1 as s9
from . import ali_stage12_lambda_extension_v1 as s12
from . import ali_stage13_lambda_search_v1 as s13

SCHEMA = "bota-ali-stage14-lambda-search-v1"
COARSE_GRID = (16., 32., 64., 128.)
HISTORICAL_GRID = (*s12.GRID, 4., 8., 16.)
ABS_GAIN = 1e-6
REL_GAIN = .01
SCENARIOS = ("L8", "L4M4")
NUMERICAL_ERRORS = {
    "singular_adamw_tangent_at_zero_second_moment",
    "nonfinite_adamw_second_moment_correction",
    "nonfinite_stable_adamw_tangent",
}
cid = s8._candidate_id
read_json = s8._json


def load_config(repo, path):
    config = read_json(path)
    if config.get("schema") != SCHEMA or config.get("test_access_policy") != "forbidden":
        raise ValueError("invalid Stage 14 identity or data boundary")
    for key in ("amendment", "stage8_config", "stage9_config", "stage13_config", "stage13_code"):
        item = config[key]
        if sha256_file(repo / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen binding changed: {key}")
    if config["protocol"] != {
        "selection_seed": 42, "confirmation_seeds": [41, 43],
        "scenarios": list(SCENARIOS), "coarse_grid": list(COARSE_GRID),
        "maximum_lambda": 128., "fine_rounds": 1, "controls": [1., 16.],
        "ablation_scale": 0., "ablation_seeds": [41, 42, 43],
        "absolute_gain_minimum": ABS_GAIN, "relative_gain_minimum": REL_GAIN,
        "optimizer_steps": 200, "split": "Development/Dev-Select", "final_test": False,
        "source_scaled": False, "endpoint_interpolation": False,
        "reproduction_atol": 1e-8, "reproduction_rtol": .001,
    }:
        raise ValueError("Stage 14 rules changed; require a new version")
    if config["output_root"] != "outputs/ali_stage14_lambda_search_v1":
        raise ValueError("unexpected output root")
    amendment = read_json(repo / config["amendment"]["path"])
    if amendment["status"] != "SPECIFIED_BEFORE_STAGE14_EXECUTION":
        raise ValueError("amendment status mismatch")
    return config


def source_map(repo, config_path):
    # Record uncommitted implementation too; no need to force a Git commit.
    paths = list((repo / "src").rglob("*.py"))
    paths += list((repo / "configs").glob("*"))
    paths += list((repo / "docs" / "audits").glob("*.json"))
    paths += [repo / "scripts/bota_if/run_ali_stage14_lambda_search_v1.ps1", config_path]
    return {str(p.relative_to(repo)).replace("\\", "/"): sha256_file(p)
            for p in sorted(set(paths)) if p.is_file()}


def verify_bundle(path):
    if (path / "COMPLETED").read_text(encoding="utf-8") != SCHEMA + "\n":
        raise ValueError(f"incomplete Stage 14 artifact: {path}")
    manifest = read_json(path / "manifest.json")
    actual = {str(p.relative_to(path)).replace("\\", "/") for p in path.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if actual != set(manifest["files"]) or manifest.get("schema") != SCHEMA:
        raise ValueError(f"Stage 14 artifact inventory changed: {path}")
    for name, expected in manifest["files"].items():
        if sha256_file(path / name) != expected:
            raise ValueError(f"Stage 14 artifact changed: {path / name}")
    s9._assert_no_test_access(manifest)
    return manifest


def publish(stage, destination):
    if destination.exists():
        raise FileExistsError(destination)
    atomic_text(stage / "COMPLETED", SCHEMA + "\n")
    files = {str(p.relative_to(stage)).replace("\\", "/"): sha256_file(p)
             for p in sorted(stage.rglob("*")) if p.is_file()}
    atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": files, "test_accessed": False})
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(stage, destination)


def staging(root):
    # Short enough for torch's UUID-based atomic temporary names on Windows.
    path = root / ".work" / uuid.uuid4().hex[:12]
    path.mkdir(parents=True)
    return path


@contextlib.contextmanager
def execution_lock(root):
    """OS-managed lock is released even after a killed process; no stale PID lock."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "execution.lock").open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("another Stage 14 Full process holds the output lock") from error
            try:
                yield
            finally:
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parent_snapshot(repo, workspace, config):
    parent_config = s13.load_config(repo, repo / config["stage13_config"]["path"])
    parent = workspace / parent_config["output_root"]
    expected_paths = {"protocol", "coarse", "selection", "summary"} | {
        f"cells/{'coarse' if seed == 42 else 'confirm'}/s{seed}_{sid}"
        for seed in (41, 42, 43) for sid in SCENARIOS}
    if set(config["parent_manifests"]) != expected_paths:
        raise ValueError("incomplete Stage 13 binding")
    for relative, expected in config["parent_manifests"].items():
        bundle = parent / relative
        if sha256_file(bundle / "manifest.json") != expected:
            raise ValueError(f"Stage 13 manifest changed: {relative}")
        s13.verify_bundle(bundle)
    s13.check_frozen(repo, parent_config, parent)
    selection = read_json(parent / "selection/decision.json")
    summary = read_json(parent / "summary/summary.json")
    if selection["selected"]["response_scale"] != 16 or summary["confirmation"]["status"] != "CROSS_SEED_DEVELOPMENT_CONFIRMED":
        raise ValueError("Stage 13 authority is not the completed lambda=16 result")
    if s13.select_rows(selection["rows"], HISTORICAL_GRID)["selected"] != selection["selected"]:
        raise ValueError("Stage 13 selection recomputation mismatch")
    if s13.confirmation_decision(summary["confirmation_rows"], 16.) != summary["confirmation"]:
        raise ValueError("Stage 13 confirmation recomputation mismatch")
    return parent, selection, summary


def historical_rows(repo, workspace, config):
    _, selection, _ = parent_snapshot(repo, workspace, config)
    return copy.deepcopy(selection["rows"])


def preflight(repo, workspace, config_path):
    config = load_config(repo, config_path)
    parent, _, _ = parent_snapshot(repo, workspace, config)
    checks = s13.preflight(repo, workspace, repo / config["stage13_config"]["path"])
    checks["schema"] = SCHEMA
    checks["parent_manifests"] = {k: sha256_file(parent / k / "manifest.json") for k in config["parent_manifests"]}
    # Bind all parent contents so a mid-run change is caught before publication.
    for p in parent.rglob("*"):
        if p.is_file() and any(part in ("protocol", "selection", "summary", "cells") for part in p.relative_to(parent).parts):
            checks["input_sha256"][str(p)] = sha256_file(p)
    return checks


def eligible(row):
    return (row.get("status", "OK") == "OK" and row.get("utility_eligible") is True
            and row.get("prediction_collapse") is False
            and isinstance(row.get("E_U"), (float, int)) and math.isfinite(row["E_U"]))


def select_rows(rows, scales):
    result = s12.select_rows(rows, scales)
    result["boundary_winner"] = bool(result["selected"] and result["selected"]["response_scale"] == 128.)
    return result


def gain_over_sixteen(selected, control):
    if selected is None:
        return {"absolute": None, "relative": None, "passes_budget_gate": False}
    improvement = control - selected["equal_scenario_mean_EU"]
    relative = improvement / control if control > 0 else None
    return {"absolute": improvement, "relative": relative,
            "passes_budget_gate": bool(relative is not None and improvement >= ABS_GAIN and relative >= REL_GAIN)}


def coarse_decision(rows, baseline_mean):
    result = select_rows(rows, COARSE_GRID)
    winner = result["selected"]
    gain = gain_over_sixteen(winner, baseline_mean)
    scale = winner["response_scale"] if winner else None
    fine = []
    if scale in (32., 64.) and gain["passes_budget_gate"]:
        i = COARSE_GRID.index(scale)
        fine = [(COARSE_GRID[i-1] + scale)/2, (scale + COARSE_GRID[i+1])/2]
        reason = "ONE_PREDEFINED_FINE_ROUND"
    elif scale == 128.:
        reason = "UPPER_BOUNDARY_STOP_NO_EXPANSION"
    elif scale == 16.:
        reason = "LOWER_BOUNDARY_RETAIN_STAGE13"
    elif winner is None:
        reason = "NO_ELIGIBLE_COARSE_CANDIDATE"
    else:
        reason = "GAIN_BELOW_BUDGET_GATE"
    result.update(fine_grid=fine, next_action=reason, gain_over_lambda16=gain)
    return result


def check_reproduction(new, old, protocol):
    differences = {}
    for key in ("E_U", "D_U", "overall_auc", "overall_log_loss"):
        delta = abs(new[key] - old[key])
        tolerance = max(protocol["reproduction_atol"], protocol["reproduction_rtol"] * abs(old[key]))
        if not math.isfinite(delta) or delta > tolerance:
            raise RuntimeError(f"lambda control historical reproduction failed: {key}, diff={delta}, tolerance={tolerance}")
        differences[key] = {"absolute_difference": delta, "tolerance": tolerance}
    return differences


def run_transport(model, dataset, indices, user_ids, selected_users, parameters, names,
                  device, pad, engine, budget, scales, progress):
    optimizer = s8.p1._optimizer(parameters, engine)
    states = {cid(scale): s8.stage1._state(parameters, len(selected_users)) for scale in scales}
    failures = {}
    batch_size = int(engine["schedule"]["batch_size"])
    if len(indices) != 200 * batch_size:
        raise ValueError("Stage 14 requires the original 200-step trajectory")
    for step, start in enumerate(range(0, len(indices), batch_size), 1):
        chosen = list(indices[start:start + batch_size])
        batch_users = [int(user_ids[i]) for i in chosen]
        batch = s8.move_batch(s8.masked_batch(dataset, chosen, pad), device)
        losses = s8.p1._sample_losses(model, batch)
        sample_rows = [[] for _ in parameters]
        for sample in range(len(chosen)):
            gradients = torch.autograd.grad(losses[sample], parameters, retain_graph=True)
            for module, gradient in enumerate(gradients):
                sample_rows[module].append(gradient.detach())
        optimizer.zero_grad(set_to_none=True)
        formal_loss = losses.sum() / len(chosen)
        formal_loss.backward()
        coefficients = [torch.stack(rows) for rows in sample_rows]
        sources = []
        for rows, parameter in zip(coefficients, parameters):
            values = []
            for target in selected_users:
                slots = [i for i, user in enumerate(batch_users) if user == target]
                values.append(-rows[slots].sum(0) / len(chosen) if slots else torch.zeros_like(parameter))
            sources.append(torch.stack(values))
        for scale in scales:
            key = cid(scale)
            if key in failures:
                continue
            try:
                s8._advance_scaled(states[key], coefficients, sources, parameters, optimizer, step, engine, scale)
            except RuntimeError as error:
                # Only known numerical failures are scientific outcomes. Never
                # turn OOM, CUDA faults or programming errors into failed rows.
                if str(error) not in NUMERICAL_ERRORS or scale in (1., 16.):
                    raise
                failures[key] = {"status": "NUMERICAL_TRANSPORT_FAILURE", "reason": str(error), "first_failed_step": step}
                del states[key]
                continue
            if not all(torch.isfinite(value).all().item() for state in states[key] for value in state.values()):
                if scale in (1., 16.):
                    raise RuntimeError("nonfinite lambda=1/16 control; reproduction blocked")
                failures[key] = {"status": "NONFINITE_TRANSPORT", "first_failed_step": step}
                del states[key]
        budget.step(optimizer, "shared_canonical_reference")
        if step == 1 or step % 10 == 0:
            atomic_json(progress, {"step": step, "of": 200, "numerical_failures": failures, "test_accessed": False})
            print(f"[Stage14] transport step={step}/200 failures={list(failures)}", flush=True)
        del batch, losses, formal_loss, sample_rows, coefficients, sources, gradients
    canonical = {name: p.detach().float().cpu().clone() for name, p in zip(names, parameters)}
    return canonical, states, failures, s8.tensor_tree_hash(optimizer.state_dict()), s8.tensor_tree_hash(s8.capture_rng())


def validate_prediction(prediction, indices):
    if prediction.get("sample_order_sha256") != canonical_hash(list(indices)):
        raise RuntimeError("Development sample order mismatch")
    if any(len(prediction[key]) != len(indices) for key in ("probability", "gold_label", "answer_loss")):
        raise RuntimeError("incomplete prediction output")
    p = np.asarray(prediction["probability"])
    return bool(np.isfinite(p).all() and np.isfinite(prediction["answer_loss"]).all() and ((p >= 0) & (p <= 1)).all())


def execute_cell(repo, workspace, config, root, contract_hash, seed, sid, scales, selection_hash=None, phase="coarse"):
    if phase not in ("coarse", "fine", "confirm") or not {1., 16.}.issubset(scales):
        raise ValueError("invalid phase or missing reproduction controls")
    if any(scale < 0 or scale > 128 for scale in scales) or len(set(scales)) != len(scales):
        raise ValueError("invalid or duplicate lambda")
    destination = root / "cells" / phase / f"s{seed}_{sid}"
    identity = {"contract_sha256": contract_hash, "seed": seed, "scenario": sid,
                "scales": list(scales), "selection_sha256": selection_hash, "phase": phase}
    if destination.exists():
        verify_bundle(destination)
        result = read_json(destination / "metrics.json")
        if result["identity"] != identity:
            raise ValueError("existing cell belongs to a different protocol or selection")
        print(f"[Stage14] verified reuse {seed}/{sid}", flush=True)
        return result
    base = s8.load_config(repo, repo / config["stage8_config"]["path"])
    cross = s9.load_config(repo, workspace, repo / config["stage9_config"]["path"])
    base["benchmark"] = cross["benchmarks"][seed]
    benchmark = s8.stage7.load_benchmark_config(repo / base["benchmark"]["config"]["path"])
    benchmark["_config_path"] = str(repo / base["benchmark"]["config"]["path"])
    _, _, registry = s8.stage7.validate_prepared(workspace, benchmark, base["benchmark"]["name"])
    _, development_users, _, select = s8.stage7._development_partition(repo, workspace, base, benchmark)
    authority = s8.stage1.preflight(repo, workspace, repo / base["stage1_config"]["path"], seed, sid)[0]
    refs = authority["references"]
    original_run, exact_run, formal_run = (Path(refs[k]["run"]) for k in ("original", "exact_masked", "formal_bota"))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requires exactly one CUDA GPU")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free / 2**30 < base["runtime"]["minimum_free_gib"]:
        raise RuntimeError("insufficient free GPU memory (requires 10 GiB)")
    torch.cuda.set_per_process_memory_fraction(s8.runner.allocator_fraction_for(
        total / 2**30, base["runtime"]["allocator_fraction"], base["runtime"]["hard_peak_reserved_gib"]), device)
    torch.cuda.reset_peak_memory_stats()
    stage = staging(root)
    previous = Path.cwd()
    begun = time.perf_counter()
    print(f"[Stage14] seed={seed} scenario={sid} scales={list(scales)} work={stage}", flush=True)
    try:
        os.chdir(workspace)
        original_state = torch.load(original_run / "scenarios" / sid / "adapter/adapter_model.pt", map_location="cpu", weights_only=True)
        formal_state = torch.load(formal_run / "scenarios" / sid / "adapter/adapter_model.pt", map_location="cpu", weights_only=True)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model, names, parameters, bases, _, tokenizer, train_data, train_users = s8.runner._model_context(
            workspace, benchmark, registry, device, original_state["A"])
        scenario = s8.behavior._scenario(registry, sid)
        budget = s8.StepBudget(200)
        canonical, states, failures, optimizer_hash, rng_hash = run_transport(
            model, train_data, registry["order"], train_users, scenario["users"], parameters, names,
            device, tokenizer.pad_token_id, s8.runner._engine(workspace, benchmark), budget, scales, stage / "progress.json")
        trajectory_seconds = time.perf_counter() - begun
        if budget.calls != 200 or budget.by_arm != {"shared_canonical_reference": 200}:
            raise RuntimeError("optimizer budget mismatch")
        if any(not torch.equal(canonical[name], original_state["B"][name]) for name in names):
            raise RuntimeError("canonical endpoint differs from frozen Original")
        publication, adapter_hashes, parameter_diagnostics, scale_one = {}, {}, {}, None
        exact_state = torch.load(exact_run / "scenarios" / sid / "adapter/adapter_model.pt", map_location="cpu", weights_only=True)
        target_delta = torch.cat([(exact_state["B"][name].float() - canonical[name]).reshape(-1) for name in names])
        for scale in scales:
            key = cid(scale)
            if key in failures:
                continue
            candidate = {name: canonical[name] + states[key][i]["theta"].sum(0).float().cpu() for i, name in enumerate(names)}
            if not all(torch.isfinite(value).all().item() for value in candidate.values()):
                if scale in (1., 16.):
                    raise RuntimeError("nonfinite reproduction-control adapter")
                failures[key] = {"status": "NONFINITE_ADAPTER"}
                continue
            candidate_delta = torch.cat([(candidate[name] - canonical[name]).reshape(-1) for name in names])
            norm = float(torch.linalg.vector_norm(candidate_delta.double()))
            target_norm = float(torch.linalg.vector_norm(target_delta.double()))
            difference = float(torch.linalg.vector_norm(candidate_delta.double() - target_delta.double()))
            dot = float(torch.dot(candidate_delta.double(), target_delta.double()))
            parameter_diagnostics[key] = {
                "update_norm": norm, "exact_update_norm": target_norm,
                "relative_l2": difference / target_norm if target_norm else None,
                "cosine": dot / (norm * target_norm) if norm and target_norm else None,
                "diagnostic_only": True,
            }
            publish_start = time.perf_counter()
            adapter_hashes[key] = s8.runner._save_fixed_ab(stage / "adapters" / key / "adapter", names, candidate, bases, f"BOTA-FS-FisherScale-{scale:g}")["sha256"]
            publication[key] = time.perf_counter() - publish_start
            if scale == 1:
                scale_one = candidate
        formal_delta = torch.cat([(formal_state["B"][name].float() - canonical[name]).reshape(-1) for name in names])
        actual_delta = torch.cat([(scale_one[name] - canonical[name]).reshape(-1) for name in names])
        alignment = s8.p2b.vector_metrics(formal_delta, actual_delta)
        gate = base["formal_reproduction"]
        if not (alignment["finite"] and alignment["cosine"] >= gate["cosine_minimum"]
                and alignment["relative_l2_error"] <= gate["relative_l2_maximum"]
                and alignment["maximum_absolute_error"] <= gate["maximum_absolute_error"]):
            raise RuntimeError(f"lambda=1 formal reproduction failed: {alignment}")
        del model, parameters, states, bases, candidate, scale_one
        gc.collect(); torch.cuda.empty_cache()
        eval_tokenizer = s8.T5Tokenizer.from_pretrained(s8.runner._base_t5_config(workspace, benchmark)["paths"]["model_dir"])
        development = s8.JsonPromptDataset(workspace / benchmark["source"]["development_json"], eval_tokenizer)
        request = set(map(int, scenario["users"]))
        local = [pos for pos, idx in enumerate(select) if development_users[idx] in request]
        if not local:
            raise ValueError("empty requested-user Development support")
        predictions = {}
        for label, run in (("Original", original_run), ("Exact-Masked", exact_run)):
            print(f"[Stage14] {seed}/{sid} evaluating {label} on Dev-Select", flush=True)
            predictions[label] = s8.behavior._predict_reference(workspace, benchmark, run, sid, development, select, device)
            if not validate_prediction(predictions[label], select):
                raise RuntimeError(f"nonfinite reference predictions: {label}")
        rows = []
        for scale in scales:
            key = cid(scale)
            if key not in failures:
                print(f"[Stage14] {seed}/{sid} evaluating lambda={scale:g}", flush=True)
                prediction = s8.behavior._predict_scenario_dir(workspace, benchmark, stage / "adapters" / key, development, select, device)
                if not validate_prediction(prediction, select):
                    if scale in (1., 16.):
                        raise RuntimeError("nonfinite control predictions")
                    failures[key] = {"status": "NONFINITE_PREDICTIONS"}
                else:
                    predictions[key] = prediction
                    row = s8._metric(key, scale, sid, prediction, predictions["Original"], predictions["Exact-Masked"], local,
                                     trajectory_seconds, publication[key], base)
                    row.update(status="OK", candidate_state_sha256=adapter_hashes[key])
            if key in failures:
                row = {"candidate_id": key, "scenario": sid, "response_scale": scale,
                       "E_U": None, "D_U": None, "local_residual": None,
                       "utility_eligible": False, "prediction_collapse": None, **failures[key]}
            row.update(seed=seed, origin="stage14", test_accessed=False)
            rows.append(row)
        reproduction = {}
        parent, _, _ = parent_snapshot(repo, workspace, config)
        parent_phase = "coarse" if seed == 42 else "confirm"
        parent_rows = read_json(parent / "cells" / parent_phase / f"s{seed}_{sid}" / "metrics.json")["rows"]
        for scale in (1., 16.):
            oldrow = next(r for r in parent_rows if r["response_scale"] == scale)
            newrow = next(r for r in rows if r["response_scale"] == scale)
            reproduction[cid(scale)] = check_reproduction(newrow, oldrow, config["protocol"])
        if seed == 42 and 0. in scales:
            oldzero = next(r for r in historical_rows(repo, workspace, config)
                          if r["response_scale"] == 0 and r["scenario"] == sid)
            newzero = next(r for r in rows if r["response_scale"] == 0)
            if newzero["status"] != "OK":
                raise RuntimeError("seed42 lambda=0 reproduction failed")
            reproduction[cid(0)] = check_reproduction(newzero, oldzero, config["protocol"])
        peak = torch.cuda.max_memory_reserved() / 2**30
        if peak > base["runtime"]["hard_peak_reserved_gib"]:
            raise RuntimeError(f"GPU reservation exceeded protocol cap: {peak:.3f} GiB")
        result = {"schema": SCHEMA, "identity": identity, "rows": rows,
                  "formal_reproduction": alignment, "historical_reproduction": reproduction,
                  "parameter_diagnostics": parameter_diagnostics,
                  "physical_optimizer_step_calls": budget.calls, "optimizer_state_sha256": optimizer_hash,
                  "rng_state_sha256": rng_hash, "peak_reserved_gib": peak,
                  "wall_time_seconds": time.perf_counter() - begun, "test_accessed": False}
        atomic_json(stage / "metrics.json", result)
        atomic_json(stage / "predictions.json", {"indices": select, "local_positions": local, "predictions": predictions, "test_accessed": False})
        check_frozen(repo, config, root)
        publish(stage, destination)
        return result
    except Exception as error:
        atomic_json(stage / "FAILED.json", {"identity": identity, "error_type": type(error).__name__, "error": str(error), "test_accessed": False})
        print(f"[Stage14] incomplete diagnostics preserved: {stage}", flush=True)
        raise
    finally:
        os.chdir(previous)
        gc.collect(); torch.cuda.empty_cache()


def check_frozen(repo, config, root):
    verify_bundle(root / "protocol")
    frozen = read_json(root / "protocol/contract.json")
    for relative, expected in frozen["source_sha256"].items():
        if sha256_file(repo / relative) != expected:
            raise RuntimeError(f"source/config changed since freeze: {relative}")
    for filename, expected in frozen["preflight"]["input_sha256"].items():
        if sha256_file(Path(filename)) != expected:
            raise RuntimeError(f"experiment input changed since freeze: {filename}")
    return sha256_file(root / "protocol/contract.json")


def freeze_protocol(repo, workspace, config_path, config, root, checks):
    if (root / "protocol").exists():
        identity = check_frozen(repo, config, root)
        frozen = read_json(root / "protocol/contract.json")
        if frozen["preflight"] != checks:
            raise RuntimeError("reference/input preflight changed since protocol freeze")
        return identity
    stage = staging(root)
    atomic_json(stage / "contract.json", {"schema": SCHEMA, "config": config,
        "source_sha256": source_map(repo, config_path), "preflight": checks, "git": git_snapshot(repo),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "test_accessed": False})
    publish(stage, root / "protocol")
    return sha256_file(root / "protocol/contract.json")


def freeze_decision(root, name, payload):
    if name not in ("coarse", "selection"):
        raise ValueError("unknown decision")
    destination = root / name
    if destination.exists():
        verify_bundle(destination)
        if read_json(destination / "decision.json") != payload:
            raise RuntimeError(f"refusing to replace frozen {name} decision")
    else:
        stage = staging(root)
        atomic_json(stage / "decision.json", payload)
        publish(stage, destination)
    return sha256_file(destination / "decision.json")


def confirmation_decision(rows, scale):
    decisions, regressions = {}, []
    for seed in (41, 43):
        cells = [r for r in rows if r["seed"] == seed]
        select_rows(cells, (0., 1., 16., scale))
        candidate = [r for r in cells if r["response_scale"] == scale]
        controls = [r for r in cells if r["response_scale"] == 16]
        valid = all(eligible(r) and r.get("local_residual") is not None and r["local_residual"] < 1 for r in candidate)
        mean = sum(r["E_U"] for r in candidate)/2 if all(r.get("E_U") is not None for r in candidate) else None
        control_mean = sum(r["E_U"] for r in controls)/2
        passed = valid and all(eligible(r) for r in controls) and mean is not None and mean < control_mean
        for row in candidate:
            control = next(r for r in controls if r["scenario"] == row["scenario"])
            if row.get("E_U") is not None and row["E_U"] > control["E_U"]:
                regressions.append({"seed": seed, "scenario": row["scenario"],
                                    "control_lambda": 16., "EU_increase": row["E_U"] - control["E_U"]})
        decisions[str(seed)] = {"candidate_mean_EU": mean, "control_mean_EU": control_mean,
            "absolute_gain": control_mean - mean if mean is not None else None,
            "relative_gain": (control_mean - mean)/control_mean if mean is not None and control_mean else None,
            "passed": bool(passed)}
    return {"status": "CROSS_SEED_DEVELOPMENT_CONFIRMED" if all(v["passed"] for v in decisions.values()) else "NOT_STABLE",
            "seeds": decisions, "cell_regressions": regressions, "test_accessed": False, "finaltest_authorized": False}


def ablation_report(rows, target_scale):
    targets = sorted({16., float(target_scale)})
    expected = {(seed, sid, scale) for seed in (41, 42, 43) for sid in SCENARIOS for scale in (0., *targets)}
    relevant = [r for r in rows if r["response_scale"] in (0., *targets)]
    keys = [(r["seed"], r["scenario"], r["response_scale"]) for r in relevant]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("missing/duplicate no-Fisher ablation cells")
    cells, aggregates = [], {}
    for scale in targets:
        comparisons = []
        for seed in (41, 42, 43):
            for sid in SCENARIOS:
                zero = next(r for r in relevant if (r["seed"], r["scenario"], r["response_scale"]) == (seed, sid, 0.))
                target = next(r for r in relevant if (r["seed"], r["scenario"], r["response_scale"]) == (seed, sid, scale))
                finite = all(r.get("status") == "OK" and isinstance(r.get("E_U"), (int, float)) and math.isfinite(r["E_U"]) for r in (zero, target))
                gain = zero["E_U"] - target["E_U"] if finite else None
                item = {"seed": seed, "scenario": sid, "target_lambda": scale,
                        "zero_E_U": zero.get("E_U"), "target_E_U": target.get("E_U"),
                        "zero_status": zero["status"], "target_status": target["status"],
                        "zero_eligible": eligible(zero), "target_eligible": eligible(target),
                        "zero_residual": zero.get("local_residual"), "target_residual": target.get("local_residual"),
                        "zero_toward_exact_rate": zero.get("toward_exact_rate"), "target_toward_exact_rate": target.get("toward_exact_rate"),
                        "absolute_gain": gain, "relative_gain": gain / zero["E_U"] if finite and zero["E_U"] else None,
                        "direction": "UNAVAILABLE" if gain is None else "IMPROVED" if gain > 0 else "WORSE" if gain < 0 else "UNCHANGED"}
                comparisons.append(item)
        complete = all(x["absolute_gain"] is not None for x in comparisons)
        zero_mean = sum(x["zero_E_U"] for x in comparisons)/6 if complete else None
        target_mean = sum(x["target_E_U"] for x in comparisons)/6 if complete else None
        aggregates[cid(scale)] = {"finite_cells": sum(x["absolute_gain"] is not None for x in comparisons),
            "improved_cells": sum(x["direction"] == "IMPROVED" for x in comparisons),
            "worsened_cells": sum(x["direction"] == "WORSE" for x in comparisons),
            "all_target_eligible": all(x["target_eligible"] for x in comparisons),
            "zero_mean_EU": zero_mean, "target_mean_EU": target_mean,
            "relative_gain": 1 - target_mean / zero_mean if complete and zero_mean else None,
            "absolute_gain": zero_mean - target_mean if complete else None}
        cells.extend(comparisons)
    return {"rows": cells, "aggregates": aggregates, "descriptive_only": True,
            "test_accessed": False, "statistical_significance_tested": False}


def final_decision(rows, baseline_mean):
    scales = sorted({r["response_scale"] for r in rows})
    result = select_rows(rows, scales)
    winner = result["selected"]
    gain = gain_over_sixteen(winner, baseline_mean)
    result["gain_over_lambda16"] = gain
    result["confirmation_triggered"] = bool(winner and winner["response_scale"] > 16 and gain["passes_budget_gate"])
    result["rows"] = rows
    result["ablation_target_scale"] = winner["response_scale"] if result["confirmation_triggered"] else 16.
    return result


def verify_decision_hash(root, name, expected):
    verify_bundle(root / name)
    if sha256_file(root / name / "decision.json") != expected:
        raise RuntimeError(f"{name} decision changed during execution")


def summary(root, selection, coarse, confirmation, rows, controls, contract_hash, ablation):
    result = {"schema": SCHEMA, "selection": selection, "coarse": coarse,
              "confirmation": confirmation, "confirmation_rows": rows, "controls": controls,
              "fisher_ablation": ablation,
              "contract_sha256": contract_hash, "test_accessed": False, "finaltest_authorized": False}
    destination = root / "summary"
    if destination.exists():
        verify_bundle(destination)
        if read_json(destination / "summary.json") != result:
            raise RuntimeError("refusing to replace existing summary")
        return result
    stage = staging(root)
    atomic_json(stage / "summary.json", result)
    all_rows = selection["rows"] + controls + rows
    fields = list(dict.fromkeys(k for row in all_rows for k in row))
    with (stage / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    lines = ["# Stage 14: bounded Fisher-response calibration", "",
             "Development only. Prior exposure is acknowledged. No FinalTest access/authorization.", "",
             f"Coarse action: {coarse['next_action']}; one-time fine grid: {coarse['fine_grid']}",
             f"Final screen winner: {selection['selected']}",
             f"Gain over lambda=16: {selection['gain_over_lambda16']}",
             f"Lambda=128 boundary winner: {selection['boundary_winner']}",
             f"Cross-seed confirmation: {confirmation['status']}", "",
             "Continuation thresholds are engineering budget decisions, not statistical significance.",
             "No inference that empirical Fisher is the correct curvature or that lambda is globally optimal.",
             "Parameter-space diagnostics are in each cell's metrics.json and do not select candidates.", "",
             "| Seed | Request | Lambda | Origin | Status | E_U | Residual | Eligible |",
             "|---|---|---|---|---|---|---|---|"]
    for row in all_rows:
        lines.append(f"| {row['seed']} | {row['scenario']} | {row['response_scale']} | {row['origin']} | {row['status']} | {row.get('E_U')} | {row.get('local_residual')} | {eligible(row)} |")
    lines += ["", "## No-Fisher ablation (descriptive; no significance test)", "",
              "| Seed | Request | Target lambda | Zero E_U | Target E_U | Relative gain | Outcome |",
              "|---|---|---|---|---|---|---|"]
    for row in ablation["rows"]:
        lines.append(f"| {row['seed']} | {row['scenario']} | {row['target_lambda']} | {row['zero_E_U']} | {row['target_E_U']} | {row['relative_gain']} | {row['direction']} |")
    atomic_json(stage / "fisher_ablation.json", ablation)
    atomic_text(stage / "report.md", "\n".join(lines) + "\n")
    publish(stage, destination)
    return result


def execute(repo, workspace, config_path, phase):
    config = load_config(repo, config_path)
    root = workspace / config["output_root"]
    if phase == "Confirm" and not (root / "selection").exists():
        raise RuntimeError("run Sweep/All to freeze final selection before Confirm")
    checks = preflight(repo, workspace, config_path)
    contract_hash = freeze_protocol(repo, workspace, config_path, config, root, checks)
    old = historical_rows(repo, workspace, config)
    baseline = next(c["equal_scenario_mean_EU"] for c in select_rows(old, HISTORICAL_GRID)["candidates"] if c["response_scale"] == 16)
    controls, coarse_rows, additions, screen_executed = [], [], [], []
    for sid in SCENARIOS:
        value = execute_cell(repo, workspace, config, root, contract_hash, 42, sid, (0., 1., *COARSE_GRID), phase="coarse")
        screen_executed.extend(value["rows"])
        coarse_rows += [r for r in value["rows"] if r["response_scale"] in COARSE_GRID]
        additions += [r for r in value["rows"] if r["response_scale"] > 16]
        controls += [dict(r, origin="stage14_coarse_control_or_ablation") for r in value["rows"] if r["response_scale"] in (0., 1., 16.)]
    coarse = coarse_decision(coarse_rows, baseline)
    coarse.update(rows=coarse_rows, contract_sha256=contract_hash, controls=controls.copy(), test_accessed=False)
    coarse_hash = freeze_decision(root, "coarse", coarse)
    print(f"[Stage14] coarse={coarse['next_action']} fine_grid={coarse['fine_grid']}", flush=True)
    if phase == "Coarse":
        return {"status": "COARSE_FROZEN", "decision": coarse["next_action"], "fine_grid": coarse["fine_grid"], "root": str(root), "test_accessed": False}
    if coarse["fine_grid"]:
        for sid in SCENARIOS:
            verify_decision_hash(root, "coarse", coarse_hash)
            value = execute_cell(repo, workspace, config, root, contract_hash, 42, sid,
                                 (1., 16., *coarse["fine_grid"]), coarse_hash, phase="fine")
            new = [r for r in value["rows"] if r["response_scale"] > 16]
            additions.extend(new)
            screen_executed.extend(new)
            controls += [dict(r, origin="stage14_fine_control") for r in value["rows"] if r["response_scale"] in (1., 16.)]
    selection = final_decision(old + additions, baseline)
    selection.update(contract_sha256=contract_hash, coarse_decision_sha256=coarse_hash, controls=controls,
        ablation_seed42_rows=[r for r in screen_executed if r["response_scale"] in (0., 16., selection["ablation_target_scale"])])
    selection_hash = freeze_decision(root, "selection", selection)
    print(f"[Stage14] frozen winner={selection['selected']} new-winner confirmation={selection['confirmation_triggered']}; no-Fisher ablation always runs", flush=True)
    if phase == "Sweep":
        return {"status": "SELECTION_FROZEN", "selected": selection["selected"], "confirmation_triggered": selection["confirmation_triggered"], "ablation_pending": True, "test_accessed": False}
    rows = []
    scales = (0., 1., 16.)
    if selection["confirmation_triggered"]:
        scales += (selection["selected"]["response_scale"],)
    for seed in (41, 43):
        for sid in SCENARIOS:
            verify_decision_hash(root, "selection", selection_hash)
            value = execute_cell(repo, workspace, config, root, contract_hash, seed, sid,
                                 scales, selection_hash, phase="confirm")
            rows.extend(value["rows"])
    if selection["confirmation_triggered"]:
        confirmation = confirmation_decision(rows, selection["selected"]["response_scale"])
    else:
        confirmation = {"status": "NOT_TRIGGERED_RETAIN_STAGE13_REFERENCE", "test_accessed": False, "finaltest_authorized": False}
    ablation = ablation_report(selection["ablation_seed42_rows"] + rows, selection["ablation_target_scale"])
    check_frozen(repo, config, root)
    summary(root, selection, coarse, confirmation, rows, controls, contract_hash, ablation)
    return {"status": "COMPLETED", "selected": selection["selected"], "confirmation": confirmation,
            "gain_over_lambda16": selection["gain_over_lambda16"], "boundary_winner": selection["boundary_winner"],
            "fisher_ablation": ablation["aggregates"],
            "summary": str(root / "summary/report.md"), "test_accessed": False}


def analyze(repo, workspace, config):
    root = workspace / config["output_root"]
    check_frozen(repo, config, root)
    contract_hash = sha256_file(root / "protocol/contract.json")
    for manifest in (root / "cells").rglob("manifest.json"):
        verify_bundle(manifest.parent)
        cell = read_json(manifest.parent / "metrics.json")
        if cell["identity"]["contract_sha256"] != contract_hash:
            raise ValueError("cell contract link mismatch")
        phase = cell["identity"]["phase"]
        if phase in ("fine", "confirm"):
            decision = "coarse" if phase == "fine" else "selection"
            verify_decision_hash(root, decision, cell["identity"]["selection_sha256"])
    verify_bundle(root / "coarse")
    coarse = read_json(root / "coarse/decision.json")
    old = historical_rows(repo, workspace, config)
    baseline = next(c["equal_scenario_mean_EU"] for c in select_rows(old, HISTORICAL_GRID)["candidates"] if c["response_scale"] == 16)
    expected = coarse_decision(coarse["rows"], baseline)
    if any(coarse[k] != v for k, v in expected.items()):
        raise ValueError("coarse decision recomputation mismatch")
    if not (root / "selection").exists():
        return {"status": "COARSE_FROZEN", "decision": coarse["next_action"], "fine_grid": coarse["fine_grid"]}
    verify_bundle(root / "selection")
    selection = read_json(root / "selection/decision.json")
    expected = final_decision(selection["rows"], baseline)
    if any(selection[k] != v for k, v in expected.items()):
        raise ValueError("final selection recomputation mismatch")
    if not (root / "summary").exists():
        return {"status": "SELECTION_FROZEN", "selected": selection["selected"],
                "confirmation_triggered": selection["confirmation_triggered"], "ablation_pending": True}
    verify_bundle(root / "summary")
    result = read_json(root / "summary/summary.json")
    if result["selection"] != selection or result["coarse"] != coarse:
        raise ValueError("summary decision links mismatch")
    if selection["confirmation_triggered"]:
        expected = confirmation_decision(result["confirmation_rows"], selection["selected"]["response_scale"])
        if expected != result["confirmation"]:
            raise ValueError("confirmation recomputation mismatch")
    expected = ablation_report(selection["ablation_seed42_rows"] + result["confirmation_rows"], selection["ablation_target_scale"])
    if expected != result["fisher_ablation"]:
        raise ValueError("ablation recomputation mismatch")
    return {"status": "COMPLETED_VERIFIED", "selected": selection["selected"], "gain_over_lambda16": selection["gain_over_lambda16"],
            "confirmation": result["confirmation"], "boundary_winner": selection["boundary_winner"],
            "fisher_ablation": expected["aggregates"], "test_accessed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ali_stage14_lambda_search_v1.json"))
    parser.add_argument("--mode", choices=("Preflight", "SyntheticDryRun", "Full", "Analyze"), default="Preflight")
    parser.add_argument("--phase", choices=("All", "Coarse", "Sweep", "Confirm"), default="All")
    args = parser.parse_args()
    repo, workspace = args.repo.resolve(), args.workspace.resolve()
    config_path = (repo / args.config).resolve()
    config = load_config(repo, config_path)
    if args.mode == "Preflight":
        checks = preflight(repo, workspace, config_path)
        result = {"status": "PREFLIGHT_PASS", "request_support": checks["request_support"], "model_loaded": False, "test_accessed": False}
    elif args.mode == "SyntheticDryRun":
        rows = [{"candidate_id": cid(scale), "scenario": sid, "response_scale": scale,
                 "E_U": .000001*(abs(scale-32)+1), "utility_eligible": True, "prediction_collapse": False}
                for scale in COARSE_GRID for sid in SCENARIOS]
        result = coarse_decision(rows, .00003)
        assert result["fine_grid"] == [24., 48.]
        result.update(mode="SyntheticDryRun", model_loaded=False)
    elif args.mode == "Full":
        with execution_lock(workspace / config["output_root"]):
            result = execute(repo, workspace, config_path, args.phase)
    else:
        result = analyze(repo, workspace, config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
