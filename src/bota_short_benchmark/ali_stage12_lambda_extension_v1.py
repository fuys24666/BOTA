"""Stage 12: bounded Development-only extension of the Fisher-response grid.

Historical Stage 8/9 artifacts are read-only. Full executes a seed-42 screen,
freezes selection, then conditionally confirms the fixed winner on seeds 41/43.
No FinalTest evaluator is called. Reuse is at atomic seed/scenario boundaries.
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

SCHEMA = "bota-ali-stage12-lambda-extension-v1"
GRID = (0., .25, .5, .75, 1., 1.5, 2.)
NEW_GRID = (1., 1.5, 2.)
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
        raise ValueError("invalid Stage 12 identity or data boundary")
    for key in ("amendment", "stage8_config", "stage9_config", "stage8_audit"):
        binding = config[key]
        if sha256_file(repo / binding["path"]) != binding["sha256"]:
            raise ValueError(f"frozen binding changed: {key}")
    if config["protocol"] != {
        "selection_seed": 42, "confirmation_seeds": [41, 43],
        "scenarios": list(SCENARIOS), "new_grid": list(NEW_GRID), "merged_grid": list(GRID),
        "optimizer_steps": 200, "split": "Development/Dev-Select", "final_test": False,
        "source_scaled": False, "endpoint_interpolation": False,
        "reproduction_atol": 1e-8, "reproduction_rtol": .001,
    }:
        raise ValueError("Stage 12 protocol changed; use a new amendment/version")
    if config["output_root"] != "outputs/ali_stage12_lambda_extension_v1":
        raise ValueError("unexpected output root")
    amendment = read_json(repo / config["amendment"]["path"])
    if amendment["status"] != "SPECIFIED_BEFORE_STAGE12_EXECUTION":
        raise ValueError("amendment status mismatch")
    return config


def source_map(repo, config_path):
    # Record uncommitted implementation too; no need to force a Git commit.
    paths = list((repo / "src").rglob("*.py"))
    paths += list((repo / "configs").glob("*"))
    paths += list((repo / "docs" / "audits").glob("*.json"))
    paths += [repo / "scripts/bota_if/run_ali_stage12_lambda_extension_v1.ps1", config_path]
    return {str(p.relative_to(repo)).replace("\\", "/"): sha256_file(p)
            for p in sorted(set(paths)) if p.is_file()}


def verify_bundle(path):
    if (path / "COMPLETED").read_text(encoding="utf-8") != SCHEMA + "\n":
        raise ValueError(f"incomplete Stage 12 artifact: {path}")
    manifest = read_json(path / "manifest.json")
    actual = {str(p.relative_to(path)).replace("\\", "/") for p in path.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if actual != set(manifest["files"]) or manifest.get("schema") != SCHEMA:
        raise ValueError(f"Stage 12 artifact inventory changed: {path}")
    for name, expected in manifest["files"].items():
        if sha256_file(path / name) != expected:
            raise ValueError(f"Stage 12 artifact changed: {path / name}")
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
                raise RuntimeError("another Stage 12 Full process holds the output lock") from error
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


def historical_rows(repo, workspace, config):
    oldconfig = s8.load_config(repo, repo / config["stage8_config"]["path"])
    oldrun = workspace / oldconfig["output_root"] / "runs/ali_stage8_bota_response_seed42_v1"
    audit = read_json(repo / config["stage8_audit"]["path"])
    for name, expected in audit["source_sha256"].items():
        if sha256_file(oldrun / name) != expected:
            raise ValueError(f"Stage 8 frozen source changed: {name}")
    s8.analyze(workspace, oldconfig, oldrun.name)
    rows = read_json(oldrun / "candidate_metrics.json")["rows"]
    for row in rows:
        row.update(seed=42, status="OK", origin="frozen_stage8")
    return rows


def preflight(repo, workspace, config_path):
    config = load_config(repo, config_path)
    base = s8.load_config(repo, repo / config["stage8_config"]["path"])
    cross = s9.load_config(repo, workspace, repo / config["stage9_config"]["path"])
    old = historical_rows(repo, workspace, config)
    select_rows(old, s8.SCALES)
    support, authorities, input_hashes = {}, {}, {}
    for seed in (42, 41, 43):
        seedconfig = copy.deepcopy(base)
        seedconfig["benchmark"] = cross["benchmarks"][seed]
        binding = seedconfig["benchmark"]
        benchmark = s8.stage7.load_benchmark_config(repo / binding["config"]["path"])
        benchmark["_config_path"] = str(repo / binding["config"]["path"])
        _, _, registry = s8.stage7.validate_prepared(workspace, benchmark, binding["name"])
        _, users, _, select = s8.stage7._development_partition(repo, workspace, seedconfig, benchmark)
        for field in ("original_checkpoint", "train_json", "development_json", "train_user_ids", "base_config", "trajectory_config"):
            if field in benchmark["source"]:
                file = workspace / benchmark["source"][field]
                if str(file) not in input_hashes:
                    input_hashes[str(file)] = sha256_file(file)
        # These runtime configs live in the experiment workspace, not the repo.
        runtime_config = s8.yaml.safe_load((workspace / benchmark["source"]["base_config"]).read_text(encoding="utf-8"))
        t5_path = workspace / runtime_config["base_config"]
        if str(t5_path) not in input_hashes:
            input_hashes[str(t5_path)] = sha256_file(t5_path)
        for sid in SCENARIOS:
            request = s8.behavior._scenario(registry, sid)
            count = sum(users[i] in set(request["users"]) for i in select)
            if count == 0:
                raise ValueError(f"empty requested-user Dev-Select support: {seed}/{sid}")
            support[f"{seed}/{sid}"] = count
            authorities[f"{seed}/{sid}"] = s8.stage1.preflight(
                repo, workspace, repo / base["stage1_config"]["path"], seed, sid)[0]
            authority = authorities[f"{seed}/{sid}"]
            model_dir = Path(authority["local_model_dir"])
            files = [model_dir / name for name in ("config.json", "pytorch_model.bin", "spiece.model", "tokenizer_config.json")]
            for reference in authority["references"].values():
                adapter = Path(reference["run"]) / "scenarios" / sid / "adapter"
                files.extend([adapter / "adapter_model.pt", adapter / "adapter_config.json"])
            for file in files:
                if str(file) not in input_hashes:
                    input_hashes[str(file)] = sha256_file(file)
    return {"schema": SCHEMA, "request_support": support, "authorities": authorities,
            "input_sha256": input_hashes, "model_loaded": False, "test_accessed": False}


def eligible(row):
    return (row.get("status", "OK") == "OK" and row.get("utility_eligible") is True
            and row.get("prediction_collapse") is False
            and isinstance(row.get("E_U"), (float, int)) and math.isfinite(row["E_U"]))


def select_rows(rows, scales=GRID):
    expected = {(cid(scale), sid) for scale in scales for sid in SCENARIOS}
    keys = [(r["candidate_id"], r["scenario"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("missing, duplicate or unexpected candidate/scenario cells")
    candidates = []
    for scale in scales:
        cells = [r for r in rows if r["candidate_id"] == cid(scale)]
        valid = all(eligible(r) for r in cells)
        values = [r.get("E_U") for r in cells]
        mean = sum(values) / len(values) if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) else None
        candidates.append({"candidate_id": cid(scale), "response_scale": scale,
                           "eligible": valid, "equal_scenario_mean_EU": mean,
                           "failures": [r["scenario"] + ":" + r.get("status", "utility_or_collapse")
                                        for r in cells if not eligible(r)]})
    valid = [c for c in candidates if c["eligible"]]
    winner = min(valid, key=lambda c: (c["equal_scenario_mean_EU"], abs(c["response_scale"] - 1), c["candidate_id"])) if valid else None
    return {"status": "SELECTED" if winner else "NO_ELIGIBLE_CALIBRATION", "selected": winner,
            "candidates": candidates, "boundary_winner": bool(winner and winner["response_scale"] == 2),
            "test_accessed": False, "finaltest_authorized": False}


def check_reproduction(new, old, protocol):
    differences = {}
    for key in ("E_U", "D_U", "overall_auc", "overall_log_loss"):
        delta = abs(new[key] - old[key])
        tolerance = max(protocol["reproduction_atol"], protocol["reproduction_rtol"] * abs(old[key]))
        if not math.isfinite(delta) or delta > tolerance:
            raise RuntimeError(f"lambda=1 historical reproduction failed: {key}, diff={delta}, tolerance={tolerance}")
        differences[key] = {"absolute_difference": delta, "tolerance": tolerance}
    return differences


def run_transport(model, dataset, indices, user_ids, selected_users, parameters, names,
                  device, pad, engine, budget, scales, progress):
    optimizer = s8.p1._optimizer(parameters, engine)
    states = {cid(scale): s8.stage1._state(parameters, len(selected_users)) for scale in scales}
    failures = {}
    batch_size = int(engine["schedule"]["batch_size"])
    if len(indices) != 200 * batch_size:
        raise ValueError("Stage 12 requires the original 200-step trajectory")
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
                if str(error) not in NUMERICAL_ERRORS or scale == 1:
                    raise
                failures[key] = {"status": "NUMERICAL_TRANSPORT_FAILURE", "reason": str(error), "first_failed_step": step}
                del states[key]
                continue
            if not all(torch.isfinite(value).all().item() for state in states[key] for value in state.values()):
                if scale == 1:
                    raise RuntimeError("nonfinite lambda=1 control; reproduction blocked")
                failures[key] = {"status": "NONFINITE_TRANSPORT", "first_failed_step": step}
                del states[key]
        budget.step(optimizer, "shared_canonical_reference")
        if step == 1 or step % 10 == 0:
            atomic_json(progress, {"step": step, "of": 200, "numerical_failures": failures, "test_accessed": False})
            print(f"[Stage12] transport step={step}/200 failures={list(failures)}", flush=True)
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


def execute_cell(repo, workspace, config, root, contract_hash, seed, sid, scales, selection_hash=None):
    destination = root / "cells" / f"s{seed}_{sid}"
    identity = {"contract_sha256": contract_hash, "seed": seed, "scenario": sid,
                "scales": list(scales), "selection_sha256": selection_hash}
    if destination.exists():
        verify_bundle(destination)
        result = read_json(destination / "metrics.json")
        if result["identity"] != identity:
            raise ValueError("existing cell belongs to a different protocol or selection")
        print(f"[Stage12] verified reuse {seed}/{sid}", flush=True)
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
    print(f"[Stage12] seed={seed} scenario={sid} scales={list(scales)} work={stage}", flush=True)
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
        publication, adapter_hashes, scale_one = {}, {}, None
        for scale in scales:
            key = cid(scale)
            if key in failures:
                continue
            candidate = {name: canonical[name] + states[key][i]["theta"].sum(0).float().cpu() for i, name in enumerate(names)}
            if not all(torch.isfinite(value).all().item() for value in candidate.values()):
                if scale == 1:
                    raise RuntimeError("nonfinite lambda=1 adapter")
                failures[key] = {"status": "NONFINITE_ADAPTER"}
                continue
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
            print(f"[Stage12] {seed}/{sid} evaluating {label} on Dev-Select", flush=True)
            predictions[label] = s8.behavior._predict_reference(workspace, benchmark, run, sid, development, select, device)
            if not validate_prediction(predictions[label], select):
                raise RuntimeError(f"nonfinite reference predictions: {label}")
        rows = []
        for scale in scales:
            key = cid(scale)
            if key not in failures:
                print(f"[Stage12] {seed}/{sid} evaluating lambda={scale:g}", flush=True)
                prediction = s8.behavior._predict_scenario_dir(workspace, benchmark, stage / "adapters" / key, development, select, device)
                if not validate_prediction(prediction, select):
                    if scale == 1:
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
            row.update(seed=seed, origin="stage12", test_accessed=False)
            rows.append(row)
        reproduction = {}
        if seed == 42:
            oldrow = next(r for r in historical_rows(repo, workspace, config) if r["candidate_id"] == cid(1) and r["scenario"] == sid)
            reproduction = check_reproduction(next(r for r in rows if r["candidate_id"] == cid(1)), oldrow, config["protocol"])
        peak = torch.cuda.max_memory_reserved() / 2**30
        if peak > base["runtime"]["hard_peak_reserved_gib"]:
            raise RuntimeError(f"GPU reservation exceeded protocol cap: {peak:.3f} GiB")
        result = {"schema": SCHEMA, "identity": identity, "rows": rows,
                  "formal_reproduction": alignment, "historical_reproduction": reproduction,
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
        print(f"[Stage12] incomplete diagnostics preserved: {stage}", flush=True)
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


def freeze_selection(root, rows, controls, contract_hash):
    result = select_rows(rows)
    result.update(schema=SCHEMA, contract_sha256=contract_hash, rows=rows, controls=controls)
    destination = root / "selection"
    if destination.exists():
        verify_bundle(destination)
        if read_json(destination / "selection.json") != result:
            raise RuntimeError("frozen selection cannot be replaced")
    else:
        stage = staging(root)
        atomic_json(stage / "selection.json", result)
        publish(stage, destination)
    return result, sha256_file(destination / "selection.json")


def confirmation_decision(rows, scale):
    decisions, regressions = {}, []
    for seed in (41, 43):
        cells = [r for r in rows if r["seed"] == seed]
        select_rows(cells, (1., scale))  # Enforce all cells, no silent omission.
        selected = [r for r in cells if r["response_scale"] == scale]
        control = [r for r in cells if r["response_scale"] == 1]
        valid = all(eligible(r) and r.get("local_residual") is not None and r["local_residual"] < 1 for r in selected)
        control_valid = all(eligible(r) for r in control)
        mean = sum(r["E_U"] for r in selected) / 2 if all(r.get("E_U") is not None for r in selected) else None
        base = sum(r["E_U"] for r in control) / 2
        for row in selected:
            ref = next(r for r in control if r["scenario"] == row["scenario"])
            if row.get("E_U") is not None and row["E_U"] > ref["E_U"]:
                regressions.append({"seed": seed, "scenario": row["scenario"], "EU_increase": row["E_U"] - ref["E_U"]})
        decisions[str(seed)] = {"candidate_mean_EU": mean, "control_mean_EU": base,
                               "passed": bool(valid and control_valid and mean is not None and mean < base)}
    return {"status": "CROSS_SEED_DEVELOPMENT_CONFIRMED" if all(d["passed"] for d in decisions.values()) else "NOT_STABLE",
            "seeds": decisions, "cell_regressions": regressions, "finaltest_authorized": False, "test_accessed": False}


def write_summary(root, selection, decision, confirmation):
    result = {"schema": SCHEMA, "selection": selection, "confirmation": decision,
              "confirmation_rows": confirmation, "test_accessed": False}
    destination = root / "summary"
    if destination.exists():
        verify_bundle(destination)
        if read_json(destination / "summary.json") != result:
            raise RuntimeError("refusing to replace existing summary")
        return result
    stage = staging(root)
    atomic_json(stage / "summary.json", result)
    rows = selection["rows"] + selection["controls"] + confirmation
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (stage / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ["# Stage 12: Development lambda extension", "", "No FinalTest access or authorization.", "",
             f"Selection: {selection['selected']}", f"Confirmation: {decision['status']}",
             f"Boundary optimum at lambda=2: {selection['boundary_winner']}", "",
             "Historical Stage 8 rows are preserved. Rerun lambda=1 is a control, not a duplicate candidate.",
             "This is a reviewer-motivated extension with prior Development exposure, not an untouched holdout.", "",
             "| Seed | Request | Lambda | Origin | Status | E_U | Residual | Eligible |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['scenario']} | {row['response_scale']} | {row['origin']} | {row['status']} | {row.get('E_U')} | {row.get('local_residual')} | {eligible(row)} |")
    atomic_text(stage / "report.md", "\n".join(lines) + "\n")
    publish(stage, destination)
    return result


def execute(repo, workspace, config_path, phase):
    config = load_config(repo, config_path)
    checks = preflight(repo, workspace, config_path)
    root = workspace / config["output_root"]
    contract_hash = freeze_protocol(repo, workspace, config_path, config, root, checks)
    old = historical_rows(repo, workspace, config)
    rows, controls = list(old), []
    for sid in SCENARIOS:
        if phase == "Confirm" and not (root / "cells" / f"s42_{sid}").exists():
            raise RuntimeError("run Sweep before Confirm")
        result = execute_cell(repo, workspace, config, root, contract_hash, 42, sid, NEW_GRID)
        controls += [r for r in result["rows"] if r["response_scale"] == 1]
        rows += [r for r in result["rows"] if r["response_scale"] > 1]
    selection, selection_hash = freeze_selection(root, rows, controls, contract_hash)
    print(f"[Stage12] frozen selection={selection['selected']}", flush=True)
    if phase == "Sweep":
        return {"status": "SWEEP_FROZEN", "selection": selection["selected"], "root": str(root), "test_accessed": False}
    winner = selection["selected"]
    confirmation = []
    if winner and winner["response_scale"] > 1:
        for seed in (41, 43):
            for sid in SCENARIOS:
                verify_bundle(root / "selection")
                if sha256_file(root / "selection/selection.json") != selection_hash:
                    raise RuntimeError("selection changed during confirmation")
                result = execute_cell(repo, workspace, config, root, contract_hash, seed, sid, (1., winner["response_scale"]), selection_hash)
                confirmation.extend(result["rows"])
        decision = confirmation_decision(confirmation, winner["response_scale"])
    else:
        decision = {"status": "NOT_TRIGGERED_NO_EXTENDED_WINNER", "finaltest_authorized": False, "test_accessed": False}
    check_frozen(repo, config, root)
    write_summary(root, selection, decision, confirmation)
    return {"status": "COMPLETED", "selected": winner, "confirmation": decision,
            "boundary_winner": selection["boundary_winner"], "summary": str(root / "summary/report.md"), "test_accessed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ali_stage12_lambda_extension_v1.json"))
    parser.add_argument("--mode", choices=("Preflight", "SyntheticDryRun", "Full", "Analyze"), default="Preflight")
    parser.add_argument("--phase", choices=("All", "Sweep", "Confirm"), default="All")
    args = parser.parse_args()
    repo, workspace = args.repo.resolve(), args.workspace.resolve()
    config_path = (repo / args.config).resolve()
    config = load_config(repo, config_path)
    if args.mode == "Preflight":
        result = preflight(repo, workspace, config_path)
        result = {"status": "PREFLIGHT_PASS", "request_support": result["request_support"], "model_loaded": False, "test_accessed": False}
    elif args.mode == "SyntheticDryRun":
        rows = [{"candidate_id": cid(s), "scenario": sid, "E_U": abs(s - 1.5) + .01,
                 "utility_eligible": True, "prediction_collapse": False} for s in GRID for sid in SCENARIOS]
        result = select_rows(rows)
        assert result["selected"]["response_scale"] == 1.5
    elif args.mode == "Full":
        with execution_lock(workspace / config["output_root"]):
            result = execute(repo, workspace, config_path, args.phase)
    else:
        root = workspace / config["output_root"]
        check_frozen(repo, config, root)
        for path in sorted((root / "cells").glob("s*")):
            verify_bundle(path)
        verify_bundle(root / "selection")
        if (root / "summary").exists():
            verify_bundle(root / "summary")
            result = read_json(root / "summary/summary.json")
        else:
            result = {"status": "SWEEP_FROZEN_CONFIRMATION_NOT_COMPLETED", "selection": read_json(root / "selection/selection.json")}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
