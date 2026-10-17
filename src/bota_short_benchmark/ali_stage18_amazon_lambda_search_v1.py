"""Bounded Amazon-only FS/Fisher calibration; never evaluates FinalTest.

One shared seed42 selection, one optional refinement, two confirmation seeds.
Each bank, prediction and decision is immutable and hash-verified on resume.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import platform
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml

from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, sha256_file, seed_everything
from . import ali_stage8_bota_response_sweep_v1 as s8
from . import ali_stage14_lambda_search_v1 as s14
from . import evaluation, protocol, runner

SCHEMA = "bota-ali-stage18-amazon-lambda-search-v1"
GRID = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
SCENARIOS = ["K2", "K4"]
RULES = {
    "selection_seed": 42, "confirmation_seeds": [41, 43], "scenarios": SCENARIOS,
    "coarse_grid": GRID, "maximum_lambda": 256, "fine_rounds": 1,
    "absolute_gain_minimum": 1e-6, "relative_gain_minimum": .01,
    "optimizer_steps": 200, "batch_size": 16, "evaluation_split": "Development",
    "source_scaled": False, "endpoint_interpolation": False,
    "confirmation_controls": [0, 1, 64], "confirmation_changes_selection": False,
    "same_historical_checkpoint_and_order_across_seeds": True,
}
SELECT = {"primary": "equal_scenario_mean_EU", "auc_damage_max": .005,
          "logloss_damage_max": .01, "tie_break": "smaller_lambda",
          "no_eligible_policy": "NO_ELIGIBLE_CANDIDATE"}
read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
cid = s8._candidate_id


def load_config(repo, path):
    c = read(path)
    if c["schema"] != SCHEMA or c["test_access_policy"] != "forbidden":
        raise ValueError("invalid Stage18 boundary")
    if c["protocol"] != RULES or c["selection"] != SELECT:
        raise ValueError("changing frozen search rules requires a new version")
    if c["output_root"] != "outputs/ali_stage18_amazon_lambda_search_v1":
        raise ValueError("invalid separate output root")
    a = read(repo / c["amendment"])
    if a["status"] != "SPECIFIED_BEFORE_STAGE18_EXECUTION" or a["test_accessed"] is not False:
        raise ValueError("invalid Amendment17")
    return c


def stage_dir(root):
    p = root / ".work" / uuid.uuid4().hex[:10]
    p.mkdir(parents=True)
    return p


def verify(path):
    m = read(path / "manifest.json")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*")
              if p.is_file() and p != path / "manifest.json"}
    if m["schema"] != SCHEMA or actual != set(m["files"]):
        raise ValueError(f"artifact inventory mismatch: {path}")
    if (path / "COMPLETED").read_text(encoding="utf-8") != SCHEMA + "\n":
        raise ValueError(f"incomplete artifact: {path}")
    for name, expected in m["files"].items():
        if sha256_file(path / name) != expected:
            raise ValueError(f"artifact changed: {path / name}")
    return m


def publish(stage, destination):
    if destination.exists():
        raise FileExistsError(destination)
    atomic_text(stage / "COMPLETED", SCHEMA + "\n")
    atomic_json(stage / "manifest.json", {
        "schema": SCHEMA, "test_accessed": False,
        "files": {p.relative_to(stage).as_posix(): sha256_file(p)
                  for p in sorted(stage.rglob("*")) if p.is_file()}})
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(stage, destination)


def freeze_json(root, name, payload):
    dest = root / name
    if dest.exists():
        verify(dest)
        if read(dest / "result.json") != payload:
            raise ValueError(f"refusing to change frozen {name}")
    else:
        stage = stage_dir(root)
        atomic_json(stage / "result.json", payload)
        publish(stage, dest)
    return sha256_file(dest / "result.json")


def audit(repo, workspace, c, config_path):
    # The workspace copy is the historical authority (its hash is in the run
    # contracts). The repo copy renamed a legacy runtime-config alias later.
    # Never substitute that edited file and silently bypass contract checking.
    b = protocol.load_config(workspace / c["benchmark_config"])
    current = protocol.load_config(repo / c["benchmark_config"])
    comparable = copy.deepcopy(b)
    comparable["source"]["base_config"] = current["source"]["base_config"]
    for value in (comparable, current):
        for key in list(value):
            if key.startswith("_"):
                del value[key]
    if comparable != current:
        raise ValueError("repo and historical Amazon config differ beyond the documented runtime alias")
    prepared, historical, reg = protocol.validate_prepared(workspace, b, c["benchmark_name"])
    for field, binding in historical["sources"].items():
        if b["source"][field] != binding["path"] or protocol.source_sha256(workspace / binding["path"]) != binding["sha256"]:
            raise ValueError(f"historical source binding changed: {field}")
    if {x["id"] for x in reg["scenarios"]} != set(SCENARIOS) or len(reg["order"]) != 3200:
        raise ValueError("Amazon request registry/budget mismatch")
    if reg["selection_uses_labels_or_predictions"] or reg["test_accessed"]:
        raise ValueError("invalid source registry boundary")
    du = read(workspace / b["source"]["development_user_ids"])
    tr = read(workspace / b["source"]["train_user_ids"])
    if len(du) != reg["development_samples"] or len(tr) != reg["global_train_samples"]:
        raise ValueError("user sidecars mismatch registry")
    support = {r["id"]: sum(int(u) in r["users"] for u in du) for r in reg["scenarios"]}
    if support != {"K2": 10, "K4": 20}:
        raise ValueError(f"unexpected frozen request support: {support}")
    if len(du) != 3840 or not set(reg["order"]) <= set(range(len(tr))):
        raise ValueError("invalid data/order")
    for r in reg["scenarios"]:
        positions = [i for i, idx in enumerate(reg["order"]) if tr[idx] in r["users"]]
        if positions != r["forget_window_positions"]:
            raise ValueError("deletion slots do not match source sidecar")
    sources = list((repo / "src").rglob("*.py")) + [config_path, repo / c["benchmark_config"],
        repo / c["amendment"], repo / "scripts/bota_if/run_ali_stage18_amazon_lambda_search_v1.ps1"]
    inputs = list(prepared.rglob("*")) + [workspace / c["benchmark_config"]]
    for field in ("train_json", "development_json", "train_user_ids", "development_user_ids", "raw_lineage_manifest", "base_config"):
        inputs.append(workspace / b["source"][field])
    runtime = yaml.safe_load((workspace / b["source"]["base_config"]).read_text(encoding="utf-8"))
    inputs.append(workspace / runtime["base_config"])
    inputs += list((workspace / b["source"]["original_checkpoint"]).rglob("*"))
    model_dir = Path(runner._base_t5_config(workspace, b)["paths"]["model_dir"])
    if not model_dir.is_absolute():
        model_dir = workspace / model_dir
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    inputs += list(model_dir.rglob("*"))
    refs = {}
    for method, name in c["references"].items():
        run = evaluation._validate_method(workspace, b, method, name, c["benchmark_name"])
        refs[method] = str(run.resolve())
        inputs += list(run.rglob("*"))
    print("[Stage18] hashing Amazon assets and source (no model execution)", flush=True)
    checks = {"schema": SCHEMA, "request_support": support, "development_samples": len(du),
              "references": refs, "registry_sha256": canonical_hash(reg),
              "input_sha256": {str(p.resolve()): sha256_file(p) for p in sorted(set(inputs)) if p.is_file()},
              "source_sha256": {p.relative_to(repo).as_posix(): sha256_file(p) for p in sorted(set(sources))},
              "test_accessed": False}
    return checks, b, reg, list(map(int, du))


def choose(rows, scales):
    candidates = []
    for scale in sorted(set(scales)):
        cells = [r for r in rows if r["response_scale"] == scale]
        if len(cells) != 2 or {r["scenario"] for r in cells} != set(SCENARIOS):
            raise ValueError(f"incomplete/duplicate candidate {scale}")
        eligible = all(r["status"] == "OK" and r["utility_eligible"] and not r["prediction_collapse"]
                       and r["E_U"] is not None and np.isfinite(r["E_U"]) for r in cells)
        mean = float(np.mean([r["E_U"] for r in cells])) if all(r["E_U"] is not None for r in cells) else None
        candidates.append({"lambda": scale, "eligible": eligible, "mean_E_U": mean})
    valid = [r for r in candidates if r["eligible"]]
    best = min(valid, key=lambda r: (r["mean_E_U"], r["lambda"])) if valid else None
    return {"status": "SELECTED" if best else "NO_ELIGIBLE_CANDIDATE", "selected": best,
            "candidates": candidates, "test_accessed": False}


def fine_grid(decision):
    best = decision["selected"]
    if not best or best["lambda"] in (0, GRID[-1]):
        return []
    zero = next(r for r in decision["candidates"] if r["lambda"] == 0)
    if not zero["eligible"] or zero["mean_E_U"] <= 0:
        return []
    gain = zero["mean_E_U"] - best["mean_E_U"]
    if gain < 1e-6 or gain / zero["mean_E_U"] < .01:
        return []
    i = GRID.index(best["lambda"])
    return [(GRID[i-1] + GRID[i]) / 2, (GRID[i] + GRID[i+1]) / 2]


def confirmation(rows, scale):
    details = {}
    for seed in (41, 43):
        seed_rows = [r for r in rows if r["seed"] == seed]
        d = choose(seed_rows, sorted({0, 1, 64, scale}))
        zero = next(x for x in d["candidates"] if x["lambda"] == 0)
        best = next(x for x in d["candidates"] if x["lambda"] == scale)
        gain = zero["mean_E_U"] - best["mean_E_U"] if zero["mean_E_U"] is not None and best["mean_E_U"] is not None else None
        rel = gain / zero["mean_E_U"] if gain is not None and zero["mean_E_U"] > 0 else None
        regressions = []
        for sid in SCENARIOS:
            a = next(x for x in seed_rows if x["scenario"] == sid and x["response_scale"] == scale)
            z = next(x for x in seed_rows if x["scenario"] == sid and x["response_scale"] == 0)
            if a["E_U"] is None or z["E_U"] is None or a["E_U"] > z["E_U"] + 1e-6:
                regressions.append(sid)
        passed = best["eligible"] and zero["eligible"] and gain is not None and gain >= 1e-6 and rel is not None and rel >= .01 and not regressions
        details[str(seed)] = {"passed": bool(passed), "absolute_gain_over_zero": gain,
                              "relative_gain_over_zero": rel, "regressed_requests": regressions,
                              "selected_candidate": best, "zero": zero}
    return {"status": "CONFIRMED" if all(x["passed"] for x in details.values()) else "NOT_CONFIRMED",
            "selected_lambda_unchanged": scale, "per_seed": details, "test_accessed": False}


def transport(model, dataset, order, users, targets, params, names, device, pad, engine, scales, progress):
    opt = s8.p1._optimizer(params, engine)
    states = {cid(x): s8.stage1._state(params, len(targets)) for x in scales}
    failed = {}
    budget = s8.StepBudget(len(order) // 16)
    for step, start in enumerate(range(0, len(order), 16), 1):
        ids = order[start:start+16]
        batch_users = [int(users[i]) for i in ids]
        batch = s8.move_batch(s8.masked_batch(dataset, ids, pad), device)
        losses = s8.p1._sample_losses(model, batch)
        rows = [[] for _ in params]
        for sample in range(len(ids)):
            gradients = torch.autograd.grad(losses[sample], params, retain_graph=True)
            for j, grad in enumerate(gradients):
                rows[j].append(grad.detach())
        opt.zero_grad(set_to_none=True)
        loss = losses.sum() / len(ids)
        loss.backward()
        coefficients = [torch.stack(x) for x in rows]
        sources = []
        for coef, p in zip(coefficients, params):
            src = []
            for u in targets:
                slots = [i for i, bu in enumerate(batch_users) if bu == u]
                src.append(-coef[slots].sum(0) / len(ids) if slots else torch.zeros_like(p))
            sources.append(torch.stack(src))
        for scale in scales:
            key = cid(scale)
            if key in failed:
                continue
            try:
                s8._advance_scaled(states[key], coefficients, sources, params, opt, step, engine, scale)
                if not all(torch.isfinite(v).all().item() for st in states[key] for v in st.values()):
                    raise FloatingPointError("nonfinite transport state")
            except (RuntimeError, FloatingPointError) as e:
                if isinstance(e, RuntimeError) and str(e) not in s14.NUMERICAL_ERRORS:
                    raise
                if scale in (0, 1):
                    raise RuntimeError(f"reference-control transport failed at lambda={scale}") from e
                failed[key] = {"status": "NUMERICAL_TRANSPORT_FAILURE", "first_failed_step": step, "reason": str(e)}
                del states[key]
        budget.step(opt, "shared_canonical_reference")
        if step == 1 or step % 10 == 0 or step == len(order) // 16:
            atomic_json(progress, {"step": step, "of": len(order)//16, "failures": failed, "test_accessed": False})
            print(f"[Stage18] transport {step}/{len(order)//16} failed={list(failed)}", flush=True)
        del batch, losses, loss, rows, coefficients, sources, gradients
    canonical = {name: p.detach().float().cpu().clone() for name, p in zip(names, params)}
    vectors = {key: torch.cat([st["theta"].detach().float().cpu().reshape(len(targets), -1) for st in value], dim=1)
               for key, value in states.items()}
    return canonical, vectors, failed, budget.calls


def compose(canonical, names, delta):
    values, start = {}, 0
    for name in names:
        p = canonical[name]
        values[name] = p + delta[start:start+p.numel()].reshape(p.shape)
        start += p.numel()
    if start != delta.numel():
        raise ValueError("coordinate dimension mismatch")
    return values


def gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Stage18 requires exactly one CUDA device")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free / 2**30 < 10:
        raise RuntimeError("at least 10 GiB free GPU memory required")
    torch.cuda.set_per_process_memory_fraction(runner.allocator_fraction_for(total/2**30, .88, 14), device)
    return device


def release():
    gc.collect()
    torch.cuda.empty_cache()


def build_bank(root, workspace, b, reg, checks, seed, scales, phase, contract):
    dest = root / "banks" / f"{phase}_{seed}"
    identity = {"seed": seed, "scales": scales, "phase": phase, "contract": contract}
    if dest.exists():
        verify(dest)
        if read(dest / "result.json")["identity"] != identity:
            raise ValueError("bank identity mismatch")
        print(f"[Stage18] verified bank {phase}/{seed}", flush=True)
        return dest
    stage = stage_dir(root)
    started = time.perf_counter()
    device = gpu()
    torch.cuda.reset_peak_memory_stats()
    config = copy.deepcopy(b)
    config["protocol"]["run_seed"] = seed
    config["coordinate"]["fixed_a_seed"] = seed
    fixed = None
    if seed == 42:
        original = torch.load(Path(checks["references"]["Original-Short"]) / "scenarios/K2/adapter/adapter_model.pt", map_location="cpu", weights_only=True)
        fixed = original["A"]
    seed_everything(seed)
    try:
        model, names, params, bases, _, tok, dataset, users = runner._model_context(workspace, config, reg, device, fixed)
        targets = sorted({int(u) for r in reg["scenarios"] for u in r["users"]})
        canonical, vectors, failed, steps = transport(model, dataset, reg["order"], users, targets,
            params, names, device, tok.pad_token_id, runner._engine(workspace, config), scales, stage / "progress.json")
        if steps != 200:
            raise RuntimeError("canonical optimizer budget mismatch")
        if seed == 42 and any(not torch.equal(canonical[n], original["B"][n]) for n in names):
            raise RuntimeError("Amazon canonical endpoint does not reproduce historical Original")
        runner._save_fixed_ab(stage / "Original/adapter", names, canonical, bases, "Stage18-Original")
        alignment = {}
        for r in reg["scenarios"]:
            slots = [targets.index(int(u)) for u in r["users"]]
            for scale in scales:
                key = cid(scale)
                if key in failed:
                    continue
                values = compose(canonical, names, vectors[key][slots].sum(0))
                if not all(torch.isfinite(x).all().item() for x in values.values()):
                    raise RuntimeError("nonfinite composed endpoint")
                runner._save_fixed_ab(stage / r["id"] / key / "adapter", names, values, bases, f"Stage18-FS-lambda{scale:g}")
                if seed == 42 and scale == 1:
                    old = torch.load(Path(checks["references"]["BOTA-T2-Short"]) / "scenarios" / r["id"] / "adapter/adapter_model.pt", map_location="cpu", weights_only=True)
                    expected = torch.cat([(old["B"][n] - canonical[n]).flatten() for n in names])
                    actual = torch.cat([(values[n] - canonical[n]).flatten() for n in names])
                    metric = s8.p2b.vector_metrics(expected, actual)
                    if not (metric["finite"] and metric["cosine"] >= .9999 and metric["relative_l2_error"] <= .01 and metric["maximum_absolute_error"] <= .001):
                        raise RuntimeError(f"historical lambda1 reproduction failed: {metric}")
                    alignment[r["id"]] = metric
        del model, params, bases, vectors, dataset, tok
        release()
        peak = torch.cuda.max_memory_reserved() / 2**30
        if peak > 14:
            raise RuntimeError("GPU memory cap exceeded")
        atomic_json(stage / "result.json", {"identity": identity, "failures": failed, "formal_reproduction": alignment,
            "physical_optimizer_steps": steps, "peak_reserved_gib": peak, "seconds": time.perf_counter()-started, "test_accessed": False})
        publish(stage, dest)
    except Exception as e:
        atomic_json(stage / "FAILED.json", {"error": repr(e), "identity": identity, "test_accessed": False})
        print(f"[Stage18] incomplete bank retained at {stage}", flush=True)
        raise
    return dest


def exact_reference(root, workspace, b, reg, checks, seed, sid, bank, contract):
    if seed == 42:
        return Path(checks["references"]["Retrain-Short"]) / "scenarios" / sid
    dest = root / "exact" / f"s{seed}_{sid}"
    identity = {"seed": seed, "scenario": sid, "contract": contract}
    if dest.exists():
        verify(dest)
        if read(dest / "result.json")["identity"] != identity:
            raise ValueError("Exact reference identity changed")
        return dest
    stage = stage_dir(root)
    device = gpu()
    config = copy.deepcopy(b)
    config["protocol"]["run_seed"] = seed
    config["coordinate"]["fixed_a_seed"] = seed
    state = torch.load(bank / "Original/adapter/adapter_model.pt", map_location="cpu", weights_only=True)
    seed_everything(seed)
    started = time.perf_counter()
    print(f"[Stage18] Exact-Masked seed={seed} request={sid} 200 steps", flush=True)
    model, names, params, bases, _, tok, dataset, users = runner._model_context(workspace, config, reg, device, state["A"])
    budget = s8.StepBudget(200)
    targets = next(r["users"] for r in reg["scenarios"] if r["id"] == sid)
    values, opt, trace = runner._train_named(model, dataset, reg["order"], users, targets, names,
        params, device, tok.pad_token_id, runner._engine(workspace, config), budget, "Exact-Masked")
    if budget.calls != 200:
        raise RuntimeError("Exact reference optimizer budget mismatch")
    runner._save_fixed_ab(stage / "adapter", names, values, bases, "Stage18-Exact-Masked")
    atomic_json(stage / "result.json", {"identity": identity, "physical_optimizer_steps": budget.calls,
        "trace_sha256": canonical_hash(trace), "seconds": time.perf_counter()-started, "test_accessed": False})
    del model, params, opt, dataset, tok, bases
    release()
    publish(stage, dest)
    return dest


def predict(root, workspace, b, adapter, key, dataset, indices, contract, device):
    dest = root / "predictions" / key
    identity = {"adapter_sha256": sha256_file(adapter / "adapter/adapter_model.pt"),
                "contract": contract, "indices_sha256": canonical_hash(indices)}
    if dest.exists():
        verify(dest)
        result = read(dest / "result.json")
        if result["identity"] != identity:
            raise ValueError(f"prediction identity changed: {key}")
        return result["prediction"]
    stage = stage_dir(root)
    print(f"[Stage18] Development inference {key} n={len(indices)}", flush=True)
    model = evaluation._load_fixed(workspace, b, adapter, device)
    result = {"probability": [], "gold_label": [], "answer_loss": [], "sample_order_sha256": canonical_hash(indices)}
    try:
        for start in range(0, len(indices), 256):
            piece = evaluation._single_predictions(model, dataset, indices[start:start+256], device, 4)
            for field in ("probability", "gold_label", "answer_loss"):
                result[field].extend(piece[field])
            if start == 0 or (start // 256) % 4 == 0 or start+256 >= len(indices):
                print(f"[Stage18] {key} samples={min(start+256,len(indices))}/{len(indices)}", flush=True)
    finally:
        del model
        release()
    valid = s14.validate_prediction(result, indices)
    # JSON remains standards-compliant even for failed numerical predictions.
    atomic_json(stage / "result.json", {"identity": identity, "valid": valid,
        "prediction": result if valid else None, "status": "OK" if valid else "NONFINITE_PREDICTIONS", "test_accessed": False})
    publish(stage, dest)
    return result if valid else None


def phase_rows(root, workspace, b, reg, checks, dev_users, seed, scales, phase, contract):
    bank = build_bank(root, workspace, b, reg, checks, seed, scales, phase, contract)
    result = read(bank / "result.json")
    source_check(Path(__file__).resolve().parents[2], checks)
    device = gpu()
    tokenizer = s8.T5Tokenizer.from_pretrained(runner._base_t5_config(workspace, b)["paths"]["model_dir"])
    dataset = s8.JsonPromptDataset(workspace / b["source"]["development_json"], tokenizer)
    indices = list(range(len(dev_users)))
    if len(dataset) != len(indices):
        raise ValueError("Development length changed")
    original = predict(root, workspace, b, bank / "Original", f"s{seed}_Original", dataset, indices, contract, device)
    if original is None:
        raise RuntimeError("invalid Original predictions")
    rows = []
    for sid in SCENARIOS:
        exact_dir = exact_reference(root, workspace, b, reg, checks, seed, sid, bank, contract)
        exact = predict(root, workspace, b, exact_dir, f"s{seed}_{sid}_Exact", dataset, indices, contract, device)
        if exact is None:
            raise RuntimeError("invalid Exact-Masked predictions")
        requested = set(next(r["users"] for r in reg["scenarios"] if r["id"] == sid))
        local = [i for i, u in enumerate(dev_users) if u in requested]
        for scale in scales:
            key = cid(scale)
            failure = result["failures"].get(key)
            pred = None if failure else predict(root, workspace, b, bank / sid / key,
                f"s{seed}_{sid}_{key}", dataset, indices, contract, device)
            if pred is not None:
                row = s8._metric(key, scale, sid, pred, original, exact, local, result["seconds"], 0, {"selection": SELECT})
                row["status"] = "OK"
            else:
                row = {"candidate_id": key, "response_scale": scale, "scenario": sid,
                    "E_U": None, "D_U": None, "local_residual": None, "utility_eligible": False,
                    "prediction_collapse": None, **(failure or {"status": "NONFINITE_PREDICTIONS"})}
            row.update(seed=seed, local_samples=len(local), phase=phase, test_accessed=False)
            rows.append(row)
    freeze_json(root, f"metrics/{phase}_{seed}", {"rows": rows, "test_accessed": False})
    return rows


def source_check(repo, checks):
    for rel, expected in checks["source_sha256"].items():
        if sha256_file(repo / rel) != expected:
            raise RuntimeError(f"source/config changed during execution: {rel}")


def report(root, selection, rows, confirmed):
    best = selection["selected"]
    lines = ["# Stage 18: Amazon Development lambda calibration", "",
        f"Selected lambda: {best['lambda'] if best else 'none'}; confirmation: {confirmed['status']}", "",
        "Calibration only: same historical checkpoint/cohort/order in all seeds. No FinalTest access.",
        "K2/K4 have only 10/20 requested-user Development samples; K2 is nested in K4.",
        "The existing Development cohort has been used previously. These are not independent holdout results.",
        "Only FS Fisher response strength was searched; all other hyperparameters and deletion sources were fixed.", "",
        "| Seed | Request | Lambda | E_U | Residual | AUC damage | Logloss damage | Eligible | Status |",
        "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def fmt(k):
            v = r.get(k)
            return "--" if v is None else f"{v:.8g}"
        lines.append(f"| {r['seed']} | {r['scenario']} | {r['response_scale']:g} | {fmt('E_U')} | {fmt('local_residual')} | {fmt('auc_damage')} | {fmt('logloss_damage')} | {r['utility_eligible']} | {r['status']} |")
    lines += ["", "The frozen seed42 choice is not changed by confirmation. A failed confirmation requires a separately specified next step, not choosing another lambda after looking at seeds41/43.",
              "A winner at 256 remains boundary-optimal within this grid, not a located global optimum. No automatic further expansion.",
              "All numerical failures and utility violations above remain part of the search record."]
    return "\n".join(lines) + "\n"


def run(repo, workspace, c, config_path, phase):
    root = workspace / c["output_root"]
    with s14.execution_lock(root):
        checks, b, reg, users = audit(repo, workspace, c, config_path)
        env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
        contract = freeze_json(root, "protocol", {"checks": checks, "config": c, "environment": env, "test_accessed": False})
        prev = Path.cwd()
        os.chdir(workspace)
        try:
            rows = phase_rows(root, workspace, b, reg, checks, users, 42, GRID, "coarse", contract)
            coarse = choose(rows, GRID)
            fine = fine_grid(coarse)
            freeze_json(root, "coarse", {**coarse, "fine_grid": fine})
            source_check(repo, checks)
            if phase == "Coarse":
                return {"status": "COARSE_COMPLETED", "decision": coarse, "fine_grid": fine, "test_accessed": False}
            if fine:
                rows += phase_rows(root, workspace, b, reg, checks, users, 42, fine, "fine", contract)
            selected = choose(rows, GRID + fine)
            freeze_json(root, "selection", selected)
            source_check(repo, checks)
            if phase == "Sweep":
                return {"status": "SELECTION_FROZEN", "selection": selected, "test_accessed": False}
            if selected["selected"]:
                scale = selected["selected"]["lambda"]
                for seed in (41, 43):
                    rows += phase_rows(root, workspace, b, reg, checks, users, seed,
                        sorted({0, 1, 64, scale}), "confirm", contract)
                confirmed = confirmation(rows, scale)
            else:
                confirmed = {"status": "NO_ELIGIBLE_CANDIDATE", "test_accessed": False}
            # Detect source/data mutations before publishing a final report.
            end_checks, _, _, _ = audit(repo, workspace, c, config_path)
            if end_checks != checks:
                raise RuntimeError("inputs changed during Stage18; refusing final publication")
            summary = {"schema": SCHEMA, "selection": selected, "confirmation": confirmed, "rows": rows,
                       "status": "COMPLETED", "test_accessed": False}
            dest = root / "summary"
            if dest.exists():
                verify(dest)
                if read(dest / "result.json") != summary:
                    raise ValueError("existing summary differs")
            else:
                stage = stage_dir(root)
                atomic_json(stage / "result.json", summary)
                atomic_text(stage / "report.md", report(root, selected, rows, confirmed))
                publish(stage, dest)
            return {"status": "COMPLETED", "selection": selected["selected"], "confirmation": confirmed,
                    "summary": str(dest), "test_accessed": False}
        finally:
            os.chdir(prev)
            release()


def smoke(repo, workspace, c, config_path):
    """Two real training steps plus adapter reload/inference; never selection evidence."""
    checks, b, reg, _ = audit(repo, workspace, c, config_path)
    root = workspace / c["output_root"]
    with s14.execution_lock(root):
        stage = stage_dir(root)
        prev = Path.cwd()
        os.chdir(workspace)
        try:
            device = gpu()
            seed_everything(42)
            state = torch.load(Path(checks["references"]["Original-Short"]) / "scenarios/K2/adapter/adapter_model.pt", map_location="cpu", weights_only=True)
            model, names, params, bases, _, tok, dataset, users = runner._model_context(workspace, b, reg, device, state["A"])
            targets = next(r["users"] for r in reg["scenarios"] if r["id"] == "K4")
            position = next(i for i, ix in enumerate(reg["order"]) if users[ix] in targets)
            start = position // 16 * 16
            canonical, vectors, failed, calls = transport(model, dataset, reg["order"][start:start+32], users,
                targets, params, names, device, tok.pad_token_id, runner._engine(workspace, b), GRID, stage / "progress.json")
            assert calls == 2
            assert not failed, failed
            for scale in (0, 1, 64, 256):
                values = compose(canonical, names, vectors[cid(scale)].sum(0))
                runner._save_fixed_ab(stage / cid(scale) / "adapter", names, values, bases, "SMOKE-NOT-SCIENTIFIC")
            del model, params, dataset, bases, vectors
            release()
            dev = s8.JsonPromptDataset(workspace / b["source"]["development_json"], tok)
            m = evaluation._load_fixed(workspace, b, stage / cid(64), device)
            pred = evaluation._single_predictions(m, dev, [0, 1, 2, 3], device, 4)
            assert s14.validate_prediction(pred, [0, 1, 2, 3])
            del m
            release()
            result = {"status": "SMOKE_PASS", "optimizer_steps": calls, "scales_tested": GRID,
                      "eval_samples": 4, "scientific_result": False, "test_accessed": False}
            atomic_json(stage / "smoke.json", result)
            return {**result, "diagnostics": str(stage)}
        finally:
            os.chdir(prev)
            release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--workspace", type=Path, default=Path("C:/python/E2URec-main"))
    p.add_argument("--mode", choices=["Preflight", "Smoke", "Full", "Analyze"], default="Preflight")
    p.add_argument("--phase", choices=["All", "Coarse", "Sweep"], default="All")
    args = p.parse_args()
    repo, workspace = args.repo.resolve(), args.workspace.resolve()
    path = repo / "configs/ali_stage18_amazon_lambda_search_v1.json"
    c = load_config(repo, path)
    if args.mode == "Preflight":
        checks, _, _, _ = audit(repo, workspace, c, path)
        result = {"status": "PREFLIGHT_PASS", "input_files": len(checks["input_sha256"]),
                  "source_files": len(checks["source_sha256"]), "request_support": checks["request_support"],
                  "grid": GRID, "gpu_not_loaded": True, "test_accessed": False}
    elif args.mode == "Smoke":
        result = smoke(repo, workspace, c, path)
    elif args.mode == "Full":
        result = run(repo, workspace, c, path, args.phase)
    else:
        root = workspace / c["output_root"]
        with s14.execution_lock(root):
            if not (root / "summary").exists():
                result = {"status": "INCOMPLETE", "completed_metrics": sorted(str(x.parent.relative_to(root)) for x in (root / "metrics").glob("*/result.json"))}
            else:
                verify(root / "summary")
                for x in root.rglob("manifest.json"):
                    if ".work" not in x.parts:
                        verify(x.parent)
                result = read(root / "summary/result.json")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
