"""Group C: trajectory-preserving GPU-resident HVP execution audit."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import yaml
from transformers import T5Tokenizer

from src.diagnostics.t5_lora_influence_feasibility_audit import conjugate_gradient, flatten_tensors, project_update_space, self_kl_loss, split_vector
from src.diagnostics.t5_lora_influence_feasibility_audit_a2 import analytic_b_gradient, build_fixed_a_basis, collect_qv_modules, estimate_lambda_max, install_fixed_ab_coordinate, select_retain_panels
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, load_config, load_legacy_model, move_batch
from src.diagnostics.t5_step817_forget_conflict_audit import tensor_tree_hash
from src.if_a2_optimization.group_a_gradient_audit import masked_batch, masked_forward, stream_weight_gradients
from src.if_a2_optimization.group_b_scale_audit import _all_finite, make_masked_curvature_operator, masked_self_kl_gradient, quantized_delta
from src.paper_if_a2.common import atomic_json, canonical_hash, git_snapshot, hardware_snapshot, safe_run_name, sha256_file
from src.paper_if_a2.if_a2_method import _method_lineage, load_method_config

SCHEMA = "if-a2-group-c-gpu-resident-hvp-v1"
COMPLETED_MARKER = "IF_A2_GROUP_C_GPU_RESIDENT_HVP_V1_COMPLETED"
STOPPED_MARKER = "IF_A2_GROUP_C_GPU_RESIDENT_HVP_V1_STOPPED_SAFELY"
GIB = 1024 ** 3


def _validate_config(config: dict[str, Any]) -> None:
    expected = {"schema", "development_only", "test_access_policy", "audit_target", "group_f_authority", "method_config", "output_root", "input_masking", "runtime", "curvature", "cg", "projection", "equivalence", "scientific_scope"}
    if set(config) != expected or config["schema"] != SCHEMA or config["development_only"] is not True or config["test_access_policy"] != "forbidden": raise ValueError("invalid Group-C schema/policy")
    if config["audit_target"] != "ratio_i02s42v1_interaction_2pct_corrected_masking": raise ValueError("Group-C target changed")
    if config["method_config"] != "outputs/ru1/i02s42v1/configs/if_a2.yaml" or config["output_root"] != "outputs/if_a2_optimization/group_c_gpu_resident_hvp_v1": raise ValueError("Group-C path contract changed")
    authority = {"run_dir": "outputs/if_a2_optimization/group_f_projection_decomposition_v1/if_a2_group_f_i02s42v1_masked_seed42_v1", "result_sha256": "c47424591553a777b75710b7b6d714c50c447f1f8149ff648ceda6e3f7844330", "manifest_sha256": "7ea47eee45aad6e6c808f24b34c398829dfd7e21de119b4ce8930e3a59420d35", "basis_sha256": "61a1b2540c40d06e49261fb0aeabeaf8be1f9ca2448621b87d820110463461d0", "forget_gradient_sha256": "dc1c5403e4ebfe88e9ace56ab0c82105a4f363b3cde69a75d16047246a5e87be", "lambda_max": .003208373308317939, "damping": 3.208373308317939e-05, "f_both_candidate_sha256": "8c487d5c2e5e98101043427fc5a2ddb43f12afa6117172d5566da5a03ad5b60f"}
    if config["group_f_authority"] != authority: raise ValueError("Group-F authority changed")
    mask = {"protocol": "explicit_encoder_attention_mask", "mask_definition": "input_ids_ne_pad_token_id", "applies_to": ["basis", "forget_gradient", "curvature", "projection_constraints"], "legacy_unmasked_equivalence_claimed": False}
    if config["input_masking"] != mask: raise ValueError("corrected masking contract changed")
    runtime = {"forget_batch_size": 16, "curvature_batch_size": 8, "safety_batch_size": 8, "allocator_fraction": .88, "minimum_free_gib_before_load": 8., "hard_cap_reserved_gib": 14., "resident_panel_max_gib": 1.}
    if config["runtime"] != runtime: raise ValueError("runtime contract changed")
    curvature = {"primary_panel_samples": 4096, "primary_panel_seed": 42, "reference_strategy": "detached_same_forward", "power_iterations": 12, "convergence_tolerance": 1e-4, "numerical_lower_bound": 1e-14, "relative_damping_ratio": .01}
    if config["curvature"] != curvature: raise ValueError("curvature contract changed")
    cg = {"relative_residual_tolerance": 1e-4, "absolute_residual_tolerance": 1e-10, "max_iterations": 40, "residual_explosion_factor": 1000., "pap_absolute_tolerance": 1e-14}
    if config["cg"] != cg: raise ValueError("CG contract changed")
    projection = {"active_constraints": ["retain_supervised", "retain_self_kl"], "safety_panel_samples": 2048, "safety_panel_seed": 44, "relative_singular_tolerance": 1e-10, "normalized_constraint_tolerance": 1e-8, "formal_dtype": "float32", "scale": .01}
    if config["projection"] != projection: raise ValueError("projection contract changed")
    equivalence = {"probe_order": [["streaming", "resident"], ["resident", "streaming"]], "probe_vectors": ["power_seed42", "normalized_forget_gradient"], "require_exact_probe_sha": True, "probe_max_absolute_error": 0., "probe_relative_l2_error": 0., "lambda_max_must_equal_group_f": True, "damping_must_equal_group_f": True, "f_both_candidate_sha_must_equal_group_f": True}
    if config["equivalence"] != equivalence: raise ValueError("equivalence gates changed")
    scope = {"purpose": "trajectory_preserving_hvp_execution_optimization", "mathematical_objective_changed": False, "curvature_panel_changed": False, "damping_changed": False, "cg_changed": False, "projection_changed": False, "scale_changed": False, "candidate_model_published": False}
    if config["scientific_scope"] != scope: raise ValueError("scientific scope changed")


def validate_group_f_authority(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    authority = config["group_f_authority"]; run = root / authority["run_dir"]; required = {"COMPLETED", "group_f_projection_decomposition.json", "manifest.json", "run_state.json"}
    if not run.is_dir() or {item.name for item in run.iterdir()} != required: raise ValueError("invalid Group-F authority inventory")
    if (run / "COMPLETED").read_text(encoding="utf-8") != "IF_A2_GROUP_F_PROJECTION_DECOMPOSITION_V1_COMPLETED\n": raise ValueError("invalid Group-F marker")
    if sha256_file(run / "group_f_projection_decomposition.json") != authority["result_sha256"] or sha256_file(run / "manifest.json") != authority["manifest_sha256"]: raise ValueError("Group-F SHA mismatch")
    report = json.loads((run / "group_f_projection_decomposition.json").read_text(encoding="utf-8")); manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")); state = json.loads((run / "run_state.json").read_text(encoding="utf-8"))
    both = next((row for row in report.get("arms", []) if row.get("arm") == "F-Both"), None)
    checks = [manifest.get("status") == "COMPLETED", state.get("status") == "COMPLETED", report.get("group_b_anchor_exact") is True, report.get("selection_performed") is False, report.get("coordinate", {}).get("basis_sha256") == authority["basis_sha256"], report.get("forget_gradient", {}).get("sha256") == authority["forget_gradient_sha256"], report.get("curvature", {}).get("lambda_max", {}).get("lambda_max_hat") == authority["lambda_max"], report.get("curvature", {}).get("damping") == authority["damping"], both is not None and both.get("candidate_sha256") == authority["f_both_candidate_sha256"], report.get("optimizer_steps_committed") == 0, report.get("test_accessed") is False, all(report.get("integrity", {}).values())]
    if not all(checks): raise ValueError("Group-F scientific authority invalid")
    return {**authority, "run_dir": str(run), "status": "COMPLETED", "hardware": report["hardware"], "timing": report["timing"], "test_accessed": False}


def materialize_resident_panel(dataset: JsonPromptDataset, indices: Sequence[int], batch_size: int, pad_token_id: int, device: torch.device) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    batches: list[dict[str, torch.Tensor]] = []; rows = []; order = []; valid_tokens = 0; resident_bytes = 0
    for start in range(0, len(indices), batch_size):
        chosen = list(indices[start:start + batch_size]); batch = {key: value.to(device).contiguous() for key, value in masked_batch(dataset, chosen, pad_token_id).items()}; tokens = int(batch["target_ids"].ne(-100).sum()); ids = batch["sample_id"].detach().cpu().tolist()
        if ids != chosen or not bool(batch["attention_mask"].any(dim=1).all()): raise RuntimeError("resident panel order/mask mismatch")
        batch_bytes = sum(value.numel() * value.element_size() for value in batch.values()); resident_bytes += batch_bytes; valid_tokens += tokens; order.extend(ids); rows.append({"batch": len(rows), "sample_ids_sha256": canonical_hash(ids), "tensor_sha256": tensor_tree_hash(batch), "shape": {key: list(value.shape) for key, value in batch.items()}, "valid_tokens": tokens, "bytes": batch_bytes}); batches.append(batch)
    if order != list(indices) or valid_tokens <= 0: raise RuntimeError("resident panel lineage mismatch")
    return batches, {"samples": len(order), "batches": len(batches), "batch_size": batch_size, "valid_tokens": valid_tokens, "sample_order_sha256": canonical_hash(order), "batch_registry_sha256": canonical_hash(rows), "resident_bytes": resident_bytes, "resident_gib": resident_bytes / GIB, "device": str(device), "all_tensors_resident": all(value.device == device for batch in batches for value in batch.values()), "rows": rows}


def resident_hessian_vector_product(loss: torch.Tensor, parameters: Sequence[torch.Tensor], vector_pieces: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(parameters) != len(vector_pieces): raise ValueError("resident HVP piece count mismatch")
    first = torch.autograd.grad(loss, parameters, create_graph=True, retain_graph=True, allow_unused=False); dot = sum((gradient * piece).sum() for gradient, piece in zip(first, vector_pieces)); second = torch.autograd.grad(dot, parameters, create_graph=False, retain_graph=False, allow_unused=False)
    if any(parameter.grad is not None for parameter in parameters): raise RuntimeError("resident autograd.grad populated .grad")
    result = torch.cat([value.reshape(-1).to(dtype=torch.float64) for value in second])
    if not torch.isfinite(result).all(): raise FloatingPointError("nonfinite resident HVP")
    return result


def make_resident_curvature_operator(model: torch.nn.Module, batches: Sequence[dict[str, torch.Tensor]], parameters: Sequence[torch.Tensor], counter: dict[str, int]) -> Callable[[torch.Tensor], torch.Tensor]:
    if not batches: raise ValueError("empty resident batch registry")
    device = parameters[0].device
    if any(value.device != device for batch in batches for value in batch.values()): raise RuntimeError("nonresident panel tensor")
    token_counts = [int(batch["target_ids"].ne(-100).sum().cpu()) for batch in batches]
    def operator(vector: torch.Tensor) -> torch.Tensor:
        pieces = split_vector(vector, parameters); total = torch.zeros(vector.numel(), dtype=torch.float64, device=device); tokens_total = 0; started = time.perf_counter()
        for batch, tokens in zip(batches, token_counts):
            mask = batch["target_ids"].ne(-100); current = masked_forward(model, batch).logits; loss = self_kl_loss(current.detach(), current, mask); weighted = resident_hessian_vector_product(loss, parameters, pieces) * tokens; total.add_(weighted); tokens_total += tokens; counter["hvp_batches"] = counter.get("hvp_batches", 0) + 1; del mask, current, loss, weighted
        if tokens_total <= 0: raise RuntimeError("empty curvature panel")
        result = total.cpu(); result.div_(tokens_total); curvature = float(torch.dot(vector, result)); tolerance = 1e-10 + 1e-8 * float(torch.linalg.vector_norm(vector)) * float(torch.linalg.vector_norm(result))
        if curvature < -tolerance: raise RuntimeError("significant_negative_curvature")
        counter["operator_calls"] = counter.get("operator_calls", 0) + 1; counter.setdefault("operator_seconds", []).append(time.perf_counter() - started); print(f"[group-c:hvp] operator_call={counter['operator_calls']} resident_batches={len(batches)} seconds={counter['operator_seconds'][-1]:.3f}", flush=True); return result
    return operator


def compare_operator_probes(streaming: Callable[[torch.Tensor], torch.Tensor], resident: Callable[[torch.Tensor], torch.Tensor], vectors: Sequence[tuple[str, torch.Tensor]], order: Sequence[Sequence[str]]) -> dict[str, Any]:
    rows = []
    for (name, vector), execution_order in zip(vectors, order):
        values = {}; timing = {}
        for implementation in execution_order:
            started = time.perf_counter(); values[implementation] = (streaming if implementation == "streaming" else resident)(vector); timing[implementation] = time.perf_counter() - started
        difference = values["resident"] - values["streaming"]; reference_norm = float(torch.linalg.vector_norm(values["streaming"])); resident_norm = float(torch.linalg.vector_norm(values["resident"])); denominator = max(reference_norm * resident_norm, 1e-300)
        rows.append({"probe": name, "execution_order": list(execution_order), "streaming_sha256": tensor_tree_hash({"hvp": values["streaming"]}), "resident_sha256": tensor_tree_hash({"hvp": values["resident"]}), "sha_exact": torch.equal(values["streaming"], values["resident"]), "max_absolute_error": float(torch.max(torch.abs(difference))), "relative_l2_error": float(torch.linalg.vector_norm(difference)) / max(reference_norm, 1e-300), "cosine": float(torch.dot(values["streaming"], values["resident"])) / denominator, "streaming_seconds": timing["streaming"], "resident_seconds": timing["resident"], "speedup": timing["streaming"] / max(timing["resident"], 1e-300)})
    return {"rows": rows, "aggregate_streaming_seconds": sum(row["streaming_seconds"] for row in rows), "aggregate_resident_seconds": sum(row["resident_seconds"] for row in rows), "aggregate_speedup": sum(row["streaming_seconds"] for row in rows) / max(sum(row["resident_seconds"] for row in rows), 1e-300)}


def equivalence_checks(probes: dict[str, Any], basis_sha: str, gradient_sha: str, estimate: dict[str, Any], damping: float, candidate_sha: str, authority: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gate = config["equivalence"]
    return {"probe_sha_exact": all(row["sha_exact"] is True for row in probes["rows"]) if gate["require_exact_probe_sha"] else True, "probe_max_absolute_error": all(row["max_absolute_error"] <= gate["probe_max_absolute_error"] for row in probes["rows"]), "probe_relative_l2_error": all(row["relative_l2_error"] <= gate["probe_relative_l2_error"] for row in probes["rows"]), "basis_exact": basis_sha == authority["basis_sha256"], "forget_gradient_exact": gradient_sha == authority["forget_gradient_sha256"], "lambda_max_exact": estimate["lambda_max_hat"] == authority["lambda_max"], "damping_exact": damping == authority["damping"], "f_both_candidate_exact": candidate_sha == authority["f_both_candidate_sha256"]}


def _publish(stage: Path, destination: Path, report: dict[str, Any], status: str, implementation_sha: str) -> None:
    if status not in {"COMPLETED", "STOPPED_SAFELY"} or not _all_finite(report): raise ValueError("invalid Group-C publication")
    result_name = "group_c_gpu_resident_hvp.json"; atomic_json(stage / result_name, report); atomic_json(stage / "run_state.json", {"schema": SCHEMA, "status": status, "reason": None if status == "COMPLETED" else "numerical_equivalence_gate_failed", "optimizer_constructed": False, "optimizer_steps_committed": 0, "candidate_model_published": False, "test_accessed": False}); marker = "COMPLETED" if status == "COMPLETED" else "STOPPED_SAFELY"; marker_text = COMPLETED_MARKER if status == "COMPLETED" else STOPPED_MARKER; (stage / marker).write_text(marker_text + "\n", encoding="utf-8", newline="\n"); atomic_json(stage / "manifest.json", {"schema": SCHEMA, "status": status, "result_sha256": sha256_file(stage / result_name), "run_state_sha256": sha256_file(stage / "run_state.json"), "implementation_sha256": implementation_sha, "published_atomically": True, "optimizer_steps_committed": 0, "candidate_model_published": False, "test_accessed": False}); destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)


def preflight(root: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")); _validate_config(config); authority = validate_group_f_authority(root, config); method = load_method_config(root / config["method_config"])
    return {"schema": SCHEMA, "mode": "Preflight", "group_f_authority": authority, "original_sha256": sha256_file(root / method["original"]), "resident_strategy": "gpu_batches_plus_single_vector_transfer_plus_device_float64_accumulation", "equivalence": config["equivalence"], "model_loaded": False, "optimizer_constructed": False, "test_accessed": False}


def analyze(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")); _validate_config(config); run = root / config["output_root"] / safe_run_name(run_name)
    if not run.is_dir(): raise ValueError("missing Group-C run")
    markers = {name for name in ("COMPLETED", "STOPPED_SAFELY") if (run / name).is_file()}
    if len(markers) != 1: raise ValueError("invalid Group-C terminal marker")
    required = {next(iter(markers)), "group_c_gpu_resident_hvp.json", "manifest.json", "run_state.json"}
    if {item.name for item in run.iterdir()} != required: raise ValueError("invalid Group-C inventory")
    report = json.loads((run / "group_c_gpu_resident_hvp.json").read_text(encoding="utf-8")); manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")); state = json.loads((run / "run_state.json").read_text(encoding="utf-8")); expected_marker = COMPLETED_MARKER if state["status"] == "COMPLETED" else STOPPED_MARKER
    if (run / next(iter(markers))).read_text(encoding="utf-8") != expected_marker + "\n" or manifest.get("result_sha256") != sha256_file(run / "group_c_gpu_resident_hvp.json") or manifest.get("status") != state.get("status") or report.get("test_accessed") is not False: raise ValueError("Group-C terminal evidence mismatch")
    return {"status": state["status"], "run_dir": str(run), "equivalence_checks": report["equivalence_checks"], "exact": report["exact"], "probe_aggregate_speedup": report["operator_equivalence"]["aggregate_speedup"], "legacy_group_f_hvp_seconds": report["timing_reference"]["group_f_lambda_plus_cg_seconds"], "resident_lambda_plus_cg_seconds": report["timing"]["resident_lambda_plus_cg_seconds"], "end_to_end_speedup_estimate": report["speedup_summary"]["lambda_plus_cg_speedup_vs_group_f"], "resident_panel_gib": report["resident_panel"]["resident_gib"], "peak_reserved_gib": report["memory"]["peak_reserved_gib"], "optimizer_steps_committed": 0, "test_accessed": False}


def run(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")); _validate_config(config); run_name = safe_run_name(run_name); destination = (root / config["output_root"] / run_name).resolve()
    if destination.exists(): raise FileExistsError(destination)
    authority = validate_group_f_authority(root, config); git = git_snapshot(root)
    if not git["clean"]: raise RuntimeError("formal Group-C audit requires clean Git")
    method_path = root / config["method_config"]; method = load_method_config(method_path); base = load_config(root / method["base_config"], root); checkpoint = root / method["original"]
    for key in ("original", "forget", "retain", "development"):
        path = checkpoint if key == "original" else root / method[key]
        if sha256_file(path) != method[f"{key}_sha256"]: raise ValueError(f"{key} authority SHA mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("Group C requires exactly one CUDA GPU")
    device = torch.device("cuda:0"); free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes / GIB < config["runtime"]["minimum_free_gib_before_load"]: raise RuntimeError("insufficient clean-GPU free VRAM")
    current_hardware = hardware_snapshot(); hardware_keys = ("gpu_uuid", "gpu_name", "gpu_memory_mib", "driver", "torch", "cuda", "python", "single_gpu"); same_hardware = all(current_hardware.get(key) == authority["hardware"].get(key) for key in hardware_keys)
    if not same_hardware: raise RuntimeError("Group-C timing comparison requires the exact Group-F hardware/runtime")
    work = destination.parent / ".work"; work.mkdir(parents=True, exist_ok=True); stage = work / f"{run_name}.{uuid.uuid4().hex}.stage"; stage.mkdir(); model = None; resident_batches = None; started = time.perf_counter(); timing = {}; checkpoint_before = sha256_file(checkpoint)
    try:
        torch.cuda.set_per_process_memory_fraction(config["runtime"]["allocator_fraction"], device); torch.cuda.reset_peak_memory_stats(); print("[group-c:start] reconstructing exact Group-F coordinate", flush=True); model = load_legacy_model(checkpoint).to(device).eval(); tokenizer = T5Tokenizer.from_pretrained(base["paths"]["model_dir"]); pad = tokenizer.pad_token_id
        if pad is None: raise RuntimeError("corrected masking requires pad_token_id")
        forget = JsonPromptDataset(root / method["forget"], tokenizer); retain = JsonPromptDataset(root / method["retain"], tokenizer); lineage, indices, users = _method_lineage(root, base, method)
        if len(forget) != 1258 or len(retain) != 58742: raise RuntimeError("2% lineage/count mismatch")
        a2 = yaml.safe_load((root / method["if_a2_config"]).read_text(encoding="utf-8")); panels = select_retain_panels(users["retain_train"], a2["panels"]); primary = panels["primary"]["indices"]; safety = panels["safety"]["indices"]
        if len(primary) != 4096 or len(safety) != 2048: raise RuntimeError("Retain panel contract changed")
        modules = collect_qv_modules(model); module_names = [name for name, _ in modules]; weights = [module.weight for _, module in modules]; official = list(model.parameters()); buffers = list(model.buffers()); official_before = tensor_tree_hash({str(i): value.detach() for i, value in enumerate(official)}); buffers_before = tensor_tree_hash({str(i): value.detach() for i, value in enumerate(buffers)}); rng_before = tensor_tree_hash({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
        for parameter in model.parameters(): parameter.requires_grad_(False)
        for weight in weights: weight.requires_grad_(True)
        tick = time.perf_counter(); matrices, basis_meta = stream_weight_gradients(model, forget, indices["forget_train"], weights, device, 16, pad); timing["basis_gradient_seconds"] = time.perf_counter() - tick; bases = {}; basis_reports = []
        for name, matrix in zip(module_names, matrices): bases[name], row = build_fixed_a_basis(matrix, rank=16, name=name, seed=42); basis_reports.append(row)
        basis_sha = canonical_hash([(row["name"], row["basis_sha256"]) for row in basis_reports]); sample = move_batch(masked_batch(forget, [0], pad), device)
        with torch.no_grad(): before = masked_forward(model, sample).logits.detach().cpu()
        for weight in weights: weight.requires_grad_(False)
        names, parameters = install_fixed_ab_coordinate(model, bases, 32); names = [name[:-2] if name.endswith(".B") else name for name in names]
        with torch.no_grad(): after = masked_forward(model, sample).logits.detach().cpu()
        if names != module_names or not torch.equal(before, after) or any(torch.count_nonzero(parameter).item() for parameter in parameters): raise RuntimeError("fixed-A/B equivalence failed")
        del sample, before, after
        tick = time.perf_counter(); parts, gradient_meta = stream_weight_gradients(model, forget, indices["forget_train"], parameters, device, 16, pad); g_f = flatten_tensors(parts); timing["forget_gradient_seconds"] = time.perf_counter() - tick; analytic = flatten_tensors([analytic_b_gradient(matrix, bases[name], 32, 16) for name, matrix in zip(module_names, matrices)]); analytic_error = float(torch.linalg.vector_norm(g_f - analytic) / torch.linalg.vector_norm(analytic))
        if analytic_error > 2e-5: raise RuntimeError("analytic gradient mismatch")
        print("[group-c:resident] materializing 4096-sample panel", flush=True); tick = time.perf_counter(); resident_batches, resident_meta = materialize_resident_panel(retain, primary, 8, pad, device); timing["resident_materialization_seconds"] = time.perf_counter() - tick
        if resident_meta["resident_gib"] > config["runtime"]["resident_panel_max_gib"] or torch.cuda.max_memory_reserved() / GIB > config["runtime"]["hard_cap_reserved_gib"]: raise RuntimeError("resident panel memory safety gate failed")
        streaming_counter = {"hvp_batches": 0}; resident_counter = {"hvp_batches": 0}; streaming = make_masked_curvature_operator(model, retain, primary, parameters, device, 8, pad, streaming_counter); resident = make_resident_curvature_operator(model, resident_batches, parameters, resident_counter)
        generator = torch.Generator(device="cpu"); generator.manual_seed(42); power_probe = torch.randn(g_f.numel(), dtype=torch.float64, generator=generator); power_probe /= torch.linalg.vector_norm(power_probe); forget_probe = g_f / torch.linalg.vector_norm(g_f)
        print("[group-c:equivalence] two full-panel crossed probes", flush=True); probes = compare_operator_probes(streaming, resident, [("power_seed42", power_probe), ("normalized_forget_gradient", forget_probe)], config["equivalence"]["probe_order"])
        probe_exact = all(row["sha_exact"] is True and row["max_absolute_error"] == 0. and row["relative_l2_error"] == 0. for row in probes["rows"])
        if not probe_exact: raise RuntimeError(f"resident HVP probe equivalence failed before lambda/CG: {json.dumps(probes, sort_keys=True)}")
        print("[group-c:lambda] resident operator", flush=True); tick = time.perf_counter(); estimate = estimate_lambda_max(resident, g_f.numel(), seed=42, iterations=12, convergence_tolerance=1e-4, numerical_lower_bound=1e-14); timing["resident_lambda_seconds"] = time.perf_counter() - tick; damping = .01 * estimate["lambda_max_hat"]
        print("[group-c:cg] resident operator", flush=True); tick = time.perf_counter(); cg = conjugate_gradient(resident, g_f, damping=damping, relative_tolerance=1e-4, absolute_tolerance=1e-10, max_iterations=40, residual_explosion_factor=1000., pap_tolerance=1e-14); raw = cg.pop("solution"); timing["resident_cg_seconds"] = time.perf_counter() - tick; timing["resident_lambda_plus_cg_seconds"] = timing["resident_lambda_seconds"] + timing["resident_cg_seconds"]
        print("[group-c:anchor] reconstructing F-Both candidate", flush=True); sup_parts, sup_meta = stream_weight_gradients(model, retain, safety, parameters, device, 8, pad); g_sup = flatten_tensors(sup_parts); g_kl, kl_meta = masked_self_kl_gradient(model, retain, safety, parameters, device, 8, pad); base_flat = flatten_tensors([parameter.detach() for parameter in parameters]); projection = project_update_space(raw, [g_sup, g_kl], relative_tolerance=1e-10, normalized_tolerance=1e-8, formal_dtype=torch.float32, base=base_flat); direction = projection.pop("actual"); actual = quantized_delta(base_flat, direction, .01); candidate_sha = tensor_tree_hash({"candidate": actual}); gradient_sha = tensor_tree_hash({"g_F": g_f}); checks = equivalence_checks(probes, basis_sha, gradient_sha, estimate, damping, candidate_sha, authority, config); exact = all(checks.values())
        peak_allocated = torch.cuda.max_memory_allocated() / GIB; peak_reserved = torch.cuda.max_memory_reserved() / GIB; integrity = {"checkpoint_unchanged": sha256_file(checkpoint) == checkpoint_before, "official_parameters_unchanged": tensor_tree_hash({str(i): value.detach() for i, value in enumerate(official)}) == official_before, "buffers_unchanged": tensor_tree_hash({str(i): value.detach() for i, value in enumerate(buffers)}) == buffers_before, "B_restored_zero": all(torch.count_nonzero(parameter).item() == 0 for parameter in parameters), "parameter_grad_absent": all(parameter.grad is None for parameter in official + list(parameters)), "rng_unchanged": tensor_tree_hash({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}) == rng_before, "model_eval": model.training is False, "peak_reserved_within_cap": peak_reserved <= 14., "same_hardware_as_group_f": same_hardware}
        if not all(integrity.values()): raise RuntimeError("Group-C transactional restoration failed")
        implementation_files = [Path(__file__), Path(__file__).with_name("group_b_scale_audit.py"), config_path, method_path]; implementation = {str(path.resolve().relative_to(root.resolve())).replace("\\", "/"): sha256_file(path) for path in implementation_files}; implementation_sha = canonical_hash(implementation); group_f_timing = authority["timing"]; reference_seconds = group_f_timing["lambda_max_seconds"] + group_f_timing["cg_seconds"]
        report = {"schema": SCHEMA, "run_name": run_name, "status": "COMPLETED" if exact else "STOPPED_SAFELY", "runtime_protocol": "2pct_corrected_masking_gpu_resident_hvp", "group_f_authority": authority, "lineage": lineage, "input_masking": config["input_masking"], "resident_strategy": {"batches_preconstructed_once": True, "panel_tensors_gpu_resident": True, "vector_pieces_transferred_once_per_operator_call": True, "hvp_accumulation": "gpu_float64_original_batch_order", "operator_result_cpu_transfers_per_call": 1, "reference_logits_cached": False}, "resident_panel": resident_meta, "coordinate": {"basis_sha256": basis_sha, "basis_reports": basis_reports, "module_order": names, "A_frozen": True, "B_only": True, "rank": 16, "alpha": 32}, "forget_gradient": {**gradient_meta, "basis_gradient": basis_meta, "norm": float(torch.linalg.vector_norm(g_f)), "sha256": gradient_sha, "analytic_relative_error": analytic_error}, "operator_equivalence": probes, "resident_curvature": {"lambda_max": estimate, "damping": damping, "counter": resident_counter}, "streaming_probe_counter": streaming_counter, "cg": {**cg, "solution_sha256": tensor_tree_hash({"solution": raw})}, "projection": {**projection, "active_constraints": ["retain_supervised", "retain_self_kl"], "scale": .01, "direction_sha256": tensor_tree_hash({"direction": direction}), "candidate_sha256": candidate_sha, "sup_gradient": {**sup_meta, "norm": float(torch.linalg.vector_norm(g_sup))}, "kl_gradient": kl_meta}, "equivalence_checks": checks, "exact": exact, "timing": {**timing, "wall_time_seconds": time.perf_counter() - started}, "timing_reference": {"group_f_lambda_plus_cg_seconds": reference_seconds, "group_f_total_seconds": group_f_timing["wall_time_seconds"]}, "speedup_summary": {"probe_speedup": probes["aggregate_speedup"], "lambda_plus_cg_speedup_vs_group_f": reference_seconds / max(timing["resident_lambda_plus_cg_seconds"], 1e-300), "same_hardware_verified": same_hardware, "formal_speedup_claim_requires_same_hardware": True}, "memory": {"device_total_gib": total_bytes / GIB, "free_before_load_gib": free_bytes / GIB, "peak_allocated_gib": peak_allocated, "peak_reserved_gib": peak_reserved, "hard_cap_reserved_gib": 14.}, "scientific_scope": config["scientific_scope"], "integrity": integrity, "checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": sha256_file(checkpoint), "git": git, "hardware": current_hardware, "implementation_files": implementation, "implementation_sha256": implementation_sha, "optimizer_constructed": False, "optimizer_steps_committed": 0, "candidate_model_published": False, "retrain_loaded": False, "test_loader_built": False, "test_accessed": False}
        status = "COMPLETED" if exact else "STOPPED_SAFELY"; _publish(stage, destination, report, status, implementation_sha); return {"status": status, "run_dir": str(destination), "exact": exact, "equivalence_checks": checks, "probe_speedup": probes["aggregate_speedup"], "lambda_plus_cg_speedup_vs_group_f": report["speedup_summary"]["lambda_plus_cg_speedup_vs_group_f"], "optimizer_steps_committed": 0, "test_accessed": False}
    finally:
        if resident_batches is not None: del resident_batches
        if model is not None: del model
        if stage.exists(): shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def synthetic() -> dict[str, Any]:
    class Tiny(torch.nn.Module):
        def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([[.2, -.1], [.4, .3]]))
        def forward(self, input_ids, attention_mask, labels):
            logits = torch.nn.functional.embedding(input_ids, self.weight); loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), ignore_index=-100); return type("Output", (), {"logits": logits, "loss": loss})()
    model = Tiny().eval(); batch = {"sample_id": torch.tensor([0, 1]), "input_ids": torch.tensor([[0, 1], [1, 0]]), "target_ids": torch.tensor([[0, 1], [1, 0]]), "attention_mask": torch.ones((2, 2), dtype=torch.bool)}; counter = {}; resident = make_resident_curvature_operator(model, [batch], [model.weight], counter); vector = torch.tensor([.1, .2, -.3, .4], dtype=torch.float64); left = resident(vector); right = resident(vector)
    return {"schema": SCHEMA, "deterministic": torch.equal(left, right), "hvp_sha256": tensor_tree_hash({"hvp": left}), "operator_calls": counter["operator_calls"], "optimizer_constructed": False, "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--run-name"); parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "Full", "Analyze"], default="Preflight"); args = parser.parse_args(); root = args.root.resolve(); config = args.config.resolve()
    if args.mode == "Preflight": value = preflight(root, config)
    elif args.mode == "SyntheticDryRun": value = synthetic()
    elif not args.run_name: raise ValueError(f"{args.mode} requires --run-name")
    elif args.mode == "Analyze": value = analyze(root, config, args.run_name)
    else: value = run(root, config, args.run_name)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__": main()
