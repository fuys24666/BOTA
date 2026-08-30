"""Frozen ML-1M L8 source-only versus block empirical-Fisher BOTA audit."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from transformers import T5Tokenizer

from src.bota_if import p1_trajectory_transport_audit as p1
from src.bota_if import p2b_full_module_adamw_transport_audit as p2b
from src.bota_if.p1_trajectory_transport_audit import StepBudget
from src.diagnostics.ml1m_development_protocol import reconstruct_authoritative_rows
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, move_batch
from src.if_a2_optimization.group_a_gradient_audit import masked_batch
from src.paper_baselines.common import capture_rng, tensor_tree_hash
from src.paper_if_a2.common import atomic_json, canonical_hash, directory_hash, git_snapshot, safe_run_name, sha256_file

from . import evaluation as short_eval
from . import runner
from . import t012_behavior_ablation as behavior
from .protocol import load_config as load_benchmark_config
from .protocol import validate_prepared

SCHEMA = "bota-short-fisher-ablation-v1"
MARKER = "BOTA_SHORT_FISHER_ABLATION_V1_COMPLETED"
ARMS = ("T2_SourceOnly", "T2_BlockEmpiricalFisher")
DISPLAY = {
    "T2_SourceOnly": "BOTA-T2-SourceOnly",
    "T2_BlockEmpiricalFisher": "BOTA-T2-BlockEmpiricalFisher",
}
REQUIRED = {"COMPLETED", "adapters", "contract.json", "manifest.json", "metrics.csv", "metrics.json", "provenance.json", "report.md", "run_state.json", "timing.json"}


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema", "test_access_policy", "output_root", "benchmark_config", "authority", "protocol", "evaluation", "decision", "runtime"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA or value["test_access_policy"] != "forbidden":
        raise ValueError("invalid ML-1M Fisher-ablation config")
    if value["authority"] != {"benchmark_name": "bota_short_i02_seed42_v3", "scenario": "L8", "original_run_name": "bota_short_original_seed42_v3", "exact_masked_run_name": "bota_short_exact_masked_seed42_v3", "formal_bota_run_name": "bota_short_bota_seed42_v3"}:
        raise ValueError("Fisher-ablation authority changed")
    if value["protocol"] != {"seed": 42, "optimizer_steps": 200, "batch_size": 16, "arms": list(ARMS), "shared_canonical_trajectory": True, "physical_optimizer_step_calls": 200, "online_optimizer_steps": 0, "source_only_gradient_response": "deletion_source_only", "fisher_gradient_response": "deletion_source_plus_per_sample_block_empirical_fisher", "authoritative_optimizer_steps_committed": 0}:
        raise ValueError("Fisher-ablation protocol changed")
    if value["evaluation"] != {"split": "Development", "final_test": False, "inference_batch_size": 4, "bootstrap_resamples": 1000, "bootstrap_seed": 42501}:
        raise ValueError("Fisher-ablation evaluation changed")
    expected_decision = {"parameter_relative_l2_material_margin": .02, "behavioral_absolute_residual_material_margin": .02, "auc_damage_noninferiority_margin": .001, "log_loss_damage_noninferiority_margin": .002, "single_seed_stability_claim_forbidden": True, "posthoc_candidate_selection_forbidden": True}
    if value["decision"] != expected_decision:
        raise ValueError("Fisher-ablation decision rule changed")
    return value


def _state(parameters: Sequence[torch.Tensor], users: int) -> list[dict[str, torch.Tensor]]:
    return [{key: torch.zeros((users, *parameter.shape), device=parameter.device, dtype=parameter.dtype) for key in ("theta", "m", "v")} for parameter in parameters]


def _advance_source_only(states, sources, parameters, optimizer, step: int, config: dict[str, Any]) -> None:
    opt = config["optimizer"]; beta1, beta2 = map(float, opt["betas"])
    for parameter, source, current in zip(parameters, sources, states):
        optimizer_state = optimizer.state.get(parameter, {})
        m = optimizer_state.get("exp_avg", torch.zeros_like(parameter)).detach()
        v = optimizer_state.get("exp_avg_sq", torch.zeros_like(parameter)).detach()
        theta, dm, dv = p2b.stable_adamw_tangent_step(
            parameter.detach(), parameter.grad.detach(), m, v,
            current["theta"], current["m"], current["v"], source,
            step=step, lr=float(opt["learning_rate"]), beta1=beta1,
            beta2=beta2, eps=float(opt["eps"]),
            weight_decay=float(opt["weight_decay"]), full_v=True,
        )
        current.update(theta=theta, m=dm, v=dv)


def run_shared_canonical(model, dataset, indices, user_ids, selected_users, parameters, names, device, pad, config, budget):
    """Run one formal trajectory and propagate only the closure toggle."""
    optimizer = p1._optimizer(parameters, config)
    fisher = p2b.new_full_states(parameters, len(selected_users))
    source_only = _state(parameters, len(selected_users))
    trace = []; batch_size = config["schedule"]["batch_size"]
    for step, start in enumerate(range(0, len(indices), batch_size), 1):
        chosen = list(indices[start:start + batch_size]); batch_users = [int(user_ids[index]) for index in chosen]
        batch = move_batch(masked_batch(dataset, chosen, pad), device); losses = p1._sample_losses(model, batch)
        sample_rows = [[] for _ in parameters]
        for sample in range(len(chosen)):
            gradients = torch.autograd.grad(losses[sample], parameters, retain_graph=True)
            for module, gradient in enumerate(gradients): sample_rows[module].append(gradient.detach())
        optimizer.zero_grad(set_to_none=True); formal_loss = losses.sum() / len(chosen); formal_loss.backward()
        coefficients = [torch.stack(rows) for rows in sample_rows]; sources = []
        for rows, parameter in zip(coefficients, parameters):
            values = []
            for target in selected_users:
                slots = [slot for slot, user in enumerate(batch_users) if user == target]
                values.append(-rows[slots].sum(0) / len(chosen) if slots else torch.zeros_like(parameter))
            sources.append(torch.stack(values))
        _advance_source_only(source_only, sources, parameters, optimizer, step, config)
        p2b.advance_full_transports(fisher, coefficients, sources, parameters, optimizer, step, config)
        budget.step(optimizer, "shared_canonical_reference")
        trace.append({"step": step, "batch_hash": canonical_hash(chosen), "selected_user_slot_counts": [batch_users.count(user) for user in selected_users]})
        del batch, losses, formal_loss, sample_rows, coefficients, sources
    canonical = {name: parameter.detach().float().cpu().clone() for name, parameter in zip(names, parameters)}
    result = {
        "T2_SourceOnly": [{key: value.detach().cpu() for key, value in module.items()} for module in source_only],
        "T2_BlockEmpiricalFisher": [{key: value.detach().cpu() for key, value in module.items()} for module in fisher["T2_AdamW_full_state"]],
    }
    return canonical, result, trace, tensor_tree_hash(optimizer.state_dict()), tensor_tree_hash(capture_rng())


def _authority(root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_config(config_path); benchmark_path = root / config["benchmark_config"]; benchmark = load_benchmark_config(benchmark_path); authority = config["authority"]
    _, contract, registry = validate_prepared(root, benchmark, authority["benchmark_name"]); scenario = behavior._scenario(registry, "L8")
    original_run, original_manifest, original = behavior._validate_reference(root, benchmark, "Original-Short", authority["original_run_name"], authority["benchmark_name"], "L8")
    exact_run, exact_manifest, exact = behavior._validate_reference(root, benchmark, "Retrain-Short", authority["exact_masked_run_name"], authority["benchmark_name"], "L8")
    bota_run, bota_manifest, bota = behavior._validate_reference(root, benchmark, "BOTA-T2-Short", authority["formal_bota_run_name"], authority["benchmark_name"], "L8")
    if not behavior._same_coordinate(original, exact) or not behavior._same_coordinate(original, bota): raise ValueError("Fisher-ablation fixed-A mismatch")
    if any(row.get("request_hash") != scenario["request_hash"] for row in (original_manifest, exact_manifest, bota_manifest)): raise ValueError("Fisher-ablation request mismatch")
    pre = {"schema": SCHEMA, "authority": authority, "benchmark_config_sha256": sha256_file(benchmark_path), "registry_sha256": registry["registry_sha256"], "request_hash": scenario["request_hash"], "optimizer_steps": contract["optimizer_steps"], "references": {"original": {"run": str(original_run), "adapter_sha256": original_manifest["artifact"]["sha256"]}, "exact_masked": {"run": str(exact_run), "adapter_sha256": exact_manifest["artifact"]["sha256"]}, "formal_bota": {"run": str(bota_run), "adapter_sha256": bota_manifest["artifact"]["sha256"]}}, "config_sha256": sha256_file(config_path), "test_accessed": False}
    return pre, benchmark, registry


def classify(parameter: dict[str, Any], metrics: dict[str, dict[str, Any]], decision: dict[str, Any]) -> dict[str, Any]:
    source_p, fisher_p = parameter[ARMS[0]], parameter[ARMS[1]]; source_b, fisher_b = metrics[ARMS[0]], metrics[ARMS[1]]
    p_gain = source_p["relative_l2_error"] - fisher_p["relative_l2_error"]
    b_gain = abs(source_b["local_residual"]) - abs(fisher_b["local_residual"])
    utility_safe = fisher_b["auc_damage_vs_original"] <= source_b["auc_damage_vs_original"] + decision["auc_damage_noninferiority_margin"] and fisher_b["log_loss_damage_vs_original"] <= source_b["log_loss_damage_vs_original"] + decision["log_loss_damage_noninferiority_margin"]
    if p_gain >= decision["parameter_relative_l2_material_margin"] and b_gain >= decision["behavioral_absolute_residual_material_margin"] and utility_safe: label = "block_empirical_fisher_materially_helpful_single_seed"
    elif p_gain <= decision["parameter_relative_l2_material_margin"] and b_gain <= decision["behavioral_absolute_residual_material_margin"]: label = "source_only_explains_primary_effect_single_seed"
    else: label = "mixed_fisher_contribution_single_seed"
    return {"classification": label, "parameter_relative_l2_improvement": p_gain, "behavioral_absolute_residual_improvement": b_gain, "fisher_utility_noninferior": utility_safe, "single_seed_stability_claimed": False, "posthoc_candidate_selection_used": False}


def execute(root: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = load_config(config_path); pre, benchmark, registry = _authority(root, config_path); git = git_snapshot(root)
    if not git["clean"]: raise RuntimeError("formal Fisher ablation requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("exactly one CUDA GPU required")
    torch.cuda.set_device(0); device = torch.device("cuda:0"); free, total = torch.cuda.mem_get_info(device)
    if free / 2**30 < config["runtime"]["minimum_free_gib"]: raise RuntimeError("insufficient dedicated GPU memory")
    torch.cuda.set_per_process_memory_fraction(runner.allocator_fraction_for(total / 2**30, config["runtime"]["allocator_fraction"], config["runtime"]["hard_peak_reserved_gib"]), device); torch.cuda.reset_peak_memory_stats()
    destination = root / config["output_root"] / "runs" / safe_run_name(run_name)
    if destination.exists(): raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True)
    authority = config["authority"]; scenario = behavior._scenario(registry, "L8"); original_run = Path(pre["references"]["original"]["run"]); exact_run = Path(pre["references"]["exact_masked"]["run"]); bota_run = Path(pre["references"]["formal_bota"]["run"])
    original_state = torch.load(original_run / "scenarios/L8/adapter/adapter_model.pt", map_location="cpu", weights_only=True); exact_state = torch.load(exact_run / "scenarios/L8/adapter/adapter_model.pt", map_location="cpu", weights_only=True); formal_bota = torch.load(bota_run / "scenarios/L8/adapter/adapter_model.pt", map_location="cpu", weights_only=True)
    source_checkpoint = root / benchmark["source"]["original_checkpoint"]; source_before = sha256_file(source_checkpoint); started = time.perf_counter(); model = None
    try:
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        init_started = time.perf_counter(); model, names, parameters, bases, _, tokenizer, dataset, users = runner._model_context(root, benchmark, registry, device, original_state["A"]); initialization = time.perf_counter() - init_started
        budget = StepBudget(200); trajectory_started = time.perf_counter(); canonical, states, trace, optimizer_hash, rng_hash = run_shared_canonical(model, dataset, registry["order"], users, scenario["users"], parameters, names, device, tokenizer.pad_token_id, runner._engine(root, benchmark), budget); trajectory_seconds = time.perf_counter() - trajectory_started
        if budget.calls != 200 or budget.by_arm != {"shared_canonical_reference": 200}: raise RuntimeError("Fisher-ablation step budget mismatch")
        if any(not torch.equal(canonical[name], original_state["B"][name]) for name in names): raise RuntimeError("shared canonical endpoint differs from formal Original")
        actual = torch.cat([(exact_state["B"][name].float() - canonical[name]).reshape(-1) for name in names]); candidates = {}; parameter_metrics = {}; adapter_hashes = {}
        for arm in ARMS:
            candidate = {name: canonical[name] + states[arm][module]["theta"].sum(0).float() for module, name in enumerate(names)}; candidates[arm] = candidate; predicted = torch.cat([(candidate[name] - canonical[name]).reshape(-1) for name in names]); parameter_metrics[arm] = p2b.vector_metrics(actual, predicted); adapter_hashes[arm] = runner._save_fixed_ab(stage / "adapters" / arm / "adapter", names, candidate, bases, DISPLAY[arm])["sha256"]
        formal_bota_exact = all(torch.equal(candidates["T2_BlockEmpiricalFisher"][name], formal_bota["B"][name]) for name in names)
        formal_delta = torch.cat([(formal_bota["B"][name].float() - original_state["B"][name].float()).reshape(-1) for name in names])
        reconstructed_delta = torch.cat([(candidates["T2_BlockEmpiricalFisher"][name] - canonical[name]).reshape(-1) for name in names])
        formal_bota_alignment = p2b.vector_metrics(formal_delta, reconstructed_delta)
        del model, parameters, states; model = None; gc.collect(); torch.cuda.empty_cache()
        evaluation_started = time.perf_counter(); base = runner._base_t5_config(root, benchmark); eval_tokenizer = T5Tokenizer.from_pretrained(base["paths"]["model_dir"]); development = JsonPromptDataset(root / benchmark["source"]["development_json"], eval_tokenizer); _, development_rows, _ = reconstruct_authoritative_rows(root / benchmark["source"]["raw_data"]); development_users = [int(row.authoritative_user_id) for row in development_rows]; indices = list(range(len(development))); selected = [index for index, user in enumerate(development_users) if user in set(scenario["users"])]
        if len(development) != 20000 or len(selected) < 80: raise RuntimeError("Development support mismatch")
        original_prediction = behavior._predict_reference(root, benchmark, original_run, "L8", development, indices, device); exact_prediction = behavior._predict_reference(root, benchmark, exact_run, "L8", development, indices, device); metric_rows = []; metric_map = {}
        if original_prediction["gold_label"] != exact_prediction["gold_label"] or original_prediction["sample_order_sha256"] != exact_prediction["sample_order_sha256"]: raise RuntimeError("Development reference order mismatch")
        for number, arm in enumerate(ARMS):
            prediction = behavior._predict_scenario_dir(root, benchmark, stage / "adapters" / arm, development, indices, device)
            if prediction["gold_label"] != exact_prediction["gold_label"] or prediction["sample_order_sha256"] != exact_prediction["sample_order_sha256"]: raise RuntimeError(f"{arm} Development order mismatch")
            row = behavior.behavior_metrics(DISPLAY[arm], prediction, exact_prediction, original_prediction, selected, development_users, config["evaluation"]["bootstrap_resamples"], config["evaluation"]["bootstrap_seed"] + number); metric_rows.append(row); metric_map[arm] = row
        evaluation_seconds = time.perf_counter() - evaluation_started; decision = classify(parameter_metrics, metric_map, config["decision"]); peak = torch.cuda.max_memory_reserved() / 2**30
        if peak > config["runtime"]["hard_peak_reserved_gib"] or sha256_file(source_checkpoint) != source_before: raise RuntimeError("Fisher-ablation integrity gate failed")
        atomic_json(stage / "metrics.json", {"schema": SCHEMA, "scenario": "L8", "reference": "Exact-Masked-Reference-200step", "parameter_transport": parameter_metrics, "formal_bota_alignment": formal_bota_alignment, "formal_bota_tensor_exact": formal_bota_exact, "behavior": metric_rows, "decision": decision, "test_accessed": False})
        with (stage / "metrics.csv").open("w", encoding="utf-8", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0])); writer.writeheader(); writer.writerows(metric_rows)
        timing = {"initialization_seconds": initialization, "shared_offline_trajectory_seconds": trajectory_seconds, "development_evaluation_seconds": evaluation_seconds, "end_to_end_seconds": time.perf_counter() - started}; atomic_json(stage / "timing.json", timing)
        lines = ["# ML-1M L8 minimal Fisher ablation", "", "Both arms share the exact same 200-step canonical trajectory and T2 AdamW state linearization; only the gradient-response closure differs.", "", "| Arm | Delta cosine | Norm ratio | Relative L2 | Y/N JSD | L2 RMS | Local residual [95% CI] | AUC | LogLoss |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm, row in zip(ARMS, metric_rows):
            p = parameter_metrics[arm]; lines.append(f"| {DISPLAY[arm]} | {behavior._fmt(p['cosine'])} | {behavior._fmt(p['norm_ratio'])} | {behavior._fmt(p['relative_l2_error'])} | {behavior._fmt(row['yes_no_jsd_to_exact'])} | {behavior._fmt(row['probability_l2_rms_to_exact'])} | {behavior._fmt(row['local_residual'],4)} [{behavior._fmt(row['ci_lower'],4)}, {behavior._fmt(row['ci_upper'],4)}] | {behavior._fmt(row['overall_auc'])} | {behavior._fmt(row['overall_log_loss'])} |")
        lines.extend(["", f"Classification: `{decision['classification']}`.", "", "This is a single-seed explanatory ablation, not evidence of cross-seed stability and not a post-hoc model-selection run.", ""]); (stage / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
        atomic_json(stage / "contract.json", pre); atomic_json(stage / "provenance.json", {"schema": SCHEMA, "git": git, "source_checkpoint_sha256": source_before, "canonical_trace_sha256": canonical_hash(trace), "optimizer_state_sha256": optimizer_hash, "rng_sha256": rng_hash, "formal_bota_tensor_exact": formal_bota_exact, "formal_bota_alignment": formal_bota_alignment, "development_accessed": True, "final_test_accessed": False, "test_loader_built": False, "posthoc_candidate_selection_used": False, "test_accessed": False})
        state = {"schema": SCHEMA, "status": "COMPLETED", "scenario": "L8", "arms": list(ARMS), "physical_optimizer_step_calls": 200, "canonical_optimizer_steps": 200, "online_optimizer_steps": 0, "authoritative_optimizer_steps_committed": 0, "peak_reserved_gib": peak, "classification": decision["classification"], "test_accessed": False}; atomic_json(stage / "run_state.json", state); (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        files = {name: sha256_file(stage / name) for name in ("COMPLETED", "contract.json", "metrics.csv", "metrics.json", "provenance.json", "report.md", "run_state.json", "timing.json")}; atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": files, "adapter_sha256": adapter_hashes, "published_atomically": True, "test_accessed": False}); destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination)
        return {"status": "COMPLETED", "run_dir": str(destination), **decision, "physical_optimizer_step_calls": 200, "test_accessed": False}
    except Exception as error:
        shutil.rmtree(stage, ignore_errors=True); raise RuntimeError("ML-1M Fisher ablation failed before atomic publication") from error
    finally:
        if model is not None: del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def analyze(root: Path, config: dict[str, Any], run_name: str) -> dict[str, Any]:
    run = root / config["output_root"] / "runs" / safe_run_name(run_name)
    if not run.is_dir() or {path.name for path in run.iterdir()} != REQUIRED or (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n": raise ValueError("invalid Fisher-ablation run")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")); state = json.loads((run / "run_state.json").read_text(encoding="utf-8")); metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if any(sha256_file(run / name) != expected for name, expected in manifest["files"].items()) or state.get("physical_optimizer_step_calls") != 200 or metrics.get("test_accessed") is not False: raise ValueError("Fisher-ablation evidence mismatch")
    for arm, expected in manifest["adapter_sha256"].items():
        if arm not in ARMS or directory_hash(run / "adapters" / arm / "adapter") != expected: raise ValueError("Fisher-ablation adapter mismatch")
    return {"status": "COMPLETED", "run_dir": str(run), "classification": state["classification"], "parameter_transport": metrics["parameter_transport"], "formal_bota_alignment": metrics["formal_bota_alignment"], "formal_bota_tensor_exact": metrics["formal_bota_tensor_exact"], "behavior": metrics["behavior"], "decision": metrics["decision"], "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path.cwd()); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--mode", choices=("Preflight", "SyntheticDryRun", "Full", "Analyze"), default="Preflight"); parser.add_argument("--run-name", default=""); args = parser.parse_args(); root, path = args.root.resolve(), args.config.resolve(); config = load_config(path)
    if args.mode == "Preflight": result = _authority(root, path)[0]
    elif args.mode == "SyntheticDryRun": result = {"schema": SCHEMA, "status": "COMPLETED", "arms": list(ARMS), "physical_optimizer_step_calls": 0, "real_model_loaded": False, "test_accessed": False}
    elif not args.run_name: parser.error(f"{args.mode} requires RunName")
    elif args.mode == "Analyze": result = analyze(root, config, args.run_name)
    else: result = execute(root, path, args.run_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
