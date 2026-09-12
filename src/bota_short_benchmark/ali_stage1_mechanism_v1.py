"""Development-only Stage 1 mechanism stabilization for the Ali amendment."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import shutil
import statistics
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
from src.bota_short_benchmark import runner
from src.bota_short_benchmark import t012_behavior_ablation as behavior
from src.bota_short_benchmark.protocol import load_config as load_benchmark_config
from src.bota_short_benchmark.protocol import validate_prepared
from src.diagnostics.ml1m_development_protocol import reconstruct_authoritative_rows
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, move_batch
from src.if_a2_optimization.group_a_gradient_audit import masked_batch
from src.paper_baselines.common import capture_rng, tensor_tree_hash
from src.paper_if_a2.common import (
    atomic_json,
    canonical_hash,
    directory_hash,
    git_snapshot,
    safe_run_name,
    sha256_file,
)


SCHEMA = "bota-ali-stage1-mechanism-v1"
MARKER = "BOTA_ALI_STAGE1_MECHANISM_V1_COMPLETED"
SUMMARY_SCHEMA = "bota-ali-stage1-mechanism-summary-v1"
SUMMARY_MARKER = "BOTA_ALI_STAGE1_MECHANISM_SUMMARY_V1_COMPLETED"
SEEDS = (41, 42, 43)
SCENARIOS = ("L8", "L4M4")
ARMS = ("FM_source_only", "FS_source_only", "FS_empirical_fisher")
DISPLAY = {
    "FM_source_only": "BOTA-FM-SourceOnly",
    "FS_source_only": "BOTA-FS-SourceOnly",
    "FS_empirical_fisher": "BOTA-FS-EmpiricalFisher",
}
RUN_REQUIRED = {
    "COMPLETED",
    "adapters",
    "contract.json",
    "manifest.json",
    "metrics.csv",
    "metrics.json",
    "per_sample_metrics.jsonl",
    "provenance.json",
    "report.md",
    "run_state.json",
    "timing.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_config(repo: Path, path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "test_access_policy",
        "parent_amendment",
        "output_root",
        "benchmark_configs",
        "protocol",
        "evaluation",
        "decision",
        "formal_reproduction",
        "runtime",
    }
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA:
        raise ValueError("invalid Stage 1 mechanism configuration")
    if value["test_access_policy"] != "forbidden":
        raise ValueError("Stage 1 must forbid FinalTest")
    parent = value["parent_amendment"]
    parent_path = repo / parent["path"]
    if not parent_path.is_file() or sha256_file(parent_path) != parent["sha256"]:
        raise ValueError("Stage 0 Amendment 01 binding changed")
    parent_value = _load_json(parent_path)
    if parent_value.get("amendment_id") != parent["amendment_id"]:
        raise ValueError("Stage 0 Amendment 01 identity changed")
    protocol = value["protocol"]
    if protocol != {
        "seeds": list(SEEDS),
        "scenarios": list(SCENARIOS),
        "arms": list(ARMS),
        "optimizer_steps": 200,
        "batch_size": 16,
        "shared_canonical_trajectory": True,
        "online_optimizer_steps": 0,
        "authoritative_optimizer_steps_committed": 0,
    }:
        raise ValueError("Stage 1 mechanism protocol changed")
    evaluation = value["evaluation"]
    if evaluation["split"] != "Development" or evaluation["final_test"] is not False:
        raise ValueError("Stage 1 evaluation is not Development-only")
    if set(int(seed) for seed in value["benchmark_configs"]) != set(SEEDS):
        raise ValueError("Stage 1 benchmark registry changed")
    if value["formal_reproduction"] != {
        "exact_tensor_equality_required": False,
        "cosine_minimum": 0.9999,
        "relative_l2_maximum": 0.01,
        "maximum_absolute_error": 0.001,
        "scientific_selection_uses_this_gate": False,
    }:
        raise ValueError("Stage 1 formal-reproduction integrity gate changed")
    return value


def default_run_name(seed: int, scenario: str) -> str:
    return f"ali_stage1_mechanism_seed{seed}_{scenario.lower()}_v1"


def _state(parameters: Sequence[torch.Tensor], users: int) -> list[dict[str, torch.Tensor]]:
    return [
        {
            key: torch.zeros(
                (users, *parameter.shape),
                device=parameter.device,
                dtype=parameter.dtype,
            )
            for key in ("theta", "m", "v")
        }
        for parameter in parameters
    ]


def _advance_arm(
    states: list[dict[str, torch.Tensor]],
    coefficients: Sequence[torch.Tensor],
    sources: Sequence[torch.Tensor],
    parameters: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    step: int,
    engine: dict[str, Any],
    *,
    full_v: bool,
    empirical_fisher: bool,
) -> None:
    opt = engine["optimizer"]
    beta1, beta2 = map(float, opt["betas"])
    for module, (parameter, rows, source) in enumerate(zip(parameters, coefficients, sources)):
        current = states[module]
        dgradient = source
        if empirical_fisher:
            dgradient = p2b.full_fisher_product(rows, current["theta"]) + source
        optimizer_state = optimizer.state.get(parameter, {})
        moment = optimizer_state.get("exp_avg", torch.zeros_like(parameter)).detach()
        second_moment = optimizer_state.get("exp_avg_sq", torch.zeros_like(parameter)).detach()
        theta, dm, dv = p2b.stable_adamw_tangent_step(
            parameter.detach(),
            parameter.grad.detach(),
            moment,
            second_moment,
            current["theta"],
            current["m"],
            current["v"],
            dgradient,
            step=step,
            lr=float(opt["learning_rate"]),
            beta1=beta1,
            beta2=beta2,
            eps=float(opt["eps"]),
            weight_decay=float(opt["weight_decay"]),
            full_v=full_v,
        )
        current.update(theta=theta, m=dm, v=dv)


def run_shared_canonical(
    model: Any,
    dataset: Any,
    indices: Sequence[int],
    user_ids: Sequence[int],
    selected_users: Sequence[int],
    parameters: Sequence[torch.Tensor],
    names: Sequence[str],
    device: torch.device,
    pad: int,
    engine: dict[str, Any],
    budget: StepBudget,
    arms: Sequence[str] = ARMS,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]], str, str]:
    optimizer = p1._optimizer(parameters, engine)
    requested_arms = tuple(arms)
    if not requested_arms or len(set(requested_arms)) != len(requested_arms) or any(arm not in ARMS for arm in requested_arms):
        raise ValueError("invalid Stage 1 transport-arm subset")
    states = {arm: _state(parameters, len(selected_users)) for arm in requested_arms}
    trace: list[dict[str, Any]] = []
    batch_size = int(engine["schedule"]["batch_size"])
    for step, start in enumerate(range(0, len(indices), batch_size), 1):
        chosen = list(indices[start : start + batch_size])
        batch_users = [int(user_ids[index]) for index in chosen]
        batch = move_batch(masked_batch(dataset, chosen, pad), device)
        losses = p1._sample_losses(model, batch)
        sample_rows: list[list[torch.Tensor]] = [[] for _ in parameters]
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
            user_sources = []
            for target in selected_users:
                slots = [slot for slot, user in enumerate(batch_users) if user == target]
                user_sources.append(
                    -rows[slots].sum(0) / len(chosen) if slots else torch.zeros_like(parameter)
                )
            sources.append(torch.stack(user_sources))
        if "FM_source_only" in states:
            _advance_arm(
                states["FM_source_only"], coefficients, sources, parameters, optimizer,
                step, engine, full_v=False, empirical_fisher=False,
            )
        if "FS_source_only" in states:
            _advance_arm(
                states["FS_source_only"], coefficients, sources, parameters, optimizer,
                step, engine, full_v=True, empirical_fisher=False,
            )
        if "FS_empirical_fisher" in states:
            _advance_arm(
                states["FS_empirical_fisher"], coefficients, sources, parameters, optimizer,
                step, engine, full_v=True, empirical_fisher=True,
            )
        budget.step(optimizer, "shared_canonical_reference")
        trace.append(
            {
                "step": step,
                "batch_hash": canonical_hash(chosen),
                "selected_user_slot_counts": [batch_users.count(user) for user in selected_users],
            }
        )
        del batch, losses, formal_loss, sample_rows, coefficients, sources
    canonical = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in zip(names, parameters)
    }
    cpu_states = {
        arm: [
            {key: value.detach().cpu() for key, value in module.items()}
            for module in modules
        ]
        for arm, modules in states.items()
    }
    return (
        canonical,
        cpu_states,
        trace,
        tensor_tree_hash(optimizer.state_dict()),
        tensor_tree_hash(capture_rng()),
    )


def _artifact_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def preflight(
    repo: Path,
    workspace: Path,
    config_path: Path,
    seed: int,
    scenario_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_config(repo, config_path)
    if seed not in SEEDS or scenario_id not in SCENARIOS:
        raise ValueError("unsupported Stage 1 seed or scenario")
    benchmark_path = repo / config["benchmark_configs"][seed]
    benchmark = load_benchmark_config(benchmark_path)
    if int(benchmark["protocol"]["run_seed"]) != seed:
        raise ValueError("benchmark run-seed binding changed")
    base = runner._base_t5_config(workspace, benchmark)
    model_dir = Path(base["paths"]["model_dir"])
    if not model_dir.is_absolute():
        model_dir = workspace / model_dir
    required_model_files = ("config.json", "pytorch_model.bin", "spiece.model", "tokenizer_config.json")
    missing_model_files = [name for name in required_model_files if not (model_dir / name).is_file()]
    if missing_model_files:
        raise ValueError(f"local T5-base directory is incomplete: {missing_model_files}")
    benchmark_name = f"bota_short_i02_seed{seed}_v3"
    _, contract, registry = validate_prepared(workspace, benchmark, benchmark_name)
    scenario = behavior._scenario(registry, scenario_id)
    run_names = {
        "original": f"bota_short_original_seed{seed}_v3",
        "exact_masked": f"bota_short_exact_masked_seed{seed}_v3",
        "formal_bota": f"bota_short_bota_seed{seed}_v3",
    }
    original_run, original_manifest, original = behavior._validate_reference(
        workspace, benchmark, "Original-Short", run_names["original"], benchmark_name, scenario_id
    )
    exact_run, exact_manifest, exact = behavior._validate_reference(
        workspace, benchmark, "Retrain-Short", run_names["exact_masked"], benchmark_name, scenario_id
    )
    bota_run, bota_manifest, bota = behavior._validate_reference(
        workspace, benchmark, "BOTA-T2-Short", run_names["formal_bota"], benchmark_name, scenario_id
    )
    if not behavior._same_coordinate(original, exact) or not behavior._same_coordinate(original, bota):
        raise ValueError("Stage 1 fixed-A coordinate mismatch")
    manifests = (original_manifest, exact_manifest, bota_manifest)
    if any(row.get("request_hash") != scenario["request_hash"] for row in manifests):
        raise ValueError("Stage 1 request hash mismatch")
    authority = {
        "schema": SCHEMA,
        "parent_amendment": config["parent_amendment"],
        "benchmark_name": benchmark_name,
        "benchmark_config": str(benchmark_path),
        "benchmark_config_sha256": sha256_file(benchmark_path),
        "local_model_dir": str(model_dir.resolve()),
        "seed": seed,
        "scenario": scenario_id,
        "request_hash": scenario["request_hash"],
        "registry_sha256": registry["registry_sha256"],
        "optimizer_steps": contract["optimizer_steps"],
        "references": {
            "original": {"run": str(original_run), "adapter_sha256": original_manifest["artifact"]["sha256"]},
            "exact_masked": {"run": str(exact_run), "adapter_sha256": exact_manifest["artifact"]["sha256"]},
            "formal_bota": {"run": str(bota_run), "adapter_sha256": bota_manifest["artifact"]["sha256"]},
        },
        "config_sha256": sha256_file(config_path),
        "development_accessed": False,
        "final_test_accessed": False,
        "test_accessed": False,
    }
    return authority, benchmark, registry


def execute(
    repo: Path,
    workspace: Path,
    config_path: Path,
    seed: int,
    scenario_id: str,
    run_name: str,
) -> dict[str, Any]:
    config = load_config(repo, config_path)
    authority, benchmark, registry = preflight(repo, workspace, config_path, seed, scenario_id)
    code_git = git_snapshot(repo)
    if config["runtime"]["code_repository_must_be_clean"] and not code_git["clean"]:
        raise RuntimeError("formal Stage 1 run requires a clean BOTA code repository")
    workspace_git = git_snapshot(workspace)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal Stage 1 run requires exactly one CUDA GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free / 2**30 < float(config["runtime"]["minimum_free_gib"]):
        raise RuntimeError("insufficient free GPU memory for Stage 1")
    fraction = runner.allocator_fraction_for(
        total / 2**30,
        float(config["runtime"]["allocator_fraction"]),
        float(config["runtime"]["hard_peak_reserved_gib"]),
    )
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    torch.cuda.reset_peak_memory_stats()

    destination = workspace / config["output_root"] / "runs" / safe_run_name(run_name)
    if destination.exists():
        raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True)
    scenario = behavior._scenario(registry, scenario_id)
    original_run = Path(authority["references"]["original"]["run"])
    exact_run = Path(authority["references"]["exact_masked"]["run"])
    bota_run = Path(authority["references"]["formal_bota"]["run"])
    original_state = torch.load(
        original_run / "scenarios" / scenario_id / "adapter" / "adapter_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    exact_state = torch.load(
        exact_run / "scenarios" / scenario_id / "adapter" / "adapter_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    formal_bota = torch.load(
        bota_run / "scenarios" / scenario_id / "adapter" / "adapter_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    source_checkpoint = workspace / benchmark["source"]["original_checkpoint"]
    source_before = sha256_file(source_checkpoint)
    started = time.perf_counter()
    model = None
    original_working_directory = Path.cwd()
    os.chdir(workspace)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        initialization_started = time.perf_counter()
        model, names, parameters, bases, _, tokenizer, train_data, train_users = runner._model_context(
            workspace, benchmark, registry, device, original_state["A"]
        )
        initialization_seconds = time.perf_counter() - initialization_started
        engine = runner._engine(workspace, benchmark)
        budget = StepBudget(200)
        trajectory_started = time.perf_counter()
        canonical, states, trace, optimizer_hash, rng_hash = run_shared_canonical(
            model,
            train_data,
            registry["order"],
            train_users,
            scenario["users"],
            parameters,
            names,
            device,
            tokenizer.pad_token_id,
            engine,
            budget,
        )
        trajectory_seconds = time.perf_counter() - trajectory_started
        if budget.calls != 200 or budget.by_arm != {"shared_canonical_reference": 200}:
            raise RuntimeError("Stage 1 optimizer-step budget mismatch")
        if any(not torch.equal(canonical[name], original_state["B"][name]) for name in names):
            raise RuntimeError("Stage 1 canonical endpoint differs from formal Original")

        actual = torch.cat(
            [(exact_state["B"][name].float() - canonical[name]).reshape(-1) for name in names]
        )
        candidates: dict[str, dict[str, torch.Tensor]] = {}
        parameter_metrics: dict[str, dict[str, Any]] = {}
        adapter_hashes: dict[str, str] = {}
        artifact_bytes: dict[str, int] = {}
        publication_seconds: dict[str, float] = {}
        for arm in ARMS:
            candidate = {
                name: canonical[name] + states[arm][module]["theta"].sum(0).float()
                for module, name in enumerate(names)
            }
            candidates[arm] = candidate
            predicted = torch.cat([(candidate[name] - canonical[name]).reshape(-1) for name in names])
            parameter_metrics[arm] = p2b.vector_metrics(actual, predicted)
            publish_started = time.perf_counter()
            adapter_dir = stage / "adapters" / arm / "adapter"
            adapter_hashes[arm] = runner._save_fixed_ab(
                adapter_dir, names, candidate, bases, DISPLAY[arm]
            )["sha256"]
            publication_seconds[arm] = time.perf_counter() - publish_started
            artifact_bytes[arm] = _artifact_bytes(adapter_dir)

        formal_bota_exact = all(
            torch.equal(candidates["FS_empirical_fisher"][name], formal_bota["B"][name])
            for name in names
        )
        formal_delta = torch.cat(
            [
                (formal_bota["B"][name].float() - original_state["B"][name].float()).reshape(-1)
                for name in names
            ]
        )
        reconstructed_delta = torch.cat(
            [
                (candidates["FS_empirical_fisher"][name] - canonical[name]).reshape(-1)
                for name in names
            ]
        )
        formal_bota_alignment = p2b.vector_metrics(formal_delta, reconstructed_delta)
        reproduction = config["formal_reproduction"]
        formal_bota_reproduction_gate = bool(
            formal_bota_alignment["finite"]
            and formal_bota_alignment["cosine"] >= float(reproduction["cosine_minimum"])
            and formal_bota_alignment["relative_l2_error"]
            <= float(reproduction["relative_l2_maximum"])
            and formal_bota_alignment["maximum_absolute_error"]
            <= float(reproduction["maximum_absolute_error"])
        )
        if not formal_bota_reproduction_gate:
            raise RuntimeError(
                "Stage 1 FS+Fisher failed the numerical formal-BOTA reproduction gate: "
                + json.dumps(formal_bota_alignment, sort_keys=True)
            )
        del model, parameters, states, candidates
        model = None
        gc.collect()
        torch.cuda.empty_cache()

        evaluation_started = time.perf_counter()
        base = runner._base_t5_config(workspace, benchmark)
        eval_tokenizer = T5Tokenizer.from_pretrained(base["paths"]["model_dir"])
        development = JsonPromptDataset(
            workspace / benchmark["source"]["development_json"], eval_tokenizer
        )
        _, development_rows, _ = reconstruct_authoritative_rows(
            workspace / benchmark["source"]["raw_data"]
        )
        development_users = [int(row.authoritative_user_id) for row in development_rows]
        indices = list(range(len(development)))
        selected = [
            index
            for index, user in enumerate(development_users)
            if user in set(scenario["users"])
        ]
        if len(development) != 20000 or len(selected) < 80:
            raise RuntimeError("Stage 1 Development support mismatch")
        original_prediction = behavior._predict_reference(
            workspace, benchmark, original_run, scenario_id, development, indices, device
        )
        exact_prediction = behavior._predict_reference(
            workspace, benchmark, exact_run, scenario_id, development, indices, device
        )
        if (
            original_prediction["gold_label"] != exact_prediction["gold_label"]
            or original_prediction["sample_order_sha256"]
            != exact_prediction["sample_order_sha256"]
        ):
            raise RuntimeError("Stage 1 Development reference order mismatch")
        metric_rows: list[dict[str, Any]] = []
        safe_samples: list[dict[str, Any]] = []
        for number, arm in enumerate(ARMS):
            prediction = behavior._predict_scenario_dir(
                workspace, benchmark, stage / "adapters" / arm, development, indices, device
            )
            if (
                prediction["gold_label"] != exact_prediction["gold_label"]
                or prediction["sample_order_sha256"]
                != exact_prediction["sample_order_sha256"]
            ):
                raise RuntimeError(f"{arm} Development order mismatch")
            row = behavior.behavior_metrics(
                DISPLAY[arm],
                prediction,
                exact_prediction,
                original_prediction,
                selected,
                development_users,
                int(config["evaluation"]["bootstrap_resamples"]),
                int(config["evaluation"]["bootstrap_seed"]) + seed * 10 + number,
            )
            probability = np.asarray(prediction["probability"], dtype=np.float64)
            exact_probability = np.asarray(exact_prediction["probability"], dtype=np.float64)
            original_probability = np.asarray(original_prediction["probability"], dtype=np.float64)
            chosen = np.asarray(selected, dtype=np.int64)
            row.update(
                arm=arm,
                seed=seed,
                scenario=scenario_id,
                absolute_counterfactual_error_EU=float(
                    np.mean(np.abs(probability[chosen] - exact_probability[chosen]))
                ),
                shared_build_wall_time_seconds=trajectory_seconds,
                adapter_publication_seconds=publication_seconds[arm],
                artifact_bytes=artifact_bytes[arm],
            )
            metric_rows.append(row)
            safe_samples.extend(
                {
                    "arm": arm,
                    "development_index": index,
                    "user_hash": canonical_hash([seed, development_users[index]]),
                    "probability": float(probability[index]),
                    "exact_probability": float(exact_probability[index]),
                    "original_probability": float(original_probability[index]),
                }
                for index in selected
            )
        evaluation_seconds = time.perf_counter() - evaluation_started
        peak = torch.cuda.max_memory_reserved() / 2**30
        if peak > float(config["runtime"]["hard_peak_reserved_gib"]):
            raise RuntimeError("Stage 1 GPU hard cap exceeded")
        if sha256_file(source_checkpoint) != source_before:
            raise RuntimeError("Stage 1 source checkpoint changed")

        atomic_json(
            stage / "metrics.json",
            {
                "schema": SCHEMA,
                "seed": seed,
                "scenario": scenario_id,
                "reference": "Exact-Masked-Reference-200step",
                "parameter_transport": parameter_metrics,
                "behavior": metric_rows,
                "formal_bota_tensor_exact": formal_bota_exact,
                "formal_bota_alignment": formal_bota_alignment,
                "formal_bota_reproduction_gate_passed": formal_bota_reproduction_gate,
                "test_accessed": False,
            },
        )
        with (stage / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)
        with (stage / "per_sample_metrics.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            for row in safe_samples:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        timing = {
            "initialization_seconds": initialization_seconds,
            "shared_offline_trajectory_seconds": trajectory_seconds,
            "adapter_publication_seconds": publication_seconds,
            "development_evaluation_seconds": evaluation_seconds,
            "end_to_end_seconds": time.perf_counter() - started,
        }
        atomic_json(stage / "timing.json", timing)
        report = [
            f"# Ali Stage 1 mechanism stabilization — seed {seed}, {scenario_id}",
            "",
            "Development only. All arms share one canonical 200-step trajectory and differ only in optimizer-state or response transport.",
            "",
            "| Arm | Delta cosine | Norm ratio | Relative L2 | EU | Local residual [95% CI] | AUC | LogLoss | Bytes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for arm, row in zip(ARMS, metric_rows):
            parameter = parameter_metrics[arm]
            report.append(
                f"| {DISPLAY[arm]} | {behavior._fmt(parameter['cosine'])} | "
                f"{behavior._fmt(parameter['norm_ratio'])} | "
                f"{behavior._fmt(parameter['relative_l2_error'])} | "
                f"{behavior._fmt(row['absolute_counterfactual_error_EU'])} | "
                f"{behavior._fmt(row['local_residual'], 4)} "
                f"[{behavior._fmt(row['ci_lower'], 4)}, {behavior._fmt(row['ci_upper'], 4)}] | "
                f"{behavior._fmt(row['overall_auc'])} | "
                f"{behavior._fmt(row['overall_log_loss'])} | {row['artifact_bytes']} |"
            )
        report.extend(
            [
                "",
                "The seed-42 ablation was known before Amendment 01; cross-seed selection is deferred to the six-run summary.",
                "",
            ]
        )
        (stage / "report.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
        atomic_json(stage / "contract.json", authority)
        atomic_json(
            stage / "provenance.json",
            {
                "schema": SCHEMA,
                "code_repository_git": code_git,
                "experiment_workspace_git": workspace_git,
                "experiment_workspace_dirty_allowed": True,
                "source_checkpoint_sha256": source_before,
                "canonical_trace_sha256": canonical_hash(trace),
                "optimizer_state_sha256": optimizer_hash,
                "rng_sha256": rng_hash,
                "formal_bota_tensor_exact": formal_bota_exact,
                "formal_bota_alignment": formal_bota_alignment,
                "formal_bota_reproduction_gate_passed": formal_bota_reproduction_gate,
                "formal_reproduction_is_scientific_selection_metric": False,
                "known_seed42_result_disclosed": True,
                "development_accessed": True,
                "final_test_accessed": False,
                "test_loader_built": False,
                "test_accessed": False,
            },
        )
        state = {
            "schema": SCHEMA,
            "status": "COMPLETED",
            "seed": seed,
            "scenario": scenario_id,
            "arms": list(ARMS),
            "physical_optimizer_step_calls": 200,
            "canonical_optimizer_steps": 200,
            "online_optimizer_steps": 0,
            "authoritative_optimizer_steps_committed": 0,
            "peak_reserved_gib": peak,
            "formal_bota_tensor_exact": formal_bota_exact,
            "formal_bota_reproduction_gate_passed": formal_bota_reproduction_gate,
            "test_accessed": False,
        }
        atomic_json(stage / "run_state.json", state)
        (stage / "COMPLETED").write_text(MARKER + "\n", encoding="utf-8", newline="\n")
        files = {
            name: sha256_file(stage / name)
            for name in (
                "COMPLETED",
                "contract.json",
                "metrics.csv",
                "metrics.json",
                "per_sample_metrics.jsonl",
                "provenance.json",
                "report.md",
                "run_state.json",
                "timing.json",
            )
        }
        atomic_json(
            stage / "manifest.json",
            {
                "schema": SCHEMA,
                "files": files,
                "adapter_sha256": adapter_hashes,
                "published_atomically": True,
                "test_accessed": False,
            },
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, destination)
        return {
            "status": "COMPLETED",
            "run_dir": str(destination),
            "seed": seed,
            "scenario": scenario_id,
            "arms": list(ARMS),
            "formal_bota_tensor_exact": formal_bota_exact,
            "formal_bota_reproduction_gate_passed": True,
            "physical_optimizer_step_calls": 200,
            "test_accessed": False,
        }
    except Exception as error:
        shutil.rmtree(stage, ignore_errors=True)
        raise RuntimeError("Stage 1 mechanism run failed before atomic publication") from error
    finally:
        os.chdir(original_working_directory)
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze(config: dict[str, Any], workspace: Path, run_name: str) -> dict[str, Any]:
    run = workspace / config["output_root"] / "runs" / safe_run_name(run_name)
    if not run.is_dir() or {item.name for item in run.iterdir()} != RUN_REQUIRED:
        raise ValueError(f"invalid Stage 1 run layout: {run}")
    if (run / "COMPLETED").read_text(encoding="utf-8") != MARKER + "\n":
        raise ValueError("invalid Stage 1 completion marker")
    manifest = _load_json(run / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("test_accessed") is not False:
        raise ValueError("invalid Stage 1 manifest")
    for name, expected in manifest.get("files", {}).items():
        if sha256_file(run / name) != expected:
            raise ValueError(f"Stage 1 artifact mismatch: {name}")
    for arm, expected in manifest.get("adapter_sha256", {}).items():
        if arm not in ARMS or directory_hash(run / "adapters" / arm / "adapter") != expected:
            raise ValueError(f"Stage 1 adapter mismatch: {arm}")
    state = _load_json(run / "run_state.json")
    metrics = _load_json(run / "metrics.json")
    if (
        state.get("status") != "COMPLETED"
        or state.get("physical_optimizer_step_calls") != 200
        or state.get("formal_bota_reproduction_gate_passed") is not True
        or metrics.get("test_accessed") is not False
    ):
        raise ValueError("Stage 1 evidence-state mismatch")
    if [row.get("arm") for row in metrics.get("behavior", [])] != list(ARMS):
        raise ValueError("Stage 1 behavior arm order mismatch")
    return {
        "status": "COMPLETED",
        "run_dir": str(run),
        "seed": state["seed"],
        "scenario": state["scenario"],
        "parameter_transport": metrics["parameter_transport"],
        "formal_bota_alignment": metrics["formal_bota_alignment"],
        "behavior": metrics["behavior"],
        "test_accessed": False,
    }


def _mean(rows: Sequence[float]) -> float:
    return float(statistics.mean(float(value) for value in rows))


def _stdev(rows: Sequence[float]) -> float:
    return float(statistics.stdev(float(value) for value in rows))


def decide(summary: dict[str, dict[str, dict[str, float]]], decision: dict[str, Any]) -> dict[str, Any]:
    residual_margin = float(decision["residual_noninferiority_margin"])
    parameter_margin = float(decision["relative_l2_noninferiority_margin"])

    def utility_safe(arm: str) -> bool:
        return all(
            summary[scenario][arm]["overall_auc_damage_mean"]
            <= float(decision["overall_auc_damage_max"])
            and summary[scenario][arm]["overall_logloss_damage_mean"]
            <= float(decision["overall_logloss_damage_max"])
            for scenario in SCENARIOS
        )

    fisher_checks = {
        scenario: {
            "residual_difference": summary[scenario]["FS_source_only"]["local_residual_mean"]
            - summary[scenario]["FS_empirical_fisher"]["local_residual_mean"],
            "relative_l2_difference": summary[scenario]["FS_source_only"]["relative_l2_mean"]
            - summary[scenario]["FS_empirical_fisher"]["relative_l2_mean"],
        }
        for scenario in SCENARIOS
    }
    remove_fisher = utility_safe("FS_source_only") and all(
        row["residual_difference"] <= residual_margin
        and row["relative_l2_difference"] <= parameter_margin
        for row in fisher_checks.values()
    )
    selected_fs = "FS_source_only" if remove_fisher else "FS_empirical_fisher"
    fm_checks = {
        scenario: {
            "residual_difference": summary[scenario]["FM_source_only"]["local_residual_mean"]
            - summary[scenario][selected_fs]["local_residual_mean"],
            "relative_l2_difference": summary[scenario]["FM_source_only"]["relative_l2_mean"]
            - summary[scenario][selected_fs]["relative_l2_mean"],
        }
        for scenario in SCENARIOS
    }
    promote_fm = utility_safe("FM_source_only") and all(
        row["residual_difference"] < 0
        and row["relative_l2_difference"] <= parameter_margin
        for row in fm_checks.values()
    )
    selected = "FM_source_only" if promote_fm else selected_fs
    return {
        "selected_arm": selected,
        "fisher_removed": remove_fisher,
        "fm_promoted": promote_fm,
        "fisher_checks": fisher_checks,
        "fm_checks_against_selected_fs": fm_checks,
        "utility_safe": {arm: utility_safe(arm) for arm in ARMS},
        "selection_frozen_before_attack": True,
        "known_seed42_result_disclosed": True,
        "significance_claimed": False,
    }


def summarize(
    repo: Path,
    workspace: Path,
    config_path: Path,
    summary_name: str,
) -> dict[str, Any]:
    config = load_config(repo, config_path)
    records = [
        analyze(config, workspace, default_run_name(seed, scenario))
        for seed in SEEDS
        for scenario in SCENARIOS
    ]
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for scenario in SCENARIOS:
        aggregate[scenario] = {}
        for arm in ARMS:
            cells = []
            for record in records:
                if record["scenario"] != scenario:
                    continue
                behavior_row = next(row for row in record["behavior"] if row["arm"] == arm)
                parameter = record["parameter_transport"][arm]
                cells.append({**behavior_row, **parameter})
            if len(cells) != 3:
                raise ValueError(f"expected three seeds for {scenario}/{arm}")
            aggregate[scenario][arm] = {
                "local_residual_mean": _mean([row["local_residual"] for row in cells]),
                "local_residual_sample_sd": _stdev([row["local_residual"] for row in cells]),
                "absolute_counterfactual_error_EU_mean": _mean(
                    [row["absolute_counterfactual_error_EU"] for row in cells]
                ),
                "delta_cosine_mean": _mean([row["cosine"] for row in cells]),
                "norm_ratio_mean": _mean([row["norm_ratio"] for row in cells]),
                "relative_l2_mean": _mean([row["relative_l2_error"] for row in cells]),
                "overall_auc_mean": _mean([row["overall_auc"] for row in cells]),
                "overall_logloss_mean": _mean([row["overall_log_loss"] for row in cells]),
                "overall_auc_damage_mean": _mean([row["auc_damage_vs_original"] for row in cells]),
                "overall_logloss_damage_mean": _mean(
                    [row["log_loss_damage_vs_original"] for row in cells]
                ),
                "shared_build_wall_time_mean_seconds": _mean(
                    [row["shared_build_wall_time_seconds"] for row in cells]
                ),
                "artifact_bytes_mean": _mean([row["artifact_bytes"] for row in cells]),
            }
    selection = decide(aggregate, config["decision"])
    destination = workspace / config["output_root"] / "summaries" / safe_run_name(summary_name)
    if destination.exists():
        raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{safe_run_name(summary_name)}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True)
    payload = {
        "schema": SUMMARY_SCHEMA,
        "status": "COMPLETED",
        "seeds": list(SEEDS),
        "scenarios": list(SCENARIOS),
        "arms": list(ARMS),
        "aggregate": aggregate,
        "selection": selection,
        "source_runs": [record["run_dir"] for record in records],
        "development_only": True,
        "final_test_accessed": False,
        "test_accessed": False,
    }
    atomic_json(stage / "summary.json", payload)
    lines = [
        "# Ali Stage 1 mechanism stabilization summary",
        "",
        "Development-only mean +/- sample SD across seeds 41, 42 and 43.",
        "",
        "| Scenario | Arm | Residual | EU mean | Delta cosine | Norm ratio | Relative L2 | AUC | LogLoss |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for arm in ARMS:
            row = aggregate[scenario][arm]
            lines.append(
                f"| {scenario} | {DISPLAY[arm]} | "
                f"{row['local_residual_mean']:.4f} +/- {row['local_residual_sample_sd']:.4f} | "
                f"{row['absolute_counterfactual_error_EU_mean']:.6f} | "
                f"{row['delta_cosine_mean']:.6f} | {row['norm_ratio_mean']:.6f} | "
                f"{row['relative_l2_mean']:.6f} | {row['overall_auc_mean']:.6f} | "
                f"{row['overall_logloss_mean']:.6f} |"
            )
    lines.extend(
        [
            "",
            f"Frozen selected arm: `{selection['selected_arm']}`.",
            f"Empirical Fisher removed: `{selection['fisher_removed']}`.",
            f"FM promoted: `{selection['fm_promoted']}`.",
            "",
            "No significance claim is made from three seeds. The seed-42 observation was known before Amendment 01.",
            "",
        ]
    )
    (stage / "summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (stage / "COMPLETED").write_text(SUMMARY_MARKER + "\n", encoding="utf-8", newline="\n")
    atomic_json(
        stage / "manifest.json",
        {
            "schema": SUMMARY_SCHEMA,
            "files": {
                name: sha256_file(stage / name)
                for name in ("COMPLETED", "summary.json", "summary.md")
            },
            "test_accessed": False,
        },
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, destination)
    return {
        "status": "COMPLETED",
        "summary_dir": str(destination),
        "selected_arm": selection["selected_arm"],
        "fisher_removed": selection["fisher_removed"],
        "fm_promoted": selection["fm_promoted"],
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ali_stage1_mechanism_v1.yaml"))
    parser.add_argument(
        "--mode",
        choices=("Preflight", "SyntheticDryRun", "Full", "Analyze", "Summarize"),
        default="Preflight",
    )
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--scenario", choices=SCENARIOS, default="L8")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--summary-name", default="ali_stage1_mechanism_summary_v1")
    args = parser.parse_args()
    repo = args.repo.resolve()
    workspace = args.workspace.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    config_path = config_path.resolve()
    config = load_config(repo, config_path)
    run_name = args.run_name or default_run_name(args.seed, args.scenario)
    if args.mode == "Preflight":
        result = preflight(repo, workspace, config_path, args.seed, args.scenario)[0]
    elif args.mode == "SyntheticDryRun":
        result = {
            "schema": SCHEMA,
            "status": "COMPLETED",
            "arms": list(ARMS),
            "real_model_loaded": False,
            "final_test_accessed": False,
            "test_accessed": False,
        }
    elif args.mode == "Analyze":
        result = analyze(config, workspace, run_name)
    elif args.mode == "Summarize":
        result = summarize(repo, workspace, config_path, args.summary_name)
    else:
        result = execute(repo, workspace, config_path, args.seed, args.scenario, run_name)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
