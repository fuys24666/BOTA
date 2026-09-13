"""Stage 7A: preregistered Development-only baseline hyperparameter sweep.

Each method is executed independently. Candidate selection is joint across L8
and L4M4, uses only the frozen Dev-Select subset, and never opens FinalTest.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import T5Tokenizer

from src.diagnostics.ml1m_development_protocol import reconstruct_authoritative_rows
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from src.paper_baselines import partitioned
from src.paper_if_a2.common import atomic_json, canonical_hash, directory_hash, git_snapshot, safe_run_name, seed_everything, sha256_file

from . import e2urec_short_v1 as e2urec
from . import evaluation as short_eval
from . import paper_v2
from . import runner
from .protocol import load_config as load_benchmark_config
from .protocol import validate_prepared

SCHEMA = "bota-ali-stage7-baseline-sweep-v1"
MARKER = "BOTA_ALI_STAGE7_BASELINE_SWEEP_V1_COMPLETED"
METHODS = ("NegGrad", "PCGrad", "IFRU", "E2URec", "SISA", "RecEraser")
SCENARIOS = ("L8", "L4M4")
RUN_REQUIRED = {"COMPLETED", "candidate_metrics.csv", "candidate_metrics.json", "contract.json", "manifest.json", "report.md", "run_state.json", "selection.json"}
EXPECTED_GRIDS = {
    "IFRU":{"relative_damping_ratio":[.001,.01,.1],"curvature_panel_samples":[512,1024],"cg_max_iterations":40},
    "E2URec":{"teacher_steps":1200,"teacher_learning_rate":.00005,"student_learning_rate":.001,"forget_weight":[.2,.4,.6],"checkpoints":[200,500,1000]},
    "NegGrad":{"learning_rate":.001,"forget_weight":[.05,.1,.2,.4,.8],"checkpoints":[25,50,100,200]},
    "PCGrad":{"learning_rate":[.0005,.001,.002],"checkpoints":[25,50,100,200]},
    "SISA":{"shards":4,"slices":4,"learning_rate":[.0001,.0005,.001],"epochs_per_slice":[1,3,5]},
    "RecEraser":{"shards":4,"feature_dimensions":32,"learning_rate":[.0005,.001,.002],"epochs_per_slice":[1,3,5],"attention_steps":200,"attention_learning_rate":.1},
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _label(row: dict[str, Any]) -> int:
    value = str(row.get("output", row.get("target", row.get("answer", "")))).strip().lower()
    if value.startswith("yes"):
        return 1
    if value.startswith("no"):
        return 0
    raise ValueError("Development row is not a binary Yes/No target")


def _candidate_ids(config: dict[str, Any], method: str) -> list[str]:
    grid = config["grids"][method]
    if method == "NegGrad":
        return [f"neggrad_w{weight:g}_s{step}" for weight in grid["forget_weight"] for step in grid["checkpoints"]]
    if method == "PCGrad":
        return [f"pcgrad_lr{lr:g}_s{step}" for lr in grid["learning_rate"] for step in grid["checkpoints"]]
    if method == "IFRU":
        return [f"ifru_p{panel}_d{ratio:g}" for panel in grid["curvature_panel_samples"] for ratio in grid["relative_damping_ratio"]]
    if method == "E2URec":
        return [f"e2urec_w{weight:g}_s{step}" for weight in grid["forget_weight"] for step in grid["checkpoints"]]
    if method == "SISA":
        return [f"sisa_lr{lr:g}_e{epoch}" for lr in grid["learning_rate"] for epoch in grid["epochs_per_slice"]]
    return [f"receraser_lr{lr:g}_e{epoch}" for lr in grid["learning_rate"] for epoch in grid["epochs_per_slice"]]


def load_config(repo: Path, workspace: Path, path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema", "test_access_policy", "parent_amendment", "benchmark", "development_partition", "grids", "selection", "output_root", "runtime"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA or value["test_access_policy"] != "forbidden":
        raise ValueError("invalid Stage 7 config surface")
    parent = value["parent_amendment"]
    amendment = _json(repo / parent["path"])
    if sha256_file(repo / parent["path"]) != parent["sha256"] or amendment.get("amendment_id") != parent["amendment_id"] or amendment.get("status") != "FROZEN_BEFORE_STAGE7_EXECUTION":
        raise ValueError("Amendment 06 identity/status changed")
    benchmark = value["benchmark"]
    for field in ("config", "paper_config"):
        binding = benchmark[field]
        if sha256_file(repo / binding["path"]) != binding["sha256"]:
            raise ValueError(f"Stage 7 {field} binding changed")
    frozen = workspace / benchmark["frozen_development_evaluation"]["path"]
    binding = benchmark["frozen_development_evaluation"]
    if sha256_file(frozen / "manifest.json") != binding["manifest_sha256"] or sha256_file(frozen / "per_sample_metrics.jsonl") != binding["per_sample_sha256"]:
        raise ValueError("frozen Development authority changed")
    expected_methods = set(METHODS)
    if set(value["grids"]) != expected_methods or value["grids"] != EXPECTED_GRIDS or value["selection"] != {
        "scenarios": list(SCENARIOS), "auc_damage_max": .005, "logloss_damage_max": .01,
        "primary": "equal_scenario_mean_EU", "tie_break": ["compute_count", "candidate_id"],
        "no_eligible_policy": "NO_ELIGIBLE_CANDIDATE",
    }:
        raise ValueError("Stage 7 grid/selection surface changed")
    return value


def _authority(repo: Path, workspace: Path, config: dict[str, Any]):
    binding = config["benchmark"]["config"]
    benchmark = load_benchmark_config(repo / binding["path"])
    benchmark["_config_path"] = str((repo / binding["path"]).resolve())
    _, _, registry = validate_prepared(workspace, benchmark, config["benchmark"]["name"])
    return benchmark, registry


def _development_partition(repo: Path, workspace: Path, config: dict[str, Any], benchmark: dict[str, Any]):
    rows = json.loads((workspace / benchmark["source"]["development_json"]).read_text(encoding="utf-8"))
    _, development, _ = reconstruct_authoritative_rows(workspace / benchmark["source"]["raw_data"])
    users = [int(row.authoritative_user_id) for row in development]
    if len(rows) != 20000 or len(users) != 20000:
        raise RuntimeError("Development reconstruction mismatch")
    split = config["development_partition"]; fit: list[int] = []; select: list[int] = []
    for index, (row, user) in enumerate(zip(rows, users)):
        digest = canonical_hash([split["namespace"], 42, user, _label(row), index])
        (fit if int(digest, 16) % 2 == split["fit_parity"] else select).append(index)
    if len(fit) != split["fit_count"] or len(select) != split["select_count"] or canonical_hash(fit) != split["fit_index_sha256"] or canonical_hash(select) != split["select_index_sha256"]:
        raise RuntimeError("frozen Development partition changed")
    return rows, users, fit, select


def _reference_runs(workspace: Path, config: dict[str, Any]) -> dict[str, Path]:
    frozen = workspace / config["benchmark"]["frozen_development_evaluation"]["path"]
    manifest = _json(frozen / "manifest.json")
    for name, expected in manifest["files"].items():
        if sha256_file(frozen / name) != expected:
            raise ValueError(f"frozen Development file changed: {name}")
    provenance = _json(frozen / "provenance.json")
    return {key: Path(value) for key, value in provenance["model_runs"].items()}


def _predict_reference(workspace: Path, benchmark: dict[str, Any], run: Path, method_id: str, scenario: str, dataset, indices, device):
    return short_eval._predict(workspace, benchmark, run, method_id, scenario, dataset, indices, device)


def _metric(candidate_id: str, method: str, scenario: str, prediction: dict[str, Any], original: dict[str, Any], exact: dict[str, Any], local_positions: Sequence[int], compute_count: int, wall: float, hyperparameters: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    p = np.asarray(prediction["probability"], dtype=np.float64); o = np.asarray(original["probability"], dtype=np.float64); e = np.asarray(exact["probability"], dtype=np.float64); y = np.asarray(prediction["gold_label"], dtype=np.int64); selected = np.asarray(local_positions, dtype=np.int64)
    if prediction["gold_label"] != original["gold_label"] or prediction["gold_label"] != exact["gold_label"]:
        raise RuntimeError("Development label/order mismatch")
    eu = float(np.mean(np.abs(p[selected] - e[selected]))); du = float(np.mean(np.abs(o[selected] - e[selected]))); auc = float(roc_auc_score(y, p)); original_auc = float(roc_auc_score(y, o)); clipped = p.clip(1e-12, 1-1e-12); original_clipped = o.clip(1e-12, 1-1e-12); ll = float(np.mean(-(y*np.log(clipped)+(1-y)*np.log(1-clipped)))); original_ll = float(np.mean(-(y*np.log(original_clipped)+(1-y)*np.log(1-original_clipped)))); auc_damage = original_auc-auc; ll_damage = ll-original_ll; utility = auc_damage <= config["selection"]["auc_damage_max"] and ll_damage <= config["selection"]["logloss_damage_max"]
    online_steps=0 if method=="IFRU" else int(hyperparameters["checkpoint"]) if method=="E2URec" else int(compute_count)
    return {"candidate_id": candidate_id, "method": method, "scenario": scenario, "E_U": eu, "D_U": du, "local_residual": eu/du if du else None, "toward_exact_rate": float(np.mean(np.abs(p[selected]-e[selected]) < np.abs(o[selected]-e[selected]))), "overall_auc": auc, "overall_acc": float(accuracy_score(y, p >= .5)), "overall_log_loss": ll, "original_auc": original_auc, "original_log_loss": original_ll, "auc_damage": auc_damage, "logloss_damage": ll_damage, "utility_eligible": utility, "compute_count": int(compute_count), "optimizer_steps": 200 if method=="IFRU" else int(compute_count), "online_optimizer_steps": online_steps, "hvp_calls": int(compute_count) if method=="IFRU" else 0, "wall_time_seconds": float(wall), "hyperparameters": hyperparameters, "prediction_collapse": bool(np.std(p) < 1e-6), "test_accessed": False}


def _select(rows: list[dict[str, Any]], config: dict[str, Any], method: str) -> dict[str, Any]:
    candidates = _candidate_ids(config, method); aggregates = []
    for candidate in candidates:
        cells = [row for row in rows if row["candidate_id"] == candidate]
        if {row["scenario"] for row in cells} != set(SCENARIOS):
            raise RuntimeError(f"incomplete candidate: {candidate}")
        eligible = all(row["utility_eligible"] and not row["prediction_collapse"] for row in cells)
        aggregates.append({"candidate_id": candidate, "eligible": eligible, "equal_scenario_mean_EU": float(np.mean([row["E_U"] for row in cells])), "compute_count": max(row["compute_count"] for row in cells), "failure_reasons": [f"{row['scenario']}:utility_or_collapse" for row in cells if not row["utility_eligible"] or row["prediction_collapse"]]})
    eligible = [row for row in aggregates if row["eligible"]]
    selected = min(eligible, key=lambda row: (row["equal_scenario_mean_EU"], row["compute_count"], row["candidate_id"])) if eligible else None
    return {"method": method, "status": "SELECTED" if selected else config["selection"]["no_eligible_policy"], "selected": selected, "candidates": aggregates, "selection_uses_finaltest": False, "test_accessed": False}


def _pair_indices(retain: Sequence[int], forget: Sequence[int], step: int, seed: int = 42) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed + 1_000_003 * (step // max(1, math.ceil(len(retain) / 16))))
    order = torch.randperm(len(retain), generator=generator).tolist(); start = (step * 16) % len(retain)
    chosen = order[start:start+16] if start + 16 <= len(retain) else order[start:] + order[:16-(len(retain)-start)]
    return [int(retain[index]) for index in chosen], list(map(int, forget))


def _gradient_sweep(repo: Path, workspace: Path, config: dict[str, Any], benchmark: dict[str, Any], registry: dict[str, Any], method: str, destination: Path) -> list[dict[str, Any]]:
    device = torch.device("cuda:0"); _, users, _, select = _development_partition(repo, workspace, config, benchmark); tokenizer = T5Tokenizer.from_pretrained(runner._base_t5_config(workspace, benchmark)["paths"]["model_dir"]); development = JsonPromptDataset(workspace / benchmark["source"]["development_json"], tokenizer); references = _reference_runs(workspace, config); rows = []; grid = config["grids"][method]
    outer_values = grid["forget_weight"] if method == "NegGrad" else grid["learning_rate"]
    for scenario in registry["scenarios"]:
        sid = scenario["id"]; request = set(map(int, scenario["users"])); local_positions = [position for position, index in enumerate(select) if users[index] in request]
        original = _predict_reference(workspace, benchmark, references["Original-Short-200step"], "Original-Short", sid, development, select, device); exact = _predict_reference(workspace, benchmark, references["Exact-Masked-Reference-200step"], "Retrain-Short", sid, development, select, device)
        forget = list(map(int, scenario["forget_train_indices"])); forget_set = set(forget); retain = [int(index) for index in registry["order"] if int(index) not in forget_set]
        for outer in outer_values:
            seed_everything(42); model, names, parameters, bases, train_tokenizer, train, _, _ = paper_v2._load_original_endpoint(workspace, benchmark, registry, "bota_short_original_seed42_v3", sid, device); lr = float(grid["learning_rate"] if method == "NegGrad" else outer); optimizer = torch.optim.AdamW(parameters, lr=lr, betas=(.9,.999), eps=1e-8, weight_decay=.01); begun = time.perf_counter(); checkpoints = set(map(int, grid["checkpoints"])); max_step = max(checkpoints)
            try:
                for step in range(1, max_step+1):
                    ridx, fidx = _pair_indices(retain, forget, step-1); optimizer.zero_grad(set_to_none=True)
                    if method == "NegGrad":
                        weight = float(outer); paper_v2._backward_mean(model, train, ridx, device, train_tokenizer.pad_token_id, 1., 4); paper_v2._backward_mean(model, train, fidx, device, train_tokenizer.pad_token_id, -weight, 4)
                    else:
                        rg, _ = paper_v2._mean_grads(model, train, ridx, parameters, device, train_tokenizer.pad_token_id, 1., 4); fg, _ = paper_v2._mean_grads(model, train, fidx, parameters, device, train_tokenizer.pad_token_id, -1., 4); merged, _ = paper_v2._pcgrad(rg, fg)
                        for parameter, gradient in zip(parameters, merged): parameter.grad = gradient
                    optimizer.step()
                    if step in checkpoints:
                        prediction = short_eval._single_predictions(model.eval(), development, select, device, 4); model.train(); candidate = f"neggrad_w{outer:g}_s{step}" if method == "NegGrad" else f"pcgrad_lr{outer:g}_s{step}"; hyper = {"forget_weight": float(outer), "learning_rate": lr, "checkpoint": step} if method == "NegGrad" else {"learning_rate": lr, "checkpoint": step, "projection": "deterministic_symmetric_two_task"}; row = _metric(candidate, method, sid, prediction, original, exact, local_positions, step, time.perf_counter()-begun, hyper, config); rows.append(row)
                        candidate_dir = destination / "candidate_states" / candidate / sid; artifact=runner._save_fixed_ab(candidate_dir/"adapter",names,runner._copy_parameters(names,parameters),bases,candidate);row["candidate_state_sha256"] = artifact["sha256"]
            finally:
                del model, optimizer, train; gc.collect(); torch.cuda.empty_cache()
    return rows


def _ifru_sweep(repo: Path, workspace: Path, config: dict[str, Any], benchmark: dict[str, Any], registry: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    device=torch.device("cuda:0");_,users,_,select=_development_partition(repo,workspace,config,benchmark);tokenizer=T5Tokenizer.from_pretrained(runner._base_t5_config(workspace,benchmark)["paths"]["model_dir"]);development=JsonPromptDataset(workspace/benchmark["source"]["development_json"],tokenizer);references=_reference_runs(workspace,config);grid=config["grids"]["IFRU"];rows=[];seed_everything(42)
    model=optimizer=resident=None
    try:
        model,names,parameters,bases,_,train_tokenizer,train,train_users=runner._model_context(workspace,benchmark,registry,device);budget=runner.StepBudget(200);_,optimizer,_=runner._train_named(model,train,registry["order"],train_users,[],names,parameters,device,train_tokenizer.pad_token_id,runner._engine(workspace,benchmark),budget,"Stage7_IFRU_base");runner._prepare_influence_endpoint(model,optimizer,parameters);base_flat=runner.flatten_tensors([parameter.detach() for parameter in parameters])
        reference_cache={}
        for scenario in registry["scenarios"]:
            sid=scenario["id"];request=set(map(int,scenario["users"]));local=[position for position,index in enumerate(select) if users[index] in request];original=_predict_reference(workspace,benchmark,references["Original-Short-200step"],"Original-Short",sid,development,select,device);exact=_predict_reference(workspace,benchmark,references["Exact-Masked-Reference-200step"],"Retrain-Short",sid,development,select,device);gradient_parts,_=runner.sample_mean_gradients(model,train,scenario["forget_train_indices"],parameters,device,8,train_tokenizer.pad_token_id);reference_cache[sid]=(local,original,exact,runner.flatten_tensors(gradient_parts));del gradient_parts
        for panel in grid["curvature_panel_samples"]:
            panel_indices=[registry["order"][index] for index in runner.deterministic_panel(3200,int(panel),42)];resident,_=runner.materialize_resident_panel(train,panel_indices,8,train_tokenizer.pad_token_id,device);counter={};operator=runner.make_sample_mean_ggn_operator(model,resident,parameters,counter);estimate=runner.estimate_lambda_max(operator,base_flat.numel(),seed=42,iterations=12,convergence_tolerance=.0001,numerical_lower_bound=1e-14)
            for scenario in registry["scenarios"]:
                sid=scenario["id"];local,original,exact,gradient=reference_cache[sid]
                for ratio in grid["relative_damping_ratio"]:
                    begun=time.perf_counter();cg=runner.conjugate_gradient(operator,gradient,damping=estimate["lambda_max_hat"]*float(ratio),relative_tolerance=.0001,absolute_tolerance=1e-10,max_iterations=int(grid["cg_max_iterations"]),residual_explosion_factor=1000.,pap_tolerance=1e-14,allow_truncated_solution=True);direction=cg.pop("solution");actual=runner.quantized_delta(base_flat,direction,scenario["deleted_interactions"]/scenario["window_interactions"]);offset=0;candidate_values={}
                    for name,parameter in zip(names,parameters): size=parameter.numel();candidate_values[name]=(parameter.detach().reshape(-1).cpu()+actual[offset:offset+size].cpu()).reshape_as(parameter);offset+=size
                    candidate=f"ifru_p{panel}_d{ratio:g}";candidate_dir=destination/"candidate_states"/candidate/sid;artifact=runner._save_fixed_ab(candidate_dir/"adapter",names,candidate_values,bases,candidate);candidate_model=short_eval._load_fixed(workspace,benchmark,candidate_dir,device)
                    try:prediction=short_eval._single_predictions(candidate_model,development,select,device,4)
                    finally:short_eval._release(candidate_model)
                    hyper={"curvature_panel_samples":int(panel),"relative_damping_ratio":float(ratio),"cg_max_iterations":int(grid["cg_max_iterations"]),"lambda_max":float(estimate["lambda_max_hat"]),"cg_iterations":int(cg["iterations"])};row=_metric(candidate,"IFRU",sid,prediction,original,exact,local,12+int(cg["hvp_calls"]),time.perf_counter()-begun,hyper,config);row["candidate_state_sha256"]=artifact["sha256"];rows.append(row);del direction,actual,candidate_values
            del operator,resident;resident=None;gc.collect();torch.cuda.empty_cache()
        return rows
    finally:
        del model,optimizer,resident;gc.collect();torch.cuda.empty_cache()


def _e2urec_sweep(repo: Path, workspace: Path, config: dict[str, Any], benchmark: dict[str, Any], registry: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    device=torch.device("cuda:0");_,users,_,select=_development_partition(repo,workspace,config,benchmark);tokenizer=T5Tokenizer.from_pretrained(runner._base_t5_config(workspace,benchmark)["paths"]["model_dir"]);development=JsonPromptDataset(workspace/benchmark["source"]["development_json"],tokenizer);references=_reference_runs(workspace,config);grid=config["grids"]["E2URec"];rows=[]
    for scenario in registry["scenarios"]:
        sid=scenario["id"];request=set(map(int,scenario["users"]));local=[position for position,index in enumerate(select) if users[index] in request];original_prediction=_predict_reference(workspace,benchmark,references["Original-Short-200step"],"Original-Short",sid,development,select,device);exact=_predict_reference(workspace,benchmark,references["Exact-Masked-Reference-200step"],"Retrain-Short",sid,development,select,device);forget=list(map(int,scenario["forget_train_indices"]));forget_set=set(forget);retain=[int(index) for index in registry["order"] if int(index) not in forget_set]
        augmented=original_model=None
        try:
            augmented,names,augmented_parameters,bases,train_tokenizer,train,_,_=paper_v2._load_original_endpoint(workspace,benchmark,registry,"bota_short_original_seed42_v3",sid,device);original_model,original_names,_,_,_,_,_,_=paper_v2._load_original_endpoint(workspace,benchmark,registry,"bota_short_original_seed42_v3",sid,device)
            if names!=original_names:raise RuntimeError("E2URec coordinate mismatch")
            teacher_optimizer=torch.optim.AdamW(augmented_parameters,lr=float(grid["teacher_learning_rate"]),betas=(.9,.999),eps=1e-8,weight_decay=.01);augmented.train();teacher_begun=time.perf_counter()
            for step in range(int(grid["teacher_steps"])):
                logical=e2urec._wrapped(forget,16,42,step,70_000);teacher_optimizer.zero_grad(set_to_none=True)
                for offset in range(0,16,4):
                    value=e2urec._batch(train,logical[offset:offset+4],device,train_tokenizer.pad_token_id);loss=runner.p1._sample_losses(augmented,value).mean()/4;loss.backward();del value,loss
                teacher_optimizer.step()
            teacher_seconds=time.perf_counter()-teacher_begun;e2urec._freeze(augmented);e2urec._freeze(original_model);del teacher_optimizer
            for weight in grid["forget_weight"]:
                current=None
                try:
                    current,current_names,current_parameters,current_bases,current_tokenizer,current_train,_,_=paper_v2._load_original_endpoint(workspace,benchmark,registry,"bota_short_original_seed42_v3",sid,device)
                    if current_names!=names:raise RuntimeError("E2URec student coordinate mismatch")
                    optimizer=torch.optim.AdamW(current_parameters,lr=float(grid["student_learning_rate"]),betas=(.9,.999),eps=1e-8,weight_decay=.01);checkpoints=set(map(int,grid["checkpoints"]));begun=time.perf_counter();warmup=math.ceil(len(forget)/16);current.train()
                    for step in range(1,max(checkpoints)+1):
                        fidx=e2urec._wrapped(forget,16,42,step-1,90_000);ridx=e2urec._wrapped(retain,16,42,step-1,110_000);optimizer.zero_grad(set_to_none=True)
                        for offset in range(0,16,4):
                            fb=e2urec._batch(current_train,fidx[offset:offset+4],device,current_tokenizer.pad_token_id)
                            if step<=warmup:loss=e2urec.forget_loss_light(current,original_model,augmented,fb,2.0)
                            else:
                                rb=e2urec._batch(current_train,ridx[offset:offset+4],device,current_tokenizer.pad_token_id);parts=e2urec.joint_losses_light(current,original_model,augmented,fb,rb,2.0);loss=(1-float(weight))*(parts["L_sup"]+parts["L_retain_KL"])+float(weight)*parts["L_forget"];del rb,parts
                            (loss/4).backward();del fb,loss
                        optimizer.step()
                        if step in checkpoints:
                            prediction=short_eval._single_predictions(current.eval(),development,select,device,4);current.train();candidate=f"e2urec_w{weight:g}_s{step}";candidate_dir=destination/"candidate_states"/candidate/sid;artifact=runner._save_fixed_ab(candidate_dir,current_names,runner._copy_parameters(current_names,current_parameters),current_bases,candidate);hyper={"teacher_steps":int(grid["teacher_steps"]),"teacher_learning_rate":float(grid["teacher_learning_rate"]),"student_learning_rate":float(grid["student_learning_rate"]),"forget_weight":float(weight),"checkpoint":step,"teacher_shared_seconds":teacher_seconds};row=_metric(candidate,"E2URec",sid,prediction,original_prediction,exact,local,int(grid["teacher_steps"])+step,time.perf_counter()-begun,hyper,config);row["candidate_state_sha256"]=artifact["sha256"];rows.append(row)
                finally:
                    if current is not None:del current
                    if 'optimizer' in locals():del optimizer
                    if 'current_train' in locals():del current_train
                    gc.collect();torch.cuda.empty_cache()
        finally:
            if augmented is not None:del augmented
            if original_model is not None:del original_model
            if 'train' in locals():del train
            gc.collect();torch.cuda.empty_cache()
    return rows


def _write_identical(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text: raise RuntimeError(f"frozen Stage 7 protocol collision: {path}")
        return
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp");temporary.write_text(text,encoding="utf-8",newline="\n");os.replace(temporary,path)


def _partition_sweep(repo: Path, workspace: Path, config: dict[str, Any], benchmark: dict[str, Any], registry: dict[str, Any], method: str, destination: Path) -> tuple[list[dict[str, Any]],dict[str,list[Path]]]:
    device=torch.device("cuda:0");development_rows,users,fit,select=_development_partition(repo,workspace,config,benchmark);tokenizer=T5Tokenizer.from_pretrained(runner._base_t5_config(workspace,benchmark)["paths"]["model_dir"]);development=JsonPromptDataset(workspace/benchmark["source"]["development_json"],tokenizer);references=_reference_runs(workspace,config);grid=config["grids"][method];kind="sisa" if method=="SISA" else "receraser";protocol_root=workspace/config["output_root"]/"protocol";fit_path=protocol_root/"development_fit.json";fit_indices_path=protocol_root/"development_fit_indices.json";fit_text=json.dumps([development_rows[index] for index in fit],ensure_ascii=False,indent=2)+"\n";fit_indices_text=json.dumps(fit,indent=2)+"\n";_write_identical(fit_path,fit_text);_write_identical(fit_indices_path,fit_indices_text);rows=[];components:dict[str,list[Path]]={candidate:[] for candidate in _candidate_ids(config,method)};short_id="SISA-Short-T5" if method=="SISA" else "RecEraser-Adapter-Short"
    protocol_dir=workspace/benchmark["output_root"]/"protocols"/config["benchmark"]["name"]
    for scenario in registry["scenarios"]:
        sid=scenario["id"];request=set(map(int,scenario["users"]));local=[position for position,index in enumerate(select) if users[index] in request];original=_predict_reference(workspace,benchmark,references["Original-Short-200step"],"Original-Short",sid,development,select,device);exact=_predict_reference(workspace,benchmark,references["Exact-Masked-Reference-200step"],"Retrain-Short",sid,development,select,device);base=yaml.safe_load((protocol_dir/"data"/sid/f"{kind}.yaml").read_text(encoding="utf-8"));base_steps=int(base["budget"]["total_expected_optimizer_steps"])
        for lr in grid["learning_rate"]:
            for epoch in grid["epochs_per_slice"]:
                candidate=(f"sisa_lr{lr:g}_e{epoch}" if method=="SISA" else f"receraser_lr{lr:g}_e{epoch}");candidate_config=json.loads(json.dumps(base));full_development=dict(candidate_config["sources"]["development"]);candidate_config["sources"]["development"]={"path":str(fit_path.relative_to(workspace)).replace("\\","/"),"sha256":sha256_file(fit_path)};candidate_config["development_subset_authority"]={"full_path":full_development["path"],"full_sha256":full_development["sha256"],"indices_path":str(fit_indices_path.relative_to(workspace)).replace("\\","/"),"indices_sha256":sha256_file(fit_indices_path)};candidate_config["protocol"]["learning_rate"]=float(lr);candidate_config["protocol"]["epochs_per_slice"]=int(epoch);candidate_config["budget"]["total_expected_optimizer_steps"]=base_steps*int(epoch);candidate_config["budget"]["official_epoch_or_step_budget"]=f"Stage 7 Development sweep candidate: {epoch} epoch(s) per slice/local model";candidate_config["output_root"]=f"{config['output_root']}/components/{method}/{candidate}/{sid}";candidate_config["synthetic_root"]=f"{config['output_root']}/synthetic/{method}/{candidate}/{sid}";config_path=protocol_root/method/candidate/f"{sid}.yaml";_write_identical(config_path,yaml.safe_dump(candidate_config,sort_keys=False,allow_unicode=True));component=workspace/candidate_config["output_root"]/"model";begun=time.perf_counter()
                if component.exists():partitioned.analyze_run(component,partitioned.MARKERS[kind])
                else:partitioned.execute(workspace,config_path,"model",kind,False,partition_seed_override=42,formal=False)
                components[candidate].append(component);scenario_manifest={"component_run":str(component),"method_id":short_id,"test_accessed":False,"model_type":"full_model_ensemble" if method=="SISA" else "lora_adapter_ensemble"};prediction=short_eval._ensemble_predictions(workspace,benchmark,scenario_manifest,development,select,device,4);manifest_path=component/"paper_model_manifest.json";manifest=_json(manifest_path);hyper={"learning_rate":float(lr),"epochs_per_slice":int(epoch),"shards":int(grid["shards"]),"slices":int(grid.get("slices",1)),"development_aggregation_fit":"Dev-Fit" if method=="RecEraser" else "fixed_mean"};row=_metric(candidate,method,sid,prediction,original,exact,local,int(manifest["optimizer_steps"]),time.perf_counter()-begun,hyper,config);row["candidate_state_sha256"]=sha256_file(manifest_path);rows.append(row);state_dir=destination/"candidate_states"/candidate/sid;state_dir.mkdir(parents=True,exist_ok=True);atomic_json(state_dir/"component.json",{"component_run":str(component),"paper_model_manifest_sha256":row["candidate_state_sha256"],"model_tensors_retained_until_selection":True,"test_accessed":False})
    return rows,components


def preflight(repo: Path, workspace: Path, config_path: Path, method: str) -> dict[str, Any]:
    config = load_config(repo, workspace, config_path); benchmark, registry = _authority(repo, workspace, config); _, users, fit, select = _development_partition(repo, workspace, config, benchmark)
    model_directory=short_eval._local_model_directory(workspace,runner._base_t5_config(workspace,benchmark)["paths"]["model_dir"])
    support = {scenario["id"]: {"fit": sum(users[index] in set(scenario["users"]) for index in fit), "select": sum(users[index] in set(scenario["users"]) for index in select)} for scenario in registry["scenarios"]}
    budget_cells=0
    if method in {"SISA","RecEraser"}:
        kind="sisa" if method=="SISA" else "receraser";protocol_dir=workspace/benchmark["output_root"]/"protocols"/config["benchmark"]["name"]
        for sid in SCENARIOS:
            base=yaml.safe_load((protocol_dir/"data"/sid/f"{kind}.yaml").read_text(encoding="utf-8"));base_steps=int(base["budget"]["total_expected_optimizer_steps"])
            for lr in config["grids"][method]["learning_rate"]:
                for epoch in config["grids"][method]["epochs_per_slice"]:
                    candidate=json.loads(json.dumps(base));candidate["protocol"]["learning_rate"]=float(lr);candidate["protocol"]["epochs_per_slice"]=int(epoch);candidate["budget"]["total_expected_optimizer_steps"]=base_steps*int(epoch);partitioned.validate_partition_config(candidate,kind);partitioned.partitioned_budget(candidate,kind);budget_cells+=1
    return {"schema": SCHEMA, "mode": "Preflight", "method": method, "candidate_count": len(_candidate_ids(config, method)), "scenarios": list(SCENARIOS), "development_fit": len(fit), "development_select": len(select), "request_support": support, "partitioned_budget_cells_verified":budget_cells, "local_model_directory":str(model_directory), "local_model_directory_verified":True, "model_loaded": False, "optimizer_constructed": False, "finaltest_accessed": False, "test_accessed": False}


def synthetic(repo: Path, workspace: Path, config_path: Path, method: str) -> dict[str, Any]:
    config = load_config(repo, workspace, config_path); rows = []
    for candidate in _candidate_ids(config, method):
        for number, scenario in enumerate(SCENARIOS):
            rows.append({"candidate_id": candidate, "scenario": scenario, "E_U": .1 + number*.01, "utility_eligible": True, "prediction_collapse": False, "compute_count": 10})
    selection = _select(rows, config, method)
    return {"schema": SCHEMA, "method": method, "candidate_count": len(_candidate_ids(config, method)), "selected": selection["selected"]["candidate_id"], "model_loaded": False, "optimizer_constructed": False, "test_accessed": False}


def execute(repo: Path, workspace: Path, config_path: Path, method: str, run_name: str) -> dict[str, Any]:
    config = load_config(repo, workspace, config_path); benchmark, registry = _authority(repo, workspace, config); git = git_snapshot(repo)
    if config["runtime"]["repository_must_be_clean"] and not git["clean"]:
        raise RuntimeError("commit Amendment 06 and Stage 7 code before a formal Full run")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU required")
    free, _ = torch.cuda.mem_get_info(0)
    if free / (1024**3) < config["runtime"]["minimum_free_gib"]:
        raise RuntimeError("insufficient dedicated GPU memory")
    destination = workspace / config["output_root"] / "runs" / safe_run_name(run_name)
    if destination.exists():
        raise FileExistsError(destination)
    stage = destination.parent / ".work" / f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True); begun = time.perf_counter(); previous_directory=Path.cwd();os.chdir(workspace)
    try:
        components=None
        if method in {"NegGrad", "PCGrad"}: rows=_gradient_sweep(repo,workspace,config,benchmark,registry,method,stage)
        elif method=="IFRU": rows=_ifru_sweep(repo,workspace,config,benchmark,registry,stage)
        elif method=="E2URec": rows=_e2urec_sweep(repo,workspace,config,benchmark,registry,stage)
        else: rows,components=_partition_sweep(repo,workspace,config,benchmark,registry,method,stage)
        selection = _select(rows, config, method)
        if components is not None:
            selected_id=selection["selected"]["candidate_id"] if selection["selected"] else None;allowed_root=(workspace/config["output_root"]/"components"/method).resolve()
            for candidate,paths in components.items():
                retained=candidate==selected_id
                if not retained:
                    for component in paths:
                        resolved=component.resolve()
                        if allowed_root not in resolved.parents:raise RuntimeError("refusing to remove component outside Stage 7 output root")
                        if resolved.exists():shutil.rmtree(resolved)
                for sid in SCENARIOS:
                    state_path=stage/"candidate_states"/candidate/sid/"component.json"
                    if state_path.exists():
                        value=_json(state_path);value["model_tensors_retained_after_selection"]=retained;value.pop("model_tensors_retained_until_selection",None);atomic_json(state_path,value)
        atomic_json(stage / "candidate_metrics.json", {"schema": SCHEMA, "method": method, "rows": rows, "test_accessed": False})
        flat = [{**{key:value for key,value in row.items() if key != "hyperparameters"}, "hyperparameters": json.dumps(row["hyperparameters"], sort_keys=True)} for row in rows]
        with (stage / "candidate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        atomic_json(stage / "selection.json", selection); report = [f"# Stage 7A baseline sweep: {method}", "", f"Status: `{selection['status']}`.", "", "All candidates were evaluated on frozen Dev-Select; FinalTest was not opened.", ""]
        if selection["selected"]: report.append(f"Selected candidate: `{selection['selected']['candidate_id']}`.")
        (stage / "report.md").write_text("\n".join(report)+"\n", encoding="utf-8", newline="\n"); state = {"schema": SCHEMA, "status": "COMPLETED", "method": method, "run_name": run_name, "candidate_cells": len(rows), "wall_time_seconds": time.perf_counter()-begun, "finaltest_accessed": False, "test_accessed": False}; atomic_json(stage / "run_state.json", state); atomic_json(stage / "contract.json", {"schema": SCHEMA, "config_sha256": sha256_file(config_path), "amendment_sha256": sha256_file(repo / config["parent_amendment"]["path"]), "git": git, "split": "Development/Dev-Select", "finaltest_accessed": False, "test_accessed": False}); (stage / "COMPLETED").write_text(MARKER+"\n", encoding="utf-8", newline="\n"); state_hashes={str(path.parent.relative_to(stage/"candidate_states")).replace("\\","/"):directory_hash(path.parent) for path in sorted((stage/"candidate_states").rglob("adapter_model.pt"))};state_hashes.update({str(path.parent.relative_to(stage/"candidate_states")).replace("\\","/"):directory_hash(path.parent) for path in sorted((stage/"candidate_states").rglob("component.json"))});atomic_json(stage / "manifest.json", {"schema": SCHEMA, "files": {name: sha256_file(stage/name) for name in RUN_REQUIRED-{"manifest.json"}}, "candidate_states":state_hashes, "published_atomically": True, "test_accessed": False}); destination.parent.mkdir(parents=True, exist_ok=True); os.replace(stage, destination); return {"status": "COMPLETED", "method": method, "run_dir": str(destination), "selection": selection["status"], "selected_candidate": selection["selected"]["candidate_id"] if selection["selected"] else None, "test_accessed": False}
    finally:
        os.chdir(previous_directory)
        if stage.exists(): shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def analyze(workspace: Path, config: dict[str, Any], run_name: str) -> dict[str, Any]:
    run = workspace / config["output_root"] / "runs" / safe_run_name(run_name)
    if not run.is_dir() or {path.name for path in run.iterdir()} != RUN_REQUIRED | {"candidate_states"} or (run/"COMPLETED").read_text(encoding="utf-8") != MARKER+"\n":
        raise ValueError("invalid Stage 7 run")
    manifest = _json(run/"manifest.json")
    for name, expected in manifest["files"].items():
        if sha256_file(run/name) != expected: raise ValueError(f"Stage 7 artifact changed: {name}")
    for relative,expected in manifest["candidate_states"].items():
        if directory_hash(run/"candidate_states"/relative)!=expected:raise ValueError(f"Stage 7 candidate state changed: {relative}")
    for component_file in (run/"candidate_states").rglob("component.json"):
        value=_json(component_file)
        if value.get("model_tensors_retained_after_selection"):
            component=Path(value["component_run"]);manifest_path=component/"paper_model_manifest.json"
            if not manifest_path.is_file() or sha256_file(manifest_path)!=value["paper_model_manifest_sha256"]:raise ValueError("selected partitioned candidate changed")
    return {"status": "COMPLETED", "run_dir": str(run), "selection": _json(run/"selection.json")["status"], "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, default=Path.cwd()); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--config", type=Path, default=Path("configs/ali_stage7_baseline_sweep_v1.yaml")); parser.add_argument("--mode", choices=["Preflight", "SyntheticDryRun", "Full", "Analyze"], default="Preflight"); parser.add_argument("--method", choices=METHODS, required=True); parser.add_argument("--run-name", default=""); args = parser.parse_args(); repo=args.repo.resolve(); workspace=args.workspace.resolve(); config_path=(repo/args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    if args.mode == "Preflight": result=preflight(repo,workspace,config_path,args.method)
    elif args.mode == "SyntheticDryRun": result=synthetic(repo,workspace,config_path,args.method)
    elif not args.run_name: parser.error(f"{args.mode} requires --run-name")
    elif args.mode == "Analyze": result=analyze(workspace,load_config(repo,workspace,config_path),args.run_name)
    else: result=execute(repo,workspace,config_path,args.method,args.run_name)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == "__main__": main()
