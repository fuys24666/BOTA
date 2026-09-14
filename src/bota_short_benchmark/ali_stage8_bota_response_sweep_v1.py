"""Stage 8: preregistered Development-only BOTA response-strength sweep."""
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
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import T5Tokenizer

from src.bota_if import p1_trajectory_transport_audit as p1
from src.bota_if import p2b_full_module_adamw_transport_audit as p2b
from src.bota_if.p1_trajectory_transport_audit import StepBudget
from src.diagnostics.ml1m_development_protocol import reconstruct_authoritative_rows
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, move_batch
from src.if_a2_optimization.group_a_gradient_audit import masked_batch
from src.paper_baselines.common import capture_rng, tensor_tree_hash
from src.paper_if_a2.common import atomic_json, canonical_hash, directory_hash, git_snapshot, safe_run_name, sha256_file

from . import ali_stage1_mechanism_v1 as stage1
from . import ali_stage7_baseline_sweep_v1 as stage7
from . import runner
from . import t012_behavior_ablation as behavior


SCHEMA = "bota-ali-stage8-bota-response-sweep-v1"
MARKER = "BOTA_ALI_STAGE8_BOTA_RESPONSE_SWEEP_V1_COMPLETED"
SCENARIOS = ("L8", "L4M4")
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
RUN_REQUIRED = {"COMPLETED", "adapters", "candidate_metrics.csv", "candidate_metrics.json", "contract.json", "manifest.json", "report.md", "run_state.json", "selection.json"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _candidate_id(scale: float) -> str:
    return f"bota_r{scale:g}"


def load_config(repo: Path, path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema", "test_access_policy", "parent_amendment", "baseline_audit", "stage1_config", "benchmark", "development_partition", "protocol", "selection", "formal_reproduction", "evaluation", "output_root", "runtime"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA or value["test_access_policy"] != "forbidden":
        raise ValueError("invalid Stage 8 configuration surface")
    for field in ("parent_amendment", "baseline_audit", "stage1_config"):
        binding = value[field]
        if sha256_file(repo / binding["path"]) != binding["sha256"]:
            raise ValueError(f"Stage 8 {field} binding changed")
    amendment = _json(repo / value["parent_amendment"]["path"])
    if amendment.get("amendment_id") != value["parent_amendment"]["amendment_id"] or amendment.get("status") != "FROZEN_BEFORE_STAGE8_EXECUTION":
        raise ValueError("Amendment 07 identity/status changed")
    audit = _json(repo / value["baseline_audit"]["path"])
    if audit.get("status") != value["baseline_audit"]["status"] or audit.get("finaltest_accessed") is not False or set(audit.get("selections", {})) != set(stage7.METHODS):
        raise ValueError("frozen Stage 7 baseline audit changed")
    for field in ("config", "paper_config"):
        binding = value["benchmark"][field]
        if sha256_file(repo / binding["path"]) != binding["sha256"]:
            raise ValueError(f"Stage 8 benchmark {field} changed")
    if value["protocol"] != {"scenarios": list(SCENARIOS), "optimizer_steps": 200, "response_scales": list(SCALES), "full_state_adamw": True, "empirical_fisher_response": True, "source_scaled": False, "recursive_transport": True, "endpoint_interpolation": False, "online_optimizer_steps": 0}:
        raise ValueError("Stage 8 protocol changed")
    if value["selection"] != {"scenarios": list(SCENARIOS), "auc_damage_max": .005, "logloss_damage_max": .01, "primary": "equal_scenario_mean_EU", "tie_break": ["absolute_deviation_from_scale_1", "candidate_id"], "no_eligible_policy": "NO_ELIGIBLE_CALIBRATION"}:
        raise ValueError("Stage 8 selection rule changed")
    if value["evaluation"] != {"split": "Development/Dev-Select", "final_test": False, "inference_batch_size": 4}:
        raise ValueError("Stage 8 evaluation boundary changed")
    return value


def _select(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    summaries = []
    for scale in SCALES:
        candidate = _candidate_id(scale)
        cells = [row for row in rows if row["candidate_id"] == candidate]
        if {row["scenario"] for row in cells} != set(SCENARIOS):
            raise RuntimeError(f"incomplete BOTA candidate: {candidate}")
        eligible = all(row["utility_eligible"] and not row["prediction_collapse"] for row in cells)
        summaries.append({"candidate_id": candidate, "response_scale": scale, "eligible": eligible, "equal_scenario_mean_EU": float(np.mean([row["E_U"] for row in cells])), "failure_reasons": [f"{row['scenario']}:utility_or_collapse" for row in cells if not row["utility_eligible"] or row["prediction_collapse"]]})
    eligible = [row for row in summaries if row["eligible"]]
    selected = min(eligible, key=lambda row: (row["equal_scenario_mean_EU"], abs(row["response_scale"] - 1.0), row["candidate_id"])) if eligible else None
    return {"method": "BOTA", "status": "SELECTED" if selected else config["selection"]["no_eligible_policy"], "selected": selected, "effective_response_scale": selected["response_scale"] if selected else 1.0, "candidates": summaries, "selection_uses_finaltest": False, "test_accessed": False}


def _advance_scaled(states: list[dict[str, torch.Tensor]], coefficients: Sequence[torch.Tensor], sources: Sequence[torch.Tensor], parameters: Sequence[torch.Tensor], optimizer: torch.optim.Optimizer, step: int, engine: dict[str, Any], response_scale: float) -> None:
    opt = engine["optimizer"]; beta1, beta2 = map(float, opt["betas"])
    for current, parameter, rows, source in zip(states, parameters, coefficients, sources):
        dgradient = source if response_scale == 0.0 else source + response_scale * p2b.full_fisher_product(rows, current["theta"])
        optimizer_state = optimizer.state.get(parameter, {})
        theta, dm, dv = p2b.stable_adamw_tangent_step(
            parameter.detach(), parameter.grad.detach(), optimizer_state.get("exp_avg", torch.zeros_like(parameter)).detach(), optimizer_state.get("exp_avg_sq", torch.zeros_like(parameter)).detach(), current["theta"], current["m"], current["v"], dgradient,
            step=step, lr=float(opt["learning_rate"]), beta1=beta1, beta2=beta2, eps=float(opt["eps"]), weight_decay=float(opt["weight_decay"]), full_v=True,
        )
        current.update(theta=theta, m=dm, v=dv)


def _run_scaled(model: Any, dataset: Any, indices: Sequence[int], user_ids: Sequence[int], selected_users: Sequence[int], parameters: Sequence[torch.Tensor], names: Sequence[str], device: torch.device, pad: int, engine: dict[str, Any], budget: StepBudget) -> tuple[dict[str, torch.Tensor], dict[str, list[dict[str, torch.Tensor]]], str, str]:
    optimizer = p1._optimizer(parameters, engine)
    states = {_candidate_id(scale): stage1._state(parameters, len(selected_users)) for scale in SCALES}
    batch_size = int(engine["schedule"]["batch_size"])
    for step, start in enumerate(range(0, len(indices), batch_size), 1):
        chosen = list(indices[start:start + batch_size]); batch_users = [int(user_ids[index]) for index in chosen]
        batch = move_batch(masked_batch(dataset, chosen, pad), device); losses = p1._sample_losses(model, batch)
        sample_rows: list[list[torch.Tensor]] = [[] for _ in parameters]
        for sample in range(len(chosen)):
            gradients = torch.autograd.grad(losses[sample], parameters, retain_graph=True)
            for module, gradient in enumerate(gradients): sample_rows[module].append(gradient.detach())
        optimizer.zero_grad(set_to_none=True); formal_loss = losses.sum() / len(chosen); formal_loss.backward(); coefficients = [torch.stack(rows) for rows in sample_rows]
        sources = []
        for rows, parameter in zip(coefficients, parameters):
            user_sources = []
            for target in selected_users:
                slots = [slot for slot, user in enumerate(batch_users) if user == target]
                user_sources.append(-rows[slots].sum(0) / len(chosen) if slots else torch.zeros_like(parameter))
            sources.append(torch.stack(user_sources))
        for scale in SCALES: _advance_scaled(states[_candidate_id(scale)], coefficients, sources, parameters, optimizer, step, engine, scale)
        budget.step(optimizer, "shared_canonical_reference")
        del batch, losses, formal_loss, sample_rows, coefficients, sources
    canonical = {name: parameter.detach().float().cpu().clone() for name, parameter in zip(names, parameters)}
    return canonical, states, tensor_tree_hash(optimizer.state_dict()), tensor_tree_hash(capture_rng())


def _metric(candidate_id: str, scale: float, scenario: str, prediction: dict[str, Any], original: dict[str, Any], exact: dict[str, Any], local_positions: Sequence[int], shared_seconds: float, publication_seconds: float, config: dict[str, Any]) -> dict[str, Any]:
    p = np.asarray(prediction["probability"], dtype=np.float64); o = np.asarray(original["probability"], dtype=np.float64); e = np.asarray(exact["probability"], dtype=np.float64); y = np.asarray(prediction["gold_label"], dtype=np.int64); selected = np.asarray(local_positions, dtype=np.int64)
    if prediction["gold_label"] != original["gold_label"] or prediction["gold_label"] != exact["gold_label"]: raise RuntimeError("Stage 8 Development order mismatch")
    eu = float(np.mean(np.abs(p[selected] - e[selected]))); du = float(np.mean(np.abs(o[selected] - e[selected]))); auc = float(roc_auc_score(y, p)); original_auc = float(roc_auc_score(y, o)); clipped = p.clip(1e-12, 1-1e-12); original_clipped = o.clip(1e-12, 1-1e-12); loss = float(np.mean(-(y*np.log(clipped)+(1-y)*np.log(1-clipped)))); original_loss = float(np.mean(-(y*np.log(original_clipped)+(1-y)*np.log(1-original_clipped)))); auc_damage = original_auc-auc; loss_damage = loss-original_loss
    return {"candidate_id": candidate_id, "method": "BOTA", "scenario": scenario, "response_scale": scale, "E_U": eu, "D_U": du, "local_residual": eu/du if du else None, "toward_exact_rate": float(np.mean(np.abs(p[selected]-e[selected]) < np.abs(o[selected]-e[selected]))), "overall_auc": auc, "overall_acc": float(accuracy_score(y,p>=.5)), "overall_log_loss": loss, "original_auc": original_auc, "original_log_loss": original_loss, "auc_damage": auc_damage, "logloss_damage": loss_damage, "utility_eligible": auc_damage <= config["selection"]["auc_damage_max"] and loss_damage <= config["selection"]["logloss_damage_max"], "prediction_collapse": bool(np.std(p)<1e-6), "offline_optimizer_steps": 200, "online_optimizer_steps": 0, "hvp_calls": 0, "shared_transport_wall_time_seconds": shared_seconds, "adapter_publication_wall_time_seconds": publication_seconds, "test_accessed": False}


def preflight(repo: Path, workspace: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(repo, config_path); benchmark = stage7.load_benchmark_config(repo / config["benchmark"]["config"]["path"]); benchmark["_config_path"] = str((repo / config["benchmark"]["config"]["path"]).resolve()); _, _, registry = stage7.validate_prepared(workspace, benchmark, config["benchmark"]["name"])
    frozen = workspace / config["benchmark"]["frozen_development_evaluation"]["path"]; binding = config["benchmark"]["frozen_development_evaluation"]
    if sha256_file(frozen/"manifest.json") != binding["manifest_sha256"] or sha256_file(frozen/"per_sample_metrics.jsonl") != binding["per_sample_sha256"]: raise ValueError("Stage 8 frozen Development authority changed")
    _, users, fit, select = stage7._development_partition(repo, workspace, config, benchmark)
    stage1_path = repo / config["stage1_config"]["path"]
    authorities = {scenario: stage1.preflight(repo, workspace, stage1_path, 42, scenario)[0] for scenario in SCENARIOS}
    support = {scenario["id"]: {"dev_fit": sum(users[index] in set(scenario["users"]) for index in fit), "dev_select": sum(users[index] in set(scenario["users"]) for index in select)} for scenario in registry["scenarios"]}
    return {"schema": SCHEMA, "mode": "Preflight", "candidate_count": len(SCALES), "candidate_cells": len(SCALES)*len(SCENARIOS), "scenarios": list(SCENARIOS), "request_support": support, "reference_adapter_sha256": {scenario: authorities[scenario]["references"] for scenario in SCENARIOS}, "model_loaded": False, "optimizer_constructed": False, "finaltest_accessed": False, "test_accessed": False}


def synthetic(config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for scale in SCALES:
        for number, scenario in enumerate(SCENARIOS): rows.append({"candidate_id": _candidate_id(scale), "scenario": scenario, "E_U": abs(scale-.5)+number*.01, "utility_eligible": True, "prediction_collapse": False})
    selection = _select(rows, config)
    return {"schema": SCHEMA, "mode": "SyntheticDryRun", "selected": selection["selected"]["candidate_id"], "candidate_count": len(SCALES), "model_loaded": False, "optimizer_constructed": False, "test_accessed": False}


def execute(repo: Path, workspace: Path, config_path: Path, run_name: str) -> dict[str, Any]:
    config = load_config(repo, config_path); preflight(repo, workspace, config_path); git = git_snapshot(repo)
    if config["runtime"]["repository_must_be_clean"] and not git["clean"]: raise RuntimeError("commit Amendment 07 and Stage 8 code before Full")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("Stage 8 requires exactly one CUDA GPU")
    device=torch.device("cuda:0");free,total=torch.cuda.mem_get_info(device)
    if free/2**30<float(config["runtime"]["minimum_free_gib"]):raise RuntimeError("insufficient free GPU memory for Stage 8")
    torch.cuda.set_per_process_memory_fraction(runner.allocator_fraction_for(total/2**30,float(config["runtime"]["allocator_fraction"]),float(config["runtime"]["hard_peak_reserved_gib"])),device);torch.cuda.reset_peak_memory_stats()
    destination=workspace/config["output_root"]/"runs"/safe_run_name(run_name)
    if destination.exists():raise FileExistsError(destination)
    stage=destination.parent/".work"/f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage";stage.mkdir(parents=True);started=time.perf_counter();previous=Path.cwd();os.chdir(workspace);rows=[];reproduction={};source_hashes={}
    try:
        benchmark=stage7.load_benchmark_config(repo/config["benchmark"]["config"]["path"]);benchmark["_config_path"]=str((repo/config["benchmark"]["config"]["path"]).resolve());_,_,registry=stage7.validate_prepared(workspace,benchmark,config["benchmark"]["name"]);_,development_users,_,select=stage7._development_partition(repo,workspace,config,benchmark);stage1_path=repo/config["stage1_config"]["path"]
        for scenario_id in SCENARIOS:
            authority,_,_=stage1.preflight(repo,workspace,stage1_path,42,scenario_id);scenario=behavior._scenario(registry,scenario_id);original_run=Path(authority["references"]["original"]["run"]);exact_run=Path(authority["references"]["exact_masked"]["run"]);formal_run=Path(authority["references"]["formal_bota"]["run"])
            original_state=torch.load(original_run/"scenarios"/scenario_id/"adapter"/"adapter_model.pt",map_location="cpu",weights_only=True);formal_state=torch.load(formal_run/"scenarios"/scenario_id/"adapter"/"adapter_model.pt",map_location="cpu",weights_only=True);source_checkpoint=workspace/benchmark["source"]["original_checkpoint"];source_before=sha256_file(source_checkpoint);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            model,names,parameters,bases,_,tokenizer,train_data,train_users=runner._model_context(workspace,benchmark,registry,device,original_state["A"]);engine=runner._engine(workspace,benchmark);budget=StepBudget(200);trajectory_started=time.perf_counter();canonical,states,optimizer_hash,rng_hash=_run_scaled(model,train_data,registry["order"],train_users,scenario["users"],parameters,names,device,tokenizer.pad_token_id,engine,budget);trajectory_seconds=time.perf_counter()-trajectory_started
            if budget.calls!=200 or budget.by_arm!={"shared_canonical_reference":200}:raise RuntimeError("Stage 8 optimizer-step budget mismatch")
            if any(not torch.equal(canonical[name],original_state["B"][name]) for name in names):raise RuntimeError("Stage 8 canonical endpoint differs from Original")
            scale_one=None;publication={};adapter_hashes={}
            for scale in SCALES:
                candidate_id=_candidate_id(scale);candidate={name:canonical[name]+states[candidate_id][module]["theta"].sum(0).float().cpu() for module,name in enumerate(names)};publish=time.perf_counter();adapter_dir=stage/"adapters"/candidate_id/scenario_id/"adapter";adapter_hashes[candidate_id]=runner._save_fixed_ab(adapter_dir,names,candidate,bases,f"BOTA-FS-FisherScale-{scale:g}")["sha256"];publication[candidate_id]=time.perf_counter()-publish
                if scale==1.0:scale_one=candidate
            assert scale_one is not None
            formal_delta=torch.cat([(formal_state["B"][name].float()-canonical[name]).reshape(-1) for name in names]);reproduced_delta=torch.cat([(scale_one[name]-canonical[name]).reshape(-1) for name in names]);alignment=p2b.vector_metrics(formal_delta,reproduced_delta);gate=config["formal_reproduction"];passed=bool(alignment["finite"] and alignment["cosine"]>=gate["cosine_minimum"] and alignment["relative_l2_error"]<=gate["relative_l2_maximum"] and alignment["maximum_absolute_error"]<=gate["maximum_absolute_error"])
            if not passed:raise RuntimeError(f"Stage 8 response-scale-1 formal reproduction failed: {alignment}")
            reproduction[scenario_id]={"metrics":alignment,"passed":passed,"formal_adapter_sha256":authority["references"]["formal_bota"]["adapter_sha256"],"reproduced_adapter_sha256":adapter_hashes[_candidate_id(1.0)]}
            source_hashes[scenario_id]={"source_checkpoint_sha256":source_before,"optimizer_state_sha256":optimizer_hash,"rng_state_sha256":rng_hash}
            del model,parameters,states,scale_one;gc.collect();torch.cuda.empty_cache()
            eval_tokenizer=T5Tokenizer.from_pretrained(runner._base_t5_config(workspace,benchmark)["paths"]["model_dir"]);development=JsonPromptDataset(workspace/benchmark["source"]["development_json"],eval_tokenizer);request=set(map(int,scenario["users"]));local=[position for position,index in enumerate(select) if development_users[index] in request];original=behavior._predict_reference(workspace,benchmark,original_run,scenario_id,development,select,device);exact=behavior._predict_reference(workspace,benchmark,exact_run,scenario_id,development,select,device)
            for scale in SCALES:
                candidate_id=_candidate_id(scale);prediction=behavior._predict_scenario_dir(workspace,benchmark,stage/"adapters"/candidate_id/scenario_id,development,select,device);row=_metric(candidate_id,scale,scenario_id,prediction,original,exact,local,trajectory_seconds,publication[candidate_id],config);row["candidate_state_sha256"]=adapter_hashes[candidate_id];rows.append(row)
            if sha256_file(source_checkpoint)!=source_before:raise RuntimeError("Stage 8 source checkpoint changed")
        selection=_select(rows,config);atomic_json(stage/"candidate_metrics.json",{"schema":SCHEMA,"rows":rows,"formal_reproduction":reproduction,"test_accessed":False});flat=[{**row} for row in rows]
        with (stage/"candidate_metrics.csv").open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        atomic_json(stage/"selection.json",selection);report=["# Stage 8 BOTA response-strength sweep","","Development/Dev-Select only; FinalTest was not opened.","",f"Status: `{selection['status']}`.","",f"Effective response scale: `{selection['effective_response_scale']}`.",""];(stage/"report.md").write_text("\n".join(report),encoding="utf-8",newline="\n");state={"schema":SCHEMA,"status":"COMPLETED","run_name":run_name,"candidate_cells":len(rows),"wall_time_seconds":time.perf_counter()-started,"maximum_peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,"finaltest_accessed":False,"test_accessed":False};atomic_json(stage/"run_state.json",state);atomic_json(stage/"contract.json",{"schema":SCHEMA,"config_sha256":sha256_file(config_path),"amendment_sha256":sha256_file(repo/config["parent_amendment"]["path"]),"baseline_audit_sha256":sha256_file(repo/config["baseline_audit"]["path"]),"git":git,"split":"Development/Dev-Select","source_hashes":source_hashes,"finaltest_accessed":False,"test_accessed":False});(stage/"COMPLETED").write_text(MARKER+"\n",encoding="utf-8",newline="\n");atomic_json(stage/"manifest.json",{"schema":SCHEMA,"files":{name:sha256_file(stage/name) for name in RUN_REQUIRED-{"manifest.json","adapters"}},"adapters":{str(path.parent.relative_to(stage/"adapters")).replace("\\","/"):directory_hash(path.parent) for path in sorted((stage/"adapters").rglob("adapter_model.pt"))},"published_atomically":True,"test_accessed":False});destination.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,destination);return {"status":"COMPLETED","run_dir":str(destination),"selection":selection["status"],"selected_candidate":selection["selected"]["candidate_id"] if selection["selected"] else None,"effective_response_scale":selection["effective_response_scale"],"test_accessed":False}
    finally:
        os.chdir(previous)
        if stage.exists():shutil.rmtree(stage)
        gc.collect();torch.cuda.empty_cache()


def analyze(workspace: Path, config: dict[str, Any], run_name: str) -> dict[str, Any]:
    run=workspace/config["output_root"]/"runs"/safe_run_name(run_name)
    if not run.is_dir() or {path.name for path in run.iterdir()}!=RUN_REQUIRED or (run/"COMPLETED").read_text(encoding="utf-8")!=MARKER+"\n":raise ValueError("invalid Stage 8 run")
    manifest=_json(run/"manifest.json")
    for name,expected in manifest["files"].items():
        if sha256_file(run/name)!=expected:raise ValueError(f"Stage 8 artifact changed: {name}")
    for relative,expected in manifest["adapters"].items():
        if directory_hash(run/"adapters"/relative)!=expected:raise ValueError(f"Stage 8 adapter changed: {relative}")
    selection=_json(run/"selection.json")
    return {"status":"COMPLETED","run_dir":str(run),"selection":selection["status"],"selected_candidate":selection["selected"]["candidate_id"] if selection["selected"] else None,"effective_response_scale":selection["effective_response_scale"],"test_accessed":False}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--repo",type=Path,default=Path.cwd());parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--config",type=Path,default=Path("configs/ali_stage8_bota_response_sweep_v1.yaml"));parser.add_argument("--mode",choices=["Preflight","SyntheticDryRun","Full","Analyze"],default="Preflight");parser.add_argument("--run-name",default="ali_stage8_bota_response_seed42_v1");args=parser.parse_args();repo=args.repo.resolve();workspace=args.workspace.resolve();config_path=(repo/args.config).resolve() if not args.config.is_absolute() else args.config.resolve();config=load_config(repo,config_path)
    if args.mode=="Preflight":result=preflight(repo,workspace,config_path)
    elif args.mode=="SyntheticDryRun":result=synthetic(config)
    elif args.mode=="Full":result=execute(repo,workspace,config_path,args.run_name)
    else:result=analyze(workspace,config,args.run_name)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
