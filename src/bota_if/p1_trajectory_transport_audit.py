"""P0 + P1-A audit for blockwise optimizer-aware trajectory transport.

The only optimizer steps in this module occur in four disposable in-memory
arms.  No model, parameter, optimizer, token, prompt, gradient, or raw user ID
is published.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import T5Tokenizer

from src.diagnostics.t5_lora_influence_feasibility_audit_a2 import (
    collect_qv_modules,
    install_fixed_ab_coordinate,
)
from src.diagnostics.t5_reconstructed_official import (
    JsonPromptDataset,
    load_config,
    load_legacy_model,
    move_batch,
)
from src.if_a2_optimization.group_a_gradient_audit import (
    GIB,
    masked_batch,
    masked_forward,
    token_and_sample_numerators,
)
from src.paper_baselines.common import capture_rng, restore_rng, tensor_tree_hash
from src.paper_if_a2.common import (
    atomic_json,
    canonical_hash,
    git_snapshot,
    hardware_snapshot,
    safe_run_name,
    sha256_file,
)

SCHEMA = "bota-if-p0-p1a-trajectory-transport-v1"
MARKER = "BOTA_IF_P0_P1A_TRAJECTORY_TRANSPORT_V1_COMPLETED"
VARIANTS = ("T0_SGD", "T1_AdamW_frozen_v", "T2_AdamW_full_state")


def load_frozen_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema", "test_access_policy", "experiment_root", "base_config",
        "train_data", "raw_data", "original_checkpoint", "output_root",
        "authority", "coordinate", "schedule", "optimizer", "transport",
        "gates", "runtime", "privacy", "scientific_scope",
    }
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA:
        raise ValueError("invalid BOTA-IF P1 schema")
    if value["test_access_policy"] != "forbidden":
        raise ValueError("FinalTest access must remain forbidden")
    if value["coordinate"] != {
        "target_modules": ["q", "v"], "module_count": 72,
        "lora_rank": 16, "lora_alpha": 32, "trainable_coordinate": "B_only",
        "initial_B": "zero",
        "fixed_a_source": "deterministic_request_independent_random_orthonormal",
        "fixed_a_seed": 42, "transport_rank_per_module": 8,
        "total_transport_rank": 576,
        "transport_source": "disjoint_full_train_calibration_batches",
    }:
        raise ValueError("P0 coordinate changed")
    if value["schedule"] != {
        "seed": 42, "steps": 50, "batch_size": 16,
        "calibration_batches": 16,
        "calibration_disjoint_from_training_window": True,
        "users": ["high_frequency", "median_frequency", "low_frequency"],
        "user_selection_uses_outcomes": False,
    }:
        raise ValueError("P1 schedule changed")
    if value["optimizer"] != {
        "name": "AdamW", "learning_rate": .001, "betas": [.9, .999],
        "eps": 1e-8, "weight_decay": .01, "scheduler": "none",
        "gradient_clipping": "none",
    }:
        raise ValueError("optimizer changed")
    if value["transport"] != {
        "curvature": "per_sample_block_diagonal_empirical_fisher",
        "variants": list(VARIANTS), "primary": "T2_AdamW_full_state",
        "adamw_v_linearization": "full_first_order_diagonal_state",
    }:
        raise ValueError("transport changed")
    if value["runtime"]["physical_optimizer_step_limit"] != 200:
        raise ValueError("physical optimizer-step limit changed")
    scope = value["scientific_scope"]
    if scope != {
        "masked_reference": True, "masked_slots_preserved": True,
        "masked_batch_denominator_preserved": True,
        "zero_authoritative_update": True,
        "compacted_retrain_reference": False,
        "certified_unlearning_claimed": False,
        "historical_equivalence_claimed": False,
    }:
        raise ValueError("scientific scope changed")
    forbidden = ("development", "validation", "final_test", "finaltest")
    for key in ("train_data", "raw_data", "original_checkpoint", "output_root"):
        if any(part.lower() in forbidden for part in Path(value[key]).parts):
            raise ValueError(f"forbidden split path: {key}")
    return value


def _train_user_ids_only(raw_dir: Path) -> tuple[list[int], dict[str, Any]]:
    """Replay only the formal train slice; do not materialize Development."""
    user_ids = {int(line.split("::", 1)[0]) for line in (raw_dir / "users.dat").read_text(encoding="utf-8").splitlines()}
    movie_ids = {int(line.split("::", 1)[0]) for line in (raw_dir / "movies.dat").read_text(encoding="ISO-8859-1").splitlines()}
    ratings = []
    for row, line in enumerate((raw_dir / "ratings.dat").read_text(encoding="utf-8").splitlines()):
        user, movie, rating, timestamp = map(int, line.split("::"))
        if user in user_ids and movie in movie_ids:
            ratings.append((timestamp, user, movie, rating, row))
    frame = pd.DataFrame(ratings, columns=["timestamp", "user", "movie", "rating", "raw_row"])
    frame.sort_values(["timestamp", "user", "movie"], kind="stable", inplace=True)
    counts = frame.groupby("user").size()
    filtered_count = len(frame) - sum(min(5, int(count)) for count in counts)
    train_begin, train_end = filtered_count - 100_000, filtered_count - 40_000
    histories = Counter(); filtered_index = 0; train_users: list[int] = []
    for row in frame.itertuples(index=False):
        user = int(row.user)
        if histories[user] >= 5:
            if train_begin <= filtered_index < train_end:
                train_users.append(user)
            filtered_index += 1
        histories[user] += 1
    if filtered_index != filtered_count or len(train_users) != 60_000:
        raise RuntimeError("train-only raw replay mismatch")
    return train_users, {
        "raw_unsplit_source_read": True, "train_rows": len(train_users),
        "train_slice": [train_begin, train_end],
        "development_rows_materialized": 0, "final_test_rows_materialized": 0,
        "train_user_order_sha256": canonical_hash(train_users),
    }


def _seed_for_name(name: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little") % (2**63 - 1)


def deterministic_fixed_a(names_and_modules: Sequence[tuple[str, torch.nn.Linear]], rank: int, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    bases: dict[str, torch.Tensor] = {}; hashes = {}
    for name, module in names_and_modules:
        generator = torch.Generator(device="cpu"); generator.manual_seed(_seed_for_name(name, seed))
        matrix = torch.randn((module.in_features, rank), dtype=torch.float64, generator=generator)
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        for column in range(rank):
            pivot = int(torch.argmax(torch.abs(q[:, column])))
            if float(q[pivot, column]) < 0: q[:, column].mul_(-1)
        basis = q.T.contiguous()
        residual = float(torch.linalg.matrix_norm(basis @ basis.T - torch.eye(rank, dtype=torch.float64)))
        if residual > 1e-10: raise RuntimeError("fixed-A orthonormality failure")
        bases[name] = basis; hashes[name] = tensor_tree_hash({name: basis})
    return bases, {"module_hashes": hashes, "aggregate_sha256": canonical_hash(hashes), "request_data_used": False}


def freeze_schedule(user_ids: Sequence[int], *, seed: int, steps: int, batch_size: int, calibration_batches: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu"); generator.manual_seed(seed)
    order = torch.randperm(len(user_ids), generator=generator).tolist()
    train_count = steps * batch_size; calibration_count = calibration_batches * batch_size
    train = order[:train_count]; calibration = order[train_count:train_count + calibration_count]
    if set(train) & set(calibration): raise RuntimeError("calibration/training overlap")
    global_counts = Counter(map(int, user_ids)); window_users = sorted({int(user_ids[index]) for index in train})
    by_frequency = sorted(window_users, key=lambda user: (global_counts[user], user))
    low = by_frequency[0]; high = by_frequency[-1]
    median_frequency = float(np.median([global_counts[user] for user in window_users]))
    median = min((user for user in window_users if user not in {low, high}), key=lambda user: (abs(global_counts[user]-median_frequency), user))
    selected = [high, median, low]
    if len(set(selected)) != 3 or any(not any(int(user_ids[i]) == user for i in train) for user in selected):
        raise RuntimeError("three frequency-stratified users unavailable")
    return {
        "train_indices": train, "calibration_indices": calibration,
        "selected_users": selected,
        "public": {
            "train_samples": len(train), "calibration_samples": len(calibration),
            "train_order_sha256": canonical_hash(train),
            "batch_order_sha256": canonical_hash([train[i:i+batch_size] for i in range(0, len(train), batch_size)]),
            "calibration_order_sha256": canonical_hash(calibration),
            "selected_user_roles": ["high_frequency", "median_frequency", "low_frequency"],
            "selected_user_frequency": [global_counts[user] for user in selected],
            "selected_user_window_visits": [sum(int(user_ids[i]) == user for i in train) for user in selected],
            "selected_user_hashes": [hashlib.sha256(f"bota-p1:{seed}:{user}".encode()).hexdigest() for user in selected],
            "raw_user_ids_persisted": False, "selection_uses_outcomes": False,
            "calibration_disjoint": True,
        },
    }


def _sample_losses(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    output = masked_forward(model, batch)
    _, sample_sum, _ = token_and_sample_numerators(output.logits, batch["target_ids"])
    # Recompute the vector explicitly; token_and_sample_numerators intentionally returns sums.
    labels = batch["target_ids"]; losses = torch.nn.functional.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]), labels.reshape(-1),
        ignore_index=-100, reduction="none").reshape_as(labels)
    mask = labels.ne(-100); per_sample = (losses * mask).sum(1) / mask.sum(1)
    if not torch.allclose(per_sample.sum(), sample_sum, atol=1e-6, rtol=1e-6):
        raise RuntimeError("sample-loss reduction mismatch")
    return per_sample


def _fresh_runtime(root: Path, config: dict[str, Any], device: torch.device, fixed_a: dict[str, torch.Tensor] | None = None):
    model = load_legacy_model(root / config["original_checkpoint"]).to(device).train()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    modules = collect_qv_modules(model)
    if fixed_a is None:
        fixed_a, fixed_report = deterministic_fixed_a(modules, config["coordinate"]["lora_rank"], config["coordinate"]["fixed_a_seed"])
    else: fixed_report = None
    names, parameters = install_fixed_ab_coordinate(model, fixed_a, config["coordinate"]["lora_alpha"])
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if name not in set(names)):
        raise RuntimeError("non-B parameter trainable")
    return model, names, parameters, fixed_a, fixed_report


def build_transport_bases(model, dataset, parameters, names, indices, device, pad, batch_size, rank):
    snapshots: dict[str, list[torch.Tensor]] = {name: [] for name in names}; losses = []
    for start in range(0, len(indices), batch_size):
        chosen = indices[start:start+batch_size]; batch = move_batch(masked_batch(dataset, chosen, pad), device)
        sample_losses = _sample_losses(model, batch); loss = sample_losses.mean()
        gradients = torch.autograd.grad(loss, parameters)
        for name, gradient in zip(names, gradients): snapshots[name].append(gradient.detach().float().cpu().reshape(-1))
        losses.append(float(loss.detach())); del batch, sample_losses, loss, gradients
    bases: dict[str, torch.Tensor] = {}; reports = {}
    for name in names:
        matrix = torch.stack(snapshots[name]).double(); _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        rows = vh[:rank].clone()
        if rows.shape[0] != rank: raise RuntimeError("transport basis rank unavailable")
        for row in rows:
            pivot = int(torch.argmax(torch.abs(row)))
            if float(row[pivot]) < 0: row.mul_(-1)
        basis = rows.T.contiguous(); residual = float(torch.linalg.matrix_norm(basis.T @ basis - torch.eye(rank, dtype=torch.float64)))
        bases[name] = basis
        reports[name] = {"singular_values": singular[:rank].tolist(), "orthonormality_residual": residual, "basis_sha256": tensor_tree_hash({name: basis})}
    return bases, {"modules": reports, "aggregate_sha256": canonical_hash({name: row["basis_sha256"] for name, row in reports.items()}), "calibration_mean_loss": float(np.mean(losses)), "gradients_persisted": False, "bases_persisted": False}


def adamw_tangent_step(theta, gradient, m, v, dtheta, dm, dv, dgradient, *, step, lr, beta1, beta2, eps, weight_decay, full_v):
    m_new = beta1*m + (1-beta1)*gradient; v_new = beta2*v + (1-beta2)*gradient.square()
    dm_new = beta1*dm + (1-beta1)*dgradient
    dv_new = beta2*dv + 2*(1-beta2)*gradient*dgradient if full_v else torch.zeros_like(dv)
    bc1, bc2 = 1-beta1**step, 1-beta2**step
    mhat, vhat = m_new/bc1, v_new/bc2; root = torch.sqrt(vhat); denominator = root + eps
    d_mhat = dm_new/bc1; d_vhat = dv_new/bc2
    d_update = d_mhat/denominator - mhat*d_vhat/(2*torch.clamp(root, min=1e-30)*denominator.square())
    next_theta = (1-lr*weight_decay)*dtheta - lr*d_update
    return next_theta, dm_new, dv_new


def _new_transport_state(user_count: int, module_count: int, rank: int) -> dict[str, list[list[dict[str, torch.Tensor]]]]:
    return {variant: [[{"theta": torch.zeros(rank, dtype=torch.float64), "m": torch.zeros(rank, dtype=torch.float64), "v": torch.zeros(rank, dtype=torch.float64)} for _ in range(module_count)] for _ in range(user_count)] for variant in VARIANTS}


def _advance_transports(states, coefficient_rows, sources, bases, parameters, optimizer, step, opt):
    beta1, beta2 = map(float, opt["betas"]); lr=float(opt["learning_rate"]); eps=float(opt["eps"]); wd=float(opt["weight_decay"])
    for module_index, (basis, parameter, coefficients) in enumerate(zip(bases, parameters, coefficient_rows)):
        basis = basis.double(); gradient = parameter.grad.detach().double().cpu().reshape(-1)
        state = optimizer.state.get(parameter, {}); m = state.get("exp_avg", torch.zeros_like(parameter)).detach().double().cpu().reshape(-1); v = state.get("exp_avg_sq", torch.zeros_like(parameter)).detach().double().cpu().reshape(-1)
        for user_index, source in enumerate(sources[module_index]):
            for variant in VARIANTS:
                current = states[variant][user_index][module_index]; z = current["theta"]
                fisher_z = coefficients.T @ (coefficients @ z) / coefficients.shape[0]
                dz = fisher_z + source
                if variant == "T0_SGD":
                    current["theta"] = (1-lr*wd)*z - lr*dz
                    continue
                full_theta = basis @ z; full_dm = basis @ current["m"]; full_dv = basis @ current["v"]; full_dg = basis @ dz
                next_theta, next_m, next_v = adamw_tangent_step(
                    parameter.detach().double().cpu().reshape(-1), gradient, m, v,
                    full_theta, full_dm, full_dv, full_dg, step=step, lr=lr,
                    beta1=beta1, beta2=beta2, eps=eps, weight_decay=wd,
                    full_v=variant == "T2_AdamW_full_state")
                current["theta"] = basis.T @ next_theta; current["m"] = basis.T @ next_m; current["v"] = basis.T @ next_v


class StepBudget:
    def __init__(self, limit: int): self.limit=limit; self.calls=0; self.by_arm: dict[str,int]={}
    def step(self, optimizer: torch.optim.Optimizer, arm: str) -> None:
        if self.calls >= self.limit: raise RuntimeError("physical optimizer-step budget exhausted")
        optimizer.step(); self.calls += 1; self.by_arm[arm] = self.by_arm.get(arm, 0) + 1


def _optimizer(parameters, config):
    opt=config["optimizer"]
    return torch.optim.AdamW(parameters, lr=opt["learning_rate"], betas=tuple(opt["betas"]), eps=opt["eps"], weight_decay=opt["weight_decay"])


def run_canonical(model, dataset, indices, user_ids, selected_users, parameters, names, transport_bases, device, pad, config, budget):
    optimizer=_optimizer(parameters,config); states=_new_transport_state(3,len(parameters),config["coordinate"]["transport_rank_per_module"]); traces=[]
    batch_size=config["schedule"]["batch_size"]
    for step,start in enumerate(range(0,len(indices),batch_size),1):
        chosen=indices[start:start+batch_size]; batch_users=[int(user_ids[i]) for i in chosen]; batch=move_batch(masked_batch(dataset,chosen,pad),device); losses=_sample_losses(model,batch)
        per_module_samples=[[] for _ in parameters]; summed=[torch.zeros_like(parameter) for parameter in parameters]
        for sample in range(len(chosen)):
            gradients=torch.autograd.grad(losses[sample],parameters,retain_graph=sample+1<len(chosen))
            for module_index,(gradient,basis) in enumerate(zip(gradients,transport_bases)):
                summed[module_index].add_(gradient.detach()/len(chosen)); per_module_samples[module_index].append((basis.T@gradient.detach().double().cpu().reshape(-1)).detach())
        optimizer.zero_grad(set_to_none=True)
        for parameter,gradient in zip(parameters,summed): parameter.grad=gradient
        coefficient_rows=[torch.stack(rows) for rows in per_module_samples]
        sources=[]
        for coefficients in coefficient_rows:
            sources.append([-coefficients[[i for i,user in enumerate(batch_users) if user==target]].sum(0)/len(chosen) if target in batch_users else torch.zeros(coefficients.shape[1],dtype=torch.float64) for target in selected_users])
        _advance_transports(states,coefficient_rows,sources,transport_bases,parameters,optimizer,step,config["optimizer"])
        total_loss=float(losses.mean().detach()); grad_norm=math.sqrt(sum(float(torch.sum(g.double().square())) for g in summed)); budget.step(optimizer,"canonical_reference")
        traces.append({"step":step,"loss":total_loss,"gradient_norm":grad_norm,"batch_hash":canonical_hash(chosen),"selected_user_slot_counts":[batch_users.count(user) for user in selected_users]})
        del batch,losses,per_module_samples,summed,coefficient_rows,sources
    return {name:parameter.detach().double().cpu().clone() for name,parameter in zip(names,parameters)}, states, traces, tensor_tree_hash(optimizer.state_dict()), tensor_tree_hash(capture_rng())


def run_masked(model,dataset,indices,user_ids,target_user,parameters,names,device,pad,config,budget,arm):
    optimizer=_optimizer(parameters,config); traces=[]; batch_size=config["schedule"]["batch_size"]
    for step,start in enumerate(range(0,len(indices),batch_size),1):
        chosen=indices[start:start+batch_size]; batch_users=[int(user_ids[i]) for i in chosen]; batch=move_batch(masked_batch(dataset,chosen,pad),device); losses=_sample_losses(model,batch)
        weights=torch.tensor([0. if user==target_user else 1. for user in batch_users],dtype=losses.dtype,device=device); loss=torch.sum(losses*weights)/len(chosen)
        optimizer.zero_grad(set_to_none=True); loss.backward(); grad_norm=math.sqrt(sum(float(torch.sum(parameter.grad.detach().double().square())) for parameter in parameters)); budget.step(optimizer,arm)
        traces.append({"step":step,"loss":float(loss.detach()),"gradient_norm":grad_norm,"batch_hash":canonical_hash(chosen),"masked_slots":batch_users.count(target_user),"batch_size_preserved":len(chosen)})
        del batch,losses,weights,loss
    return {name:parameter.detach().double().cpu().clone() for name,parameter in zip(names,parameters)},traces,tensor_tree_hash(optimizer.state_dict()),tensor_tree_hash(capture_rng())


def compare_prediction(actual_by_name, canonical_by_name, state, names, bases, gates):
    actual=[]; predicted=[]; modules=[]
    for index,name in enumerate(names):
        left=(actual_by_name[name]-canonical_by_name[name]).reshape(-1).double(); right=bases[index].double()@state[index]["theta"].double()
        actual.append(left); predicted.append(right); denom=float(torch.linalg.vector_norm(left)*torch.linalg.vector_norm(right)); cosine=float(torch.dot(left,right))/denom if denom else None
        modules.append({"module_hash":hashlib.sha256(name.encode()).hexdigest(),"cosine":cosine,"actual_norm":float(torch.linalg.vector_norm(left)),"predicted_norm":float(torch.linalg.vector_norm(right)),"positive":cosine is not None and cosine>0})
    left=torch.cat(actual); right=torch.cat(predicted); ln=float(torch.linalg.vector_norm(left)); rn=float(torch.linalg.vector_norm(right)); cosine=float(torch.dot(left,right))/max(ln*rn,1e-300); ratio=rn/max(ln,1e-300); relative=float(torch.linalg.vector_norm(right-left))/max(ln,1e-300); positive=sum(row["positive"] for row in modules)/len(modules); high_energy=sorted(modules,key=lambda row:row["actual_norm"],reverse=True)[:max(1,math.ceil(len(modules)*.25))]; high_energy_positive=sum(row["positive"] for row in high_energy)/len(high_energy)
    return {"cosine":cosine,"norm_ratio":ratio,"relative_l2_error":relative,"positive_module_fraction":positive,"high_energy_positive_module_fraction":high_energy_positive,"high_energy_definition":"top_actual_delta_norm_quartile","actual_delta_norm":ln,"predicted_delta_norm":rn,"actual_delta_sha256":tensor_tree_hash({"delta":left}),"predicted_delta_sha256":tensor_tree_hash({"delta":right}),"modules":modules,"base_gates_passed":cosine>=gates["cosine_minimum"] and gates["norm_ratio_minimum"]<=ratio<=gates["norm_ratio_maximum"] and relative<=gates["relative_l2_maximum"] and positive>=gates["positive_module_fraction_minimum"] and high_energy_positive>=gates["positive_module_fraction_minimum"]}


def _finite(value: Any) -> bool:
    if isinstance(value,float): return math.isfinite(value)
    if isinstance(value,dict): return all(_finite(item) for item in value.values())
    if isinstance(value,list): return all(_finite(item) for item in value)
    return True


def validate_authority(root: Path, config: dict[str,Any]) -> dict[str,Any]:
    experiment=root/config["experiment_root"]; manifest=json.loads((experiment/"experiment_manifest.json").read_text(encoding="utf-8")); authority=config["authority"]
    if manifest.get("experiment_contract_sha256")!=authority["experiment_contract_sha256"] or manifest.get("counts",{}).get("train")!=authority["train_samples"] or manifest.get("test_accessed") is not False or manifest.get("processed_test_split_read") is not False: raise ValueError("2% experiment authority mismatch")
    train=root/config["train_data"]; checkpoint=root/config["original_checkpoint"]
    if sha256_file(train)!=authority["train_sha256"] or sha256_file(checkpoint)!=authority["original_sha256"]: raise ValueError("train/checkpoint SHA mismatch")
    return {"experiment_contract_sha256":authority["experiment_contract_sha256"],"train_sha256":sha256_file(train),"original_sha256":sha256_file(checkpoint),"test_accessed":False}


def preflight(root:Path,config_path:Path)->dict[str,Any]:
    config=load_frozen_config(config_path); authority=validate_authority(root,config)
    return {"schema":SCHEMA,"mode":"Preflight","authority":authority,"coordinate":config["coordinate"],"schedule":config["schedule"],"transport":config["transport"],"physical_optimizer_step_budget":200,"expected_breakdown":{"canonical_reference":50,"masked_high_frequency":50,"masked_median_frequency":50,"masked_low_frequency":50},"model_loaded":False,"development_loaded":False,"retrain_loaded":False,"test_loader_built":False,"test_accessed":False}


def budget_audit(config:dict[str,Any])->dict[str,Any]:
    steps=config["schedule"]["steps"]; arms=1+len(config["schedule"]["users"]); total=steps*arms
    if total!=config["runtime"]["physical_optimizer_step_limit"]: raise RuntimeError("budget arithmetic mismatch")
    return {"schema":SCHEMA,"mode":"BudgetAudit","step_positions":steps,"canonical_optimizer_steps":steps,"masked_optimizer_steps":3*steps,"total_physical_optimizer_step_calls":total,"hard_limit":200,"zero_authoritative_update":True,"test_accessed":False}


def synthetic()->dict[str,Any]:
    theta=torch.tensor([.3,-.2],dtype=torch.float64); gradient=torch.tensor([.2,-.1],dtype=torch.float64); m=torch.tensor([.05,.02],dtype=torch.float64); v=torch.tensor([.03,.04],dtype=torch.float64); dtheta=torch.tensor([.01,-.02],dtype=torch.float64); dm=torch.tensor([.003,-.004],dtype=torch.float64); dv=torch.tensor([.002,.001],dtype=torch.float64); dg=torch.tensor([-.01,.03],dtype=torch.float64); kwargs={"step":4,"lr":.001,"beta1":.9,"beta2":.999,"eps":1e-8,"weight_decay":.01,"full_v":True}
    analytic=adamw_tangent_step(theta,gradient,m,v,dtheta,dm,dv,dg,**kwargs)[0]; epsilon=1e-6
    def update(t,g,mm,vv):
        mn=.9*mm+.1*g; vn=.999*vv+.001*g.square(); return (1-.001*.01)*t-.001*(mn/(1-.9**4))/(torch.sqrt(vn/(1-.999**4))+1e-8)
    numeric=(update(theta+epsilon*dtheta,gradient+epsilon*dg,m+epsilon*dm,v+epsilon*dv)-update(theta,gradient,m,v))/epsilon; error=float(torch.linalg.vector_norm(analytic-numeric))/float(torch.linalg.vector_norm(numeric))
    users=[1]*20+[2]*10+[3]*5+[4]*4+[5]*3; frozen=freeze_schedule(users,seed=42,steps=2,batch_size=4,calibration_batches=2)
    return {"schema":SCHEMA,"adamw_full_state_finite_difference_relative_error":error,"adamw_full_state_gate":error<1e-5,"schedule_disjoint":not bool(set(frozen["train_indices"])&set(frozen["calibration_indices"])),"real_model_loaded":False,"optimizer_steps":0,"test_accessed":False}


def publish(stage:Path,destination:Path,report:dict[str,Any],implementation_sha:str)->None:
    if not _finite(report): raise ValueError("nonfinite P1 report")
    atomic_json(stage/"p0_p1a_trajectory_transport.json",report); atomic_json(stage/"run_state.json",{"schema":SCHEMA,"status":"COMPLETED","classification":report["classification"],"all_gates_passed":report["all_gates_passed"],"physical_optimizer_step_calls":report["execution"]["physical_optimizer_step_calls"],"authoritative_parameters_modified":False,"test_accessed":False}); (stage/"COMPLETED").write_text(MARKER+"\n",encoding="utf-8",newline="\n"); atomic_json(stage/"manifest.json",{"schema":SCHEMA,"status":"COMPLETED","report_sha256":sha256_file(stage/"p0_p1a_trajectory_transport.json"),"run_state_sha256":sha256_file(stage/"run_state.json"),"implementation_sha256":implementation_sha,"published_atomically":True,"model_artifact_published":False,"optimizer_artifact_published":False,"test_accessed":False}); destination.parent.mkdir(parents=True,exist_ok=True); os.replace(stage,destination)


def analyze(root:Path,config_path:Path,run_name:str)->dict[str,Any]:
    config=load_frozen_config(config_path); run=root/config["output_root"]/safe_run_name(run_name); expected={"COMPLETED","manifest.json","p0_p1a_trajectory_transport.json","run_state.json"}
    if not run.is_dir() or {p.name for p in run.iterdir()}!=expected or (run/"COMPLETED").read_text(encoding="utf-8")!=MARKER+"\n": raise ValueError("invalid P1 run")
    report=json.loads((run/"p0_p1a_trajectory_transport.json").read_text(encoding="utf-8")); manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8")); state=json.loads((run/"run_state.json").read_text(encoding="utf-8"))
    if manifest.get("report_sha256")!=sha256_file(run/"p0_p1a_trajectory_transport.json") or manifest.get("run_state_sha256")!=sha256_file(run/"run_state.json") or state.get("physical_optimizer_step_calls")!=200 or report.get("test_accessed") is not False: raise ValueError("P1 evidence mismatch")
    return {"status":"COMPLETED","run_dir":str(run),"classification":report["classification"],"all_gates_passed":report["all_gates_passed"],"users":report["users"],"execution":report["execution"],"primary_summary":report["primary_summary"],"test_accessed":False}


def execute(root:Path,config_path:Path,run_name:str)->dict[str,Any]:
    config=load_frozen_config(config_path); run_name=safe_run_name(run_name); destination=(root/config["output_root"]/run_name).resolve()
    if destination.exists(): raise FileExistsError(destination)
    authority=validate_authority(root,config); git=git_snapshot(root)
    if not git["clean"]: raise RuntimeError("formal P1 audit requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=config["runtime"]["required_cuda_devices"]: raise RuntimeError("one CUDA GPU required")
    device=torch.device("cuda:0"); free,total=torch.cuda.mem_get_info(device)
    if free/GIB<config["runtime"]["minimum_free_gib"]: raise RuntimeError("insufficient clean dedicated GPU memory")
    train_users,replay=_train_user_ids_only(root/config["raw_data"]); schedule=freeze_schedule(train_users,seed=42,steps=50,batch_size=16,calibration_batches=config["schedule"]["calibration_batches"])
    work=destination.parent/".work"; work.mkdir(parents=True,exist_ok=True); stage=work/f"{run_name}.{uuid.uuid4().hex}.stage"; stage.mkdir(); started=time.perf_counter(); checkpoint_before=sha256_file(root/config["original_checkpoint"]); budget=StepBudget(200); model=None; initial_rng=None
    try:
        torch.cuda.set_per_process_memory_fraction(config["runtime"]["allocator_fraction"],device); torch.cuda.reset_peak_memory_stats(); base=load_config(root/config["base_config"],root); tokenizer=T5Tokenizer.from_pretrained(base["paths"]["model_dir"]); pad=tokenizer.pad_token_id
        if pad is None: raise RuntimeError("masked protocol requires pad token")
        dataset=JsonPromptDataset(root/config["train_data"],tokenizer)
        if len(dataset)!=len(train_users): raise RuntimeError("train/user lineage mismatch")
        initial_rng=capture_rng(); random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42); arm_rng=capture_rng()
        model,names,parameters,fixed_a,fixed_report=_fresh_runtime(root,config,device); transport_bases,transport_report=build_transport_bases(model,dataset,parameters,names,schedule["calibration_indices"],device,pad,16,8); del model; model=None; gc.collect(); torch.cuda.empty_cache()
        basis_list=[transport_bases[name] for name in names]
        restore_rng(arm_rng); model,names,parameters,_,_=_fresh_runtime(root,config,device,fixed_a); canonical,states,canonical_trace,canonical_optimizer_hash,canonical_rng_hash=run_canonical(model,dataset,schedule["train_indices"],train_users,schedule["selected_users"],parameters,names,basis_list,device,pad,config,budget); canonical_parameter_hash=tensor_tree_hash(canonical); del model; model=None; gc.collect(); torch.cuda.empty_cache()
        users=[]
        for user_index,(role,target) in enumerate(zip(config["schedule"]["users"],schedule["selected_users"])):
            restore_rng(arm_rng); model,names,parameters,_,_=_fresh_runtime(root,config,device,fixed_a); actual,trace,optimizer_hash,rng_hash=run_masked(model,dataset,schedule["train_indices"],train_users,target,parameters,names,device,pad,config,budget,f"masked_{role}"); comparisons={variant:compare_prediction(actual,canonical,states[variant][user_index],names,basis_list,config["gates"]) for variant in VARIANTS}; t0=comparisons["T0_SGD"];t1=comparisons["T1_AdamW_frozen_v"];t2=comparisons["T2_AdamW_full_state"]; optimizer_aware=t2["relative_l2_error"]<t0["relative_l2_error"] and t2["relative_l2_error"]<=t1["relative_l2_error"]+1e-12; rng_exact=rng_hash==canonical_rng_hash; passed=t2["base_gates_passed"] and optimizer_aware and rng_exact
            users.append({"role":role,"user_hash":schedule["public"]["selected_user_hashes"][user_index],"global_frequency":schedule["public"]["selected_user_frequency"][user_index],"window_visits":schedule["public"]["selected_user_window_visits"][user_index],"masked_reference":{"slots_preserved":True,"batch_denominator_preserved":True,"steps":50,"trace_sha256":canonical_hash(trace),"final_parameter_sha256":tensor_tree_hash(actual),"optimizer_state_sha256":optimizer_hash,"final_rng_sha256":rng_hash,"canonical_rng_exact":rng_exact,"model_mode":"train"},"predictions":comparisons,"optimizer_aware_gate_passed":optimizer_aware,"all_gates_passed":passed}); del model,actual; model=None; gc.collect(); torch.cuda.empty_cache()
        if budget.calls!=200 or budget.by_arm!={"canonical_reference":50,"masked_high_frequency":50,"masked_median_frequency":50,"masked_low_frequency":50}: raise RuntimeError("physical optimizer-step accounting mismatch")
        all_passed=all(row["all_gates_passed"] for row in users); classification="trajectory_transport_supported" if all_passed else "trajectory_transport_not_supported"
        source_unchanged=sha256_file(root/config["original_checkpoint"])==checkpoint_before; peak=torch.cuda.max_memory_reserved()/GIB
        if not source_unchanged or peak>config["runtime"]["hard_peak_reserved_gib"]: raise RuntimeError("source/memory integrity failure")
        implementation={str(path.relative_to(root)).replace("\\","/"):sha256_file(path) for path in (Path(__file__),config_path,root/"scripts/bota_if/run_p1_trajectory_transport_audit_v1.ps1")}; implementation_sha=canonical_hash(implementation)
        report={"schema":SCHEMA,"run_name":run_name,"status":"COMPLETED","classification":classification,"all_gates_passed":all_passed,"authority":authority,"p0":{"fixed_a":fixed_report,"transport_coordinate":transport_report,"module_count":72,"rank_per_module":8,"total_rank":576,"request_independent":True},"schedule":schedule["public"],"users":users,"primary_summary":{"primary":"T2_AdamW_full_state","users_passing":sum(row["all_gates_passed"] for row in users),"users_total":3,"t2_beats_t0_all_users":all(row["predictions"]["T2_AdamW_full_state"]["relative_l2_error"]<row["predictions"]["T0_SGD"]["relative_l2_error"] for row in users),"t2_not_worse_than_t1_all_users":all(row["predictions"]["T2_AdamW_full_state"]["relative_l2_error"]<=row["predictions"]["T1_AdamW_frozen_v"]["relative_l2_error"]+1e-12 for row in users),"canonical_rng_exact_all_users":all(row["masked_reference"]["canonical_rng_exact"] for row in users)},"execution":{"step_positions":50,"canonical_optimizer_steps":50,"masked_optimizer_steps":150,"physical_optimizer_step_calls":budget.calls,"physical_optimizer_step_limit":budget.limit,"authoritative_optimizer_steps_committed":0,"by_arm":budget.by_arm,"canonical_trace_sha256":canonical_hash(canonical_trace),"canonical_parameter_sha256":canonical_parameter_hash,"canonical_optimizer_state_sha256":canonical_optimizer_hash,"canonical_final_rng_sha256":canonical_rng_hash},"lineage":replay,"integrity":{"source_checkpoint_unchanged":source_unchanged,"source_checkpoint_sha256_before":checkpoint_before,"source_checkpoint_sha256_after":sha256_file(root/config["original_checkpoint"]),"authoritative_parameters_modified":False,"authoritative_optimizer_steps_committed":0,"model_artifact_published":False,"optimizer_artifact_published":False,"development_loaded":False,"retrain_loaded":False,"test_loader_built":False},"scientific_scope":config["scientific_scope"],"privacy":config["privacy"],"memory":{"peak_reserved_gib":peak,"hard_peak_reserved_gib":config["runtime"]["hard_peak_reserved_gib"],"device_total_gib":total/GIB},"hardware":hardware_snapshot(),"git":git,"implementation":implementation,"implementation_sha256":implementation_sha,"wall_time_seconds":time.perf_counter()-started,"test_accessed":False}; publish(stage,destination,report,implementation_sha); restore_rng(initial_rng)
        return {"status":"COMPLETED","run_dir":str(destination),"classification":classification,"all_gates_passed":all_passed,"physical_optimizer_step_calls":200,"test_accessed":False}
    except Exception as error:
        if stage.exists() and not destination.exists():
            failure={"schema":SCHEMA,"status":"INTERRUPTED","reason":type(error).__name__,"message":str(error),"physical_optimizer_step_calls":budget.calls,"physical_optimizer_step_limit":budget.limit,"by_arm":budget.by_arm,"source_checkpoint_sha256_before":checkpoint_before,"source_checkpoint_sha256_after":sha256_file(root/config["original_checkpoint"]),"authoritative_parameters_modified":False,"model_artifact_published":False,"optimizer_artifact_published":False,"development_loaded":False,"retrain_loaded":False,"test_loader_built":False,"test_accessed":False}
            atomic_json(stage/"run_state.json",failure); (stage/"INTERRUPTED").write_text("BOTA_IF_P0_P1A_INTERRUPTED\n",encoding="utf-8",newline="\n"); atomic_json(stage/"manifest.json",{"schema":SCHEMA,"status":"INTERRUPTED","run_state_sha256":sha256_file(stage/"run_state.json"),"physical_optimizer_step_calls":budget.calls,"published_atomically":True,"test_accessed":False}); destination.parent.mkdir(parents=True,exist_ok=True); os.replace(stage,destination)
        raise RuntimeError(f"P1 audit interrupted; immutable evidence published at {destination}") from error
    finally:
        if model is not None: del model
        if initial_rng is not None: restore_rng(initial_rng)
        if stage.exists(): shutil.rmtree(stage)
        gc.collect(); torch.cuda.empty_cache()


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=Path.cwd());parser.add_argument("--config",type=Path,required=True);parser.add_argument("--mode",choices=["Preflight","BudgetAudit","SyntheticDryRun","Full","Analyze"],default="Preflight");parser.add_argument("--run-name");args=parser.parse_args();root=args.root.resolve();config_path=args.config.resolve();config=load_frozen_config(config_path)
    if args.mode=="Preflight": result=preflight(root,config_path)
    elif args.mode=="BudgetAudit": result=budget_audit(config)
    elif args.mode=="SyntheticDryRun": result=synthetic()
    elif not args.run_name: raise ValueError(f"{args.mode} requires --run-name")
    elif args.mode=="Analyze": result=analyze(root,config_path,args.run_name)
    else: result=execute(root,config_path,args.run_name)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__": main()
