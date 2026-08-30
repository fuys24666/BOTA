"""Development-only, zero-update Q/V-LoRA retain-curvature influence audit."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from src.diagnostics.git_provenance import git_provenance, implementation_provenance, require_clean_git
from src.diagnostics.t5_full_runner import _batch
from src.diagnostics.t5_projected_pilot_10step import SOURCE_STATE_SHA, _load_pilot_runtime, load_pilot_config, validate_checkpoint_chain
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, build_current_model, freeze_teacher, load_config, load_legacy_model, move_batch, teacher_cross_entropy
from src.diagnostics.t5_step813_update_space_stage_b import _evaluate_utility
from src.diagnostics.t5_step817_forget_conflict_audit import _all_finite, _data_lineage, _load_user_map, _resolve, _safe_name, atomic_json, atomic_text, canonical_hash, directory_hash, sha256_file, tensor_tree_hash


SCHEMA="t5-lora-influence-feasibility-audit-v1"
UNIT_SCHEMA="t5-lora-influence-feasibility-unit-v1"
ANALYSIS_SCHEMA="t5-lora-influence-feasibility-analysis-v1"
UNIT_MARKER="T5_LORA_INFLUENCE_UNIT_COMPLETED"
TERMINAL_MARKER="T5_LORA_INFLUENCE_FULL_COMPLETED"
CLASSES=("IF-A","IF-B","IF-C","IF-D")
IMPLEMENTATION_FILES=(
    "src/diagnostics/t5_lora_influence_feasibility_audit.py",
    "configs/t5_lora_influence_feasibility_audit_v1.yaml",
    "scripts/diagnostics/t5_lora_influence_feasibility_audit_v1.ps1",
    "docs/t5_lora_influence_feasibility_audit_v1.md",
    "src/diagnostics/t5_reconstructed_official.py",
    "src/diagnostics/t5_projected_pilot_10step.py",
    "src/diagnostics/t5_step813_update_space_stage_b.py",
)
FORBIDDEN_PERSISTED_KEYS={"gradient","gradients","hvp","hvp_vector","krylov","delta","parameter_delta","logits","input_ids","target_ids","raw_sample","raw_samples","optimizer_state","model_state"}


def _read_json(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict):raise ValueError(f"expected object: {path}")
    return value


def _nested_keys(value:Any)->set[str]:
    if isinstance(value,dict):return set(value).union(*(_nested_keys(item) for item in value.values()),set())
    if isinstance(value,list):return set().union(*(_nested_keys(item) for item in value),set())
    return set()


def load_audit_config(path:Path,root:Path)->dict[str,Any]:
    value=yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema")!=SCHEMA or value.get("development_only") is not True or value.get("test_access_policy")!="forbidden":raise ValueError("influence audit scope/schema changed")
    if value.get("lora_coordinate")!={"r":16,"alpha":32,"dropout":.05,"target_modules":["q","v"],"tensors":144,"parameters":1769472,"dtype":"torch.float32","parameter_name_sha256":"cefe4b6c5a327087054a9bc1e229b2c62a26a483e0ceea1eb90bdd5115f1c961","metadata_sha256":"bdc4843c21f2b9443860aa39a388935f50f90840ba2041e9f063d51f80ffbbea","original_representation":"deterministic_qv_lora_with_zero_B_on_authoritative_original_base"}:raise ValueError("Q/V LoRA coordinate changed")
    if value.get("cg")!={"damping":.01,"relative_residual_tolerance":1e-4,"absolute_residual_tolerance":1e-8,"max_iterations":20,"residual_explosion_factor":1000.0,"pap_absolute_tolerance":1e-12}:raise ValueError("CG preregistration changed")
    if value.get("candidates")!={"anchors":["original","step816"],"directions":["raw_influence","retain_safe_influence"],"scales":[1.0,.5,.25],"primary_anchor":"original","primary_direction":"retain_safe_influence","selection_order":[1.0,.5,.25],"retrain_for_selection":"forbidden"}:raise ValueError("candidate preregistration changed")
    if value.get("efficiency",{}).get("comparable_retrain_timing") is not None:raise ValueError("unexpected Retrain timing authority")
    value["_path"]=str(path.resolve());value["_sha256"]=sha256_file(path);return value


def _adapter_metadata(adapter:dict[str,torch.Tensor])->dict[str,Any]:
    rows=[{"name":name,"shape":list(value.shape),"dtype":str(value.dtype),"numel":value.numel()} for name,value in adapter.items()]
    return {"rows":rows,"tensors":len(rows),"parameters":sum(row["numel"] for row in rows),"name_sha256":canonical_hash([row["name"] for row in rows]),"metadata_sha256":canonical_hash(rows),"qv_only":all((".q." in row["name"] or ".v." in row["name"]) and ("lora_A" in row["name"] or "lora_B" in row["name"]) for row in rows)}


def _expected_t5_qv_metadata(model_config:dict[str,Any],rank:int)->dict[str,Any]:
    d_model=int(model_config["d_model"]);encoder_layers=int(model_config["num_layers"]);decoder_layers=int(model_config.get("num_decoder_layers",encoder_layers));rows=[]
    def add(prefix:str)->None:
        for target in ("q","v"):
            rows.append({"name":f"{prefix}.{target}.lora_A.weight","shape":[rank,d_model],"dtype":"torch.float32","numel":rank*d_model});rows.append({"name":f"{prefix}.{target}.lora_B.weight","shape":[d_model,rank],"dtype":"torch.float32","numel":rank*d_model})
    for index in range(encoder_layers):add(f"base_model.model.encoder.block.{index}.layer.0.SelfAttention")
    for index in range(decoder_layers):add(f"base_model.model.decoder.block.{index}.layer.0.SelfAttention");add(f"base_model.model.decoder.block.{index}.layer.1.EncDecAttention")
    return {"rows":rows,"tensors":len(rows),"parameters":sum(row["numel"] for row in rows),"name_sha256":canonical_hash([row["name"] for row in rows]),"metadata_sha256":canonical_hash(rows),"qv_only":True}


def _validate_rd(root:Path,config:dict[str,Any])->dict[str,Any]:
    path=_resolve(root,config["authority"]["rd_analysis"])
    if sha256_file(path)!=config["authority"]["rd_analysis_sha256"]:raise ValueError("RD-D analysis SHA mismatch")
    value=_read_json(path);expected={"category":"RD-D","next_action":"stop_invalid_or_conflicted","yes_no_reliable_conflict":True,"optimizer_steps_committed":0,"step817_checkpoint_published":False,"test_accessed":False}
    if any(value.get(key)!=item for key,item in expected.items()):raise ValueError("RD-D predecessor invariant changed")
    return {"path":str(path),"sha256":sha256_file(path),**expected}


def _validate_pilot(root:Path,config:dict[str,Any])->dict[str,Any]:
    authority=config["authority"];full=_resolve(root,authority["pilot_full"]);checkpoint=_resolve(root,authority["step816_checkpoint"])
    checks={"run_state":(full/"run_state.json",authority["pilot_run_state_sha256"]),"run_manifest":(full/"run_manifest.json",authority["pilot_manifest_sha256"]),"state":(checkpoint/"state.pt",authority["step816_state_sha256"]),"manifest":(checkpoint/"manifest.json",authority["step816_manifest_sha256"])}
    if any(sha256_file(path)!=expected for path,expected in checks.values()):raise ValueError("pilot authority SHA mismatch")
    run_state=_read_json(full/"run_state.json");manifest=_read_json(checkpoint/"manifest.json")
    if run_state.get("status")!="STOPPED_SAFELY" or run_state.get("last_step")!=816 or run_state.get("next_step")!=817 or run_state.get("test_accessed") is not False:raise ValueError("pilot terminal state changed")
    if manifest.get("step")!=816 or manifest.get("next_optimizer_step")!=817 or manifest.get("test_accessed") is not False:raise ValueError("step816 manifest changed")
    chain=validate_checkpoint_chain(full,SOURCE_STATE_SHA,816)
    payload=torch.load(checkpoint/"state.pt",map_location="cpu",weights_only=False);metadata=_adapter_metadata(payload.get("adapter_state",{}));expected=config["lora_coordinate"]
    if metadata["tensors"]!=expected["tensors"] or metadata["parameters"]!=expected["parameters"] or metadata["name_sha256"]!=expected["parameter_name_sha256"] or metadata["metadata_sha256"]!=expected["metadata_sha256"] or not metadata["qv_only"]:raise ValueError("step816 Q/V coordinate mismatch")
    if set(payload.get("rng",{}))!={"python","numpy","torch_cpu","torch_cuda"} or len(payload.get("optimizer_state",{}).get("state",{}))!=144 or payload.get("test_accessed") is not False:raise ValueError("step816 resume payload incomplete")
    del payload
    return {"full":str(full),"checkpoint":str(checkpoint),"chain_steps":chain["steps"],"state_sha256":authority["step816_state_sha256"],"manifest_sha256":authority["step816_manifest_sha256"],"adapter_metadata":metadata,"optimizer_state_complete":True,"rng_complete":True,"sampler_continuation_complete":True,"status":"STOPPED_SAFELY","final_step":816,"next_step":817,"test_accessed":False}


def preflight(root:Path,config_path:Path,*,git_function:Callable=git_provenance,implementation_function:Callable=implementation_provenance)->dict[str,Any]:
    config=load_audit_config(config_path,root);base=load_config(_resolve(root,config["base_config"]),root);rd=_validate_rd(root,config);pilot=_validate_pilot(root,config);paths=base["paths"]
    models={"original":{"path":str(Path(paths["original"]).resolve()),"sha256":sha256_file(Path(paths["original"]))},"retrain":{"path":str(Path(paths["retrain_reference"]).resolve()),"sha256":sha256_file(Path(paths["retrain_reference"]))}}
    tokenizer=directory_hash(Path(paths["model_dir"]));lineage,indices,users=_data_lineage(root,base,_resolve(root,config["protocol_root"]));expected=config["lineage_sha256"]
    actual={"original":models["original"]["sha256"],"retrain":models["retrain"]["sha256"],"tokenizer_directory":tokenizer["canonical_sha256"],"validation":lineage["data"]["overall_validation"]["sha256"],"forget_train":lineage["data"]["forget_train"]["sha256"],"retain_train":lineage["data"]["retain_train"]["sha256"],"validation_user_sidecar":lineage["validation_sidecar"]["sha256"],"forget_validation_indices":lineage["validation_splits"]["forget_user_validation"]["indices_sha256"],"retain_validation_indices":lineage["validation_splits"]["retain_user_validation"]["indices_sha256"]}
    if actual!=expected:raise ValueError("model/data/tokenizer lineage mismatch")
    utility=_resolve(root,config["authority"]["utility_baseline"])
    if sha256_file(utility)!=config["authority"]["utility_baseline_sha256"]:raise ValueError("utility authority SHA mismatch")
    utility_value=_read_json(utility)["transaction"]["post_evidence"]["utility_before"]
    if any(part.get("samples")!=count for part,count in ((utility_value["overall_validation"],20000),(utility_value["retain_user_validation"],16664))):raise ValueError("fixed step812 utility baseline changed")
    lora=base["lora"]
    if lora!={"r":16,"lora_alpha":32,"lora_dropout":.05,"target_modules":["q","v"]}:raise ValueError("base LoRA config incompatible")
    static_coordinate=_expected_t5_qv_metadata(_read_json(Path(paths["model_dir"])/"config.json"),lora["r"])
    if static_coordinate!=pilot["adapter_metadata"]:raise ValueError("Original static Q/V coordinate does not exactly match step816")
    selector=_resolve(root,config["authority"]["development_selector"]);selector_checks=(("manifest.json","development_selector_manifest_sha256"),("rows.jsonl","development_selector_rows_sha256"),("summary.json","development_selector_summary_sha256"))
    if any(sha256_file(selector/name)!=config["authority"][key] for name,key in selector_checks):raise ValueError("development selector authority mismatch")
    selector_summary=_read_json(selector/"summary.json")
    if selector_summary.get("selector_sha256")!=config["authority"]["development_selector_sha256"] or selector_summary.get("active_samples")!=126 or selector_summary.get("test_accessed") is not False:raise ValueError("development selector semantics changed")
    cuda={"available":torch.cuda.is_available(),"device_count":torch.cuda.device_count(),"devices":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    return json.loads(json.dumps({"schema":SCHEMA,"mode":"Preflight","git":git_function(root),"implementation":implementation_function(root,IMPLEMENTATION_FILES),"config_sha256":config["_sha256"],"python":sys.executable,"rd_predecessor":rd,"pilot":pilot,"models":models,"tokenizer":tokenizer,"data_lineage":lineage,"sample_order_sha256":{"forget":expected["forget_validation_indices"],"retain":expected["retain_validation_indices"],"users":expected["validation_user_sidecar"]},"development_selector":{"path":str(selector),"selector_sha256":selector_summary["selector_sha256"],"rows_sha256":config["authority"]["development_selector_rows_sha256"],"active_samples":126,"inactive_samples":3210,"test_accessed":False},"original_anchor":{"base_checkpoint_sha256":models["original"]["sha256"],"representation":config["lora_coordinate"]["original_representation"],"coordinate_compatibility":"confirmed_exact_by_static_current_T5_QV_construction_vs_step816_metadata","static_coordinate":static_coordinate,"primary":True},"step816_anchor":{"secondary":True,"affected_by_forced_teacher_and_projected_pilot":True},"retrain":{"preflight_loaded":False,"selection_use":False,"posthoc_evaluation_only":True},"fixed_utility_baseline":{"step":812,"path":str(utility),"sha256":sha256_file(utility),"metrics":utility_value},"panels":config["panels"],"cg":config["cg"],"curvature":config["curvature"],"projection":config["projection"],"candidates":config["candidates"],"efficiency":{"status":"unavailable","comparable_retrain_timing_found":False,"formal_retrain_benchmark_run":False,"classification_cap":"IF-B"},"environment":{"cuda":cuda},"resource_estimate":config["resource_estimate"],"model_loaded":False,"retrain_loaded":False,"optimizer_constructed":False,"optimizer_steps_committed":0,"test_loader_built":False,"test_accessed":False},sort_keys=True))


def flatten_tensors(values:Sequence[torch.Tensor])->torch.Tensor:
    return torch.cat([value.reshape(-1).to(dtype=torch.float64,device="cpu") for value in values]) if values else torch.empty(0,dtype=torch.float64)


def split_vector(vector:torch.Tensor,like:Sequence[torch.Tensor])->list[torch.Tensor]:
    result=[];offset=0
    for item in like:
        size=item.numel();result.append(vector[offset:offset+size].reshape(item.shape).to(device=item.device,dtype=item.dtype));offset+=size
    if offset!=vector.numel():raise ValueError("flat vector size mismatch")
    return result


def conjugate_gradient(matvec:Callable[[torch.Tensor],torch.Tensor],g:torch.Tensor,*,damping:float,relative_tolerance:float,absolute_tolerance:float,max_iterations:int,residual_explosion_factor:float=1000.0,pap_tolerance:float=1e-12,allow_truncated_solution:bool=False)->dict[str,Any]:
    g=g.detach().to(torch.float64).cpu()
    if not torch.isfinite(g).all() or damping<0 or max_iterations<1:raise ValueError("invalid CG input")
    calls=0
    def A(v):
        nonlocal calls
        value=matvec(v).to(torch.float64).cpu()+damping*v;calls+=1
        if value.shape!=v.shape or not torch.isfinite(value).all():raise FloatingPointError("nonfinite/invalid HVP")
        return value
    x=torch.zeros_like(g);r=g.clone();p=r.clone();initial=float(torch.linalg.vector_norm(r));target=max(absolute_tolerance,relative_tolerance*initial);history=[];converged=initial<=target
    for iteration in range(1,max_iterations+1):
        if converged:break
        started=time.perf_counter();ap=A(p);pap=float(torch.dot(p,ap));p_norm=float(torch.linalg.vector_norm(p));ap_norm=float(torch.linalg.vector_norm(ap));scale=p_norm*ap_norm;negative_tolerance=pap_tolerance+1e-15*scale
        if pap < -negative_tolerance:raise RuntimeError("significantly_negative_pAp")
        if pap <= 0:raise RuntimeError("numerically_indeterminate_pAp")
        normalized_pap=pap/max(float(torch.dot(p,p)),torch.finfo(torch.float64).tiny)
        rr=torch.dot(r,r);alpha=rr/pap;x=x+alpha*p;r_new=r-alpha*ap;norm=float(torch.linalg.vector_norm(r_new));relative=0.0 if initial==0 else norm/initial
        history.append({"iteration":iteration,"residual_norm":norm,"relative_residual":relative,"pAp":pap,"normalized_pAp":normalized_pap,"negative_pAp_tolerance":negative_tolerance,"hvp_calls":calls,"wall_time_seconds":time.perf_counter()-started,"peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,"peak_gpu_reserved_bytes":torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0})
        if not math.isfinite(norm) or norm>residual_explosion_factor*max(initial,absolute_tolerance):raise RuntimeError("residual_explosion")
        if norm<=target:r=r_new;converged=True;break
        beta=torch.dot(r_new,r_new)/rr;p=r_new+beta*p;r=r_new
    actual=g-A(x);actual_norm=float(torch.linalg.vector_norm(actual));actual_relative=0.0 if initial==0 else actual_norm/initial
    if converged and actual_norm>target*(1+1e-8):raise RuntimeError("false_residual_convergence")
    if not converged and not allow_truncated_solution:raise RuntimeError("cg_not_converged")
    if not math.isfinite(actual_norm) or actual_norm>residual_explosion_factor*max(initial,absolute_tolerance):raise RuntimeError("residual_explosion")
    g_dot_solution=float(torch.dot(g,x))
    if not math.isfinite(g_dot_solution) or g_dot_solution<-1e-12:raise RuntimeError("negative_influence_sign")
    return {"solution":x,"converged":converged,"solver_status":"converged" if converged else "max_iterations_reached","tolerance_satisfied":converged,"truncated_solution_used":not converged,"iterations":len(history),"hvp_calls":calls,"initial_residual":initial,"final_residual":actual_norm,"final_relative_residual":actual_relative,"history":history,"influence_sign":"positive","g_dot_solution":g_dot_solution}


def self_kl_loss(reference_logits:torch.Tensor,current_logits:torch.Tensor,valid_mask:torch.Tensor)->torch.Tensor:
    reference=reference_logits.detach();prob=F.softmax(reference,dim=-1);token=(prob*(F.log_softmax(reference,dim=-1)-F.log_softmax(current_logits,dim=-1))).sum(-1);mask=valid_mask.to(token.dtype)
    if mask.shape!=token.shape or float(mask.sum())<=0:raise ValueError("invalid KL token mask")
    return (token*mask).sum()/mask.sum()


def hessian_vector_product(loss:torch.Tensor,parameters:Sequence[torch.Tensor],vector:torch.Tensor)->torch.Tensor:
    first=torch.autograd.grad(loss,parameters,create_graph=True,retain_graph=True,allow_unused=False);pieces=split_vector(vector,parameters);dot=sum((gradient*piece).sum() for gradient,piece in zip(first,pieces));second=torch.autograd.grad(dot,parameters,create_graph=False,retain_graph=False,allow_unused=False)
    if any(parameter.grad is not None for parameter in parameters):raise RuntimeError("autograd.grad populated .grad")
    result=flatten_tensors(second)
    if not torch.isfinite(result).all():raise FloatingPointError("nonfinite HVP")
    return result


def project_update_space(delta:torch.Tensor,constraints:Sequence[torch.Tensor],*,relative_tolerance:float=1e-10,normalized_tolerance:float=1e-8,formal_dtype:torch.dtype=torch.float32,base:torch.Tensor|None=None)->dict[str,Any]:
    d=delta.detach().to(torch.float64).cpu();columns=[value.detach().to(torch.float64).cpu() for value in constraints if float(torch.linalg.vector_norm(value))>0]
    if columns:
        matrix=torch.stack(columns,dim=1);u,s,_=torch.linalg.svd(matrix,full_matrices=False);threshold=relative_tolerance*float(s.max()) if len(s) else 0.0;rank=int(torch.sum(s>threshold));basis=u[:,:rank];safe=d-basis@(basis.T@d)
    else:s=torch.empty(0,dtype=torch.float64);rank=0;safe=d.clone()
    if base is None:actual=safe.to(formal_dtype).to(torch.float64)
    else:
        original=base.detach().to(torch.float64).cpu();actual=(original+safe).to(formal_dtype).to(torch.float64)-original.to(formal_dtype).to(torch.float64)
    dots=[]
    for constraint in constraints:
        c=constraint.to(torch.float64).cpu();den=max(float(torch.linalg.vector_norm(c))*max(float(torch.linalg.vector_norm(actual)),1e-300),1e-300);dots.append({"dot":float(torch.dot(c,actual)),"normalized_dot":float(torch.dot(c,actual))/den})
    passed=all(item["normalized_dot"]<=normalized_tolerance for item in dots) and torch.isfinite(actual).all()
    return {"actual":actual,"rank":rank,"singular_values":[float(value) for value in s],"residual_norm":float(torch.linalg.vector_norm(actual-safe)),"raw_norm":float(torch.linalg.vector_norm(d)),"safe_norm":float(torch.linalg.vector_norm(actual)),"safe_raw_ratio":float(torch.linalg.vector_norm(actual))/max(float(torch.linalg.vector_norm(d)),1e-300),"constraint_dots":dots,"dtype_roundtrip_verified":True,"passed":bool(passed)}


def candidate_content_hash(anchor:str,direction:str,scale:float,actual_delta:torch.Tensor)->str:
    digest=hashlib.sha256();digest.update(json.dumps({"anchor":anchor,"direction":direction,"scale":scale},sort_keys=True,separators=(",",":")).encode());digest.update(actual_delta.detach().cpu().contiguous().numpy().tobytes());return digest.hexdigest()


def freeze_candidates(candidates:list[dict[str,Any]],*,retrain_loaded:bool)->dict[str,Any]:
    if retrain_loaded:raise RuntimeError("Retrain loaded before candidate freeze")
    identities=[item["candidate_id"] for item in candidates]
    if len(identities)!=len(set(identities)):raise ValueError("duplicate candidate")
    scalar=[{key:value for key,value in item.items() if key not in {"actual_delta"}} for item in candidates]
    if any(_nested_keys(item)&FORBIDDEN_PERSISTED_KEYS for item in scalar):raise ValueError("vector leaked into candidate artifact")
    return {"candidate_order":identities,"candidates":scalar,"candidate_registry_sha256":canonical_hash(scalar),"retrain_not_loaded_during_selection":True,"frozen":True}


def select_primary(candidates:list[dict[str,Any]],*,retrain_loaded:bool)->dict[str,Any]:
    if retrain_loaded:raise RuntimeError("Retrain cannot select candidate")
    ordered=[item for scale in (1.0,.5,.25) for item in candidates if item["anchor"]=="original" and item["direction"]=="retain_safe_influence" and item["scale"]==scale]
    if len(ordered)!=3:raise ValueError("primary candidate set incomplete")
    selected=next((item for item in ordered if item["directional_gate_pass"] is True and item["utility_pass"] is True),None)
    return {"selected_primary_candidate_id":None if selected is None else selected["candidate_id"],"selection_reason":"first_preregistered_scale_passing_retain_directional_and_utility" if selected else "no_primary_scale_passed","retrain_used_for_selection":False}


def validate_utility_gate(value:dict[str,Any])->bool:
    required={"fixed_baseline_step","baseline_metrics","candidate_metrics","damage","utility_checks","utility_pass"}
    if not required<=set(value) or value["fixed_baseline_step"]!=812 or type(value["utility_pass"]) is not bool or not isinstance(value["utility_checks"],dict) or not value["utility_checks"] or any(type(item) is not bool for item in value["utility_checks"].values()) or not _all_finite(value):raise ValueError("utility schema invalid")
    return value["utility_pass"]


def classify_if(*,valid:bool,primary_scientific:bool,retain_safety:bool,utility_pass:bool,efficiency_status:str,efficiency_ratio:float|None,original_pass:bool,step816_pass:bool,reliable_reverse:bool=False)->dict[str,Any]:
    anchor_conflict=bool(step816_pass and not original_pass)
    if not valid:return {"category":"IF-D","next_action":"stop_invalid_or_conflicted","anchor_conflict":anchor_conflict}
    scientific=bool(primary_scientific and original_pass and not reliable_reverse)
    if not scientific or not retain_safety or not utility_pass:return {"category":"IF-C","next_action":"stop_influence_training_and_design_clean_adapter_retraining_plan_b","anchor_conflict":anchor_conflict}
    if efficiency_status!="available" or efficiency_ratio is None or efficiency_ratio>.5:return {"category":"IF-B","next_action":"implement_low_rank_retain_curvature_cache_before_training","anchor_conflict":anchor_conflict}
    return {"category":"IF-A","next_action":"implement_one_step_reversible_curvature_influence_update","anchor_conflict":anchor_conflict}


def _unit_binding(pre:dict[str,Any],contract_sha:str,unit_id:str,kind:str,index:int)->dict[str,Any]:
    return {"schema":UNIT_SCHEMA,"unit_id":unit_id,"kind":kind,"index":index,"config_sha256":pre["config_sha256"],"contract_sha256":contract_sha,"predecessor_sha256":pre["rd_predecessor"]["sha256"],"git_commit":pre["git"]["git_commit"],"implementation_sha256":pre["implementation"]["canonical_sha256"],"checkpoint_sha256":pre["pilot"]["state_sha256"],"data_sha256":pre["data_lineage"]["data"]["overall_validation"]["sha256"],"test_accessed":False}


def publish_unit(path:Path,value:dict[str,Any],binding:dict[str,Any],*,interrupt:Callable[[],None]|None=None)->None:
    if path.exists():raise FileExistsError(path)
    if not _all_finite(value) or _nested_keys(value)&FORBIDDEN_PERSISTED_KEYS:raise ValueError("unsafe/nonfinite unit")
    stage=path.parent/f".{path.name}.{uuid.uuid4().hex[:10]}.stage";stage.mkdir(parents=True)
    try:
        atomic_json(stage/"unit.json",value)
        if interrupt:interrupt()
        manifest={**binding,"unit_sha256":sha256_file(stage/"unit.json"),"published_atomically":True,"optimizer_constructed":False,"optimizer_steps_committed":0,"test_accessed":False};atomic_json(stage/"manifest.json",manifest);atomic_text(stage/"COMPLETED",UNIT_MARKER+"\n");path.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,path)
    except BaseException:
        if stage.exists():
            for item in stage.iterdir():item.unlink()
            stage.rmdir()
        raise


def validate_unit(path:Path,binding:dict[str,Any]|None=None)->dict[str,Any]:
    if not path.is_dir() or {item.name for item in path.iterdir()}!={"unit.json","manifest.json","COMPLETED"}:raise ValueError("unit inventory invalid")
    value=_read_json(path/"unit.json");manifest=_read_json(path/"manifest.json")
    if (path/"COMPLETED").read_text(encoding="utf-8")!=UNIT_MARKER+"\n" or manifest.get("unit_sha256")!=sha256_file(path/"unit.json") or manifest.get("published_atomically") is not True or manifest.get("optimizer_constructed") is not False or manifest.get("optimizer_steps_committed")!=0 or manifest.get("test_accessed") is not False or (binding and any(manifest.get(key)!=item for key,item in binding.items())) or not _all_finite(value) or _nested_keys(value)&FORBIDDEN_PERSISTED_KEYS:raise ValueError("unit validation failed")
    return {"value":value,"manifest":manifest}


def validate_resume(run_dir:Path,pre:dict[str,Any])->dict[str,Any]:
    if not run_dir.is_dir() or (run_dir/"COMPLETED").exists() or any(item.name.endswith(".stage") or item.name.startswith(".") and ".stage" in item.name for item in run_dir.rglob("*")):raise ValueError("Resume requires clean nonterminal run")
    state=_read_json(run_dir/"run_state.json");contract=_read_json(run_dir/"contract.json")
    if state.get("status")!="INTERRUPTED" or state.get("optimizer_steps_committed")!=0 or state.get("test_accessed") is not False or contract.get("git",{}).get("git_commit")!=pre["git"]["git_commit"] or contract.get("implementation",{}).get("canonical_sha256")!=pre["implementation"]["canonical_sha256"] or contract.get("config_sha256")!=pre["config_sha256"]:raise ValueError("Resume provenance/status mismatch")
    return {"state":state,"contract":contract}


def _toy_psd_hvp(matrix:torch.Tensor)->Callable[[torch.Tensor],torch.Tensor]:return lambda vector:matrix.to(torch.float64)@vector.to(torch.float64)


def synthetic_run(root:Path,config_path:Path,run_name:str,mode:str)->dict[str,Any]:
    config=load_audit_config(config_path,root);branch="synthetic_runs" if mode=="SyntheticDryRun" else "dry_runs";final=_resolve(root,Path(config["output_root"])/branch/_safe_name(run_name),output=True)
    if final.exists():raise FileExistsError(final)
    matrix=torch.tensor([[4.,1.],[1.,3.]],dtype=torch.float64);g=torch.tensor([1.,2.],dtype=torch.float64);cg=conjugate_gradient(_toy_psd_hvp(matrix),g,damping=.01,relative_tolerance=1e-12,absolute_tolerance=1e-12,max_iterations=5);expected=torch.linalg.solve(matrix+.01*torch.eye(2,dtype=torch.float64),g);small_cg=conjugate_gradient(lambda value:.01*value,torch.tensor([1e-6,-1e-6],dtype=torch.float64),damping=0,relative_tolerance=1e-12,absolute_tolerance=1e-20,max_iterations=2,pap_tolerance=1e-12)
    if not torch.allclose(cg["solution"],expected,atol=1e-10,rtol=1e-10):raise RuntimeError("synthetic CG mismatch")
    projection=project_update_space(torch.tensor([1.,2.,3.]),[torch.tensor([1.,0.,0.]),torch.tensor([0.,1.,0.])]);candidates=[]
    for anchor in ("original","step816"):
        for direction in ("raw_influence","retain_safe_influence"):
            for scale in (1.0,.5,.25):
                delta=cg["solution"]*scale;candidate_id=f"{anchor}:{direction}:{scale}";candidates.append({"candidate_id":candidate_id,"anchor":anchor,"direction":direction,"scale":scale,"candidate_content_sha256":candidate_content_hash(anchor,direction,scale,delta),"directional_gate_pass":direction=="retain_safe_influence","utility_pass":scale<=.5,"actual_delta":delta})
    frozen=freeze_candidates(candidates,retrain_loaded=False);selection=select_primary(candidates,retrain_loaded=False);examples={"IF-A":classify_if(valid=True,primary_scientific=True,retain_safety=True,utility_pass=True,efficiency_status="available",efficiency_ratio=.25,original_pass=True,step816_pass=True),"IF-B":classify_if(valid=True,primary_scientific=True,retain_safety=True,utility_pass=True,efficiency_status="unavailable",efficiency_ratio=None,original_pass=True,step816_pass=True),"IF-C":classify_if(valid=True,primary_scientific=False,retain_safety=True,utility_pass=True,efficiency_status="unavailable",efficiency_ratio=None,original_pass=False,step816_pass=True),"IF-D":classify_if(valid=False,primary_scientific=False,retain_safety=False,utility_pass=False,efficiency_status="unavailable",efficiency_ratio=None,original_pass=False,step816_pass=False)}
    final.mkdir(parents=True);contract={"schema":SCHEMA,"mode":mode,"run_name":run_name,"synthetic":True,"test_accessed":False};atomic_json(final/"contract.json",contract);unit_value={"schema":UNIT_SCHEMA,"kind":"synthetic_numerics","cg":{key:value for key,value in cg.items() if key!="solution"},"small_positive_pAp_regression":{key:value for key,value in small_cg.items() if key!="solution"},"projection":{key:value for key,value in projection.items() if key!="actual"},"candidate_registry":frozen,"selection":selection,"classification_examples":examples,"model_loaded":False,"retrain_loaded":False,"optimizer_constructed":False,"optimizer_steps_committed":0,"test_accessed":False};binding={"schema":UNIT_SCHEMA,"unit_id":"synthetic_numerics","kind":"synthetic","index":0,"contract_sha256":sha256_file(final/"contract.json"),"test_accessed":False};publish_unit(final/"units/synthetic_numerics",unit_value,binding);validate_unit(final/"units/synthetic_numerics",binding);result={"schema":SCHEMA,"mode":mode,"cg_exact":True,"small_positive_pAp_accepted":small_cg["history"][0]["pAp"]<1e-12 and small_cg["history"][0]["normalized_pAp"]>0,"positive_influence_sign":cg["influence_sign"]=="positive","projection_pass":projection["passed"],"candidate_registry_frozen":True,"selected_primary_candidate_id":selection["selected_primary_candidate_id"],"classification_examples":examples,"model_loaded":False,"retrain_loaded":False,"optimizer_constructed":False,"optimizer_steps_committed":0,"test_accessed":False};atomic_json(final/"result.json",result);atomic_text(final/"COMPLETED","T5_LORA_INFLUENCE_SYNTHETIC_COMPLETED\n");return {**result,"run_dir":str(final)}


def build_contract(pre:dict[str,Any],run_name:str)->dict[str,Any]:
    return {"schema":SCHEMA,"run_name":run_name,"git":pre["git"],"implementation":pre["implementation"],"config_sha256":pre["config_sha256"],"rd_predecessor":pre["rd_predecessor"],"pilot":pre["pilot"],"models":pre["models"],"tokenizer":pre["tokenizer"],"data_lineage":pre["data_lineage"],"sample_order_sha256":pre["sample_order_sha256"],"development_selector":pre["development_selector"],"original_anchor":pre["original_anchor"],"step816_anchor":pre["step816_anchor"],"panels":pre["panels"],"cg":pre["cg"],"curvature":pre["curvature"],"projection":pre["projection"],"candidates":pre["candidates"],"fixed_utility_baseline":pre["fixed_utility_baseline"],"efficiency":pre["efficiency"],"optimizer_constructed":False,"optimizer_steps_committed":0,"test_accessed":False}


def _select_panel(indices:list[int],user_map:dict[int,int],users:int,maximum:int)->list[int]:
    selected_users=set(sorted({user_map[index] for index in indices})[:users]);selected=[index for index in indices if user_map[index] in selected_users][:maximum]
    if len(selected)!=maximum or len({user_map[index] for index in selected})<2:raise ValueError("authoritative grouped panel unavailable")
    return selected


def _trainable(model:torch.nn.Module)->tuple[list[str],list[torch.Tensor]]:
    named=[(name,value) for name,value in model.named_parameters() if value.requires_grad];names=[name for name,_ in named];parameters=[value for _,value in named]
    if len(parameters)!=144 or sum(value.numel() for value in parameters)!=1769472 or not all((".q." in name or ".v." in name) and "lora_" in name for name in names):raise ValueError("runtime is not exact Q/V LoRA coordinate")
    return names,parameters


def _load_anchor(root:Path,config:dict[str,Any],base:dict[str,Any],anchor:str,device:torch.device)->dict[str,Any]:
    if anchor=="step816":
        runtime,_=_load_pilot_runtime(root,load_pilot_config(_resolve(root,config["pilot_config"]),root),_resolve(root,config["authority"]["step816_checkpoint"]),device);current=runtime["current"];original=runtime["original"];tokenizer=runtime["tokenizer"]
    elif anchor=="original":
        cpu_state=torch.random.get_rng_state();cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        try:
            torch.manual_seed(config["panels"]["seed"])
            if torch.cuda.is_available():torch.cuda.manual_seed_all(config["panels"]["seed"])
            current=build_current_model(Path(base["paths"]["original"]),base["lora"])
            with torch.no_grad():
                for name,value in current.named_parameters():
                    if value.requires_grad and "lora_B" in name:value.zero_()
            original=freeze_teacher(load_legacy_model(Path(base["paths"]["original"])))
            tokenizer=__import__("transformers",fromlist=["T5Tokenizer"]).T5Tokenizer.from_pretrained(base["paths"]["model_dir"])
        finally:
            torch.random.set_rng_state(cpu_state)
            if torch.cuda.is_available():torch.cuda.set_rng_state_all(cuda_state)
        current=current.to(device);original=original.to(device)
    else:raise ValueError(anchor)
    current.eval();original.eval();names,parameters=_trainable(current);adapter=get_peft_model_state_dict(current);metadata=_adapter_metadata(adapter)
    if metadata["name_sha256"]!=config["lora_coordinate"]["parameter_name_sha256"] or metadata["metadata_sha256"]!=config["lora_coordinate"]["metadata_sha256"]:raise ValueError("anchor Q/V mapping incompatible")
    before=tensor_tree_hash(adapter);return {"anchor":anchor,"current":current,"original":original,"tokenizer":tokenizer,"names":names,"parameters":parameters,"before_sha256":before,"device":device}


def _release_runtime(runtime:dict[str,Any])->None:
    runtime.clear()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def _panel_gradient(model:torch.nn.Module,dataset:JsonPromptDataset,indices:list[int],parameters:list[torch.Tensor],device:torch.device,batch_size:int,loss_function:Callable[[dict[str,torch.Tensor]],torch.Tensor])->tuple[torch.Tensor,float,int]:
    accumulation=torch.zeros(sum(value.numel() for value in parameters),dtype=torch.float64);loss_sum=0.0;calls=0
    for start in range(0,len(indices),batch_size):
        selected=indices[start:start+batch_size];batch=move_batch(_batch(dataset,selected),device);loss=loss_function(batch);grads=torch.autograd.grad(loss,parameters,create_graph=False,retain_graph=False,allow_unused=False);weight=len(selected)/len(indices);accumulation+=weight*flatten_tensors(grads);loss_sum+=weight*float(loss.detach().cpu());calls+=1;del batch,loss,grads
    if any(parameter.grad is not None for parameter in parameters):raise RuntimeError("panel autograd populated .grad")
    return accumulation,loss_sum,calls


def _construct_anchor_direction(runtime:dict[str,Any],validation:JsonPromptDataset,forget_panel:list[int],retain_panel:list[int],retain_indices:list[int],config:dict[str,Any],pre:dict[str,Any],device:torch.device,anchor:str)->tuple[dict[str,Any],list[dict[str,Any]]]:
    model,original,parameters=runtime["current"],runtime["original"],runtime["parameters"];batch_size=config["panels"]["batch_size"];base_flat=flatten_tensors([value.detach() for value in parameters]);peak_before=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0;forget_started=time.perf_counter()
    g_f,forget_loss,forget_calls=_panel_gradient(model,validation,forget_panel,parameters,device,batch_size,lambda batch:model(input_ids=batch["input_ids"],labels=batch["target_ids"]).loss);forget_time=time.perf_counter()-forget_started
    g_sup,retain_sup_loss,sup_calls=_panel_gradient(model,validation,retain_panel,parameters,device,batch_size,lambda batch:model(input_ids=batch["input_ids"],labels=batch["target_ids"]).loss)
    def kl_constraint(batch):
        current=model(input_ids=batch["input_ids"],labels=batch["target_ids"]).logits
        with torch.no_grad():reference=original(input_ids=batch["input_ids"],labels=batch["target_ids"]).logits
        return teacher_cross_entropy(reference,current)
    g_kl,retain_kl_loss,kl_calls=_panel_gradient(model,validation,retain_panel,parameters,device,batch_size,kl_constraint)
    hvp_calls={"batches":0};curvatures=[]
    def matvec(vector:torch.Tensor)->torch.Tensor:
        total=torch.zeros_like(vector,dtype=torch.float64)
        for start in range(0,len(retain_panel),batch_size):
            selected=retain_panel[start:start+batch_size];batch=move_batch(_batch(validation,selected),device)
            with torch.no_grad():reference=model(input_ids=batch["input_ids"],labels=batch["target_ids"]).logits.detach()
            current=model(input_ids=batch["input_ids"],labels=batch["target_ids"]).logits;loss=self_kl_loss(reference,current,batch["target_ids"]!=-100);piece=hessian_vector_product(loss,parameters,vector);total+=len(selected)/len(retain_panel)*piece;hvp_calls["batches"]+=1;del batch,reference,current,loss,piece
        curvature=float(torch.dot(vector,total));curvatures.append(curvature);tolerance=config["curvature"]["absolute_negative_tolerance"]+config["curvature"]["relative_negative_tolerance"]*max(float(torch.linalg.vector_norm(vector))*float(torch.linalg.vector_norm(total)),1.0)
        if curvature < -tolerance:raise RuntimeError("significant_negative_curvature")
        return total
    cg_started=time.perf_counter();cg=conjugate_gradient(matvec,g_f,damping=config["cg"]["damping"],relative_tolerance=config["cg"]["relative_residual_tolerance"],absolute_tolerance=config["cg"]["absolute_residual_tolerance"],max_iterations=config["cg"]["max_iterations"],residual_explosion_factor=config["cg"]["residual_explosion_factor"],pap_tolerance=config["cg"]["pap_absolute_tolerance"]);cg_time=time.perf_counter()-cg_started;raw=cg.pop("solution");projection_started=time.perf_counter();projection=project_update_space(raw,[g_kl,g_sup],relative_tolerance=config["projection"]["relative_singular_tolerance"],normalized_tolerance=config["projection"]["normalized_constraint_tolerance"],formal_dtype=torch.float32,base=base_flat);safe=projection.pop("actual");projection["module_retention_ratio"]=_module_retention(runtime["names"],raw,safe);projection_time=time.perf_counter()-projection_started;candidate_rows=[];utility_started=time.perf_counter()
    for direction,vector in (("raw_influence",raw),("retain_safe_influence",safe)):
        for scale in config["candidates"]["scales"]:
            actual=(base_flat+scale*vector).to(torch.float32).to(torch.float64)-base_flat.to(torch.float32).to(torch.float64);candidate_id=f"{anchor}:{direction}:{scale}";dots={"forget":float(torch.dot(g_f,actual)),"retain_kl":float(torch.dot(g_kl,actual)),"retain_supervised":float(torch.dot(g_sup,actual))};norm=float(torch.linalg.vector_norm(actual));normalized={key:value/max(float(torch.linalg.vector_norm(gradient))*max(norm,1e-300),1e-300) for key,value,gradient in (("forget",dots["forget"],g_f),("retain_kl",dots["retain_kl"],g_kl),("retain_supervised",dots["retain_supervised"],g_sup))};directional=math.isfinite(norm) and norm>0 and dots["forget"]>0 and normalized["retain_kl"]<=config["projection"]["normalized_constraint_tolerance"] and normalized["retain_supervised"]<=config["projection"]["normalized_constraint_tolerance"]
            with _temporary_delta(parameters,actual):candidate_metrics=_evaluate_utility({"current":model},validation,retain_indices,device)
            utility=_utility_evidence(pre["fixed_utility_baseline"]["metrics"],candidate_metrics,config);candidate_rows.append({"candidate_id":candidate_id,"anchor":anchor,"direction":direction,"scale":scale,"candidate_content_sha256":candidate_content_hash(anchor,direction,scale,actual),"delta_norm":norm,"first_order":dots,"normalized_first_order":normalized,"directional_gate_pass":bool(directional),"utility":utility,"utility_pass":utility["utility_pass"],"actual_delta":actual})
    utility_time=time.perf_counter()-utility_started
    if runtime["before_sha256"]!=tensor_tree_hash(get_peft_model_state_dict(model)) or any(parameter.grad is not None for parameter in parameters):raise RuntimeError("anchor parameters/grad changed")
    module_norms=_module_norms(runtime["names"],g_f)
    report={"anchor":anchor,"anchor_parameter_sha256":runtime["before_sha256"],"parameter_order_sha256":canonical_hash(runtime["names"]),"forget_loss":forget_loss,"forget_gradient_norm":float(torch.linalg.vector_norm(g_f)),"forget_module_gradient_norms":module_norms,"retain_supervised_loss":retain_sup_loss,"retain_kl_loss":retain_kl_loss,"cg":cg,"curvature":{"kind":"retain_self_kl_fisher_ggn","reference_logits_detached":True,"minimum_observed":min(curvatures),"maximum_observed":max(curvatures),"hvp_batch_calls":hvp_calls["batches"],"significant_negative_curvature":False},"projection":projection,"timing":{"forget_gradient_seconds":forget_time,"retain_curvature_and_hvp_seconds":cg_time,"cg_seconds":cg_time,"projection_seconds":projection_time,"utility_seconds":utility_time},"forward_batch_estimate":forget_calls+sup_calls+2*kl_calls+hvp_calls["batches"]*2,"autograd_grad_calls":forget_calls+sup_calls+kl_calls+hvp_calls["batches"]*2,"peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,"peak_gpu_reserved_bytes":torch.cuda.max_memory_reserved(device) if device.type=="cuda" else 0,"all_finite":True,"parameters_unchanged":True,"parameter_grad_absent":True,"optimizer_constructed":False,"optimizer_steps_committed":0,"test_accessed":False};return report,candidate_rows


def _module_norms(names:list[str],vector:torch.Tensor)->dict[str,float]:
    result={};offset=0
    for name in names:
        # Q/V tensors all have the frozen 12,288 elements.
        size=12288;piece=vector[offset:offset+size];result[name]=float(torch.linalg.vector_norm(piece));offset+=size
    if offset!=vector.numel():raise ValueError("module segmentation mismatch")
    return result


def _module_retention(names:list[str],raw:torch.Tensor,safe:torch.Tensor)->dict[str,float]:
    result={};offset=0
    for name in names:
        size=12288;raw_norm=float(torch.linalg.vector_norm(raw[offset:offset+size]));safe_norm=float(torch.linalg.vector_norm(safe[offset:offset+size]));result[name]=safe_norm/max(raw_norm,1e-300);offset+=size
    if offset!=raw.numel() or safe.numel()!=raw.numel():raise ValueError("module retention segmentation mismatch")
    return result


@contextlib.contextmanager
def _temporary_delta(parameters:list[torch.Tensor],delta:torch.Tensor):
    before=[value.detach().cpu().clone() for value in parameters];pieces=split_vector(delta,parameters)
    try:
        with torch.no_grad():
            for parameter,piece in zip(parameters,pieces):parameter.add_(piece)
        yield
    finally:
        with torch.no_grad():
            for parameter,value in zip(parameters,before):parameter.copy_(value.to(parameter.device))


def _utility_evidence(baseline:dict[str,Any],candidate:dict[str,Any],config:dict[str,Any])->dict[str,Any]:
    damage={"overall_auc":baseline["overall_validation"]["auc"]-candidate["overall_validation"]["auc"],"retain_user_auc":baseline["retain_user_validation"]["auc"]-candidate["retain_user_validation"]["auc"],"overall_log_loss":candidate["overall_validation"]["log_loss"]-baseline["overall_validation"]["log_loss"],"retain_user_log_loss":candidate["retain_user_validation"]["log_loss"]-baseline["retain_user_validation"]["log_loss"]};checks={"finite":_all_finite(candidate) and _all_finite(damage),"overall_auc":damage["overall_auc"]<=config["utility"]["overall_auc_damage_max"],"retain_user_auc":damage["retain_user_auc"]<=config["utility"]["retain_user_auc_damage_max"],"overall_log_loss":damage["overall_log_loss"]<=config["utility"]["overall_log_loss_damage_max"],"retain_user_log_loss":damage["retain_user_log_loss"]<=config["utility"]["retain_user_log_loss_damage_max"],"probability_not_collapsed":candidate["overall_validation"]["probability_std"]>1e-6 and candidate["retain_user_validation"]["probability_std"]>1e-6};value={"fixed_baseline_step":812,"baseline_metrics":baseline,"candidate_metrics":candidate,"damage":damage,"utility_checks":checks,"utility_pass":bool(all(checks.values()))};validate_utility_gate(value);return value


def _cluster_mean_ci(values:list[float],users:list[int],resamples:int,seed:int)->dict[str,Any]:
    array=np.asarray(values,dtype=np.float64);groups=np.asarray(users);unique=sorted(set(users));by={user:np.flatnonzero(groups==user) for user in unique};rng=np.random.default_rng(seed);draws=[]
    for _ in range(resamples):
        sampled=rng.choice(unique,len(unique),replace=True);indices=np.concatenate([by[int(user)] for user in sampled]);draws.append(float(np.mean(array[indices])))
    return {"point":float(np.mean(array)),"ci95_low":float(np.percentile(draws,2.5)),"ci95_high":float(np.percentile(draws,97.5)),"users":len(unique),"samples":len(values),"resamples":resamples,"cluster":"authoritative_user_id"}


def _bernoulli_jsd(left:torch.Tensor,right:torch.Tensor)->torch.Tensor:
    l=torch.stack((1-left,left),-1).clamp_min(1e-12);r=torch.stack((1-right,right),-1).clamp_min(1e-12);m=(l+r)/2;return .5*(l*(l.log()-m.log())).sum(-1)+.5*(r*(r.log()-m.log())).sum(-1)


def _evaluate_against_retrain(anchor:torch.nn.Module,retrain:torch.nn.Module,validation:JsonPromptDataset,indices:list[int],user_map:dict[int,int],active_by_source:dict[int,bool],delta:torch.Tensor,parameters:list[torch.Tensor],config:dict[str,Any],device:torch.device)->dict[str,Any]:
    full_jsd=[];binary_jsd=[];loss_improvement=[];probability_improvement=[];truth_delta=[];candidate_delta=[];users=[];yes=[];active=[];batch_size=config["panels"]["batch_size"]
    with _temporary_delta(parameters,torch.zeros_like(delta)):
        anchor_state=[value.detach().cpu().clone() for value in parameters]
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            selected=indices[start:start+batch_size];batch=move_batch(_batch(validation,selected),device)
            anchor_logits=anchor(input_ids=batch["input_ids"],labels=batch["target_ids"]).logits
            with _temporary_delta(parameters,delta):candidate_logits=anchor(input_ids=batch["input_ids"],labels=batch["target_ids"]).logits
            retrain_logits=retrain(input_ids=batch["input_ids"],labels=batch["target_ids"]).logits;mask=batch["target_ids"]!=-100;pa=F.softmax(anchor_logits,-1);pc=F.softmax(candidate_logits,-1);pr=F.softmax(retrain_logits,-1);m_ar=(pa+pr)/2;m_cr=(pc+pr)/2;jsd_ar=.5*((pa*(pa.clamp_min(1e-12).log()-m_ar.clamp_min(1e-12).log())).sum(-1)+(pr*(pr.clamp_min(1e-12).log()-m_ar.clamp_min(1e-12).log())).sum(-1));jsd_cr=.5*((pc*(pc.clamp_min(1e-12).log()-m_cr.clamp_min(1e-12).log())).sum(-1)+(pr*(pr.clamp_min(1e-12).log()-m_cr.clamp_min(1e-12).log())).sum(-1));per_full=((jsd_ar-jsd_cr)*mask).sum(-1)/mask.sum(-1)
            target=batch["target_ids"][:,0];yes_id=2163;no_id=465;local=(target==yes_id).long();a2=F.softmax(anchor_logits[:,0,[no_id,yes_id]],-1);c2=F.softmax(candidate_logits[:,0,[no_id,yes_id]],-1);r2=F.softmax(retrain_logits[:,0,[no_id,yes_id]],-1);aobs=torch.gather(a2,1,local[:,None]).squeeze(1);cobs=torch.gather(c2,1,local[:,None]).squeeze(1);robs=torch.gather(r2,1,local[:,None]).squeeze(1);per_binary=_bernoulli_jsd(a2[:,1],r2[:,1])-_bernoulli_jsd(c2[:,1],r2[:,1]);la=-aobs.clamp_min(1e-12).log();lc=-cobs.clamp_min(1e-12).log();lr=-robs.clamp_min(1e-12).log();per_loss=(la-lr).abs()-(lc-lr).abs();per_prob=(aobs-robs).abs()-(cobs-robs).abs();full_jsd.extend(per_full.cpu().tolist());binary_jsd.extend(per_binary.cpu().tolist());loss_improvement.extend(per_loss.cpu().tolist());probability_improvement.extend(per_prob.cpu().tolist());truth_delta.extend((robs-aobs).cpu().tolist());candidate_delta.extend((cobs-aobs).cpu().tolist());users.extend(user_map[index] for index in selected);yes.extend(bool(value==yes_id) for value in target.cpu().tolist());active.extend(active_by_source[index] for index in selected);del batch,anchor_logits,candidate_logits,retrain_logits,pa,pc,pr
    if any(not torch.equal(value.detach().cpu(),before) for value,before in zip(parameters,anchor_state)):raise RuntimeError("evaluation changed anchor")
    from src.diagnostics.t5_retrain_direction_separability_audit import binary_metrics,continuous_metrics,direction_label
    truth=[direction_label(value,.001) for value in truth_delta];pred=[direction_label(value,.001) for value in candidate_delta];counts={"toward_retrain":sum(value>.001 for value in probability_improvement),"away_from_retrain":sum(value<-.001 for value in probability_improvement),"equivalent":sum(abs(value)<=.001 for value in probability_improvement)};by_user={}
    for user,value in zip(users,probability_improvement):by_user.setdefault(user,[]).append(value)
    mixed=sum(min(values)<-.001 and max(values)>.001 for values in by_user.values())/len(by_user);groups={}
    for name,mask_values in (("all",[True]*len(users)),("observed_yes",yes),("observed_no",[not value for value in yes]),("active",active),("inactive",[not value for value in active])):
        chosen=[i for i,value in enumerate(mask_values) if value];groups[name]={"full_vocabulary_jsd_improvement":_cluster_mean_ci([full_jsd[i] for i in chosen],[users[i] for i in chosen],config["evaluation"]["bootstrap_resamples"],42),"yes_no_jsd_improvement":_cluster_mean_ci([binary_jsd[i] for i in chosen],[users[i] for i in chosen],config["evaluation"]["bootstrap_resamples"],42),"answer_loss_improvement":_cluster_mean_ci([loss_improvement[i] for i in chosen],[users[i] for i in chosen],config["evaluation"]["bootstrap_resamples"],42),"probability_distance_improvement":_cluster_mean_ci([probability_improvement[i] for i in chosen],[users[i] for i in chosen],config["evaluation"]["bootstrap_resamples"],42)}
    return {"samples":len(indices),"users":len(set(users)),"groups":groups,"direction":{"counts":counts,"binary":binary_metrics(truth,pred,len(truth)),"continuous":continuous_metrics(candidate_delta,truth_delta),"mixed_direction_user_rate":mixed,"worst_user_mean_quantile_05":float(np.percentile([np.mean(value) for value in by_user.values()],5))},"reliable_reverse_direction":groups["observed_yes"]["full_vocabulary_jsd_improvement"]["ci95_high"]<0 or groups["observed_no"]["full_vocabulary_jsd_improvement"]["ci95_high"]<0,"full_vocabulary_evaluated":True,"test_accessed":False}


def _scientific_gate(evaluation:dict[str,Any],config:dict[str,Any])->bool:
    all_group=evaluation["groups"]["all"]
    return bool(all_group["full_vocabulary_jsd_improvement"]["ci95_low"]>0 and all_group["yes_no_jsd_improvement"]["ci95_low"]>0 and all_group["answer_loss_improvement"]["ci95_low"]>=config["evaluation"]["answer_loss_noninferiority_bound"] and not evaluation["reliable_reverse_direction"])


def execute_full(root:Path,config_path:Path,run_name:str,*,resume:bool)->dict[str,Any]:
    """Create the strict two-phase container; real numerical units are fail-closed.

    Full runtime is intentionally dispatched through ``run_real_phase1`` and
    ``run_real_phase2`` so Retrain cannot exist in the Phase-1 call graph.
    """
    pre=preflight(root,config_path);require_clean_git(pre["git"],"influence Resume" if resume else "influence Full");config=load_audit_config(config_path,root);run_dir=_resolve(root,Path(config["output_root"])/"full_runs"/_safe_name(run_name),output=True)
    if resume:validate_resume(run_dir,pre)
    else:
        if run_dir.exists():raise FileExistsError(run_dir)
        run_dir.mkdir(parents=True);atomic_json(run_dir/"contract.json",build_contract(pre,run_name));atomic_json(run_dir/"run_state.json",{"status":"RUNNING","phase":"direction_construction","optimizer_steps_committed":0,"retrain_loaded":False,"test_accessed":False})
    lock=run_dir/"RUN.lock"
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    except FileExistsError as error:raise RuntimeError("RunName locked") from error
    try:
        phase1=run_real_phase1(root,config,pre,run_dir)
        if phase1["retrain_loaded"] is not False or phase1["candidate_registry_frozen"] is not True:raise RuntimeError("Phase-1 isolation failed")
        phase2=run_real_phase2(root,config,pre,run_dir,phase1)
        state={"status":"COMPLETED","phase":"posthoc_evaluation","phase1_sha256":phase1["unit_manifest_sha256"],"phase2_sha256":phase2["unit_manifest_sha256"],"optimizer_constructed":False,"optimizer_steps_committed":0,"step817_checkpoint_published":False,"test_accessed":False};atomic_json(run_dir/"run_state.json",state);atomic_json(run_dir/"full_manifest.json",{"schema":SCHEMA,"contract_sha256":sha256_file(run_dir/"contract.json"),"run_state_sha256":sha256_file(run_dir/"run_state.json"),"phase1_manifest_sha256":phase1["unit_manifest_sha256"],"phase2_manifest_sha256":phase2["unit_manifest_sha256"],"published_atomically":True,"optimizer_steps_committed":0,"test_accessed":False});atomic_text(run_dir/"COMPLETED",TERMINAL_MARKER+"\n");return {**state,"run_dir":str(run_dir)}
    except BaseException:
        if run_dir.exists() and not (run_dir/"COMPLETED").exists():atomic_json(run_dir/"run_state.json",{"status":"INTERRUPTED","optimizer_steps_committed":0,"step817_checkpoint_published":False,"test_accessed":False})
        raise
    finally:
        if lock.exists():lock.unlink()


def run_real_phase1(root:Path,config:dict[str,Any],pre:dict[str,Any],run_dir:Path)->dict[str,Any]:
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");base=load_config(_resolve(root,config["base_config"]),root);validation=None;anchor_runtime={};all_candidates=[];anchor_reports={};started=time.perf_counter();retrain_loaded=False
    lineage,indices,_=_data_lineage(root,base,_resolve(root,config["protocol_root"]));user_map=_load_user_map(root,pre);forget_panel=_select_panel(indices["forget_user_validation"],user_map,config["panels"]["forget_users"],config["panels"]["forget_samples_max"]);retain_panel=_select_panel(indices["retain_user_validation"],user_map,config["panels"]["retain_users"],config["panels"]["retain_samples_max"])
    try:
        for anchor in config["candidates"]["anchors"]:
            runtime=_load_anchor(root,config,base,anchor,device);anchor_runtime[anchor]=runtime
            if validation is None:validation=JsonPromptDataset(Path(base["paths"]["validation"]),runtime["tokenizer"])
            report,candidates=_construct_anchor_direction(runtime,validation,forget_panel,retain_panel,indices["retain_user_validation"],config,pre,device,anchor)
            anchor_reports[anchor]=report;all_candidates.extend(candidates)
        frozen=freeze_candidates(all_candidates,retrain_loaded=retrain_loaded);selection=select_primary(all_candidates,retrain_loaded=retrain_loaded)
        value={"schema":UNIT_SCHEMA,"phase":1,"kind":"direction_construction","anchors":anchor_reports,"candidate_registry":frozen,"selection":selection,"panel":{"forget":{"samples":len(forget_panel),"users":len({user_map[i] for i in forget_panel}),"sample_order_sha256":canonical_hash(forget_panel),"user_order_sha256":canonical_hash([user_map[i] for i in forget_panel]),"batch_order_sha256":canonical_hash([forget_panel[i:i+config['panels']['batch_size']] for i in range(0,len(forget_panel),config['panels']['batch_size'])])},"retain":{"samples":len(retain_panel),"users":len({user_map[i] for i in retain_panel}),"sample_order_sha256":canonical_hash(retain_panel),"user_order_sha256":canonical_hash([user_map[i] for i in retain_panel]),"batch_order_sha256":canonical_hash([retain_panel[i:i+config['panels']['batch_size']] for i in range(0,len(retain_panel),config['panels']['batch_size'])])}},"direction_construction_wall_time_seconds":time.perf_counter()-started,"retrain_loaded":False,"retrain_not_loaded_during_selection":True,"candidate_registry_frozen":True,"model_parameters_modified":False,"optimizer_constructed":False,"optimizer_steps_committed":0,"step817_checkpoint_published":False,"vectors_persisted":False,"test_accessed":False}
        unit=run_dir/"units/phase1_direction_construction";binding=_unit_binding(pre,sha256_file(run_dir/"contract.json"),"phase1_direction_construction","direction_construction",0)
        if unit.exists():
            existing=validate_unit(unit,binding)["value"]
            if existing["candidate_registry"]["candidate_registry_sha256"]!=frozen["candidate_registry_sha256"]:raise ValueError("recomputed Phase-1 candidate SHA mismatch")
        else:publish_unit(unit,value,binding)
        return {"retrain_loaded":False,"candidate_registry_frozen":True,"unit_manifest_sha256":sha256_file(unit/"manifest.json"),"candidates":all_candidates,"selection":selection,"anchor_runtime":anchor_runtime,"validation":validation,"forget_indices":indices["forget_user_validation"],"retain_indices":indices["retain_user_validation"],"user_map":user_map}
    except BaseException:
        for runtime in anchor_runtime.values():_release_runtime(runtime)
        raise


def run_real_phase2(root:Path,config:dict[str,Any],pre:dict[str,Any],run_dir:Path,phase1:dict[str,Any])->dict[str,Any]:
    completed=run_dir/"units/phase2_posthoc_evaluation";completed_binding=_unit_binding(pre,sha256_file(run_dir/"contract.json"),"phase2_posthoc_evaluation","posthoc_evaluation",1)
    if completed.exists():
        validate_unit(completed,completed_binding)
        for runtime in phase1["anchor_runtime"].values():_release_runtime(runtime)
        return {"unit_manifest_sha256":sha256_file(completed/"manifest.json")}
    phase1_unit=run_dir/"units/phase1_direction_construction";phase1_value=validate_unit(phase1_unit)["value"]
    if phase1_value["candidate_registry"]["candidate_registry_sha256"]!=freeze_candidates(phase1["candidates"],retrain_loaded=False)["candidate_registry_sha256"] or phase1_value["retrain_loaded"] is not False:raise ValueError("Phase-1 freeze invalid")
    device=next(iter(phase1["anchor_runtime"].values()))["device"];base=load_config(_resolve(root,config["base_config"]),root);selector_rows=[json.loads(line) for line in (_resolve(root,config["authority"]["development_selector"])/"rows.jsonl").read_text(encoding="utf-8").splitlines()];active_by_source={int(row["source_index"]):bool(row["active"]) for row in selector_rows};retrain_loaded_at=time.time();retrain=freeze_teacher(load_legacy_model(Path(base["paths"]["retrain_reference"]))).to(device);evaluations={};started=time.perf_counter()
    try:
        for candidate in phase1["candidates"]:
            runtime=phase1["anchor_runtime"][candidate["anchor"]];evaluations[candidate["candidate_id"]]=_evaluate_against_retrain(runtime["current"],retrain,phase1["validation"],phase1["forget_indices"],phase1["user_map"],active_by_source,candidate["actual_delta"],runtime["parameters"],config,device)
        selected=phase1["selection"]["selected_primary_candidate_id"];selected_candidate=next((item for item in phase1["candidates"] if item["candidate_id"]==selected),None);selected_eval=evaluations.get(selected) if selected else None
        original_pass=bool(selected_eval and _scientific_gate(selected_eval,config));step816_pass=any(item["anchor"]=="step816" and item["direction"]=="retain_safe_influence" and item["directional_gate_pass"] and item["utility_pass"] and _scientific_gate(evaluations[item["candidate_id"]],config) for item in phase1["candidates"]);reliable_reverse=bool(selected_eval and selected_eval["reliable_reverse_direction"]);retain_pass=bool(selected_candidate and selected_candidate["directional_gate_pass"]);utility_pass=bool(selected_candidate and selected_candidate["utility_pass"]);scientific=bool(original_pass and not reliable_reverse)
        value={"schema":UNIT_SCHEMA,"phase":2,"kind":"posthoc_evaluation","candidate_registry_sha256":phase1_value["candidate_registry"]["candidate_registry_sha256"],"selected_primary_candidate_id":selected,"evaluations":evaluations,"old_forced_teacher_authority":{"analysis_sha256":pre["rd_predecessor"]["sha256"],"category":pre["rd_predecessor"]["category"],"yes_no_reliable_conflict":pre["rd_predecessor"]["yes_no_reliable_conflict"],"rerun":False},"valid":True,"scientific_pass":scientific,"retain_safety_pass":retain_pass,"utility_pass":utility_pass,"original_pass":original_pass,"step816_pass":step816_pass,"anchor_conflict":bool(step816_pass and not original_pass),"reliable_reverse_direction":reliable_reverse,"efficiency_status":"unavailable","efficiency_ratio":None,"retrain_loaded_after_candidate_freeze":True,"retrain_loaded_timestamp":retrain_loaded_at,"retrain_used_for_selection":False,"posthoc_evaluation_wall_time_seconds":time.perf_counter()-started,"optimizer_constructed":False,"optimizer_steps_committed":0,"step817_checkpoint_published":False,"vectors_persisted":False,"test_accessed":False}
        unit=run_dir/"units/phase2_posthoc_evaluation";binding=_unit_binding(pre,sha256_file(run_dir/"contract.json"),"phase2_posthoc_evaluation","posthoc_evaluation",1)
        if unit.exists():validate_unit(unit,binding)
        else:publish_unit(unit,value,binding)
        return {"unit_manifest_sha256":sha256_file(unit/"manifest.json")}
    finally:
        del retrain
        for runtime in phase1["anchor_runtime"].values():_release_runtime(runtime)
        if torch.cuda.is_available():torch.cuda.empty_cache()


def verify_full(root:Path,config:dict[str,Any],pre:dict[str,Any],run_name:str)->dict[str,Any]:
    path=_resolve(root,Path(config["output_root"])/"full_runs"/_safe_name(run_name),output=True)
    if not path.is_dir() or {item.name for item in path.iterdir()}!={"contract.json","run_state.json","full_manifest.json","units","COMPLETED"} or (path/"COMPLETED").read_text(encoding="utf-8")!=TERMINAL_MARKER+"\n":raise ValueError("Analyze refuses incomplete Full")
    if _read_json(path/"contract.json")!=build_contract(pre,run_name):raise ValueError("Full contract mismatch")
    state=_read_json(path/"run_state.json");manifest=_read_json(path/"full_manifest.json")
    units=path/"units"
    if not units.is_dir() or {item.name for item in units.iterdir()}!={"phase1_direction_construction","phase2_posthoc_evaluation"}:raise ValueError("Full unit inventory invalid")
    contract_sha=sha256_file(path/"contract.json");phase1=validate_unit(units/"phase1_direction_construction",_unit_binding(pre,contract_sha,"phase1_direction_construction","direction_construction",0));phase2=validate_unit(units/"phase2_posthoc_evaluation",_unit_binding(pre,contract_sha,"phase2_posthoc_evaluation","posthoc_evaluation",1))
    if state.get("status")!="COMPLETED" or state.get("optimizer_steps_committed")!=0 or state.get("test_accessed") is not False or manifest.get("contract_sha256")!=contract_sha or manifest.get("run_state_sha256")!=sha256_file(path/"run_state.json") or manifest.get("phase1_manifest_sha256")!=sha256_file(units/"phase1_direction_construction/manifest.json") or manifest.get("phase2_manifest_sha256")!=sha256_file(units/"phase2_posthoc_evaluation/manifest.json") or manifest.get("test_accessed") is not False or phase1["value"].get("retrain_loaded") is not False or phase2["value"].get("retrain_used_for_selection") is not False:raise ValueError("Full terminal evidence invalid")
    return {"path":path,"state":state,"manifest":manifest}


def analyze(root:Path,config_path:Path,run_name:str)->dict[str,Any]:
    pre=preflight(root,config_path);require_clean_git(pre["git"],"influence Analyze");config=load_audit_config(config_path,root);verified=verify_full(root,config,pre,run_name);destination=_resolve(root,Path(config["output_root"])/"analysis_runs"/_safe_name(run_name),output=True)
    if destination.exists():raise FileExistsError(destination)
    # The Phase-2 unit owns frozen scalar gates; no vectors or models are read.
    phase2=validate_unit(verified["path"]/"units/phase2_posthoc_evaluation")["value"];decision=classify_if(valid=phase2["valid"],primary_scientific=phase2["scientific_pass"],retain_safety=phase2["retain_safety_pass"],utility_pass=phase2["utility_pass"],efficiency_status=phase2["efficiency_status"],efficiency_ratio=phase2.get("efficiency_ratio"),original_pass=phase2["original_pass"],step816_pass=phase2["step816_pass"],reliable_reverse=phase2["reliable_reverse_direction"])
    result={"schema":ANALYSIS_SCHEMA,"run_name":run_name,**decision,"scientific_pass":phase2["scientific_pass"],"retain_safety_pass":phase2["retain_safety_pass"],"utility_pass":phase2["utility_pass"],"efficiency_status":phase2["efficiency_status"],"reliable_reverse_direction":phase2["reliable_reverse_direction"],"retrain_used_for_selection":False,"optimizer_constructed":False,"optimizer_steps_committed":0,"step817_checkpoint_published":False,"test_accessed":False};stage=destination.parent/f".{destination.name}.{uuid.uuid4().hex[:10]}.stage";stage.mkdir(parents=True);atomic_json(stage/"analysis.json",result);atomic_json(stage/"manifest.json",{"schema":ANALYSIS_SCHEMA,"analysis_sha256":sha256_file(stage/"analysis.json"),"source_full_manifest_sha256":sha256_file(verified["path"]/"full_manifest.json"),"published_atomically":True,"test_accessed":False});atomic_text(stage/"COMPLETED","T5_LORA_INFLUENCE_ANALYSIS_COMPLETED\n");destination.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,destination);return result


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=("Preflight","SyntheticDryRun","DryRun","Full","Resume","Analyze"),default="Preflight");parser.add_argument("--config",type=Path,required=True);parser.add_argument("--project-root",type=Path,default=Path.cwd());parser.add_argument("--run-name");args=parser.parse_args();root=args.project_root.resolve();config=args.config.resolve()
    if args.mode=="Preflight":result=preflight(root,config)
    else:
        if not args.run_name:parser.error(f"{args.mode} requires --run-name")
        if args.mode in {"SyntheticDryRun","DryRun"}:result=synthetic_run(root,config,args.run_name,args.mode)
        elif args.mode=="Full":result=execute_full(root,config,args.run_name,resume=False)
        elif args.mode=="Resume":result=execute_full(root,config,args.run_name,resume=True)
        else:result=analyze(root,config,args.run_name)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
