from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import T5Tokenizer

from src.diagnostics.t5_full_runner import _batch
from src.diagnostics.t5_lora_influence_feasibility_audit import conjugate_gradient, flatten_tensors, project_update_space, self_kl_loss, split_vector
from src.diagnostics.t5_lora_influence_feasibility_audit_a2 import (_make_curvature_operator, _stream_weight_gradients, analytic_b_gradient, build_fixed_a_basis, collect_qv_modules, estimate_lambda_max, install_fixed_ab_coordinate, select_retain_panels)
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, load_config, load_legacy_model, move_batch
from src.diagnostics.t5_step817_forget_conflict_audit import _data_lineage, tensor_tree_hash
from src.diagnostics.ml1m_development_protocol import reconstruct_authoritative_rows

from .artifacts import atomic_torch_save, complete, publish_manifest, write_contract
from .common import atomic_json, directory_hash, gpu_timer, require_formal_preflight, safe_new_directory, safe_run_name, seed_everything, sha256_file


SCHEMA="paper-if-a2-method-v1"; MARKER="PAPER_IF_A2_METHOD_COMPLETED"
RATIO_CURVATURE_RUNTIME={"curvature_microbatch_size":8,"safety_microbatch_size":8,"reference_strategy":"detached_same_forward","cuda_allocator_fraction":.88,"automatic_fallback":False,"panel_samples_unchanged":True}


def load_method_config(path: Path) -> dict[str,Any]:
    value=yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema")!=SCHEMA or value.get("coordinate")!={"target_modules":["q","v"],"rank":16,"alpha":32,"trainable":"B_only","initial_B":"zero"} or value.get("relative_damping_ratio")!=.01 or value.get("trust_scale")!=.5: raise ValueError("frozen IF-A2 method config changed")
    if value.get("deletion_experiment") and value.get("curvature_runtime",RATIO_CURVATURE_RUNTIME)!=RATIO_CURVATURE_RUNTIME:raise ValueError("ratio IF-A2 curvature runtime changed")
    return value


def curvature_runtime(config:dict[str,Any])->dict[str,Any]:
    return dict(config.get("curvature_runtime",RATIO_CURVATURE_RUNTIME) if config.get("deletion_experiment") else {"curvature_microbatch_size":16,"safety_microbatch_size":16,"reference_strategy":"separate_no_grad_forward","cuda_allocator_fraction":None,"automatic_fallback":False,"panel_samples_unchanged":True})


def configure_cuda_allocator(device:torch.device,runtime:dict[str,Any])->dict[str,Any]:
    fraction=runtime["cuda_allocator_fraction"]
    if device.type=="cuda" and fraction is not None:torch.cuda.set_per_process_memory_fraction(float(fraction),device.index or 0)
    return {**runtime,"allocator_limit_applied":bool(device.type=="cuda" and fraction is not None),"shared_gpu_memory_paging_authorized":False,"mathematical_objective_changed":False,"curvature_panel_changed":False,"floating_reduction_partition_changed":runtime["curvature_microbatch_size"]!=16}


def _method_lineage(root:Path,base:dict[str,Any],config:dict[str,Any])->tuple[dict[str,Any],dict[str,list[int]],dict[str,list[int]]]:
    if not config.get("deletion_experiment"):return _data_lineage(root,base,root/config["protocol_root"])
    protocol=root/config["protocol_root"];manifest=json.loads((protocol/"experiment_manifest.json").read_text(encoding="utf-8"));partition=json.loads((protocol/"validation_partition.json").read_text(encoding="utf-8"));train,validation,_=reconstruct_authoritative_rows(root/"data/ml-1m/raw_data");forget=set(manifest["selected_user_ids"]);forget_train=[row for row in train if row.authoritative_user_id in forget];retain_train=[row for row in train if row.authoritative_user_id not in forget]
    indices={"forget_train":list(range(len(forget_train))),"retain_train":list(range(len(retain_train))),"overall_validation":list(range(len(validation))),"forget_user_validation":partition["forget_indices"],"retain_user_validation":partition["retain_indices"]}
    users={"forget_train":[int(row.authoritative_user_id) for row in forget_train],"retain_train":[int(row.authoritative_user_id) for row in retain_train],"overall_validation":partition["authoritative_user_ids"],"forget_user_validation":[partition["authoritative_user_ids"][i] for i in partition["forget_indices"]],"retain_user_validation":[partition["authoritative_user_ids"][i] for i in partition["retain_indices"]]}
    return {"schema":manifest["schema"],"experiment_contract_sha256":manifest["experiment_contract_sha256"],"test_accessed":False},indices,users


def preflight(root:Path, config_path:Path, *, formal:bool=False)->dict[str,Any]:
    config=load_method_config(config_path); original=root/config["original"]
    if sha256_file(original)!=config["original_sha256"]: raise ValueError("Original SHA mismatch")
    for name in ("forget","retain","development"):
        if sha256_file(root/config[name])!=config[f"{name}_sha256"]:raise ValueError(f"{name} lineage mismatch")
    conflicts=__import__("src.paper_if_a2.common",fromlist=["conflicting_processes"]).conflicting_processes()
    result={"schema":SCHEMA,"mode":"Preflight","original_sha256":sha256_file(original),"candidate_sha256":config["candidate_sha256"],"target_modules":["q","v"],"rank":16,"alpha":32,"relative_damping_ratio":.01,"trust_scale":.5,"curvature_runtime":curvature_runtime(config),"optimizer_steps":0,"model_loaded":False,"test_accessed":False,"conflicts":conflicts}
    if config.get("deletion_experiment") is not None:result["deletion_experiment"] = config["deletion_experiment"]
    if formal: result["formal"]=require_formal_preflight(root)
    return result


def save_adapter(path:Path,names:list[str],parameters:list[torch.Tensor],bases:dict[str,torch.Tensor],candidate_sha:str)->dict[str,Any]:
    path.mkdir(parents=True); state={"B":{name:value.detach().cpu() for name,value in zip(names,parameters)},"A":{name:value.detach().cpu() for name,value in bases.items()}}
    atomic_torch_save(path/"adapter_model.pt",state); atomic_json(path/"adapter_config.json",{"format":"paper-fixed-A-B-LoRA-v1","target_modules":["q","v"],"rank":16,"alpha":32,"candidate_sha256":candidate_sha})
    loaded=torch.load(path/"adapter_model.pt",map_location="cpu",weights_only=True)
    if tensor_tree_hash(state)!=tensor_tree_hash(loaded): raise RuntimeError("adapter reload mismatch")
    return {"adapter_sha256":directory_hash(path),"adapter_bytes":sum(item.stat().st_size for item in path.rglob("*") if item.is_file()),"reload_exact":True}


def apply_candidate_vector(parameters:list[torch.Tensor],vector:torch.Tensor)->None:
    """Apply a flat B-space vector using the verified tensor-aware splitter."""
    pieces=split_vector(vector,parameters)
    with torch.no_grad():
        for parameter,value in zip(parameters,pieces):parameter.add_(value)


def construct_method(root:Path,config:dict[str,Any],run_dir:Path)->dict[str,Any]:
    base=load_config(root/config["base_config"],root); device=torch.device("cuda:0"); runtime=configure_cuda_allocator(device,curvature_runtime(config)); checkpoint=root/config["original"]; before=sha256_file(checkpoint); timings={}; counts={"forward_calls":0,"autograd_grad_calls":0,"hvp_calls":0}; model=None; started=time.perf_counter()
    try:
        print("[method:start]",flush=True); seed_everything(42); torch.cuda.reset_peak_memory_stats()
        with gpu_timer() as box: model=load_legacy_model(checkpoint).to(device).eval()
        timings["model_load_seconds"]=box["seconds"]
        tokenizer=T5Tokenizer.from_pretrained(base["paths"]["model_dir"]); forget=JsonPromptDataset(root/config["forget"],tokenizer); retain=JsonPromptDataset(root/config["retain"],tokenizer); development=JsonPromptDataset(root/config["development"],tokenizer)
        _,indices,users=_method_lineage(root,base,config); a2=yaml.safe_load((root/config["if_a2_config"]).read_text(encoding="utf-8")); panels=select_retain_panels(users["retain_train"],a2["panels"])
        official=list(model.parameters()); official_hash=tensor_tree_hash({str(i):p.detach() for i,p in enumerate(official)}); modules=collect_qv_modules(model); weights=[module.weight for _,module in modules]
        for p in model.parameters():p.requires_grad_(False)
        for p in weights:p.requires_grad_(True)
        with gpu_timer() as box: matrices,report=_stream_weight_gradients(model,forget,indices["forget_train"],weights,device,16)
        timings["forget_gradient_seconds"]=box["seconds"]; counts["forward_calls"]+=report["forward_batches"];counts["autograd_grad_calls"]+=report["autograd_grad_calls"]
        for p in weights:p.requires_grad_(False)
        print("[method:basis]",flush=True); basis_started=time.perf_counter(); bases={}; basis_reports=[]
        for (name,_),matrix in zip(modules,matrices): bases[name],row=build_fixed_a_basis(matrix,rank=16,name=name,seed=42);basis_reports.append(row)
        timings["basis_seconds"]=time.perf_counter()-basis_started
        sample=move_batch(_batch(forget,[0]),device)
        with torch.no_grad(): original_logits=model(input_ids=sample["input_ids"],labels=sample["target_ids"]).logits.detach().cpu()
        names,parameters=install_fixed_ab_coordinate(model,bases,32)
        with torch.no_grad(): zero_logits=model(input_ids=sample["input_ids"],labels=sample["target_ids"]).logits.detach().cpu()
        if not torch.equal(original_logits,zero_logits):raise RuntimeError("B=0 equivalence failed")
        with gpu_timer() as box: direct,direct_report=_stream_weight_gradients(model,forget,indices["forget_train"],parameters,device,16)
        timings["B_gradient_seconds"]=box["seconds"];g_f=flatten_tensors(direct);counts["forward_calls"]+=direct_report["forward_batches"];counts["autograd_grad_calls"]+=direct_report["autograd_grad_calls"]
        analytic=flatten_tensors([analytic_b_gradient(matrix,bases[name],32,16) for (name,_),matrix in zip(modules,matrices)])
        if float(torch.linalg.vector_norm(g_f-analytic)/torch.linalg.vector_norm(analytic))>2e-5:raise RuntimeError("analytic B gradient mismatch")
        print("[method:curvature]",flush=True); counter={"hvp_batches":0};operator=_make_curvature_operator(model,retain,panels["primary"]["indices"],parameters,device,runtime["curvature_microbatch_size"],counter,reuse_detached_current_logits=runtime["reference_strategy"]=="detached_same_forward")
        with gpu_timer() as box: estimate=estimate_lambda_max(operator,g_f.numel(),seed=42,iterations=12,convergence_tolerance=1e-4,numerical_lower_bound=1e-14)
        timings["lambda_max_seconds"]=box["seconds"];damping=.01*estimate["lambda_max_hat"]
        print("[method:cg]",flush=True)
        with gpu_timer() as box: cg=conjugate_gradient(operator,g_f,damping=damping,relative_tolerance=1e-4,absolute_tolerance=1e-10,max_iterations=40,residual_explosion_factor=1000.,pap_tolerance=1e-14)
        timings["cg_seconds"]=box["seconds"];raw=cg.pop("solution");counts["hvp_calls"]=counter["hvp_batches"]
        print("[method:projection]",flush=True);projection_started=time.perf_counter(); safety=panels["safety"]["indices"]
        g_sup_parts,sup_report=_stream_weight_gradients(model,retain,safety,parameters,device,runtime["safety_microbatch_size"]);g_sup=flatten_tensors(g_sup_parts);counts["forward_calls"]+=sup_report["forward_batches"];counts["autograd_grad_calls"]+=sup_report["autograd_grad_calls"]
        total=torch.zeros_like(g_f);tokens=0
        for offset in range(0,len(safety),runtime["safety_microbatch_size"]):
            batch=move_batch(_batch(retain,safety[offset:offset+runtime["safety_microbatch_size"]]),device);mask=batch["target_ids"]!=-100;count=int(mask.sum())
            with torch.no_grad():reference=model(input_ids=batch["input_ids"],labels=batch["target_ids"]).logits.detach()
            current=model(input_ids=batch["input_ids"],labels=batch["target_ids"]).logits;loss=self_kl_loss(reference,current,mask);total+=count*flatten_tensors(torch.autograd.grad(loss,parameters));tokens+=count;del batch,reference,current,loss
        g_kl=total/tokens;base_flat=flatten_tensors([p.detach() for p in parameters]);projection=project_update_space(raw,[g_kl,g_sup],relative_tolerance=1e-10,normalized_tolerance=1e-8,formal_dtype=torch.float32,base=base_flat);safe=projection.pop("actual");actual=(base_flat+.5*safe).float().double()-base_flat.float().double();candidate=tensor_tree_hash({"candidate":actual})
        timings["projection_seconds"]=time.perf_counter()-projection_started
        if config["candidate_sha256"]!="derive_and_record" and candidate!=config["candidate_sha256"]:raise RuntimeError("candidate mismatch")
        apply_candidate_vector(parameters,actual)
        print("[method:save]",flush=True);save_started=time.perf_counter();adapter=save_adapter(run_dir/"adapter",names,parameters,bases,candidate);smoke_indices=list(range(int(config["smoke_samples"])));smoke_batch=move_batch(_batch(development,smoke_indices),device)
        with torch.inference_mode(): smoke=tensor_tree_hash({"logits":model(input_ids=smoke_batch["input_ids"],labels=smoke_batch["target_ids"]).logits.detach().cpu()})
        trainable_parameters=sum(p.numel() for p in parameters)
        if official_hash!=tensor_tree_hash({str(i):p.detach() for i,p in enumerate(official)}) or before!=sha256_file(checkpoint):raise RuntimeError("Original changed")
        del model;model=None;gc.collect();torch.cuda.empty_cache();reloaded=load_legacy_model(checkpoint).to(device).eval();saved=torch.load(run_dir/"adapter/adapter_model.pt",map_location=device,weights_only=True);reload_names,reload_parameters=install_fixed_ab_coordinate(reloaded,saved["A"],32)
        with torch.no_grad():
            for name,parameter in zip(reload_names,reload_parameters):parameter.copy_(saved["B"][name].to(parameter))
            reload_smoke=tensor_tree_hash({"logits":reloaded(input_ids=smoke_batch["input_ids"],labels=smoke_batch["target_ids"]).logits.detach().cpu()})
        del reloaded,saved,smoke_batch;gc.collect();torch.cuda.empty_cache()
        if reload_smoke!=smoke:raise RuntimeError("adapter reload prediction mismatch")
        timings["save_reload_seconds"]=time.perf_counter()-save_started
        timing={**timings,"method_end_to_end_seconds":time.perf_counter()-started}; result={"schema":SCHEMA,"candidate_sha256":candidate,"original_sha256_before":before,"original_sha256_after":sha256_file(checkpoint),"optimizer_steps":0,"trainable_parameters":trainable_parameters,"hvp_calls":counts["hvp_calls"],"forward_calls":counts["forward_calls"],"autograd_grad_calls":counts["autograd_grad_calls"],"curvature_runtime":runtime,"adapter_bytes":adapter["adapter_bytes"],"adapter_reload_exact":adapter["reload_exact"],"smoke_prediction_sha256":smoke,"reload_smoke_prediction_sha256":reload_smoke,"peak_vram_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_vram_bytes":torch.cuda.max_memory_reserved(),"peak_cpu_ram_bytes":__import__("psutil").Process().memory_info().rss,"test_accessed":False}
        return {"result":result,"timing":timing}
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()


def run_full(root:Path,config_path:Path,run_name:str)->dict[str,Any]:
    config=load_method_config(config_path);pre=preflight(root,config_path,formal=True);base=(root/config.get("output_root","outputs/paper_if_a2_v1/method_runs")).resolve();run_dir=(base/safe_run_name(run_name)).resolve()
    if base not in run_dir.parents or run_dir.exists():raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True);write_contract(run_dir,{"schema":SCHEMA,"run_name":run_name,"config_sha256":sha256_file(config_path),**pre})
    try:
        payload=construct_method(root,config,run_dir);atomic_json(run_dir/"method_result.json",payload["result"]);atomic_json(run_dir/"timing.json",payload["timing"]);atomic_json(run_dir/"provenance.json",pre);publish_manifest(run_dir,["adapter","contract.json","method_result.json","timing.json","provenance.json"],{"test_accessed":False});complete(run_dir,MARKER);print("[method:completed]",flush=True);return payload
    except BaseException as error:
        atomic_json(run_dir/"run_state.json",{"status":"INTERRUPTED","reason":f"{type(error).__name__}: {error}","optimizer_steps":0,"automatic_fallback_attempted":False,"test_accessed":False})
        raise


def synthetic(root:Path,run_name:str,config:dict[str,Any]|None=None)->dict[str,Any]:
    base=(root/(config or {}).get("synthetic_root",(config or {}).get("output_root","outputs/paper_if_a2_v1/method_runs"))).resolve();run_dir=(base/safe_run_name(run_name)).resolve()
    if run_dir.exists():raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True);model=torch.nn.Linear(3,2,bias=False);before=model.weight.detach().clone();adapter=run_dir/"adapter";adapter.mkdir();atomic_torch_save(adapter/"adapter_model.pt",{"delta":torch.ones_like(before)*.01});loaded=torch.load(adapter/"adapter_model.pt",weights_only=True);exact=torch.equal(loaded["delta"],torch.ones_like(before)*.01);result={"synthetic":True,"adapter_reload_exact":exact,"original_unchanged":torch.equal(before,model.weight),"optimizer_steps":0,"real_t5_loaded":False,"test_accessed":False};atomic_json(run_dir/"method_result.json",result);complete(run_dir,"PAPER_IF_A2_SYNTHETIC_COMPLETED");return result


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=["Preflight","Full","SyntheticDryRun"],default="Preflight");parser.add_argument("--project-root",type=Path,default=Path.cwd());parser.add_argument("--config",type=Path);parser.add_argument("--run-name",default="");args=parser.parse_args();root=args.project_root.resolve();config=args.config or root/"configs/paper_if_a2_method_v1.yaml"
    loaded=load_method_config(config);result=preflight(root,config) if args.mode=="Preflight" else synthetic(root,args.run_name,loaded) if args.mode=="SyntheticDryRun" else run_full(root,config,args.run_name);print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":main()
