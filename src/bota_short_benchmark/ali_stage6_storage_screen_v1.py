"""Stage 6A: measured storage and parameter-fidelity screen for impact banks."""
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

import torch
import yaml

from src.bota_if.p1_trajectory_transport_audit import StepBudget
from src.bota_short_benchmark import ali_stage2_stronger_signal_v1 as stage2
from src.bota_short_benchmark import ali_stage5_sequential_v1 as stage5
from src.bota_short_benchmark import runner
from src.bota_short_benchmark import t012_behavior_ablation as behavior
from src.bota_short_benchmark.ali_stage1_mechanism_v1 import run_shared_canonical
from src.bota_short_benchmark.protocol import load_config as load_benchmark_config
from src.bota_short_benchmark.protocol import validate_prepared
from src.paper_if_a2.artifacts import atomic_torch_save
from src.paper_if_a2.common import atomic_json, canonical_hash, git_snapshot, safe_run_name, seed_everything, sha256_file

SCHEMA="bota-ali-stage6-storage-screen-v1"
MARKER="BOTA_ALI_STAGE6_STORAGE_SCREEN_V1_COMPLETED"
SUMMARY_MARKER="BOTA_ALI_STAGE6_STORAGE_SUMMARY_V1_COMPLETED"
SEEDS=(41,42,43)
BANK_SIZES=(12,32,64,128,256)
RANKS=(4,8,16,32,64)
TOPQ=(0.0625,0.03125,0.015625)
SELECTED_ARM="FS_empirical_fisher"
RUN_REQUIRED={"COMPLETED","bank.pt","contract.json","manifest.json","metrics.csv","metrics.json","provenance.json","report.md","request_registry.json","run_state.json","timing.json"}


def _json(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict):raise ValueError(f"expected JSON object: {path}")
    return value


def load_config(repo:Path,workspace:Path,path:Path)->dict[str,Any]:
    value=yaml.safe_load(path.read_text(encoding="utf-8"));required={"schema","test_access_policy","parent_amendment","stage1_selection","stage5_audit","benchmark_configs","output_root","protocol","representations","decision","runtime"}
    if not isinstance(value,dict) or set(value)!=required or value["schema"]!=SCHEMA or value["test_access_policy"]!="forbidden":raise ValueError("invalid Stage 6 config")
    for field in ("parent_amendment","stage5_audit"):
        binding=value[field];source=repo/binding["path"]
        if sha256_file(source)!=binding["sha256"]:raise ValueError(f"{field} binding changed")
    parent=value["parent_amendment"]
    if _json(repo/parent["path"]).get("amendment_id")!=parent["amendment_id"]:raise ValueError("Amendment 05 identity changed")
    selection=value["stage1_selection"];selection_path=workspace/selection["path"]
    if sha256_file(selection_path)!=selection["sha256"] or _json(selection_path).get("selection",{}).get("selected_arm")!=SELECTED_ARM:raise ValueError("Stage 1 selection changed")
    expected={"seeds":list(SEEDS),"optimizer_steps_per_chunk":200,"selected_arm":SELECTED_ARM,"bank_sizes":list(BANK_SIZES),"maximum_bank_users":256,"user_chunk_size":32,"registered_users_first":True,"query_scenarios":["L8","L4M4"]}
    if value["protocol"]!=expected:raise ValueError("Stage 6 protocol changed")
    if value["representations"]!={"fp16":True,"int8_per_vector":True,"int8_per_block":{"block_size":4096},"topq_fractions":list(TOPQ),"shared_low_rank_ranks":list(RANKS)}:raise ValueError("Stage 6 representation grid changed")
    for seed in SEEDS:
        binding=value["benchmark_configs"][seed]
        if sha256_file(repo/binding["path"])!=binding["sha256"]:raise ValueError(f"benchmark config changed: {seed}")
    return value


def _enrich_registry(protocol_dir:Path,private:dict[str,Any])->dict[str,Any]:
    public=_json(protocol_dir/"request_registry.json");public_by_id={row["id"]:row for row in public["scenarios"]}
    return {**private,"scenarios":[{**row,"public_members":public_by_id[row["id"]]["public_members"]} for row in private["scenarios"]]}


def _bank_users(registry:dict[str,Any],train_users:Sequence[int])->tuple[list[int],list[str]]:
    registered=[];public_by_user={}
    for scenario in registry["scenarios"]:
        for user,public in zip(scenario["users"],scenario["public_members"]):public_by_user[int(user)]=public["user_hash"]
    registered=sorted(public_by_user,key=lambda user:public_by_user[user]);window_users={int(train_users[index]) for index in registry["order"]};remaining=[user for user in window_users if user not in public_by_user];remaining.sort(key=lambda user:canonical_hash(["ali-stage6-public-user",user]));users=registered+remaining
    if len(registered)!=12 or len(users)<256:raise RuntimeError(f"insufficient nested bank population: registered={len(registered)}, eligible={len(users)}")
    users=users[:256];hashes=[public_by_user.get(user,canonical_hash(["ali-stage6-public-user",user])) for user in users]
    if len(set(users))!=256 or len(set(hashes))!=256:raise RuntimeError("bank user/hash collision")
    return users,hashes


def _stage5_query_bindings(workspace:Path,seed:int)->dict[str,dict[str,Any]]:
    run=workspace/"outputs/ali_stage5_sequential_v1/runs"/f"ali_stage5_sequential_seed{seed}_v1";metrics=_json(run/"metrics.json")["cells"];manifest=_json(run/"manifest.json");result={}
    for scenario in ("L8","L4M4"):
        candidates=[row for row in metrics if row["scenario"]==scenario and row["request_size"]==8]
        hashes={row["union_hash"] for row in candidates}
        if len(hashes)!=1:raise ValueError(f"Stage 5 final union changed: {scenario}")
        union_hash=next(iter(hashes));entry=manifest["endpoint_sha256"][union_hash];result[scenario]={"union_hash":union_hash,"directory_key":entry["directory_key"],"fp32_parameter_residual":candidates[0]["parameter_metrics"]["relative_l2"]}
    return result


def preflight(repo:Path,workspace:Path,config_path:Path,seed:int)->tuple[dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any],list[int],list[str],dict[str,dict[str,Any]]]:
    config=load_config(repo,workspace,config_path)
    if seed not in SEEDS:raise ValueError(seed)
    stage5.analyze(stage5.load_config(repo,workspace,repo/"configs/ali_stage5_sequential_v1.yaml"),workspace,seed,f"ali_stage5_sequential_seed{seed}_v1")
    binding=config["benchmark_configs"][seed];benchmark=load_benchmark_config(repo/binding["path"]);benchmark_name=f"bota_short_i02_seed{seed}_v3";protocol_dir,contract,private=validate_prepared(workspace,benchmark,benchmark_name);registry=_enrich_registry(protocol_dir,private);train_users,_,lineage=stage2._source_users(workspace,benchmark);users,hashes=_bank_users(registry,train_users);queries=_stage5_query_bindings(workspace,seed);original_run,manifest,state=behavior._validate_reference(workspace,benchmark,"Original-Short",f"bota_short_original_seed{seed}_v3",benchmark_name,"L8")
    authority={"schema":SCHEMA,"seed":seed,"benchmark_name":benchmark_name,"benchmark_config_sha256":binding["sha256"],"prepared_registry_sha256":registry["registry_sha256"],"bank_users":256,"registered_prefix_users":12,"bank_user_hashes_sha256":canonical_hash(hashes),"query_bindings":queries,"original_run":str(original_run),"original_adapter_sha256":manifest["artifact"]["sha256"],"source_checkpoint_sha256":sha256_file(workspace/benchmark["source"]["original_checkpoint"]),"lineage":lineage,"development_predictions_accessed":False,"model_loaded":False,"final_test_accessed":False,"test_accessed":False}
    return authority,config,benchmark,registry,users,hashes,queries


def _flatten_map(values:dict[str,torch.Tensor],names:Sequence[str])->torch.Tensor:
    return torch.cat([values[name].detach().float().reshape(-1).cpu() for name in names])


def _adapter_delta(path:Path,original:torch.Tensor,names:Sequence[str])->torch.Tensor:
    state=torch.load(path/"adapter"/"adapter_model.pt",map_location="cpu",weights_only=True);return _flatten_map(state["B"],names)-original


def _encode(name:str,x:torch.Tensor,low_rank:dict[str,torch.Tensor]|None=None,topq:dict[str,torch.Tensor]|None=None)->dict[str,Any]:
    if name=="FP32":return {"kind":name,"values":x}
    if name=="FP16":return {"kind":name,"values":x.half()}
    if name=="INT8_per_vector":
        scale=x.abs().amax(1,keepdim=True).clamp_min(1e-30)/127;return {"kind":name,"q":torch.round(x/scale).clamp(-127,127).to(torch.int8),"scale":scale}
    if name=="INT8_per_block4096":
        if x.shape[1]%4096:raise ValueError("impact dimension is not divisible by block size")
        blocks=x.reshape(x.shape[0],-1,4096);scale=blocks.abs().amax(2,keepdim=True).clamp_min(1e-30)/127;return {"kind":name,"q":torch.round(blocks/scale).clamp(-127,127).to(torch.int8),"scale":scale,"dimension":x.shape[1]}
    if name.startswith("topq_"):
        fraction=float(name.split("_",1)[1]);count=max(1,math.ceil(x.shape[1]*fraction))
        if topq is None:
            _,indices=torch.topk(x.abs(),count,dim=1,sorted=False);values=torch.gather(x,1,indices).half()
        else:
            # Clone slices so each measured artifact owns only its requested
            # fraction rather than the backing storage of the largest Top-q.
            indices=topq["indices"][:,:count].clone();values=topq["values"][:,:count].clone()
        return {"kind":"topq","fraction":fraction,"indices":indices.to(torch.int32),"values":values,"dimension":x.shape[1]}
    if name.startswith("shared_low_rank_r"):
        rank=int(name.rsplit("r",1)[1]);effective=min(rank,x.shape[0]);assert low_rank is not None;return {"kind":"shared_low_rank","requested_rank":rank,"effective_rank":effective,"coefficients":low_rank["u"][:,:effective].clone(),"basis":low_rank["basis"][:effective].clone()}
    raise ValueError(name)


def _compose(encoded:dict[str,Any],rows:Sequence[int])->torch.Tensor:
    kind=encoded["kind"]
    if kind in ("FP32","FP16"):return encoded["values"][list(rows)].float().sum(0)
    if kind=="INT8_per_vector":return (encoded["q"][list(rows)].float()*encoded["scale"][list(rows)]).sum(0)
    if kind=="INT8_per_block4096":return (encoded["q"][list(rows)].float()*encoded["scale"][list(rows)]).sum(0).reshape(-1)[:encoded["dimension"]]
    if kind=="topq":
        result=torch.zeros(encoded["dimension"],dtype=torch.float32)
        for row in rows:result.scatter_add_(0,encoded["indices"][row].long(),encoded["values"][row].float())
        return result
    if kind=="shared_low_rank":return encoded["coefficients"][list(rows)].sum(0)@encoded["basis"]
    raise ValueError(kind)


def _representation_names()->list[str]:
    return ["FP32","FP16","INT8_per_vector","INT8_per_block4096",*[f"topq_{value}" for value in TOPQ],*[f"shared_low_rank_r{rank}" for rank in RANKS]]


def _measure_artifact(stage:Path,name:str,encoded:dict[str,Any])->tuple[int,float]:
    file_stem=safe_run_name(name.replace(".","p"))
    target=stage/"measure"/f"{file_stem}.pt";target.parent.mkdir(parents=True,exist_ok=True);begun=time.perf_counter();atomic_torch_save(target,encoded);seconds=time.perf_counter()-begun;size=target.stat().st_size;target.unlink();return size,seconds


def execute(repo:Path,workspace:Path,config_path:Path,seed:int,run_name:str)->dict[str,Any]:
    authority,config,benchmark,registry,users,hashes,queries=preflight(repo,workspace,config_path,seed);git=git_snapshot(repo)
    if not git["clean"]:raise RuntimeError("formal Stage 6A requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("exactly one CUDA GPU required")
    torch.cuda.set_device(0);device=torch.device("cuda:0");free,total=torch.cuda.mem_get_info(device)
    if free/2**30<config["runtime"]["minimum_free_gib"]:raise RuntimeError("insufficient free GPU memory")
    torch.cuda.set_per_process_memory_fraction(runner.allocator_fraction_for(total/2**30,config["runtime"]["allocator_fraction"],config["runtime"]["hard_peak_reserved_gib"]),device);torch.cuda.reset_peak_memory_stats();destination=workspace/config["output_root"]/"runs"/safe_run_name(run_name)
    if destination.exists():raise FileExistsError(destination)
    stage=destination.parent/".work"/f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage";stage.mkdir(parents=True);source=workspace/benchmark["source"]["original_checkpoint"];source_before=sha256_file(source);original_run=Path(authority["original_run"]);original_state=torch.load(original_run/"scenarios"/"L8"/"adapter"/"adapter_model.pt",map_location="cpu",weights_only=True);old=Path.cwd();os.chdir(workspace);model=None;started=time.perf_counter()
    try:
        bank=None;canonical=None;names=None;bases_cpu=None;chunk_times=[];trace_hashes=[];optimizer_hashes=[];chunk_size=config["protocol"]["user_chunk_size"]
        for start in range(0,len(users),chunk_size):
            chunk=users[start:start+chunk_size];seed_everything(seed);model,current_names,parameters,bases,_,tokenizer,train_data,train_users=runner._model_context(workspace,benchmark,registry,device,original_state["A"]);current_bases={name:value.detach().cpu().clone() for name,value in bases.items()}
            if not stage2._same_tensor_map(current_bases,original_state["A"]):raise RuntimeError("fixed LoRA-A coordinate mismatch")
            begun=time.perf_counter();budget=StepBudget(200);current,states,trace,optimizer_hash,rng_hash=run_shared_canonical(model,train_data,registry["order"],train_users,chunk,parameters,current_names,device,tokenizer.pad_token_id,runner._engine(workspace,benchmark),budget,arms=(SELECTED_ARM,));chunk_times.append(time.perf_counter()-begun)
            if budget.calls!=200 or any(not torch.equal(current[name],original_state["B"][name]) for name in current_names):raise RuntimeError("chunk canonical endpoint mismatch")
            if canonical is None:
                canonical=current;names=list(current_names);bases_cpu=current_bases;dimension=sum(current[name].numel() for name in names);bank=torch.empty((len(users),dimension),dtype=torch.float32)
            elif list(current_names)!=names or any(not torch.equal(current[name],canonical[name]) for name in names):raise RuntimeError("canonical endpoint differs across chunks")
            offset=0
            for module,name in enumerate(names):
                values=states[SELECTED_ARM][module]["theta"].reshape(len(chunk),-1).float();bank[start:start+len(chunk),offset:offset+values.shape[1]]=values;offset+=values.shape[1]
            trace_hashes.append(canonical_hash(trace));optimizer_hashes.append(optimizer_hash);del model,parameters,states,current;model=None;gc.collect();torch.cuda.empty_cache()
        assert bank is not None and canonical is not None and names is not None
        atomic_torch_save(stage/"bank.pt",{"schema":SCHEMA,"seed":seed,"user_hashes":hashes,"impacts":bank,"dimension":bank.shape[1],"selected_arm":SELECTED_ARM})
        stage5_run=workspace/"outputs/ali_stage5_sequential_v1/runs"/f"ali_stage5_sequential_seed{seed}_v1";original_flat=_flatten_map(torch.load(stage5_run/"endpoints/original/adapter/adapter_model.pt",map_location="cpu",weights_only=True)["B"],names);query_data={}
        registered_map={user:index for index,user in enumerate(users)}
        private_by_id={row["id"]:row for row in registry["scenarios"]}
        for scenario,binding in queries.items():
            endpoint=stage5_run/"endpoints"/binding["directory_key"];exact_delta=_adapter_delta(endpoint/"exact_masked",original_flat,names);fp32_delta=_adapter_delta(endpoint/"bota",original_flat,names);row_indices=[registered_map[int(user)] for user in private_by_id[scenario]["users"]];query_data[scenario]={"rows":row_indices,"exact":exact_delta,"stage5_fp32":fp32_delta,"stage5_residual":binding["fp32_parameter_residual"]}
        metric_rows=[];names_grid=_representation_names();serialization_seconds={}
        for bank_size in BANK_SIZES:
            # Clone the prefix so torch.save measures this K-user artifact rather
            # than preserving the backing storage of the full 256-user bank.
            x=bank[:bank_size].clone()
            # The model has already been released.  Use the otherwise idle GPU
            # for the expensive exact Gram decomposition, then return the
            # representation to CPU before measuring serialized bytes.
            x_device=x.to(device)
            gram=x_device@x_device.T
            eigenvalues,u_device=torch.linalg.eigh(gram)
            order=torch.argsort(eigenvalues,descending=True)
            u_device=u_device[:,order[:min(max(RANKS),bank_size)]]
            low_rank={"u":u_device.cpu(),"basis":(u_device.T@x_device).cpu()}
            top_count=max(1,math.ceil(x.shape[1]*max(TOPQ)))
            _,top_indices_device=torch.topk(x_device.abs(),top_count,dim=1,sorted=True)
            topq={"indices":top_indices_device.to(torch.int32).cpu(),"values":torch.gather(x_device,1,top_indices_device).half().cpu()}
            del x_device,u_device,top_indices_device
            encoded_cache={}
            for representation in names_grid:
                begun=time.perf_counter();encoded=_encode(representation,x,low_rank,topq);encode_seconds=time.perf_counter()-begun;artifact_bytes,serialize_seconds=_measure_artifact(stage,f"k{bank_size}_{representation}",encoded);serialization_seconds[f"k{bank_size}_{representation}"]=serialize_seconds;encoded_cache[representation]=(encoded,artifact_bytes,encode_seconds)
            fp32_bytes=encoded_cache["FP32"][1]
            for representation,(encoded,artifact_bytes,encode_seconds) in encoded_cache.items():
                for scenario,query in query_data.items():
                    begun=time.perf_counter();delta=_compose(encoded,query["rows"]);composition_seconds=time.perf_counter()-begun;exact=query["exact"];residual=float(torch.linalg.vector_norm(delta-exact)/torch.linalg.vector_norm(exact).clamp_min(1e-30));fp32_reconstructed=_compose(encoded_cache["FP32"][0],query["rows"]);fp32_residual=float(torch.linalg.vector_norm(fp32_reconstructed-exact)/torch.linalg.vector_norm(exact).clamp_min(1e-30));reproduction=float(torch.linalg.vector_norm(fp32_reconstructed-query["stage5_fp32"])/torch.linalg.vector_norm(query["stage5_fp32"]).clamp_min(1e-30));increase=residual/max(fp32_residual,1e-30)-1
                    metric_rows.append({"seed":seed,"bank_size":bank_size,"representation":representation,"scenario":scenario,"artifact_bytes":artifact_bytes,"bytes_per_user":artifact_bytes/bank_size,"compression_ratio":fp32_bytes/artifact_bytes,"parameter_residual":residual,"fp32_parameter_residual":fp32_residual,"relative_residual_increase":increase,"fp32_stage5_reproduction_relative_l2":reproduction,"encode_seconds":encode_seconds,"serialize_seconds":serialization_seconds[f"k{bank_size}_{representation}"],"composition_seconds":composition_seconds,"finite":all(math.isfinite(value) for value in (residual,fp32_residual,increase,reproduction))})
            del x,gram,eigenvalues,order,low_rank,topq,encoded_cache;gc.collect();torch.cuda.empty_cache()
        if any(not row["finite"] or row["fp32_stage5_reproduction_relative_l2"]>1e-6 for row in metric_rows if row["representation"]=="FP32"):raise RuntimeError("FP32 bank does not reproduce Stage 5")
        public={"schema":SCHEMA,"seed":seed,"bank_sizes":list(BANK_SIZES),"maximum_bank_users":256,"registered_prefix_users":12,"user_hashes":hashes,"selection_uses_predictions_or_labels":False,"raw_user_ids_persisted":False,"registry_sha256":canonical_hash(hashes),"test_accessed":False};atomic_json(stage/"request_registry.json",public);atomic_json(stage/"metrics.json",{"schema":SCHEMA,"seed":seed,"rows":metric_rows,"representations":names_grid,"bank_sizes":list(BANK_SIZES),"screen_only":True,"development_predictions_accessed":False,"final_test_accessed":False,"test_accessed":False})
        with (stage/"metrics.csv").open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=list(metric_rows[0]));writer.writeheader();writer.writerows(metric_rows)
        lines=["# Ali Stage 6A storage screen","",f"Seed {seed}; measured serialized artifacts; no Development inference.","","| K | Representation | Compression | Worst residual increase |","|---:|---|---:|---:|"]
        for bank_size in BANK_SIZES:
            for representation in names_grid:
                selected=[row for row in metric_rows if row["bank_size"]==bank_size and row["representation"]==representation];lines.append(f"| {bank_size} | {representation} | {selected[0]['compression_ratio']:.2f}x | {max(row['relative_residual_increase'] for row in selected):.4f} |")
        (stage/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n");peak=torch.cuda.max_memory_reserved()/2**30
        if peak>config["runtime"]["hard_peak_reserved_gib"] or sha256_file(source)!=source_before:raise RuntimeError("resource or source-integrity guard failed")
        atomic_json(stage/"timing.json",{"chunk_trajectory_seconds":chunk_times,"serialization_seconds":serialization_seconds,"end_to_end_seconds":time.perf_counter()-started});atomic_json(stage/"contract.json",authority);atomic_json(stage/"provenance.json",{"schema":SCHEMA,"git":git,"source_checkpoint_sha256":source_before,"chunk_trace_sha256":trace_hashes,"chunk_optimizer_sha256":optimizer_hashes,"raw_user_ids_persisted":False,"development_predictions_accessed":False,"final_test_loader_built":False,"test_accessed":False});atomic_json(stage/"run_state.json",{"schema":SCHEMA,"status":"COMPLETED","seed":seed,"bank_users":256,"chunks":math.ceil(256/chunk_size),"optimizer_steps_per_chunk":200,"physical_optimizer_step_calls":200*math.ceil(256/chunk_size),"screen_rows":len(metric_rows),"peak_reserved_gib":peak,"test_accessed":False});(stage/"COMPLETED").write_text(MARKER+"\n",encoding="utf-8",newline="\n");files={name:sha256_file(stage/name) for name in ("COMPLETED","bank.pt","contract.json","metrics.csv","metrics.json","provenance.json","report.md","request_registry.json","run_state.json","timing.json")};atomic_json(stage/"manifest.json",{"schema":SCHEMA,"files":files,"published_atomically":True,"test_accessed":False});destination.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,destination);return {"status":"COMPLETED","run_dir":str(destination),"seed":seed,"bank_users":256,"screen_rows":len(metric_rows),"physical_optimizer_step_calls":200*math.ceil(256/chunk_size),"test_accessed":False}
    except Exception as error:
        shutil.rmtree(stage,ignore_errors=True);raise RuntimeError("Stage 6A failed before atomic publication") from error
    finally:
        os.chdir(old)
        if model is not None:del model
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def analyze(config:dict[str,Any],workspace:Path,seed:int,run_name:str)->dict[str,Any]:
    run=workspace/config["output_root"]/"runs"/safe_run_name(run_name)
    if not run.is_dir() or {path.name for path in run.iterdir()}!=RUN_REQUIRED or (run/"COMPLETED").read_text(encoding="utf-8")!=MARKER+"\n":raise ValueError("invalid Stage 6A run")
    manifest=_json(run/"manifest.json")
    for name,digest in manifest["files"].items():
        if sha256_file(run/name)!=digest:raise ValueError(f"Stage 6A artifact changed: {name}")
    metrics=_json(run/"metrics.json");state=_json(run/"run_state.json");expected=len(BANK_SIZES)*len(_representation_names())*2
    if len(metrics.get("rows",[]))!=expected or state.get("physical_optimizer_step_calls")!=1600 or metrics.get("development_predictions_accessed") is not False or any(row.get("test_accessed") is not False for row in (metrics,state)):raise ValueError("Stage 6A completeness invariant failed")
    return {"status":"VALID","run_dir":str(run),"seed":seed,"bank_users":256,"screen_rows":expected,"test_accessed":False}


def summarize(repo:Path,workspace:Path,config_path:Path)->dict[str,Any]:
    config=load_config(repo,workspace,config_path);all_rows=[];runs=[]
    for seed in SEEDS:
        name=f"ali_stage6_storage_screen_seed{seed}_v1";analyze(config,workspace,seed,name);run=workspace/config["output_root"]/"runs"/name;runs.append(run);all_rows.extend(_json(run/"metrics.json")["rows"])
    candidates=[]
    for representation in _representation_names():
        rows=[row for row in all_rows if row["bank_size"]==config["decision"]["candidate_bank_size"] and row["representation"]==representation];minimum_ratio=min(row["compression_ratio"] for row in rows);worst=max(row["relative_residual_increase"] for row in rows);bytes_max=max(row["artifact_bytes"] for row in rows);eligible=minimum_ratio>=config["decision"]["minimum_storage_reduction"] and worst<=config["decision"]["maximum_relative_parameter_residual_increase"];candidates.append({"representation":representation,"minimum_compression_ratio":minimum_ratio,"worst_case_residual_increase":worst,"maximum_artifact_bytes":bytes_max,"eligible":eligible})
    eligible=sorted((row for row in candidates if row["eligible"]),key=lambda row:(row["worst_case_residual_increase"],row["maximum_artifact_bytes"],row["representation"]));selected=eligible[0] if eligible else None;status="CANDIDATE_FROZEN" if selected else "NO_CANDIDATE";destination=workspace/config["output_root"]/"summaries"/"ali_stage6_storage_screen_summary_v1"
    if destination.exists():raise FileExistsError(destination)
    stage=destination.parent/".work"/f"summary.{uuid.uuid4().hex}.stage";stage.mkdir(parents=True);summary={"schema":SCHEMA,"status":status,"selected_candidate":selected,"all_candidates":candidates,"stage6b_required":selected is not None,"storage_claim_passed":False,"reason":"Behavioral confirmation remains required." if selected else "No representation passed the frozen 16x/10% parameter screen.","final_test_accessed":False,"test_accessed":False};atomic_json(stage/"summary.json",summary);lines=["# Ali Stage 6A storage screen summary","",f"Status: **{status}**.",""]
    if selected:lines.append(f"Frozen Stage 6B candidate: `{selected['representation']}` ({selected['minimum_compression_ratio']:.2f}x minimum compression; worst parameter-residual increase {selected['worst_case_residual_increase']:.4f}).")
    else:lines.append("No representation passed the frozen screen; Stage 6B must not run.")
    lines.extend(["","| Representation | Min compression | Worst increase | Eligible |","|---|---:|---:|---|"])
    for row in candidates:lines.append(f"| {row['representation']} | {row['minimum_compression_ratio']:.2f}x | {row['worst_case_residual_increase']:.4f} | {row['eligible']} |")
    (stage/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n");(stage/"COMPLETED").write_text(SUMMARY_MARKER+"\n",encoding="utf-8",newline="\n");atomic_json(stage/"manifest.json",{"schema":SCHEMA,"files":{name:sha256_file(stage/name) for name in ("COMPLETED","summary.json","summary.md")},"source_manifests":{str(seed):sha256_file(runs[i]/"manifest.json") for i,seed in enumerate(SEEDS)},"test_accessed":False});destination.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,destination);return {"status":status,"summary_dir":str(destination),"selected_candidate":selected,"stage6b_required":selected is not None,"test_accessed":False}


def synthetic()->dict[str,Any]:
    rows=[{"representation":"topq_0.03125","compression_ratio":20.0,"residual_increase":.08},{"representation":"shared_low_rank_r16","compression_ratio":15.9,"residual_increase":.01}];eligible=[row for row in rows if row["compression_ratio"]>=16 and row["residual_increase"]<=.10];return {"schema":SCHEMA,"bank_sizes":list(BANK_SIZES),"representations":_representation_names(),"rows_per_seed":len(BANK_SIZES)*len(_representation_names())*2,"selected":eligible[0]["representation"],"model_loaded":False,"development_predictions_accessed":False,"test_accessed":False}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--repo",type=Path,required=True);parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--config",type=Path,default=Path("configs/ali_stage6_storage_screen_v1.yaml"));parser.add_argument("--seed",type=int,choices=SEEDS,default=42);parser.add_argument("--mode",choices=("Preflight","SyntheticDryRun","Full","Analyze","Summary"),default="Preflight");parser.add_argument("--run-name",default="ali_stage6_storage_screen_seed42_v1");args=parser.parse_args();repo=args.repo.resolve();workspace=args.workspace.resolve();config_path=args.config if args.config.is_absolute() else repo/args.config
    if args.mode=="Preflight":
        authority,_,_,_,users,hashes,queries=preflight(repo,workspace,config_path,args.seed);result={"status":"READY","authority":authority,"bank_users":len(users),"public_hashes":len(hashes),"queries":list(queries),"model_loaded":False,"test_accessed":False}
    elif args.mode=="SyntheticDryRun":result=synthetic()
    elif args.mode=="Full":result=execute(repo,workspace,config_path,args.seed,args.run_name)
    elif args.mode=="Analyze":result=analyze(load_config(repo,workspace,config_path),workspace,args.seed,args.run_name)
    else:result=summarize(repo,workspace,config_path)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
