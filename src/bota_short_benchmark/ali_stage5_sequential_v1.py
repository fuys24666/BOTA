"""Development-only cumulative registered-user deletion experiment."""
from __future__ import annotations

import argparse
import csv
import gc
import json
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

from src.bota_if.p1_trajectory_transport_audit import StepBudget
from src.bota_short_benchmark import ali_stage2_stronger_signal_v1 as stage2
from src.bota_short_benchmark import runner
from src.bota_short_benchmark import t012_behavior_ablation as behavior
from src.bota_short_benchmark.ali_stage1_mechanism_v1 import run_shared_canonical
from src.bota_short_benchmark.ali_stage4_attack_v1 import _attack, _paired_bootstrap, _row_from_prediction
from src.bota_short_benchmark.protocol import load_config as load_benchmark_config
from src.bota_short_benchmark.protocol import validate_prepared
from src.diagnostics.ml1m_development_protocol import reconstruct_authoritative_rows
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from src.diagnostics.t5_zero_training_analysis import ATTACKS
from src.paper_baselines.common import capture_rng, restore_rng, tensor_tree_hash
from src.paper_if_a2.common import atomic_json, canonical_hash, directory_hash, git_snapshot, safe_run_name, seed_everything, sha256_file

SCHEMA="bota-ali-stage5-sequential-v1"
MARKER="BOTA_ALI_STAGE5_SEQUENTIAL_V1_COMPLETED"
SUMMARY_MARKER="BOTA_ALI_STAGE5_SEQUENTIAL_SUMMARY_V1_COMPLETED"
SEEDS=(41,42,43)
SIZES=(1,2,4,8)
PRIMARY_ORDERS=("low_first","middle_first","alternating","deterministic_hash")
SELECTED_ARM="FS_empirical_fisher"
RUN_REQUIRED={"COMPLETED","contract.json","endpoints","manifest.json","metrics.csv","metrics.json","attacks.json","per_sample_metrics.jsonl","provenance.json","report.md","request_registry.json","run_state.json","timing.json"}


def _json(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict):raise ValueError(f"expected JSON object: {path}")
    return value


def load_config(repo:Path,workspace:Path,path:Path)->dict[str,Any]:
    value=yaml.safe_load(path.read_text(encoding="utf-8"))
    required={"schema","test_access_policy","parent_amendment","stage1_selection","benchmark_configs","output_root","protocol","evaluation","decision","runtime"}
    if not isinstance(value,dict) or set(value)!=required or value["schema"]!=SCHEMA or value["test_access_policy"]!="forbidden":raise ValueError("invalid Stage 5 config")
    parent=value["parent_amendment"];parent_path=repo/parent["path"]
    if sha256_file(parent_path)!=parent["sha256"] or _json(parent_path).get("amendment_id")!=parent["amendment_id"]:raise ValueError("Amendment 04 binding changed")
    selection=value["stage1_selection"];selection_path=workspace/selection["path"]
    if sha256_file(selection_path)!=selection["sha256"] or _json(selection_path).get("selection",{}).get("selected_arm")!=SELECTED_ARM:raise ValueError("Stage 1 selection changed")
    protocol=value["protocol"]
    expected={"seeds":list(SEEDS),"optimizer_steps":200,"selected_arm":SELECTED_ARM,"primary_scenario":"L4M4","primary_orders":list(PRIMARY_ORDERS),"secondary_scenario":"L8","secondary_orders":["deterministic_hash"],"cumulative_sizes":list(SIZES),"recompose_from_canonical":True,"chain_from_published_adapter":False,"online_optimizer_steps":0}
    if protocol!=expected:raise ValueError("Stage 5 protocol changed")
    evaluation=value["evaluation"]
    if evaluation!={"split":"Development","final_test":False,"inference_batch_size":4,"behavior_bootstrap_resamples":1000,"behavior_bootstrap_seed":45100,"attack_bootstrap_resamples":2000,"attack_bootstrap_seed":42}:raise ValueError("Stage 5 evaluation changed")
    for seed in SEEDS:
        binding=value["benchmark_configs"][seed]
        if sha256_file(repo/binding["path"])!=binding["sha256"]:raise ValueError(f"benchmark config changed: {seed}")
    return value


def _ordered_people(scenario:dict[str,Any],train_users:Sequence[int])->list[dict[str,Any]]:
    rows=[]
    for source_index,public in zip(scenario["forget_train_indices"],scenario["public_members"]):
        rows.append({"user":int(train_users[int(source_index)]),"user_hash":public["user_hash"],"role":public["role"]})
    if len(rows)!=8 or len({row["user"] for row in rows})!=8:raise ValueError("registered scenario users changed")
    return rows


def _sequence(people:list[dict[str,Any]],order_name:str)->list[dict[str,Any]]:
    low=sorted((row for row in people if row["role"]=="low"),key=lambda row:row["user_hash"])
    middle=sorted((row for row in people if row["role"]=="middle"),key=lambda row:row["user_hash"])
    if order_name=="low_first":return low+middle
    if order_name=="middle_first":return middle+low
    if order_name=="alternating":return [row for pair in zip(low,middle) for row in pair]
    if order_name=="deterministic_hash":return sorted(people,key=lambda row:row["user_hash"])
    raise ValueError(order_name)


def build_registry(registry:dict[str,Any],train_users:Sequence[int])->tuple[dict[str,Any],list[dict[str,Any]],list[int]]:
    scenarios={row["id"]:row for row in registry["scenarios"]};cells=[]
    for scenario_id,orders in (("L4M4",PRIMARY_ORDERS),("L8",("deterministic_hash",))):
        people=_ordered_people(scenarios[scenario_id],train_users)
        for order_name in orders:
            sequence=_sequence(people,order_name)
            for size in SIZES:
                prefix=sequence[:size];public_hashes=sorted(row["user_hash"] for row in prefix);union_hash=canonical_hash(public_hashes)
                cells.append({"cell_id":f"{scenario_id}_{order_name}_k{size}","scenario":scenario_id,"order_family":order_name,"request_size":size,"users":[row["user"] for row in prefix],"public_user_hashes":public_hashes,"union_hash":union_hash})
    public_cells=[{key:value for key,value in row.items() if key!="users"} for row in cells]
    union_users=sorted({user for row in cells for user in row["users"]},key=lambda user:canonical_hash(["stage5",user]))
    public={"schema":SCHEMA,"cells":public_cells,"cell_count":len(public_cells),"unique_union_count":len({row["union_hash"] for row in cells}),"registry_sha256":canonical_hash(public_cells),"selection_uses_predictions_or_labels":False,"raw_user_ids_persisted":False,"final_test_accessed":False,"test_accessed":False}
    return public,cells,union_users


def preflight(repo:Path,workspace:Path,config_path:Path,seed:int)->tuple[dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any],list[dict[str,Any]],list[int]]:
    config=load_config(repo,workspace,config_path)
    if seed not in SEEDS:raise ValueError(seed)
    binding=config["benchmark_configs"][seed];benchmark=load_benchmark_config(repo/binding["path"]);benchmark_name=f"bota_short_i02_seed{seed}_v3"
    protocol_dir,contract,base_registry=validate_prepared(workspace,benchmark,benchmark_name)
    if contract.get("optimizer_steps")!=200 or base_registry.get("test_accessed") is not False:raise ValueError("prepared protocol changed")
    public_registry=_json(protocol_dir/"request_registry.json");public_by_id={row["id"]:row for row in public_registry["scenarios"]}
    base_registry={**base_registry,"scenarios":[{**row,"public_members":public_by_id[row["id"]]["public_members"]} for row in base_registry["scenarios"]]}
    train_users,development_users,lineage=stage2._source_users(workspace,benchmark)
    public,cells,union_users=build_registry(base_registry,train_users)
    original_run,manifest,state=behavior._validate_reference(workspace,benchmark,"Original-Short",f"bota_short_original_seed{seed}_v3",benchmark_name,"L8")
    model_dir=Path(runner._base_t5_config(workspace,benchmark)["paths"]["model_dir"]);model_dir=model_dir if model_dir.is_absolute() else workspace/model_dir
    missing=[name for name in ("config.json","pytorch_model.bin","spiece.model","tokenizer_config.json") if not (model_dir/name).is_file()]
    if missing:raise ValueError(f"local T5-base incomplete: {missing}")
    authority={"schema":SCHEMA,"seed":seed,"benchmark_name":benchmark_name,"benchmark_config_sha256":binding["sha256"],"prepared_registry_sha256":base_registry["registry_sha256"],"sequential_registry_sha256":public["registry_sha256"],"cells":len(cells),"unique_unions":public["unique_union_count"],"union_users":len(union_users),"original_run":str(original_run),"original_adapter_sha256":manifest["artifact"]["sha256"],"source_checkpoint_sha256":sha256_file(workspace/benchmark["source"]["original_checkpoint"]),"lineage":lineage,"model_loaded":False,"development_predictions_accessed":False,"final_test_accessed":False,"test_accessed":False}
    return authority,config,benchmark,base_registry,cells,union_users


def _classification(prediction:dict[str,Any],indices:Sequence[int]|None=None)->dict[str,Any]:
    idx=np.arange(len(prediction["probability"])) if indices is None else np.asarray(indices,dtype=np.int64);p=np.asarray(prediction["probability"],dtype=np.float64)[idx];y=np.asarray(prediction["gold_label"],dtype=np.int64)[idx];p=p.clip(1e-12,1-1e-12)
    auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
    return {"auc":auc,"accuracy":float(accuracy_score(y,p>=.5)),"log_loss":float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))),"prediction_collapse":bool(np.std(p)<1e-6),"samples":len(idx),"auc_defined":auc is not None}


def _parameter_metrics(original_dir:Path,exact_dir:Path,bota_dir:Path)->dict[str,float]:
    states=[torch.load(path/"adapter"/"adapter_model.pt",map_location="cpu",weights_only=True)["B"] for path in (original_dir,exact_dir,bota_dir)];keys=list(states[0]);dot=ne=nb=error=0.0
    for key in keys:
        exact=(states[1][key]-states[0][key]).double();bota=(states[2][key]-states[0][key]).double();dot+=float((exact*bota).sum());ne+=float((exact*exact).sum());nb+=float((bota*bota).sum());error+=float(((bota-exact)**2).sum())
    return {"delta_cosine":dot/max((ne*nb)**.5,1e-30),"norm_ratio":nb**.5/max(ne**.5,1e-30),"relative_l2":error**.5/max(ne**.5,1e-30)}


def _same_prediction_order(*rows:dict[str,Any])->bool:
    return all(row["gold_label"]==rows[0]["gold_label"] and row["sample_order_sha256"]==rows[0]["sample_order_sha256"] for row in rows[1:])


def execute(repo:Path,workspace:Path,config_path:Path,seed:int,run_name:str)->dict[str,Any]:
    authority,config,benchmark,base_registry,cells,union_users=preflight(repo,workspace,config_path,seed);git=git_snapshot(repo)
    if not git["clean"]:raise RuntimeError("formal Stage 5 requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("exactly one CUDA GPU required")
    torch.cuda.set_device(0);device=torch.device("cuda:0");free,total=torch.cuda.mem_get_info(device)
    if free/2**30<float(config["runtime"]["minimum_free_gib"]):raise RuntimeError("insufficient free GPU memory")
    torch.cuda.set_per_process_memory_fraction(runner.allocator_fraction_for(total/2**30,float(config["runtime"]["allocator_fraction"]),float(config["runtime"]["hard_peak_reserved_gib"])),device);torch.cuda.reset_peak_memory_stats()
    destination=workspace/config["output_root"] / "runs" / safe_run_name(run_name)
    if destination.exists():raise FileExistsError(destination)
    stage=destination.parent/".work"/f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage";stage.mkdir(parents=True)
    source=workspace/benchmark["source"]["original_checkpoint"];source_before=sha256_file(source);original_run=Path(authority["original_run"]);original_state=torch.load(original_run/"scenarios"/"L8"/"adapter"/"adapter_model.pt",map_location="cpu",weights_only=True)
    model=None;old=Path.cwd();os.chdir(workspace);started=time.perf_counter()
    try:
        seed_everything(seed);model,names,parameters,bases,_,tokenizer,train_data,train_users=runner._model_context(workspace,benchmark,base_registry,device,original_state["A"]);bases_cpu={name:value.detach().cpu().clone() for name,value in bases.items()};initial_rng=capture_rng();budget=StepBudget(200)
        canonical,all_states,trace,optimizer_hash,rng_hash=run_shared_canonical(model,train_data,base_registry["order"],train_users,union_users,parameters,names,device,tokenizer.pad_token_id,runner._engine(workspace,benchmark),budget,arms=(SELECTED_ARM,))
        if not stage2._same_tensor_map(bases_cpu,original_state["A"]):raise RuntimeError("fixed LoRA-A coordinate mismatch")
        if budget.calls!=200 or any(not torch.equal(canonical[name],original_state["B"][name]) for name in names):raise RuntimeError("canonical endpoint mismatch")
        states=all_states[SELECTED_ARM];del all_states,model,parameters;model=None;gc.collect();torch.cuda.empty_cache()
        original_dir=stage/"endpoints"/"original";runner._save_fixed_ab(original_dir/"adapter",names,canonical,bases_cpu,"Original-Stage5")
        unique={row["union_hash"]:row for row in cells};user_slot={user:index for index,user in enumerate(union_users)};request_times={};endpoint_hashes={}
        for union_hash,row in unique.items():
            begun=time.perf_counter();candidate={name:canonical[name]+states[module]["theta"][[user_slot[user] for user in row["users"]]].sum(0).float() for module,name in enumerate(names)};root=stage/"endpoints"/union_hash;artifact=runner._save_fixed_ab(root/"bota"/"adapter",names,candidate,bases_cpu,"BOTA-FS-EmpiricalFisher-Stage5");request_times[union_hash]=time.perf_counter()-begun;endpoint_hashes[union_hash]={"bota":artifact["sha256"]};del candidate
        del states
        exact_times={};exact_optimizer_hashes={}
        for union_hash,row in unique.items():
            seed_everything(seed);model,exact_names,exact_parameters,exact_bases,_,exact_tokenizer,exact_data,exact_users=runner._model_context(workspace,benchmark,base_registry,device,original_state["A"])
            if list(exact_names)!=list(names) or not stage2._same_tensor_map(exact_bases,bases_cpu):raise RuntimeError("paired Exact-Masked coordinate mismatch")
            restore_rng(initial_rng);exact_budget=StepBudget(200);begun=time.perf_counter();values,optimizer,exact_trace=runner._train_named(model,exact_data,base_registry["order"],exact_users,row["users"],exact_names,exact_parameters,device,exact_tokenizer.pad_token_id,runner._engine(workspace,benchmark),exact_budget,f"stage5_exact_{union_hash[:12]}");exact_times[union_hash]=time.perf_counter()-begun
            if exact_budget.calls!=200 or [item["batch_hash"] for item in exact_trace]!=[item["batch_hash"] for item in trace]:raise RuntimeError("paired Exact-Masked trace mismatch")
            exact_optimizer_hashes[union_hash]=tensor_tree_hash(optimizer.state_dict());artifact=runner._save_fixed_ab(stage/"endpoints"/union_hash/"exact_masked"/"adapter",names,values,bases_cpu,"Exact-Masked-Stage5");endpoint_hashes[union_hash]["exact_masked"]=artifact["sha256"];del model,exact_parameters,optimizer,values;model=None;gc.collect();torch.cuda.empty_cache()
        base=runner._base_t5_config(workspace,benchmark);model_dir=Path(base["paths"]["model_dir"]);model_dir=model_dir if model_dir.is_absolute() else workspace/model_dir;eval_tokenizer=T5Tokenizer.from_pretrained(model_dir,local_files_only=True);development=JsonPromptDataset(workspace/benchmark["source"]["development_json"],eval_tokenizer);train_authority,development_users,_=stage2._source_users(workspace,benchmark);indices=list(range(len(development)));original_prediction=behavior._predict_scenario_dir(workspace,benchmark,original_dir,development,indices,device);original_utility=_classification(original_prediction);train_records=json.loads((workspace/benchmark["source"]["train_json"]).read_text(encoding="utf-8"));window_records=[train_records[index] for index in base_registry["order"]];window_users=[int(train_authority[index]) for index in base_registry["order"]]
        endpoint_results={};attack_rows=[];safe_samples=[]
        for number,(union_hash,row) in enumerate(unique.items()):
            root=stage/"endpoints"/union_hash;exact_prediction=behavior._predict_scenario_dir(workspace,benchmark,root/"exact_masked",development,indices,device);bota_prediction=behavior._predict_scenario_dir(workspace,benchmark,root/"bota",development,indices,device)
            if not _same_prediction_order(original_prediction,exact_prediction,bota_prediction):raise RuntimeError("Development prediction order mismatch")
            requested=set(row["users"]);selected=[i for i,user in enumerate(development_users) if int(user) in requested];retained=[i for i,user in enumerate(development_users) if int(user) not in requested];metric=behavior.behavior_metrics("BOTA-Stage5",bota_prediction,exact_prediction,original_prediction,selected,development_users,int(config["evaluation"]["behavior_bootstrap_resamples"]),int(config["evaluation"]["behavior_bootstrap_seed"])+number);op=np.asarray(original_prediction["probability"]);ep=np.asarray(exact_prediction["probability"]);bp=np.asarray(bota_prediction["probability"]);chosen=np.asarray(selected,dtype=np.int64)
            positions=[position for position,user in enumerate(window_users) if user in requested];members=[window_records[position] for position in positions];member_users=[window_users[position] for position in positions];nonmember_users=[development_users[i] for i in selected];attack_data=JsonPromptDataset(workspace/benchmark["source"]["development_json"],eval_tokenizer);attack_data.records=members+[development.records[i] for i in selected];attack_indices=list(range(len(attack_data)));attack_predictions={"Original":behavior._predict_scenario_dir(workspace,benchmark,original_dir,attack_data,attack_indices,device),"Exact-Masked":behavior._predict_scenario_dir(workspace,benchmark,root/"exact_masked",attack_data,attack_indices,device),"BOTA":behavior._predict_scenario_dir(workspace,benchmark,root/"bota",attack_data,attack_indices,device)}
            if not _same_prediction_order(*attack_predictions.values()):raise RuntimeError("matched-user attack prediction order mismatch")
            caches={};split=len(member_users)
            for endpoint,prediction in attack_predictions.items():
                member_rows=[_row_from_prediction(prediction,i,member_users[i]) for i in range(split)];nonmember_rows=[_row_from_prediction(prediction,split+i,nonmember_users[i]) for i in range(len(nonmember_users))];caches[endpoint]=(member_rows,nonmember_rows)
                for attack_name,score in ATTACKS.items():attack_rows.append({"union_hash":union_hash,"request_size":len(requested),"model":endpoint,"attack":attack_name,"formal_interpretation_permitted":False,**_attack(member_rows,nonmember_rows,score,int(config["evaluation"]["attack_bootstrap_seed"]),int(config["evaluation"]["attack_bootstrap_resamples"]))})
            paired=_paired_bootstrap(caches,"negative_answer_loss",int(config["evaluation"]["attack_bootstrap_seed"]),int(config["evaluation"]["attack_bootstrap_resamples"]));bota_utility=_classification(bota_prediction);guard={"auc_damage":original_utility["auc"]-bota_utility["auc"],"logloss_damage":bota_utility["log_loss"]-original_utility["log_loss"],"prediction_collapse":bota_utility["prediction_collapse"]};guard["pass"]=guard["auc_damage"]<=config["decision"]["overall_auc_damage_max"] and guard["logloss_damage"]<=config["decision"]["overall_logloss_damage_max"] and not guard["prediction_collapse"]
            endpoint_results[union_hash]={"DU":float(np.mean(np.abs(op[chosen]-ep[chosen]))),"EU":float(np.mean(np.abs(bp[chosen]-ep[chosen]))),"local_residual":metric["local_residual"],"residual_ci":[metric["ci_lower"],metric["ci_upper"]],"toward_exact_rate":metric["toward_rate"],"toward_exact":metric["toward_exact"],"away_from_exact":metric["away_from_exact"],"overall_original":original_utility,"overall_exact":_classification(exact_prediction),"overall_bota":bota_utility,"requested_original":_classification(original_prediction,selected),"requested_exact":_classification(exact_prediction,selected),"requested_bota":_classification(bota_prediction,selected),"retained_original":_classification(original_prediction,retained),"retained_exact":_classification(exact_prediction,retained),"retained_bota":_classification(bota_prediction,retained),"utility_guardrail":guard,"parameter_metrics":_parameter_metrics(original_dir,root/"exact_masked",root/"bota"),"request_stage_wall_time_seconds":request_times[union_hash],"descriptive_primary_attack":paired,"attack_interpretation_permitted":False}
            for i in selected:safe_samples.append({"union_hash":union_hash,"development_index":i,"user_hash":canonical_hash([seed,development_users[i]]),"original_probability":float(op[i]),"exact_probability":float(ep[i]),"bota_probability":float(bp[i])})
        metric_rows=[]
        for cell in cells:metric_rows.append({"seed":seed,"cell_id":cell["cell_id"],"scenario":cell["scenario"],"order_family":cell["order_family"],"request_size":cell["request_size"],"union_hash":cell["union_hash"],**endpoint_results[cell["union_hash"]]})
        peak=torch.cuda.max_memory_reserved()/2**30
        if peak>config["runtime"]["hard_peak_reserved_gib"] or sha256_file(source)!=source_before:raise RuntimeError("resource or source-integrity guard failed")
        public,cells_again,_=build_registry(base_registry,train_users)
        if [row["union_hash"] for row in cells_again]!=[row["union_hash"] for row in cells]:raise RuntimeError("request registry changed")
        atomic_json(stage/"request_registry.json",public);atomic_json(stage/"metrics.json",{"schema":SCHEMA,"seed":seed,"cells":metric_rows,"cell_count":len(metric_rows),"unique_union_count":len(unique),"all_cells_reported":len(metric_rows)==20,"attack_interpretation_permitted":False,"final_test_accessed":False,"test_accessed":False});atomic_json(stage/"attacks.json",{"schema":SCHEMA,"rows":attack_rows,"unique_unions":len(unique),"models":3,"attacks_per_model":4,"formal_interpretation_permitted":False,"parent_attack_result_unchanged":True,"test_accessed":False})
        flat=[]
        for row in metric_rows:flat.append({"seed":seed,"cell_id":row["cell_id"],"scenario":row["scenario"],"order_family":row["order_family"],"request_size":row["request_size"],"union_hash":row["union_hash"],"DU":row["DU"],"EU":row["EU"],"local_residual":row["local_residual"],"ci_lower":row["residual_ci"][0],"ci_upper":row["residual_ci"][1],"toward_exact_rate":row["toward_exact_rate"],"delta_cosine":row["parameter_metrics"]["delta_cosine"],"norm_ratio":row["parameter_metrics"]["norm_ratio"],"relative_l2":row["parameter_metrics"]["relative_l2"],"bota_auc":row["overall_bota"]["auc"],"bota_log_loss":row["overall_bota"]["log_loss"],"utility_pass":row["utility_guardrail"]["pass"],"request_stage_seconds":row["request_stage_wall_time_seconds"]})
        with (stage/"metrics.csv").open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        with (stage/"per_sample_metrics.jsonl").open("w",encoding="utf-8",newline="\n") as handle:
            for row in safe_samples:handle.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
        lines=["# Ali Stage 5 sequential cumulative-union result","",f"Seed {seed}; Development only; MIA is descriptive.","","| Cell | Residual [95% CI] | Toward | Utility |","|---|---:|---:|---|"]
        for row in metric_rows:lines.append(f"| {row['cell_id']} | {row['local_residual']:.4f} [{row['residual_ci'][0]:.4f}, {row['residual_ci'][1]:.4f}] | {row['toward_exact_rate']:.3f} | {'PASS' if row['utility_guardrail']['pass'] else 'FAIL'} |")
        (stage/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n");atomic_json(stage/"timing.json",{"exact_replay_seconds":exact_times,"bota_request_stage_seconds":request_times,"end_to_end_seconds":time.perf_counter()-started});atomic_json(stage/"contract.json",authority);atomic_json(stage/"provenance.json",{"schema":SCHEMA,"git":git,"source_checkpoint_sha256":source_before,"canonical_trace_sha256":canonical_hash(trace),"canonical_optimizer_sha256":optimizer_hash,"canonical_rng_sha256":rng_hash,"paired_initial_rng_sha256":tensor_tree_hash(initial_rng),"exact_optimizer_sha256":exact_optimizer_hashes,"request_registry_sha256":public["registry_sha256"],"raw_user_ids_persisted":False,"development_accessed":True,"final_test_loader_built":False,"test_accessed":False});atomic_json(stage/"run_state.json",{"schema":SCHEMA,"status":"COMPLETED","seed":seed,"cells":20,"unique_unions":len(unique),"canonical_optimizer_steps":200,"exact_masked_optimizer_steps":200*len(unique),"physical_optimizer_step_calls":200*(1+len(unique)),"online_optimizer_steps":0,"peak_reserved_gib":peak,"test_accessed":False});(stage/"COMPLETED").write_text(MARKER+"\n",encoding="utf-8",newline="\n");files={name:sha256_file(stage/name) for name in ("COMPLETED","contract.json","metrics.csv","metrics.json","attacks.json","per_sample_metrics.jsonl","provenance.json","report.md","request_registry.json","run_state.json","timing.json")};atomic_json(stage/"manifest.json",{"schema":SCHEMA,"files":files,"original_adapter_sha256":directory_hash(original_dir/"adapter"),"endpoint_sha256":endpoint_hashes,"published_atomically":True,"test_accessed":False});destination.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,destination)
        return {"status":"COMPLETED","run_dir":str(destination),"seed":seed,"cells":20,"unique_unions":len(unique),"physical_optimizer_step_calls":200*(1+len(unique)),"test_accessed":False}
    except Exception as error:
        shutil.rmtree(stage,ignore_errors=True);raise RuntimeError("Stage 5 failed before atomic publication") from error
    finally:
        os.chdir(old)
        if model is not None:del model
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def analyze(config:dict[str,Any],workspace:Path,seed:int,run_name:str)->dict[str,Any]:
    run=workspace/config["output_root"]/"runs"/safe_run_name(run_name)
    if not run.is_dir() or {path.name for path in run.iterdir()}!=RUN_REQUIRED or (run/"COMPLETED").read_text(encoding="utf-8")!=MARKER+"\n":raise ValueError("invalid Stage 5 run")
    manifest=_json(run/"manifest.json")
    for name,digest in manifest["files"].items():
        if sha256_file(run/name)!=digest:raise ValueError(f"Stage 5 artifact changed: {name}")
    if directory_hash(run/"endpoints"/"original"/"adapter")!=manifest["original_adapter_sha256"]:raise ValueError("Stage 5 Original endpoint changed")
    for union_hash,endpoints in manifest["endpoint_sha256"].items():
        for endpoint,digest in endpoints.items():
            if directory_hash(run/"endpoints"/union_hash/endpoint/"adapter")!=digest:raise ValueError(f"Stage 5 endpoint changed: {union_hash}/{endpoint}")
    metrics=_json(run/"metrics.json");registry=_json(run/"request_registry.json");attacks=_json(run/"attacks.json");state=_json(run/"run_state.json")
    if len(metrics.get("cells",[]))!=20 or registry.get("cell_count")!=20 or len(attacks.get("rows",[]))!=registry["unique_union_count"]*12 or state.get("status")!="COMPLETED" or any(row.get("test_accessed") is not False for row in (metrics,registry,attacks,state)):raise ValueError("Stage 5 completeness invariant failed")
    return {"status":"VALID","run_dir":str(run),"seed":seed,"cells":20,"unique_unions":registry["unique_union_count"],"test_accessed":False}


def summarize(repo:Path,workspace:Path,config_path:Path)->dict[str,Any]:
    config=load_config(repo,workspace,config_path);runs=[];rows=[]
    for seed in SEEDS:
        name=f"ali_stage5_sequential_seed{seed}_v1";analyze(config,workspace,seed,name);run=workspace/config["output_root"]/"runs"/name;runs.append(run);rows.extend(_json(run/"metrics.json")["cells"])
    primary={}
    for order in PRIMARY_ORDERS:
        values=[row["local_residual"] for row in rows if row["scenario"]=="L4M4" and row["order_family"]==order and row["request_size"]==2];primary[order]={"mean_local_residual":float(np.mean(values)),"sample_sd":float(np.std(values,ddof=1)),"all_cell_utility_pass":all(row["utility_guardrail"]["pass"] for row in rows if row["scenario"]=="L4M4" and row["order_family"]==order and row["request_size"]==2)}
    means=[value["mean_local_residual"] for value in primary.values()];spread=max(means)-min(means);same_final=all(len({row["union_hash"] for row in rows if row["seed"]==seed and row["scenario"]=="L4M4" and row["request_size"]==8})==1 for seed in SEEDS);passed=all(value["mean_local_residual"]<config["decision"]["residual_maximum"] and value["all_cell_utility_pass"] for value in primary.values()) and spread<=config["decision"]["order_spread_maximum"] and same_final
    destination=workspace/config["output_root"]/"summaries"/"ali_stage5_sequential_summary_v1"
    if destination.exists():raise FileExistsError(destination)
    stage=destination.parent/".work"/f"summary.{uuid.uuid4().hex}.stage";stage.mkdir(parents=True);summary={"schema":SCHEMA,"status":"PASS" if passed else "FAIL","primary_k2":primary,"order_spread":spread,"order_spread_maximum":config["decision"]["order_spread_maximum"],"same_l4m4_k8_union_invariant":same_final,"cells":len(rows),"interpretation":"conditional cumulative-recomposition support" if passed else "retain one-shot-only scope","attack_interpretation_permitted":False,"final_test_accessed":False,"test_accessed":False};atomic_json(stage/"summary.json",summary);lines=["# Ali Stage 5 sequential summary","",f"Decision: **{summary['status']}**. {summary['interpretation']}.","","| L4M4 K=2 order | Mean residual | SD | Utility |","|---|---:|---:|---|"]
    for order,value in primary.items():lines.append(f"| {order} | {value['mean_local_residual']:.4f} | {value['sample_sd']:.4f} | {'PASS' if value['all_cell_utility_pass'] else 'FAIL'} |")
    lines.extend(["",f"Order spread: {spread:.4f} (maximum 0.10).",f"Common K=8 union invariant: {same_final}.",""]);(stage/"summary.md").write_text("\n".join(lines),encoding="utf-8",newline="\n");(stage/"COMPLETED").write_text(SUMMARY_MARKER+"\n",encoding="utf-8",newline="\n");atomic_json(stage/"manifest.json",{"schema":SCHEMA,"files":{name:sha256_file(stage/name) for name in ("COMPLETED","summary.json","summary.md")},"source_manifests":{str(seed):sha256_file(runs[i]/"manifest.json") for i,seed in enumerate(SEEDS)},"test_accessed":False});destination.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,destination);return {"status":summary["status"],"summary_dir":str(destination),"order_spread":spread,"one_shot_only":not passed,"test_accessed":False}


def synthetic()->dict[str,Any]:
    means={order:value for order,value in zip(PRIMARY_ORDERS,(.61,.63,.66,.65))};spread=max(means.values())-min(means.values());return {"schema":SCHEMA,"cells_per_seed":20,"total_cells":60,"primary_orders":list(PRIMARY_ORDERS),"sizes":list(SIZES),"example_order_spread":spread,"example_pass":all(value<1 for value in means.values()) and spread<=.10,"recompose_from_canonical":True,"chain_from_published_adapter":False,"model_loaded":False,"test_accessed":False}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--repo",type=Path,required=True);parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--config",type=Path,default=Path("configs/ali_stage5_sequential_v1.yaml"));parser.add_argument("--seed",type=int,choices=SEEDS,default=42);parser.add_argument("--mode",choices=("Preflight","SyntheticDryRun","Full","Analyze","Summary"),default="Preflight");parser.add_argument("--run-name",default="ali_stage5_sequential_seed42_v1");args=parser.parse_args();repo=args.repo.resolve();workspace=args.workspace.resolve();config_path=args.config if args.config.is_absolute() else repo/args.config
    if args.mode=="Preflight":
        authority,_,_,_,cells,unions=preflight(repo,workspace,config_path,args.seed);result={"status":"READY","authority":authority,"cells":len(cells),"union_users":len(unions),"model_loaded":False,"test_accessed":False}
    elif args.mode=="SyntheticDryRun":result=synthetic()
    elif args.mode=="Full":result=execute(repo,workspace,config_path,args.seed,args.run_name)
    elif args.mode=="Analyze":result=analyze(load_config(repo,workspace,config_path),workspace,args.seed,args.run_name)
    else:result=summarize(repo,workspace,config_path)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
