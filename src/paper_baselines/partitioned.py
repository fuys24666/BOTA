from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import T5ForConditionalGeneration

from src.paper_if_a2.artifacts import atomic_torch_save, complete, publish_manifest
from src.paper_if_a2.common import atomic_json, canonical_hash, directory_hash, sha256_file

from .common import (
    SafeStop, analyze_run, atomic_publish_directory, enforce_memory_guard, existing_run_directory, load_config,
    logical_batches, new_run_directory, paper_model_manifest, preflight as shared_preflight,
    stopped_safely, tensor_tree_hash, write_model_manifest,
)
from .runtime import (
    answer_loss, batch, clean_lora_model, development_binary_predictions,
    load_trainable_state, model_counts, release, tokenizer_and_dataset, trainable_state,
)
from .budget import partitioned_budget


SCHEMAS={"sisa":"paper-sisa-v1","receraser":"paper-receraser-adapter-v1"}
METHODS={"sisa":"SISA-T5","receraser":"RecEraser-Adapter"}
MARKERS={"sisa":"PAPER_SISA_V1_COMPLETED","receraser":"PAPER_RECERASER_ADAPTER_V1_COMPLETED"}


def row_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def locate_subset_rows(full: list[dict[str,Any]], subset: list[dict[str,Any]]) -> list[int]:
    locations: dict[str,collections.deque[int]]=collections.defaultdict(collections.deque)
    for index,row in enumerate(full):locations[row_key(row)].append(index)
    result=[]
    for row in subset:
        key=row_key(row)
        if not locations[key]:raise ValueError("subset row is absent or duplicated beyond full data")
        result.append(locations[key].popleft())
    return result


def assert_interaction_disjoint(
    train: list[dict[str, Any]],
    development: list[dict[str, Any]],
    train_entity_ids: list[int] | None = None,
    development_entity_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Reject identity leakage without confusing equal model inputs with one interaction."""
    train_by_key: dict[str, list[int]] = collections.defaultdict(list)
    development_by_key: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(train): train_by_key[row_key(row)].append(index)
    for index, row in enumerate(development): development_by_key[row_key(row)].append(index)
    overlap = set(train_by_key) & set(development_by_key)
    if not overlap:
        return {"semantic_overlap_keys": 0, "cross_entity_equal_rows": 0, "identity_overlap": False}
    if (train_entity_ids is None) != (development_entity_ids is None):
        raise ValueError("training/Development entity sidecars must be provided together")
    if train_entity_ids is None or development_entity_ids is None:
        raise ValueError("training and Development contain identical interactions")
    if len(train_entity_ids) != len(train) or len(development_entity_ids) != len(development):
        raise ValueError("training/Development entity sidecar length mismatch")
    if not all(isinstance(value, int) and value > 0 for value in train_entity_ids + development_entity_ids):
        raise ValueError("training/Development entity sidecars are invalid")
    cross_entity = 0
    for key in overlap:
        for train_index in train_by_key[key]:
            for development_index in development_by_key[key]:
                if train_entity_ids[train_index] == development_entity_ids[development_index]:
                    raise ValueError("training and Development contain identical interactions for the same entity")
                cross_entity += 1
    return {"semantic_overlap_keys": len(overlap), "cross_entity_equal_rows": cross_entity, "identity_overlap": False}


def _verified_short_benchmark_entity_ids(
    root: Path, config: dict[str, Any], train: list[dict[str, Any]], development: list[dict[str, Any]],
) -> tuple[list[int] | None, list[int] | None, dict[str, Any]]:
    """Recover manifest-bound identities for already-published short benchmarks."""
    train_path = (root / config["sources"]["train"]["path"]).resolve()
    protocol = train_path.parent.parent
    required = {"PREPARED", "contract.json", "data", "manifest.json", "request_registry.json", "request_registry.private.json"}
    if not protocol.is_dir() or {path.name for path in protocol.iterdir()} != required:
        return None, None, {"type": "strict_semantic_disjointness", "verified": True}
    manifest = json.loads((protocol / "manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((protocol / "contract.json").read_text(encoding="utf-8"))
    registry = json.loads((protocol / "request_registry.private.json").read_text(encoding="utf-8"))
    if (protocol / "PREPARED").read_text(encoding="utf-8") != "BOTA_SHORT_BENCHMARK_V1_PREPARED\n":
        raise ValueError("short benchmark identity authority marker mismatch")
    for name, expected in manifest.get("files", {}).items():
        if sha256_file(protocol / name) != expected: raise ValueError(f"short benchmark identity authority mismatch: {name}")
    data_root = protocol / "data"
    data_hash = canonical_hash([(str(path.relative_to(data_root)).replace("\\", "/"), sha256_file(path)) for path in sorted(data_root.rglob("*")) if path.is_file()])
    if data_hash != manifest.get("data_tree_sha256"): raise ValueError("short benchmark identity data tree mismatch")
    if contract.get("test_accessed") is not False or registry.get("test_accessed") is not False:
        raise ValueError("short benchmark identity authority accessed test")
    sources = contract.get("sources", {})
    needed = ("train_json", "development_json", "train_user_ids", "development_user_ids", "raw_lineage_manifest")
    if not all(name in sources for name in needed):
        return None, None, {"type": "strict_semantic_disjointness", "verified": True}
    authority_paths: dict[str, Path] = {}
    for name in needed:
        entry = sources[name]; path = (root / entry["path"]).resolve()
        if any(part.lower() in {"test", "final_test", "finaltest"} for part in path.parts):
            raise ValueError("test-like path forbidden in split identity authority")
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"split identity authority SHA mismatch: {name}")
        authority_paths[name] = path
    development_path = (root / config["sources"]["development"]["path"]).resolve()
    if development_path != authority_paths["development_json"]:
        raise ValueError("Development identity authority path mismatch")
    raw_manifest = json.loads(authority_paths["raw_lineage_manifest"].read_text(encoding="utf-8"))
    prepared_root = authority_paths["raw_lineage_manifest"].parent
    if raw_manifest.get("test_accessed") is not False:
        raise ValueError("raw lineage identity authority accessed test")
    for name, expected in raw_manifest.get("files", {}).items():
        if sha256_file(prepared_root / name) != expected: raise ValueError(f"raw lineage artifact mismatch: {name}")
    full_train = json.loads(authority_paths["train_json"].read_text(encoding="utf-8"))
    full_train_entities = json.loads(authority_paths["train_user_ids"].read_text(encoding="utf-8"))
    development_entities = json.loads(authority_paths["development_user_ids"].read_text(encoding="utf-8"))
    order = registry.get("order")
    if not isinstance(order, list) or len(order) != len(train) or len(set(order)) != len(order):
        raise ValueError("short benchmark identity order is invalid")
    if any(not isinstance(index, int) or index < 0 or index >= len(full_train) for index in order):
        raise ValueError("short benchmark identity order is out of range")
    if len(full_train_entities) != len(full_train) or len(development_entities) != len(development):
        raise ValueError("split identity authority length mismatch")
    if [full_train[index] for index in order] != train:
        raise ValueError("train window is not the authority order projection")
    train_entities = [full_train_entities[index] for index in order]
    return train_entities, development_entities, {
        "type": "manifest_bound_entity_identity", "verified": True,
        "protocol_manifest_sha256": sha256_file(protocol / "manifest.json"),
        "raw_lineage_manifest_sha256": sha256_file(authority_paths["raw_lineage_manifest"]),
        "test_accessed": False,
    }


def sisa_partition(rows:list[dict[str,Any]],shards:int,slices:int,seed:int)->dict[int,dict[int,list[int]]]:
    result={shard:{slice_id:[] for slice_id in range(slices)} for shard in range(shards)}
    for index,row in enumerate(rows):
        digest=hashlib.sha256(f"{seed}:{row_key(row)}".encode()).digest();shard=int.from_bytes(digest[:8],"big")%shards;slice_id=int.from_bytes(digest[8:16],"big")%slices;result[shard][slice_id].append(index)
    assigned=[index for shard in result.values() for part in shard.values() for index in part]
    if len(assigned)!=len(rows) or len(set(assigned))!=len(rows):raise ValueError("SISA partition is not isolated/exhaustive")
    return result


def deletion_plan(partition:dict[int,dict[int,list[int]]],forget_indices:list[int])->dict[int,int]:
    lookup={index:(shard,slice_id) for shard,parts in partition.items() for slice_id,indices in parts.items() for index in indices};plan={}
    for index in forget_indices:
        shard,slice_id=lookup[index];plan[shard]=min(plan.get(shard,slice_id),slice_id)
    return plan


def text_features(rows:list[dict[str,Any]],dimensions:int=32)->np.ndarray:
    features=np.zeros((len(rows),dimensions),dtype=np.float32)
    for index,row in enumerate(rows):
        for token in str(row["input"]).lower().split():features[index,int.from_bytes(hashlib.sha256(token.encode()).digest()[:4],"big")%dimensions]+=1
        norm=np.linalg.norm(features[index]);features[index]/=norm if norm else 1
    return features


def balanced_similarity_partition(features:np.ndarray,shards:int,seed:int)->list[int]:
    if len(features)<shards:raise ValueError("fewer rows than shards")
    first=seed%len(features);centers=[first]
    while len(centers)<shards:
        distances=np.min([[float(np.linalg.norm(row-features[c])) for c in centers] for row in features],axis=1);distances[centers]=-1;centers.append(int(np.argmax(distances)))
    capacity=math.ceil(len(features)/shards);counts=[0]*shards;assignment=[]
    for row in features:
        order=sorted(range(shards),key=lambda shard:(float(np.linalg.norm(row-features[centers[shard]])),shard));chosen=next(shard for shard in order if counts[shard]<capacity);assignment.append(chosen);counts[chosen]+=1
    if max(counts)-min(counts)>1:raise ValueError("RecEraser partition is not balanced")
    return assignment


def aggregate_probabilities(probabilities:np.ndarray,weights:np.ndarray)->np.ndarray:
    if probabilities.ndim!=2 or weights.shape!=(probabilities.shape[1],):raise ValueError("aggregation shape mismatch")
    if np.any(weights<0) or not np.isclose(weights.sum(),1):raise ValueError("aggregation weights invalid")
    return probabilities@weights


def fit_attention(probabilities:np.ndarray,labels:np.ndarray,steps:int=200,lr:float=.1)->np.ndarray:
    p=torch.tensor(probabilities,dtype=torch.float64);y=torch.tensor(labels,dtype=torch.float64);logits=torch.zeros(p.shape[1],dtype=torch.float64,requires_grad=True);optimizer=torch.optim.Adam([logits],lr=lr)
    for _ in range(steps):
        optimizer.zero_grad();weights=torch.softmax(logits,0);prediction=(p@weights).clamp(1e-8,1-1e-8);loss=-(y*prediction.log()+(1-y)*(1-prediction).log()).mean();loss.backward();optimizer.step()
    return torch.softmax(logits.detach(),0).numpy()


def validate_partition_config(config:dict[str,Any],kind:str)->None:
    protocol=config["protocol"]
    if kind=="sisa":
        if protocol.get("sharded") is not True or protocol.get("isolated") is not True or protocol.get("sliced") is not True or protocol.get("aggregation")!="mean_yes_no_probability":raise ValueError("SISA definition incomplete")
        if protocol.get("partition_unit")!="interaction":raise ValueError("processed data only authorizes interaction-level SISA")
    else:
        if protocol.get("method_name")!="RecEraser-Adapter" or protocol.get("partition")!="balanced_text_similarity" or protocol.get("aggregation")!="learned_development_probability_attention":raise ValueError("RecEraser-Adapter definition incomplete")
    if protocol["shards"]<2:raise ValueError("partitioned baseline requires multiple shards")


def preflight(root:Path,config_path:Path,kind:str,formal:bool=False,partition_seed_override:int|None=None)->dict[str,Any]:
    config=load_config(config_path,SCHEMAS[kind]);validate_partition_config(config,kind);audit=partitioned_budget(config,kind);partition_seed=int(partition_seed_override if partition_seed_override is not None else config["protocol"].get("partition_seed",config["seed"]));files=[Path(__file__),Path(__file__).with_name("common.py"),Path(__file__).with_name("runtime.py"),Path(__file__).with_name(f"{kind}.py")];result=shared_preflight(root,config_path,SCHEMAS[kind],METHODS[kind],files,formal);result["protocol"]=config["protocol"];result["partition_seed"]=partition_seed;result["training_seed"]=int(config["seed"]);result["contract"]={**result["contract"],"partition_seed":partition_seed,"training_seed":int(config["seed"])};result["sequential_gpu_loading"]=True;result["test_accessed"]=False;result["budget_audit"]=audit;return result


def _train_indices(model:torch.nn.Module,dataset:Any,indices:list[int],config:dict[str,Any],device:torch.device,epochs:int)->int:
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=config["protocol"]["learning_rate"]);steps=0
    for epoch in range(epochs):
        order=sorted(indices,key=lambda index:hashlib.sha256(f"{config['seed']}:{epoch}:{index}".encode()).digest())
        for start in range(0,len(order),config["protocol"]["effective_batch"]):
            logical=order[start:start+config["protocol"]["effective_batch"]]
            if not logical:continue
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0,len(logical),config["protocol"]["physical_microbatch"]):
                micro=batch(dataset,logical[offset:offset+config["protocol"]["physical_microbatch"]],device);loss=answer_loss(model,micro)/math.ceil(len(logical)/config["protocol"]["physical_microbatch"]);loss.backward();del micro,loss
            enforce_memory_guard(config["memory_guard"]);optimizer.step();steps+=1
    return steps


def _new_local_model(root:Path,config:dict[str,Any],kind:str,device:torch.device)->torch.nn.Module:
    if kind=="sisa":return T5ForConditionalGeneration.from_pretrained(root/config["tokenizer"],attn_implementation="eager").float().to(device)
    return clean_lora_model(root/config["sources"]["original"]["path"],config["lora"],device)


def execute(root:Path,config_path:Path,run_name:str,kind:str,resume:bool,partition_seed_override:int|None=None)->dict[str,Any]:
    config=load_config(config_path,SCHEMAS[kind]);validate_partition_config(config,kind);audit=partitioned_budget(config,kind);partition_seed=int(partition_seed_override if partition_seed_override is not None else config["protocol"].get("partition_seed",config["seed"]));pre=preflight(root,config_path,kind,formal=True,partition_seed_override=partition_seed);contract=pre["contract"]
    full=json.loads((root/config["sources"]["train"]["path"]).read_text(encoding="utf-8"));forget_rows=json.loads((root/config["sources"]["forget"]["path"]).read_text(encoding="utf-8"));development_rows=json.loads((root/config["sources"]["development"]["path"]).read_text(encoding="utf-8"));train_entities,development_entities,identity_authority=_verified_short_benchmark_entity_ids(root,config,full,development_rows);split_disjointness=assert_interaction_disjoint(full,development_rows,train_entities,development_entities);forget=set(locate_subset_rows(full,forget_rows));shards=config["protocol"]["shards"]
    partition=sisa_partition(full,shards,config["protocol"].get("slices",1),partition_seed) if kind=="sisa" else {shard:{0:[]} for shard in range(shards)}
    if kind=="receraser":
        assignment=balanced_similarity_partition(text_features(full,config["protocol"]["feature_dimensions"]),shards,partition_seed)
        for index,shard in enumerate(assignment):partition[shard][0].append(index)
    plan=deletion_plan(partition,list(forget));actual_initial=[[len(partition[s][t]) for t in sorted(partition[s])] for s in range(shards)] if kind=="sisa" else [len(partition[s][0]) for s in range(shards)];actual_remaining=[[len([i for i in partition[s][t] if i not in forget]) for t in sorted(partition[s])] for s in range(shards)] if kind=="sisa" else [len([i for i in partition[s][0] if i not in forget]) for s in range(shards)]
    if actual_initial!=config["budget"]["initial_partition_counts"] or actual_remaining!=config["budget"]["post_delete_partition_counts"]:raise ValueError("actual partition budget differs from frozen BudgetAudit before model load")
    run_dir=existing_run_directory(root,config["output_root"],run_name) if resume else new_run_directory(root,config["output_root"],run_name)
    if resume:
        if (run_dir/"COMPLETED").exists() or (run_dir/"STOPPED_SAFELY").exists():raise ValueError("terminal run cannot Resume")
        if json.loads((run_dir/"contract.json").read_text(encoding="utf-8"))!=contract:raise ValueError("Resume contract mismatch")
    else:atomic_json(run_dir/"contract.json",contract);atomic_json(run_dir/"provenance.json",pre);atomic_json(run_dir/"budget_audit.json",audit)
    device=torch.device("cuda:0");_,dataset=tokenizer_and_dataset(root/config["tokenizer"],root/config["sources"]["train"]["path"]);_,development=tokenizer_and_dataset(root/config["tokenizer"],root/config["sources"]["development"]["path"]);atomic_json(run_dir/"partition.json",{"kind":kind,"partition_seed":partition_seed,"training_seed":int(config["seed"]),"shards":{str(s):{str(t):indices for t,indices in parts.items()} for s,parts in partition.items()},"forget_indices_sha256":canonical_hash(sorted(forget)),"affected_shards":plan,"split_disjointness":split_disjointness,"identity_authority":identity_authority,"test_accessed":False});model_root=run_dir/"models";model_root.mkdir(exist_ok=True);state_path=run_dir/"run_state.json";state=json.loads(state_path.read_text(encoding="utf-8")) if resume and state_path.exists() else {"status":"RUNNING","method_id":METHODS[kind],"next_shard":0,"optimizer_steps":0,"initial_training_seconds":0.0,"unlearning_seconds":0.0,"test_accessed":False};wall=time.perf_counter();trainable=total=0
    try:
        for shard in range(state["next_shard"],shards):
            local_started=time.perf_counter();model=_new_local_model(root,config,kind,device);trainable,total=model_counts(model);initial_steps=0;unlearning_steps=0;shard_dir=model_root/f"shard_{shard:02d}";shard_work=run_dir/"work"/f"shard_{shard:02d}"
            if shard_dir.exists():raise FileExistsError(shard_dir)
            if shard_work.exists():shutil.rmtree(shard_work)
            shard_work.mkdir(parents=True)
            for slice_id in sorted(partition[shard]):
                initial_steps+=_train_indices(model,dataset,partition[shard][slice_id],config,device,config["protocol"]["epochs_per_slice"]);atomic_torch_save(shard_work/f"initial_slice_{slice_id:02d}.pt",trainable_state(model,"full_model" if kind=="sisa" else "lora_adapter"))
            state["initial_training_seconds"]+=time.perf_counter()-local_started;unlearn_started=time.perf_counter();earliest=plan.get(shard)
            if earliest is not None:
                release(model);model=_new_local_model(root,config,kind,device)
                if earliest>0:load_trainable_state(model,torch.load(shard_work/f"initial_slice_{earliest-1:02d}.pt",map_location="cpu",weights_only=True),"full_model" if kind=="sisa" else "lora_adapter")
                for slice_id in range(earliest,config["protocol"].get("slices",1)):unlearning_steps+=_train_indices(model,dataset,[i for i in partition[shard][slice_id] if i not in forget],config,device,config["protocol"]["epochs_per_slice"])
            final=trainable_state(model,"full_model" if kind=="sisa" else "lora_adapter");atomic_torch_save(shard_work/"final_state.pt",final);atomic_json(shard_work/"manifest.json",{"shard":shard,"earliest_retrained_slice":earliest,"initial_optimizer_steps":initial_steps,"unlearning_optimizer_steps":unlearning_steps,"final_sha256":tensor_tree_hash(final),"test_accessed":False});atomic_publish_directory(shard_work,shard_dir);state["unlearning_seconds"]+=time.perf_counter()-unlearn_started;state["optimizer_steps"]+=initial_steps+unlearning_steps;state["next_shard"]=shard+1;atomic_json(state_path,state);release(model)
        if kind=="receraser":
            columns=[];labels=None
            for shard in range(shards):
                model=_new_local_model(root,config,kind,device);load_trainable_state(model,torch.load(model_root/f"shard_{shard:02d}"/"final_state.pt",map_location="cpu",weights_only=True),"lora_adapter");prob,current_labels,_=development_binary_predictions(model,development,device,config["protocol"]["evaluation_batch_size"]);columns.append(prob);labels=current_labels if labels is None else labels
                if current_labels!=labels:raise ValueError("RecEraser Development order mismatch")
                release(model)
            weights=fit_attention(np.asarray(columns).T,np.asarray(labels),config["protocol"]["attention_steps"],config["protocol"]["attention_learning_rate"]);aggregation={"type":"learned_development_probability_attention","weights":weights.tolist(),"fitting":"frozen Development Yes/No cross-entropy; the same weights aggregate full-vocabulary probabilities in unified evaluation","test_accessed":False}
        else:aggregation={"type":"mean_yes_no_probability","weights":[1/shards]*shards,"fitting":"fixed","test_accessed":False}
        atomic_json(run_dir/"aggregation.json",aggregation);timing={"augmentation_seconds":0.0,"initial_training_seconds":state["initial_training_seconds"],"unlearning_incremental_seconds":state["unlearning_seconds"],"training_seconds":state["initial_training_seconds"]+state["unlearning_seconds"],"end_to_end_wall_seconds":time.perf_counter()-wall,"test_accessed":False};resources={"peak_gpu_allocated":torch.cuda.max_memory_allocated(),"peak_gpu_reserved":torch.cuda.max_memory_reserved(),"cpu_rss":__import__("psutil").Process().memory_info().rss,"shared_gpu_memory_used":False,"inference_model_count":shards,"sequential_training":True};manifest=paper_model_manifest(method_id=METHODS[kind],display_name=config["display_name"],run_name=run_name,model_type="full_model_ensemble" if kind=="sisa" else "lora_adapter_ensemble",artifacts=[model_root,run_dir/"aggregation.json"],root=root,contract=contract,config_sha256=sha256_file(config_path),trainable_parameters=trainable*shards,total_parameters=total*shards,optimizer_steps=state["optimizer_steps"],timing=timing,resources=resources,completion_marker=MARKERS[kind],extra={"shards":shards,"slices":config["protocol"].get("slices",1),"affected_shards":len(plan),"retrained_samples":sum(len([i for i in partition[shard][slice_id] if i not in forget]) for shard in plan for slice_id in range(plan[shard],config["protocol"].get("slices",1))),"aggregation_cost_model_count":shards,"adaptation_disclosure":config["adaptation_disclosure"]});write_model_manifest(run_dir,manifest);atomic_json(run_dir/"timing.json",timing);state.update({"status":"COMPLETED","test_accessed":False});atomic_json(state_path,state);publish_manifest(run_dir,["models","partition.json","aggregation.json","contract.json","provenance.json","budget_audit.json","timing.json","paper_model_manifest.json"],{"schema":SCHEMAS[kind],"test_accessed":False});complete(run_dir,MARKERS[kind]);return {"status":"COMPLETED","method_id":METHODS[kind],"run_dir":str(run_dir),"test_accessed":False}
    except SafeStop as stop:return stopped_safely(run_dir,stop)
    except BaseException as error:state.update({"status":"INTERRUPTED","reason":type(error).__name__,"message":str(error),"test_accessed":False});atomic_json(state_path,state);raise


def synthetic(root:Path,config_path:Path,run_name:str,kind:str)->dict[str,Any]:
    config=load_config(config_path,SCHEMAS[kind]);validate_partition_config(config,kind);run_dir=new_run_directory(root,config["synthetic_root"],run_name);rows=[{"input":str(i),"output":"Yes."} for i in range(24)];partition=sisa_partition(rows,4,3,42);forget=[1,7];plan=deletion_plan(partition,forget);result={"method_id":METHODS[kind],"partition_complete":sum(len(v) for p in partition.values() for v in p.values())==len(rows),"affected_shards":plan,"real_t5_loaded":False,"optimizer_constructed":False,"test_accessed":False};atomic_json(run_dir/"synthetic_result.json",result);return result


def run_cli(kind:str,args:Any)->dict[str,Any]:
    root=args.project_root.resolve();config_path=args.config or root/f"configs/paper_{kind}_v1.yaml"
    if args.mode=="Preflight":return preflight(root,config_path,kind)
    if args.mode=="BudgetAudit":return partitioned_budget(load_config(config_path,SCHEMAS[kind]),kind)
    if not args.run_name:raise ValueError("RunName is required")
    if args.mode=="SyntheticDryRun":return synthetic(root,config_path,args.run_name,kind)
    if args.mode=="Analyze":return analyze_run(existing_run_directory(root,load_config(config_path,SCHEMAS[kind])["output_root"],args.run_name),MARKERS[kind])
    return execute(root,config_path,args.run_name,kind,args.mode=="Resume")
