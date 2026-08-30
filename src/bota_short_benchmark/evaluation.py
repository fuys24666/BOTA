"""Development-only aggregate evaluation for the short benchmark."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.diagnostics.ml1m_development_protocol import reconstruct_authoritative_rows
from src.diagnostics.t5_lora_influence_feasibility_audit_a2 import install_fixed_ab_coordinate
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset, load_config as load_t5_config, load_legacy_model, move_batch
from src.paper_baselines.runtime import clean_lora_model, load_trainable_state
from src.paper_if_a2.common import atomic_json, canonical_hash, git_snapshot, safe_run_name, sha256_file
from src.paper_if_a2.evaluate_v2 import _batch

from .protocol import SCHEMA, load_config, validate_prepared
from .runner import MARKER as METHOD_MARKER, METHODS
from .runner import _base_t5_config

EVAL_MARKER = "BOTA_SHORT_EVALUATION_V1_COMPLETED"
ORDER = ["Original-Short", "Retrain-Short", "IFRU-Short-LoRA", "SISA-Short-T5", "RecEraser-Adapter-Short", "BOTA-T2-Short"]


def _validate_method(root: Path, config: dict[str, Any], method_id: str, run_name: str, benchmark_name: str) -> Path:
    run = root / config["output_root"] / "models" / method_id / safe_run_name(run_name); required = {"COMPLETED", "contract.json", "manifest.json", "run_state.json", "scenarios"}
    if not run.is_dir() or {path.name for path in run.iterdir()} != required or (run / "COMPLETED").read_text(encoding="utf-8") != METHOD_MARKER + "\n": raise ValueError(f"invalid {method_id} run")
    state = json.loads((run / "run_state.json").read_text(encoding="utf-8")); manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")); contract = json.loads((run / "contract.json").read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETED" or state.get("method_id") != method_id or state.get("benchmark_name") != benchmark_name or state.get("test_accessed") is not False or sha256_file(run / "run_state.json") != manifest.get("run_state_sha256") or contract.get("test_accessed") is not False: raise ValueError(f"{method_id} run binding mismatch")
    return run


def resolve_registry(root: Path, config: dict[str, Any], benchmark_name: str, names: dict[str, str]) -> tuple[dict[str, Path], dict[str, Any]]:
    _, contract, registry = validate_prepared(root, config, benchmark_name); expected = {METHODS[key] for key in ("Original", "Retrain", "IFRU", "SISA", "RecEraser", "BOTA")}
    if set(names) != expected or any(not value for value in names.values()): raise ValueError("all six frozen short methods are required")
    runs = {method: _validate_method(root, config, method, names[method], benchmark_name) for method in ORDER}
    return runs, registry


def _load_fixed(root: Path, config: dict[str, Any], scenario_run: Path, device: torch.device):
    state = torch.load(scenario_run / "adapter/adapter_model.pt", map_location="cpu", weights_only=True); model = load_legacy_model(root / config["source"]["original_checkpoint"]); names, parameters = install_fixed_ab_coordinate(model, state["A"], 32)
    with torch.no_grad():
        for name, parameter in zip(names, parameters): parameter.copy_(state["B"][name].to(parameter))
    return model.to(device).eval()


def _release(model):
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def _single_predictions(model, dataset, indices: Sequence[int], device: torch.device, batch_size: int) -> dict[str, Any]:
    probabilities = []; labels = []; losses = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = list(indices[start:start+batch_size]); cpu = _batch(dataset, selected); value = move_batch(cpu, device); logits = model(input_ids=value["input_ids"], labels=value["target_ids"]).logits; probability = F.softmax(logits, -1); targets = value["target_ids"]; mask = targets.ne(-100); yes_no = probability[:, 0, [465, 2163]]; yes = yes_no[:, 1] / yes_no.sum(-1); gathered = torch.gather(probability, 2, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1); answer = (-(gathered.clamp_min(1e-30).log()) * mask).sum(-1) / mask.sum(-1); probabilities.extend(yes.cpu().tolist()); labels.extend((targets[:, 0] == 2163).int().cpu().tolist()); losses.extend(answer.cpu().tolist()); del cpu, value, logits, probability
    return {"probability": probabilities, "gold_label": labels, "answer_loss": losses, "sample_order_sha256": canonical_hash(list(indices))}


def _ensemble_predictions(root: Path, config: dict[str, Any], scenario_manifest: dict[str, Any], dataset, indices, device, batch_size):
    component = Path(scenario_manifest["component_run"]); aggregation = json.loads((component / "aggregation.json").read_text(encoding="utf-8")); weights = list(map(float, aggregation["weights"])); n = len(indices); yes = torch.zeros(n, dtype=torch.float64); no = torch.zeros(n, dtype=torch.float64); target = None; masks = None; labels = None
    base = _base_t5_config(root, config)
    for shard, weight in enumerate(weights):
        if scenario_manifest["method_id"] == "SISA-Short-T5": model = T5ForConditionalGeneration.from_pretrained(Path(base["paths"]["model_dir"]), attn_implementation="eager").float(); kind = "full_model"
        else: model = clean_lora_model(root / config["source"]["original_checkpoint"], {"r":16,"lora_alpha":32,"lora_dropout":.05,"target_modules":["q","v"]}, torch.device("cpu")); kind = "lora_adapter"
        state = torch.load(component / f"models/shard_{shard:02d}/final_state.pt", map_location="cpu", weights_only=True); load_trainable_state(model, state, kind); model = model.to(device).eval(); offset = 0
        with torch.inference_mode():
            for start in range(0, n, batch_size):
                selected = list(indices[start:start+batch_size]); cpu = _batch(dataset, selected); value = move_batch(cpu, device); probability = F.softmax(model(input_ids=value["input_ids"], labels=value["target_ids"]).logits, -1); current = len(selected); yes[offset:offset+current].add_(probability[:,0,2163].double().cpu(), alpha=weight); no[offset:offset+current].add_(probability[:,0,465].double().cpu(), alpha=weight); gathered = torch.gather(probability,2,value["target_ids"].clamp_min(0).unsqueeze(-1)).squeeze(-1).double().cpu(); mask = value["target_ids"].ne(-100).cpu()
                if target is None: target = torch.zeros((n, gathered.shape[1]), dtype=torch.float64); masks = torch.zeros((n, gathered.shape[1]), dtype=torch.bool); labels = torch.zeros(n, dtype=torch.int64)
                target[offset:offset+current].add_(gathered, alpha=weight); masks[offset:offset+current] = mask; labels[offset:offset+current] = (value["target_ids"][:,0] == 2163).long().cpu(); offset += current; del cpu, value, probability, gathered
        _release(model)
    probability = yes / (yes + no).clamp_min(1e-30); answer = (-(target.clamp_min(1e-30).log()) * masks).sum(-1) / masks.sum(-1)
    return {"probability": probability.tolist(), "gold_label": labels.tolist(), "answer_loss": answer.tolist(), "sample_order_sha256": canonical_hash(list(indices))}


def _predict(root, config, run, method_id, scenario, dataset, indices, device):
    scenario_run = run / "scenarios" / scenario; manifest = json.loads((scenario_run / "scenario_manifest.json").read_text(encoding="utf-8"))
    if manifest["method_id"] != method_id or manifest["test_accessed"] is not False: raise ValueError("scenario manifest mismatch")
    if manifest["model_type"] == "if_a2_fixed_ab":
        model = _load_fixed(root, config, scenario_run, device)
        try: return _single_predictions(model, dataset, indices, device, config["evaluation"]["inference_batch_size"])
        finally: _release(model)
    return _ensemble_predictions(root, config, manifest, dataset, indices, device, config["evaluation"]["inference_batch_size"])


def bernoulli_jsd(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = np.stack([1-left, left], axis=1).clip(1e-12, 1); q = np.stack([1-right, right], axis=1).clip(1e-12, 1); m = (p+q)/2
    return .5*np.sum(p*np.log(p/m), axis=1)+.5*np.sum(q*np.log(q/m), axis=1)


def _metrics(method: str, prediction: dict[str, Any], retrain: dict[str, Any], original: dict[str, Any], request_indices: Sequence[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    p = np.asarray(prediction["probability"], dtype=np.float64); r = np.asarray(retrain["probability"], dtype=np.float64); o = np.asarray(original["probability"], dtype=np.float64); y = np.asarray(prediction["gold_label"], dtype=np.int64); loss = np.asarray(prediction["answer_loss"], dtype=np.float64); selected = np.asarray(request_indices, dtype=np.int64); ps, rs, os = p[selected], r[selected], o[selected]; distance = np.abs(ps-rs); baseline = np.abs(os-rs); denominator = float(np.mean(baseline)); residual = float(np.mean(distance)/denominator) if denominator else (0. if np.all(distance==0) else None); toward = int(np.sum(distance < baseline-1e-15)); away = int(np.sum(distance > baseline+1e-15)); equal = len(selected)-toward-away
    row = {"method_id": method, "overall_auc": float(roc_auc_score(y,p)), "overall_acc": float(accuracy_score(y,p>=.5)), "overall_log_loss": float(np.mean(-(y*np.log(p.clip(1e-12,1))+(1-y)*np.log((1-p).clip(1e-12,1))))), "mean_answer_loss": float(np.mean(loss)), "forget_samples": len(selected), "yes_no_jsd_to_retrain": float(np.mean(bernoulli_jsd(ps,rs))), "probability_l2_rms_to_retrain": float(np.sqrt(np.mean((ps-rs)**2))), "probability_mae_to_retrain": float(np.mean(distance)), "residual_ratio_to_retrain": residual, "toward_retrain": toward, "away_from_retrain": away, "equivalent": equal, "sign_agreement": float(np.mean(np.sign(ps-os)==np.sign(rs-os))), "mcc": float(matthews_corrcoef(np.sign(rs-os),np.sign(ps-os))) if len(set(np.sign(rs-os)))>1 and len(set(np.sign(ps-os)))>1 else None, "prediction_collapse": bool(np.std(p)<1e-6), "finite": bool(np.isfinite(p).all() and np.isfinite(loss).all())}
    samples = [{"method_id":method,"development_index":int(index),"gold_label":int(y[index]),"probability":float(p[index]),"retrain_probability":float(r[index]),"original_probability":float(o[index]),"absolute_distance_to_retrain":float(abs(p[index]-r[index])),"yes_no_jsd_to_retrain":float(bernoulli_jsd(np.asarray([p[index]]),np.asarray([r[index]]))[0])} for index in selected]
    return row, samples


def execute(root: Path, config_path: Path, benchmark_name: str, run_name: str, names: dict[str,str]) -> dict[str, Any]:
    config = load_config(config_path); runs, registry = resolve_registry(root, config, benchmark_name, names); git = git_snapshot(root)
    if not git["clean"]: raise RuntimeError("formal short evaluation requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("exactly one CUDA GPU required")
    run_name=safe_run_name(run_name); destination=root/config["output_root"]/"evaluations"/run_name
    if destination.exists(): raise FileExistsError(destination)
    stage=destination.parent/".work"/f"{run_name}.{uuid.uuid4().hex}.stage";stage.mkdir(parents=True);device=torch.device("cuda:0");base=_base_t5_config(root,config);tokenizer=T5Tokenizer.from_pretrained(base["paths"]["model_dir"]);dataset=JsonPromptDataset(root/config["source"]["development_json"],tokenizer)
    if "development_user_ids" in config["source"]: dev_users=json.loads((root/config["source"]["development_user_ids"]).read_text(encoding="utf-8"))
    else:
        _,development,_=reconstruct_authoritative_rows(root/config["source"]["raw_data"]);dev_users=[int(row.authoritative_user_id) for row in development]
    if len(dataset)!=registry["development_samples"] or len(dev_users)!=registry["development_samples"]: raise RuntimeError("Development count mismatch")
    all_metrics=[];all_samples=[];started=time.perf_counter()
    try:
        for scenario in registry["scenarios"]:
            indices=list(range(len(dataset))); request=set(map(int,scenario["users"])); request_indices=[index for index,user in enumerate(dev_users) if user in request]
            minimum_support = config["protocol"]["minimum_development_samples_per_user"] * len(request)
            if len(request_indices)<minimum_support: raise RuntimeError(f"scenario {scenario['id']} has insufficient Development support")
            predictions={}
            for method in ORDER: predictions[method]=_predict(root,config,runs[method],method,scenario["id"],dataset,indices,device)
            reference=predictions["Retrain-Short"];original=predictions["Original-Short"]
            if any(predictions[method]["gold_label"]!=reference["gold_label"] or predictions[method]["sample_order_sha256"]!=reference["sample_order_sha256"] for method in ORDER):raise RuntimeError("Development label/order mismatch")
            for method in ORDER:
                row,samples=_metrics(method,predictions[method],reference,original,request_indices);all_metrics.append({"scenario":scenario["id"],"composition":scenario["composition"],**row});all_samples.extend({"scenario":scenario["id"],**sample} for sample in samples)
        atomic_json(stage/"metrics.json",{"schema":SCHEMA,"benchmark_name":benchmark_name,"metrics":all_metrics,"test_accessed":False});
        with (stage/"metrics.csv").open("w",encoding="utf-8",newline="") as handle:
            fields=sorted({key for row in all_metrics for key in row});writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(all_metrics)
        with (stage/"per_sample_metrics.jsonl").open("w",encoding="utf-8",newline="\n") as handle:
            for row in all_samples:handle.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
        report=["# BOTA short-step benchmark v1","","Development-only; no FinalTest or MIA was accessed.",""]
        scenario_ids = [row["id"] for row in registry["scenarios"]]
        for scenario in scenario_ids:
            report.extend([f"## {scenario}","","| Method | AUC | LogLoss | Residual→Retrain | Toward/Away |","|---|---:|---:|---:|---:|"])
            for row in [item for item in all_metrics if item["scenario"]==scenario]:report.append(f"| {row['method_id']} | {row['overall_auc']:.6f} | {row['overall_log_loss']:.6f} | {row['residual_ratio_to_retrain']:.6f} | {row['toward_retrain']}/{row['away_from_retrain']} |")
            report.append("")
        (stage/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8",newline="\n");state={"schema":SCHEMA,"status":"COMPLETED","benchmark_name":benchmark_name,"run_name":run_name,"models":ORDER,"scenarios":scenario_ids,"wall_time_seconds":time.perf_counter()-started,"test_accessed":False};atomic_json(stage/"run_state.json",state);atomic_json(stage/"provenance.json",{"schema":SCHEMA,"git":git,"registry_sha256":registry["registry_sha256"],"model_runs":{method:str(path.resolve()) for method,path in runs.items()},"split":"Development","final_test_accessed":False,"test_accessed":False});(stage/"COMPLETED").write_text(EVAL_MARKER+"\n",encoding="utf-8",newline="\n");atomic_json(stage/"manifest.json",{"schema":SCHEMA,"files":{name:sha256_file(stage/name) for name in ("metrics.json","metrics.csv","per_sample_metrics.jsonl","report.md","run_state.json","provenance.json","COMPLETED")},"published_atomically":True,"test_accessed":False});destination.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,destination);return {"status":"COMPLETED","run_dir":str(destination),"models":ORDER,"scenarios":scenario_ids,"split":"Development","test_accessed":False}
    finally:
        if stage.exists():__import__("shutil").rmtree(stage)
        gc.collect();torch.cuda.empty_cache()


def analyze(root:Path,config_path:Path,run_name:str)->dict[str,Any]:
    config=load_config(config_path);run=root/config["output_root"]/"evaluations"/safe_run_name(run_name);required={"COMPLETED","manifest.json","metrics.csv","metrics.json","per_sample_metrics.jsonl","provenance.json","report.md","run_state.json"}
    if not run.is_dir() or {path.name for path in run.iterdir()}!=required or (run/"COMPLETED").read_text(encoding="utf-8")!=EVAL_MARKER+"\n":raise ValueError("invalid short evaluation")
    manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"));
    for name,expected in manifest["files"].items():
        if sha256_file(run/name)!=expected:raise ValueError(f"evaluation artifact mismatch: {name}")
    state=json.loads((run/"run_state.json").read_text(encoding="utf-8"));return {"status":"COMPLETED","run_dir":str(run),"models":state["models"],"scenarios":state["scenarios"],"test_accessed":False}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=Path.cwd());parser.add_argument("--config",type=Path,default=Path("configs/bota_short_benchmark_v1.yaml"));parser.add_argument("--mode",choices=["Preflight","SyntheticDryRun","Full","Analyze"],default="Preflight");parser.add_argument("--benchmark-name",default="");parser.add_argument("--run-name",default="");
    for key in ("original","retrain","ifru","sisa","receraser","bota"):parser.add_argument(f"--{key}-run-name",default="")
    args=parser.parse_args();root=args.root.resolve();cp=(root/args.config).resolve() if not args.config.is_absolute() else args.config.resolve();names={METHODS["Original"]:args.original_run_name,METHODS["Retrain"]:args.retrain_run_name,METHODS["IFRU"]:args.ifru_run_name,METHODS["SISA"]:args.sisa_run_name,METHODS["RecEraser"]:args.receraser_run_name,METHODS["BOTA"]:args.bota_run_name}
    if args.mode=="SyntheticDryRun":result={"schema":SCHEMA,"models":ORDER,"scenarios":[row["id"] for row in load_config(cp)["protocol"]["scenarios"]],"real_model_loaded":False,"test_accessed":False}
    elif args.mode=="Preflight":result={"schema":SCHEMA,"mode":"Preflight","benchmark_name":args.benchmark_name,"models":list(names),"split":"Development","model_loaded":False,"test_accessed":False}
    elif not args.run_name:parser.error(f"{args.mode} requires RunName")
    elif args.mode=="Analyze":result=analyze(root,cp,args.run_name)
    else:result=execute(root,cp,args.benchmark_name,args.run_name,names)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":main()
