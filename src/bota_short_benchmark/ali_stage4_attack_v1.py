"""Preregistered Development-only matched-user MIA verification (Ali Stage 4)."""
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
from typing import Any, Callable, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, Subset
from transformers import T5Tokenizer

from src.bota_if import p1_trajectory_transport_audit as p1
from src.bota_short_benchmark import e2urec_short_v1 as e2
from src.bota_short_benchmark import evaluation as short_eval
from src.bota_short_benchmark import paper_evaluation_v2 as paper_eval
from src.bota_short_benchmark import paper_v2, runner
from src.bota_short_benchmark.protocol import load_config as load_benchmark_config
from src.bota_short_benchmark.protocol import validate_prepared
from src.diagnostics.ml1m_development_protocol import reconstruct_authoritative_rows
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from src.diagnostics.t5_zero_training_analysis import ATTACKS, attack_metrics, macro_user_auc, matched_user_cluster_bootstrap
from src.paper_if_a2.common import atomic_json, canonical_hash, git_snapshot, safe_run_name, sha256_file

SCHEMA = "bota-ali-stage4-attack-v1"
MARKER = "BOTA_ALI_STAGE4_ATTACK_V1_COMPLETED"
SUMMARY_SCHEMA = "bota-ali-stage4-attack-summary-v1"
SUMMARY_MARKER = "BOTA_ALI_STAGE4_ATTACK_SUMMARY_V1_COMPLETED"
SEEDS = (41, 42, 43)
SCENARIOS = ("L8", "L4M4")
DISPLAY = ("Original", "Exact-Masked", "BOTA", "IFRU", "E2URec", "NegGrad", "PCGrad", "SISA", "RecEraser", "Retain-Retrain-P5")
PAPER_NAMES = {
    "Original-Short-200step": "bota_short_original_seed{seed}_v3",
    "Exact-Masked-Reference-200step": "bota_short_exact_masked_seed{seed}_v3",
    "FullControl-P5-Short": "bota_short_full_control_p5_seed{seed}_v3",
    "Retain-Retrain-P5-Short": "bota_short_retain_p5_seed{seed}_v3",
    "BOTA-T2-Short": "bota_short_bota_seed{seed}_v3",
    "IFRU-Short-LoRA": "bota_short_ifru_seed{seed}_v3",
    "NegGrad-Mixed-Short-BOnly": "bota_short_neggrad_seed{seed}_v3",
    "PCGrad-Short-BOnly": "bota_short_pcgrad_seed{seed}_v3",
    "SISA-Short-T5": "bota_short_sisa_seed{seed}_v3",
    "RecEraser-Adapter-Short": "bota_short_receraser_seed{seed}_v3",
}
PAPER_TO_DISPLAY = {
    "Original-Short-200step": "Original", "Exact-Masked-Reference-200step": "Exact-Masked",
    "Retain-Retrain-P5-Short": "Retain-Retrain-P5", "BOTA-T2-Short": "BOTA",
    "IFRU-Short-LoRA": "IFRU", "NegGrad-Mixed-Short-BOnly": "NegGrad",
    "PCGrad-Short-BOnly": "PCGrad", "SISA-Short-T5": "SISA", "RecEraser-Adapter-Short": "RecEraser",
}
RUN_REQUIRED = {"COMPLETED", "manifest.json", "metrics.csv", "metrics.json", "per_sample_metrics.jsonl", "provenance.json", "report.md", "run_state.json"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_manifest(directory: Path, expected_hash: str) -> None:
    manifest_path = directory / "manifest.json"
    if sha256_file(manifest_path) != expected_hash: raise ValueError(f"frozen manifest changed: {manifest_path}")
    manifest = _json(manifest_path)
    if manifest.get("test_accessed") is not False: raise ValueError("frozen evaluation accessed FinalTest")
    for name, digest in manifest.get("files", {}).items():
        if sha256_file(directory / name) != digest: raise ValueError(f"frozen artifact changed: {directory / name}")


def load_config(repo: Path, workspace: Path, path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema", "test_access_policy", "parent_preregistration", "stage3_result_audit", "benchmark_configs", "paper_configs", "frozen_development_evaluations", "output_root", "protocol", "evaluation", "decision", "runtime"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != SCHEMA or value["test_access_policy"] != "forbidden": raise ValueError("invalid Stage 4 configuration")
    for field in ("parent_preregistration", "stage3_result_audit"):
        binding = value[field]; source = repo / binding["path"]
        if sha256_file(source) != binding["sha256"]: raise ValueError(f"{field} binding changed")
    prereg = _json(repo / value["parent_preregistration"]["path"])["attack_verification"]
    expected = value["evaluation"]
    if (prereg["primary_score"] != value["protocol"]["primary_score"] or prereg["secondary_scores"] != value["protocol"]["secondary_scores"]
        or prereg["bootstrap"]["resamples"] != expected["bootstrap_resamples"] or prereg["bootstrap"]["seed"] != expected["bootstrap_seed"]): raise ValueError("Stage 4 differs from parent attack preregistration")
    if tuple(value["protocol"]["seeds"]) != SEEDS or tuple(value["protocol"]["scenarios"]) != SCENARIOS or tuple(value["protocol"]["models"]) != DISPLAY: raise ValueError("Stage 4 population changed")
    for seed in SEEDS:
        for group in ("benchmark_configs", "paper_configs"):
            binding = value[group][seed]; source = repo / binding["path"]
            if sha256_file(source) != binding["sha256"]: raise ValueError(f"{group}/{seed} binding changed")
            workspace_copy = workspace / binding["path"]
            if not workspace_copy.is_file() or sha256_file(workspace_copy) != binding["sha256"]: raise ValueError(f"workspace config differs: {workspace_copy}")
        frozen = value["frozen_development_evaluations"][seed]
        _verify_manifest(workspace / frozen["path"], frozen["manifest_sha256"])
    frozen = value["frozen_development_evaluations"]["e2urec"]
    _verify_manifest(workspace / frozen["path"], frozen["manifest_sha256"])
    return value


def _model_bindings(workspace: Path, config: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], Path]:
    paper_config = paper_v2.load_config(workspace / config["paper_configs"][seed]["path"])
    names = {key: value.format(seed=seed) for key, value in PAPER_NAMES.items()}
    benchmark_name = f"bota_short_i02_seed{seed}_v3"
    v1, registry, runs = paper_eval.resolve(workspace, paper_config, benchmark_name, names)
    e2run = e2._model_run(workspace, seed, f"bota_short_e2urec_seed{seed}_v1")
    return paper_config, v1, runs, e2run


def preflight(repo: Path, workspace: Path, config_path: Path, seed: int) -> dict[str, Any]:
    config = load_config(repo, workspace, config_path)
    paper_config, v1, runs, e2run = _model_bindings(workspace, config, seed)
    _, contract, registry = validate_prepared(workspace, v1, f"bota_short_i02_seed{seed}_v3")
    if contract.get("test_accessed") is not False or registry.get("test_accessed") is not False: raise ValueError("prepared protocol accessed FinalTest")
    if {row["id"] for row in registry["scenarios"]} != set(SCENARIOS): raise ValueError("scenario registry changed")
    train_users,_=p1._train_user_ids_only(workspace/v1["source"]["raw_data"])
    _,development,_=reconstruct_authoritative_rows(workspace/v1["source"]["raw_data"]); dev_users=[int(row.authoritative_user_id) for row in development]
    populations={}
    for scenario in registry["scenarios"]:
        requested=set(map(int,scenario["users"])); members=[int(train_users[index]) for index in scenario["forget_train_indices"]]; nonmembers=[user for user in dev_users if user in requested]
        if set(members)!=requested or set(nonmembers)!=requested or len(members)!=len(requested): raise ValueError(f"invalid matched-user population: {scenario['id']}")
        populations[scenario["id"]]={"member_samples":len(members),"nonmember_samples":len(nonmembers),"matched_users":len(requested)}
    return {"schema": SCHEMA, "status": "READY", "seed": seed, "models": list(DISPLAY), "scenarios": list(SCENARIOS), "matched_user_populations":populations,"bound_model_runs": len(runs) + 1, "e2urec_run": str(e2run), "model_loaded": False, "final_test_accessed": False, "test_accessed": False}


def _row_from_prediction(prediction: dict[str, Any], index: int, user: int) -> dict[str, Any]:
    p = float(prediction["probability"][index]); loss = float(prediction["answer_loss"][index])
    p = min(max(p, 1e-12), 1 - 1e-12)
    return {"user_id": int(user), "answer_sequence_loss": loss, "confidence": max(p, 1-p), "binary_entropy": float(-(p*math.log(p)+(1-p)*math.log(1-p))), "yes_no_margin": 2*p-1}


def _arrays(member: list[dict[str, Any]], nonmember: list[dict[str, Any]], score: Callable[[dict[str, Any]], float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.asarray([score(row) for row in member]), np.asarray([score(row) for row in nonmember]), np.asarray([row["user_id"] for row in member]), np.asarray([row["user_id"] for row in nonmember]))


def _attack(member: list[dict[str, Any]], nonmember: list[dict[str, Any]], score: Callable[[dict[str, Any]], float], seed: int, resamples: int) -> dict[str, Any]:
    m, n, mu, nu = _arrays(member, nonmember, score)
    if set(mu) != set(nu): raise ValueError("strict matched-user population violated")
    return {"pooled": attack_metrics(m, n), "matched_user_bootstrap": matched_user_cluster_bootstrap(m, mu, n, nu, seed=seed, resamples=resamples), "macro_user_auc": macro_user_auc(m, mu, n, nu, seed=seed, resamples=resamples)}


def _paired_bootstrap(caches: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]], attack_name: str, seed: int, resamples: int) -> dict[str, Any]:
    score = ATTACKS[attack_name]; arrays = {name: _arrays(*rows, score) for name, rows in caches.items()}
    users = sorted(set(arrays["Original"][2]) & set(arrays["Original"][3]))
    if any(set(value[2]) != set(users) or set(value[3]) != set(users) for value in arrays.values()): raise ValueError("paired attack user population differs")
    def auc(name: str, sampled: Sequence[int]) -> float:
        m,n,mu,nu=arrays[name]; sm=np.concatenate([m[mu==u] for u in sampled]); sn=np.concatenate([n[nu==u] for u in sampled]); return float(roc_auc_score(np.r_[np.ones(len(sm)),np.zeros(len(sn))],np.r_[sm,sn]))
    points={name:attack_metrics(value[0],value[1])["roc_auc"] for name,value in arrays.items()}
    reference_point=abs(points["Original"]-.5)-abs(points["Exact-Masked"]-.5)
    improvement_point=abs(points["Original"]-points["Exact-Masked"])-abs(points["BOTA"]-points["Exact-Masked"])
    rng=np.random.default_rng(seed); reference=[]; improvement=[]
    for _ in range(resamples):
        sampled=list(map(int,rng.choice(users,size=len(users),replace=True))); original=auc("Original",sampled); exact=auc("Exact-Masked",sampled); bota=auc("BOTA",sampled)
        reference.append(abs(original-.5)-abs(exact-.5)); improvement.append(abs(original-exact)-abs(bota-exact))
    def result(point: float, values: list[float]) -> dict[str, Any]:
        return {"point_estimate":point,"percentile_95_ci":[float(x) for x in np.percentile(values,[2.5,97.5])],"bootstrap_unit":"authoritative_user","resamples":resamples,"seed":seed}
    return {"reference_signal":result(reference_point,reference),"bota_gap_improvement":result(improvement_point,improvement),"endpoint_auc":points}


def _utility(workspace: Path, config: dict[str, Any], seed: int, scenario: str) -> dict[str, dict[str, Any]]:
    frozen = workspace / config["frozen_development_evaluations"][seed]["path"]
    paper_rows = _json(frozen / "metrics.json")["metrics"]
    output={PAPER_TO_DISPLAY[row["method_id"]]:row for row in paper_rows if row["scenario"]==scenario and row["method_id"] in PAPER_TO_DISPLAY}
    e2rows=_json(workspace / config["frozen_development_evaluations"]["e2urec"]["path"] / "metrics.json")["rows"]
    output["E2URec"]=next(row for row in e2rows if int(row["seed"])==seed and row["scenario"]==scenario)
    if set(output)!=set(DISPLAY): raise ValueError("frozen utility rows incomplete")
    original=output["Original"]; bota=output["BOTA"]; decision=config["decision"]
    guard={"auc_damage":float(original["overall_auc"]-bota["overall_auc"]),"logloss_damage":float(bota["overall_log_loss"]-original["overall_log_loss"])}
    guard["pass"]=guard["auc_damage"]<=decision["overall_auc_damage_max"] and guard["logloss_damage"]<=decision["overall_logloss_damage_max"] and not bool(bota.get("prediction_collapse",False))
    output["_bota_guardrail"]=guard
    return output


def execute(repo: Path, workspace: Path, config_path: Path, seed: int, run_name: str) -> dict[str, Any]:
    config=load_config(repo,workspace,config_path); git=git_snapshot(repo)
    if not git["clean"]: raise RuntimeError("formal Stage 4 execution requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("exactly one CUDA GPU required")
    free_bytes,_=torch.cuda.mem_get_info(0)
    if free_bytes/2**30 < config["runtime"]["minimum_free_gib"]: raise RuntimeError("insufficient free GPU memory")
    paper_config,v1,runs,e2run=_model_bindings(workspace,config,seed); protocol_dir,_,registry=validate_prepared(workspace,v1,f"bota_short_i02_seed{seed}_v3")
    destination=workspace/config["output_root"] / "runs" / safe_run_name(run_name)
    if destination.exists(): raise FileExistsError(destination)
    stage=destination.parent/".work"/f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage"; stage.mkdir(parents=True)
    try:
        device=torch.device("cuda:0"); base=runner._base_t5_config(workspace,v1)
        model_dir=Path(base["paths"]["model_dir"])
        if not model_dir.is_absolute(): model_dir=workspace/model_dir
        if not model_dir.is_dir(): raise FileNotFoundError(f"local tokenizer directory not found: {model_dir}")
        tokenizer=T5Tokenizer.from_pretrained(model_dir,local_files_only=True)
        development=JsonPromptDataset(workspace/v1["source"]["development_json"],tokenizer)
        train_users,_=p1._train_user_ids_only(workspace/v1["source"]["raw_data"]); _,dev_lineage,_=reconstruct_authoritative_rows(workspace/v1["source"]["raw_data"]); dev_users=[int(row.authoritative_user_id) for row in dev_lineage]
        all_metrics=[]; safe_samples=[]; decisions=[]; started=time.perf_counter()
        for scenario_number,scenario in enumerate(registry["scenarios"]):
            sid=scenario["id"]; users=list(map(int,scenario["users"])); member_users=[int(train_users[i]) for i in scenario["forget_train_indices"]]
            if set(member_users)!=set(users) or len(set(member_users))!=len(member_users): raise ValueError("member lineage differs from registered users")
            nonmember_indices=[i for i,u in enumerate(dev_users) if int(u) in set(users)]; nonmember_users=[dev_users[i] for i in nonmember_indices]
            if set(nonmember_users)!=set(users): raise ValueError("Development lacks a registered user")
            forget=JsonPromptDataset(protocol_dir/"data"/sid/"forget.json",tokenizer); dataset=ConcatDataset([forget,Subset(development,nonmember_indices)]); indices=list(range(len(dataset)))
            predictions={}
            for paper_name in paper_eval.ORDER:
                if paper_name=="FullControl-P5-Short": continue
                display=PAPER_TO_DISPLAY[paper_name]; predictions[display]=paper_eval._predict(workspace,paper_config,v1,runs[paper_name],paper_name,sid,dataset,indices,device)
            scenario_run=e2run/"scenarios"/sid; model=short_eval._load_fixed(workspace,v1,scenario_run,device)
            try: predictions["E2URec"]=short_eval._single_predictions(model,dataset,indices,device,config["evaluation"]["inference_batch_size"])
            finally: short_eval._release(model)
            authority=predictions["Exact-Masked"]
            if any(p["gold_label"]!=authority["gold_label"] or p["sample_order_sha256"]!=authority["sample_order_sha256"] for p in predictions.values()): raise ValueError("attack sample order differs")
            caches={}
            split=len(member_users)
            for method,prediction in predictions.items():
                members=[_row_from_prediction(prediction,i,member_users[i]) for i in range(split)]
                nonmembers=[_row_from_prediction(prediction,split+i,nonmember_users[i]) for i in range(len(nonmember_users))]
                caches[method]=(members,nonmembers)
                for attack_number,(attack_name,score) in enumerate(ATTACKS.items()):
                    result=_attack(members,nonmembers,score,config["evaluation"]["bootstrap_seed"]+seed*100+scenario_number*10+attack_number,config["evaluation"]["bootstrap_resamples"])
                    all_metrics.append({"seed":seed,"scenario":sid,"method":method,"attack":attack_name,"role":"sensitivity" if method=="Retain-Retrain-P5" else "primary_comparison","member_samples":len(members),"nonmember_samples":len(nonmembers),**result})
                for role,rows in (("member",members),("nonmember",nonmembers)):
                    for i,row in enumerate(rows): safe_samples.append({"seed":seed,"scenario":sid,"method":method,"role":role,"user_hash":canonical_hash([seed,sid,row["user_id"]]),"ordinal":i,"answer_sequence_loss":row["answer_sequence_loss"],"confidence":row["confidence"],"binary_entropy":row["binary_entropy"],"yes_no_margin":row["yes_no_margin"]})
            paired=_paired_bootstrap({key:caches[key] for key in ("Original","Exact-Masked","BOTA")},config["protocol"]["primary_score"],config["evaluation"]["bootstrap_seed"]+seed*1000+scenario_number,config["evaluation"]["bootstrap_resamples"])
            utility=_utility(workspace,config,seed,sid); ref=paired["reference_signal"]; imp=paired["bota_gap_improvement"]
            ref_pass=ref["point_estimate"]>=config["decision"]["reference_signal_minimum_point_difference"] and ref["percentile_95_ci"][0]>0
            imp_pass=imp["point_estimate"]>0 and imp["percentile_95_ci"][0]>0
            guardrail=utility.pop("_bota_guardrail")
            decisions.append({"seed":seed,"scenario":sid,"primary_attack":config["protocol"]["primary_score"],"paired_bootstrap":paired,"reference_signal_pass":ref_pass,"bota_gap_improvement_pass":imp_pass,"utility_guardrail":guardrail,"utility":utility,"primary_cell_pass":bool(ref_pass and imp_pass and guardrail["pass"])})
        atomic_json(stage/"metrics.json",{"schema":SCHEMA,"metrics":all_metrics,"decisions":decisions,"all_attacks_reported":True,"retain_retrain_is_sensitivity":True,"final_test_accessed":False,"test_accessed":False})
        flat=[]
        for row in all_metrics:
            flat.append({"seed":row["seed"],"scenario":row["scenario"],"method":row["method"],"attack":row["attack"],"role":row["role"],"pooled_roc_auc":row["pooled"]["roc_auc"],"macro_user_auc":row["macro_user_auc"]["mean"],"ci_lower":row["matched_user_bootstrap"]["pooled_auc"]["percentile_95_ci"][0],"ci_upper":row["matched_user_bootstrap"]["pooled_auc"]["percentile_95_ci"][1]})
        with (stage/"metrics.csv").open("w",encoding="utf-8",newline="") as h: w=csv.DictWriter(h,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
        with (stage/"per_sample_metrics.jsonl").open("w",encoding="utf-8",newline="\n") as h:
            for row in safe_samples:h.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
        report=["# Ali Stage 4 matched-user MIA verification","","Development only; FinalTest was not constructed. Retain-Retrain-P5 is reported only as a sensitivity endpoint.",""]
        for row in decisions:
            p=row["paired_bootstrap"];report.extend([f"## {row['scenario']}","",f"Reference signal: {p['reference_signal']['point_estimate']:.4f} [{p['reference_signal']['percentile_95_ci'][0]:.4f}, {p['reference_signal']['percentile_95_ci'][1]:.4f}] — {'PASS' if row['reference_signal_pass'] else 'FAIL'}.",f"BOTA gap improvement: {p['bota_gap_improvement']['point_estimate']:.4f} [{p['bota_gap_improvement']['percentile_95_ci'][0]:.4f}, {p['bota_gap_improvement']['percentile_95_ci'][1]:.4f}] — {'PASS' if row['bota_gap_improvement_pass'] else 'FAIL'}.",f"Utility guardrail: {'PASS' if row['utility_guardrail']['pass'] else 'FAIL'}; primary cell: {'PASS' if row['primary_cell_pass'] else 'FAIL'}.",""])
        (stage/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8",newline="\n")
        peak=torch.cuda.max_memory_reserved()/2**30
        if peak>config["runtime"]["hard_peak_reserved_gib"]: raise RuntimeError("GPU hard cap exceeded")
        atomic_json(stage/"run_state.json",{"schema":SCHEMA,"status":"COMPLETED","seed":seed,"scenarios":list(SCENARIOS),"models":list(DISPLAY),"attacks":list(ATTACKS),"wall_time_seconds":time.perf_counter()-started,"peak_reserved_gib":peak,"test_accessed":False})
        atomic_json(stage/"provenance.json",{"schema":SCHEMA,"git":git,"config_sha256":sha256_file(config_path),"registered_protocol":str(protocol_dir),"model_runs":{PAPER_TO_DISPLAY[k]:str(v) for k,v in runs.items() if k in PAPER_TO_DISPLAY}|{"E2URec":str(e2run)},"split":"Development","final_test_loader_built":False,"test_accessed":False})
        (stage/"COMPLETED").write_text(MARKER+"\n",encoding="utf-8",newline="\n")
        atomic_json(stage/"manifest.json",{"schema":SCHEMA,"files":{name:sha256_file(stage/name) for name in ("COMPLETED","metrics.csv","metrics.json","per_sample_metrics.jsonl","provenance.json","report.md","run_state.json")},"published_atomically":True,"test_accessed":False})
        destination.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,destination)
        return {"status":"COMPLETED","run_dir":str(destination),"seed":seed,"cells":2,"models":10,"attacks":4,"test_accessed":False}
    except Exception as error:
        shutil.rmtree(stage,ignore_errors=True);raise RuntimeError("Stage 4 attack run failed before atomic publication") from error
    finally:
        gc.collect();torch.cuda.empty_cache()


def _validate_run(config: dict[str, Any], workspace: Path, seed: int, run_name: str) -> dict[str, Any]:
    run=workspace/config["output_root"]/"runs"/safe_run_name(run_name)
    if not run.is_dir() or {p.name for p in run.iterdir()}!=RUN_REQUIRED or (run/"COMPLETED").read_text(encoding="utf-8")!=MARKER+"\n":raise ValueError("invalid Stage 4 run")
    manifest=_json(run/"manifest.json")
    for name,digest in manifest["files"].items():
        if sha256_file(run/name)!=digest:raise ValueError(f"Stage 4 artifact changed: {name}")
    value=_json(run/"metrics.json")
    if len(value["metrics"])!=80 or len(value["decisions"])!=2 or value.get("test_accessed") is not False:raise ValueError("Stage 4 completeness invariant failed")
    return {"run":run,"metrics":value}


def summarize(repo: Path, workspace: Path, config_path: Path, summary_name: str) -> dict[str, Any]:
    config=load_config(repo,workspace,config_path);runs={};cells=[]
    for seed in SEEDS:
        value=_validate_run(config,workspace,seed,f"ali_stage4_attack_seed{seed}_v1");runs[str(seed)]=sha256_file(value["run"]/"manifest.json");cells.extend(value["metrics"]["decisions"])
    if len(cells)!=6:raise ValueError("Stage 4 summary requires all six cells")
    destination=workspace/config["output_root"]/"summaries"/safe_run_name(summary_name)
    if destination.exists():raise FileExistsError(destination)
    stage=destination.parent/".work"/f"{safe_run_name(summary_name)}.{uuid.uuid4().hex}.stage";stage.mkdir(parents=True)
    payload={"schema":SUMMARY_SCHEMA,"status":"COMPLETED","cells":cells,"primary_cells_passed":sum(bool(x["primary_cell_pass"]) for x in cells),"all_six_cells_reported":True,"retain_retrain_is_sensitivity":True,"final_test_accessed":False,"test_accessed":False};atomic_json(stage/"summary.json",payload)
    lines=["# Ali Stage 4 attack summary","","All six preregistered seed-by-scenario cells are shown; Retain-Retrain-P5 remains sensitivity-only.","","| Seed | Scenario | Reference | BOTA improvement | Utility | Cell |","|---:|---|---|---|---|---|"]
    for row in cells:lines.append(f"| {row['seed']} | {row['scenario']} | {'PASS' if row['reference_signal_pass'] else 'FAIL'} | {'PASS' if row['bota_gap_improvement_pass'] else 'FAIL'} | {'PASS' if row['utility_guardrail']['pass'] else 'FAIL'} | {'PASS' if row['primary_cell_pass'] else 'FAIL'} |")
    (stage/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n");(stage/"COMPLETED").write_text(SUMMARY_MARKER+"\n",encoding="utf-8",newline="\n")
    atomic_json(stage/"manifest.json",{"schema":SUMMARY_SCHEMA,"files":{n:sha256_file(stage/n) for n in ("COMPLETED","summary.json","summary.md")},"source_run_manifest_sha256":runs,"published_atomically":True,"test_accessed":False});destination.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,destination)
    return {"status":"COMPLETED","summary_dir":str(destination),"cells":6,"test_accessed":False}


def synthetic() -> dict[str, Any]:
    users=np.array([1,2]); member=np.array([.9,.8]); nonmember=np.array([.1,.2]); result=matched_user_cluster_bootstrap(member,users,nonmember,users,seed=42,resamples=20)
    return {"schema":SCHEMA,"pooled_auc":result["pooled_auc"]["point_estimate"],"bootstrap_unit":result["unit"],"real_model_loaded":False,"test_accessed":False}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--repo",type=Path,required=True);parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--config",type=Path,default=Path("configs/ali_stage4_attack_v1.yaml"));parser.add_argument("--mode",choices=("Preflight","SyntheticDryRun","Full","Analyze","Summarize"),default="Preflight");parser.add_argument("--seed",type=int,choices=SEEDS,default=41);parser.add_argument("--run-name",default="");parser.add_argument("--summary-name",default="ali_stage4_attack_summary_v1");args=parser.parse_args()
    repo=args.repo.resolve();workspace=args.workspace.resolve();config_path=args.config if args.config.is_absolute() else repo/args.config;run_name=args.run_name or f"ali_stage4_attack_seed{args.seed}_v1"
    if args.mode=="Preflight":result=preflight(repo,workspace,config_path,args.seed)
    elif args.mode=="SyntheticDryRun":result=synthetic()
    elif args.mode=="Full":result=execute(repo,workspace,config_path,args.seed,run_name)
    elif args.mode=="Analyze":result={"status":"VALID","run_dir":str(_validate_run(load_config(repo,workspace,config_path),workspace,args.seed,run_name)["run"]),"test_accessed":False}
    else:result=summarize(repo,workspace,config_path,args.summary_name)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
