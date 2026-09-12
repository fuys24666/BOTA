"""Secondary matched-user MIA on all nine frozen Stage 2 requests."""
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
from typing import Any

import torch
import yaml
from transformers import T5Tokenizer

from src.bota_short_benchmark import ali_stage2_stronger_signal_v1 as stage2
from src.bota_short_benchmark import runner
from src.bota_short_benchmark import t012_behavior_ablation as behavior
from src.bota_short_benchmark.ali_stage4_attack_v1 import _attack, _paired_bootstrap, _row_from_prediction
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from src.diagnostics.t5_zero_training_analysis import ATTACKS
from src.paper_if_a2.common import atomic_json, canonical_hash, git_snapshot, safe_run_name, sha256_file

SCHEMA="bota-ali-stage4a-stronger-signal-attack-v1"
MARKER="BOTA_ALI_STAGE4A_STRONGER_SIGNAL_ATTACK_V1_COMPLETED"
REQUESTS=("single_k2","single_k4","single_k8","moderate_k2","moderate_k4","moderate_k8","high_k2","high_k4","high_k8")
MODELS=("Original","Exact-Masked","BOTA")
INTERPRETABLE=("high_k4",)
REQUIRED={"COMPLETED","manifest.json","metrics.csv","metrics.json","per_sample_metrics.jsonl","provenance.json","report.md","run_state.json"}


def _json(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict):raise ValueError(f"expected JSON object: {path}")
    return value


def load_config(repo:Path,workspace:Path,path:Path)->dict[str,Any]:
    value=yaml.safe_load(path.read_text(encoding="utf-8"));required={"schema","test_access_policy","parent_amendment","stage2_config","stage2_run","output_root","protocol","evaluation","decision","runtime"}
    if not isinstance(value,dict) or set(value)!=required or value["schema"]!=SCHEMA or value["test_access_policy"]!="forbidden":raise ValueError("invalid Stage 4A config")
    amendment=value["parent_amendment"];amendment_path=repo/amendment["path"]
    if sha256_file(amendment_path)!=amendment["sha256"] or _json(amendment_path).get("amendment_id")!=amendment["amendment_id"]:raise ValueError("Amendment 03 binding changed")
    source=value["stage2_config"]
    if sha256_file(repo/source["path"])!=source["sha256"]:raise ValueError("Stage 2 config binding changed")
    run=workspace/value["stage2_run"]["path"]
    if sha256_file(run/"manifest.json")!=value["stage2_run"]["manifest_sha256"]:raise ValueError("Stage 2 run binding changed")
    protocol=value["protocol"]
    if protocol!={"seed":42,"requests":list(REQUESTS),"models":list(MODELS),"primary_score":"negative_answer_loss","secondary_scores":["confidence","negative_entropy","absolute_yes_no_margin"],"adequate_du_minimum":.001,"formally_interpretable_requests":list(INTERPRETABLE),"report_every_request":True}:raise ValueError("Stage 4A protocol changed")
    if value["evaluation"]!={"split":"Development","final_test":False,"inference_batch_size":4,"bootstrap_unit":"authoritative_user","bootstrap_resamples":2000,"bootstrap_seed":42}:raise ValueError("Stage 4A evaluation changed")
    return value


def preflight(repo:Path,workspace:Path,config_path:Path)->tuple[dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any],list[dict[str,Any]],Path]:
    config=load_config(repo,workspace,config_path);stage2_path=repo/config["stage2_config"]["path"]
    authority,stage2_config,benchmark,base_registry,requests,_=stage2.preflight(repo,workspace,stage2_path)
    source=stage2.analyze(stage2_config,workspace,"ali_stage2_stronger_signal_seed42_v1")
    if tuple(row["id"] for row in requests)!=REQUESTS or source["adequate_requests"]!=list(INTERPRETABLE):raise ValueError("frozen Stage 2 request eligibility changed")
    run=Path(source["run_dir"]);public=_json(run/"request_registry.json")
    if public.get("registry_sha256")!=authority["stronger_signal_registry_sha256"]:raise ValueError("Stage 2 registry changed")
    populations={row["id"]:{"users":len(row["users"]),"member_samples":row["deleted_interactions"],"formally_interpretable":row["id"] in INTERPRETABLE} for row in requests}
    result={"schema":SCHEMA,"status":"READY","requests":list(REQUESTS),"models":list(MODELS),"attacks":list(ATTACKS),"populations":populations,"formally_interpretable_requests":list(INTERPRETABLE),"model_loaded":False,"final_test_accessed":False,"test_accessed":False}
    return result,config,benchmark,base_registry,requests,run


def _guardrail(row:dict[str,Any],decision:dict[str,Any])->dict[str,Any]:
    auc=float(row["bota_auc_damage_vs_original"]);loss=float(row["bota_logloss_damage_vs_original"])
    return {"auc_damage":auc,"logloss_damage":loss,"prediction_collapse":bool(row["prediction_collapse"]),"pass":auc<=decision["overall_auc_damage_max"] and loss<=decision["overall_logloss_damage_max"] and not bool(row["prediction_collapse"])}


def _request_decision(request_id:str,adequate:bool,paired:dict[str,Any],guardrail:dict[str,Any],config:dict[str,Any])->dict[str,Any]:
    permitted=request_id in INTERPRETABLE
    if permitted!=adequate:raise ValueError("formal interpretation differs from frozen DU gate")
    ref=paired["reference_signal"];imp=paired["bota_gap_improvement"]
    ref_pass=ref["point_estimate"]>=config["decision"]["reference_signal_minimum_point_difference"] and ref["percentile_95_ci"][0]>0
    imp_pass=imp["point_estimate"]>0 and imp["percentile_95_ci"][0]>0
    return {"request_id":request_id,"adequate_du":adequate,"formal_interpretation_permitted":permitted,"reference_signal_pass":ref_pass,"bota_gap_improvement_pass":imp_pass,"utility_guardrail":guardrail,"formal_request_pass":bool(permitted and ref_pass and imp_pass and guardrail["pass"]),"status":"FORMAL_PASS" if permitted and ref_pass and imp_pass and guardrail["pass"] else ("FORMAL_FAIL" if permitted else "DESCRIPTIVE_WEAK_DU")}


def execute(repo:Path,workspace:Path,config_path:Path,run_name:str)->dict[str,Any]:
    ready,config,benchmark,base_registry,requests,source_run=preflight(repo,workspace,config_path);git=git_snapshot(repo)
    if not git["clean"]:raise RuntimeError("formal Stage 4A execution requires clean Git")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("exactly one CUDA GPU required")
    free,_=torch.cuda.mem_get_info(0)
    if free/2**30<config["runtime"]["minimum_free_gib"]:raise RuntimeError("insufficient free GPU memory")
    destination=workspace/config["output_root"]/"runs"/safe_run_name(run_name)
    if destination.exists():raise FileExistsError(destination)
    stage=destination.parent/".work"/f"{safe_run_name(run_name)}.{uuid.uuid4().hex}.stage";stage.mkdir(parents=True)
    device=torch.device("cuda:0")
    try:
        base=runner._base_t5_config(workspace,benchmark);model_dir=Path(base["paths"]["model_dir"]);model_dir=model_dir if model_dir.is_absolute() else workspace/model_dir
        tokenizer=T5Tokenizer.from_pretrained(model_dir,local_files_only=True)
        development=JsonPromptDataset(workspace/benchmark["source"]["development_json"],tokenizer)
        train_users,development_users,_=stage2._source_users(workspace,benchmark)
        train_records=json.loads((workspace/benchmark["source"]["train_json"]).read_text(encoding="utf-8"));window_records=[train_records[index] for index in base_registry["order"]];window_users=[int(train_users[index]) for index in base_registry["order"]]
        stage2_metrics={row["request_id"]:row for row in _json(source_run/"metrics.json")["requests"]}
        all_metrics=[];safe_samples=[];decisions=[];started=time.perf_counter()
        for request_number,request in enumerate(requests):
            rid=request["id"];positions=list(map(int,request["forget_window_positions"]));members=[window_records[index] for index in positions];member_users=[window_users[index] for index in positions];requested=set(map(int,request["users"]));nonmember_indices=[i for i,user in enumerate(development_users) if int(user) in requested];nonmember_users=[development_users[i] for i in nonmember_indices]
            if set(member_users)!=requested or set(nonmember_users)!=requested:raise ValueError(f"matched-user population changed: {rid}")
            dataset=JsonPromptDataset(workspace/benchmark["source"]["development_json"],tokenizer);dataset.records=members+[development.records[index] for index in nonmember_indices];indices=list(range(len(dataset)))
            endpoints={"Original":source_run/"references"/"original","Exact-Masked":source_run/"requests"/rid/"exact_masked","BOTA":source_run/"requests"/rid/"bota"}
            predictions={name:behavior._predict_scenario_dir(workspace,benchmark,path,dataset,indices,device) for name,path in endpoints.items()}
            authority=predictions["Exact-Masked"]
            if any(value["gold_label"]!=authority["gold_label"] or value["sample_order_sha256"]!=authority["sample_order_sha256"] for value in predictions.values()):raise ValueError("attack prediction order mismatch")
            caches={};split=len(member_users)
            for model,prediction in predictions.items():
                member_rows=[_row_from_prediction(prediction,i,member_users[i]) for i in range(split)];nonmember_rows=[_row_from_prediction(prediction,split+i,nonmember_users[i]) for i in range(len(nonmember_users))];caches[model]=(member_rows,nonmember_rows)
                for attack_number,(attack_name,score) in enumerate(ATTACKS.items()):
                    result=_attack(member_rows,nonmember_rows,score,config["evaluation"]["bootstrap_seed"],config["evaluation"]["bootstrap_resamples"])
                    all_metrics.append({"request_id":rid,"stratum":request["stratum"],"request_size":request["request_size"],"deleted_interactions":len(member_users),"model":model,"attack":attack_name,"formal_interpretation_permitted":rid in INTERPRETABLE,"member_samples":len(member_rows),"nonmember_samples":len(nonmember_rows),**result})
                for role,rows in (("member",member_rows),("nonmember",nonmember_rows)):
                    for ordinal,row in enumerate(rows):safe_samples.append({"request_id":rid,"model":model,"role":role,"ordinal":ordinal,"user_hash":canonical_hash([42,rid,row["user_id"]]),"answer_sequence_loss":row["answer_sequence_loss"],"confidence":row["confidence"],"binary_entropy":row["binary_entropy"],"yes_no_margin":row["yes_no_margin"]})
            paired=_paired_bootstrap(caches,config["protocol"]["primary_score"],config["evaluation"]["bootstrap_seed"],config["evaluation"]["bootstrap_resamples"]);frozen=stage2_metrics[rid];guard=_guardrail(frozen,config["decision"]);decision=_request_decision(rid,bool(frozen["adequate_effect"]),paired,guard,config);decision.update({"DU":frozen["DU"],"EU":frozen["EU"],"local_residual":frozen["local_residual"],"toward_rate":frozen["toward_rate"],"paired_bootstrap":paired});decisions.append(decision)
        atomic_json(stage/"metrics.json",{"schema":SCHEMA,"evidence_class":"secondary_stronger_signal_attack","metrics":all_metrics,"decisions":decisions,"all_nine_requests_reported":True,"only_high_k4_formally_interpretable":True,"parent_stage4_result_unchanged":True,"final_test_accessed":False,"test_accessed":False})
        flat=[{"request_id":row["request_id"],"stratum":row["stratum"],"request_size":row["request_size"],"deleted_interactions":row["deleted_interactions"],"model":row["model"],"attack":row["attack"],"formal_interpretation_permitted":row["formal_interpretation_permitted"],"pooled_roc_auc":row["pooled"]["roc_auc"],"macro_user_auc":row["macro_user_auc"]["mean"],"ci_lower":row["matched_user_bootstrap"]["pooled_auc"]["percentile_95_ci"][0],"ci_upper":row["matched_user_bootstrap"]["pooled_auc"]["percentile_95_ci"][1]} for row in all_metrics]
        with (stage/"metrics.csv").open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        with (stage/"per_sample_metrics.jsonl").open("w",encoding="utf-8",newline="\n") as handle:
            for row in safe_samples:handle.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
        report=["# Ali Stage 4A stronger-signal attack","","All nine frozen Stage 2 requests are reported. Only high_k4 is eligible for formal interpretation; the parent Stage 4 result remains unchanged.","","| Request | DU | Reference | BOTA improvement | Utility | Status |","|---|---:|---|---|---|---|"]
        for row in decisions:report.append(f"| {row['request_id']} | {row['DU']:.6f} | {'PASS' if row['reference_signal_pass'] else 'FAIL'} | {'PASS' if row['bota_gap_improvement_pass'] else 'FAIL'} | {'PASS' if row['utility_guardrail']['pass'] else 'FAIL'} | {row['status']} |")
        (stage/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8",newline="\n")
        peak=torch.cuda.max_memory_reserved()/2**30
        if peak>config["runtime"]["hard_peak_reserved_gib"]:raise RuntimeError("GPU hard cap exceeded")
        atomic_json(stage/"run_state.json",{"schema":SCHEMA,"status":"COMPLETED","seed":42,"requests":list(REQUESTS),"models":list(MODELS),"attacks":list(ATTACKS),"formal_request":"high_k4","wall_time_seconds":time.perf_counter()-started,"peak_reserved_gib":peak,"test_accessed":False})
        atomic_json(stage/"provenance.json",{"schema":SCHEMA,"git":git,"config_sha256":sha256_file(config_path),"amendment_sha256":config["parent_amendment"]["sha256"],"stage2_manifest_sha256":sha256_file(source_run/"manifest.json"),"stage2_run":str(source_run),"raw_user_ids_persisted":False,"split":"Development","final_test_loader_built":False,"test_accessed":False})
        (stage/"COMPLETED").write_text(MARKER+"\n",encoding="utf-8",newline="\n");atomic_json(stage/"manifest.json",{"schema":SCHEMA,"files":{name:sha256_file(stage/name) for name in ("COMPLETED","metrics.csv","metrics.json","per_sample_metrics.jsonl","provenance.json","report.md","run_state.json")},"published_atomically":True,"test_accessed":False});destination.parent.mkdir(parents=True,exist_ok=True);os.replace(stage,destination)
        return {"status":"COMPLETED","run_dir":str(destination),"requests":9,"metrics":108,"formal_request":"high_k4","test_accessed":False}
    except Exception as error:
        shutil.rmtree(stage,ignore_errors=True);raise RuntimeError("Stage 4A failed before atomic publication") from error
    finally:
        gc.collect();torch.cuda.empty_cache()


def analyze(repo:Path,workspace:Path,config_path:Path,run_name:str)->dict[str,Any]:
    config=load_config(repo,workspace,config_path);run=workspace/config["output_root"]/"runs"/safe_run_name(run_name)
    if not run.is_dir() or {p.name for p in run.iterdir()}!=REQUIRED or (run/"COMPLETED").read_text(encoding="utf-8")!=MARKER+"\n":raise ValueError("invalid Stage 4A run")
    manifest=_json(run/"manifest.json")
    for name,digest in manifest["files"].items():
        if sha256_file(run/name)!=digest:raise ValueError(f"Stage 4A artifact changed: {name}")
    metrics=_json(run/"metrics.json")
    if len(metrics["metrics"])!=108 or len(metrics["decisions"])!=9 or [row["request_id"] for row in metrics["decisions"] if row["formal_interpretation_permitted"]]!=list(INTERPRETABLE) or metrics.get("test_accessed") is not False:raise ValueError("Stage 4A completeness invariant failed")
    return {"status":"VALID","run_dir":str(run),"formal_decision":next(row for row in metrics["decisions"] if row["request_id"]=="high_k4"),"test_accessed":False}


def synthetic()->dict[str,Any]:
    config={"decision":{"reference_signal_minimum_point_difference":.02}};paired={"reference_signal":{"point_estimate":.03,"percentile_95_ci":[.01,.05]},"bota_gap_improvement":{"point_estimate":.02,"percentile_95_ci":[.001,.04]}};guard={"pass":True}
    return {"schema":SCHEMA,"high_k4":_request_decision("high_k4",True,paired,guard,config),"weak":_request_decision("single_k2",False,paired,guard,config),"real_model_loaded":False,"test_accessed":False}


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--repo",type=Path,required=True);parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--config",type=Path,default=Path("configs/ali_stage4a_stronger_signal_attack_v1.yaml"));parser.add_argument("--mode",choices=("Preflight","SyntheticDryRun","Full","Analyze"),default="Preflight");parser.add_argument("--run-name",default="ali_stage4a_stronger_signal_attack_seed42_v1");args=parser.parse_args();repo=args.repo.resolve();workspace=args.workspace.resolve();config_path=args.config if args.config.is_absolute() else repo/args.config
    if args.mode=="Preflight":result=preflight(repo,workspace,config_path)[0]
    elif args.mode=="SyntheticDryRun":result=synthetic()
    elif args.mode=="Full":result=execute(repo,workspace,config_path,args.run_name)
    else:result=analyze(repo,workspace,config_path,args.run_name)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
