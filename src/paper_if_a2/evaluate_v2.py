from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any,Callable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import set_peft_model_state_dict
from transformers import T5ForConditionalGeneration,T5Tokenizer

from src.diagnostics.t5_full_runner import _batch
from src.diagnostics.t5_lora_influence_feasibility_audit_a2 import install_fixed_ab_coordinate
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset,build_current_model,load_config as load_base_config,load_legacy_model,move_batch
from src.diagnostics.t5_step817_forget_conflict_audit import _data_lineage
from src.diagnostics.t5_zero_training_analysis import mia_for_model

from .artifacts import complete,publish_manifest,verify_completed,write_contract
from .common import atomic_json,atomic_text,canonical_hash,directory_hash,require_formal_preflight,safe_new_directory,sha256_file
from .metrics import binary_metrics,classify_paper,clustered_bootstrap,clustered_ratio,direction_metrics,membership_metrics


SCHEMA="paper-if-a2-evaluation-v2";MARKER="PAPER_IF_A2_EVALUATION_V2_COMPLETED";MODELS=("original","if_a2","e2urec","retrain")
ALLOWED_SAMPLE_KEYS={"canonical_sample_index","authoritative_user_id","split","group","gold_label",*(f"{model}_probability" for model in MODELS),*(f"{model}_answer_loss" for model in MODELS),"full_vocab_jsd_original_retrain","full_vocab_jsd_if_a2_retrain","full_vocab_jsd_e2urec_retrain","yes_no_jsd_original_retrain","yes_no_jsd_if_a2_retrain","yes_no_jsd_e2urec_retrain"}


def load_evaluation_config(path:Path)->dict[str,Any]:
    value=yaml.safe_load(path.read_text(encoding="utf-8"));expected={"schema":SCHEMA,"inference_batch_size":4,"bootstrap":{"resamples":2000,"confidence":.95,"cluster":"authoritative_user","seed":42},"split":"Development","test_access_policy":"forbidden","primary_mia":"forget.primary_matched_user.negative_answer_loss.pooled_matched_user.roc_auc","sensitivity_mia":"absolute_attack_advantage","primary_retrain_schema":"paper-retain-retrain-v3","primary_retrain_patience":5,"primary_retrain_requires_amendment_disclosure":True,"historical_patience3_run":"paper_retain_retrain_ml1m_seed42_v3","historical_patience3_schema":"paper-retain-retrain-v2"}
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError("frozen four-model evaluation protocol changed")
    return value


def reject_non_development(split:str)->None:
    if split!="Development":raise PermissionError("evaluation v2 is Development-only; test is forbidden")


def paired_difference(left:list[float],right:list[float],users:list[int],*,resamples:int=2000,seed:int=42)->dict[str,Any]:
    if not (len(left)==len(right)==len(users)) or not left:raise ValueError("paired sample binding mismatch")
    differences=np.asarray(left,dtype=float)-np.asarray(right,dtype=float);summary=clustered_bootstrap(differences.tolist(),users,resamples=resamples,seed=seed)
    lower,upper=summary["ci_lower"],summary["ci_upper"]
    summary.update({"definition":"distance(IF-A2, Retrain) - distance(E2URec, Retrain)","negative_means":"IF-A2_better","positive_means":"E2URec_better","decision":"IF-A2_better" if upper<0 else "E2URec_better" if lower>0 else "inconclusive"});return summary


def absolute_attack_advantage(auc:float)->float:return abs(float(auc)-.5)


def efficiency_rows(method:dict[str,Any],retrain:dict[str,Any],e2urec:dict[str,Any],*,authority_mode:bool,historical_retrain:dict[str,Any]|None=None)->list[dict[str,Any]]:
    retrain_seconds=float(retrain["seconds"]);rows=[]
    for name,payload,scope in (("IF-A2",method,"end_to_end"),("Retain Retrain",retrain,"end_to_end"),("E2URec",e2urec,"historical_runtime_unavailable" if authority_mode else "online_only")):
        seconds=None if authority_mode and name=="E2URec" else payload.get("seconds");comparable=name!="E2URec" or (not authority_mode and payload.get("inclusive_seconds") is not None)
        rows.append({"method":name,"seconds":seconds,"speedup_vs_retrain":None if seconds is None else retrain_seconds/seconds,"time_ratio_vs_retrain":None if seconds is None else seconds/retrain_seconds,"peak_vram_bytes":payload.get("peak_vram_bytes"),"trainable_parameters":payload.get("trainable_parameters"),"optimizer_steps":payload.get("optimizer_steps"),"forward_calls":payload.get("forward_calls"),"backward_or_hvp_calls":payload.get("backward_or_hvp_calls"),"artifact_bytes":payload.get("artifact_bytes"),"timing_scope":scope,"inclusive_comparability":"comparable" if comparable else "unavailable"})
    rows[1].update({"patience":retrain.get("configured_patience",retrain.get("patience")),"best_epoch":retrain.get("best_epoch"),"stopping_epoch":retrain.get("stopping_epoch",retrain.get("epochs")),"training_seconds":retrain.get("training_seconds"),"validation_seconds":retrain.get("validation_seconds"),"checkpoint_seconds":retrain.get("checkpoint_seconds"),"physical_batch_size":retrain.get("physical_batch_size"),"effective_batch_size":retrain.get("effective_batch_size"),"precision":retrain.get("precision")})
    if historical_retrain is not None:rows.append({"method":"Retain Retrain patience3 sensitivity","seconds":historical_retrain.get("seconds"),"speedup_vs_retrain":None,"time_ratio_vs_retrain":None,"peak_vram_bytes":historical_retrain.get("peak_vram_bytes"),"trainable_parameters":historical_retrain.get("trainable_parameters"),"optimizer_steps":historical_retrain.get("optimizer_steps"),"forward_calls":historical_retrain.get("forward_calls"),"backward_or_hvp_calls":historical_retrain.get("backward_or_hvp_calls"),"artifact_bytes":historical_retrain.get("artifact_bytes"),"timing_scope":"historical_patience3_sensitivity","inclusive_comparability":"sensitivity_only","patience":3,"best_epoch":historical_retrain.get("best_epoch"),"stopping_epoch":historical_retrain.get("stopping_epoch"),"training_seconds":historical_retrain.get("training_seconds"),"validation_seconds":historical_retrain.get("validation_seconds"),"checkpoint_seconds":historical_retrain.get("checkpoint_seconds"),"physical_batch_size":historical_retrain.get("physical_batch_size"),"effective_batch_size":historical_retrain.get("effective_batch_size"),"precision":historical_retrain.get("precision")})
    return rows


def efficiency_csv(rows:list[dict[str,Any]])->str:
    fields=["method","seconds","speedup_vs_retrain","time_ratio_vs_retrain","timing_scope","inclusive_comparability","patience","best_epoch","stopping_epoch","optimizer_steps","training_seconds","validation_seconds","checkpoint_seconds","physical_batch_size","effective_batch_size","precision","peak_vram_bytes"]
    return ",".join(fields)+"\n"+"".join(",".join(str(row.get(field,"")) for field in fields)+"\n" for row in rows)


def validate_retrain_binding(root:Path,run_name:str,sensitivity_mode:str="")->dict[str,Any]:
    if sensitivity_mode not in {"","Patience3Historical"}:raise ValueError("invalid RetrainSensitivityMode")
    run_dir=root/"outputs/paper_if_a2_v1/retrain_runs"/run_name;verify_completed(run_dir,"PAPER_RETAIN_RETRAIN_COMPLETED");manifest=json.loads((run_dir/"manifest.json").read_text(encoding="utf-8"));result=json.loads((run_dir/"retrain_result.json").read_text(encoding="utf-8"))
    if manifest.get("test_accessed") is not False or result.get("test_accessed") is not False:raise ValueError("Retrain test invariant failed")
    if not sensitivity_mode:
        required={"schema":"paper-retain-retrain-v3","configured_patience":5,"protocol_amendment_after_development_inspection":True,"previous_primary_retrain_preserved":True,"previous_results_not_overwritten":True,"amendment_does_not_use_test":True,"best_model_restored":True}
        if manifest.get("schema")!="paper-retain-retrain-v3" or any(result.get(k)!=v for k,v in required.items()):raise ValueError("Primary evaluation requires amended patience-five Retain Retrain v3")
        return {"scope":"primary","primary_eligible":True,"patience":5,"schema":result["schema"],"best_epoch":result["best_epoch"],"stopping_epoch":result["stopping_epoch"],"best_validation_loss":result["best_validation_loss"],"manifest_sha256":sha256_file(run_dir/"manifest.json"),"test_accessed":False}
    if manifest.get("schema")!="paper-retain-retrain-v2" or result.get("schema")!="paper-retain-retrain-v2":raise ValueError("Patience3Historical requires the preserved v2 run")
    return {"scope":"sensitivity_only","primary_eligible":False,"patience":3,"schema":result["schema"],"best_epoch":result["best_epoch"],"stopping_epoch":result["epochs"],"best_validation_loss":result["best_validation_loss"],"manifest_sha256":sha256_file(run_dir/"manifest.json"),"test_accessed":False}


def classify_method(*,valid:bool,forgetting:bool,utility:bool,mia:bool,efficient:bool,efficiency_available:bool=True)->dict[str,str]:
    if not efficiency_available and valid and forgetting and utility and mia:return {"category":"PAPER-EFFICIENCY-UNAVAILABLE","next_action":"measure_inclusive_end_to_end_runtime"}
    return classify_paper(valid=valid,forgetting=forgetting,utility=utility,mia=mia,efficient=efficient)


def preflight(root:Path,config_path:Path,method_name:str="",retrain_name:str="",e2urec_name:str="",split:str="Development",authority_mode:str="",retrain_sensitivity_mode:str="",*,formal:bool=False)->dict[str,Any]:
    reject_non_development(split);config=load_evaluation_config(config_path)
    if authority_mode not in {"","ReconstructedStep1000"}:raise ValueError("unknown E2URec authority mode")
    if e2urec_name and authority_mode:raise ValueError("E2URecRunName and E2URecAuthorityMode are mutually exclusive")
    if sha256_file(root/config["original"])!=config["original_sha256"] or sha256_file(root/config["development"])!=config["development_sha256"]:raise ValueError("evaluation source SHA mismatch")
    if authority_mode and sha256_file(root/config["authority_checkpoint"])!=config["authority_checkpoint_sha256"]:raise ValueError("authority checkpoint SHA mismatch")
    retrain_binding=validate_retrain_binding(root,retrain_name,retrain_sensitivity_mode) if retrain_name else None
    result={"schema":SCHEMA,"mode":"Preflight","split":"Development","method_run_name":method_name,"retrain_run_name":retrain_name,"retrain_binding":retrain_binding,"retrain_sensitivity_mode":retrain_sensitivity_mode or None,"e2urec_run_name":e2urec_name or None,"e2urec_authority_mode":authority_mode or None,"efficiency_status":"historical_runtime_unavailable" if authority_mode else "formal_run_required","model_loaded":False,"optimizer_constructed":False,"test_loader_built":False,"test_accessed":False}
    if formal:result["formal"]=require_formal_preflight(root)
    return result


def _release(model:Any)->None:
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def _load_retrain(path:Path,device:torch.device)->Any:return T5ForConditionalGeneration.from_pretrained(path/"best_model",attn_implementation="eager").float().to(device).eval()


def _load_named(root:Path,base:dict[str,Any],kind:str,path:Path|None,device:torch.device,authority_checkpoint:Path|None=None)->Any:
    if kind=="original":return load_legacy_model(Path(base["paths"]["original"])).to(device).eval()
    if kind=="if_a2":
        model=load_legacy_model(Path(base["paths"]["original"])).to(device).eval();adapter=torch.load(path/"adapter/adapter_model.pt",map_location=device,weights_only=True);names,parameters=install_fixed_ab_coordinate(model,adapter["A"],32)
        with torch.no_grad():
            for name,parameter in zip(names,parameters):parameter.copy_(adapter["B"][name].to(parameter))
        return model.eval()
    if kind=="e2urec":
        model=build_current_model(Path(base["paths"]["original"]),base["lora"]);state=torch.load(authority_checkpoint,map_location="cpu",weights_only=False)["adapter_state"] if authority_checkpoint else torch.load(path/"adapter/adapter_model.pt",map_location="cpu",weights_only=True);set_peft_model_state_dict(model,state);return model.to(device).eval()
    raise ValueError(kind)


def evaluate_pair(left_name:str,left:Any,retrain:Any,dataset:Any,indices:list[int],users:list[int],device:torch.device,active:dict[int,bool])->list[dict[str,Any]]:
    rows=[];yes_id=2163;no_id=465
    with torch.inference_mode():
        for start in range(0,len(indices),4):
            selected=indices[start:start+4];batch=move_batch(_batch(dataset,selected),device);outputs=[model(input_ids=batch["input_ids"],labels=batch["target_ids"]).logits for model in (left,retrain)];probs=[F.softmax(value,-1) for value in outputs];mask=batch["target_ids"]!=-100;targets=batch["target_ids"][:,0];yes=[F.softmax(value[:,0,[no_id,yes_id]],-1)[:,1] for value in outputs]
            losses=[]
            for value in probs:losses.append((-(torch.gather(value,2,batch["target_ids"].clamp_min(0).unsqueeze(-1)).squeeze(-1).clamp_min(1e-30).log())*mask).sum(-1)/mask.sum(-1))
            midpoint=(probs[0]+probs[1])/2;full=.5*((probs[0]*(probs[0].clamp_min(1e-30).log()-midpoint.clamp_min(1e-30).log())).sum(-1)+(probs[1]*(probs[1].clamp_min(1e-30).log()-midpoint.clamp_min(1e-30).log())).sum(-1));full=(full*mask).sum(-1)/mask.sum(-1);binary=[]
            l=torch.stack([1-yes[0],yes[0]],-1).clamp_min(1e-12);r=torch.stack([1-yes[1],yes[1]],-1).clamp_min(1e-12);m=(l+r)/2;binary=.5*(l*(l.log()-m.log())).sum(-1)+.5*(r*(r.log()-m.log())).sum(-1)
            for local,index in enumerate(selected):rows.append({"canonical_sample_index":int(index),"authoritative_user_id":int(users[index]),"split":"Development","group":"active" if active.get(index,False) else "inactive","gold_label":int(targets[local]==yes_id),f"{left_name}_probability":float(yes[0][local]),"retrain_probability":float(yes[1][local]),f"{left_name}_answer_loss":float(losses[0][local]),"retrain_answer_loss":float(losses[1][local]),f"full_vocab_jsd_{left_name}_retrain":float(full[local]),f"yes_no_jsd_{left_name}_retrain":float(binary[local])})
            del batch,outputs,probs
    return rows


def merge_bound_pairs(pairs:dict[str,list[dict[str,Any]]])->list[dict[str,Any]]:
    reference=None;merged={}
    for name,rows in pairs.items():
        identity=[(r["canonical_sample_index"],r["authoritative_user_id"],r["gold_label"]) for r in rows]
        if reference is None:reference=identity
        elif identity!=reference:raise ValueError("four-model sample/user/order binding mismatch")
        for row in rows:
            key=row["canonical_sample_index"]
            if key not in merged:merged[key]=dict(row)
            else:
                for field,value in row.items():
                    if field in merged[key] and merged[key][field]!=value:raise ValueError(f"inconsistent shared field: {field}")
                    merged[key][field]=value
    return [merged[index] for index,_,_ in reference]


def subgroup_masks(rows:list[dict[str,Any]])->dict[str,list[bool]]:
    users={r["authoritative_user_id"] for r in rows};scores={u:np.mean([r["full_vocab_jsd_if_a2_retrain"] for r in rows if r["authoritative_user_id"]==u]) for u in users};count=max(1,math.ceil(.05*len(users)));worst={u for u,_ in sorted(scores.items(),key=lambda pair:(-pair[1],pair[0]))[:count]}
    return {"all":[True]*len(rows),"observed_yes":[r["gold_label"]==1 for r in rows],"observed_no":[r["gold_label"]==0 for r in rows],"active":[r["group"]=="active" for r in rows],"inactive":[r["group"]=="inactive" for r in rows],"worst_5_percent_users":[r["authoritative_user_id"] in worst for r in rows]}


def summarize_four(rows:list[dict[str,Any]],overall:list[dict[str,Any]],retain:list[dict[str,Any]],config:dict[str,Any],mia:dict[str,Any],efficiency:list[dict[str,Any]])->dict[str,Any]:
    groups={};masks=subgroup_masks(rows)
    for group,mask in masks.items():
        chosen=[r for r,keep in zip(rows,mask) if keep];users=[r["authoritative_user_id"] for r in chosen];groups[group]={}
        for method in ("if_a2","e2urec"):
            groups[group][method]={"full_vocab_jsd":clustered_bootstrap([r[f"full_vocab_jsd_{method}_retrain"] for r in chosen],users),"yes_no_jsd":clustered_bootstrap([r[f"yes_no_jsd_{method}_retrain"] for r in chosen],users),"full_vocab_residual_ratio":clustered_ratio([r[f"full_vocab_jsd_{method}_retrain"] for r in chosen],[r["full_vocab_jsd_original_retrain"] for r in chosen],users),"yes_no_residual_ratio":clustered_ratio([r[f"yes_no_jsd_{method}_retrain"] for r in chosen],[r["yes_no_jsd_original_retrain"] for r in chosen],users),"answer_loss_gap_improvement":clustered_bootstrap([abs(r["original_answer_loss"]-r["retrain_answer_loss"])-abs(r[f"{method}_answer_loss"]-r["retrain_answer_loss"]) for r in chosen],users),"observed_probability_distance_improvement":clustered_bootstrap([abs(r["original_probability"]-r["retrain_probability"])-abs(r[f"{method}_probability"]-r["retrain_probability"]) for r in chosen],users),"direction":direction_metrics([r["original_probability"] for r in chosen],[r[f"{method}_probability"] for r in chosen],[r["retrain_probability"] for r in chosen])}
        groups[group]["pairwise"]={"full_vocab_jsd":paired_difference([r["full_vocab_jsd_if_a2_retrain"] for r in chosen],[r["full_vocab_jsd_e2urec_retrain"] for r in chosen],users),"yes_no_jsd":paired_difference([r["yes_no_jsd_if_a2_retrain"] for r in chosen],[r["yes_no_jsd_e2urec_retrain"] for r in chosen],users),"answer_loss_absolute_gap":paired_difference([abs(r["if_a2_answer_loss"]-r["retrain_answer_loss"]) for r in chosen],[abs(r["e2urec_answer_loss"]-r["retrain_answer_loss"]) for r in chosen],users),"observed_probability_distance":paired_difference([abs(r["if_a2_probability"]-r["retrain_probability"]) for r in chosen],[abs(r["e2urec_probability"]-r["retrain_probability"]) for r in chosen],users)}
    utility={}
    for scope,selected in (("overall_development",overall),("retain_user_development",retain)):
        labels=[r["gold_label"] for r in selected];utility[scope]={model:binary_metrics(labels,[r[f"{model}_probability"] for r in selected]) for model in MODELS}
    decisions={}
    efficiency_map={r["method"]:r for r in efficiency}
    for method,label in (("if_a2","IF-A2"),("e2urec","E2URec")):
        utility_pass=all(utility[s]["original"]["auc"]-utility[s][method]["auc"]<=config["utility"]["overall_auc_damage_max"] and utility[s][method]["log_loss"]-utility[s]["original"]["log_loss"]<=config["utility"]["overall_logloss_damage_max"] and not utility[s][method]["prediction_collapse"] for s in utility);forgetting=groups["all"][method]["full_vocab_residual_ratio"]["ci_upper"]<1 and groups["all"][method]["yes_no_residual_ratio"]["ci_upper"]<1;mia_pass=mia["gaps"][method]<=config["mia_gap_max"];erow=efficiency_map[label];available=erow["inclusive_comparability"]=="comparable";efficient=available and erow["time_ratio_vs_retrain"]<=config["efficiency"]["pass_time_ratio_max"];decisions[method]=classify_method(valid=True,forgetting=forgetting,utility=utility_pass,mia=mia_pass,efficient=efficient,efficiency_available=available)|{"forgetting_pass":forgetting,"utility_pass":utility_pass,"mia_pass":mia_pass}
    return {"schema":SCHEMA,"methods":decisions,"groups":groups,"utility":utility,"mia":mia,"efficiency":efficiency,"pairwise_sign_definition":"distance(IF-A2, Retrain) - distance(E2URec, Retrain); negative=IF-A2_better; positive=E2URec_better","bootstrap_resamples":2000,"test_accessed":False}


def _scalar_cache(model:Any,dataset:Any,indices:list[int],users:list[int],device:torch.device)->list[dict[str,Any]]:
    rows=[];yes_id=2163;no_id=465
    with torch.inference_mode():
        for start in range(0,len(indices),4):
            chosen=indices[start:start+4];batch=move_batch(_batch(dataset,chosen),device);logits=model(input_ids=batch["input_ids"],labels=batch["target_ids"]).logits;mask=batch["target_ids"]!=-100;loss=-(torch.gather(F.log_softmax(logits,-1),2,batch["target_ids"].clamp_min(0).unsqueeze(-1)).squeeze(-1)*mask).sum(-1)/mask.sum(-1);yes=F.softmax(logits[:,0,[no_id,yes_id]],-1)[:,1]
            for local,index in enumerate(chosen):rows.append({"answer_sequence_loss":float(loss[local]),"confidence":float(torch.maximum(yes[local],1-yes[local])),"binary_entropy":float(-(yes[local].clamp_min(1e-12).log()*yes[local]+(1-yes[local]).clamp_min(1e-12).log()*(1-yes[local]))),"yes_no_margin":float(2*yes[local]-1),"user_id":int(users[index])})
    return rows


def run_full(root:Path,config_path:Path,method_name:str,retrain_name:str,e2urec_name:str,run_name:str,authority_mode:str,retrain_sensitivity_mode:str="")->dict[str,Any]:
    if not method_name or not retrain_name or not (e2urec_name or authority_mode):raise ValueError("formal evaluation requires MethodRunName, RetrainRunName, and one E2URec source")
    config=load_evaluation_config(config_path);pre=preflight(root,config_path,method_name,retrain_name,e2urec_name,"Development",authority_mode,retrain_sensitivity_mode,formal=True);method_dir=root/"outputs/paper_if_a2_v1/method_runs"/method_name;retrain_dir=root/"outputs/paper_if_a2_v1/retrain_runs"/retrain_name;verify_completed(method_dir,"PAPER_IF_A2_METHOD_COMPLETED");retrain_binding=validate_retrain_binding(root,retrain_name,retrain_sensitivity_mode)
    e2_dir=None
    if not authority_mode:
        e2_dir=root/"outputs/paper_if_a2_v1/e2urec_runs"/e2urec_name;verify_completed(e2_dir,"PAPER_E2UREC_BASELINE_COMPLETED");e2_result=json.loads((e2_dir/"e2urec_result.json").read_text(encoding="utf-8"))
        if e2_result.get("classification")!="E2R-AUTHORITY-EXACT":raise ValueError("primary evaluation requires E2R-AUTHORITY-EXACT")
    run_dir=safe_new_directory(root,"evaluation_v2_runs",run_name);write_contract(run_dir,{**pre,"config_sha256":sha256_file(config_path),"method_manifest_sha256":sha256_file(method_dir/"manifest.json"),"retrain_manifest_sha256":sha256_file(retrain_dir/"manifest.json"),"e2urec_manifest_sha256":None if authority_mode else sha256_file(e2_dir/"manifest.json")})
    device=torch.device("cuda:0");base=load_base_config(root/config["base_config"],root);tokenizer=T5Tokenizer.from_pretrained(base["paths"]["model_dir"]);dataset=JsonPromptDataset(root/config["development"],tokenizer);_,indices,users=_data_lineage(root,base,root/config["protocol_root"]);active={}
    for line in (root/config["selector"]/"rows.jsonl").read_text(encoding="utf-8").splitlines():row=json.loads(line);active[int(row["source_index"])]=bool(row["active"])
    pairs={}
    for kind in ("original","if_a2","e2urec"):
        left=_load_named(root,base,kind,method_dir if kind=="if_a2" else e2_dir,device,root/config["authority_checkpoint"] if authority_mode and kind=="e2urec" else None);retrain=_load_retrain(retrain_dir,device);pairs[kind]=evaluate_pair(kind,left,retrain,dataset,indices["overall_validation"],users["overall_validation"],device,active);del left,retrain;gc.collect();torch.cuda.empty_cache()
    all_rows=merge_bound_pairs(pairs);by_index={r["canonical_sample_index"]:r for r in all_rows};forget_rows=[by_index[i] for i in indices["forget_user_validation"]];retain_rows=[by_index[i] for i in indices["retain_user_validation"]]
    forget_data=JsonPromptDataset(Path(base["paths"]["forget"]),tokenizer);retain_data=JsonPromptDataset(Path(base["paths"]["retain"]),tokenizer);mia={"models":{},"primary_attack":config["primary_mia"],"sensitivity_only":"absolute_attack_advantage"}
    for kind in MODELS:
        model=_load_retrain(retrain_dir,device) if kind=="retrain" else _load_named(root,base,kind,method_dir if kind=="if_a2" else e2_dir,device,root/config["authority_checkpoint"] if authority_mode and kind=="e2urec" else None);caches={"forget_train":_scalar_cache(model,forget_data,indices["forget_train"],users["forget_train"],device),"retain_train":_scalar_cache(model,retain_data,indices["retain_train"],users["retain_train"],device),"forget_user_validation":_scalar_cache(model,dataset,indices["forget_user_validation"],users["overall_validation"],device),"retain_user_validation":_scalar_cache(model,dataset,indices["retain_user_validation"],users["overall_validation"],device)};mia["models"][kind]=mia_for_model(caches,seed=42,resamples=2000);del model;gc.collect();torch.cuda.empty_cache()
    def auc(kind:str)->float:return float(mia["models"][kind]["forget"]["primary_matched_user"]["attacks"]["negative_answer_loss"]["pooled_matched_user"]["roc_auc"])
    mia["gaps"]={"if_a2":abs(auc("if_a2")-auc("retrain")),"e2urec":abs(auc("e2urec")-auc("retrain"))};mia["sensitivity"]={kind:absolute_attack_advantage(auc(kind)) for kind in MODELS}
    mt=json.loads((method_dir/"timing.json").read_text(encoding="utf-8"));mr=json.loads((method_dir/"method_result.json").read_text(encoding="utf-8"));rt=json.loads((retrain_dir/"timing.json").read_text(encoding="utf-8"));rr=json.loads((retrain_dir/"retrain_result.json").read_text(encoding="utf-8"));et={} if authority_mode else json.loads((e2_dir/"timing.json").read_text(encoding="utf-8"));er={} if authority_mode else json.loads((e2_dir/"e2urec_result.json").read_text(encoding="utf-8"));historical=None
    if not retrain_sensitivity_mode:
        historical_dir=root/"outputs/paper_if_a2_v1/retrain_runs"/config["historical_patience3_run"];historical_binding=validate_retrain_binding(root,config["historical_patience3_run"],"Patience3Historical");ht=json.loads((historical_dir/"timing.json").read_text(encoding="utf-8"));hr=json.loads((historical_dir/"retrain_result.json").read_text(encoding="utf-8"));historical={"seconds":ht["retrain_end_to_end_seconds"],**hr,**{key:ht.get(key) for key in ("training_seconds","validation_seconds","checkpoint_seconds")},"stopping_epoch":hr["epochs"],"artifact_bytes":sum(p.stat().st_size for p in historical_dir.rglob("*") if p.is_file())}
    eff=efficiency_rows({"seconds":mt["method_end_to_end_seconds"],**mr,"backward_or_hvp_calls":mr.get("hvp_calls"),"artifact_bytes":sum(p.stat().st_size for p in method_dir.rglob("*") if p.is_file())},{"seconds":rt["retrain_end_to_end_seconds"],**rr,**{key:rt.get(key) for key in ("training_seconds","validation_seconds","checkpoint_seconds")},"backward_or_hvp_calls":rr.get("backward_calls"),"artifact_bytes":sum(p.stat().st_size for p in retrain_dir.rglob("*") if p.is_file())},{"seconds":et.get("e2urec_online_unlearning_seconds"),"inclusive_seconds":et.get("e2urec_end_to_end_inclusive_seconds"),**er,"backward_or_hvp_calls":er.get("backward_calls"),"artifact_bytes":None if authority_mode else sum(p.stat().st_size for p in e2_dir.rglob("*") if p.is_file())},authority_mode=bool(authority_mode),historical_retrain=historical);metrics=summarize_four(forget_rows,all_rows,retain_rows,config,mia,eff)|{"method_run_name":method_name,"retrain_run_name":retrain_name,"retrain_binding":retrain_binding,"retrain_sensitivity_mode":retrain_sensitivity_mode or None,"classification_scope":retrain_binding["scope"],"primary_classification_published":retrain_binding["primary_eligible"],"e2urec_run_name":e2urec_name or None,"e2urec_authority_mode":authority_mode or None,"split":"Development"}
    atomic_json(run_dir/"metrics_v2.json",metrics);atomic_text(run_dir/"per_sample_metrics.jsonl","".join(json.dumps({k:r[k] for k in ALLOWED_SAMPLE_KEYS},sort_keys=True)+"\n" for r in forget_rows));atomic_text(run_dir/"summary_methods.csv","method,category,forgetting_pass,utility_pass,mia_pass\n"+"".join(f"{name},{value['category']},{value['forgetting_pass']},{value['utility_pass']},{value['mia_pass']}\n" for name,value in metrics["methods"].items()))
    pair_header="group,metric,point,ci_lower,ci_upper,decision,sign_definition\n";atomic_text(run_dir/"pairwise_if_a2_vs_e2urec.csv",pair_header+"".join(f"{group},{metric},{value['point']},{value['ci_lower']},{value['ci_upper']},{value['decision']},distance(IF-A2 Retrain)-distance(E2URec Retrain)\n" for group,g in metrics["groups"].items() for metric,value in g["pairwise"].items()));atomic_text(run_dir/"subgroup_metrics.csv","group,method,full_residual,yes_no_residual\n"+"".join(f"{group},{method},{value['full_vocab_residual_ratio']['point']},{value['yes_no_residual_ratio']['point']}\n" for group,g in metrics["groups"].items() for method,value in ((m,g[m]) for m in ("if_a2","e2urec"))));atomic_text(run_dir/"utility_metrics.csv","scope,model,auc,log_loss,accuracy,confidence,positive_rate,prediction_collapse\n"+"".join(f"{scope},{model},{v['auc']},{v['log_loss']},{v['accuracy']},{v['confidence_mean']},{v['positive_rate']},{v['prediction_collapse']}\n" for scope,s in metrics["utility"].items() for model,v in s.items()));atomic_text(run_dir/"mia_metrics.csv","model,primary_auc,absolute_attack_advantage_sensitivity_only\n"+"".join(f"{m},{auc(m)},{metrics['mia']['sensitivity'][m]}\n" for m in MODELS));atomic_text(run_dir/"efficiency_metrics.csv",efficiency_csv(eff));atomic_text(run_dir/"report_v2.md",f"# Four-model Development evaluation\n\nIF-A2: **{metrics['methods']['if_a2']['category']}**  \nE2URec: **{metrics['methods']['e2urec']['category']}**\n\nThe primary Retain-only baseline used five consecutive non-improving validation epochs to reduce premature stopping. A patience-three run is retained as a timing and convergence sensitivity analysis.\n\nPairwise sign: distance(IF-A2, Retrain) - distance(E2URec, Retrain); negative favors IF-A2 and positive favors E2URec.\n");atomic_json(run_dir/"provenance.json",pre);files=["metrics_v2.json","summary_methods.csv","pairwise_if_a2_vs_e2urec.csv","subgroup_metrics.csv","utility_metrics.csv","mia_metrics.csv","efficiency_metrics.csv","per_sample_metrics.jsonl","report_v2.md","provenance.json","contract.json"];publish_manifest(run_dir,files,{"schema":SCHEMA,"test_accessed":False});complete(run_dir,MARKER);return metrics


def synthetic(root:Path,run_name:str)->dict[str,Any]:
    run_dir=safe_new_directory(root,"evaluation_v2_runs",run_name);users=[i//2 for i in range(20)];pair=paired_difference([.1]*20,[.2]*20,users,resamples=100);eff=efficiency_rows({"seconds":1.},{"seconds":10.},{"seconds":2.,"inclusive_seconds":None},authority_mode=False);classes={"A":classify_method(valid=True,forgetting=True,utility=True,mia=True,efficient=True),"B":classify_method(valid=True,forgetting=True,utility=True,mia=False,efficient=True),"C":classify_method(valid=True,forgetting=False,utility=True,mia=True,efficient=True),"D":classify_method(valid=False,forgetting=True,utility=True,mia=True,efficient=True),"unavailable":classify_method(valid=True,forgetting=True,utility=True,mia=True,efficient=False,efficiency_available=False)};result={"schema":SCHEMA,"synthetic":True,"pairwise":pair,"efficiency":eff,"classes":classes,"real_t5_loaded":False,"test_accessed":False};atomic_json(run_dir/"synthetic_result.json",result);complete(run_dir,"PAPER_EVALUATION_V2_SYNTHETIC_COMPLETED");return result


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["Preflight","Full","SyntheticDryRun"],default="Preflight");p.add_argument("--project-root",type=Path,default=Path.cwd());p.add_argument("--config",type=Path);p.add_argument("--method-run-name",default="");p.add_argument("--retrain-run-name",default="");p.add_argument("--retrain-sensitivity-mode",choices=["","Patience3Historical"],default="");p.add_argument("--e2urec-run-name",default="");p.add_argument("--e2urec-authority-mode",choices=["","ReconstructedStep1000"],default="");p.add_argument("--run-name",default="");p.add_argument("--split",default="Development");a=p.parse_args();root=a.project_root.resolve();config=a.config or root/"configs/paper_if_a2_evaluation_v2.yaml"
    if a.mode!="Preflight" and not a.run_name:raise ValueError("RunName is required")
    result=preflight(root,config,a.method_run_name,a.retrain_run_name,a.e2urec_run_name,a.split,a.e2urec_authority_mode,a.retrain_sensitivity_mode) if a.mode=="Preflight" else synthetic(root,a.run_name) if a.mode=="SyntheticDryRun" else run_full(root,config,a.method_run_name,a.retrain_run_name,a.e2urec_run_name,a.run_name,a.e2urec_authority_mode,a.retrain_sensitivity_mode);print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
