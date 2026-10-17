"""Frozen historical rescoring. Development first; explicit separate FinalTest lane."""
from __future__ import annotations
import argparse
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import T5Tokenizer
from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, sha256_file
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from . import ali_stage25_inventory as inventory
from . import ali_stage25_scoring as scoring
from . import ali_stage24_masked_rescore_v1 as s24

SCHEMA="bota-ali-stage25-historical-mask-audit-v1"
CONFIG="configs/ali_stage25_historical_mask_audit_v1.json"
ROOT="outputs/ali_stage25_historical_mask_audit_v1"
read=s24.read


def verify(path,contract=None):
    m=read(path/"manifest.json")
    actual={p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and p.name!="manifest.json"}
    if m["schema"]!=SCHEMA or actual!=set(m["files"]) or (path/"COMPLETED").read_text(encoding="utf-8")!=SCHEMA+"\n":raise ValueError(f"invalid bundle {path}")
    for rel,digest in m["files"].items():
        if sha256_file(path/rel)!=digest:raise ValueError(f"changed artifact {path/rel}")
    r=read(path/"result.json")
    if contract is not None and r["contract"]!=contract:raise ValueError("contract changed")
    return r


def task(root,key,contract,fn,accessed=False):
    dest=root/key
    if dest.exists():return verify(dest,contract)
    stage=root/".work"/uuid.uuid4().hex[:10];stage.mkdir(parents=True)
    started=time.perf_counter();print(f"[Stage25] {key} work={stage}",flush=True)
    try:
        result={**fn(stage),"contract":contract,"test_accessed":accessed,"optimizer_updates":0,
                "wall_seconds":time.perf_counter()-started}
        atomic_json(stage/"result.json",result);atomic_text(stage/"COMPLETED",SCHEMA+"\n")
        atomic_json(stage/"manifest.json",{"schema":SCHEMA,"files":{
            p.relative_to(stage).as_posix():sha256_file(p) for p in sorted(stage.rglob("*")) if p.is_file()}})
        dest.parent.mkdir(parents=True,exist_ok=True);os.rename(stage,dest);return result
    except BaseException as e:
        atomic_json(stage/"FAILED.json",{"error":repr(e),"key":key,"contract":contract,"test_accessed":accessed})
        raise


def load_config(repo):
    c=read(repo/CONFIG)
    if c["schema"]!=SCHEMA or c["output_root"]!=ROOT or c["lambda"]!={"ML1M":64,"Amazon":48} or c["optimizer_updates"]!=0 or c["hyperparameter_selection"] or c["all_candidate_resweep"]:
        raise ValueError("frozen repair scope changed")
    if c["audit"]!=s24.load_config(repo)["audit"]:raise ValueError("audit tolerances changed")
    return c


def prepare(repo,workspace,config):
    print("[Stage25] auditing historical bundles, retained endpoints and frozen cohorts; no model execution",flush=True)
    plan=inventory.build(repo,workspace,config)
    for p in [repo/CONFIG,Path(__file__),Path(inventory.__file__),Path(scoring.__file__),
              repo/"scripts/bota_if/run_ali_stage25_historical_mask_audit_v1.ps1"]:
        plan["hashes"][str(p.resolve())]=sha256_file(p)
    plan.update(schema=SCHEMA,config=config,runtime={**s24.s23.c17.runtime(),
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},selection_performed=False)
    return plan


def check_plan(plan,config):
    if plan["config"]!=config:raise ValueError("frozen config changed")
    runtime={**s24.s23.c17.runtime(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if plan["runtime"]!=runtime:raise ValueError("runtime changed")
    for path,digest in plan["hashes"].items():
        if sha256_file(Path(path))!=digest:raise ValueError(f"frozen source/input changed: {path}")


def authorize(scope,confirm):
    if scope=="FinalTest" and not confirm:
        raise PermissionError("FinalTest scoring requires -ConfirmFinalTest; no cohort was loaded")


def dataset(workspace,cohort,confirm):
    authorize(cohort["scope"],confirm)
    base=s24.s23.runner._base_t5_config(workspace,cohort["benchmark"])
    model_dir=Path(base["paths"]["model_dir"])
    if not model_dir.is_absolute():model_dir=workspace/model_dir
    tok=T5Tokenizer.from_pretrained(str(model_dir),local_files_only=True)
    if cohort["lane"]=="ranking":
        users,movies=inventory.s15.lineage.load_raw_metadata(workspace/"data/ml-1m/raw_data")
        data=inventory.s15.RankingDataset(cohort["plan"],tok,users,movies)
        if len(data)!=len(cohort["indices"]):raise ValueError("candidate count changed")
    else:
        path=Path(cohort["data_path"])
        if cohort.get("data_sha256") and sha256_file(path)!=cohort["data_sha256"]:raise ValueError("FinalTest content changed")
        data=JsonPromptDataset(path,tok)
        if max(cohort["indices"])>=len(data):raise ValueError("cohort indices out of range")
    return data


def read_old(job,cohort):
    spec=job.get("old")
    if not spec:return None
    value=read(Path(spec["path"]))
    for key in spec["keys"]:value=value[key]
    if value is None:return None
    if spec.get("scores_only"):
        if len(value)!=len(cohort["indices"]):raise ValueError("old ranking length changed")
        return {"probability":value}
    if value["sample_order_sha256"]!=canonical_hash(cohort["indices"]):raise ValueError("old cohort order mismatch")
    return value


def delta(old,new):
    fields=("E_U","D_U","local_residual","local_exact_masked_residual","p5_retrain_residual","retain_retrain_absolute_error",
            "overall_auc","overall_log_loss","toward_exact_rate","ndcg@5","ndcg@10","ndcg@20",
            "recall@5","recall@10","recall@20")
    out={}
    for field in fields:
        a,b=old.get(field),new.get(field)
        if field in old or field in new:
            out.update({"legacy_"+field:a,"masked_"+field:b,
                        "delta_"+field:float(b-a) if a is not None and b is not None else None})
    return out


def metric_rows(cohort,jobs,results):
    """Never fill missing methods with legacy predictions or change eligibility."""
    rows=[];by_request={}
    for job in jobs:by_request.setdefault(job["request"],[]).append(job)
    for rid,group in by_request.items():
        available={j["method"]:results[j["id"]]["prediction"] for j in group if results[j["id"]]["status"]=="OK"}
        old={j["method"]:read_old(j,cohort) for j in group if j.get("old")}
        old={k:v for k,v in old.items() if v is not None}
        for job in group:
            result=results[job["id"]]
            common={"cohort":job["cohort"],"dataset":cohort["dataset"],"scope":cohort["scope"],"lane":cohort["lane"],
                    "seed":job["seed"],"request":rid,"method":job["method"],"status":result["status"],
                    "historical_eligible":job["provenance"].get("historical_eligible",False),
                    "calibration_status":job["provenance"].get("calibration_status","FROZEN"),
                    "selection_performed":False,"test_accessed":cohort["scope"]=="FinalTest"}
            if result["status"]!="OK":
                rows.append(common);continue
            method=job["method"];requested=set(job["provenance"]["users"])
            if cohort["lane"]=="ranking":
                metric=inventory.s15.metrics_for(cohort["plan"],available[method]["probability"],requested)
                legacy=inventory.s15.metrics_for(cohort["plan"],old[method]["probability"],requested) if method in old else None
                for g in metric["aggregates"]:
                    ref=next(x for x in legacy["aggregates"] if x["group"]==g["group"]) if legacy else {}
                    rows.append({**common,**g,**delta(ref,g)})
            else:
                local=[i for i,u in enumerate(cohort["users"]) if u in requested]
                if not local:raise ValueError("no requested-user coverage; do not replace request")
                if not {"Original","Exact-Masked"}<=available.keys():raise ValueError("missing corrected references")
                # Generic metrics match historical E_U/AUC/logloss formulas.
                req={"id":rid,"local_indices":[cohort["indices"][i] for i in local]}
                cfg={"selection":cohort.get("selection",{"auc_damage_max":.005,"logloss_damage_max":.01})}
                def basic(preds):
                    return s24.s23.s20.metric(method,method,req,preds[method],preds["Original"],preds["Exact-Masked"],
                                             cohort["indices"],cfg)
                metric=basic(available)
                legacy=basic(old) if {method,"Original","Exact-Masked"}<=old.keys() else (job.get("old_metrics") or {})
                if cohort["dataset"]=="ML1M":
                    p5=inventory.s16.s10._metric(method,available,local,cohort["users"],
                        {"evaluation":inventory.s16.s10.EXPECTED_EVALUATION},job["seed"]*100+inventory.s16.SCENARIOS.index(rid))
                    metric.update(p5)
                    if {method,"Original","Exact-Masked","Retain-Retrain-P5","FullControl-P5"}<=old.keys():
                        legacy.update(inventory.s16.s10._metric(method,old,local,cohort["users"],
                            {"evaluation":inventory.s16.s10.EXPECTED_EVALUATION},job["seed"]*100+inventory.s16.SCENARIOS.index(rid)))
                # Override historical helper's hard-coded seed42/test flag.
                rows.append({**metric,**common,"local_samples":len(local),"supported_users":len({cohort['users'][i] for i in local}),
                             "missing_users":sorted(requested-set(cohort["users"])),**delta(legacy,metric)})
    return rows


def audit_report(plan):
    statuses=Counter(j["status"] for j in plan["jobs"])
    lines=["# Stage25 historical padding audit", "", "Frozen evaluation repair; no retraining, reselection or new requests.",
           "", "## Scoring paths", "",
           "- AFFECTED: evaluation._single_predictions and _ensemble_predictions omit encoder attention_mask.",
           "- AFFECTED: Stage7-9/12-14 Development calibration/reference paths reach the same evaluator.",
           "- AFFECTED: Stage15 score_model explicitly uses the legacy no-mask convention for ranking.",
           "- AFFECTED: Stage16 binary AND ranking call Stage10._predict_endpoint -> common evaluator.",
           "- AFFECTED: Stage18/20/21 Amazon Development prediction uses the same no-mask paths.",
           "- AFFECTED, NOT REPAIRED HERE: Stage17 behavior/ranking/attack call that shared evaluator.",
           "- ALREADY MASKED: trajectory _sample_losses / masked_forward uses encoder attention_mask.",
           "  Parameter-space diagnostics cannot be explained away by the evaluation defect.", "",
           "## This run", "",
           "ML: all 72 frozen main endpoints on Development and Stage16 FinalTest binary/ranking cohorts.",
           "Amazon: Stage20/21 comparisons; Stage20 selected (or prespecified ineligible diagnostic) calibration checkpoints;",
           "Stage18 frozen lambda48 and source-only on K2/K4 across all three seeds.",
           "Historical lambda1 controls in ML remain an internal trace, not a newly chosen method.",
           "No full hyperparameter grid rescoring and no claim that old selections remain optimal.",
           "Development repair does not authorize a new FinalTest-informed selection.", "",
           f"Job inventory: {dict(statuses)}", "", "## Unavailable endpoints", ""]
    for j in plan["jobs"]:
        if j["status"]!="READY":lines.append(f"- {j['cohort']}/{j['request']}/{j['method']}: {j['status']}")
    lines += ["", "Missing Amazon SISA/RecEraser weights were intentionally released by the historical run.",
              "Their stored unmasked predictions cannot be converted into masked predictions.",
              "Restore exact weights or authorize a separately specified reconstruction; never silently mix scores.",
              "Historical numerical solver failures remain failures, not missing-data wins."]
    return "\n".join(lines)+"\n"


def summarize(root,plan,scope):
    contract=canonical_hash(plan);rows=[];missing=[];pred_changes=[]
    cohorts={k:v for k,v in plan["cohorts"].items() if v["scope"]==scope}
    for key,cohort in cohorts.items():
        jobs=[j for j in plan["jobs"] if j["cohort"]==key]
        paths=[root/"cells"/j["id"] for j in jobs]
        if not all(p.exists() for p in paths):
            missing.extend(j["id"] for j,p in zip(jobs,paths) if not p.exists());continue
        results={j["id"]:verify(p,contract) for j,p in zip(jobs,paths)}
        if any(r["status"]=="INVARIANCE_FAILED" for r in results.values()):
            raise RuntimeError("summary blocked: batch invariance failure is not a missing comparator")
        r=task(root,"metrics/"+key,contract,lambda stage:{"rows":metric_rows(cohort,jobs,results)},scope=="FinalTest")
        rows.extend(r["rows"])
        for job in jobs:
            r=results[job["id"]]
            if r.get("legacy_delta") is not None:pred_changes.append({"cohort":key,"request":job["request"],"method":job["method"],**r["legacy_delta"]})
    if missing:return {"status":"INCOMPLETE","scope":scope,"pending_cells":len(missing),"completed_metric_rows":len(rows)}
    def work(stage):
        unavailable=[r for r in rows if r["status"]!="OK"]
        s24.write_csv(stage/"metrics.csv",rows);s24.write_csv(stage/"prediction_changes.csv",pred_changes)
        lines=[f"# Stage25 {scope} masked historical rescoring", "", "Frozen configurations; no retuning. All available methods scored with encoder masks.",
               "Historical eligibility labels are preserved, even if corrected utility now passes.",
               "Missing weights / historical solver failures are NOT wins. No complete-comparator claim for an incomplete group.", "",
               "| Cohort | Request | Method | Status | Legacy E_U | Masked E_U | Legacy residual | Masked residual |",
               "|---|---|---|---|---:|---:|---:|---:|"]
        def fmt(x):return "NA" if x is None else f"{x:.7g}"
        for r in rows:
            if r["lane"]=="binary":lines.append(f"| {r['cohort']} | {r['request']} | {r['method']} | {r['status']} | {fmt(r.get('legacy_E_U'))} | {fmt(r.get('E_U'))} | {fmt(r.get('legacy_local_residual'))} | {fmt(r.get('local_residual'))} |")
        lines += ["", "Ranking, P5 reference, absolute errors, utility and paired legacy deltas are in metrics.csv.",
                  "The repair does not validate optimality of historical tuning, Stage17 stress tests, privacy guarantees, or missing baselines."]
        atomic_text(stage/"REPORT.md","\n".join(lines)+"\n")
        return {"status":"COMPLETED_WITH_UNAVAILABLE" if unavailable else "COMPLETED","scope":scope,
                "rows":rows,"unavailable_rows":len(unavailable),"selection_performed":False}
    return task(root,"summary/"+scope,contract,work,scope=="FinalTest")


def execute(root,workspace,plan,scope,lane,confirm):
    authorize(scope,confirm)
    contract=canonical_hash(plan)
    if scope=="FinalTest":
        task(root,"access/FinalTest",contract,lambda stage:{"status":"EXPLICIT_FROZEN_REEVALUATION",
             "historical_finaltest_exposure":True,"selection_performed":False},True)
    for key,cohort in plan["cohorts"].items():
        if cohort["scope"]!=scope or lane not in ("All",cohort["lane"]):continue
        jobs=[j for j in plan["jobs"] if j["cohort"]==key];data=None
        try:
            for job in jobs:
                path=root/"cells"/job["id"]
                if path.exists():
                    cached=verify(path,contract)
                    if cached["status"]=="INVARIANCE_FAILED":raise RuntimeError("frozen audit failed; diagnose before retrying")
                    continue
                if job["status"]!="READY":
                    task(root,"cells/"+job["id"],contract,lambda stage:{"status":job["status"],"job":job,"prediction":None},scope=="FinalTest")
                    continue
                if data is None:data=dataset(workspace,cohort,confirm)
                def work(stage):
                    def progress(shard,done,total):
                        atomic_json(stage/"progress.json",{"cohort":key,"request":job["request"],"method":job["method"],"shard":shard,"done":done,"total":total})
                        print(f"[Stage25] {key}/{job['request']}/{job['method']} shard={shard} {done}/{total}",flush=True)
                    r=scoring.score_endpoint(workspace,cohort,job,data,plan["config"],progress,read)
                    old=read_old(job,cohort)
                    diff=None
                    if old is not None and r["status"]=="OK":
                        a=np.asarray(old["probability"]);b=np.asarray(r["prediction"]["probability"])
                        if a.shape!=b.shape or not np.isfinite(a).all():raise ValueError("invalid legacy predictions")
                        diff={"max_probability_delta":float(np.abs(a-b).max()),"mean_probability_delta":float(np.abs(a-b).mean())}
                    return {**r,"job":job,"legacy_delta":diff}
                result=task(root,"cells/"+job["id"],contract,work,scope=="FinalTest")
                if result["status"]=="INVARIANCE_FAILED":raise RuntimeError(f"invariance failed: {key}/{job['method']}; diagnostics preserved")
            check_plan(plan,plan["config"])
        finally:
            del data
    return summarize(root,plan,scope)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,default=Path.cwd());p.add_argument("--workspace",type=Path,default=Path("C:/python/E2URec-main"))
    p.add_argument("--mode",choices=["Preflight","Freeze","Full","Analyze"],default="Preflight")
    p.add_argument("--scope",choices=["Development","FinalTest"],default="Development")
    p.add_argument("--lane",choices=["All","binary","ranking"],default="All")
    p.add_argument("--confirm-finaltest",action="store_true");args=p.parse_args()
    if args.mode=="Full":authorize(args.scope,args.confirm_finaltest)
    repo,workspace=args.repo.resolve(),args.workspace.resolve();config=load_config(repo);root=workspace/ROOT
    if (root/"protocol").exists():
        plan=verify(root/"protocol")["plan"];check_plan(plan,config)
    else:plan=prepare(repo,workspace,config)
    contract=canonical_hash(plan)
    if args.mode=="Preflight":
        result={"status":"PREFLIGHT_PASS","jobs":len(plan["jobs"]),"inventory":dict(Counter(j["status"] for j in plan["jobs"])),
                "model_loaded":False,"raw_finaltest_loaded":False,"optimizer_updates":0}
    else:
        with s24.s23.s20.lock(root):
            task(root,"protocol",contract,lambda stage:{"plan":plan,"status":"FROZEN"})
            def audit(stage):
                atomic_text(stage/"REPORT.md",audit_report(plan));s24.write_csv(stage/"endpoint_inventory.csv",plan["jobs"])
                return {"status":"AUDITED","job_status_counts":dict(Counter(j["status"] for j in plan["jobs"]))}
            task(root,"audit",contract,audit)
            if args.mode=="Freeze":result={"status":"FROZEN","contract":contract,"audit":str(root/"audit/REPORT.md")}
            elif args.mode=="Analyze":result=summarize(root,plan,args.scope)
            else:result=execute(root,workspace,plan,args.scope,args.lane,args.confirm_finaltest)
    if "rows" in result:result={k:v for k,v in result.items() if k!="rows"}|{"metric_rows":len(result["rows"]),"report":str(root/"summary"/args.scope/"REPORT.md")}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
