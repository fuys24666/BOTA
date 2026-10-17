"""Separately sealed lambda=64 FinalTest re-evaluation; no tuning/training.

Preflight/Seal never materialize FinalTest. Full requires explicit confirmation.
Resume is only an infrastructure continuation of the identical sealed protocol.
Historical FinalTest exposure is preserved, not relabelled as a new holdout.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import T5Tokenizer

from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, directory_hash, git_snapshot, sha256_file
from . import ali_stage10_finaltest_v1 as s10
from . import ali_stage15_ranking_v1 as s15

SCHEMA = "bota-ali-stage16-finaltest-v1"
ROOT = "outputs/ali_stage16_finaltest_v1"
METHODS = ("Original", "Exact-Masked", "BOTA64", "BOTA1", "BOTA0", "FullControl-P5",
           "Retain-Retrain-P5", "IFRU", "E2URec", "NegGrad", "PCGrad", "RecEraser")
SEEDS, SCENARIOS = s15.SEEDS, s15.SCENARIOS
LANES = ("binary", "ranking")
CAPABILITY = object()
PROTOCOL = {
    "seeds": list(SEEDS), "scenarios": list(SCENARIOS), "methods": list(METHODS),
    "split": "FinalTest", "binary_rows": 20000, "lanes": list(LANES),
    "response_scale": 64, "ablation_scales": [0, 1], "optimizer_updates": 0,
    "hyperparameter_selection": False, "historical_finaltest_exposure": True,
    "request_replacement": False, "primary": "binary_requested_user_mean_absolute_error_to_ExactMasked",
    "binary_utility_guards": {"auc_damage_max": .005, "logloss_damage_max": .01},
    "bootstrap_resamples": 2000, "bootstrap_unit": "authoritative_requested_user",
    "ranking": {
        "positive": "all_FinalTest_rating_ge_4_unseen_before_FinalTest_in_catalogue",
        "catalogue": "items_observed_before_FinalTest",
        "history": "last_10_before_first_FinalTest_event_per_user",
        "negatives": 100, "namespace": "ali-stage15-ranking-v1",
        "exclude": "all_user_observations_through_FinalTest_end",
        "ks": [5, 10, 20], "primary": "sampled_NDCG@10", "secondary": "NDCG@5/20_Recall@5/10/20",
        "no_positive": "exclude_from_macro_and_report", "insufficient_negatives": "use_all_and_report",
        "tie_break": "ascending_movie_id", "groups": ["all", "requested", "retained"],
        "new_acceptance_threshold": None,
    },
    "selection_on_results": False, "inference_batch_size": 4,
    "resume": "explicit_same_seal_only;reuse_completed_cells;restart_incomplete_cells;no_new_cohort",
    "publication": "all_144_cells_required_for_combined_summary",
    "legacy_reproduction": {"atol": 1e-8, "rtol": .001,
        "fields": ["E_U", "D_U", "overall_auc", "overall_log_loss", "p5_retrain_residual"]},
}


def read(path): return s15.read(path)


def load_config(repo):
    config = read(repo / "configs/ali_stage16_finaltest_v1.json")
    if config.get("schema") != SCHEMA or config.get("protocol") != PROTOCOL or config.get("output_root") != ROOT:
        raise ValueError("Stage 16 protocol changed; no within-run amendment allowed")
    if config.get("test_access_policy") != "explicit_separately_sealed_reevaluation":
        raise ValueError("invalid access policy")
    for name, expected in config["bindings"].items():
        if sha256_file(repo / name) != expected:
            raise ValueError(f"frozen binding changed: {name}")
    amendment = read(repo / "configs/ali_stage0_amendment15_v1.json")
    if amendment["protocol"] != PROTOCOL or amendment["status"] != "SPECIFIED_BEFORE_STAGE16_FINALTEST_ACCESS":
        raise ValueError("amendment disagreement")
    return config


def source_hashes(repo):
    paths = list((repo / "src").rglob("*.py")) + list((repo / "data_preprocess").glob("*.py"))
    paths += [p for p in (repo / "configs").rglob("*") if p.is_file()]
    paths += list((repo / "scripts/bota_if").glob("*.ps1"))
    paths += list((repo / "docs/audits").glob("*.json"))
    return {p.relative_to(repo).as_posix(): sha256_file(p) for p in sorted(set(paths))}


def preflight(repo, workspace):
    """Read Development authorities/models only; never inspect a test data file."""
    config = load_config(repo)
    audit = read(repo / "docs/audits/ali_stage15_result_audit_v1.json")
    if audit["status"] != "FROZEN_COMPLETE_STAGE15" or audit["endpoint_cells"] != 66 or audit["lambda_frozen"] != 64:
        raise ValueError("Stage 15 must be completely sealed first")
    root15 = workspace / s15.ROOT
    for relative, expected in audit["bundles"].items():
        path = root15 / relative
        if sha256_file(path / "manifest.json") != expected:
            raise ValueError("sealed Stage 15 manifest changed")
        s15.verify_bundle(path)
    parent = read(root15 / "protocol/contract.json")
    if canonical_hash(parent) != audit["contract_sha256"]:
        raise ValueError("Stage 15 contract changed")
    s15.verify_inputs(parent, workspace)
    config15 = s15.load_config(repo, repo / "configs/ali_stage15_ranking_v1.json")
    config9, entries, dependencies = s15.authority(repo, workspace, config15)
    if entries != parent["endpoints"]:
        raise ValueError("Stage 15 endpoints changed")
    for seed in SEEDS:
        for sid in SCENARIOS:
            base = next(e for e in entries if e["seed"] == seed and e["scenario"] == sid and e["method"] == "BOTA64")
            path = (workspace / base["path"]).parent.parent / "bota_r0/adapter"
            if not (path / "adapter_model.pt").is_file(): raise FileNotFoundError(path)
            entries.append({**base, "method": "BOTA0", "path": path.relative_to(workspace).as_posix(),
                            "directory_sha256": directory_hash(path)})
    expected = {(s, sid, m) for s in SEEDS for sid in SCENARIOS for m in METHODS}
    if len(entries) != 72 or {(e["seed"],e["scenario"],e["method"]) for e in entries} != expected:
        raise ValueError("incomplete Stage 16 endpoint matrix")
    benchmarks = {}
    for seed in SEEDS:
        benchmarks[str(seed)], registry = s15.s9.stage7._authority(repo, workspace, s15.s9._seed_config(config9,seed))
        for e in (e for e in entries if e["seed"] == seed):
            sc = next(r for r in registry["scenarios"] if r["id"] == e["scenario"])
            if e["users"] != sc["users"] or e["registry_sha256"] != registry["registry_sha256"]:
                raise ValueError("request registration changed")
    legacy = workspace / config["legacy_result"]["path"]
    # Only hash the old manifest during admission. Numerical FinalTest results
    # are opened only by post-inference analysis for reproducibility checks.
    if sha256_file(legacy / "manifest.json") != config["legacy_result"]["manifest_sha256"]:
        raise ValueError("historical FinalTest manifest binding changed")
    contract = {"schema":SCHEMA,"protocol":PROTOCOL,"config_sha256":sha256_file(repo/"configs/ali_stage16_finaltest_v1.json"),
        "stage15_audit_sha256":sha256_file(repo/"docs/audits/ali_stage15_result_audit_v1.json"),
        "endpoints":entries,"dependencies":dependencies,"benchmarks":benchmarks,
        "input_sha256":parent["input_sha256"],"source_sha256":source_hashes(repo),
        "runtime":{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"numpy":np.__version__},
        "historical_finaltest_exposure":True,"test_accessed":False}
    return config, contract


def publish(stage, target, accessed):
    if target.exists(): raise FileExistsError(target)
    atomic_text(stage/"COMPLETED", SCHEMA+"\n")
    atomic_json(stage/"manifest.json",{"schema":SCHEMA,"test_accessed":accessed,
        "files":{p.name:sha256_file(p) for p in sorted(stage.iterdir()) if p.is_file()}})
    target.parent.mkdir(parents=True,exist_ok=True)
    os.rename(stage,target)


def verify(path):
    manifest=read(path/"manifest.json")
    if manifest["schema"]!=SCHEMA or (path/"COMPLETED").read_text(encoding="utf-8")!=SCHEMA+"\n":
        raise ValueError("invalid Stage 16 artifact")
    if {p.name for p in path.iterdir()} != set(manifest["files"])|{"manifest.json"}:
        raise ValueError("Stage 16 inventory changed")
    for name,expected in manifest["files"].items():
        if sha256_file(path/name)!=expected: raise ValueError(f"artifact changed: {path/name}")


def seal(repo, workspace, contract):
    root=workspace/ROOT; target=root/"protocol"
    if target.exists():
        verify(target)
        if read(target/"contract.json")!=contract: raise ValueError("sealed code/config/input changed")
    else:
        if (root/"access").exists(): raise ValueError("access exists without a seal; do not reset ledger")
        stage=s15.stage_path(root)
        atomic_json(stage/"contract.json",contract)
        atomic_json(stage/"git_snapshot.json",git_snapshot(repo))
        publish(stage,target,False)
    return canonical_hash(contract)


def check_seal(repo, workspace, contract):
    root=workspace/ROOT; verify(root/"protocol")
    if read(root/"protocol/contract.json")!=contract: raise ValueError("pre-access seal mismatch")
    if source_hashes(repo)!=contract["source_sha256"]: raise ValueError("sealed source inventory/hash changed")
    for name,expected in contract["input_sha256"].items():
        if sha256_file(Path(name))!=expected: raise ValueError("sealed input changed")
    for dep in contract["dependencies"].values():
        path=s15.safe_path(workspace,dep["path"])
        if (directory_hash(path) if path.is_dir() else sha256_file(path))!=dep["sha256"]:
            raise ValueError("model dependency changed")


def begin_access(root, binding, resume, confirm):
    if not confirm: raise PermissionError("Full/Resume requires -ConfirmFinalTest")
    access=root/"access"
    if resume:
        if not (access/"started.json").is_file(): raise ValueError("nothing to resume; use Full")
        if read(access/"started.json")["seal_sha256"]!=binding: raise ValueError("cannot resume a different seal")
    else:
        try: access.mkdir()
        except FileExistsError as e: raise RuntimeError("FinalTest run already started; use Analyze or explicit Resume, never a new Full") from e
        atomic_json(access/"started.json",{"schema":SCHEMA,"seal_sha256":binding,"status":"FINALTEST_ACCESS_STARTED",
            "historical_finaltest_exposure":True,"test_accessed":True})
    return access


def ranking_plan(train, development, final):
    """Same Stage 15 rule, with the cutoff moved to the start of FinalTest."""
    before=list(train)+list(development)
    catalogue={item for row in before for item in [row.movie_id,*row.history_ids]}
    first={}
    for row in final: first.setdefault(row.authoritative_user_id,row)
    for row in first.values(): catalogue.update(row.history_ids)
    plan=s15.build_candidates(before,final,list(range(len(final))),catalogue)
    for row in plan["coverage"]:
        row["finaltest_rows"]=row.pop("development_rows")
        row["evaluation_rows"]=row.pop("select_rows")
    plan.update(protocol=PROTOCOL["ranking"],split="FinalTest",test_accessed=True)
    return plan


def materialize(workspace, root, contract, capability):
    if capability is not CAPABILITY: raise PermissionError("FinalTest capability required")
    raw=workspace/"data/ml-1m/raw_data"
    final=s10.lineage._final_lineage(raw)
    final_path=s10.lineage._validate_final_file(workspace,final)
    train,dev,_=s15.lineage.reconstruct_authoritative_rows(raw)
    plan=ranking_plan(train,dev,final)
    cohort={"schema":SCHEMA,"seal_sha256":canonical_hash(contract),"rows":len(final),
        "final_path":str(final_path),"processed_sha256":sha256_file(final_path),
        "users":[r.authoritative_user_id for r in final],"labels":[r.label for r in final],
        "candidate_sha256":canonical_hash(plan),"test_accessed":True}
    if len(final)!=20000: raise ValueError("FinalTest count mismatch")
    target=root/"cohort"
    if target.exists():
        verify(target)
        if read(target/"cohort.json")!=cohort or read(target/"candidates.json")!=plan:
            raise ValueError("same-seal resume cohort/candidates changed")
    else:
        stage=s15.stage_path(root); atomic_json(stage/"cohort.json",cohort); atomic_json(stage/"candidates.json",plan)
        publish(stage,target,True)
    return cohort,plan


class BinaryDataset(JsonPromptDataset):
    def __init__(self,path,tokenizer,capability):
        if capability is not CAPABILITY: raise PermissionError("FinalTest dataset capability required")
        self.records=read(path);self.tokenizer=tokenizer


class ProgressDataset:
    def __init__(self,dataset,path): self.dataset=dataset;self.path=path;self.passes=0
    def __len__(self): return len(self.dataset)
    def __getitem__(self,i): return self.dataset[i]
    def collate_fn(self,rows):
        start=rows[0]["sample_id"];end=rows[-1]["sample_id"]+1
        if start==0:self.passes+=1
        if start==0 or start%2000==0 or end==len(self):
            atomic_json(self.path,{"pass":self.passes,"done":end,"total":len(self),"test_accessed":True})
            print(f"[Stage16] pass={self.passes} samples={end}/{len(self)}",flush=True)
        return self.dataset.collate_fn(rows)


def validate_prediction(prediction,n,labels=None):
    if prediction.get("sample_order_sha256")!=canonical_hash(list(range(n))): raise ValueError("prediction order mismatch")
    for key in ("probability","gold_label","answer_loss"):
        if len(prediction[key])!=n: raise ValueError("incomplete prediction")
    p=np.asarray(prediction["probability"])
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all() or not np.isfinite(prediction["answer_loss"]).all():
        raise ValueError("nonfinite/invalid frozen endpoint prediction")
    if labels is not None and prediction["gold_label"]!=labels: raise ValueError("prediction labels mismatch")


def identity_for(contract,cohort,entry,lane):
    return {"seal_sha256":canonical_hash(contract),"cohort_sha256":canonical_hash(cohort),"lane":lane,"endpoint":entry}


def cell_path(root,entry,lane): return root/"cells"/lane/f"s{entry['seed']}_{entry['scenario']}_{entry['method']}"


def run_cell(repo,workspace,root,contract,cohort,entry,lane,dataset):
    target=cell_path(root,entry,lane); identity=identity_for(contract,cohort,entry,lane)
    if target.exists():
        verify(target)
        if read(target/"identity.json")!=identity: raise ValueError("completed cell binding changed")
        print(f"[Stage16] verified reuse {lane}/{target.name}",flush=True);return
    stage=s15.stage_path(root);atomic_json(stage/"identity.json",identity);begun=time.perf_counter()
    print(f"[Stage16] {lane}/{target.name} work={stage}",flush=True)
    try:
        wrapped=ProgressDataset(dataset,stage/"progress.json")
        benchmark=contract["benchmarks"][str(entry["seed"])] if str(entry["seed"]) in contract["benchmarks"] else contract["benchmarks"][entry["seed"]]
        torch.cuda.reset_peak_memory_stats()
        prediction=s10._predict_endpoint(workspace,benchmark,entry,wrapped,torch.device("cuda:0"),list(range(len(dataset))))
        validate_prediction(prediction,len(dataset),cohort["labels"] if lane=="binary" else [1]*len(dataset))
        if torch.cuda.max_memory_reserved()/2**30>14:raise RuntimeError("GPU memory contract exceeded")
        atomic_json(stage/"predictions.json",{**prediction,"test_accessed":True})
        atomic_json(stage/"run_state.json",{"status":"COMPLETED","wall_seconds":time.perf_counter()-begun,
            "optimizer_updates":0,"test_accessed":True,"peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30})
        check_seal(repo,workspace,contract)
        if directory_hash(workspace/entry["path"])!=entry["directory_sha256"]:raise ValueError("endpoint changed during inference")
        publish(stage,target,True)
    except Exception as e:
        atomic_json(stage/"failure.json",{"type":type(e).__name__,"message":str(e),"test_accessed":True})
        raise


def binary_rows_for(predictions,users,request,seed,sid):
    selected=[i for i,u in enumerate(users) if u in request]
    if not selected: raise ValueError("no binary requested-user support; no request replacement permitted")
    support=Counter(users[i] for i in selected); rows=[]
    for method in METHODS:
        metric=s10._metric(method,predictions,selected,users,{"evaluation":s10.EXPECTED_EVALUATION},seed*100+SCENARIOS.index(sid))
        rows.append({**metric,"seed":seed,"scenario":sid,"selected_samples":len(selected),
            "requested_users":len(request),"supported_users":len(support),"missing_users":sorted(request-set(support))})
    return rows


def aggregate_binary(rows):
    expected={(s,sid,m) for s in SEEDS for sid in SCENARIOS for m in METHODS}
    if len(rows)!=len(expected) or {(r['seed'],r['scenario'],r['method']) for r in rows}!=expected:
        raise ValueError("incomplete binary metric matrix")
    summary=[]
    for sid in SCENARIOS:
        for method in METHODS:
            cells=[r for r in rows if r["scenario"]==sid and r["method"]==method]
            item={"scenario":sid,"method":method,"seeds":3}
            for field in s10.SUMMARY_FIELDS:
                values=[r[field] for r in cells]
                valid=all(v is not None and math.isfinite(v) for v in values)
                item[field+"_mean"]=float(np.mean(values)) if valid else None
                item[field+"_std"]=float(np.std(values,ddof=1)) if valid else None
            summary.append(item)
    bota=[r for r in rows if r['method']=='BOTA64'];mean=float(np.mean([r['E_U'] for r in bota]))
    comparative={m:mean<float(np.mean([r['E_U'] for r in rows if r['method']==m])) for m in s15.s9.METHODS[1:]}
    utility=all(r['utility_eligible'] and not r['prediction_collapse'] for r in bota)
    residual=all(r['local_exact_masked_residual'] is not None and r['local_exact_masked_residual']<1 for r in bota)
    decision={"status":"LOCAL_COMPARISON_CONFIRMED" if utility and residual and all(comparative.values()) else "LOCAL_COMPARISON_NOT_CONFIRMED",
        "utility_all_six":utility,"residual_below_one_all_six":residual,"mean_EU_below_baselines":comparative,
        "BOTA64_six_cell_mean_EU":mean,"retuning_authorized":False,"historical_finaltest_exposure":True}
    decision['cells_below_all_five_baselines']=sum(all(r['E_U']<next(v['E_U'] for v in rows
        if v['seed']==r['seed'] and v['scenario']==r['scenario'] and v['method']==m)
        for m in s15.s9.METHODS[1:]) for r in bota)
    comparisons=[]
    for r in bota:
        for control in ('BOTA0','BOTA1','Original'):
            old=next(v for v in rows if v['seed']==r['seed'] and v['scenario']==r['scenario'] and v['method']==control)
            comparisons.append({"seed":r['seed'],"scenario":r['scenario'],"control":control,
                "EU_old":old['E_U'],"EU_new":r['E_U'],"EU_absolute_reduction":old['E_U']-r['E_U'],
                "EU_relative_reduction":1-r['E_U']/old['E_U'] if old['E_U'] else None,
                "residual_reduction":old['local_exact_masked_residual']-r['local_exact_masked_residual']
                    if old['local_exact_masked_residual'] is not None and r['local_exact_masked_residual'] is not None else None})
    return summary,decision,comparisons


def summarize_ranking(rows):
    result=[]
    for sid in SCENARIOS:
        for method in METHODS:
            for group in ('all','requested','retained'):
                cells=[r for r in rows if r['scenario']==sid and r['method']==method and r['group']==group]
                if len(cells)!=3 or {r['seed'] for r in cells}!=set(SEEDS): raise ValueError("incomplete ranking matrix")
                item={"scenario":sid,"method":method,"group":group,"users_by_seed":{str(r['seed']):r['users'] for r in cells}}
                for key in (f'{m}@{k}' for m in ('ndcg','recall') for k in s15.KS):
                    values=[r[key] for r in cells]
                    item[key]={"mean":float(np.mean(values)),"sample_sd":float(np.std(values,ddof=1))} if all(v is not None for v in values) else None
                result.append(item)
    return result


def reproduction(workspace,config,rows):
    path=workspace/config['legacy_result']['path'];manifest=read(path/'manifest.json')
    if sha256_file(path/'manifest.json')!=config['legacy_result']['manifest_sha256']:raise ValueError('legacy manifest changed')
    if sha256_file(path/'metrics.json')!=manifest['files']['metrics.json']:raise ValueError('legacy metrics changed')
    old=read(path/'metrics.json')['rows']; checks=[]
    for row in rows:
        if row['method'] in ('BOTA64','BOTA0'):continue
        method='BOTA' if row['method']=='BOTA1' else row['method']
        ref=next(r for r in old if r['seed']==row['seed'] and r['scenario']==row['scenario'] and r['method']==method)
        diffs={}
        for field in PROTOCOL['legacy_reproduction']['fields']:
            a,b=row[field],ref[field]
            delta=abs(a-b) if a is not None and b is not None else None
            passed=a is None and b is None if delta is None else delta<=max(1e-8,.001*abs(b))
            diffs[field]={"absolute_difference":delta,"pass":bool(passed)}
        checks.append({"seed":row['seed'],"scenario":row['scenario'],"method":row['method'],"fields":diffs})
    return {"pass":all(v['pass'] for r in checks for v in r['fields'].values()),"cells":checks}


def analyze(workspace,config,contract):
    root=workspace/ROOT
    if not (root/'cohort').exists():return {"status":"NOT_STARTED","missing_cells":144,"raw_test_materialized":False}
    verify(root/'cohort');cohort=read(root/'cohort/cohort.json');plan=read(root/'cohort/candidates.json')
    if cohort['seal_sha256']!=canonical_hash(contract) or cohort['candidate_sha256']!=canonical_hash(plan):raise ValueError('cohort binding changed')
    missing=[f"{lane}/{cell_path(root,e,lane).name}" for e in contract['endpoints'] for lane in LANES if not cell_path(root,e,lane).exists()]
    # Validate every completed cell even during partial analysis.
    for e in contract['endpoints']:
        for lane in LANES:
            path=cell_path(root,e,lane)
            if path.exists():
                verify(path)
                if read(path/'identity.json')!=identity_for(contract,cohort,e,lane):raise ValueError('cell binding changed')
    if missing:return {"status":"PARTIAL","missing_cells":len(missing),"missing":missing,"raw_test_materialized":False}
    binary=[]; ranking=[]; details=[]
    for seed in SEEDS:
        for sid in SCENARIOS:
            entries=[e for e in contract['endpoints'] if e['seed']==seed and e['scenario']==sid]
            request=set(entries[0]['users']);predictions={}
            for e in entries:
                p=read(cell_path(root,e,'binary')/'predictions.json');validate_prediction(p,cohort['rows'],cohort['labels']);predictions[e['method']]=p
                rp=read(cell_path(root,e,'ranking')/'predictions.json');validate_prediction(rp,plan['candidate_count'],[1]*plan['candidate_count'])
                metrics=s15.metrics_for(plan,rp['probability'],request)
                details.append({"seed":seed,"scenario":sid,"method":e['method'],**metrics})
                ranking.extend({"seed":seed,"scenario":sid,"method":e['method'],**group} for group in metrics['aggregates'])
            binary.extend(binary_rows_for(predictions,cohort['users'],request,seed,sid))
    summary,decision,paired=aggregate_binary(binary); ranking_summary=summarize_ranking(ranking)
    ranking_paired=[]
    for row in (r for r in ranking if r['method']=='BOTA64'):
        for control in ('Original','Exact-Masked','BOTA0','BOTA1','Retain-Retrain-P5'):
            ref=next(r for r in ranking if r['seed']==row['seed'] and r['scenario']==row['scenario']
                and r['group']==row['group'] and r['method']==control)
            ranking_paired.append({**{k:row[k] for k in ('seed','scenario','group','users')},'control':control,
                **{key+'_delta':row[key]-ref[key] if row[key] is not None and ref[key] is not None else None
                    for key in (f'{m}@{k}' for m in ('ndcg','recall') for k in s15.KS)}})
    check=reproduction(workspace,config,binary)
    result={"schema":SCHEMA,"status":"COMPLETED" if check['pass'] else "COMPLETED_REPRODUCTION_REVIEW_REQUIRED",
        "binary_rows":binary,"binary_summary":summary,"decision":decision,"paired_BOTA_comparisons":paired,
        "ranking_rows":ranking,"ranking_summary":ranking_summary,"ranking_paired_deltas":ranking_paired,"legacy_reproduction":check,
        "endpoint_cells":72,"prediction_cells":144,"lambda_frozen":64,"test_accessed":True,
        "historical_finaltest_exposure":True,"retuning_authorized":False,"raw_test_materialized_during_analysis":False}
    target=root/'summary'
    if target.exists():
        verify(target)
        if read(target/'summary.json')!=result:raise ValueError('independent summary recomputation changed')
    else:
        stage=s15.stage_path(root);atomic_json(stage/'summary.json',result);atomic_json(stage/'ranking_per_user.json',details)
        s10._write_csv(stage/'binary_cells.csv',binary);s10._write_csv(stage/'binary_summary.csv',summary)
        s10._write_csv(stage/'ranking_cells.csv',ranking);s10._write_csv(stage/'ranking_summary.csv',ranking_summary)
        s10._write_csv(stage/'BOTA_paired_comparisons.csv',paired);s10._write_csv(stage/'ranking_paired_deltas.csv',ranking_paired)
        lines=['# Stage 16: frozen lambda=64 formal re-evaluation','',f"Status: {result['status']}; binary comparison: {decision['status']}.",'',
            'Previously used FinalTest cohort; not a fresh independent holdout. All 144 prediction cells retained; no tuning or training.',
            'Binary metrics use the original 20,000 prompts. Ranking uses separate common-history sampled candidates, not full catalogue.',
            'P5 sensitivity is a distinct denominator. Seed SD is descriptive; ranking similarity is not privacy forgetting.',
            'No updated claim about long horizons, cumulative deletion, attacks, compression or online latency follows from this evaluation.','',
            '| Request | Method | E_U | Residual | P5 residual | TER | AUC |','|---|---|---:|---:|---:|---:|---:|']
        fmt=lambda x:'NA' if x is None else f'{x:.8f}'
        for r in summary:lines.append('| '+ ' | '.join([r['scenario'],r['method']]+[fmt(r[k+'_mean']) for k in ('E_U','local_exact_masked_residual','p5_retrain_residual','toward_exact_rate','overall_auc')])+' |')
        lines+=['','## Sampled ranking','', '| Request | Method | Group | NDCG@10 | Recall@10 |','|---|---|---|---:|---:|']
        for r in ranking_summary:lines.append('| '+' | '.join([r['scenario'],r['method'],r['group'],fmt(r['ndcg@10']['mean']) if r['ndcg@10'] else 'NA',fmt(r['recall@10']['mean']) if r['recall@10'] else 'NA'])+' |')
        atomic_text(stage/'report.md','\n'.join(lines)+'\n');publish(stage,target,True)
    return {"status":result['status'],"prediction_cells":144,"missing_cells":0,"legacy_reproduction_pass":check['pass'],
        "comparison":decision['status'],"output_root":str(root),"raw_test_materialized":False,"test_predictions_read":True}


def execute(repo,workspace,config,contract,resume,confirm,lane_filter='All',seed_filter='All'):
    if not confirm:raise PermissionError('Full/Resume requires -ConfirmFinalTest')
    check_seal(repo,workspace,contract)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('requires exactly one CUDA GPU')
    free,total=torch.cuda.mem_get_info()
    if free/2**30<10:raise RuntimeError('requires 10 GiB free GPU memory')
    torch.cuda.set_per_process_memory_fraction(min(.88,14/(total/2**30)))
    root=workspace/ROOT;access=begin_access(root,canonical_hash(contract),resume,confirm)
    previous=Path.cwd()
    try:
        cohort,plan=materialize(workspace,root,contract,CAPABILITY)
        print(f"[Stage16] binary={cohort['rows']} ranking_candidates={plan['candidate_count']} ranking_users={len(plan['users'])}",flush=True)
        tokenizer=T5Tokenizer.from_pretrained(str(workspace/contract['dependencies']['pretrained_model_directory']['path']),local_files_only=True)
        if tokenizer.encode('Yes.')[0]!=2163 or tokenizer.encode('No.')[0]!=465:raise ValueError('token identity changed')
        users,movies=s15.lineage.load_raw_metadata(workspace/'data/ml-1m/raw_data')
        datasets={}
        if lane_filter in ('All','binary'):datasets['binary']=BinaryDataset(Path(cohort['final_path']),tokenizer,CAPABILITY)
        if lane_filter in ('All','ranking'):
            datasets['ranking']=s15.RankingDataset(plan,tokenizer,users,movies)
            print('[Stage16] token audit',datasets['ranking'].validate_tokens(),flush=True)
        os.chdir(workspace)
        for lane in LANES:
            if lane not in datasets:continue
            for entry in sorted(contract['endpoints'],key=lambda e:(e['seed'],e['scenario'],METHODS.index(e['method']))):
                if seed_filter!='All' and entry['seed']!=int(seed_filter):continue
                atomic_json(access/'progress.json',{"lane":lane,"seed":entry['seed'],"scenario":entry['scenario'],"method":entry['method'],"test_accessed":True})
                run_cell(repo,workspace,root,contract,cohort,entry,lane,datasets[lane])
        check_seal(repo,workspace,contract)
        result=analyze(workspace,config,contract)
        atomic_json(access/'last_outcome.json',result)
        return result
    except Exception as e:
        atomic_json(access/('failure_'+str(time.time_ns())+'.json'),{"type":type(e).__name__,"message":str(e),
            "test_accessed":True,"reselection_allowed":False,"recovery":"explicit Resume with identical seal only"})
        raise
    finally:
        os.chdir(previous);gc.collect();torch.cuda.empty_cache()


def synthetic():
    p=np.linspace(.15,.85,32);users=[i//4 for i in range(32)];pred={}
    for method in METHODS:
        offset={'Original':0,'Exact-Masked':.01,'FullControl-P5':.02,'Retain-Retrain-P5':.03,'BOTA64':.009,'BOTA1':.008,'BOTA0':.007}.get(method,.04)
        pred[method]={"probability":(p+offset).tolist(),"gold_label":[i%2 for i in range(32)],"answer_loss":[1.]*32,"sample_order_sha256":canonical_hash(list(range(32)))}
    rows=[]
    for seed in SEEDS:
        for sid in SCENARIOS:rows.extend(binary_rows_for(pred,users,{0,1},seed,sid))
    _,decision,_=aggregate_binary(rows)
    return {"status":"SYNTHETIC_PASS","synthetic_cells":len(rows),"decision":decision['status'],"actual_test_accessed":False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--mode',choices=('Preflight','Seal','SyntheticDryRun','Full','Resume','Analyze'),default='Preflight')
    p.add_argument('--confirm-final-test',action='store_true');p.add_argument('--lane',choices=('All',*LANES),default='All')
    p.add_argument('--seed',choices=('All','41','42','43'),default='All');a=p.parse_args()
    repo,workspace=a.repo.resolve(),a.workspace.resolve()
    if a.mode in ('Full','Resume') and not a.confirm_final_test:raise PermissionError('Full/Resume requires -ConfirmFinalTest')
    if a.mode=='SyntheticDryRun':load_config(repo);result=synthetic()
    else:
        with s15.s14.execution_lock(workspace/ROOT):
            config,contract=preflight(repo,workspace)
            if a.mode=='Preflight':result={"status":"PREFLIGHT_PASS","endpoints":72,"prediction_cells":144,"test_accessed":False,"test_loader_built":False}
            elif a.mode=='Seal':result={"status":"SEALED","seal_sha256":seal(repo,workspace,contract),"test_accessed":False}
            elif a.mode=='Analyze':check_seal(repo,workspace,contract);result=analyze(workspace,config,contract)
            else:result=execute(repo,workspace,config,contract,a.mode=='Resume',a.confirm_final_test,a.lane,a.seed)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
