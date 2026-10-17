"""Frozen post-hoc BOTA single/pair diagnosis; Development only, no retuning."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import os
import platform
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, sha256_file, seed_everything
from src.paper_if_a2.artifacts import atomic_torch_save
from . import ali_stage23_decomposition as math23
from . import ali_stage17_common as c17
from . import ali_stage18_amazon_lambda_search_v2 as s18
from . import ali_stage20_amazon_baselines_v1 as s20
from . import ali_stage21_amazon_seed43_kernel_v1 as k21
from . import runner, evaluation

SCHEMA = "bota-ali-stage23-pair-decomposition-v1"
CONFIG = "configs/ali_stage23_pair_decomposition_v1.json"
ROOT = "outputs/ali_stage23_pair_decomposition_v1"
read = s18.read


def load_config(repo):
    c = read(repo/CONFIG)
    for key, value in {"schema": SCHEMA, "output_root": ROOT, "datasets": ["Amazon", "ML1M"],
                       "seeds": [41,42,43], "lambda": {"Amazon":48,"ML1M":64}, "ablation_lambda":0,
                       "pairs_per_dataset":3, "optimizer_steps":200, "batch_size":16,
                       "selection_performed":False, "final_test_forbidden":True,
                       "test_accessed":False, "diagnostic_only":True}.items():
        if c.get(key) != value: raise ValueError(f"Stage23 scope changed: {key}")
    if c["evaluation"] != dict(split="Development", all_requested_user_samples=True, matched_samples_per_user=5,
                              nonrequested_control_samples=512, selection_salt="stage23-metadata-only-v1", inference_batch_size=4):
        raise ValueError("frozen evaluation scope changed")
    return c


def verify(path, contract=None):
    m = read(path/"manifest.json")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if m["schema"] != SCHEMA or actual != set(m["files"]) or (path/"COMPLETED").read_text(encoding="utf-8") != SCHEMA+"\n":
        raise ValueError(f"invalid Stage23 bundle: {path}")
    for rel, digest in m["files"].items():
        if sha256_file(path/rel) != digest: raise ValueError(f"changed artifact: {path/rel}")
    r = read(path/"result.json")
    if contract is not None and r["contract"] != contract: raise ValueError(f"contract changed: {path}")
    return r


def task(root, key, contract, fn):
    dest = root/key
    if dest.exists():
        out = verify(dest, contract); print(f"[Stage23] reused {key}", flush=True); return out
    stage = root/".work"/uuid.uuid4().hex[:10]; stage.mkdir(parents=True)
    started = time.perf_counter()
    print(f"[Stage23] start {key}; work={stage}", flush=True)
    try:
        out = {**fn(stage), "contract": contract, "test_accessed": False, "wall_seconds": time.perf_counter()-started}
        atomic_json(stage/"result.json", out); atomic_text(stage/"COMPLETED", SCHEMA+"\n")
        atomic_json(stage/"manifest.json", {"schema": SCHEMA, "files": {
            p.relative_to(stage).as_posix(): sha256_file(p) for p in sorted(stage.rglob("*")) if p.is_file()}})
        dest.parent.mkdir(parents=True, exist_ok=True); os.rename(stage, dest)
        return out
    except BaseException as error:
        atomic_json(stage/"FAILED.json", {"key": key, "error": repr(error), "contract": contract,
                    "interrupted": isinstance(error, KeyboardInterrupt), "test_accessed": False})
        print(f"[Stage23] incomplete unit retained: {stage}", flush=True)
        raise


def audit(repo, workspace, c):
    print("[Stage23] building metadata-only cohorts and hashing sources; no model execution", flush=True)
    inputs = set()
    def add(path):
        path = Path(path).resolve()
        if not path.exists(): raise FileNotFoundError(path)
        if path.is_dir():
            inputs.update(p for p in path.rglob("*") if p.is_file())
        else: inputs.add(path)
    for rel, digest in c["parent_bindings"].items():
        if sha256_file(workspace/rel) != digest: raise ValueError(f"parent changed: {rel}")
        add(workspace/rel)
    ar = workspace/"outputs/ali_stage20_amazon_baselines_v1"
    br = workspace/"outputs/ali_stage21_amazon_seed43_v1"
    mr = workspace/"outputs/ali_stage17_lambda64_suite_v1"
    s20.verify(ar/"protocol"); k21.verify(br/"protocol"); c17.verify(mr/"protocol")
    amz = read(ar/"protocol/result.json"); amz43 = read(br/"protocol/result.json")
    ml = read(mr/"protocol/contract.json")
    contexts = {}
    for dataset in c["datasets"]:
        b0 = amz["benchmark"] if dataset == "Amazon" else ml["contexts"]["42"]["benchmark"]
        if dataset == "Amazon":
            tu = list(map(int, read(workspace/b0["source"]["train_user_ids"])))
            du = list(map(int, read(workspace/b0["source"]["development_user_ids"])))
            raw_pairs = [{"id": r["id"], "users": r["users"], "roles": [m["role"] for m in r["metadata"]]}
                         for r in amz["comparison"]["requests"]]
            for field in ("train_user_ids", "development_user_ids"): add(workspace/b0["source"][field])
        else:
            tu, du, lineage = c17.s2._source_users(workspace, b0)
            add(workspace/b0["source"]["raw_data"])
            raw_pairs = math23.ml_pairs(ml["contexts"]["42"]["registry"], tu, du)
        for seed in c["seeds"]:
            b = copy.deepcopy(b0 if dataset == "Amazon" else ml["contexts"][str(seed)]["benchmark"])
            b["protocol"]["run_seed"] = seed; b["coordinate"]["fixed_a_seed"] = seed
            reg = amz["registry"] if dataset == "Amazon" else ml["contexts"][str(seed)]["registry"]
            if len(reg["order"]) != 3200 or len(tu) != reg["global_train_samples"] or len(du) != reg["development_samples"]:
                raise ValueError("source/order counts changed")
            pairs, indices, control = math23.enrich_pairs(raw_pairs, tu, du, reg["order"], c["evaluation"])
            ctx = {"dataset": dataset, "seed": seed, "benchmark": b, "registry": reg, "pairs": pairs,
                   "lambda": c["lambda"][dataset], "indices": indices, "control_indices": control,
                   "train_users_sha256": canonical_hash(tu), "development_users_sha256": canonical_hash(du),
                   "historical_joint": {}, "historical_bota": {}}
            if dataset == "Amazon":
                bank18 = workspace/"outputs/ali_stage18_amazon_lambda_search_v2/banks"/("coarse_42" if seed==42 else f"confirm_{seed}")
                s18.verify(bank18)
                original = bank18/"Original/adapter"
                add(bank18/"manifest.json"); add(bank18/"result.json")
                if seed in (42,43):
                    oldroot, p, verifier = (ar,amz,s20.verify) if seed==42 else (br,amz43,k21.verify)
                    verifier(oldroot/"bank"); add(oldroot/"bank/manifest.json")
                    for r in pairs:
                        rid = r["id"]; oldr = next(x for x in p["comparison"]["requests"] if x["id"] == rid)
                        if oldr["users"] != r["users"] or oldr["forget_train_indices"] != r["forget_train_indices"]:
                            raise ValueError("Amazon request lineage changed")
                        dest = oldroot/"exact"/rid; verifier(dest)
                        identity = read(dest/"result.json")["identity"]
                        if identity != {"contract": canonical_hash(p), "request": canonical_hash(oldr)}:
                            raise ValueError("historical Exact contract mismatch")
                        ctx["historical_joint"][rid] = str(dest/"adapter/adapter_model.pt"); add(dest)
                        for scale in (0,48):
                            path = oldroot/"bank"/rid/s18.cid(scale)/"adapter/adapter_model.pt"
                            add(path); ctx["historical_bota"][f"{rid}_{scale}"] = str(path)
            else:
                mctx = ml["contexts"][str(seed)]
                original = workspace/next(e["path"] for e in mctx["endpoints"] if e["method"]=="Original" and e["scenario"]=="L8")
                bank = mr/"cells"/f"bank_primary_s{seed}"; c17.verify(bank)
                if read(bank/"identity.json")["contract_sha256"] != canonical_hash(ml):
                    raise ValueError("historical bank contract mismatch")
                ctx["historical_bank"] = str(bank/"bank.pt"); add(bank)
            add(original); ctx["original"] = str(original/"adapter_model.pt")
            contexts[f"{dataset}_s{seed}"] = ctx
        for field in ("train_json", "development_json", "original_checkpoint", "base_config"):
            add(workspace/b0["source"][field])
        runtime = yaml.safe_load((workspace/b0["source"]["base_config"]).read_text(encoding="utf-8"))
        add(workspace/runtime["base_config"])
        model_dir = Path(runner._base_t5_config(workspace,b0)["paths"]["model_dir"])
        add(model_dir if model_dir.is_absolute() else workspace/model_dir)
    # Bind current code explicitly; historical canonical replay is an independent
    # hard gate against any runtime implementation drift.
    sources = list((repo/"src").rglob("*.py"))+[repo/CONFIG, repo/"scripts/bota_if/run_ali_stage23_pair_decomposition_v1.ps1"]
    forbidden = [p for p in inputs if "finaltest" in str(p).lower() or p.name.lower() in ("test.json","test.pkl")]
    if forbidden: raise ValueError(f"FinalTest input outside scope: {forbidden}")
    # Where ancestors bind the same files, reject changes instead of resealing.
    inherited = {**ml["hashes"], **amz["checks"]["input_sha256"], **amz43["checks"]["input_sha256"]}
    hashes = {}
    for path in sorted(inputs):
        h = sha256_file(path)
        if str(path) in inherited and inherited[str(path)] != h: raise ValueError(f"inherited input changed: {path}")
        hashes[str(path)] = h
    return {"schema": SCHEMA, "config": c, "contexts": contexts, "input_sha256": hashes,
            "source_sha256": {str(p.resolve()):sha256_file(p) for p in sorted(set(sources))},
            "runtime": {**c17.runtime(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "test_accessed": False, "selection_performed": False,
            "scope": "Post-hoc BOTA diagnosis, not independent validation. Cross-dataset differences include exposure, checkpoint and optimizer differences; seeds change fixed-A and training RNG jointly. IFRU is not rerun."}


def check_inputs(protocol):
    for group in ("input_sha256", "source_sha256"):
        for path, digest in protocol[group].items():
            if sha256_file(Path(path)) != digest: raise ValueError(f"frozen input/source changed: {path}")


def seal(root, protocol):
    identity = canonical_hash(protocol)
    if (root/"protocol").exists():
        if verify(root/"protocol", identity)["protocol"] != protocol: raise ValueError("frozen plan changed")
    else: task(root, "protocol", identity, lambda stage: {"status":"FROZEN", "protocol":protocol})
    return identity


def state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def flatten(values, names):
    return runner.flatten_tensors([values[n] for n in names])


def same_map(left, right):
    return set(left)==set(right) and all(torch.equal(left[n],right[n]) for n in left)


def fresh(workspace, ctx):
    device = s18.gpu(); seed_everything(ctx["seed"])
    original = state(ctx["original"])
    values = s20.runtime_context(workspace, ctx["benchmark"], ctx["registry"], device, original["A"])
    model,names,params,bases,_,tok,data,users = values
    if not same_map({n:v.detach().cpu() for n,v in bases.items()}, original["A"]): raise ValueError("fixed-A coordinate changed")
    if canonical_hash(list(map(int,users))) != ctx["train_users_sha256"]: raise ValueError("training users changed")
    return model,names,params,bases,tok,data,users,device,original


class Budget(runner.StepBudget):
    def step(self, opt, arm):
        super().step(opt,arm)
        if self.calls==1 or self.calls%25==0: print(f"[Stage23] {arm} optimizer_step={self.calls}/200",flush=True)


def replay(root, workspace, key, ctx, users, contract, expected=None):
    def work(stage):
        model,names,params,bases,tok,data,ids,device,original = fresh(workspace,ctx)
        try:
            budget = Budget(200)
            values,opt,trace = runner._train_named(model,data,ctx["registry"]["order"],ids,users,names,params,
                device,tok.pad_token_id,runner._engine(workspace,ctx["benchmark"]),budget,key)
            masked = sum(int(ids[i]) in users for i in ctx["registry"]["order"])
            if budget.calls!=200 or sum(r["masked_slots"] for r in trace)!=masked: raise RuntimeError("replay schedule/mask changed")
            if any(not torch.isfinite(v).all() for v in values.values()): raise RuntimeError("nonfinite exact reference")
            if expected is not None and not same_map(values,expected): raise RuntimeError("no-delete replay does not reproduce historical Original")
            runner._save_fixed_ab(stage/"adapter",names,values,bases,"Stage23-Exact-Diagnostic")
            del opt
            return {"status":"COMPLETED", "users":users,"optimizer_steps":200,"masked_slots":masked,
                    "trace_sha256":canonical_hash(trace),"original_reproduction":expected is not None}
        finally:
            del model,params,bases,tok,data; s18.release()
    task(root,key,contract,work)
    return root/key/"adapter/adapter_model.pt"


def build_bank(root, workspace, prefix, ctx, contract):
    key = prefix+"/bank"
    def work(stage):
        targets = [u for r in ctx["pairs"] for u in r["users"]]
        original = state(ctx["original"])
        if ctx["dataset"] == "ML1M":
            old = state(ctx["historical_bank"])
            if not same_map(old["canonical"],original["B"]) or not same_map(old["bases"],original["A"]):
                raise ValueError("historical bank coordinate mismatch")
            slots = [old["users"].index(u) for u in targets]
            vectors={}; failures={}
            for scale, arm in ((0,"FS0"),(64,"FS64")):
                if arm in old["failures"]: failures[str(scale)] = old["failures"][arm]
                else: vectors[str(scale)] = old["vectors"][arm][slots]
            bank={"users":targets,"names":old["names"],"vectors":vectors,"failures":failures}
            atomic_torch_save(stage/"bank.pt",bank)
            return {"status":"REUSED", "failures":failures,"optimizer_steps":0,"source":ctx["historical_bank"]}
        model,names,params,bases,tok,data,ids,device,original = fresh(workspace,ctx)
        try:
            canonical,vectors,failures,steps,control = s18.transport(model,data,ctx["registry"]["order"],ids,targets,params,names,
                device,tok.pad_token_id,runner._engine(workspace,ctx["benchmark"]),[0,48],stage/"progress.json")
            if steps!=200 or not same_map(canonical,original["B"]) or control["status"]!="TENSOR_EXACT_PASS":
                raise RuntimeError("transport/canonical control failure")
            checks={}
            for request in ctx["pairs"]:
                slots=[targets.index(u) for u in request["users"]]
                for scale in (0,48):
                    arm=s18.cid(scale); oldpath=ctx["historical_bota"].get(f"{request['id']}_{scale}")
                    if oldpath and arm not in failures:
                        candidate=s18.compose(canonical,names,vectors[arm][slots].sum(0))
                        historical=state(oldpath)
                        if not same_map(historical["A"],original["A"]): raise ValueError("historical BOTA A changed")
                        if not same_map(candidate,historical["B"]): raise RuntimeError("BOTA joint reproduction not tensor exact")
                        checks[f"{request['id']}_{scale}"]="TENSOR_EXACT_PASS"
            bank={"users":targets,"names":list(names),"vectors":{str(s):vectors[s18.cid(s)] for s in (0,48) if s18.cid(s) not in failures},
                  "failures":{str(s):failures[s18.cid(s)] for s in (0,48) if s18.cid(s) in failures}}
            atomic_torch_save(stage/"bank.pt",bank)
            return {"status":"COMPLETED_WITH_FAILURES" if bank["failures"] else "COMPLETED", "failures":bank["failures"],
                    "reproduction":checks,"optimizer_steps":steps,"control":control}
        finally:
            del model,params,bases,tok,data; s18.release()
    task(root,key,contract,work)
    return state(root/key/"bank.pt")


def make_endpoints(original, names, bank, single, pair, requests):
    out={"Original": original["B"]}
    for u, value in single.items(): out[f"Exact_u{u}"]=value
    for request in requests:
        rid=request["id"]; u,v=request["users"]
        out[f"Exact_{rid}"]=pair[rid]
        # Sum *parameter changes*, then evaluate the resulting model. Never sum probabilities.
        out[f"OracleSum_{rid}"]={n:(single[u][n].double()+single[v][n].double()-original["B"][n].double()).float() for n in names}
    for scale, vectors in bank["vectors"].items():
        for u in bank["users"]:
            out[f"B{scale}_u{u}"]=s18.compose(original["B"],names,vectors[bank["users"].index(u)])
        for request in requests:
            delta=vectors[[bank["users"].index(u) for u in request["users"]]].sum(0)
            out[f"B{scale}_{request['id']}"]=s18.compose(original["B"],names,delta)
    return out


def parameter_analysis(root,prefix,contract,ctx,endpoints,bank,names,config):
    def work(stage):
        o=flatten(endpoints["Original"],names); rows=[]
        for request in ctx["pairs"]:
            rid=request["id"];u,v=request["users"]
            for scale in (0,ctx["lambda"]):
                if str(scale) in bank["failures"]:
                    rows.append({"request":rid,"lambda_value":scale,"status":"NUMERICAL_TRANSPORT_FAILURE","failure":bank["failures"][str(scale)]});continue
                result=math23.decompose(o,flatten(endpoints[f"Exact_u{u}"],names),flatten(endpoints[f"Exact_u{v}"],names),
                    flatten(endpoints[f"Exact_{rid}"],names),flatten(endpoints[f"B{scale}_u{u}"],names),
                    flatten(endpoints[f"B{scale}_u{v}"],names),flatten(endpoints[f"B{scale}_{rid}"],names),config["ratio_norm_floor"])
                result.update(request=rid,users=[u,v],dataset=ctx["dataset"],seed=ctx["seed"],lambda_value=scale,status="OK")
                result["oracle_sum_error_norm"]=float(torch.linalg.vector_norm(flatten(endpoints[f"OracleSum_{rid}"],names)-flatten(endpoints[f"Exact_{rid}"],names)))
                rows.append(result)
        return {"status":"COMPLETED", "rows":rows,"failures":bank["failures"]}
    return task(root,prefix+"/parameters",contract,work)


def predict_all(root,workspace,prefix,ctx,contract,endpoints,config):
    remaining=[key for key in endpoints if not (root/prefix/"predictions"/key).exists()]
    for key in endpoints:
        path=root/prefix/"predictions"/key
        if path.exists(): verify(path,contract)
    if not remaining:return
    model,names,params,bases,tok,data,ids,device,original=fresh(workspace,ctx)
    dev=s18.s8.JsonPromptDataset(workspace/ctx["benchmark"]["source"]["development_json"],tok)
    try:
        model.eval()
        for key in remaining:
            def work(stage):
                values=endpoints[key]
                if any(not torch.isfinite(v).all() for v in values.values()):
                    return {"status":"NONFINITE_ENDPOINT","prediction":None}
                with torch.no_grad():
                    for name,param in zip(names,params):param.copy_(values[name].to(param))
                pred={"probability":[],"gold_label":[],"answer_loss":[],"sample_order_sha256":canonical_hash(ctx["indices"])}
                for start in range(0,len(ctx["indices"]),256):
                    piece=evaluation._single_predictions(model,dev,ctx["indices"][start:start+256],device,config["evaluation"]["inference_batch_size"])
                    for field in ("probability","gold_label","answer_loss"):pred[field].extend(piece[field])
                    print(f"[Stage23] {prefix}/{key} samples={min(start+256,len(ctx['indices']))}/{len(ctx['indices'])}",flush=True)
                valid=s18.s14.validate_prediction(pred,ctx["indices"])
                return {"status":"OK" if valid else "NONFINITE_PREDICTIONS","prediction":pred if valid else None,
                        "endpoint_sha256":s18.s8.tensor_tree_hash(values)}
            task(root,prefix+"/predictions/"+key,contract,work)
    finally:
        del model,params,bases,tok,data,dev;s18.release()


def behavioral_analysis(root,prefix,ctx,contract,endpoints,bank):
    def work(stage):
        preds={key:verify(root/prefix/"predictions"/key,contract)["prediction"] for key in endpoints}
        original=preds["Original"]
        if original is None:raise RuntimeError("nonfinite Original predictions")
        lookup={i:j for j,i in enumerate(ctx["indices"])}
        rows=[]
        def score(key, exactkey, groups, unit):
            exact=preds[exactkey];pred=preds.get(key)
            if exact is None:raise RuntimeError("nonfinite Exact predictions")
            if pred is None:return {**unit,"method":key,"status":"UNAVAILABLE_OR_NONFINITE"}
            for other in (exact,pred):
                if other["gold_label"]!=original["gold_label"] or other["sample_order_sha256"]!=original["sample_order_sha256"]:
                    raise ValueError("prediction order/labels mismatch")
            groups={**groups,"diagnostic_control":ctx["control_indices"]}
            metrics={label:math23.probability_metrics(pred["probability"],original["probability"],exact["probability"],[lookup[i] for i in ix])
                     for label,ix in groups.items()}
            return {**unit,"method":key,"status":"OK","metrics":metrics}
        for request in ctx["pairs"]:
            rid=request["id"]
            groups={"all_local":request["local_indices"],"matched_5_per_user":request["matched_indices"]}
            for member in request["members"]:groups[f"user_{member['user']}"]=member["indices"]
            for key in ("Original",f"OracleSum_{rid}",f"B0_{rid}",f"B{ctx['lambda']}_{rid}"):
                rows.append(score(key,f"Exact_{rid}",groups,{"request":rid,"kind":"pair"}))
            for member in request["members"]:
                u=member["user"];groups={"all_local":member["indices"],"matched_5_per_user":member["matched_indices"]}
                for key in ("Original",f"B0_u{u}",f"B{ctx['lambda']}_u{u}"):
                    rows.append(score(key,f"Exact_u{u}",groups,{"request":rid,"kind":"single","user":u}))
        return {"status":"COMPLETED","rows":rows,"dataset":ctx["dataset"],"seed":ctx["seed"],
                "control_is_not_full_utility_evaluation":True,"failures":bank["failures"]}
    return task(root,prefix+"/behavior",contract,work)


def run_context(root,workspace,ctx,contract,config):
    prefix=f"{ctx['dataset']}_s{ctx['seed']}"
    original=state(ctx["original"])
    replay(root,workspace,prefix+"/control",ctx,[],contract,original["B"])
    bank=build_bank(root,workspace,prefix,ctx,contract); names=bank["names"]
    single={};pairs={}
    for request in ctx["pairs"]:
        for u in request["users"]:
            path=replay(root,workspace,prefix+f"/exact/u{u}",ctx,[u],contract)
            value=state(path)
            if not same_map(value["A"],original["A"]):raise ValueError("single coordinate mismatch")
            single[u]=value["B"]
        rid=request["id"]
        if rid in ctx["historical_joint"]:
            path=Path(ctx["historical_joint"][rid])
        else:path=replay(root,workspace,prefix+f"/exact/{rid}",ctx,request["users"],contract)
        value=state(path)
        if not same_map(value["A"],original["A"]):raise ValueError("pair coordinate mismatch")
        pairs[rid]=value["B"]
    endpoints=make_endpoints(original,names,bank,single,pairs,ctx["pairs"])
    parameter_analysis(root,prefix,contract,ctx,endpoints,bank,names,config)
    predict_all(root,workspace,prefix,ctx,contract,endpoints,config)
    behavioral_analysis(root,prefix,ctx,contract,endpoints,bank)


def analyze(root,protocol):
    contract=canonical_hash(protocol);missing=[];parameters=[];behavior=[]
    for key,ctx in protocol["contexts"].items():
        for suffix,acc in (("parameters",parameters),("behavior",behavior)):
            path=root/key/suffix
            if not path.exists():missing.append(key+"/"+suffix)
            else:
                value=verify(path,contract)
                acc.extend({**row,"dataset":ctx["dataset"],"seed":ctx["seed"]} for row in value["rows"])
    if missing:return {"status":"INCOMPLETE","missing":missing,"parameter_rows":len(parameters),"behavior_rows":len(behavior)}
    def work(stage):
        flat=[]
        for row in behavior:
            if row["status"]!="OK":flat.append({k:v for k,v in row.items() if k!="metrics"});continue
            for scope,metrics in row["metrics"].items():flat.append({**{k:v for k,v in row.items() if k!="metrics"},"scope":scope,**metrics})
        def write_csv(name,rows):
            fields=sorted({k for r in rows for k,v in r.items() if not isinstance(v,(dict,list))})
            with (stage/name).open("w",encoding="utf-8-sig",newline="") as handle:
                writer=csv.DictWriter(handle,fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
        write_csv("parameter_decomposition.csv",parameters);write_csv("behavior_metrics.csv",flat)
        lines=["# Stage23 single/pair decomposition", "", "Post-hoc diagnostic; no tuning, FinalTest, or privacy claim.", "",
               "| Dataset | Seed | Request | Lambda | Single-error sum norm | Interaction norm | Pair-error norm |", "|---|---:|---|---:|---:|---:|---:|"]
        for r in parameters:
            if r["status"]=="OK":lines.append(f"| {r['dataset']} | {r['seed']} | {r['request']} | {r['lambda_value']} | {r['summed_single_error_norm']:.6g} | {r['interaction_norm']:.6g} | {r['pair_error_norm']:.6g} |")
            else:lines.append(f"| {r['dataset']} | {r['seed']} | {r['request']} | unavailable | failure retained | | |")
        lines += ["", "Read behavior_metrics.csv alongside parameter errors. OracleSum evaluates a model assembled from exact single-user parameter changes, not summed probabilities.",
                  "Squared error cross terms can cancel. Do not interpret norm ratios as causal contribution percentages.",
                  "Cross-dataset exposures differ (Amazon 2/user, ML1M 1/user). Checkpoints, optimizers and calibrated lambdas also differ. This is not an isolated dataset intervention.",
                  "Seeds change fixed-A and training RNG jointly. Only three fixed pairs per dataset; no universal grouping rule is established.",
                  "All seeds including 41 and all failed arms must remain visible. IFRU is not rerun; no updated baseline claim."]
        atomic_text(stage/"REPORT.md","\n".join(lines)+"\n")
        return {"status":"COMPLETED","parameters":parameters,"behavior":behavior,"selection_performed":False}
    return task(root,"summary",contract,work)


def smoke():
    o=torch.zeros(3);u=torch.tensor([1.,0.,0.]);v=torch.tensor([0.,1.,0.]);pair=u+v
    r=math23.decompose(o,u,v,pair,u,v,pair)
    assert r["interaction_norm"]==0 and r["pair_error_norm"]==0
    r=math23.decompose(o,u,v,pair+1,u,v,pair)
    assert r["identity_closure_norm"]==0 and r["interaction_norm"]>0
    return {"status":"SMOKE_PASS","model_loaded":False,"test_accessed":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,default=Path.cwd());parser.add_argument("--workspace",type=Path,default=Path("C:/python/E2URec-main"))
    parser.add_argument("--mode",choices=["Preflight","Freeze","Smoke","Full","Analyze"],default="Preflight")
    parser.add_argument("--dataset",choices=["All","Amazon","ML1M"],default="All");parser.add_argument("--seed",choices=["All","41","42","43"],default="All")
    args=parser.parse_args();repo=args.repo.resolve();workspace=args.workspace.resolve();root=workspace/ROOT
    if args.mode=="Smoke":result=smoke()
    else:
        config=load_config(repo)
        if (root/"protocol").exists():
            protocol=verify(root/"protocol")["protocol"]
            if protocol["config"]!=config:raise ValueError("frozen config changed")
            if protocol["runtime"]!={**c17.runtime(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}:raise ValueError("runtime changed")
            check_inputs(protocol)
        else:protocol=audit(repo,workspace,config)
        if args.mode=="Preflight":
            result={"status":"PREFLIGHT_PASS","contexts":{k:{"pairs":[{"id":r["id"],"users":r["users"],"deleted_interactions":r["deleted_interactions"]} for r in v["pairs"]],"samples":len(v["indices"])} for k,v in protocol["contexts"].items()},"model_loaded":False,"test_accessed":False}
        else:
            with s20.lock(root):
                if args.mode=="Analyze":result=analyze(root,protocol)
                else:
                    contract=seal(root,protocol)
                    if args.mode=="Freeze":result={"status":"FROZEN","contract":contract,"model_loaded":False}
                    else:
                        for ctx in protocol["contexts"].values():
                            if args.dataset not in ("All",ctx["dataset"]) or args.seed not in ("All",str(ctx["seed"])):continue
                            run_context(root,workspace,ctx,contract,config);check_inputs(protocol)
                        result=analyze(root,protocol)
    if result.get("status")=="COMPLETED":result={"status":"COMPLETED","parameter_rows":len(result["parameters"]),"behavior_rows":len(result["behavior"]),"report":str(root/"summary/REPORT.md"),"test_accessed":False}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
