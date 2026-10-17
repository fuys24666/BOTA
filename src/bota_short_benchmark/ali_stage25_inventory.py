"""Read-only historical inventory. No model, optimizer or new data selection."""
from pathlib import Path
import copy
from src.paper_if_a2.common import canonical_hash, directory_hash, sha256_file
from . import ali_stage24_masked_rescore_v1 as s24
from . import ali_stage15_ranking_v1 as s15
from . import ali_stage16_finaltest_v1 as s16

read = s24.read


def build(repo, workspace, config):
    outputs = workspace/"outputs"
    hashes = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"historical binding changed: {path}")
        hashes[str(path)] = digest
        return path
    def bundle(path, verifier):
        verifier(path)
        for file in path.rglob("*"):
            if file.is_file(): bind(file)
    for rel, digest in config["parents"].items(): bind(outputs/rel, digest)
    # Stage24 freezes the shared code and the Development source assets.
    p24 = s24.verify(outputs/s24.ROOT.split('/')[-1]/"protocol")["plan"]
    for path, digest in p24["hashes"].items(): bind(path, digest)
    c16 = read(outputs/"ali_stage16_finaltest_v1/protocol/contract.json")
    c15 = read(outputs/"ali_stage15_ranking_v1/protocol/contract.json")
    bundle(outputs/"ali_stage16_finaltest_v1/protocol", s16.verify)
    bundle(outputs/"ali_stage16_finaltest_v1/cohort", s16.verify)
    bundle(outputs/"ali_stage15_ranking_v1/protocol", s15.verify_bundle)
    cohorts, jobs = {}, []
    def add(cohort, seed, request, method, path, *, kind="fixed_ab", old=None, old_metrics=None,
            directory_sha=None, model_sha=None, status="READY", provenance=None):
        path = Path(path).resolve() if path is not None else None
        if status == "READY":
            if path is None or not path.is_dir():
                raise FileNotFoundError(f"missing retained endpoint: {path}")
            if directory_sha and directory_hash(path) != directory_sha:
                raise ValueError(f"endpoint directory changed: {path}")
            if kind == "fixed_ab": bind(path/"adapter_model.pt", model_sha)
            for file in path.rglob("*"):
                if file.is_file(): bind(file)
        if old:
            bind(old["path"])
        job = {"cohort": cohort, "seed": seed, "request": request, "method": method,
               "path": str(path) if path else None, "kind": kind, "old": old, "old_metrics": old_metrics,
               "status": status, "provenance": provenance or {}}
        job["id"] = canonical_hash(job)[:16]
        jobs.append(job)
    # ML Development indices are the frozen Stage14 calibration split, not a new split.
    ml14 = outputs/"ali_stage14_lambda_search_v1/cells"
    select_path = ml14/"coarse/s42_L8/predictions.json"
    bind(select_path)
    select = read(select_path)["indices"]
    _, dev_users, _ = s24.s23.c17.s2._source_users(workspace, c16["benchmarks"]["42"])
    rank_dev = read(outputs/"ali_stage15_ranking_v1/protocol/candidates.json")
    final_meta = read(outputs/"ali_stage16_finaltest_v1/cohort/cohort.json")
    rank_final = read(outputs/"ali_stage16_finaltest_v1/cohort/candidates.json")
    for seed in (41,42,43):
        bench = c16["benchmarks"][str(seed)]
        for split in ("Development", "FinalTest"):
            for lane in ("binary", "ranking"):
                key = f"ML1M_{split}_{lane}_s{seed}"
                plan = rank_dev if split=="Development" else rank_final
                meta = {"dataset": "ML1M", "scope": split, "lane": lane, "seed": seed,
                        "benchmark": bench, "role": "frozen_main_and_calibration_endpoints"}
                if lane == "ranking":
                    meta.update(plan=plan, indices=list(range(plan["candidate_count"])))
                elif split == "Development":
                    meta.update(data_path=str(workspace/bench["source"]["development_json"]), indices=select,
                                users=[dev_users[i] for i in select])
                else:
                    # Metadata only here. The raw FinalTest file is opened only after explicit admission.
                    meta.update(data_path=final_meta["final_path"], indices=list(range(final_meta["rows"])),
                                users=final_meta["users"], data_sha256=final_meta["processed_sha256"])
                cohorts[key] = meta
                for e in [e for e in c16["endpoints"] if e["seed"]==seed]:
                    name=f"s{seed}_{e['scenario']}_{e['method']}"
                    old=None
                    if split=="FinalTest":
                        cell=outputs/"ali_stage16_finaltest_v1/cells"/lane/name
                        bundle(cell,s16.verify)
                        old={"path":str(cell/"predictions.json"),"keys":[]}
                    elif lane=="ranking" and e["method"]!="BOTA0":
                        cell=outputs/"ali_stage15_ranking_v1/cells"/name
                        bundle(cell,s15.verify_bundle)
                        old={"path":str(cell/"scores.json"),"keys":["scores"],"scores_only":True}
                    elif lane=="binary" and e["method"] in ("Original","Exact-Masked","BOTA0","BOTA1","BOTA64"):
                        cell=ml14/("coarse" if seed==42 else "confirm")/f"s{seed}_{e['scenario']}"
                        bundle(cell,s24.s23.s18.s14.verify_bundle)
                        if read(cell/"predictions.json")["indices"] != select: raise ValueError("ML Dev split changed")
                        label={"BOTA0":"bota_r0","BOTA1":"bota_r1","BOTA64":"bota_r64"}.get(e["method"],e["method"])
                        old={"path":str(cell/"predictions.json"),"keys":["predictions",label]}
                    add(key,seed,e["scenario"],e["method"],workspace/e["path"],kind=e["kind"],old=old,
                        directory_sha=e["directory_sha256"],provenance={"users":e["users"],"historical_eligible":True})
    # Amazon Stage20/21 comparison and frozen Stage20 calibration choices.
    for stage, seed in ((20,42),(21,43)):
        base=outputs/("ali_stage20_amazon_baselines_v1" if stage==20 else "ali_stage21_amazon_seed43_v1")
        verifier=s24.s23.s20.verify if stage==20 else s24.s23.k21.verify
        bundle(base/"protocol",verifier);bundle(base/"summary",verifier)
        p=read(base/"protocol/result.json")
        indices=p["development_partition"]["select"]
        units=[]
        for file in sorted((base/"units").glob("*/result.json")):
            unit=read(file)
            units.append((file.parent,unit))
        phases=[("comparison",p["comparison"]["requests"],read(base/"summary/result.json")["rows"])]
        if stage==20:
            bundle(base/"all_selections",verifier)
            decisions=read(base/"all_selections/result.json")["decisions"]
            calrows=[]
            for method,d in decisions.items():
                frozen=d["selected"] or d.get("diagnostic")
                if frozen is None: continue
                for path,unit in units:
                    if unit["identity"]["phase"]=="calibration" and unit["identity"]["method"]==method:
                        calrows.extend({**r,"calibration_status":d["status"],"diagnostic_only":d["selected"] is None}
                                       for r in unit["rows"] if r["candidate_id"]==frozen["candidate_id"])
            phases.append(("calibration",p["calibration"],calrows))
        for phase,requests,rows in phases:
            key=f"Amazon_{phase}_s{seed}"
            cohorts[key]={"dataset":"Amazon","scope":"Development","lane":"binary","seed":seed,
                          "benchmark":p["benchmark"],"indices":indices,"users":[p["development_users"][i] for i in indices],
                          "data_path":str(workspace/p["benchmark"]["source"]["development_json"]),
                          "requests":requests,"role":phase,"selection":p["config"]["selection"]}
            if phase=="calibration":
                rows += [{"method":m,"request":r["id"],"status":"OK","candidate_id":m}
                         for r in requests for m in ("Original","Exact-Masked")]
            for row in rows:
                method,rid=row["method"],row["request"]
                req=next(r for r in requests if r["id"]==rid)
                provenance={"users":req["users"],"historical_eligible":not row.get("diagnostic_only",False),
                            "historical_status":row["status"],"calibration_status":row.get("calibration_status","REFERENCE")}
                if row["status"]!="OK":
                    add(key,seed,rid,method,None,status="HISTORICAL_NUMERICAL_FAILURE",old_metrics=row,provenance=provenance)
                    continue
                if method in ("Original","Exact-Masked","BOTA","BOTA-source-only"):
                    oldkey="Original" if method=="Original" else (f"Exact_{rid}" if method=="Exact-Masked" else f"{method}_{rid}")
                    saved=base/"predictions"/oldkey
                    verifier(saved)
                    record=read(saved/"result.json")
                    path=Path(record["identity"]["path"])/"adapter"
                    if not path.is_dir(): path=Path(record["identity"]["path"])
                    add(key,seed,rid,method,path,model_sha=record["identity"]["model_sha256"],
                        old={"path":str(saved/"result.json"),"keys":["prediction"]},old_metrics=row,provenance=provenance)
                    continue
                found=[(path,unit) for path,unit in units if unit["identity"]["phase"]==phase and
                       unit["identity"]["method"]==method and unit["identity"]["request"]==rid and
                       any(r["candidate_id"]==row["candidate_id"] for r in unit["rows"])]
                if len(found)!=1:raise ValueError(f"ambiguous frozen Amazon endpoint: {key}/{rid}/{method}")
                path,unit=found[0];bundle(path,verifier)
                if row.get("model_tensors_retained") is False:
                    add(key,seed,rid,method,None,status="WEIGHTS_NOT_RETAINED",old_metrics=row,provenance=provenance)
                else:
                    add(key,seed,rid,method,path/row["candidate_id"]/"adapter",directory_sha=row["model_sha256"],
                        old={"path":str(path/(row["candidate_id"]+"_prediction.json")),"keys":["prediction"]},
                        old_metrics=row,provenance=provenance)
    # Amazon BOTA calibration: same full Development K2/K4 and all confirmation seeds.
    base=outputs/"ali_stage18_amazon_lambda_search_v2"
    p18=read(base/"protocol/result.json")
    p20=read(outputs/"ali_stage20_amazon_baselines_v1/protocol/result.json")
    for seed in (41,42,43):
        key=f"Amazon_lambda_calibration_s{seed}"
        cohorts[key]={"dataset":"Amazon","scope":"Development","lane":"binary","seed":seed,
                      "benchmark":p20["benchmark"],"indices":list(range(len(p20["development_users"]))),
                      "users":p20["development_users"],"data_path":str(workspace/p20["benchmark"]["source"]["development_json"]),
                      "requests":p20["calibration"],"role":"frozen_lambda_calibration","selection":p18["config"]["selection"]}
        for req in p20["calibration"]:
            rid=req["id"]
            for method,label in (("Original","Original"),("Exact-Masked",f"{rid}_Exact"),
                                 ("BOTA",f"{rid}_bota_r48"),("BOTA-source-only",f"{rid}_bota_r0")):
                pred=base/"predictions"/f"s{seed}_{label}"
                bundle(pred,s24.s23.s18.verify)
                expected=read(pred/"result.json")["identity"]["adapter_sha256"]
                if method=="Original":
                    path=base/"banks"/("coarse_42" if seed==42 else f"confirm_{seed}")/"Original/adapter"
                elif method=="Exact-Masked":
                    path=(Path(p18["checks"]["references"]["Retrain-Short"])/"scenarios"/rid/"adapter") if seed==42 else base/"exact"/f"s{seed}_{rid}"/"adapter"
                else:
                    phase=("fine_42" if method=="BOTA" else "coarse_42") if seed==42 else f"confirm_{seed}"
                    path=base/"banks"/phase/rid/("bota_r48" if method=="BOTA" else "bota_r0")/"adapter"
                add(key,seed,rid,method,path,model_sha=expected,
                    old={"path":str(pred/"result.json"),"keys":["prediction"]},provenance={"users":req["users"],"historical_eligible":True})
    for key,cohort in cohorts.items():
        if cohort["scope"]=="Development" and cohort["lane"]=="binary":bind(cohort["data_path"])
    # Bind all runtime dependencies from the historical main experiment.
    for dep in c16["dependencies"].values():
        path=workspace/dep["path"]
        actual=directory_hash(path) if path.is_dir() else sha256_file(path)
        if actual!=dep["sha256"]:raise ValueError(f"shared dependency changed: {path}")
        for file in (path.rglob("*") if path.is_dir() else [path]):
            if file.is_file():bind(file)
    if len({j["id"] for j in jobs})!=len(jobs):raise ValueError("duplicate job identity")
    return {"cohorts":cohorts,"jobs":jobs,"hashes":hashes}
