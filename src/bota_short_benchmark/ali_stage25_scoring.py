"""Mask-aware fixed-adapter/ensemble inference with unchanged mixture semantics."""
import gc
from pathlib import Path
import numpy as np
import torch
from src.paper_if_a2.common import canonical_hash
from . import evaluation
from . import ali_stage24_masked_scoring as s24


def raw_scores(model, data, indices, device, pad, batch_size=4, progress=None):
    if model.training: raise ValueError("model.eval() required")
    out={"yes":[],"no":[],"targets":[],"gold_label":[],"probability":[],"answer_loss":[],
         "sample_order_sha256":canonical_hash(list(indices))}
    with torch.inference_mode():
        for start in range(0,len(indices),batch_size):
            batch=data.collate_fn([data[i] for i in indices[start:start+batch_size]])
            inputs=batch["input_ids"].to(device);targets=batch["target_ids"].to(device)
            mask=inputs.ne(pad);valid=targets.ne(-100)
            if not bool(mask.any(1).all() and valid.any(1).all()):raise ValueError("empty sample")
            if not bool(((targets[:,0]==465)|(targets[:,0]==2163)).all()):raise ValueError("yes/no labels changed")
            logits=model(input_ids=inputs,attention_mask=mask,labels=targets).logits
            if not bool(torch.isfinite(logits).all()):raise ValueError("nonfinite model logits")
            probs=torch.softmax(logits,-1);yn=probs[:,0,[465,2163]]
            gathered=torch.gather(probs,2,targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            prob=yn[:,1]/yn.sum(-1)
            loss=(-(gathered.clamp_min(1e-30).log())*valid).sum(-1)/valid.sum(-1)
            if not bool(torch.isfinite(prob).all() and torch.isfinite(loss).all()):raise ValueError("nonfinite scores")
            out["yes"].extend(yn[:,1].cpu().tolist());out["no"].extend(yn[:,0].cpu().tolist())
            out["targets"].extend([row[v].cpu().tolist() for row,v in zip(gathered,valid)])
            out["gold_label"].extend((targets[:,0]==2163).int().cpu().tolist())
            out["probability"].extend(prob.cpu().tolist());out["answer_loss"].extend(loss.cpu().tolist())
            if progress and (start==0 or start%2000==0 or start+batch_size>=len(indices)):
                progress(min(start+batch_size,len(indices)),len(indices))
    return out


def combine(parts, weights):
    if not parts or len(parts)!=len(weights) or any(not np.isfinite(w) or w<0 for w in weights) or not np.isclose(sum(weights),1.,atol=1e-5,rtol=0):
        raise ValueError("invalid mixture weights")
    ref=parts[0]
    for p in parts:
        if p["gold_label"]!=ref["gold_label"] or p["sample_order_sha256"]!=ref["sample_order_sha256"]:
            raise ValueError("ensemble sample order/labels changed")
        if [len(t) for t in p["targets"]]!=[len(t) for t in ref["targets"]]:raise ValueError("ensemble target lengths changed")
    if len(parts)==1 and weights==[1.]:
        return {k:ref[k] for k in ("probability","answer_loss","gold_label","sample_order_sha256")}
    yes=sum(w*np.asarray(p["yes"],dtype=np.float64) for p,w in zip(parts,weights))
    no=sum(w*np.asarray(p["no"],dtype=np.float64) for p,w in zip(parts,weights))
    probability=yes/np.maximum(yes+no,1e-30)
    # Mix token probabilities first, then take log; do NOT average shard losses
    # or normalized Yes probabilities (both would change the baseline).
    losses=[]
    for i in range(len(yes)):
        mass=sum(w*np.asarray(p["targets"][i],dtype=np.float64) for p,w in zip(parts,weights))
        losses.append(float(-np.log(np.maximum(mass,1e-30)).mean()))
    if not np.isfinite(probability).all() or not np.isfinite(losses).all():raise ValueError("nonfinite mixture")
    return {"probability":probability.tolist(),"answer_loss":losses,"gold_label":ref["gold_label"],
            "sample_order_sha256":ref["sample_order_sha256"]}


def cohort_combine(parts, weights, cohort):
    pred=combine(parts,weights)
    if cohort["scope"]=="Development" and cohort["lane"]=="ranking" and len(parts)==1:
        # Stage15 normalized the first-token masses in float64 even for a
        # single adapter. Preserve that convention (ties can matter for NDCG).
        yes=np.asarray(parts[0]["yes"],dtype=np.float64)
        no=np.asarray(parts[0]["no"],dtype=np.float64)
        pred["probability"]=(yes/(yes+no)).tolist()
    return pred


def load_model(workspace, benchmark, job, shard, device):
    path=Path(job["path"])
    if shard is None:
        state=torch.load(path/"adapter_model.pt",map_location="cpu",weights_only=True)
        model=evaluation.load_legacy_model(workspace/benchmark["source"]["original_checkpoint"])
        names,params=evaluation.install_fixed_ab_coordinate(model,state["A"],32)
        with torch.no_grad():
            for name,param in zip(names,params):param.copy_(state["B"][name].to(param))
    else:
        model=evaluation.clean_lora_model(workspace/benchmark["source"]["original_checkpoint"],
              {"r":16,"lora_alpha":32,"lora_dropout":.05,"target_modules":["q","v"]},torch.device("cpu"))
        state=torch.load(path/f"models/shard_{shard:02d}/final_state.pt",map_location="cpu",weights_only=True)
        evaluation.load_trainable_state(model,state,"lora_adapter")
    return model.to(device).eval()


def score_endpoint(workspace, cohort, job, data, config, progress, read):
    device=torch.device("cuda:0")
    if job["kind"]=="fixed_ab":components=[(None,1.)]
    elif job["kind"]=="receraser_ensemble":
        weights=read(Path(job["path"])/"aggregation.json")["weights"]
        if not weights or any(not np.isfinite(w) or w<0 for w in weights) or not np.isclose(sum(weights),1.,atol=1e-5,rtol=0):raise ValueError("invalid frozen ensemble")
        components=list(enumerate(weights))
    else:raise ValueError("unsupported frozen endpoint kind")
    local=[i for i,u in zip(cohort["indices"],cohort.get("users",[])) if u in job["provenance"]["users"]]
    if cohort["lane"]=="ranking":
        local=[];offset=0
        for group in cohort["plan"]["users"]:
            if group["user"] in job["provenance"]["users"]:local.extend(range(offset,offset+len(group["items"])))
            offset+=len(group["items"])
    if not local:local=cohort["indices"][:32]
    ctx={"indices":cohort["indices"],"pairs":[{"local_indices":local}]}
    parts=[];probes=[];audits=[]
    for shard,weight in components:
        model=load_model(workspace,cohort["benchmark"],job,shard,device)
        try:
            audit=s24.invariance_audit(model,data,ctx,device,data.tokenizer.pad_token_id,config["audit"])
            audits.append({"shard":shard,"weight":weight,**audit})
            if audit["status"]!="PASS":
                return {"status":"INVARIANCE_FAILED","audits":audits,"prediction":None}
            probes.append(raw_scores(model,data,audit["sample_ids"],device,data.tokenizer.pad_token_id,1))
            parts.append(raw_scores(model,data,cohort["indices"],device,data.tokenizer.pad_token_id,4,
                                    lambda done,total:progress(shard,done,total)))
        finally:
            del model;gc.collect();torch.cuda.empty_cache()
    pred=cohort_combine(parts,[w for _,w in components],cohort)
    probe=cohort_combine(probes,[w for _,w in components],cohort)
    ids=audits[0]["sample_ids"];lookup={i:j for j,i in enumerate(cohort["indices"])}
    sub={k:[pred[k][lookup[i]] for i in ids] for k in ("probability","answer_loss","gold_label")}
    sub["sample_order_sha256"]=canonical_hash(ids)
    gate=s24.difference(probe,sub,ids,ids,config["audit"])
    return {"status":"OK" if gate["passed"] else "INVARIANCE_FAILED","audits":audits,
            "full_cohort_gate":gate,"prediction":pred if gate["passed"] else None}
