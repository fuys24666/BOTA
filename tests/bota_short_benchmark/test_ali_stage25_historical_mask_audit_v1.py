from pathlib import Path
from types import SimpleNamespace
import copy
import math
import numpy as np
import pytest
import torch

from src.bota_short_benchmark import ali_stage25_historical_mask_audit_v1 as s
from src.bota_short_benchmark import ali_stage25_scoring as score
from src.bota_short_benchmark import ali_stage24_masked_scoring as m24


class Data:
    tokenizer=SimpleNamespace(pad_token_id=0)
    def __getitem__(self,i):
        return {"input_ids":torch.full((i+1,),2,dtype=torch.long),"target_ids":torch.tensor([2163 if i%2 else 465,1])}
    def collate_fn(self,rows):
        inputs=torch.zeros(len(rows),max(len(r["input_ids"]) for r in rows),dtype=torch.long)
        for i,r in enumerate(rows):inputs[i,:len(r["input_ids"])]=r["input_ids"]
        return {"input_ids":inputs,"target_ids":torch.stack([r["target_ids"] for r in rows])}


class Model(torch.nn.Module):
    def forward(self,input_ids,labels,attention_mask=None):
        assert attention_mask is not None
        values=(input_ids*attention_mask).sum(1).float()/attention_mask.sum(1)
        logits=torch.zeros(len(input_ids),labels.shape[1],2200)
        logits[:,:,2163]=values[:,None]
        return SimpleNamespace(logits=logits)


def test_raw_single_reproduces_stage24_score_formula():
    model=Model().eval();data=Data();ids=list(range(8))
    raw=score.raw_scores(model,data,ids,"cpu",0)
    result=score.combine([raw],[1.])
    assert result==m24.predict(model,data,ids,"cpu",4,0)


def test_ensemble_mixes_masses_not_normalized_scores_or_losses():
    a={"yes":[.09],"no":[.01],"targets":[[.25,.5]],"gold_label":[1],"sample_order_sha256":"h"}
    b={"yes":[.10],"no":[.40],"targets":[[.75,.25]],"gold_label":[1],"sample_order_sha256":"h"}
    r=score.combine([a,b],[.5,.5])
    assert r["probability"][0]==pytest.approx(.095/.3)
    assert r["probability"][0]!=pytest.approx((.9+.2)/2)
    assert r["answer_loss"][0]==pytest.approx(-math.log(.5*.375)/2)


@pytest.mark.parametrize("weights",[[.5],[1.,1.],[-1.,2.],[float('nan'),1.]])
def test_invalid_ensemble_weights(weights):
    with pytest.raises(ValueError,match="weights"):score.combine([{},{}],weights)


def test_mismatched_ensemble_labels_rejected():
    a={"yes":[.1],"no":[.1],"targets":[[.1]],"gold_label":[0],"sample_order_sha256":"h"}
    b={**a,"gold_label":[1]}
    with pytest.raises(ValueError,match="labels"):score.combine([a,b],[.5,.5])


def test_finaltest_gate_precedes_dataset_and_model_access(tmp_path,monkeypatch):
    monkeypatch.setattr(s,"T5Tokenizer",SimpleNamespace(from_pretrained=lambda *a,**k:pytest.fail("no model access")))
    with pytest.raises(PermissionError,match="ConfirmFinalTest"):
        s.dataset(tmp_path,{"scope":"FinalTest"},False)
    with pytest.raises(PermissionError):s.execute(tmp_path,tmp_path,{},"FinalTest","All",False)
    s.authorize("Development",False)


def test_task_resume_tamper_failure_and_access_flags(tmp_path):
    a=s.task(tmp_path,"one","h",lambda stage:{"status":"OK"},True)
    assert a["test_accessed"] and a["optimizer_updates"]==0
    assert s.task(tmp_path,"one","h",lambda stage:pytest.fail("resume"),True)==a
    with pytest.raises(ValueError,match="contract"):s.verify(tmp_path/"one","other")
    (tmp_path/"one/result.json").write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError,match="artifact"):s.verify(tmp_path/"one")
    def interrupt(stage):raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):s.task(tmp_path,"two","h",interrupt)
    assert not (tmp_path/"two").exists() and list((tmp_path/".work").glob("*/FAILED.json"))


def metric_fixture():
    cohort={"dataset":"Amazon","scope":"Development","lane":"binary","seed":43,
            "indices":[0,1,2,3],"users":[1,1,2,2],"selection":{"auc_damage_max":.005,"logloss_damage_max":.01}}
    jobs=[];results={}
    for method,prob in [("Original",[.2,.8,.1,.9]),("Exact-Masked",[.3,.7,.2,.8]),("BOTA",[.28,.72,.19,.81]),("RecEraser",None),("IFRU",None)]:
        status="READY" if prob else ("WEIGHTS_NOT_RETAINED" if method=="RecEraser" else "HISTORICAL_NUMERICAL_FAILURE")
        jobs.append({"id":method,"method":method,"request":"A1","seed":43,"cohort":"amz","old":None,
                     "provenance":{"users":[1],"historical_eligible":True},"status":status})
        results[method]={"status":"OK" if prob else status,"prediction":None if prob is None else
          {"probability":prob,"gold_label":[0,1,0,1],"answer_loss":[1]*4,"sample_order_sha256":s.canonical_hash(cohort["indices"])}}
    return cohort,jobs,results


def test_missing_methods_not_scored_and_seed_not_hardcoded():
    c,j,r=metric_fixture();rows=s.metric_rows(c,j,r)
    assert len(rows)==5 and all(x["seed"]==43 and x["test_accessed"] is False for x in rows)
    b=next(x for x in rows if x["method"]=="BOTA")
    assert b["E_U"]==pytest.approx(.02) and b["local_residual"]==pytest.approx(.2)
    for x in rows:
        if x["method"] in ("IFRU","RecEraser"):assert "E_U" not in x and x["status"]!="OK"


def test_old_predictions_require_exact_order(tmp_path):
    path=tmp_path/"old.json";s.atomic_json(path,{"prediction":{"sample_order_sha256":"wrong"}})
    with pytest.raises(ValueError,match="order"):
        s.read_old({"old":{"path":str(path),"keys":["prediction"]}},{"indices":[0]})


def test_partial_and_complete_summary_with_missing_weights(tmp_path):
    c,j,results=metric_fixture();plan={"cohorts":{"amz":c},"jobs":j};h=s.canonical_hash(plan)
    assert s.summarize(tmp_path,plan,"Development")["status"]=="INCOMPLETE"
    for job in j:s.task(tmp_path,"cells/"+job["id"],h,lambda stage,job=job:results[job["id"]])
    summary=s.summarize(tmp_path,plan,"Development")
    assert summary["status"]=="COMPLETED_WITH_UNAVAILABLE" and summary["unavailable_rows"]==2
    assert s.summarize(tmp_path,plan,"Development")==summary


def test_failed_invariance_cannot_be_reported_as_completed(tmp_path):
    c,j,results=metric_fixture();plan={"cohorts":{"amz":c},"jobs":j};h=s.canonical_hash(plan)
    results["BOTA"]={"status":"INVARIANCE_FAILED","prediction":None}
    for job in j:s.task(tmp_path,"cells/"+job["id"],h,lambda stage,job=job:results[job["id"]])
    with pytest.raises(RuntimeError,match="invariance"):s.summarize(tmp_path,plan,"Development")


def test_scorer_contains_no_training_calls():
    import ast
    tree=ast.parse(Path(score.__file__).read_text(encoding="utf-8"))
    forbidden={"step","backward","replay","build_bank","transport","execute_method"}
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden]


def test_ml_p5_and_bootstrap_metrics_keep_development_flags(monkeypatch):
    c,j,r=metric_fixture()
    c["dataset"]="ML1M"
    j=[x for x in j if x["method"] not in ("RecEraser","IFRU")]
    for job in j:job["request"]="L8"
    for method,prob in [("FullControl-P5",[.15,.85,.15,.85]),("Retain-Retrain-P5",[.25,.75,.20,.80])]:
        j.append({**j[0],"id":method,"method":method})
        r[method]={"status":"OK","prediction":{**r["Original"]["prediction"],"probability":prob}}
    ev=copy.deepcopy(s.inventory.s16.s10.EXPECTED_EVALUATION);ev["bootstrap_resamples"]=10
    monkeypatch.setattr(s.inventory.s16.s10,"EXPECTED_EVALUATION",ev)
    rows=s.metric_rows(c,j,r)
    b=next(x for x in rows if x["method"]=="BOTA")
    assert b["test_accessed"] is False and b["seed"]==43
    assert b["local_exact_masked_residual"] is not None and b["p5_retrain_residual"] is not None
    assert b["E_U"]==pytest.approx(.02)


def test_stage15_double_mass_normalization_is_preserved():
    raw=score.raw_scores(Model().eval(),Data(),[0,1],"cpu",0)
    pred=score.cohort_combine([raw],[1.],{"scope":"Development","lane":"ranking"})
    expected=np.asarray(raw["yes"],dtype=np.float64)/(np.asarray(raw["yes"])+np.asarray(raw["no"]))
    assert pred["probability"]==expected.tolist()


def test_tiny_real_t5_mask_batch_invariance():
    from transformers import T5Config,T5ForConditionalGeneration
    torch.manual_seed(7)
    model=T5ForConditionalGeneration(T5Config(vocab_size=2200,d_model=16,d_kv=4,d_ff=32,num_layers=1,
        num_decoder_layers=1,num_heads=4,dropout_rate=0.,pad_token_id=0,decoder_start_token_id=0,eos_token_id=1)).eval()
    ids=list(range(8));data=Data()
    a=score.combine([score.raw_scores(model,data,ids,"cpu",0,1)],[1.])
    b=score.combine([score.raw_scores(model,data,ids[::-1],"cpu",0,4)],[1.])
    config=s.load_config(Path(__file__).resolve().parents[2])["audit"]
    assert m24.difference(a,b,ids,ids[::-1],config)["passed"]
