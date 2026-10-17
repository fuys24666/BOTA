import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.bota_short_benchmark import ali_stage23_decomposition as m
from src.bota_short_benchmark import ali_stage23_pair_decomposition_v1 as s


def test_linear_additive_exact_case():
    o=torch.zeros(3);u=torch.tensor([1.,0.,0.]);v=torch.tensor([0.,1.,0.]);p=u+v
    r=m.decompose(o,u,v,p,u,v,p)
    assert r["interaction_norm"]==r["pair_error_norm"]==r["identity_closure_norm"]==0


def test_isolated_interaction_with_exact_individual_predictions():
    o=torch.zeros(3);u=torch.tensor([1.,0.,0.]);v=torch.tensor([0.,1.,0.]);i=torch.tensor([0.,0.,.5])
    r=m.decompose(o,u,v,u+v+i,u,v,u+v)
    assert r["single_u_error_norm"]==r["single_v_error_norm"]==0
    assert r["pair_error_norm"]==r["interaction_norm"]==.5


def test_isolated_single_approximation_error():
    o=torch.zeros(2);u=torch.tensor([1.,0.]);v=torch.tensor([0.,1.]);a=torch.tensor([1.2,0.])
    r=m.decompose(o,u,v,u+v,a,v,a+v)
    assert r["interaction_norm"]==0 and r["pair_error_norm"]>0


def test_cancellation_not_misreported_as_independent_percentages():
    o=torch.zeros(2);u=torch.tensor([1.,0.]);v=torch.tensor([0.,1.]);interaction=torch.tensor([.5,.5])
    a=u+interaction;pair=u+v+interaction
    r=m.decompose(o,u,v,pair,a,v,a+v)
    assert r["pair_error_norm"]==0 and r["interaction_norm"]>0 and r["summed_single_error_norm"]>0
    assert r["single_error_interaction_cosine"]==pytest.approx(1.)
    assert r["signed_cross_terms"]["single_u_error__negative_interaction"]<0
    assert r["squared_identity_gap"]<1e-12


def test_rounding_is_explicit_and_identity_closes():
    rng=torch.Generator().manual_seed(3)
    o=torch.randn(50,generator=rng)
    a=o+torch.randn(50,generator=rng)*1e-5;b=o+torch.randn(50,generator=rng)*1e-5
    p=o+(a-o)+(b-o)
    r=m.decompose(o,a,b,p,a,b,p)
    assert r["identity_closure_norm"]<1e-12 and r["squared_identity_gap"]<1e-12


def test_zero_pair_signal_retained_without_huge_fake_ratio():
    o=torch.zeros(2);u=torch.ones(2);v=-u
    r=m.decompose(o,u,v,o,u,v,o)
    assert r["pair_signal_below_floor"] and r["pair_relative_error"] is None
    assert r["interaction_over_pair_norm"] is None


def test_nonfinite_references_rejected():
    o=torch.zeros(2);bad=torch.tensor([float("nan"),0.])
    with pytest.raises(ValueError,match="nonfinite"):m.decompose(o,o,o,bad,o,o,o)


def test_ml_metadata_resolved_by_hash_not_positional_zip():
    users=list(range(8));tr=[u for u in users for _ in range(u+1)];dev=users*5
    members=[{"role":"low" if u<4 else "middle", "user_hash":m.digest(f"bota-short:42:{'low' if u<4 else 'middle'}:{u}"),
              "global_train_frequency":u+1,"development_samples":5} for u in users]
    reg={"seed":42,"scenarios":[{"id":"L4M4","users":users[::-1],"public_members":members[3:]+members[:3]}]}
    pairs=m.ml_pairs(reg,tr,dev)
    assert len(pairs)==3
    assert all(p["users"][0]<4<=p["users"][1] for p in pairs)
    assert pairs==m.ml_pairs({"seed":42,"scenarios":[{"id":"L4M4","users":users,"public_members":members}]},tr,dev)


def test_cohort_is_metadata_only_and_reports_all_samples():
    pairs=[{"id":f"A{i}","users":[2*i,2*i+1],"roles":["low","middle"]} for i in range(3)]
    tr=list(range(6))*2;du=list(range(6))*7+[99]*10
    ev={"selection_salt":"fixed","matched_samples_per_user":5,"nonrequested_control_samples":8}
    rows,indices,control=m.enrich_pairs(pairs,tr,du,list(range(12)),ev)
    assert len(indices)==50 and len(control)==8
    assert all(r["deleted_interactions"]==4 and len(r["matched_indices"])==10 and len(r["local_indices"])==14 for r in rows)
    assert not set(control)&{i for r in rows for i in r["local_indices"]}


def test_insufficient_support_does_not_replace_users():
    pairs=[{"id":str(i),"users":[2*i,2*i+1],"roles":["low","middle"]} for i in range(3)]
    with pytest.raises(ValueError,match="support"):
        m.enrich_pairs(pairs,list(range(6)),list(range(6)),list(range(6)),{"selection_salt":"x","matched_samples_per_user":5,"nonrequested_control_samples":8})


def test_oracle_composes_parameter_changes_not_probabilities():
    original={"B":{"x":torch.tensor([10.])}}
    bank={"users":[1,2],"vectors":{"0":torch.tensor([[.5],[1.]])}}
    out=s.make_endpoints(original,["x"],bank,{1:{"x":torch.tensor([11.])},2:{"x":torch.tensor([12.])}},
                         {"A1":{"x":torch.tensor([14.])}},[{"id":"A1","users":[1,2]}])
    assert out["OracleSum_A1"]["x"].item()==13.
    assert out["B0_A1"]["x"].item()==11.5


def test_probability_metrics_preserve_absolute_scale_and_small_signal():
    r=m.probability_metrics([.1,.9],[.2,.8],[.15,.85],[0,1])
    assert r["E_U"]==pytest.approx(.05) and r["residual"]==pytest.approx(1.)
    assert m.probability_metrics([.1],[.1],[.1],[0])["residual"] is None


def test_bundle_resume_and_tamper(tmp_path):
    result=s.task(tmp_path,"unit","hash",lambda stage:{"status":"NUMERICAL_TRANSPORT_FAILURE"})
    assert s.task(tmp_path,"unit","hash",lambda stage:pytest.fail("must reuse"))==result
    with pytest.raises(ValueError,match="contract"):s.verify(tmp_path/"unit","other")
    (tmp_path/"unit/result.json").write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError,match="artifact"):s.verify(tmp_path/"unit")


def test_interrupted_unit_retained_not_published(tmp_path):
    def fail(stage):raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):s.task(tmp_path,"unit","hash",fail)
    assert not (tmp_path/"unit").exists()
    assert len(list((tmp_path/".work").glob("*/FAILED.json")))==1


def test_replay_control_uses_fresh_start_and_mask_budget(tmp_path,monkeypatch):
    parameter=torch.nn.Parameter(torch.zeros(2)); original={"B":{"B":torch.zeros(2)},"A":{"A":torch.ones(2)}}
    model=SimpleNamespace()
    fresh=lambda *a:(model,["B"],[parameter],original["A"],SimpleNamespace(pad_token_id=0),None,[1,2]*1600,"cpu",original)
    monkeypatch.setattr(s,"fresh",fresh);monkeypatch.setattr(s.s18,"release",lambda:None)
    def train(model,data,order,users,targets,names,params,device,pad,engine,budget,arm):
        budget.calls=200
        trace=[{"masked_slots":sum(users[i] in targets for i in order)}]
        return {"B":torch.zeros(2)},None,trace
    monkeypatch.setattr(s.runner,"_train_named",train)
    ctx={"registry":{"order":list(range(3200))},"benchmark":{"optimizer":{}}}
    out=s.replay(tmp_path,tmp_path,"control",ctx,[],"h",original["B"])
    assert out.exists() and s.verify(tmp_path/"control")["original_reproduction"]
    with pytest.raises(RuntimeError,match="Original"):
        s.replay(tmp_path,tmp_path,"bad",ctx,[],"h",{"B":torch.ones(2)})


def test_failures_retained_in_parameter_analysis(tmp_path):
    ctx={"pairs":[{"id":"A1","users":[1,2]}],"lambda":48,"dataset":"Amazon","seed":41}
    original={"B":{"x":torch.zeros(2)}};single={1:{"x":torch.ones(2)},2:{"x":-torch.ones(2)}};pair={"A1":{"x":torch.ones(2)}}
    bank={"users":[1,2],"vectors":{"0":torch.zeros(2,2)},"failures":{"48":{"status":"nonfinite"}}}
    endpoints=s.make_endpoints(original,["x"],bank,single,pair,ctx["pairs"])
    r=s.parameter_analysis(tmp_path,"Amazon_s41","h",ctx,endpoints,bank,["x"],{"ratio_norm_floor":1e-12})
    assert len(r["rows"])==2 and r["rows"][1]["status"]=="NUMERICAL_TRANSPORT_FAILURE"


def test_smoke():
    assert s.smoke()["model_loaded"] is False


def test_context_end_to_end_cpu_fakes_and_resume(tmp_path,monkeypatch):
    """Exercise replay -> bank -> oracle -> predictions -> all metrics, no T5."""
    parameter=torch.nn.Parameter(torch.zeros(2))
    original={"A":{"A":torch.ones(1)},"B":{"B":torch.zeros(2)}}
    path=tmp_path/"original.pt";torch.save(original,path);before=s.sha256_file(path)
    pairs=[{"id":f"A{i+1}","users":[2*i,2*i+1],"local_indices":[2*i,2*i+1],"matched_indices":[2*i,2*i+1],
            "members":[{"user":u,"indices":[u],"matched_indices":[u]} for u in (2*i,2*i+1)]} for i in range(3)]
    ctx={"dataset":"Amazon","seed":41,"lambda":48,"original":str(path),"pairs":pairs,
         "registry":{"order":list(range(12))},"historical_joint":{},"historical_bota":{},
         "indices":list(range(8)),"control_indices":[6,7],"benchmark":{"optimizer":{},"source":{"development_json":"unused"}}}
    model=SimpleNamespace(eval=lambda:None)
    def fresh(*args):
        return model,["B"],[parameter],original["A"],SimpleNamespace(pad_token_id=0),None,list(range(6))*2,"cpu",original
    monkeypatch.setattr(s,"fresh",fresh);monkeypatch.setattr(s.s18,"release",lambda:None)
    def train(model,data,order,ids,users,names,params,device,pad,engine,budget,arm):
        budget.calls=200
        delta=sum(u+1 for u in users)*.01+len(users)**2*.001
        return {"B":torch.full((2,),delta)},None,[{"masked_slots":sum(ids[i] in users for i in order)}]
    monkeypatch.setattr(s.runner,"_train_named",train)
    def transport(model,data,order,ids,targets,params,names,device,pad,engine,scales,progress):
        vectors={s.s18.cid(scale):torch.tensor([[.01*(u+1)]*2 for u in targets])*(1 if scale==0 else 1.05) for scale in [0,1,48]}
        return original["B"],vectors,{},200,{"status":"TENSOR_EXACT_PASS"}
    monkeypatch.setattr(s.s18,"transport",transport)
    monkeypatch.setattr(s.s18.s8,"JsonPromptDataset",lambda *a:None)
    def predict(model,dev,indices,device,batch):
        probability=float(torch.sigmoid(parameter[0]).detach())
        return {"probability":[probability]*len(indices),"gold_label":[i%2 for i in indices],"answer_loss":[.5]*len(indices)}
    monkeypatch.setattr(s.evaluation,"_single_predictions",predict)
    config={"ratio_norm_floor":1e-12,"evaluation":{"inference_batch_size":4}}
    s.run_context(tmp_path,tmp_path,ctx,"hash",config)
    r=s.verify(tmp_path/"Amazon_s41/behavior","hash")
    assert len(r["rows"])==30 and all(x["status"]=="OK" for x in r["rows"])
    assert len(s.verify(tmp_path/"Amazon_s41/parameters","hash")["rows"])==6
    assert s.sha256_file(path)==before
    monkeypatch.setattr(s,"fresh",lambda *a:pytest.fail("completed run must not load a model"))
    s.run_context(tmp_path,tmp_path,ctx,"hash",config)
