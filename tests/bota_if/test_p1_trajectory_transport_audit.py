from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from src.bota_if import p1_trajectory_transport_audit as audit

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"configs/bota_if/p1_trajectory_transport_audit_v1.yaml"


@pytest.fixture
def config(): return audit.load_frozen_config(CONFIG)


def test_frozen_protocol_and_budget(config):
    assert config["coordinate"]["total_transport_rank"]==576
    assert config["schedule"]["steps"]==50 and config["schedule"]["batch_size"]==16
    assert config["transport"]["primary"]=="T2_AdamW_full_state"
    result=audit.budget_audit(config)
    assert result["canonical_optimizer_steps"]==50
    assert result["masked_optimizer_steps"]==150
    assert result["total_physical_optimizer_step_calls"]==200
    assert result["zero_authoritative_update"] is True


@pytest.mark.parametrize("section,key,value",[
    ("coordinate","transport_rank_per_module",16),
    ("coordinate","fixed_a_seed",43),
    ("schedule","steps",49),
    ("schedule","calibration_batches",8),
    ("optimizer","learning_rate",.002),
    ("transport","primary","T0_SGD"),
    ("runtime","physical_optimizer_step_limit",201),
    ("scientific_scope","compacted_retrain_reference",True),
])
def test_config_tamper_rejected(tmp_path,config,section,key,value):
    changed=copy.deepcopy(config);changed[section][key]=value;path=tmp_path/"config.yaml";path.write_text(yaml.safe_dump(changed),encoding="utf-8")
    with pytest.raises(ValueError): audit.load_frozen_config(path)


def test_schedule_is_deterministic_disjoint_and_outcome_free():
    users=[1]*30+[2]*25+[3]*20+[4]*15+[5]*10+[6]*8+[7]*6+[8]*5
    first=audit.freeze_schedule(users,seed=42,steps=5,batch_size=8,calibration_batches=4)
    second=audit.freeze_schedule(users,seed=42,steps=5,batch_size=8,calibration_batches=4)
    assert first==second and not set(first["train_indices"])&set(first["calibration_indices"])
    assert first["public"]["selection_uses_outcomes"] is False
    assert first["public"]["raw_user_ids_persisted"] is False
    assert len(first["public"]["selected_user_hashes"])==3


def test_fixed_a_is_orthonormal_and_request_independent():
    modules=[("layer.q",torch.nn.Linear(7,5,bias=False)),("layer.v",torch.nn.Linear(7,5,bias=False))]
    first,report=audit.deterministic_fixed_a(modules,rank=3,seed=42);second,_=audit.deterministic_fixed_a(modules,rank=3,seed=42)
    assert report["request_data_used"] is False
    for name in first:
        assert torch.equal(first[name],second[name])
        assert torch.allclose(first[name]@first[name].T,torch.eye(3,dtype=torch.float64),atol=1e-12,rtol=0)


def test_adamw_full_state_matches_finite_difference():
    result=audit.synthetic()
    assert result["adamw_full_state_gate"] is True
    assert result["adamw_full_state_finite_difference_relative_error"]<1e-5
    assert result["optimizer_steps"]==0 and result["real_model_loaded"] is False


def test_frozen_v_and_full_state_are_distinct_when_dv_matters():
    args=[torch.tensor([.2],dtype=torch.float64),torch.tensor([.3],dtype=torch.float64),torch.tensor([.1],dtype=torch.float64),torch.tensor([.04],dtype=torch.float64),torch.tensor([.01],dtype=torch.float64),torch.tensor([.02],dtype=torch.float64),torch.tensor([.03],dtype=torch.float64),torch.tensor([.05],dtype=torch.float64)]
    kwargs={"step":3,"lr":.001,"beta1":.9,"beta2":.999,"eps":1e-8,"weight_decay":.01}
    frozen=audit.adamw_tangent_step(*args,**kwargs,full_v=False)[0]
    full=audit.adamw_tangent_step(*args,**kwargs,full_v=True)[0]
    assert not torch.equal(frozen,full)


def test_step_budget_is_physical_and_cannot_exceed_limit():
    parameter=torch.nn.Parameter(torch.tensor(1.));optimizer=torch.optim.SGD([parameter],lr=.1);budget=audit.StepBudget(2)
    budget.step(optimizer,"a");budget.step(optimizer,"b")
    with pytest.raises(RuntimeError): budget.step(optimizer,"c")
    assert budget.calls==2 and budget.by_arm=={"a":1,"b":1}


def test_prediction_gate_boundaries():
    canonical={"m":torch.zeros(2,dtype=torch.float64)};actual={"m":torch.tensor([1.,0.],dtype=torch.float64)};basis=[torch.eye(2,dtype=torch.float64)];state=[{"theta":torch.tensor([1.,0.]),"m":torch.zeros(2),"v":torch.zeros(2)}]
    gates={"cosine_minimum":.8,"norm_ratio_minimum":.5,"norm_ratio_maximum":2.,"relative_l2_maximum":.75,"positive_module_fraction_minimum":.9}
    row=audit.compare_prediction(actual,canonical,state,["m"],basis,gates)
    assert row["base_gates_passed"] is True and row["cosine"]==1 and row["relative_l2_error"]==0
    assert row["high_energy_positive_module_fraction"]==1
    state[0]["theta"]=torch.tensor([-1.,0.]);row=audit.compare_prediction(actual,canonical,state,["m"],basis,gates)
    assert row["base_gates_passed"] is False and row["positive_module_fraction"]==0


def minimal_report():
    return {"schema":audit.SCHEMA,"status":"COMPLETED","classification":"trajectory_transport_supported","all_gates_passed":True,"users":[],"execution":{"physical_optimizer_step_calls":200},"primary_summary":{"users_passing":3},"test_accessed":False}


def test_atomic_publication_contains_no_model_or_optimizer(tmp_path,config):
    config_path=tmp_path/"config.yaml";config_path.write_text(yaml.safe_dump(config),encoding="utf-8");stage=tmp_path/"stage";stage.mkdir();destination=tmp_path/config["output_root"]/"run";audit.publish(stage,destination,minimal_report(),"impl")
    assert {p.name for p in destination.iterdir()}=={"COMPLETED","manifest.json","p0_p1a_trajectory_transport.json","run_state.json"}
    result=audit.analyze(tmp_path,config_path,"run");assert result["all_gates_passed"] is True and result["test_accessed"] is False
    manifest=json.loads((destination/"manifest.json").read_text());assert manifest["model_artifact_published"] is False and manifest["optimizer_artifact_published"] is False


def test_publication_tamper_rejected(tmp_path,config):
    config_path=tmp_path/"config.yaml";config_path.write_text(yaml.safe_dump(config),encoding="utf-8");stage=tmp_path/"stage";stage.mkdir();destination=tmp_path/config["output_root"]/"run";audit.publish(stage,destination,minimal_report(),"impl");(destination/"p0_p1a_trajectory_transport.json").write_text("{}\n",encoding="utf-8")
    with pytest.raises(ValueError):audit.analyze(tmp_path,config_path,"run")


def test_train_replay_never_materializes_development_or_finaltest():
    source=inspect.getsource(audit._train_user_ids_only)
    assert "train_users.append" in source
    assert "development.append" not in source and "validation.append" not in source
    assert '"development_rows_materialized": 0' in source
    assert '"final_test_rows_materialized": 0' in source


def test_full_is_masked_reference_zero_authoritative_update():
    source=inspect.getsource(audit.execute)
    assert "run_canonical" in source and "run_masked" in source
    assert 'budget.calls!=200' in source
    assert '"authoritative_parameters_modified":False' in source
    assert '"authoritative_optimizer_steps_committed":0' in source
    assert '"model_artifact_published":False' in source
    assert '"development_loaded":False' in source and '"retrain_loaded":False' in source
    assert "development.json" not in source and "validation_partition" not in source
    assert ".train()" in inspect.getsource(audit._fresh_runtime)


def test_masked_arm_preserves_slots_and_denominator():
    source=inspect.getsource(audit.run_masked)
    assert "torch.sum(losses*weights)/len(chosen)" in source
    assert '"masked_slots"' in source and '"batch_size_preserved"' in source


def test_interrupted_run_publishes_actual_step_count_before_reraise():
    source=inspect.getsource(audit.execute)
    assert '"status":"INTERRUPTED"' in source
    assert '"physical_optimizer_step_calls":budget.calls' in source
    assert "os.replace(stage,destination)" in source


def test_disposed_model_binding_remains_defined_for_finally():
    source=inspect.getsource(audit.execute)
    assert source.count("del model; model=None") == 2
    assert "del model,actual; model=None" in source
    assert "if model is not None: del model" in source
    for line in source.splitlines():
        stripped=line.strip()
        if stripped.startswith("del model") and "actual" not in stripped:
            assert "model=None" in stripped
