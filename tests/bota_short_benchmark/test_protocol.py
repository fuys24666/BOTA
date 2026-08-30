from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.bota_short_benchmark.evaluation import ORDER, bernoulli_jsd
from src.bota_short_benchmark.protocol import load_config
from src.bota_short_benchmark.runner import METHODS
from src.bota_short_benchmark.runner import _base_t5_config, _engine, _prepare_influence_endpoint

ROOT=Path(__file__).resolve().parents[2]


def test_frozen_protocol_budget_and_scenarios():
    config=load_config(ROOT/"configs/bota_short_benchmark_v1.yaml")
    assert config["protocol"]["optimizer_steps"]==200
    assert config["protocol"]["interactions"]==3200
    assert [row["id"] for row in config["protocol"]["scenarios"]]==["L8","L4M4","L3M3H2"]
    assert all(sum(row["composition"].values())==8 for row in config["protocol"]["scenarios"])


def test_selection_is_outcome_blind_and_final_test_forbidden():
    config=load_config(ROOT/"configs/bota_short_benchmark_v1.yaml")
    assert config["protocol"]["selection_fields"]==["train_user_frequency","window_exposure","development_user_count"]
    assert config["protocol"]["selection_uses_labels_or_predictions"] is False
    assert config["evaluation"]=={"split":"Development","final_test":False,"inference_batch_size":4,"bootstrap_resamples":1000}


def test_all_six_methods_have_unique_ids():
    assert len(METHODS)==6 and len(set(METHODS.values()))==6
    assert ORDER==[METHODS[key] for key in ("Original","Retrain","IFRU","SISA","RecEraser","BOTA")]


@pytest.mark.parametrize("field,value",[("optimizer_steps",199),("batch_size",8),("interactions",3199)])
def test_budget_tampering_rejected(tmp_path,field,value):
    import yaml
    source=yaml.safe_load((ROOT/"configs/bota_short_benchmark_v1.yaml").read_text(encoding="utf-8"));source["protocol"][field]=value;path=tmp_path/"bad.yaml";path.write_text(yaml.safe_dump(source),encoding="utf-8")
    with pytest.raises(ValueError,match="budget changed"):load_config(path)


def test_scenario_tampering_rejected(tmp_path):
    import yaml
    source=yaml.safe_load((ROOT/"configs/bota_short_benchmark_v1.yaml").read_text(encoding="utf-8"));source["protocol"]["scenarios"][2]["composition"]={"low":4,"middle":2,"high":2};path=tmp_path/"bad.yaml";path.write_text(yaml.safe_dump(source),encoding="utf-8")
    with pytest.raises(ValueError,match="registry definition changed"):load_config(path)


def test_bernoulli_jsd_is_finite_symmetric_and_zero_on_identity():
    import numpy as np
    left=np.asarray([.1,.5,.9]);right=np.asarray([.2,.4,.8]);assert np.allclose(bernoulli_jsd(left,left),0);assert np.allclose(bernoulli_jsd(left,right),bernoulli_jsd(right,left));assert np.isfinite(bernoulli_jsd(left,right)).all()


def test_wrappers_do_not_offer_finaltest():
    for path in (ROOT/"scripts/bota_if").glob("run_short_*_v1.ps1"):
        text=path.read_text(encoding="utf-8");assert "FinalTest" not in text


def test_evaluation_wrapper_only_forwards_nonempty_optional_run_names():
    text=(ROOT/"scripts/bota_if/run_short_benchmark_evaluation_v1.ps1").read_text(encoding="utf-8")
    for name in ("OriginalRunName","RetrainRunName","IFRURunName","SISARunName","RecEraserRunName","BOTARunName","RunName"):
        assert f"if(${name})" in text
    assert "--original-run-name $OriginalRunName" not in text


def test_synthetic_cli_loads_no_real_model():
    from src.bota_short_benchmark.runner import preflight
    value=preflight(ROOT,ROOT/"configs/bota_short_benchmark_v1.yaml","synthetic","BOTA")
    assert value["model_loaded"] is False and value["optimizer_constructed"] is False and value["test_accessed"] is False


def test_p1_runtime_resolves_strict_t5_base_config_at_second_level():
    config=load_config(ROOT/"configs/bota_short_benchmark_v1.yaml")
    base=_base_t5_config(ROOT,config)
    assert base["protocol_name"]=="t5_reconstructed_official_code"
    assert _engine(ROOT,config)=={"schedule":{"batch_size":16},"optimizer":config["optimizer"]}


def test_ifru_endpoint_clears_training_gradients_and_enters_eval():
    import torch
    model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.01);model(torch.ones(1,2)).sum().backward();assert any(parameter.grad is not None for parameter in model.parameters())
    _prepare_influence_endpoint(model,optimizer,list(model.parameters()))
    assert model.training is False
    assert all(parameter.grad is None for parameter in model.parameters())
