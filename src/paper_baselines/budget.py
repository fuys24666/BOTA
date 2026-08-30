from __future__ import annotations

import math
from typing import Any


REQUIRED=("method_id","source_paper","official_repository","official_training_unit","official_epoch_or_step_budget","official_checkpoint_used_for_reported_result","implemented_epoch_budget","implemented_optimizer_step_budget","samples_per_epoch","effective_batch","steps_per_epoch","number_of_models_or_shards","total_expected_optimizer_steps","model_selection_rule","early_stopping_rule","development_evaluation_frequency","final_model_rule","hidden_post_selection_training","budget_matches_reference","overtraining_risk","estimated_runtime","test_accessed")


def _finish(value:dict[str,Any])->dict[str,Any]:
    missing=[key for key in REQUIRED if key not in value]
    if missing:raise ValueError(f"budget audit fields missing: {missing}")
    if value["test_accessed"] is not False or value["hidden_post_selection_training"] is not False:raise ValueError("budget audit safety invariant failed")
    value.update({"schema":"paper-baseline-budget-audit-v1","model_loaded":False,"optimizer_constructed":False,"development_samples_read":False,"final_test_accessed":False})
    return value


def gradient_budget(config:dict[str,Any],kind:str)->dict[str,Any]:
    p=config["protocol"];samples=config["data"]["forget_samples"];steps_per_epoch=math.ceil(samples/p["effective_batch"])
    if kind=="neggrad":
        if p.get("target_steps")!=3000:raise ValueError("NegGrad reported checkpoint budget must remain step3000")
        forward_backward=3000*p["accumulation_steps"];official="public E2URec NegGrad artifact model_3000.pt";checkpoint="model_3000.pt";matches=True;risk=False
    else:
        expected_warmup=math.ceil(samples/p["effective_batch"]) if config.get("deletion_experiment") else 812
        if p.get("target_steps")!=1000 or p.get("warmup_steps")!=expected_warmup or p.get("joint_steps")!=1000-expected_warmup:raise ValueError("PCGrad must share the ratio-bound E2URec student step1000 budget")
        forward_backward=(p["warmup_steps"]+2*p["joint_steps"])*p["accumulation_steps"];official="PCGrad defines gradient surgery, not training length; frozen to its E2URec student comparator";checkpoint="fixed step1000 T5 adaptation";matches=True;risk=False
    return _finish({"method_id":"NegGrad-ForgetOnly-Full" if kind=="neggrad" else "PCGrad-LoRA","source_paper":config["budget"]["source_paper"],"official_repository":config["budget"]["official_repository"],"official_training_unit":"optimizer step","official_epoch_or_step_budget":official,"official_checkpoint_used_for_reported_result":checkpoint,"implemented_epoch_budget":p["target_steps"]/steps_per_epoch,"implemented_optimizer_step_budget":p["target_steps"],"samples_per_epoch":samples,"effective_batch":p["effective_batch"],"steps_per_epoch":steps_per_epoch,"number_of_models_or_shards":1,"total_expected_optimizer_steps":p["target_steps"],"expected_forward_calls":forward_backward,"expected_backward_calls":forward_backward,"model_selection_rule":"preregistered fixed final step; Development diagnostic only","early_stopping_rule":"none","development_evaluation_frequency":len(config["checkpoints"]),"final_model_rule":f"step{p['target_steps']}","hidden_post_selection_training":False,"budget_matches_reference":matches,"overtraining_risk":risk,"estimated_runtime":">=3x E2URec student update count with full parameters" if kind=="neggrad" else "ratio-dependent E2URec-aligned gradient work plus projection","relative_to_e2urec_step1000":3.0 if kind=="neggrad" else forward_backward/(1000*p["accumulation_steps"]),"test_accessed":False})


def partitioned_budget(config:dict[str,Any],kind:str)->dict[str,Any]:
    p=config["protocol"];b=config["budget"];counts=b["initial_partition_counts"];remaining=b["post_delete_partition_counts"]
    groups=lambda values:[value for shard in values for value in (shard if isinstance(shard,list) else [shard])]
    initial_steps=sum(math.ceil(value/p["effective_batch"]) for value in groups(counts));unlearning_steps=sum(math.ceil(value/p["effective_batch"]) for value in groups(remaining));expected=initial_steps+unlearning_steps;forward_backward=sum(math.ceil(value/p["physical_microbatch"]) for value in groups(counts))+sum(math.ceil(value/p["physical_microbatch"]) for value in groups(remaining))
    if expected!=b["total_expected_optimizer_steps"]:raise ValueError("partitioned static budget does not match the executable plan")
    samples=config["data"]["train_samples"]
    return _finish({"method_id":"SISA-T5" if kind=="sisa" else "RecEraser-Adapter","source_paper":b["source_paper"],"official_repository":b["official_repository"],"official_training_unit":"one epoch per shard slice" if kind=="sisa" else "one epoch per isolated local model update","official_epoch_or_step_budget":b["official_epoch_or_step_budget"],"official_checkpoint_used_for_reported_result":b["official_checkpoint_used_for_reported_result"],"implemented_epoch_budget":p["epochs_per_slice"],"implemented_optimizer_step_budget":{"initial":initial_steps,"unlearning_incremental":unlearning_steps},"samples_per_epoch":samples,"effective_batch":p["effective_batch"],"steps_per_epoch":math.ceil(samples/p["effective_batch"]),"number_of_models_or_shards":p["shards"],"total_expected_optimizer_steps":expected,"expected_forward_calls":forward_backward,"expected_backward_calls":forward_backward,"initial_training_optimizer_steps":initial_steps,"unlearning_incremental_optimizer_steps":unlearning_steps,"model_selection_rule":"fixed one epoch per local slice; no checkpoint post-selection","early_stopping_rule":"none","development_evaluation_frequency":0 if kind=="sisa" else 1,"final_model_rule":"aggregate post-deletion affected-local-model states","hidden_post_selection_training":False,"budget_matches_reference":True,"overtraining_risk":False,"estimated_runtime":b["estimated_runtime"],"relative_to_e2urec_step1000":expected/1000,"test_accessed":False})
