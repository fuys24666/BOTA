import inspect
from pathlib import Path

from src.bota_short_benchmark import fisher_ablation_v1 as audit


CONFIG = Path("configs/bota_short_fisher_ablation_v1.yaml")


def _parameter(error): return {"relative_l2_error": error}


def _behavior(residual, auc=0.0, loss=0.0):
    return {"local_residual": residual, "auc_damage_vs_original": auc, "log_loss_damage_vs_original": loss}


def test_protocol_is_l8_shared_trajectory_and_finaltest_forbidden():
    config = audit.load_config(CONFIG)
    assert config["authority"]["scenario"] == "L8"
    assert config["protocol"]["physical_optimizer_step_calls"] == 200
    assert config["protocol"]["shared_canonical_trajectory"] is True
    assert config["evaluation"]["final_test"] is False


def test_fisher_materiality_requires_parameter_behavior_and_utility():
    decision = audit.load_config(CONFIG)["decision"]
    parameter = {audit.ARMS[0]: _parameter(.50), audit.ARMS[1]: _parameter(.40)}
    metrics = {audit.ARMS[0]: _behavior(.90), audit.ARMS[1]: _behavior(.80)}
    assert audit.classify(parameter, metrics, decision)["classification"] == "block_empirical_fisher_materially_helpful_single_seed"
    metrics[audit.ARMS[1]] = _behavior(.80, auc=.01)
    assert audit.classify(parameter, metrics, decision)["classification"] == "mixed_fisher_contribution_single_seed"


def test_source_only_is_not_mislabeled_as_cross_seed_stable():
    decision = audit.load_config(CONFIG)["decision"]
    parameter = {audit.ARMS[0]: _parameter(.41), audit.ARMS[1]: _parameter(.40)}
    metrics = {audit.ARMS[0]: _behavior(.81), audit.ARMS[1]: _behavior(.80)}
    result = audit.classify(parameter, metrics, decision)
    assert result["classification"] == "source_only_explains_primary_effect_single_seed"
    assert result["single_seed_stability_claimed"] is False


def test_execute_has_no_masked_retraining_or_finaltest_path():
    source = inspect.getsource(audit.execute)
    assert "runner._train_named" not in source
    assert "final_test_json" not in source
    assert "StepBudget(200)" in source
