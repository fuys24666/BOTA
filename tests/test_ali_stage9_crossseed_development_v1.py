from src.bota_short_benchmark import ali_stage9_crossseed_development_v1 as stage9


def test_stage9_confirmation_surface_is_fixed():
    assert stage9.HELD_OUT_SEEDS == (41, 43)
    assert stage9.REPORTED_SEEDS == (41, 42, 43)
    assert stage9.METHODS == ("BOTA", "IFRU", "E2URec", "NegGrad", "PCGrad", "RecEraser")
    assert stage9.SELECTED_IDS["BOTA"] == "bota_r1"
    assert stage9.SELECTED_IDS["RecEraser"] == "receraser_lr0.0005_e3"


def test_stage9_seed_specific_output_roots_do_not_collide():
    config = {"output_root": "outputs/x", "benchmarks": {41: {}, 43: {}}, "development_partition": {}, "selected_configurations": {
        "IFRU": {"relative_damping_ratio": .1, "curvature_panel_samples": 512, "cg_max_iterations": 40},
        "E2URec": {"teacher_steps": 1200, "teacher_learning_rate": .00005, "student_learning_rate": .001, "forget_weight": .2, "checkpoint": 200},
        "NegGrad": {"learning_rate": .001, "forget_weight": .05, "checkpoint": 25},
        "PCGrad": {"learning_rate": .0005, "checkpoint": 25},
        "RecEraser": {"shards": 4, "feature_dimensions": 32, "learning_rate": .0005, "epochs_per_slice": 3, "attention_steps": 200, "attention_learning_rate": .1},
    }, "selection": {"auc_damage_max": .005, "logloss_damage_max": .01}, "runtime": {}}
    assert stage9._seed_config(config, 41)["output_root"] != stage9._seed_config(config, 43)["output_root"]


def test_stage7_helpers_resolve_run_seed():
    assert stage9.stage7._run_seed({"protocol": {"seed": 42, "run_seed": 41}}) == 41
    assert stage9.stage7._original_run_name({"protocol": {"seed": 42, "run_seed": 43}}) == "bota_short_original_seed43_v3"
