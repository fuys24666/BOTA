from src.bota_short_benchmark.ali_stage4a_stronger_signal_attack_v1 import INTERPRETABLE, MODELS, REQUESTS, synthetic


def test_stage4a_frozen_surface() -> None:
    assert len(REQUESTS)==9
    assert MODELS==("Original","Exact-Masked","BOTA")
    assert INTERPRETABLE==("high_k4",)


def test_only_high_k4_can_receive_formal_decision() -> None:
    result=synthetic()
    assert result["high_k4"]["formal_request_pass"] is True
    assert result["high_k4"]["status"]=="FORMAL_PASS"
    assert result["weak"]["formal_request_pass"] is False
    assert result["weak"]["status"]=="DESCRIPTIVE_WEAK_DU"
    assert result["real_model_loaded"] is False
    assert result["test_accessed"] is False
