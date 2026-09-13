import torch

from src.bota_short_benchmark.ali_stage6_storage_screen_v1 import BANK_SIZES, _compose, _encode, _measure_artifact, _representation_names, synthetic


def test_fixed_stage6_surface():
    result=synthetic()
    assert result["bank_sizes"]==list(BANK_SIZES)
    assert len(result["representations"])==12
    assert result["rows_per_seed"]==120
    assert result["selected"]=="topq_0.03125"
    assert result["development_predictions_accessed"] is False
    assert result["test_accessed"] is False


def test_quantized_composition_shapes():
    x=torch.tensor([[1.,-2.,3.,-4.],[2.,1.,-1.,.5]])
    rows=[0,1]
    for name in ("FP32","FP16","INT8_per_vector"):
        value=_compose(_encode(name,x),rows)
        assert value.shape==(4,)
        assert torch.isfinite(value).all()


def test_representation_names_are_unique():
    names=_representation_names()
    assert len(names)==len(set(names))==12


def test_decimal_representation_name_is_serializable(tmp_path):
    size,seconds=_measure_artifact(tmp_path,"k256_topq_0.03125",{"kind":"FP32","values":torch.ones(2)})
    assert size>0
    assert seconds>=0
    assert list((tmp_path/"measure").iterdir())==[]


def test_low_rank_slice_owns_only_requested_storage():
    x=torch.ones(12,32)
    low_rank={"u":torch.ones(12,12),"basis":torch.ones(12,32)}
    encoded=_encode("shared_low_rank_r4",x,low_rank)
    assert encoded["coefficients"].untyped_storage().nbytes()==encoded["coefficients"].numel()*4
    assert encoded["basis"].untyped_storage().nbytes()==encoded["basis"].numel()*4
