import json
from pathlib import Path

import pytest
import torch

from src.bota_short_benchmark import ali_stage11_svd_v1 as s11


def test_orthogonal_and_rank_one_spectra():
    report, _ = s11.spectrum(torch.eye(8, dtype=torch.float64))
    assert report["entropy_effective_rank"] == pytest.approx(8)
    assert report["stable_rank"] == pytest.approx(8)
    assert report["energy_ranks"]["0.9"] == 8
    x = torch.arange(1, 6, dtype=torch.float64)[:, None] @ torch.ones(1, 7, dtype=torch.float64)
    report, _ = s11.spectrum(s11.accumulate_gram(x, 3))
    assert report["entropy_effective_rank"] == pytest.approx(1, abs=1e-6)
    assert report["numerical_rank"] == 1


def test_blocked_gram_and_singular_values_match_direct():
    x = torch.randn(9, 23, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    gram = s11.accumulate_gram(x, 4)
    torch.testing.assert_close(gram, x @ x.T)
    report, _ = s11.spectrum(gram)
    torch.testing.assert_close(torch.tensor(report["singular_values"], dtype=torch.float64), torch.linalg.svdvals(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_block_gram_matches_cpu():
    x = torch.randn(8, 29, generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(s11.accumulate_gram(x, 7, "cuda"), s11.accumulate_gram(x, 7, "cpu"))


def test_zero_bank_and_nonfinite_rejected():
    gram = s11.accumulate_gram(torch.zeros(3, 7), 2)
    report, _ = s11.spectrum(gram)
    assert report["entropy_effective_rank"] == 0
    assert report["energy_ranks"]["0.99"] == 0
    directions, normal, _, pairs = s11.direction_diagnostics(gram)
    assert directions["zero_rows"] == 3 and normal.shape == (0, 0) and pairs == []
    assert s11.spectrum(normal)[0]["numerical_rank"] == 0
    with pytest.raises(ValueError, match="Non-finite"):
        s11.accumulate_gram(torch.tensor([[float("nan")]]), 2)
    with pytest.raises(ValueError, match="negative"):
        s11.spectrum(torch.diag(torch.tensor([1., -.1], dtype=torch.float64)))


def test_normalization_distinguishes_magnitude_from_direction():
    x = torch.diag(torch.tensor([100., 1., 1., 0.], dtype=torch.float64))
    raw, _ = s11.spectrum(x @ x.T)
    directions, normal, _, pairs = s11.direction_diagnostics(x @ x.T)
    normalized, _ = s11.spectrum(normal)
    assert raw["stable_rank"] < 1.01
    assert normalized["stable_rank"] == pytest.approx(3)
    assert directions["zero_rows"] == 1 and len(pairs) == 3
    assert all(row["cosine"] == 0 for row in pairs)


def test_query_errors_match_direct_svd_and_full_rank():
    x = torch.randn(7, 19, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    rows, ranks = [0, 3, 6], [1, 3, 7]
    exact = x[rows].sum(0) + .2
    _, u = s11.spectrum(s11.accumulate_gram(x, 5))
    result = s11.query_errors(x, u, rows, exact, ranks, 4)
    du, ds, dv = torch.linalg.svd(x, full_matrices=False)
    baseline = torch.linalg.vector_norm(x[rows].sum(0)-exact)/torch.linalg.vector_norm(exact)
    for r, row in zip(ranks, result):
        compressed = ((du[:, :r]*ds[:r]) @ dv[:r])[rows].sum(0)
        reference = torch.linalg.vector_norm(compressed-exact)/torch.linalg.vector_norm(exact)
        assert row["parameter_residual"] == pytest.approx(float(reference), abs=1e-10)
        assert row["fp32_parameter_residual"] == pytest.approx(float(baseline))
    assert result[-1]["relative_residual_increase"] == pytest.approx(0, abs=1e-10)


def test_exact_and_query_zero_denominators_are_explicit():
    x = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
    _, u = s11.spectrum(x @ x.T)
    row = s11.query_errors(x, u, [0, 1], torch.zeros(2), [1, 2], 1)[0]
    assert row["parameter_residual"] is None
    assert row["relative_residual_increase"] is None
    assert row["query_compression_relative_l2"] is None


def test_synthetic_and_path_safety(tmp_path):
    assert s11.synthetic()["status"] == "SYNTHETIC_PASS"
    with pytest.raises(ValueError, match="escapes"):
        s11.checked_path(tmp_path, "../outside")


def test_module_order_is_preserved():
    result = s11.flatten({"B": {"a": torch.tensor([1.]), "z": torch.tensor([2.])}}, ["z", "a"])
    assert result.tolist() == [2., 1.]
    with pytest.raises(ValueError):
        s11.flatten({"B": {"a": torch.ones(1)}}, ["z"])


def test_small_full_run_resume_summary_and_integrity(tmp_path, monkeypatch):
    repo, workspace = tmp_path/"repo", tmp_path/"work"
    workspace.mkdir()
    repo.mkdir()
    x = torch.randn(8, 13, generator=torch.Generator().manual_seed(7))
    bank = workspace/"bank.pt"
    hashes = [f"hash{i}" for i in range(8)]
    torch.save({"impacts": x, "seed": 41, "user_hashes": hashes, "selected_arm": "FS_empirical_fisher"}, bank)
    original = workspace/"original.pt"
    torch.save({"B": {"b": torch.zeros(13)}}, original)
    queries = {}
    for sid, indices in (("L8", [0, 2]), ("L4M4", [1, 3])):
        bota, exact = workspace/f"{sid}_bota.pt", workspace/f"{sid}_exact.pt"
        q = x[indices].sum(0)
        torch.save({"B": {"b": q}}, bota)
        torch.save({"B": {"b": q+.01}}, exact)
        queries[sid] = {"rows": indices, "bota": str(bota), "exact_masked": str(exact)}
    old_metrics = workspace/"old.json"
    old_metrics.write_text('{"rows": []}')
    config = {"bank_shape": [8, 13], "output_root": "outputs/ali_stage11_svd_v1", "negative_eigenvalue_relative_tolerance": 1e-10, "numerical_eigenvalue_relative_threshold": 1e-12, "fp32_endpoint_reproduction_relative_tolerance": 1e-5}
    binding = {"config": config, "seed": 41, "bank": str(bank), "user_hashes": hashes,
               "original": str(original), "queries": queries, "stage6_metrics": str(old_metrics),
               "source_hashes": {str(bank): s11.sha256_file(bank)}, "code_hashes": {"fake": "123"}}
    monkeypatch.setattr(s11, "preflight", lambda *args: binding)
    monkeypatch.setattr(s11, "git_snapshot", lambda *args: {"clean": False})
    monkeypatch.setattr(s11, "load_config", lambda *args: config)
    monkeypatch.setattr(s11, "RANKS", (1, 4, 8))
    monkeypatch.setattr(s11, "SEEDS", (41,))
    result = s11.execute(repo, workspace, 41, "cpu", 4)
    assert result["status"] == "COMPLETED"
    assert s11.execute(repo, workspace, 41, "cpu", 4)["status"] == "VERIFIED_ALREADY_COMPLETED"
    assert s11.summarize(repo, workspace)["status"] == "DIAGNOSTICS_COMPLETE"
    assert s11.summarize(repo, workspace)["status"] == "DIAGNOSTICS_COMPLETE"
    out = Path(result["run_dir"])
    metric = json.loads((out/"metrics.json").read_text())
    assert len(metric["query_rows"]) == 6 and metric["test_accessed"] is False
    (out/"spectrum.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        s11.validate_run(out, binding)
    assert s11.sha256_file(bank) == binding["source_hashes"][str(bank)]
