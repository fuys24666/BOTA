"""CPU regression tests deliberately recreate batch-dependent padding scoring."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from src.bota_short_benchmark import ali_stage24_masked_scoring as m
from src.bota_short_benchmark import ali_stage24_masked_rescore_v1 as s

AUDIT = {"samples": 8, "local_samples": 4, "batch_sizes": [1, 4, 8],
         "orders": ["length_sorted", "short_long_interleaved"],
         "probability_atol": 1e-5, "probability_rtol": 1e-5, "loss_atol": 1e-4, "loss_rtol": 1e-5}


class Dataset:
    def __getitem__(self, i):
        return {"input_ids": torch.full((i+1,), 2, dtype=torch.long),
                "target_ids": torch.tensor([2163 if i % 2 else 465, 1])}

    def collate_fn(self, batch):
        inputs = torch.zeros(len(batch), max(len(x["input_ids"]) for x in batch), dtype=torch.long)
        for i, row in enumerate(batch):
            inputs[i, :len(row["input_ids"])] = row["input_ids"]
        return {"input_ids": inputs, "target_ids": torch.stack([x["target_ids"] for x in batch])}


class Model(torch.nn.Module):
    def __init__(self, ignore_mask=False):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.tensor(0.))
        self.ignore_mask = ignore_mask

    def forward(self, input_ids, labels, attention_mask=None):
        mask = torch.ones_like(input_ids) if attention_mask is None or self.ignore_mask else attention_mask
        value = (input_ids*mask).sum(1).float()/mask.sum(1) + self.shift
        logits = torch.zeros(len(input_ids), labels.shape[1], 2200)
        logits[:, :, 2163] = value[:, None]
        return SimpleNamespace(logits=logits)


def context():
    return {"indices": list(range(8)), "pairs": [{"local_indices": [0, 1, 2, 3]}]}


def test_padding_bug_reproduced_and_mask_fix_is_batch_invariant():
    model = Model().eval()
    r = m.invariance_audit(model, Dataset(), context(), "cpu", 0, AUDIT)
    assert r["status"] == "PASS"
    assert all(x["passed"] for x in r["rows"])
    assert any(not x["passed"] for x in r["legacy_negative_control"])
    assert max(x["probability_max_abs_delta"] for x in r["legacy_negative_control"]) > .1


def test_ignored_mask_fails_gate():
    r = m.invariance_audit(Model(ignore_mask=True).eval(), Dataset(), context(), "cpu", 0, AUDIT)
    assert r["status"] == "FAIL"


def test_single_batch_no_padding_matches_legacy_formula():
    model, data = Model().eval(), Dataset()
    a = m.predict(model, data, [2, 5], "cpu", 1, 0)
    b = m.predict(model, data, [2, 5], "cpu", 1, 0, masked=False)
    assert a == b


def test_reorder_aligns_ids_not_positions():
    model, data = Model().eval(), Dataset()
    a = m.predict(model, data, [0, 1, 4], "cpu", 1, 0)
    b = m.predict(model, data, [4, 1, 0], "cpu", 4, 0)
    assert m.difference(a, b, [0, 1, 4], [4, 1, 0], AUDIT)["passed"]
    b["gold_label"][0] = 1
    with pytest.raises(ValueError, match="labels"):
        m.difference(a, b, [0, 1, 4], [4, 1, 0], AUDIT)


def test_nonfinite_and_invalid_orders_rejected():
    a = m.predict(Model().eval(), Dataset(), [0, 1], "cpu", 1, 0)
    with pytest.raises(ValueError, match="sample IDs"):
        m.difference(a, a, [0, 1], [0, 2], AUDIT)
    a["probability"][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        m.difference(a, a, [0, 1], [0, 1], AUDIT)


def test_training_mode_and_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="model.eval"):
        m.predict(Model(), Dataset(), [0], "cpu", 1, 0)
    with pytest.raises(ValueError, match="duplicate"):
        m.predict(Model().eval(), Dataset(), [0, 0], "cpu", 1, 0)


def test_inference_does_not_change_weights_or_gradients():
    model = Model().eval()
    before = model.shift.detach().clone()
    m.predict(model, Dataset(), list(range(8)), "cpu", 4, 0)
    assert torch.equal(before, model.shift) and model.shift.grad is None


def test_bundle_resume_tamper_and_interruption(tmp_path):
    r = s.task(tmp_path, "unit", "h", lambda stage: {"status": "OK"})
    assert s.task(tmp_path, "unit", "h", lambda stage: pytest.fail("must resume")) == r
    with pytest.raises(ValueError, match="contract"):
        s.verify(tmp_path/"unit", "changed")
    (tmp_path/"unit/result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact"):
        s.verify(tmp_path/"unit")
    def interrupt(stage):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        s.task(tmp_path, "interrupted", "h", interrupt)
    assert not (tmp_path/"interrupted").exists()
    assert list((tmp_path/".work").glob("*/FAILED.json"))


def test_config_is_frozen():
    config = s.load_config(Path(__file__).resolve().parents[2])
    assert config["optimizer_step_calls"] == 0 and config["test_accessed"] is False


def integration_setup(tmp_path, monkeypatch, ignore_mask=False):
    root, parent = tmp_path/"new", tmp_path/"old"
    prefix = "Amazon_s42"
    pairs = [{"id": "A1", "users": [10, 20], "local_indices": [0, 1, 2, 3],
              "matched_indices": [0, 1, 2, 3], "members": [
                  {"user": 10, "indices": [0, 2], "matched_indices": [0, 2]},
                  {"user": 20, "indices": [1, 3], "matched_indices": [1, 3]}]}]
    ctx = {"dataset": "Amazon", "seed": 42, "lambda": 48, "pairs": pairs,
           "indices": list(range(8)), "control_indices": [4, 5, 6, 7],
           "benchmark": {"source": {"development_json": "unused"}}}
    original = {"B": {"B": torch.tensor(0.)}}
    single = {10: {"B": torch.tensor(.1)}, 20: {"B": torch.tensor(.2)}}
    pair = {"A1": {"B": torch.tensor(.4)}}
    bank = {"names": ["B"], "users": [10, 20], "vectors": {"0": torch.tensor([[.05], [.10]]),
            "48": torch.tensor([[.07], [.14]])}, "failures": {}}
    # compose expects a vector coordinate; use vector parameters consistently.
    original["B"]["B"] = original["B"]["B"].reshape(1)
    single = {u: {"B": x["B"].reshape(1)} for u, x in single.items()}
    pair = {k: {"B": x["B"].reshape(1)} for k, x in pair.items()}
    endpoints = s.s23.make_endpoints(original, ["B"], bank, single, pair, pairs)
    monkeypatch.setattr(s, "load_endpoints", lambda *a: (endpoints, bank))
    model = Model(ignore_mask).eval()
    model.shift = torch.nn.Parameter(torch.zeros(1))
    data = Dataset()
    for name, values in endpoints.items():
        with torch.no_grad():
            model.shift.copy_(values["B"])
        pred = m.predict(model, data, ctx["indices"], "cpu", 4, 0, masked=False)
        s.s23.task(parent, prefix+"/predictions/"+name, "old", lambda stage, pred=pred: {"prediction": pred})
    monkeypatch.setattr(s.s23, "fresh", lambda *a: (model, ["B"], [model.shift], None,
        SimpleNamespace(pad_token_id=0), data, None, "cpu", original))
    monkeypatch.setattr(s.s23.s18.s8, "JsonPromptDataset", lambda *a: data)
    monkeypatch.setattr(s.s23.s18, "release", lambda: None)
    for forbidden in ("replay", "build_bank"):
        monkeypatch.setattr(s.s23, forbidden, lambda *a: pytest.fail("must not train"))
    monkeypatch.setattr(s.s23.runner, "_train_named", lambda *a: pytest.fail("must not train"))
    monkeypatch.setattr(s.s23.s18, "transport", lambda *a: pytest.fail("must not transport"))
    config = {"audit": AUDIT, "chunk_size": 256, "inference_batch_size": 4}
    return root, parent, prefix, ctx, config


def test_end_to_end_scoring_metrics_and_resume_without_training(tmp_path, monkeypatch):
    root, parent, prefix, ctx, config = integration_setup(tmp_path, monkeypatch)
    r = s.rescore_context(root, parent, tmp_path, ctx, "new", "old", config)
    assert len(r["rows"]) == 10 and all(x["status"] == "OK" for x in r["rows"])
    old_digest = s.sha256_file(parent/prefix/"predictions/Original/result.json")
    monkeypatch.setattr(s.s23, "fresh", lambda *a: pytest.fail("resume should not load model"))
    assert s.rescore_context(root, parent, tmp_path, ctx, "new", "old", config) == r
    assert old_digest == s.sha256_file(parent/prefix/"predictions/Original/result.json")
    assert s.verify(root/prefix/"predictions/Original")["full_cohort_gate"]["passed"]


def test_failed_audit_prevents_scoring_publication(tmp_path, monkeypatch):
    root, parent, prefix, ctx, config = integration_setup(tmp_path, monkeypatch, ignore_mask=True)
    with pytest.raises(RuntimeError, match="batch invariance failed"):
        s.rescore_context(root, parent, tmp_path, ctx, "new", "old", config)
    assert s.verify(root/prefix/"audits/Original")["status"] == "FAIL"
    assert not (root/prefix/"predictions/Original").exists()


def test_incomplete_summary_reports_all_missing(tmp_path):
    plan = {"parent_protocol": {"contexts": {"Amazon_s42": {}, "ML1M_s43": {}}}}
    r = s.analyze(tmp_path/"new", tmp_path/"old", plan)
    assert r["status"] == "INCOMPLETE" and len(r["missing_contexts"]) == 2


def test_complete_summary_pairs_metrics_and_is_resumable(tmp_path, monkeypatch):
    root, parent, prefix, ctx, config = integration_setup(tmp_path, monkeypatch)
    plan = {"parent_protocol": {"contexts": {prefix: ctx}}}
    contract = s.canonical_hash(plan)
    r = s.rescore_context(root, parent, tmp_path, ctx, contract, "old", config)
    parent_contract = s.canonical_hash(plan["parent_protocol"])
    s.s23.task(parent, prefix+"/behavior", parent_contract, lambda stage: {"rows": r["rows"]})
    s.s23.task(parent, "summary", parent_contract, lambda stage: {"parameters": []})
    summary = s.analyze(root, parent, plan)
    assert summary["status"] == "COMPLETED" and summary["endpoint_count"] == 11
    assert len(summary["behavior"]) == 10
    assert (root/"summary/legacy_vs_masked_metrics.csv").exists()
    assert (root/"summary/batch_invariance.csv").exists()
    assert s.analyze(root, parent, plan) == summary
