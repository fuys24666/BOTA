import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.bota_short_benchmark import ali_stage15_ranking_v1 as ranking


def row(user, item, label=1, index=0, history=(1, 2)):
    return SimpleNamespace(authoritative_user_id=user, movie_id=item, label=label,
        timestamp=100+index, raw_source_row=index, processed_row_index=index,
        history_ids=list(history), history_ratings=[4]*len(history))


def test_perfect_and_bad_ranking():
    assert ranking.user_metrics([1, 2, 3], [2], [.1, .9, .2])["ndcg@5"] == 1
    assert ranking.user_metrics([1, 2, 3], [2], [.9, .1, .2])["ndcg@5"] == .5


def test_recall_denominator_and_ndcg_cutoff():
    result = ranking.user_metrics(list(range(30)), list(range(10)), list(np.linspace(1, 0, 30)))
    assert result["recall@5"] == .5
    assert result["ndcg@5"] == 1
    assert result["recall@20"] == 1


def test_ties_use_item_id_not_relevance_or_input_order():
    a = ranking.user_metrics([4, 3, 2, 1], [4], [.5]*4)
    b = ranking.user_metrics([1, 2, 3, 4], [4], [.5]*4)
    assert a == b
    assert a["ndcg@5"] == pytest.approx(1/math.log2(5))


@pytest.mark.parametrize("items,positive,scores", [([1,1],[1],[.1,.2]), ([1],[1],[float('nan')]),
    ([1],[1],[float('inf')]), ([1],[1],[1.1]), ([1],[],[.2]), ([1],[2],[.2]), ([1],[1,1],[.2]), ([1],[1],[])])
def test_invalid_scores_and_labels_fail(items, positive, scores):
    with pytest.raises(ValueError): ranking.user_metrics(items, positive, scores)


def test_candidates_exclude_training_history_and_all_development_observations():
    train = [row(10, 3)]
    dev = [row(10, 4, index=0), row(10, 5, index=1), row(10, 6, label=0, index=2),
           row(20, 7, label=0, index=3)]
    plan = ranking.build_candidates(train, dev, [0, 3], set(range(1, 160)))
    assert len(plan["users"]) == 1
    u = plan["users"][0]
    assert u["positive_items"] == [4]
    assert len(u["items"]) == 101
    assert not set(u["items"]) & {1, 2, 3, 5, 6}
    assert plan["coverage"][1]["status"] == "NO_ELIGIBLE_POSITIVE"
    assert plan == ranking.build_candidates(train, dev, [0, 3], set(reversed(range(1, 160))))


def test_common_context_does_not_include_later_development_history():
    dev = [row(10, 4, index=0), row(10, 5, index=1, history=(1,2,4))]
    plan = ranking.build_candidates([], dev, [0,1], set(range(1,150)))
    assert plan["users"][0]["history_ids"] == [1,2]
    assert plan["users"][0]["positive_items"] == [4,5]


def test_small_pool_and_out_of_catalogue_reported():
    dev = [row(10, 4), row(10, 88, index=1)]
    plan = ranking.build_candidates([], dev, [0,1], {1,2,3,4,5})
    assert plan["users"][0]["items"] == [3,4,5]
    assert plan["coverage"][0]["positive_out_of_catalogue"] == 1
    assert plan["coverage"][0]["negative_count"] == 2


def test_no_positive_is_not_silent_zero():
    with pytest.raises(ValueError, match="no eligible"):
        ranking.build_candidates([], [row(10,4,0)], [0], set(range(1,150)))


def test_conflicting_duplicate_user_item_fails():
    with pytest.raises(ValueError, match="conflicting"):
        ranking.build_candidates([], [row(10,4), row(10,4,0,1)], [0], set(range(1,150)))


def test_macro_users_not_candidate_weighted_and_coverage():
    plan = {"users": [{"user":1,"items":[1,2],"positive_items":[1]},
                       {"user":2,"items":[1,2,3],"positive_items":[1]}], "candidate_count":5}
    result = ranking.metrics_for(plan, [.9,.1, .1,.9,.8], {1,3})
    assert result["aggregates"][0]["ndcg@5"] == .75
    assert result["covered_requested_users"] == [1]
    assert result["uncovered_requested_users"] == [3]
    no_request = ranking.metrics_for(plan, [.9,.1, .1,.9,.8], {3})
    assert no_request["aggregates"][1]["ndcg@10"] is None


@pytest.mark.parametrize("name", ["../escape", "data/test/data.json", "outputs/FinalTest/predictions.json"])
def test_forbidden_paths(tmp_path, name):
    with pytest.raises(ValueError): ranking.safe_path(tmp_path, name)


def test_protocol_rejects_retuning_and_finaltest(tmp_path):
    for key, value in (("response_scale",32), ("final_test",True), ("negative_count",99)):
        protocol = copy.deepcopy(ranking.PROTOCOL); protocol[key] = value
        config = {"schema":ranking.SCHEMA,"protocol":protocol,"output_root":ranking.ROOT,"test_access_policy":"forbidden"}
        p = tmp_path / "config.json"; p.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="protocol changed"):
            ranking.load_config(tmp_path,p)


def test_atomic_bundle_and_tampering(tmp_path):
    stage = ranking.stage_path(tmp_path)
    ranking.atomic_json(stage/"value.json", {"test_accessed":False})
    target = tmp_path/"published"
    ranking.publish(stage,target); ranking.verify_bundle(target)
    (target/"value.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"): ranking.verify_bundle(target)


def test_first_token_scoring_constant_dummy_target_no_backward(tmp_path):
    class Model:
        config = SimpleNamespace(decoder_start_token_id=0)
        def __call__(self, **kwargs):
            assert not torch.is_grad_enabled()
            assert torch.all(kwargs["labels"][:,0] == 2163)
            z = torch.zeros((len(kwargs["input_ids"]),1,2200))
            z[:,:,2163] = math.log(3)
            return SimpleNamespace(logits=z)
    class Data:
        def __len__(self): return 7
        def __getitem__(self, i): return i
        def collate_fn(self, batch): return {"input_ids":torch.ones((len(batch),2),dtype=torch.long),
                                           "target_ids":torch.full((len(batch),3),2163,dtype=torch.long)}
    masses = ranking.score_model(Model(),Data(),torch.device("cpu"),tmp_path/"progress.json")
    np.testing.assert_allclose(masses[:,1]/masses.sum(1), .75, atol=1e-6)


def test_frozen_config_exists_and_matches():
    repo = Path(__file__).resolve().parents[1]
    config = ranking.load_config(repo,repo/"configs/ali_stage15_ranking_v1.json")
    assert config["protocol"]["response_scale"] == 64
    assert config["protocol"]["hyperparameter_selection"] is False


def test_partial_then_complete_summary_recomputed_and_reusable(tmp_path):
    plan = {"users": [{"user": 1, "items": [1,2], "positive_items": [1]}], "candidate_count": 2}
    entries = [{"seed":s,"scenario":sid,"method":m,"users":[1,9]}
               for s in ranking.SEEDS for sid in ranking.SCENARIOS for m in ranking.METHODS]
    contract = {"endpoints":entries}
    partial = ranking.summarize(tmp_path,plan,contract)
    assert partial["status"] == "PARTIAL" and len(partial["missing"]) == 66
    assert not (tmp_path/"summary").exists()
    for entry in entries:
        stage = ranking.stage_path(tmp_path)
        identity = {"contract_sha256":ranking.canonical_hash(contract),
                    "candidate_sha256":ranking.canonical_hash(plan),"endpoint":entry}
        ranking.atomic_json(stage/"identity.json",identity)
        ranking.atomic_json(stage/"scores.json",{"scores":[.8,.2],"candidate_sha256":ranking.canonical_hash(plan)})
        ranking.atomic_json(stage/"metrics.json",ranking.metrics_for(plan,[.8,.2],{1,9}))
        name = f"s{entry['seed']}_{entry['scenario']}_{entry['method']}"
        ranking.publish(stage,tmp_path/"cells"/name)
    full = ranking.summarize(tmp_path,plan,contract)
    assert full["status"] == "COMPLETED" and len(full["three_seed"]) == 66
    assert full["hyperparameter_reselection"] is False and full["finaltest_authorized"] is False
    assert ranking.summarize(tmp_path,plan,contract) == full
    assert full["three_seed"][2]["ndcg@10"] is None  # empty retained group


def test_prompt_token_overflow_rejected():
    dataset = ranking.RankingDataset.__new__(ranking.RankingDataset)
    dataset.records = [{"input":"too long","output":"Yes."}]
    dataset.encoded = []
    dataset.tokenizer = SimpleNamespace(encode=lambda *_args,**_kw: list(range(513)))
    with pytest.raises(ValueError, match="exceeds 512"):
        dataset[0]
