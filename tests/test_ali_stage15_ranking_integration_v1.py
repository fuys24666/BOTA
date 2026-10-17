"""Opt-in, bounded real-data/API check; never a formal ranking result."""
import copy
import os
from pathlib import Path

import numpy as np
import pytest

from src.bota_short_benchmark import ali_stage15_ranking_v1 as ranking


@pytest.mark.skipif(os.environ.get("BOTA_STAGE15_GPU_SMOKE") != "1", reason="opt-in local GPU smoke")
def test_real_data_token_audit_and_five_checkpoint_interfaces(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    workspace = Path("C:/python/E2URec-main")
    _, benchmarks, plan, contract, users, movies = ranking.prepare(repo, workspace, repo/"configs/ali_stage15_ranking_v1.json")
    tokenizer_path = ranking.safe_path(workspace,contract["dependencies"]["pretrained_model_directory"]["path"])
    tokenizer = ranking.T5Tokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    dataset = ranking.RankingDataset(plan, tokenizer, users, movies)
    stats = dataset.validate_tokens()
    print("TOKEN_AUDIT", stats, flush=True)
    assert stats["prompts"] == plan["candidate_count"] and stats["max_input_tokens"] <= 512
    requested = next(e["users"] for e in contract["endpoints"] if e["seed"] == 42 and e["scenario"] == "L8")
    print("COVERAGE", {"eligible_users":len(plan["users"]), "requested":requested,
                     "covered":[u for u in requested if u in {r['user'] for r in plan['users']}]}, flush=True)
    # Only four candidate predictions per model/component; no ranking metrics.
    small = copy.copy(dataset)
    small.records = dataset.records[:4]
    small.encoded = dataset.encoded[:4]
    scorer = ranking.score_model
    def checked(model, data, device, progress, component=0):
        masses = scorer(model, data, device, progress, component)
        legacy = ranking.evaluation._single_predictions(model, data, list(range(len(data))), device, 4)
        np.testing.assert_allclose(masses[:,1]/masses.sum(1), legacy["probability"], rtol=1e-6, atol=1e-7)
        return masses
    monkeypatch.setattr(ranking,"score_model",checked)
    monkeypatch.chdir(workspace)
    for method in ("Original","BOTA64","E2URec","Retain-Retrain-P5","RecEraser"):
        entry = next(e for e in contract["endpoints"] if e["seed"]==42 and e["scenario"]=="L8" and e["method"]==method)
        scores = ranking.predict(workspace,benchmarks[42],entry,small,tmp_path/(method+".json"))
        assert len(scores)==4 and np.isfinite(scores).all()
        print("INTERFACE_PASS",method,flush=True)
