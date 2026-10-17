"""Opt-in bounded Development-only API smoke; never a formal test result."""
import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.bota_short_benchmark import ali_stage16_finaltest_v1 as s


@pytest.mark.skipif(os.environ.get('BOTA_STAGE16_GPU_SMOKE')!='1',reason='opt-in bounded Development smoke')
def test_admission_and_three_endpoint_interfaces_on_four_development_candidates(tmp_path,monkeypatch):
    repo=Path(__file__).resolve().parents[1];workspace=Path('C:/python/E2URec-main')
    def forbidden(*a,**k):pytest.fail('FinalTest materialization during Development smoke')
    monkeypatch.setattr(s.s10.lineage,'_final_lineage',forbidden)
    monkeypatch.setattr(s.s10.lineage,'_validate_final_file',forbidden)
    _,contract=s.preflight(repo,workspace)
    assert not contract['test_accessed']
    assert json_roundtrip(contract)==contract
    # Reuse Stage 15's already-frozen Development candidates. Never materialize FinalTest.
    plan=copy.deepcopy(s.read(workspace/s.s15.ROOT/'protocol/candidates.json'))
    plan['users']=plan['users'][:1];plan['users'][0]['items']=plan['users'][0]['items'][:4]
    plan['candidate_count']=4
    tokenizer=s.T5Tokenizer.from_pretrained(str(workspace/contract['dependencies']['pretrained_model_directory']['path']),local_files_only=True)
    users,movies=s.s15.lineage.load_raw_metadata(workspace/'data/ml-1m/raw_data')
    dataset=s.s15.RankingDataset(plan,tokenizer,users,movies)
    assert dataset.validate_tokens()['prompts']==4
    monkeypatch.chdir(workspace)
    for method in ('BOTA0','BOTA64','RecEraser'):
        entry=next(e for e in contract['endpoints'] if e['seed']==42 and e['scenario']=='L8' and e['method']==method)
        pred=s.s10._predict_endpoint(workspace,contract['benchmarks']['42'],entry,dataset,torch.device('cuda:0'),list(range(4)))
        s.validate_prediction(pred,4,[1]*4)
        comparison=s.s15.predict(workspace,contract['benchmarks']['42'],entry,dataset,tmp_path/(method+'.json'))
        np.testing.assert_allclose(pred['probability'],comparison,atol=1e-7,rtol=1e-6)
        print('DEVELOPMENT_INTERFACE_PASS',method,flush=True)


def json_roundtrip(value):
    import json
    return json.loads(json.dumps(value))
