import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from src.bota_short_benchmark import ali_stage26_stress_mask_rescore_v1 as s
from src.bota_short_benchmark import ali_stage24_masked_scoring as masked


def test_atomic_resume_and_tamper(tmp_path):
    value=s.task(tmp_path,'unit','frozen',lambda p:{'status':'OK'})
    assert value['optimizer_updates']==0 and not value['selection_performed'] and not value['test_accessed']
    assert s.task(tmp_path,'unit','frozen',lambda p:pytest.fail('must resume'))==value
    with pytest.raises(ValueError,match='contract'):s.verify(tmp_path/'unit','changed')
    (tmp_path/'unit/result.json').write_text('{}')
    with pytest.raises(ValueError,match='changed artifact'):s.verify(tmp_path/'unit')


def test_interrupt_does_not_publish(tmp_path):
    def fail(p):raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):s.task(tmp_path,'unit','frozen',fail)
    assert not (tmp_path/'unit').exists()
    assert list((tmp_path/'.work').glob('*/FAILED.json'))


def test_missing_endpoint_reported_not_trained(tmp_path):
    auditor=s.Auditor()
    e=s.entry_status(tmp_path,{'path':'missing','kind':'fixed_ab','method':'BOTA64'},auditor)
    assert e['status']=='WEIGHTS_MISSING'
    assert len(e['missing'])==2
    assert not (tmp_path/'missing').exists()


def test_model_hash_change_rejected(tmp_path):
    p=tmp_path/'model';p.mkdir();(p/'adapter_model.pt').write_text('frozen');(p/'adapter_config.json').write_text('{}')
    e={'path':'model','kind':'fixed_ab','method':'BOTA64','directory_sha256':'wrong'}
    with pytest.raises(ValueError,match='differs'):s.entry_status(tmp_path,e,s.Auditor())


def test_historical_numerical_failure_not_reclassified_as_missing(tmp_path):
    e={'status':'HISTORICAL_NUMERICAL_FAILURE','method':'BOTA64','failure':{'reason':'nonfinite'}}
    assert s.entry_status(tmp_path,e,s.Auditor())==e


def test_bundle_missing_vs_corrupted(tmp_path):
    (tmp_path/'one').write_text('a')
    h=s.sha256_file(tmp_path/'one')
    s.atomic_json(tmp_path/'manifest.json',{'files':{'one':h,'absent':'x'}})
    assert s.Auditor().bundle(tmp_path)['status']=='ARTIFACT_MISSING'
    (tmp_path/'one').write_text('b')
    with pytest.raises(ValueError,match='hash mismatch'):s.Auditor().bundle(tmp_path)


def fixture():
    pred=lambda a:{'probability':a,'gold_label':[0,1,0,1],'answer_loss':[1]*4,
                   'sample_order_sha256':s.canonical_hash([0,1,2,3])}
    results={m:{'status':'OK','prediction':pred(p)} for m,p in
             [('Original',[.2,.8,.1,.9]),('Exact-Masked',[.3,.7,.2,.8]),('BOTA64',[.28,.72,.19,.81])]}
    results['IFRU']={'status':'WEIGHTS_MISSING','prediction':None}
    spec={'id':'sequential_s43_x','family':'sequential','seed':43,'request':'x','users':[1],
          'formal_attack':False,'jobs':{'binary':{m:m for m in results}}}
    plan={'cohorts':{'binary':{'users':[1,1,2,2]}}}
    return spec,plan,results


def test_missing_comparator_not_a_win_or_nonfinite():
    spec,plan,results=fixture();r=s.evaluate_spec(spec,plan,results)
    assert r['status']=='COMPLETED_WITH_UNAVAILABLE'
    b=next(x for x in r['behavior']['rows'] if x['method']=='BOTA64')
    assert b['E_U']==pytest.approx(.02) and b['residual']==pytest.approx(.2)
    missing=next(x for x in r['behavior']['rows'] if x['method']=='IFRU')
    assert missing=={'method':'IFRU','status':'WEIGHTS_MISSING'}
    assert r['seed']==43


def test_missing_reference_prevents_normalized_claim():
    spec,plan,results=fixture();results['Exact-Masked']={'status':'WEIGHTS_MISSING','prediction':None}
    assert s.evaluate_spec(spec,plan,results)['status']=='REFERENCE_UNAVAILABLE'


def test_attack_formal_permission_is_preserved(monkeypatch):
    spec,plan,results=fixture();spec['jobs']['attack']=spec['jobs']['binary'];spec['cohorts']={'attack':'attack_x'}
    plan['cohorts']['attack_x']={'members':[1],'nonmembers':[1]}
    def attack(pred,m,n,formal,utility):return {'formal_interpretation_permitted':formal,'formal_pass':formal and utility}
    monkeypatch.setattr(s.metrics,'attacks',attack)
    assert not s.evaluate_spec(spec,plan,results)['attack']['formal_interpretation_permitted']
    spec['formal_attack']=True
    assert s.evaluate_spec(spec,plan,results)['attack']['formal_interpretation_permitted']


def test_no_training_calls_in_new_driver():
    tree=ast.parse(Path(s.__file__).read_text(encoding='utf-8'))
    forbidden={'step','backward','build_bank','materialize','run_phase','run_local_masked','execute_method','screen','time_serving'}
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&forbidden


def test_finaltest_rejected_before_loading():
    with pytest.raises(PermissionError,match='forbids'):s.dataset(Path('.'),{}, {'scope':'FinalTest'})


def test_ranking_formula_remains_stage17_float32():
    raw={'yes':[.123456789],'no':[.987654321],'targets':[[.2,.3]],'gold_label':[1],
         'probability':[.111111104],'answer_loss':[1.4067054],'sample_order_sha256':'x'}
    cohort=s.scoring_cohort({'scope':'Development','lane':'ranking'})
    p=s.scoring.cohort_combine([raw],[1.],cohort)
    assert p['probability']==raw['probability']
    assert p['answer_loss']==raw['answer_loss']


def test_cache_does_not_change_sample_ids():
    class D:
        tokenizer=None
        def __init__(self):self.calls=0
        def __len__(self):return 3
        def __getitem__(self,i):self.calls+=1;return {'sample_id':i}
        def collate_fn(self,r):return r
    d=D();cache=s.CachedDataset(d)
    assert cache[2]==cache[2]=={'sample_id':2} and d.calls==1


def test_tiny_t5_masked_batch_invariance():
    from transformers import T5Config,T5ForConditionalGeneration
    class D:
        def __getitem__(self,i):return {'input_ids':torch.full((i+1,),2,dtype=torch.long),
                                      'target_ids':torch.tensor([2163 if i%2 else 465,1])}
        def collate_fn(self,rows):
            x=torch.zeros(len(rows),max(len(r['input_ids']) for r in rows),dtype=torch.long)
            for i,r in enumerate(rows):x[i,:len(r['input_ids'])]=r['input_ids']
            return {'input_ids':x,'target_ids':torch.stack([r['target_ids'] for r in rows])}
    torch.manual_seed(7)
    model=T5ForConditionalGeneration(T5Config(vocab_size=2200,d_model=16,d_kv=4,d_ff=32,num_layers=1,
          num_decoder_layers=1,num_heads=4,dropout_rate=0.,pad_token_id=0,decoder_start_token_id=0,eos_token_id=1)).eval()
    ids=list(range(8));data=D();cfg=s.read(Path(s.__file__).resolve().parents[2]/s.CONFIG)['audit']
    a=masked.predict(model,data,ids,'cpu',1,0);b=masked.predict(model,data,ids[::-1],'cpu',4,0)
    assert masked.difference(a,b,ids,ids[::-1],cfg)['passed']
