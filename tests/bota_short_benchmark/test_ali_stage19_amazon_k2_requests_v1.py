from pathlib import Path
import copy
import pytest
from src.bota_short_benchmark import ali_stage19_amazon_k2_requests_v1 as s


def metadata():
    train=[u for u in range(90) for _ in range(3)]
    dev=[u for u in range(90) for _ in range(5)]
    order=[3*u+j for u in range(90) for j in (0,1)]
    return train,dev,order


def synthetic_rows(primary=.25,zero=.5,original=1.):
    return [{'seed':seed,'request':f'R{i:02d}','response_scale':scale,'status':'OK',
        'E_U':primary if scale==48 else zero if scale==0 else .01,'D_U':original,
        'local_residual':(primary if scale==48 else zero if scale==0 else .01)/original,
        'utility_eligible':True,'prediction_collapse':False}
        for seed in [41,42,43] for i in range(1,13) for scale in [0,48,64]]


def test_deterministic_disjoint_metadata_only_registry():
    t,d,o=metadata(); excluded={0,1,2,3}
    a=s.requests_from_metadata(t,d,o,excluded)
    b=s.requests_from_metadata(t,d,o,excluded)
    assert a==b
    us=[u for r in a['requests'] for u in r['users']]
    assert len(set(us))==24 and not set(us)&excluded
    assert all(r['deleted_interactions']==4 and r['local_samples']==10 for r in a['requests'])
    assert all([m['role'] for m in r['metadata']]==['low','middle'] for r in a['requests'])
    assert a['selection_uses_labels_or_predictions'] is False


def test_no_pool_relaxation():
    t,d,o=metadata()
    with pytest.raises(ValueError,match='insufficient'):
        s.requests_from_metadata(t,d,o,set(range(89)))
    with pytest.raises(ValueError,match='insufficient'):
        s.requests_from_metadata(t,d[::5],o,set())


def test_two_distinct_evidence_levels():
    both=s.summarize(synthetic_rows())
    assert both['classification']=='L2_MATCHED_COUNTERFACTUAL_SUPPORT'
    mechanism=s.summarize(synthetic_rows(primary=.6,zero=.8,original=.5))
    assert mechanism['classification']=='L1_FISHER_GAIN_ONLY'
    assert mechanism['mechanism_gate']['passed']
    assert not mechanism['reconstruction_gate']['passed']


def test_bad_seed_and_secondary_cannot_rescue():
    rows=synthetic_rows()
    for r in rows:
        if r['seed']==41 and r['response_scale']==48:
            r['E_U']=.7; r['local_residual']=.7
    result=s.summarize(rows)
    assert result['classification']=='NOT_SUPPORTED'
    assert not result['mechanism_gate']['per_seed']['41']['passed']
    assert result['distributions']['64']['all_cell_mean_E_U']==pytest.approx(.01)


def test_failure_or_utility_violation_is_never_filtered():
    rows=synthetic_rows(); target=next(r for r in rows if r['response_scale']==48)
    target['utility_eligible']=False
    assert s.summarize(rows)['classification']=='NOT_SUPPORTED'
    target.update(status='NUMERICAL_TRANSPORT_FAILURE',E_U=None,local_residual=None)
    result=s.summarize(rows)
    assert result['classification']=='NOT_SUPPORTED'
    assert result['distributions']['48']['all_cell_mean_E_U'] is None
    assert result['distributions']['48']['valid_cells']==35


def test_request_bootstrap_and_completeness():
    rows=synthetic_rows()
    a=s.summarize(rows); b=s.summarize(rows)
    assert a['mechanism_gate']==b['mechanism_gate']
    assert a['mechanism_gate']['paired_request_bootstrap_gain_95_interval']==[.25,.25]
    with pytest.raises(ValueError,match='incomplete'):
        s.summarize(rows[:-1])
    with pytest.raises(ValueError,match='duplicate'):
        s.summarize(rows+[rows[0]])


def test_sealed_report_repeat_and_tamper(tmp_path):
    rows=synthetic_rows()
    for seed in [41,42,43]:
        for i in range(1,13):
            rid=f'R{i:02d}'
            s.freeze(tmp_path,f'cells/s{seed}_{rid}',{'rows':[r for r in rows if r['seed']==seed and r['request']==rid]})
    result=s.final_report(tmp_path)
    assert result['status']=='COMPLETED'
    assert s.final_report(tmp_path)==result
    s.atomic_json(tmp_path/'summary/unregistered.json',{})
    with pytest.raises(ValueError,match='invalid'):
        s.verify(tmp_path/'summary')


def test_incomplete_report_never_published(tmp_path):
    assert s.final_report(tmp_path)['status']=='INCOMPLETE'
    assert not (tmp_path/'summary').exists()


def test_config_is_search_free_and_parent_unchanged():
    repo=Path(__file__).resolve().parents[2]
    c,_=s.load_config(repo)
    assert c['protocol']['scales']==[0,48,64]
    assert c['protocol']['no_reselection']
    assert c['protocol']['final_test'] is False
    assert c['output_root'] != 'outputs/ali_stage18_amazon_lambda_search_v2'


def test_seed_orchestration_and_cell_resume(tmp_path,monkeypatch):
    t,dev,order=metadata()
    registry=s.requests_from_metadata(t,dev,order,set())
    dev=dev+[999]*(3840-len(dev))
    b={'source':{'development_json':'unused'}}
    monkeypatch.setattr(s.s.runner,'_base_t5_config',lambda *a:{'paths':{'model_dir':'unused'}})
    monkeypatch.setattr(s.s.s8.T5Tokenizer,'from_pretrained',lambda *a:None)
    monkeypatch.setattr(s.s.s8,'JsonPromptDataset',lambda *a:list(range(3840)))
    calls=[]
    def fake_bank(root,workspace,b,reg,refs,seed,group,shard,contract):
        p=root/f'fake_bank_{seed}_{shard}'
        p.mkdir(exist_ok=True); s.atomic_json(p/'result.json',{'failures':{}})
        return p
    def fake_exact(*args):
        calls.append(args[6]['id'])
        return tmp_path/'exact'
    def fake_predict(root,workspace,b,adapter,key,dataset,indices,contract):
        probs=[.2,.8]*1920
        if key.endswith('Exact'): probs=[x+.01 for x in probs]
        return {'probability':probs,'gold_label':[0,1]*1920,'answer_loss':[.2]*3840,
                'sample_order_sha256':s.canonical_hash(indices)}
    monkeypatch.setattr(s,'build_bank',fake_bank); monkeypatch.setattr(s,'exact',fake_exact)
    monkeypatch.setattr(s,'predict',fake_predict)
    refs={41:tmp_path/'Original'}
    a=s.run_seed(tmp_path,tmp_path,b,{},dev,registry,refs,41,'frozen')
    assert len(a)==36 and len(calls)==12
    again=s.run_seed(tmp_path,tmp_path,b,{},dev,registry,refs,41,'frozen')
    assert a==again and len(calls)==12
