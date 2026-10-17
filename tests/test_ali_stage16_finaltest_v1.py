import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.bota_short_benchmark import ali_stage16_finaltest_v1 as s


def row(user, item, label=1, index=0, history=(1, 2)):
    return SimpleNamespace(authoritative_user_id=user, movie_id=item, label=label,
        timestamp=100+index, raw_source_row=index, processed_row_index=index,
        history_ids=list(history), history_ratings=[4]*len(history))


def prediction(p, labels=None):
    return dict(probability=list(p), gold_label=labels or [0,1]*(len(p)//2),
                answer_loss=[1.]*len(p), sample_order_sha256=s.canonical_hash(list(range(len(p)))))


def bundle(root, target, files):
    stage=s.s15.stage_path(root)
    for name, value in files.items(): s.atomic_json(stage/name,value)
    s.publish(stage,target,True)


def test_config_frozen_no_selection():
    repo=Path(__file__).resolve().parents[1]
    config=s.load_config(repo)
    assert config['protocol']['response_scale']==64
    assert config['protocol']['hyperparameter_selection'] is False
    assert config['protocol']['historical_finaltest_exposure'] is True
    assert config['protocol']['ranking']['new_acceptance_threshold'] is None


def test_no_capability_or_confirmation_no_test_access(tmp_path,monkeypatch):
    def forbidden(*a,**k): raise AssertionError('test loader reached')
    monkeypatch.setattr(s.s10.lineage,'_final_lineage',forbidden)
    monkeypatch.setattr(s,'check_seal',forbidden)
    with pytest.raises(PermissionError):s.materialize(tmp_path,tmp_path,{},None)
    with pytest.raises(PermissionError):s.BinaryDataset(tmp_path/'missing',None,None)
    with pytest.raises(PermissionError):s.execute(tmp_path,tmp_path,{}, {},False,False)
    assert not (tmp_path/'access').exists()


def test_access_once_resume_same_seal_only(tmp_path):
    with pytest.raises(PermissionError):s.begin_access(tmp_path,'a',False,False)
    with pytest.raises(ValueError,match='nothing'):s.begin_access(tmp_path,'a',True,True)
    s.begin_access(tmp_path,'a',False,True)
    with pytest.raises(RuntimeError,match='already started'):s.begin_access(tmp_path,'a',False,True)
    with pytest.raises(ValueError,match='different seal'):s.begin_access(tmp_path,'b',True,True)
    s.begin_access(tmp_path,'a',True,True)


def test_seal_roundtrip_and_immutable(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'git_snapshot',lambda p:{'dirty':True})
    contract={'benchmarks':{'41':{'source':'x'}},'source_sha256':{},'protocol':s.PROTOCOL}
    value=s.seal(tmp_path,tmp_path,contract)
    assert s.seal(tmp_path,tmp_path,json.loads(json.dumps(contract)))==value
    altered=copy.deepcopy(contract);altered['protocol']['response_scale']=128
    with pytest.raises(ValueError,match='changed'):s.seal(tmp_path,tmp_path,altered)


def test_candidate_cutoff_uses_pretest_history_excludes_dev_and_all_test_observations():
    train=[row(99,i) for i in range(1,170)]+[row(10,3)]
    dev=[row(10,4),row(10,6,label=0)]
    final=[row(10,5,index=0,history=(1,2,3,4,6)),row(10,7,index=1,history=(1,2,4,6,5)),
           row(10,8,label=0,index=2),row(20,900,index=3)]
    plan=s.ranking_plan(train,dev,final)
    assert plan['test_accessed'] is True and plan['split']=='FinalTest'
    u=plan['users'][0]
    assert u['positive_items']==[5,7] and u['history_ids']==[1,2,3,4,6]
    assert len(u['items'])==102 and not set(u['items']) & {1,2,3,4,6,8}
    assert plan['coverage'][1]['status']=='NO_ELIGIBLE_POSITIVE'
    assert plan['coverage'][1]['positive_out_of_catalogue']==1
    assert 'development_rows' not in plan['coverage'][0]
    assert plan==s.ranking_plan(list(reversed(train)),dev,final)


@pytest.mark.parametrize('change', ['nan','length','order','label'])
def test_invalid_prediction_rejected(change):
    p=prediction([.2,.8])
    if change=='nan':p['probability'][0]=float('nan')
    if change=='length':p['answer_loss']=[]
    if change=='order':p['sample_order_sha256']='bad'
    if change=='label':p['gold_label']=[1,0]
    with pytest.raises(ValueError):s.validate_prediction(p,2,[0,1])


def test_binary_metrics_distinct_p5_target_and_pairing(monkeypatch):
    monkeypatch.setitem(s.s10.EXPECTED_EVALUATION,'bootstrap_resamples',20)
    p=np.linspace(.1,.8,32);pred={}
    offsets={'Original':0,'Exact-Masked':.01,'BOTA64':.009,'BOTA1':.008,'BOTA0':.007,
             'FullControl-P5':.02,'Retain-Retrain-P5':.03}
    for m in s.METHODS:pred[m]=prediction((p+offsets.get(m,.04)).tolist())
    rows=[]
    for seed in s.SEEDS:
        for sid in s.SCENARIOS:rows+=s.binary_rows_for(pred,[i//4 for i in range(32)],{0,1,99},seed,sid)
    b=next(r for r in rows if r['method']=='BOTA64')
    assert b['E_U']==pytest.approx(.001)
    assert b['local_exact_masked_residual']==pytest.approx(.1)
    assert b['retain_retrain_absolute_error']==pytest.approx(.021)
    assert b['p5_retrain_residual']==pytest.approx(2.1)
    assert b['missing_users']==[99]
    summary,decision,paired=s.aggregate_binary(rows)
    assert len(summary)==24 and len(paired)==18
    assert decision['retuning_authorized'] is False
    assert set(decision['mean_EU_below_baselines'])=={'IFRU','E2URec','NegGrad','PCGrad','RecEraser'}
    assert next(r for r in paired if r['control']=='BOTA1')['EU_relative_reduction']==pytest.approx(.5)
    with pytest.raises(ValueError):s.aggregate_binary(rows[:-1])


def test_ranking_macro_not_candidate_weighted_and_empty_group():
    plan={'users':[{'user':1,'items':[1,2],'positive_items':[1]}],'candidate_count':2}
    rows=[]
    for seed in s.SEEDS:
        for sid in s.SCENARIOS:
            for method in s.METHODS:
                r=s.s15.metrics_for(plan,[.9,.1],{99})
                rows.extend(dict(seed=seed,scenario=sid,method=method,**g) for g in r['aggregates'])
    summary=s.summarize_ranking(rows)
    assert len(summary)==72
    assert next(r for r in summary if r['group']=='requested')['ndcg@10'] is None
    assert next(r for r in summary if r['group']=='all')['ndcg@10']['mean']==1
    with pytest.raises(ValueError):s.summarize_ranking(rows[:-1])


def test_atomic_bundle_tamper(tmp_path):
    target=tmp_path/'done';bundle(tmp_path,target,{'a.json':{'a':1}})
    s.verify(target)
    (target/'a.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):s.verify(target)


def test_complete_analysis_without_raw_test_access(tmp_path,monkeypatch):
    # Exercise all 144 saved-cell paths and summary regeneration with tiny synthetic scores.
    monkeypatch.setitem(s.s10.EXPECTED_EVALUATION,'bootstrap_resamples',10)
    monkeypatch.setattr(s.s10.lineage,'_final_lineage',lambda *a:pytest.fail('raw FinalTest loader reached'))
    root=tmp_path/s.ROOT
    entries=[dict(seed=seed,scenario=sid,method=m,users=[0,1,99])
        for seed in s.SEEDS for sid in s.SCENARIOS for m in s.METHODS]
    contract={'endpoints':entries};plan={'users':[{'user':0,'items':[1,2],'positive_items':[1]}],'candidate_count':2}
    users=[i//4 for i in range(32)]
    cohort={'seal_sha256':s.canonical_hash(contract),'candidate_sha256':s.canonical_hash(plan),
            'rows':32,'users':users,'labels':[0,1]*16}
    assert s.analyze(tmp_path,{},contract)['status']=='NOT_STARTED'
    bundle(root,root/'cohort',{'cohort.json':cohort,'candidates.json':plan})
    p=np.linspace(.1,.8,32);preds={m:prediction((p+.001*s.METHODS.index(m)).tolist()) for m in s.METHODS}
    for e in entries:
        for lane in s.LANES:
            data=preds[e['method']] if lane=='binary' else prediction([.9,.1],[1,1])
            bundle(root,s.cell_path(root,e,lane),{'identity.json':s.identity_for(contract,cohort,e,lane),'predictions.json':data})
    # Frozen legacy comparison exercises the real reproduction function too.
    old=[]
    for seed in s.SEEDS:
        for sid in s.SCENARIOS:
            for r in s.binary_rows_for(preds,users,{0,1,99},seed,sid):
                if r['method'] not in ('BOTA64','BOTA0'):
                    old.append({**r,'method':'BOTA' if r['method']=='BOTA1' else r['method']})
    legacy=tmp_path/'legacy';legacy.mkdir()
    s.atomic_json(legacy/'metrics.json',{'rows':old})
    s.atomic_json(legacy/'manifest.json',{'files':{'metrics.json':s.sha256_file(legacy/'metrics.json')}})
    config={'legacy_result':{'path':'legacy','manifest_sha256':s.sha256_file(legacy/'manifest.json')}}
    assert s.analyze(tmp_path,config,contract)['status']=='COMPLETED'
    assert s.analyze(tmp_path,config,contract)['legacy_reproduction_pass'] is True
    summary=s.read(root/'summary/summary.json')
    assert len(summary['binary_rows'])==72 and len(summary['ranking_rows'])==216
    assert len(summary['legacy_reproduction']['cells'])==60


def test_partial_does_not_publish_combined_summary(tmp_path):
    root=tmp_path/s.ROOT;plan={'users':[]};contract={'endpoints':[dict(seed=41,scenario='L8',method='BOTA64')]}
    cohort={'seal_sha256':s.canonical_hash(contract),'candidate_sha256':s.canonical_hash(plan)}
    bundle(root,root/'cohort',{'cohort.json':cohort,'candidates.json':plan})
    result=s.analyze(tmp_path,{},contract)
    assert result['status']=='PARTIAL' and result['missing_cells']==2
    assert not (root/'summary').exists()


def test_full_cli_refuses_before_preflight(monkeypatch,tmp_path):
    import sys
    monkeypatch.setattr(sys,'argv',['stage16','--workspace',str(tmp_path),'--mode','Full'])
    monkeypatch.setattr(s,'preflight',lambda *a:pytest.fail('preflight should not run'))
    with pytest.raises(PermissionError):s.main()
