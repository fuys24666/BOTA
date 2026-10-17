from pathlib import Path
import copy
import numpy as np
import pytest
import torch

from src.bota_short_benchmark import ali_stage20_amazon_baselines_v1 as s


@pytest.fixture
def config():
    return s.load_config(Path(__file__).resolve().parents[2])[0]


def metadata():
    train=[u for u in range(90) for _ in range(3)]
    dev=[u for u in range(90) for _ in range(5)]
    order=[3*u+j for u in range(90) for j in (0,1)]
    return train,dev,order


def test_new_requests_are_metadata_only_disjoint(config):
    train,dev,order=metadata();excluded=set(range(10))
    r=s.requests_from_metadata(train,dev,order,excluded,config['protocol']['pairing_hash_salt'])
    assert r==s.requests_from_metadata(train,dev,order,excluded,config['protocol']['pairing_hash_salt'])
    users=[u for row in r['requests'] for u in row['users']]
    assert len(set(users))==6 and not set(users)&excluded
    assert [row['id'] for row in r['requests']]==['A1','A2','A3']
    assert all(row['deleted_interactions']==4 and len(row['local_indices'])==10 for row in r['requests'])
    assert r['selection_uses_labels_or_predictions'] is False
    with pytest.raises(ValueError,match='insufficient'):
        s.requests_from_metadata(train,dev,order,set(range(89)),'salt')


def test_fit_excludes_all_requested_users(config):
    _,dev,_=metadata();requests=[{'users':[0,1]},{'users':[31,32]}]
    part=s.development_partition(dev,requests,config['protocol']['development_fit_salt'])
    assert not set(part['fit'])&set(part['select'])
    assert set(part['fit'])|set(part['select'])==set(range(len(dev)))
    assert all(dev[i] not in {0,1,31,32} for i in part['fit'])
    assert all(i in part['select'] for i,u in enumerate(dev) if u in {0,1,31,32})


def calibration_rows(config,method):
    return [{'candidate_id':cid,'request':req,'status':'OK','E_U':.1+j*.01,
        'utility_eligible':True,'prediction_collapse':False,'compute_count':100,
        'hyperparameters':{'checkpoint':25,'forget_weight':.05}}
        for j,cid in enumerate(s.candidates(config,method)) for req in ['K2','K4']]


def test_selection_uses_both_requests_and_keeps_failures(config):
    rows=calibration_rows(config,'NegGrad');rows[0]['utility_eligible']=False
    result=s.select(rows,config,'NegGrad')
    assert result['selected']['candidate_id']==rows[2]['candidate_id']
    assert len(result['candidates'])==20 and result['candidates'][0]['failures']==['K2']
    assert result['comparison_used_for_selection'] is False
    for r in rows:r['utility_eligible']=False
    result=s.select(rows,config,'NegGrad')
    assert result['status']=='NO_ELIGIBLE_CANDIDATE' and result['selected'] is None
    assert result['diagnostic']['candidate_id']==rows[0]['candidate_id']
    for r in rows:r.update(status='NONFINITE_PREDICTIONS',E_U=None)
    assert s.select(rows,config,'NegGrad')['diagnostic'] is None
    with pytest.raises(ValueError,match='incomplete or duplicate'):
        s.select(rows[:-1],config,'NegGrad')
    with pytest.raises(ValueError,match='incomplete or duplicate'):
        s.select(calibration_rows(config,'NegGrad')+[rows[0]],config,'NegGrad')


def test_config_counts_and_chosen_hyperparameters(config):
    assert sum(len(s.candidates(config,m)) for m in s.METHODS)==65
    assert config['protocol']['seed']==42 and config['protocol']['primary_lambda']==48
    dec={'selected':{'hyperparameters':{'checkpoint':50,'learning_rate':.002}},'diagnostic':None}
    assert s.chosen_spec('PCGrad',dec)=={'outer':.002,'checkpoints':[50]}
    dec={'selected':None,'diagnostic':{'hyperparameters':{'relative_damping_ratio':.1,'curvature_panel_samples':512}}}
    assert s.chosen_spec('IFRU',dec)=={'panel':512,'ratios':[.1]}


def test_metrics_use_common_sample_projection(config):
    ix=[10,20,30,40];request={'id':'A1','local_indices':[20,40]}
    def prediction(p):
        return {'probability':p,'gold_label':[0,1,0,1],'sample_order_sha256':s.canonical_hash(ix)}
    o=prediction([.1,.8,.2,.9]);e=prediction([.1,.6,.2,.7]);p=prediction([.1,.65,.2,.75])
    r=s.metric('BOTA','BOTA',request,p,o,e,ix,config)
    assert r['E_U']==pytest.approx(.05) and r['D_U']==pytest.approx(.2)
    assert r['local_residual']==pytest.approx(.25) and r['toward_exact_rate']==1.
    assert r['local_samples']==2 and r['utility_samples']==4
    with pytest.raises(ValueError,match='omitted'):
        s.metric('BOTA','BOTA',{'id':'A1','local_indices':[99]},p,o,e,ix,config)
    p['probability'][0]=float('nan')
    assert s.metric('BOTA','BOTA',request,p,o,e,ix,config)['status']=='NONFINITE_PREDICTIONS'


def test_freeze_all_selections_blocks_partial_or_changed_contract(tmp_path):
    protocol={'seed':42}
    with pytest.raises(ValueError,match='missing IFRU'):
        s.freeze_all_selections(tmp_path,protocol)
    for m in s.METHODS:
        s.freeze(tmp_path,f'selections/{m}',{'contract':s.canonical_hash(protocol),
            'selection':{'method':m,'selected':None,'diagnostic':None}})
    assert set(s.freeze_all_selections(tmp_path,protocol))==set(s.METHODS)
    assert s.freeze_all_selections(tmp_path,protocol)==s.freeze_all_selections(tmp_path,protocol)
    with pytest.raises(ValueError,match='contract mismatch'):
        s.freeze_all_selections(tmp_path,{'seed':41})
    s.atomic_json(tmp_path/'all_selections'/'unregistered.json',{})
    with pytest.raises(ValueError,match='invalid'):
        s.verify(tmp_path/'all_selections')


def test_unit_resume_and_cleanup_scope(tmp_path):
    identity={'phase':'calibration','method':'IFRU','request':'K2'}
    dest,component=s.unit_paths(tmp_path,identity)
    assert s.unit_result(dest,identity) is None
    stage=s.stage_dir(tmp_path);s.write_unit(stage,dest,identity,[{'status':'OK'}])
    assert s.unit_result(dest,identity)==[{'status':'OK'}]
    component.mkdir(parents=True);(component/'generated.pt').write_bytes(b'local')
    s.remove_generated_component(tmp_path,component)
    assert not component.exists()
    with pytest.raises(ValueError,match='escapes'):
        s.remove_generated_component(tmp_path,tmp_path/'protocol')


def test_summaries_never_silently_omit_failed_cells():
    rows=[{'method':m,'request':r,'status':'OK','E_U':.1,'D_U':.2,
        'local_residual':.5,'toward_exact_rate':1.,'overall_auc':.8,'overall_log_loss':.4,
        'utility_eligible':True} for m in s.ORDER for r in s.COMPARISON]
    summary=s.summarize_rows(rows)
    assert summary['methods']['BOTA']['stats']['E_U']['mean']==pytest.approx(.1)
    assert summary['single_seed_exploratory']
    failed=next(r for r in rows if r['method']=='SISA');failed.update(status='NONFINITE_PREDICTIONS',E_U=None,utility_eligible=False)
    summary=s.summarize_rows(rows)
    assert summary['methods']['SISA']['stats']['E_U'] is None
    assert summary['methods']['SISA']['valid_cells']==2
    assert summary['BOTA_request_wins']['SISA']['invalid_comparator_cells']==1
    with pytest.raises(ValueError,match='incomplete or duplicate'):
        s.summarize_rows(rows[:-1])


def test_partition_rng_resume_reproduces_next_model(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(s.partitioned,'_new_local_model',lambda *a,**k:torch.rand(4))
    torch.manual_seed(42)
    with s.partition_resume_rng(run):
        first=s.partitioned._new_local_model()
        s.partitioned.atomic_json(run/'run_state.json',{'next_shard':1,'status':'RUNNING'})
        expected=s.partitioned._new_local_model()
    torch.manual_seed(999)
    with s.partition_resume_rng(run):
        resumed=s.partitioned._new_local_model()
    assert torch.equal(resumed,expected) and not torch.equal(first,expected)
    (run/'rng_1.pt').unlink()
    with pytest.raises(ValueError,match='missing RNG'):
        with s.partition_resume_rng(run):pass


def test_bounded_partition_config_uses_frozen_identity(tmp_path,monkeypatch,config):
    root=tmp_path/'outputs'/'stage20';root.mkdir(parents=True)
    old=tmp_path/'legacy'/'protocols'/'name'/'data'/'K2';old.mkdir(parents=True)
    train=[{'input':f't{i}','output':'Yes.'} for i in range(32)]
    dev=[{'input':f'd{i}','output':'No.'} for i in range(20)]
    s.atomic_json(tmp_path/'train.json',train);s.atomic_json(tmp_path/'dev.json',dev)
    base={'protocol':{'shards':4,'slices':4,'partition_seed':42,'sharded':True,'isolated':True,
        'sliced':True,'partition_unit':'interaction','aggregation':'mean_yes_no_probability',
        'effective_batch':16,'physical_microbatch':4},
        'sources':{'train':{'path':'train.json'},'forget':{},'development':{'path':'dev.json','sha256':s.sha256_file(tmp_path/'dev.json')}},
        'data':{'train_samples':32},'budget':{'source_paper':'test','official_repository':'test',
            'official_checkpoint_used_for_reported_result':'final','estimated_runtime':'test'}}
    s.atomic_text(old/'sisa.yaml',s.yaml.safe_dump(base))
    p={'benchmark':{'output_root':'legacy','source':{'development_json':'dev.json'}},
        'config':{'benchmark_name':'name'},'development_partition':{'fit':[0,1,2,3]}}
    req={'id':'A1','forget_window_positions':[1,8,16,24]}
    result=s.partition_config(root,tmp_path,p,req,'SISA',{'learning_rate':.001,'epochs_per_slice':3},root/'components'/'test')
    s.partitioned.partitioned_budget(result,'sisa')
    assert result['data']['forget_samples']==4
    assert s.read(tmp_path/result['sources']['development']['path'])==dev[:4]
    assert result['development_subset_authority']['full_path']=='dev.json'


def test_legacy_runtime_context_restores_cwd_even_on_failure(tmp_path,monkeypatch):
    before=Path.cwd()
    def fake_context(*args):
        assert Path.cwd()==tmp_path
        raise RuntimeError('simulated model load failure')
    monkeypatch.setattr(s.runner,'_model_context',fake_context)
    with pytest.raises(RuntimeError,match='simulated'):
        s.runtime_context(tmp_path,{}, {},None,None)
    assert Path.cwd()==before


def test_incomplete_analysis_never_publishes_success(tmp_path):
    r=s.analyze(tmp_path,{'seed':42})
    assert r['status']=='INCOMPLETE' and 'comparison/references' in r['missing']
    assert not (tmp_path/'summary').exists()
