import copy
import json
import importlib.util
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.bota_short_benchmark import ali_stage17_common as c
from src.bota_short_benchmark import ali_stage17_transport as t
from src.bota_short_benchmark import ali_stage17_evaluation as ev
from src.bota_short_benchmark import ali_stage17_lambda64_suite_v1 as suite
from src.bota_short_benchmark import ali_stage17_storage as storage

REPO=Path(__file__).resolve().parents[1]
ENGINE={'schedule':{'batch_size':16},'optimizer':{'betas':[.9,.99],
    'learning_rate':.001,'eps':1e-6,'weight_decay':.01}}


def test_frozen_config_and_amendment():
    cfg=c.read(REPO/'configs/ali_stage17_lambda64_suite_v1.json')
    assert cfg['protocol']==c.PROTOCOL
    for p,h in cfg['bindings'].items():assert c.sha256_file(REPO/p)==h
    assert c.PROTOCOL['lambda']==64 and c.PROTOCOL['test_access_policy']=='forbidden'
    assert c.PROTOCOL['selection'] is False
    assert list(ev.s4.ATTACKS)==c.PROTOCOL['attack_scores']


@pytest.mark.parametrize('arm,source,full_v',[('FS0',2.,True),('FS1',5.,True),
    ('FS64',194.,True),('FM0',2.,False),('FM64',194.,False)])
def test_scale_only_fisher_feedback(monkeypatch,arm,source,full_v):
    p=torch.tensor([1.],requires_grad=True);p.grad=torch.ones_like(p)
    opt=torch.optim.AdamW([p]);seen=[]
    def tangent(*args,**kwargs):
        seen.append((args[7].item(),kwargs['full_v']));return args[4:7]
    monkeypatch.setattr(t.s8.p2b,'stable_adamw_tangent_step',tangent)
    monkeypatch.setattr(t.s8.p2b,'full_fisher_product',lambda rows,theta:torch.full_like(theta,3.))
    state=t.s8.stage1._state([p],1)
    t.advance(state,[torch.ones(1,1)],[torch.full((1,1),2.)],[p],opt,1,ENGINE,arm)
    assert seen==[(source,full_v)]


@pytest.mark.parametrize('arm,old',[('FS0','FS_source_only'),('FS1','FS_empirical_fisher'),('FM0','FM_source_only')])
def test_transport_matches_historical_engine(monkeypatch,tmp_path,arm,old):
    data=torch.tensor([[.4,1.],[1.3,.2]]*32)
    for module in (t.s8,t.s8.stage1):
        monkeypatch.setattr(module,'masked_batch',lambda dataset,chosen,pad:dataset[chosen])
        monkeypatch.setattr(module,'move_batch',lambda batch,device:batch)
    monkeypatch.setattr(t.s8.p1,'_sample_losses',lambda model,batch:((batch@model).squeeze(-1)-.3).square())
    def args():
        p=torch.tensor([[.2],[.5]],requires_grad=True)
        return (p,data,list(range(64)),[1,2]*32,[1,2],[p],['p'],torch.device('cpu'),0,ENGINE)
    torch.manual_seed(42)
    original,state,_,opt_hash,rng_hash=t.s8.stage1.run_shared_canonical(*args(),t.s8.StepBudget(4),arms=[old])
    torch.manual_seed(42)
    canonical,vectors,failures,timing=t.run(*args(),[arm],tmp_path/'progress.json')
    assert torch.equal(original['p'],canonical['p']) and not failures
    assert torch.equal(state[old][0]['theta'].reshape(2,-1),vectors[arm])
    assert opt_hash==timing['optimizer_hash'] and rng_hash==timing['rng_hash']


def test_composition_is_union_not_repeat_request():
    b={'users':[10,20,30],'names':['p'],'canonical':{'p':torch.tensor([2.,3.])},
       'vectors':{'FS64':torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])}}
    assert torch.equal(t.compose(b,'FS64',[10,20])['p'],t.compose(b,'FS64',[20,10])['p'])
    with pytest.raises(ValueError):t.compose(b,'FS64',[10,10])
    with pytest.raises(ValueError):t.unflatten(torch.zeros(3),b['canonical'],b['names'])


def test_immutable_tasks_and_source_boundary(tmp_path):
    src=tmp_path/'input';src.write_text('a')
    contract={'hashes':{str(src):c.sha256_file(src)}}
    def work(dest):return {'status':'COMPLETED','answer':3}
    out=c.task(tmp_path,contract,'tiny',work)
    assert out['answer']==3 and c.task(tmp_path,contract,'tiny',lambda _:pytest.fail('reran'))['answer']==3
    with pytest.raises(ValueError):c.task(tmp_path,{'hashes':{},'changed':True},'tiny',work)
    src.write_text('b')
    with pytest.raises(ValueError):c.check_sources(contract)
    (c.path(tmp_path,'tiny')/'extra').write_text('tamper')
    with pytest.raises(ValueError):c.verify(c.path(tmp_path,'tiny'))


def test_failure_keeps_evidence(tmp_path):
    def fail(dest):raise RuntimeError('synthetic infrastructure failure')
    with pytest.raises(RuntimeError):c.task(tmp_path,{'hashes':{}},'fail',fail)
    assert not c.path(tmp_path,'fail').exists()
    assert list(c.root(tmp_path).rglob('failure.json'))


def predictions():
    y=[0,1]*16;p=np.linspace(.15,.75,32)
    offsets={'Original':0.,'Exact-Masked':.02,'BOTA64':.015,'FullControl-P5':.01,'Retain-Retrain-P5':.03}
    return {k:{'probability':(p+v).tolist(),'gold_label':y,'answer_loss':[1.]*32,
        'sample_order_sha256':c.canonical_hash(list(range(32)))} for k,v in offsets.items()}


def test_behavior_and_p5_are_development_and_correct():
    pred=predictions();pred['FM64']=None
    out=ev.behavior(pred,[1]*16+[2]*16,{1})
    row=next(r for r in out['rows'] if r['method']=='BOTA64')
    assert row['E_U']==pytest.approx(.005) and row['residual']==pytest.approx(.25)
    assert row['P5']['test_accessed'] is False
    assert row['P5']['retain_retrain_absolute_error']==pytest.approx(.015)
    assert next(r for r in out['rows'] if r['method']=='FM64')['status']=='NONFINITE_PREDICTIONS'
    pred['Original']=None
    with pytest.raises(RuntimeError):ev.behavior(pred,[1]*16+[2]*16,{1})


def test_attack_scores_and_reference_gate():
    pred=predictions()
    out=ev.attacks({k:pred[k] for k in ('Original','Exact-Masked','BOTA64')},[1]*8+[2]*8,[1]*8+[2]*8,True,True)
    assert len(out['rows'])==12 and out['privacy_guarantee'] is False
    assert out['formal_pass'] is False  # zero reference signal, not evidence of forgetting
    assert ev.attacks({},[],[],True,True)['status']=='NO_MATCHED_USER_SUPPORT'


def test_expected_matrix_keeps_all_requests():
    contexts={str(seed):{'sequential':[{'union_hash':str(i)} for i in range(13)]} for seed in c.SEEDS}
    contexts['42']['stronger']=[{'id':str(i)} for i in range(9)]
    keys=suite.expected({'contexts':contexts})
    assert len(keys)==len(set(keys))==142


def test_all_storage_formats_and_spectrum():
    torch.manual_seed(3);x=torch.randn(8,4096)*.01
    gram=storage.svd.accumulate_gram(x,32);spec,u=storage.svd.spectrum(gram)
    assert spec['numerical_rank']==8
    low={'u':u.float(),'basis':u.float().T@x}
    _,ix=x.abs().topk(256,dim=1)
    top={'indices':ix.to(torch.int32),'values':x.gather(1,ix).half()}
    for rep in c.s6._representation_names():
        encoded=c.s6._encode(rep,x,low,top);delta=c.s6._compose(encoded,[1,3])
        assert delta.shape==(4096,) and torch.isfinite(delta).all()
    d,normal,_,pairs=storage.svd.direction_diagnostics(gram)
    assert len(pairs)==28 and normal.shape==(8,8)


def test_ensemble_parameter_comparison_excluded(tmp_path):
    entries=[]
    for method,value in [('Original',0.),('Exact-Masked',1.),('BOTA64',.75)]:
        d=tmp_path/method;d.mkdir()
        c.atomic_torch_save(d/'adapter_model.pt',{'A':{'p':torch.ones(2)},'B':{'p':torch.full((2,),value)}})
        entries.append(c.endpoint(tmp_path,d,method,42,'L8',[1]))
    entries.append({'method':'RecEraser','kind':'receraser_ensemble'})
    rows=ev.parameter_metrics(tmp_path,entries)
    assert rows[2]['relative_l2_error']==pytest.approx(.25)
    assert rows[-1]['status']=='NOT_A_SHARED_PARAMETER_COORDINATE'


def legacy_worker():
    spec=importlib.util.spec_from_file_location('stage17_legacy_worker',REPO/'scripts/bota_if/ali_stage17_legacy_worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fake_legacy(monkeypatch):
    monkeypatch.setattr(t.s8.p1,'_sample_losses',lambda model,batch:((batch@model).squeeze(-1)-.3).square())
    return SimpleNamespace(p1=t.s8.p1,p2b=t.s8.p2b,p3c3=SimpleNamespace(_cpu_tree=copy.deepcopy),
        _new_t2=t.s8.stage1._state,_cpu_parameters=lambda ns,ps:{n:p.detach().clone() for n,p in zip(ns,ps)},
        capture_rng=t.s8.capture_rng,tensor_tree_hash=t.s8.tensor_tree_hash,
        masked_batch=lambda data,chosen,pad:data[chosen],move_batch=lambda batch,device:batch,
        atomic_json=c.atomic_json)


def test_legacy_window_reset_and_continuous_carry(monkeypatch,tmp_path):
    worker=legacy_worker();r=fake_legacy(monkeypatch);data=torch.tensor([[.4,1.],[1.3,.2]]*32)
    def run(windows):
        p=torch.tensor([[.2],[.5]],requires_grad=True)
        return worker.canonical_windows(torch,r,p,data,list(range(64)),[1,2]*32,[1],[p],['p'],
            torch.device('cpu'),0,ENGINE,windows,t.s8.StepBudget(4),tmp_path,64.)
    a,continuous,_=run([(0,4)]);b,reset,_=run([(0,2),(2,4)])
    assert torch.equal(continuous[4]['original']['p'],reset[4]['original']['p'])
    assert not torch.equal(continuous[4]['delta']['p'],reset[4]['delta']['p'])
    assert set(b)=={0,2}


def test_legacy_numerical_failure_does_not_poison_next_window(monkeypatch,tmp_path):
    worker=legacy_worker();r=fake_legacy(monkeypatch);original=t.s8.p2b.full_fisher_product;calls=[]
    def fail_once(*args):
        calls.append(1)
        if len(calls)==1:raise RuntimeError('nonfinite_stable_adamw_tangent')
        return original(*args)
    monkeypatch.setattr(t.s8.p2b,'full_fisher_product',fail_once)
    p=torch.tensor([[.2],[.5]],requires_grad=True);data=torch.tensor([[.4,1.],[1.3,.2]]*32)
    _,e,_=worker.canonical_windows(torch,r,p,data,list(range(64)),[1,2]*32,[1],[p],['p'],
        torch.device('cpu'),0,ENGINE,[(0,2),(2,4)],t.s8.StepBudget(4),tmp_path,64.)
    assert e[2]['failure'] and e[2]['delta'] is None
    assert e[4]['failure'] is None and e[4]['delta'] is not None


def test_storage_screen_and_serving_artifact_interfaces(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'gpu',lambda:torch.device('cpu'))
    # pytest's Windows temp prefix is much longer than the real C:/python workspace.
    monkeypatch.setattr(c,'ROOT','s17')
    protocol=copy.deepcopy(c.PROTOCOL);protocol['latency'].update(warmup=1,repeats=2)
    monkeypatch.setattr(c,'PROTOCOL',protocol)
    torch.manual_seed(3);matrix=torch.randn(256,4096)*.001
    bank={'users':list(range(256)),'names':['p'],'canonical':{'p':torch.zeros(4096)},
          'bases':{'p':torch.ones(2)},'vectors':{'FS64':matrix},'failures':{}}
    ctx={'endpoints':[],'sequence_cells':[]}
    for sid,request in [('L8',[0,1]),('L4M4',[1,2])]:
        for method,multiplier in [('Exact-Masked',1.1),('BOTA64',1.)]:
            d=tmp_path/'references'/sid/method;d.mkdir(parents=True)
            c.atomic_torch_save(d/'adapter_model.pt',{'A':bank['bases'],'B':{'p':matrix[request].sum(0)*multiplier}})
            ctx['endpoints'].append(c.endpoint(tmp_path,d,method,42,sid,request))
        ctx['sequence_cells'].append({'scenario':sid,'request_size':8,'union_hash':sid})
    contract={'hashes':{},'contexts':{'42':ctx}}
    def save(dest):
        c.atomic_torch_save(dest/'bank.pt',bank)
        return {'status':'COMPLETED','chunk_transport_seconds_sum':10.}
    c.task(tmp_path,contract,'storage_bank_s42',save)
    c.task(tmp_path,contract,'bank_timing_s42',lambda _: {'status':'COMPLETED','timing':{'transport_seconds':2.}})
    result=storage.screen(tmp_path,contract,42)
    assert len(result['rows'])==120
    assert (c.path(tmp_path,'storage_screen_s42')/'spectrum.json').is_file()
    assert len(list((c.path(tmp_path,'storage_screen_s42')/'adapters').rglob('adapter_model.pt')))==24
    c.atomic_json(tmp_path/'outputs/ali_stage5_sequential_v1/runs/ali_stage5_sequential_seed42_v1/timing.json',
                  {'exact_replay_seconds':{'L8':1.,'L4M4':1.}})
    result=storage.time_serving(tmp_path,contract,42)
    assert len(result['summary'])==4
    assert all(row['iterations']==2 and row['amortization_break_even_requests']>0 for row in result['summary'])
