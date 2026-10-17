"""Opt-in two-step GPU/Development interface smoke, not a formal experiment."""
import gc
import os
import subprocess
import sys
from pathlib import Path
import pytest
import torch
from src.bota_short_benchmark import ali_stage17_common as c
from src.bota_short_benchmark import ali_stage17_transport as t
from src.bota_short_benchmark import ali_stage17_evaluation as ev
from src.bota_short_benchmark import ali_stage17_lambda64_suite_v1 as suite


@pytest.mark.skipif(os.environ.get('BOTA_STAGE17_GPU_SMOKE')!='1',reason='bounded Development GPU smoke opt-in')
def test_two_step_transport_and_prediction_interfaces(tmp_path,monkeypatch):
    repo=Path(__file__).resolve().parents[1];w=Path('C:/python/E2URec-main')
    def forbidden(*a,**k):pytest.fail('FinalTest loader called')
    monkeypatch.setattr(c.s16.s10.lineage,'_final_lineage',forbidden)
    monkeypatch.setattr(c.s16.s10.lineage,'_validate_final_file',forbidden)
    contract=c.prepare(repo,w);ctx=contract['contexts']['42'];benchmark=ctx['benchmark'];registry=ctx['registry']
    assert len(suite.expected(contract))==142 and not contract['test_loader_built']
    monkeypatch.chdir(w);device=c.gpu()
    entry=next(e for e in ctx['endpoints'] if e['method']=='Original' and e['scenario']=='L8')
    anchor=c.torch_read(w/entry['path']/'adapter_model.pt');engine=t.s8.runner._engine(w,benchmark)
    def context():
        t.seed_everything(42)
        return t.s8.runner._model_context(w,benchmark,registry,device,anchor['A'])
    model,names,params,bases,_,tok,data,users=context()
    order=registry['order'][:32];selected=list(dict.fromkeys([users[order[0]],*ctx['primary_users']]))[:12]
    canonical,vectors,failures,timing=t.run(model,data,order,users,selected,params,names,device,tok.pad_token_id,
        engine,list(t.ARMS),tmp_path/'progress.json')
    assert not failures and timing['optimizer_steps']==2
    del model,params;gc.collect();torch.cuda.empty_cache()
    model,names,params,bases,_,tok,data,users=context()
    expected,states,_,opt_hash,rng_hash=t.s8.stage1.run_shared_canonical(model,data,order,users,selected,
        params,names,device,tok.pad_token_id,engine,t.s8.StepBudget(2))
    assert all(torch.equal(canonical[n],expected[n]) for n in names)
    for new,old in [('FS0','FS_source_only'),('FS1','FS_empirical_fisher'),('FM0','FM_source_only')]:
        old_vector=torch.cat([s['theta'].reshape(len(selected),-1) for s in states[old]],dim=1)
        assert torch.equal(vectors[new],old_vector)
    assert timing['optimizer_hash']==opt_hash and timing['rng_hash']==rng_hash
    del model,params,states;gc.collect();torch.cuda.empty_cache()
    datasets,devusers,plan,members,nonmembers=ev.datasets(w,contract,42,set(ctx['primary_users']))
    assert len(datasets['binary'])==20000 and set(members)==set(nonmembers)
    for lane in ('binary','ranking','attack'):
        pred=c.s16.s10._predict_endpoint(w,benchmark,entry,ev.Progress(datasets[lane],tmp_path/(lane+'.json')),
            device,list(range(4)))
        c.s16.validate_prediction(pred,4)
    state={'users':selected,'names':names,'canonical':canonical,'vectors':vectors,
           'bases':{n:b.detach().cpu() for n,b in bases.items()}}
    t.s8.runner._save_fixed_ab(tmp_path/'fm64',names,t.compose(state,'FM64',selected),state['bases'],'Stage17-smoke-FM64')
    print('STAGE17_GPU_DEVELOPMENT_SMOKE_PASS: two steps, five arms, three prediction lanes; no FinalTest',flush=True)


@pytest.mark.skipif(os.environ.get('BOTA_STAGE17_GPU_SMOKE')!='1',reason='bounded historical GPU smoke opt-in')
def test_historical_worker_interfaces(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(repo/'scripts/bota_if/ali_stage17_legacy_worker.py'),
        '--workspace','C:/python/E2URec-main','--mode','Smoke','--out',str(tmp_path)],check=True)
