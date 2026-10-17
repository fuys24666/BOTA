"""Immutable, Development-only lambda64 refresh. Historical runs stay read-only."""
from __future__ import annotations
import json
import os
import time
import platform
import subprocess
import sys
from pathlib import Path
import torch
from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, directory_hash, sha256_file
from src.paper_if_a2.artifacts import atomic_torch_save
from . import ali_stage16_finaltest_v1 as s16
from . import ali_stage15_ranking_v1 as s15
from . import ali_stage5_sequential_v1 as s5
from . import ali_stage2_stronger_signal_v1 as s2
from . import ali_stage3_equal_horizon_v1 as s3
from . import ali_stage6_storage_screen_v1 as s6

SCHEMA='bota-ali-stage17-lambda64-suite-v1'
ROOT='outputs/ali_stage17_lambda64_suite_v1'
SEEDS=(41,42,43)
PHASES=('Primary','Stronger','Sequential','Horizon','Storage','StorageBehavior','Latency','LegacyRelay','LegacyPosition')
PROTOCOL={
 'dataset':'MovieLens-1M','lambda':64,'seeds':list(SEEDS),'test_access_policy':'forbidden',
 'selection':False,'primary_ablation':['FS0','FS1','FS64','FM0','FM64'],
 'primary_requests':['L8','L4M4'],'stronger_seed':42,'stronger_requests':'all_nine_frozen_Stage2',
 'stronger_attack_interpretation':['high_k4'],'sequential_sizes':[1,2,4,8],
 'sequential_orders':'same_20_Stage5_cells_per_seed;deduplicate_identical_unions',
 'horizon_steps':1200,'horizon_schedule':'same_Stage3_six_P5_epoch_orders',
 'bank_sizes':[12,32,64,128,256],'bank_chunk_size':32,'storage_representations':s6._representation_names(),
 'storage_gate':{'minimum_compression':16,'maximum_relative_parameter_residual_increase':.10},
 'storage_behavior':'all_12_representations_at_256;no_candidate_selection',
 'svd':'float64_Gram;raw_and_row_normalized;fixed_256_users',
 'behavior_split':'full_Development_20000','ranking':'frozen_Stage15_DevSelect_candidates',
 'ranking_is_separate_query_set':True,'bootstrap_resamples':2000,'bootstrap_seed':42,
 'attack_scores':['negative_answer_loss','confidence','negative_entropy','absolute_yes_no_margin'],
 'attack_signal_minimum':.02,'auc_damage_max':.005,'logloss_damage_max':.01,
 'numerical_failures':'retain_explicit_outcomes;never_clip_or_shrink_lambda;no_success_only_summary',
 'latency':{'bank_sizes':[12,256],'warmup':20,'repeats':1000,'semantics':'compose_materialize_fsync_publish_reload_verify',
            'excluded':'bank_build,cold_load,network,fallback,production_service'},
 'reuse':'unchanged_original_exact_and_calibrated_baselines_with_hashes;no_retraining_of_controls',
 'legacy_position_relay':{'seed':42,'isolated_historical_runtime':True,'lambda':64,
     'relay':'same_frozen_v1_L8_600_distinct_slots_carry_theta_m_v',
     'position':'corrected_P3C5F_three_strata_four_independent_200step_windows',
     'position_metrics':'parameter_fidelity_only_same_as_historical_protocol',
     'relay_metrics':'parameter_and_full_Development_behavior_same_as_historical_protocol'},
 'cost_analysis':'dedicated_FS64_12user_build;256user_8chunks;historical_Stage5_exact_timing_break_even_no_fallback',
 'other_dataset':'not_authorized_in_this_stage',
}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def root(w):return w/ROOT
def path(w,key):return root(w)/'cells'/key
def stage(w):return s15.stage_path(root(w))

def publish(work,dest,result,identity):
    if dest.exists():raise FileExistsError(dest)
    atomic_json(work/'result.json',{**result,'test_accessed':False})
    atomic_json(work/'identity.json',identity)
    atomic_text(work/'COMPLETED',SCHEMA+'\n')
    files={p.relative_to(work).as_posix():sha256_file(p) for p in sorted(work.rglob('*')) if p.is_file()}
    atomic_json(work/'manifest.json',{'schema':SCHEMA,'files':files,'test_accessed':False})
    dest.parent.mkdir(parents=True,exist_ok=True);os.rename(work,dest)

def verify(p):
    m=read(p/'manifest.json')
    if m['schema']!=SCHEMA or m['test_accessed'] is not False:raise ValueError('invalid suite artifact')
    if (p/'COMPLETED').read_text()!=SCHEMA+'\n':raise ValueError('incomplete suite artifact')
    actual={q.relative_to(p).as_posix() for q in p.rglob('*') if q.is_file()}-{'manifest.json'}
    if actual!=set(m['files']):raise ValueError('artifact inventory changed')
    for k,v in m['files'].items():
        if sha256_file(p/k)!=v:raise ValueError(f'artifact changed: {p/k}')
    return read(p/'result.json')

def task(w,contract,key,fn):
    identity={'contract_sha256':canonical_hash(contract),'key':key};dest=path(w,key)
    if dest.exists():
        out=verify(dest)
        if read(dest/'identity.json')!=identity:raise ValueError('different source/protocol; cannot reuse')
        print('[Stage17] reuse',key,flush=True);return out
    work=stage(w);begun=time.perf_counter();print('[Stage17] start',key,'work=',work,flush=True)
    try:
        result=fn(work)
        result.update(wall_seconds=time.perf_counter()-begun)
        check_sources(contract)
        publish(work,dest,result,identity);return result
    except Exception as e:
        atomic_json(work/'failure.json',{'key':key,'exception':type(e).__name__,'message':str(e),'test_accessed':False})
        raise

def check_sources(contract):
    if 'runtime' in contract and runtime()!=contract['runtime']:raise ValueError('frozen runtime changed')
    for p,h in contract['hashes'].items():
        if sha256_file(Path(p))!=h:raise ValueError(f'frozen input changed: {p}')


def runtime():
    import numpy as np
    import transformers
    return {'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,
            'numpy':np.__version__,'transformers':transformers.__version__,'torch_threads':torch.get_num_threads()}


def legacy_preflight(repo,w):
    process=subprocess.run([sys.executable,str(repo/'scripts/bota_if/ali_stage17_legacy_worker.py'),
        '--workspace',str(w),'--mode','Preflight'],capture_output=True,text=True,check=True)
    value=json.loads(process.stdout)
    if value['status']!='READY' or value['test_accessed'] is not False:raise ValueError('legacy authority not ready')
    return value

def endpoint(w,p,method,seed,sid,users):
    p=p.resolve()
    if not p.is_relative_to(w.resolve()) or any('test' in q.lower() for q in p.relative_to(w).parts):
        raise ValueError('endpoint outside Development boundary')
    return {'path':p.relative_to(w).as_posix(),'kind':'fixed_ab','directory_sha256':directory_hash(p),
            'method':method,'seed':seed,'scenario':sid,'users':users}

def prepare(repo,w):
    longest=root(w)/'.work/1234567890/adapters/shared_low_rank_r64/L4M4/adapter'/('.adapter_config.json.'+'0'*32+'.tmp')
    if os.name=='nt' and len(str(longest))>=250:raise ValueError('use a shorter workspace path for Windows atomic artifacts')
    config=read(repo/'configs/ali_stage17_lambda64_suite_v1.json')
    if config['protocol']!=PROTOCOL or config['schema']!=SCHEMA:raise ValueError('frozen Stage17 protocol changed')
    for p,h in config['bindings'].items():
        if sha256_file(repo/p)!=h:raise ValueError('amendment/Stage16 audit changed')
    # Stage16 input authority is Development-only; do not call its FinalTest loader/execute.
    _,base=s16.preflight(repo,w)
    audit=read(repo/'docs/audits/ali_stage16_result_audit_v1.json')
    for rel,h in audit['manifest_sha256'].items():
        if sha256_file(w/s16.ROOT/rel/'manifest.json')!=h:raise ValueError('sealed Stage16 result changed')
        s16.verify(w/s16.ROOT/rel)
    contexts={};used=set();ancestors=[]
    for seed in SEEDS:
        a,_,benchmark,registry,cells,people=s5.preflight(repo,w,repo/'configs/ali_stage5_sequential_v1.yaml',seed)
        _,_,_,longreg=s3.preflight(repo,w,repo/'configs/ali_stage3_equal_horizon_v1.yaml',seed)
        if longreg['order']!=registry['order']:raise ValueError('short/long authority differs')
        run5=w/'outputs/ali_stage5_sequential_v1/runs'/f'ali_stage5_sequential_seed{seed}_v1'
        run3=w/'outputs/ali_stage3_equal_horizon_v1/runs'/f'ali_stage3_equal_horizon_seed{seed}_v1'
        run6=w/'outputs/ali_stage6_storage_screen_v1/runs'/f'ali_stage6_storage_screen_seed{seed}_v1'
        train_users,_,_=s2._source_users(w,benchmark)
        bank_users,bank_hashes=s6._bank_users(registry,train_users)
        if set(bank_users[:12])!=set(people):raise ValueError('registered bank prefix changed')
        unique={c['union_hash']:c for c in cells};seq=[]
        for h,c in unique.items():
            p=run5/'endpoints'/s5._endpoint_key(h)
            seq.append({**c,'exact':endpoint(w,p/'exact_masked/adapter','Exact-Masked',seed,c['cell_id'],c['users']),
                        'old':endpoint(w,p/'bota/adapter','BOTA1',seed,c['cell_id'],c['users'])})
        long=[]
        for sid in s16.SCENARIOS:
            users=next(c['users'] for c in registry['scenarios'] if c['id']==sid)
            long.append({'id':sid,'users':users,
                'original':endpoint(w,run3/'adapters/original/adapter','Original',seed,sid,users),
                'exact':endpoint(w,run3/'adapters'/sid/'exact_masked/adapter','Exact-Masked',seed,sid,users),
                'old':endpoint(w,run3/'adapters'/sid/'bota/adapter','BOTA1',seed,sid,users)})
        contexts[str(seed)]={'benchmark':benchmark,'registry':registry,'primary_users':bank_users[:12],
            'bank_users':bank_users,'bank_hashes':bank_hashes,'sequential':seq,'sequence_cells':cells,
            'long':long,'endpoints':[e for e in base['endpoints'] if e['seed']==seed]}
        ancestors.extend([run5,run3,run6])
    _,_,_,_,requests,people=s2.preflight(repo,w,repo/'configs/ali_stage2_stronger_signal_v1.yaml')
    run2=w/'outputs/ali_stage2_stronger_signal_v1/runs/ali_stage2_stronger_signal_seed42_v1'
    for request in requests:
        rid=request['id'];request['exact']=endpoint(w,run2/'requests'/rid/'exact_masked/adapter','Exact-Masked',42,rid,request['users'])
        request['old']=endpoint(w,run2/'requests'/rid/'bota/adapter','BOTA1',42,rid,request['users'])
    contexts['42']['stronger']=requests;contexts['42']['stronger_users']=people;ancestors.append(run2)
    # Hash complete referenced historical artifacts, not merely metadata labels.
    for d in ancestors:
        for p in d.rglob('*'):
            if p.is_file() and '.work' not in p.parts:used.add(p)
    for e in base['endpoints']:
        used.update(p for p in (w/e['path']).rglob('*') if p.is_file())
    for d in base['dependencies'].values():
        p=w/d['path'];used.update([p] if p.is_file() else [q for q in p.rglob('*') if q.is_file()])
    used.update(Path(p) for p in base['input_sha256'])
    used.update(repo/p for p in s16.source_hashes(repo))
    used.add(repo/'scripts/bota_if/ali_stage17_legacy_worker.py')
    # Foreign historical helpers are isolated in a subprocess and source-frozen too.
    used.update((w/'src').rglob('*.py'))
    used.update(p for p in (w/'configs').rglob('*') if p.is_file())
    used.update((w/s15.ROOT/'protocol').iterdir())
    legacy=legacy_preflight(repo,w)
    hashes={str(p.resolve()):sha256_file(p) for p in sorted(used) if p.is_file()}
    return {'schema':SCHEMA,'protocol':PROTOCOL,'contexts':contexts,'hashes':hashes,'runtime':runtime(),
            'legacy':legacy,'repo':str(repo.resolve()),
            'dependencies':base['dependencies'],'ranking_plan_sha256':sha256_file(w/s15.ROOT/'protocol/candidates.json'),
            'historical_development_and_finaltest_exposure':True,'test_loader_built':False}

def seal(w,contract):
    p=root(w)/'protocol'
    if p.exists():
        verify(p)
        if read(p/'contract.json')!=contract:raise ValueError('sealed suite changed; no reseal/overwrite')
    else:
        work=stage(w);atomic_json(work/'contract.json',contract)
        publish(work,p,{'status':'SEALED'},{'contract_sha256':canonical_hash(contract)})
    return contract

def load(w):
    verify(root(w)/'protocol');c=read(root(w)/'protocol/contract.json');check_sources(c);return c

def torch_read(p):return torch.load(p,map_location='cpu',weights_only=True)
def gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('exactly one CUDA GPU required')
    free,total=torch.cuda.mem_get_info()
    if free/2**30<10:raise RuntimeError('10 GiB free GPU memory required')
    torch.cuda.set_per_process_memory_fraction(min(.88,14/(total/2**30)))
    return torch.device('cuda:0')
