"""Frozen new-request K2 validation, not a parameter search or fresh holdout."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import ali_stage18_amazon_lambda_search_v2 as s

SCHEMA = "bota-ali-stage19-amazon-k2-requests-v1"
RULES = {"seeds": [41,42,43], "requests":12, "users_per_request":2,
    "composition":{"low":1,"middle":1}, "window_exposure_per_user":2,
    "minimum_development_per_user":5, "pairing_hash_salt":"bota-stage19-k2-frozen-v1",
    "optimizer_steps":200, "batch_size":16, "scales":[0,48,64], "primary_scale":48,
    "secondary_scale":64, "users_per_bank_shard":8, "bootstrap_resamples":5000,
    "bootstrap_seed":190042, "minimum_absolute_gain":1e-6, "minimum_relative_gain":.01,
    "minimum_winning_requests":8, "auc_damage_max":.005, "logloss_damage_max":.01,
    "no_reselection":True, "final_test":False}
read, atomic_json, atomic_text = s.read, s.atomic_json, s.atomic_text
sha256_file, canonical_hash = s.sha256_file, s.canonical_hash


def load_config(repo):
    path = repo / 'configs/ali_stage19_amazon_k2_requests_v1.json'
    c = read(path)
    if c['schema'] != SCHEMA or c['test_access_policy'] != 'forbidden' or c['protocol'] != RULES:
        raise ValueError('Stage19 frozen scope/rules changed')
    if c['output_root'] != 'outputs/ali_stage19_amazon_k2_requests_v1':
        raise ValueError('unexpected output directory')
    amendment = read(repo/c['amendment'])
    if amendment['status'] != 'SPECIFIED_BEFORE_STAGE19_EXECUTION' or amendment['test_accessed']:
        raise ValueError('invalid Amendment18')
    return c, path


def requests_from_metadata(train_users, dev_users, order, excluded):
    train = Counter(map(int,train_users)); dev = Counter(map(int,dev_users))
    window = Counter(int(train_users[i]) for i in order)
    ordered = sorted(train, key=lambda u:(train[u],hashlib.sha256(f'p3c1-tercile:{u}'.encode()).hexdigest()))
    n = len(ordered)
    terciles = {'low':ordered[:n//3], 'middle':ordered[n//3:2*n//3]}
    chosen, counts = {}, {}
    for role, pool in terciles.items():
        eligible = [u for u in pool if u not in excluded and window[u] == 2 and dev[u] >= 5]
        eligible.sort(key=lambda u:hashlib.sha256(f"{RULES['pairing_hash_salt']}:{role}:{u}".encode()).hexdigest())
        counts[role] = len(eligible)
        if len(eligible) < 12:
            raise ValueError(f'insufficient {role} pool: {len(eligible)}; do not relax or replace users')
        chosen[role] = eligible[:12]
    requests = []
    for i, (low,mid) in enumerate(zip(chosen['low'],chosen['middle']),1):
        users = [low,mid]
        positions = [j for j,idx in enumerate(order) if int(train_users[idx]) in users]
        local = [j for j,u in enumerate(dev_users) if int(u) in users]
        if len(positions) != 4 or len(local) < 10:
            raise ValueError('K2 exposure/support mismatch')
        requests.append({'id':f'R{i:02d}', 'users':users, 'composition':{'low':1,'middle':1},
            'forget_window_positions':positions, 'deleted_interactions':4, 'requested_users':2,
            'local_indices':local, 'local_samples':len(local), 'actual_window_ratio':4/3200,
            'metadata':[{'role':role,'user':u,'train_frequency':train[u], 'window_exposure':window[u],
                         'development_samples':dev[u]} for role,u in zip(('low','middle'),users)]})
    if len({u for r in requests for u in r['users']}) != 24:
        raise ValueError('requests must have disjoint users')
    return {'requests':requests, 'pool_counts':counts, 'excluded_users':sorted(excluded),
            'selection_uses_labels_or_predictions':False, 'test_accessed':False}


def verify(path):
    m = read(path/'manifest.json')
    actual = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file() and p != path/'manifest.json'}
    if m['schema'] != SCHEMA or set(m['files']) != actual or (path/'COMPLETED').read_text(encoding='utf-8') != SCHEMA+'\n':
        raise ValueError(f'invalid Stage19 bundle: {path}')
    for rel,h in m['files'].items():
        if sha256_file(path/rel) != h:
            raise ValueError(f'changed Stage19 artifact: {path/rel}')
    return m


def publish(stage, dest):
    if dest.exists():
        raise FileExistsError(dest)
    atomic_text(stage/'COMPLETED',SCHEMA+'\n')
    atomic_json(stage/'manifest.json',{'schema':SCHEMA,'test_accessed':False,
        'files':{p.relative_to(stage).as_posix():sha256_file(p) for p in sorted(stage.rglob('*')) if p.is_file()}})
    dest.parent.mkdir(parents=True,exist_ok=True)
    os.rename(stage,dest)


def freeze(root,name,payload):
    dest=root/name
    if dest.exists():
        verify(dest)
        if read(dest/'result.json') != payload:
            raise ValueError(f'refusing to alter frozen {name}')
    else:
        stage=s.stage_dir(root); atomic_json(stage/'result.json',payload); publish(stage,dest)
    return sha256_file(dest/'result.json')


def audit(repo,workspace,c,path):
    pc=s.load_config(repo,repo/c['parent_config'])
    checks,b,reg,du=s.audit(repo,workspace,pc,repo/c['parent_config'])
    parent=workspace/pc['output_root']
    for p in parent.rglob('manifest.json'):
        if '.work' not in p.parts:
            s.verify(p.parent)
    for name,binding in [('summary',c['parent_summary_sha256']),('selection',c['parent_selection_sha256'])]:
        if sha256_file(parent/name/'result.json') != binding:
            raise ValueError(f'Stage18 {name} changed')
    previous=read(parent/'summary/result.json')
    if previous['selection']['selected']['lambda'] != 48 or previous['confirmation']['status'] != 'NOT_CONFIRMED':
        raise ValueError('unexpected Stage18 outcome; do not silently redefine follow-up')
    s.source_check(repo,read(parent/'protocol/result.json')['checks'])
    tu=read(workspace/b['source']['train_user_ids'])
    registry=requests_from_metadata(tu,du,reg['order'],{u for r in reg['scenarios'] for u in r['users']})
    for p in [path,repo/c['amendment'],repo/'scripts/bota_if/run_ali_stage19_amazon_k2_requests_v1.ps1']:
        checks['source_sha256'][p.relative_to(repo).as_posix()]=sha256_file(p)
    for p in parent.rglob('*'):
        if p.is_file() and '.work' not in p.parts and p.name != 'execution.lock':
            checks['input_sha256'][str(p.resolve())]=sha256_file(p)
    refs={seed:parent/'banks'/('coarse_42' if seed==42 else f'confirm_{seed}')/'Original' for seed in RULES['seeds']}
    checks.update(stage19_registry_sha256=canonical_hash(registry), parent_summary_sha256=c['parent_summary_sha256'])
    return checks,b,reg,du,registry,refs


def check_inputs(repo,checks):
    s.source_check(repo,checks)
    for p,h in checks['input_sha256'].items():
        if sha256_file(Path(p)) != h:
            raise ValueError(f'frozen input changed: {p}')


def build_bank(root,workspace,b,reg,refs,seed,group,shard,contract):
    dest=root/'banks'/f's{seed}_b{shard}'
    identity={'seed':seed,'requests':[r['id'] for r in group],'contract':contract,'scales':RULES['scales']}
    if dest.exists():
        verify(dest)
        if read(dest/'result.json')['identity'] != identity: raise ValueError('bank identity mismatch')
        print(f'[Stage19] verified bank s{seed}/b{shard}',flush=True)
        return dest
    stage=s.stage_dir(root); device=s.gpu(); torch.cuda.reset_peak_memory_stats()
    config=copy.deepcopy(b); config['protocol']['run_seed']=seed; config['coordinate']['fixed_a_seed']=seed
    original=torch.load(refs[seed]/'adapter/adapter_model.pt',map_location='cpu',weights_only=True)
    s.seed_everything(seed); started=time.perf_counter()
    targets=[u for r in group for u in r['users']]
    print(f'[Stage19] bank seed={seed} shard={shard} users={len(targets)}; shared transport kernel follows',flush=True)
    try:
        model,names,params,bases,_,tok,data,users=s.runner._model_context(workspace,config,reg,device,original['A'])
        canonical,vectors,failures,steps,control=s.transport(model,data,reg['order'],users,targets,params,names,
            device,tok.pad_token_id,s.runner._engine(workspace,config),RULES['scales'],stage/'progress.json')
        if steps != 200 or any(not torch.equal(canonical[n],original['B'][n]) for n in names):
            raise RuntimeError('canonical does not reproduce sealed Stage18v2 Original')
        if control['status'] != 'TENSOR_EXACT_PASS' or control['checked_steps'] != 200:
            raise RuntimeError('same-trajectory control failed')
        for r in group:
            slots=[targets.index(u) for u in r['users']]
            for scale in RULES['scales']:
                key=s.cid(scale)
                if key in failures: continue
                values=s.compose(canonical,names,vectors[key][slots].sum(0))
                s.runner._save_fixed_ab(stage/r['id']/key/'adapter',names,values,bases,f'Stage19-FS-{scale}')
        peak=torch.cuda.max_memory_reserved()/2**30
        if peak > 14: raise RuntimeError('GPU memory cap exceeded')
        del model,params,bases,data,tok,vectors
        s.release()
        atomic_json(stage/'result.json',{'identity':identity,'failures':failures,'control':control,
            'canonical_exact':True,'physical_optimizer_steps':steps,'seconds':time.perf_counter()-started,
            'peak_reserved_gib':peak,'test_accessed':False})
        publish(stage,dest)
    except Exception as e:
        atomic_json(stage/'FAILED.json',{'identity':identity,'error':repr(e),'test_accessed':False})
        raise
    return dest


class ProgressBudget(s.s8.StepBudget):
    def step(self,optimizer,arm):
        super().step(optimizer,arm)
        if self.calls==1 or self.calls%25==0:
            print(f'[Stage19] {arm} step={self.calls}/200',flush=True)


def exact(root,workspace,b,reg,refs,seed,request,contract):
    dest=root/'exact'/f's{seed}_{request["id"]}'
    identity={'seed':seed,'request':canonical_hash(request),'contract':contract}
    if dest.exists():
        verify(dest)
        if read(dest/'result.json')['identity'] != identity: raise ValueError('Exact identity mismatch')
        return dest
    stage=s.stage_dir(root); config=copy.deepcopy(b)
    config['protocol']['run_seed']=seed; config['coordinate']['fixed_a_seed']=seed
    device=s.gpu(); original=torch.load(refs[seed]/'adapter/adapter_model.pt',map_location='cpu',weights_only=True)
    s.seed_everything(seed); started=time.perf_counter()
    try:
        model,names,params,bases,_,tok,data,users=s.runner._model_context(workspace,config,reg,device,original['A'])
        budget=ProgressBudget(200)
        values,opt,trace=s.runner._train_named(model,data,reg['order'],users,request['users'],names,params,
            device,tok.pad_token_id,s.runner._engine(workspace,config),budget,f'Exact s{seed}/{request["id"]}')
        if budget.calls!=200 or sum(t['masked_slots'] for t in trace)!=4: raise RuntimeError('Exact masked budget mismatch')
        s.runner._save_fixed_ab(stage/'adapter',names,values,bases,'Stage19-Exact-Masked')
        del model,params,opt,bases,data,tok
        s.release()
        atomic_json(stage/'result.json',{'identity':identity,'physical_optimizer_steps':200,
            'trace_sha256':canonical_hash(trace),'seconds':time.perf_counter()-started,'test_accessed':False})
        publish(stage,dest)
    except Exception as e:
        atomic_json(stage/'FAILED.json',{'identity':identity,'error':repr(e),'test_accessed':False}); raise
    return dest


def predict(root,workspace,b,adapter,key,dataset,indices,contract):
    dest=root/'predictions'/key
    identity={'contract':contract,'adapter_sha256':sha256_file(adapter/'adapter/adapter_model.pt'),
              'indices_sha256':canonical_hash(indices)}
    if dest.exists():
        verify(dest)
        result=read(dest/'result.json')
        if result['identity'] != identity: raise ValueError('prediction identity mismatch')
        return result['prediction']
    stage=s.stage_dir(root); device=s.gpu()
    model=s.evaluation._load_fixed(workspace,b,adapter,device)
    pred={'probability':[],'gold_label':[],'answer_loss':[],'sample_order_sha256':canonical_hash(indices)}
    try:
        for start in range(0,len(indices),256):
            piece=s.evaluation._single_predictions(model,dataset,indices[start:start+256],device,4)
            for k in ('probability','gold_label','answer_loss'): pred[k].extend(piece[k])
            if start==0 or start%1024==0 or start+256>=len(indices):
                print(f'[Stage19] {key} samples={min(start+256,len(indices))}/{len(indices)}',flush=True)
    finally:
        del model; s.release()
    valid=s.s14.validate_prediction(pred,indices)
    atomic_json(stage/'result.json',{'identity':identity,'prediction':pred if valid else None,
        'status':'OK' if valid else 'NONFINITE_PREDICTIONS','test_accessed':False})
    publish(stage,dest)
    return pred if valid else None


def metrics(seed,request,scale,pred,original,reference,dev_users,failure=None):
    if pred is None:
        row={'E_U':None,'D_U':None,'local_residual':None,'utility_eligible':False,
             'prediction_collapse':None,**(failure or {'status':'NONFINITE_PREDICTIONS'})}
    else:
        row=s.s8._metric(s.cid(scale),scale,request['id'],pred,original,reference,request['local_indices'],0,0,{'selection':s.SELECT})
        row['status']='OK'
        row['per_user']=[]
        for u in request['users']:
            ix=[i for i in request['local_indices'] if int(dev_users[i])==u]
            p=np.asarray(pred['probability'])[ix]; e=np.asarray(reference['probability'])[ix]; o=np.asarray(original['probability'])[ix]
            row['per_user'].append({'user':u,'n':len(ix),'E_U':float(np.abs(p-e).mean()),
                                   'D_U':float(np.abs(o-e).mean())})
    row.update(seed=seed,request=request['id'],response_scale=scale,local_samples=len(request['local_indices']),test_accessed=False)
    return row


def run_seed(root,workspace,b,reg,du,registry,refs,seed,contract):
    tok=s.s8.T5Tokenizer.from_pretrained(s.runner._base_t5_config(workspace,b)['paths']['model_dir'])
    dataset=s.s8.JsonPromptDataset(workspace/b['source']['development_json'],tok)
    indices=list(range(len(du)))
    if len(dataset)!=3840 or len(du)!=3840: raise ValueError('unexpected Development size')
    original=predict(root,workspace,b,refs[seed],f's{seed}_Original',dataset,indices,contract)
    if original is None: raise RuntimeError('nonfinite Original predictions')
    requests=registry['requests']; rows=[]
    for start in range(0,len(requests),4):
        group=requests[start:start+4]
        bank=build_bank(root,workspace,b,reg,refs,seed,group,start//4,contract)
        bank_result=read(bank/'result.json')
        for request in group:
            sid=request['id']; dest=root/'cells'/f's{seed}_{sid}'
            if dest.exists():
                verify(dest); result=read(dest/'result.json')
                if result['contract']!=contract or result['request_sha256']!=canonical_hash(request):
                    raise ValueError('completed cell identity mismatch')
                rows.extend(result['rows']); continue
            exact_dir=exact(root,workspace,b,reg,refs,seed,request,contract)
            reference=predict(root,workspace,b,exact_dir,f's{seed}_{sid}_Exact',dataset,indices,contract)
            if reference is None: raise RuntimeError('nonfinite Exact-Masked predictions')
            cell=[]
            for scale in RULES['scales']:
                key=s.cid(scale); failure=bank_result['failures'].get(key)
                pred=None if failure else predict(root,workspace,b,bank/sid/key,f's{seed}_{sid}_{key}',dataset,indices,contract)
                cell.append(metrics(seed,request,scale,pred,original,reference,du,failure))
            freeze(root,f'cells/s{seed}_{sid}',{'contract':contract,'request_sha256':canonical_hash(request),'rows':cell,'test_accessed':False})
            rows.extend(cell)
    return rows


def evidence_gate(rows,comparator):
    # Never omit a failed cell. All seeds of a request travel together in bootstrap.
    keyed={(r['seed'],r['request'],r['response_scale']):r for r in rows}
    ids=[f'R{i:02d}' for i in range(1,13)]
    a,b=np.zeros((3,12)),np.zeros((3,12))
    utility_failures=[]
    for i,seed in enumerate(RULES['seeds']):
        for j,rid in enumerate(ids):
            candidate=keyed[(seed,rid,48)]
            control=keyed[(seed,rid,0)]
            if candidate['status']!='OK' or (comparator=='lambda0' and control['status']!='OK'):
                return {'passed':False,'reason':'NUMERICAL_FAILURE_NO_CELL_EXCLUSION','comparator':comparator}
            a[i,j]=candidate['E_U']; b[i,j]=control['E_U'] if comparator=='lambda0' else candidate['D_U']
            if not candidate['utility_eligible'] or candidate['prediction_collapse']:
                utility_failures.append(f'{seed}/{rid}')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'passed':False,'reason':'NONFINITE_METRICS','comparator':comparator}
    gains=b-a; per_seed={}
    for i,seed in enumerate(RULES['seeds']):
        gain=float(gains[i].mean()); base=float(b[i].mean()); rel=gain/base if base>0 else None
        per_seed[str(seed)]={'candidate_mean_E_U':float(a[i].mean()),'comparator_mean_E_U':base,
            'absolute_gain':gain,'relative_gain':rel,'passed':gain>=1e-6 and rel is not None and rel>=.01}
    request_gains=gains.mean(0); wins=int((request_gains>0).sum())
    rng=np.random.default_rng(RULES['bootstrap_seed'])
    draws=rng.integers(0,12,size=(RULES['bootstrap_resamples'],12))
    interval=np.quantile(request_gains[draws].mean(1),[.025,.975]).tolist()
    passed=not utility_failures and all(x['passed'] for x in per_seed.values()) and wins>=8 and interval[0]>0
    return {'passed':bool(passed),'comparator':comparator,'per_seed':per_seed,'winning_requests':wins,
            'total_requests':12,'request_mean_gains':dict(zip(ids,request_gains.tolist())),
            'paired_request_bootstrap_gain_95_interval':interval,'utility_failures':utility_failures,
            'mean_gain':float(gains.mean()),'worst_cell_gain':float(gains.min()),
            'interval_scope':'conditional on reused Development and shared models; not population inference'}


def summarize(rows):
    keys=[(r['seed'],r['request'],r['response_scale']) for r in rows]
    expected={(seed,f'R{i:02d}',scale) for seed in RULES['seeds'] for i in range(1,13) for scale in RULES['scales']}
    if len(keys)!=len(expected) or set(keys)!=expected: raise ValueError('incomplete/duplicate Stage19 cells')
    mechanism=evidence_gate(rows,'lambda0'); reconstruction=evidence_gate(rows,'Original')
    outcome='L2_MATCHED_COUNTERFACTUAL_SUPPORT' if mechanism['passed'] and reconstruction['passed'] else (
        'L1_FISHER_GAIN_ONLY' if mechanism['passed'] else 'NOT_SUPPORTED')
    distributions={}
    for scale in RULES['scales']:
        selected=[r for r in rows if r['response_scale']==scale]
        valid=[r for r in selected if r['status']=='OK']
        distributions[str(scale)]={'cells':len(selected),'valid_cells':len(valid),
            'all_cell_mean_E_U':float(np.mean([r['E_U'] for r in valid])) if len(valid)==36 else None,
            'all_cell_mean_residual':float(np.mean([r['local_residual'] for r in valid])) if len(valid)==36 and all(r['local_residual'] is not None for r in valid) else None,
            'residual_below_one':sum(r['local_residual'] is not None and r['local_residual']<1 for r in valid),
            'utility_passed':sum(r['utility_eligible'] and not r['prediction_collapse'] for r in valid),
            'small_D_U_cells':[f"{r['seed']}/{r['request']}" for r in valid if r['D_U']<=1e-8],
            'worst_E_U':max((r['E_U'] for r in valid),default=None),
            'numerical_failures':[f"{r['seed']}/{r['request']}" for r in selected if r['status']!='OK']}
    return {'schema':SCHEMA,'status':'COMPLETED','classification':outcome,'primary_lambda':48,
        'secondary_lambda':64,'mechanism_gate':mechanism,'reconstruction_gate':reconstruction,
        'distributions':distributions,'rows':rows,'test_accessed':False,'new_holdout':False,'no_reselection':True}


def make_report(summary):
    lines=['# Stage19: new Amazon K2 requests','',f"Classification: **{summary['classification']}**",'',
        'Primary lambda48; lambda0 ablation; lambda64 secondary (cannot rescue primary outcome).',
        'Twelve user-disjoint requests, three seeds, four masked slots and ten local examples per request.',
        'New request identities on previously used Development. Not a new holdout or privacy-forgetting test.',
        'Shared pretrained model, cohort and order; seeds change fixed-A/runtime only.',
        'Request-cluster bootstrap intervals are conditional diagnostics, not independent population claims.','',
        '| Seed | Request | Lambda | E_U | D_U | Residual | Toward | AUC damage | Logloss damage | Utility | Status |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in summary['rows']:
        f=lambda k: '--' if r.get(k) is None else f"{r[k]:.7g}"
        lines.append(f"| {r['seed']} | {r['request']} | {r['response_scale']} | {f('E_U')} | {f('D_U')} | {f('local_residual')} | {f('toward_exact_rate')} | {f('auc_damage')} | {f('logloss_damage')} | {r['utility_eligible']} | {r['status']} |")
    for key in ('mechanism_gate','reconstruction_gate'):
        lines += ['',f'## {key}','', '```json',json.dumps(summary[key],indent=2),'```']
    lines += ['','All requests and failures retained. No per-request lambda selection or new sweep is authorized by this report.']
    return '\n'.join(lines)+'\n'


def final_report(root):
    paths=sorted((root/'cells').glob('*/result.json'))
    if len(paths)!=36:
        return {'status':'INCOMPLETE','completed_request_cells':len(paths),'required_request_cells':36,'test_accessed':False}
    rows=[]
    for p in paths:
        verify(p.parent); rows.extend(read(p)['rows'])
    summary=summarize(rows); dest=root/'summary'
    if dest.exists():
        verify(dest)
        if read(dest/'result.json')!=summary: raise ValueError('summary changed')
    else:
        stage=s.stage_dir(root); atomic_json(stage/'result.json',summary)
        atomic_text(stage/'report.md',make_report(summary)); publish(stage,dest)
    return {'status':'COMPLETED','classification':summary['classification'],'summary':str(dest),'test_accessed':False}


def smoke(root,workspace,b,reg,registry,refs):
    # Exercises the largest8-user bank shape, all controls, serialization and prediction.
    stage=s.stage_dir(root); device=s.gpu(); torch.cuda.reset_peak_memory_stats(); s.seed_everything(42)
    original=torch.load(refs[42]/'adapter/adapter_model.pt',map_location='cpu',weights_only=True)
    group=registry['requests'][:4]; targets=[u for r in group for u in r['users']]
    model,names,params,bases,_,tok,data,users=s.runner._model_context(workspace,b,reg,device,original['A'])
    pos=min(p for r in group for p in r['forget_window_positions']); start=pos//16*16
    canonical,vectors,failures,steps,control=s.transport(model,data,reg['order'][start:start+32],users,targets,
        params,names,device,tok.pad_token_id,s.runner._engine(workspace,b),RULES['scales'],stage/'progress.json')
    if steps!=2 or failures: raise RuntimeError('Stage19 smoke transport failure')
    value=s.compose(canonical,names,vectors[s.cid(48)][:2].sum(0))
    s.runner._save_fixed_ab(stage/'smoke/adapter',names,value,bases,'SMOKE-NOT-SCIENTIFIC')
    del model,params,bases,data,vectors
    s.release()
    dataset=s.s8.JsonPromptDataset(workspace/b['source']['development_json'],tok)
    p=predict(root,workspace,b,stage/'smoke','smoke_'+stage.name,dataset,group[0]['local_indices'],'SMOKE-NOT-SCIENTIFIC')
    if p is None: raise RuntimeError('smoke predictions nonfinite')
    result={'status':'SMOKE_PASS','users':8,'steps':2,'control':control,'scales':RULES['scales'],
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'scientific_result':False,'test_accessed':False}
    atomic_json(stage/'smoke.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--workspace',type=Path,default=Path('C:/python/E2URec-main'))
    parser.add_argument('--mode',choices=['Preflight','Freeze','Smoke','Full','Analyze'],default='Preflight')
    parser.add_argument('--seed',choices=['All','41','42','43'],default='All')
    args=parser.parse_args(); repo=args.repo.resolve(); workspace=args.workspace.resolve()
    c,path=load_config(repo); root=workspace/c['output_root']
    with s.s14.execution_lock(root):
        checks,b,reg,du,registry,refs=audit(repo,workspace,c,path)
        if args.mode=='Preflight':
            result={'status':'PREFLIGHT_PASS','requests':12,'unique_users':24,'pool_counts':registry['pool_counts'],
                'excluded_users':registry['excluded_users'],'request_hash':canonical_hash(registry),
                'local_samples_per_request':[r['local_samples'] for r in registry['requests']],
                'masked_slots_per_request':4,'optimizer_step_budget_full':9000,'test_accessed':False}
        elif args.mode=='Analyze':
            if not (root/'protocol').exists(): raise ValueError('no frozen Stage19 protocol')
            verify(root/'protocol')
            frozen=read(root/'protocol/result.json')
            if frozen['checks']!=checks or frozen['registry']!=registry: raise ValueError('frozen inputs changed')
            for p in root.rglob('manifest.json'):
                if '.work' not in p.parts: verify(p.parent)
            # Analyze only reads; Full publishes the final report.
            result=read(root/'summary/result.json') if (root/'summary').exists() else {'status':'INCOMPLETE','completed_request_cells':len(list((root/'cells').glob('*/result.json')))}
        elif args.mode=='Smoke':
            # Isolate QA from scientific protocol/units so later code fixes do
            # not invalidate a preregistration sealed by a two-step smoke test.
            qa_root=s.stage_dir(root)
            atomic_json(qa_root/'qa_contract.json',{'checks':checks,'registry':registry,'scientific_result':False,'test_accessed':False})
            previous=Path.cwd(); os.chdir(workspace)
            try:
                result=smoke(qa_root,workspace,b,reg,registry,refs)
                result['diagnostics']=str(qa_root)
            finally:
                os.chdir(previous); s.release()
        else:
            contract=freeze(root,'protocol',{'schema':SCHEMA,'checks':checks,'config':c,'registry':registry,
                'environment':{'torch':torch.__version__,'cuda':torch.version.cuda},'test_accessed':False})
            previous=Path.cwd(); os.chdir(workspace)
            try:
                if args.mode=='Freeze':
                    result={'status':'FROZEN','contract':contract,'requests':12,'test_accessed':False}
                else:
                    seeds=RULES['seeds'] if args.seed=='All' else [int(args.seed)]
                    for seed in seeds:
                        run_seed(root,workspace,b,reg,du,registry,refs,seed,contract)
                        check_inputs(repo,checks)
                    result=final_report(root)
            finally:
                os.chdir(previous); s.release()
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
