"""Frozen Stage17 rescore + independent artifact audit. No training or FinalTest.

Only new bundles are written. Stage17 models/requests and Stage25 audit tolerances
are immutable inputs. Missing weights are reported; never reconstructed here.
"""
from __future__ import annotations
import argparse
import copy
import gc
import json
import os
import statistics
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import T5Tokenizer
from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, sha256_file
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset
from . import ali_stage17_common as c17
from . import ali_stage17_evaluation as metrics
from . import ali_stage25_scoring as scoring
from . import ali_stage25_historical_mask_audit_v1 as s25

SCHEMA = 'bota-ali-stage26-stress-mask-rescore-v1'
ROOT = 'outputs/ali_stage26_stress_mask_rescore_v1'
CONFIG = 'configs/ali_stage26_stress_mask_rescore_v1.json'
FAMILIES = ('primary', 'stronger', 'sequential', 'horizon', 'compression', 'relay')
read = c17.read


def verify(path, contract=None):
    m = read(path/'manifest.json')
    actual = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file() and p != path/'manifest.json'}
    if m['schema'] != SCHEMA or actual != set(m['files']):
        raise ValueError(f'invalid bundle inventory: {path}')
    if (path/'COMPLETED').read_text(encoding='utf-8') != SCHEMA+'\n':
        raise ValueError(f'incomplete bundle: {path}')
    for rel, h in m['files'].items():
        if sha256_file(path/rel) != h: raise ValueError(f'changed artifact: {path/rel}')
    result = read(path/'result.json')
    if contract is not None and result['contract'] != contract: raise ValueError('contract changed')
    return result


def task(root, key, contract, fn):
    dest = root/key
    if dest.exists(): return verify(dest, contract)
    stage = root/'.work'/uuid.uuid4().hex[:10]
    stage.mkdir(parents=True)
    started = time.perf_counter()
    print(f'[Stage26] {key} work={stage}', flush=True)
    try:
        result = {**fn(stage), 'contract': contract, 'optimizer_updates': 0,
                  'test_accessed': False, 'selection_performed': False,
                  'wall_seconds': time.perf_counter()-started}
        atomic_json(stage/'result.json', result)
        atomic_text(stage/'COMPLETED', SCHEMA+'\n')
        atomic_json(stage/'manifest.json', {'schema': SCHEMA, 'files': {
            p.relative_to(stage).as_posix(): sha256_file(p) for p in sorted(stage.rglob('*')) if p.is_file()}})
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, dest)
        return result
    except BaseException as error:
        atomic_json(stage/'FAILED.json', {'key': key, 'error': repr(error), 'contract': contract})
        raise


class Auditor:
    """Check existing bytes, distinguishing absence from a changed retained file."""
    def __init__(self): self.hashes = {}; self.missing = set(); self.bundles = {}
    def file(self, path, expected=None):
        path = Path(path).resolve(); name = str(path)
        if not path.is_file(): self.missing.add(name); return None
        h = self.hashes.get(name)
        if h is None: h = self.hashes[name] = sha256_file(path)
        if expected is not None and h != expected: raise ValueError(f'input hash mismatch: {path}')
        return h
    def bundle(self, path):
        path = Path(path).resolve()
        if str(path) in self.bundles: return self.bundles[str(path)]
        if self.file(path/'manifest.json') is None:
            result = {'status': 'ARTIFACT_MISSING', 'path': str(path)}
        else:
            m = read(path/'manifest.json'); missing = []
            actual = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file() and p != path/'manifest.json'}
            if actual-set(m['files']): raise ValueError(f'unregistered files in {path}')
            for rel, h in m['files'].items():
                if self.file(path/rel, h) is None: missing.append(rel)
            result = {'status': 'VERIFIED' if not missing else 'ARTIFACT_MISSING',
                      'path': str(path), 'manifest_sha256': self.hashes[str(path/'manifest.json')], 'missing': missing}
        self.bundles[str(path)] = result
        return result


def entry_status(workspace, entry, auditor):
    if entry.get('status') == 'HISTORICAL_NUMERICAL_FAILURE': return entry
    path = (workspace/entry['path']).resolve()
    if not path.is_relative_to(workspace.resolve()): raise ValueError('endpoint outside workspace')
    required = ['adapter_model.pt', 'adapter_config.json'] if entry['kind']=='fixed_ab' else ['aggregation.json']
    missing = [rel for rel in required if auditor.file(path/rel) is None]
    if not missing and entry['kind']=='receraser_ensemble':
        weights = read(path/'aggregation.json')['weights']
        missing.extend(f'models/shard_{i:02d}/final_state.pt' for i in range(len(weights))
                       if auditor.file(path/f'models/shard_{i:02d}/final_state.pt') is None)
    files = [(p.relative_to(path).as_posix(), auditor.file(p)) for p in sorted(path.rglob('*')) if p.is_file()]
    digest = canonical_hash(files)
    if not missing and entry.get('directory_sha256') and digest != entry['directory_sha256']:
        raise ValueError(f'frozen endpoint differs: {path}')
    return {**entry, 'path': str(path), 'directory_sha256': digest,
            'status': 'WEIGHTS_MISSING' if missing else 'READY', 'missing': missing}


def materialized(workspace, auditor, bank, seed, sid, users, arm='FS64', method='BOTA64'):
    key = 'adapter_'+canonical_hash([bank, sid, users, arm])[:16]
    path = c17.path(workspace, key)
    audit = auditor.bundle(path)
    entry = {'path': str(path/'adapter'), 'kind': 'fixed_ab', 'method': method, 'seed': seed, 'users': users}
    if audit['status'] == 'VERIFIED':
        old = read(path/'result.json')
        if old['status'] == 'UNAVAILABLE_NUMERICAL_TRANSPORT':
            return {**entry, 'status': 'HISTORICAL_NUMERICAL_FAILURE', 'failure': old.get('failure')}
    return entry


def specifications(workspace, parent, auditor):
    specs = []
    def add(family, seed, sid, users, entries, formal=False, order=None):
        # Unit-response endpoints remain in the historical archive, not this
        # fixed-lambda64 refresh. Retain lambda0 and matched FM ablations.
        entries = [entry_status(workspace, dict(e), auditor) for e in entries if e['method']!='BOTA1']
        assert len({e['method'] for e in entries}) == len(entries)
        specs.append({'id': family+f'_s{seed}_'+canonical_hash(sid)[:12], 'family': family,
                      'seed': seed, 'request': sid, 'users': users, 'entries': entries,
                      'formal_attack': formal, 'order': order,
                      'lanes': ['binary'] if family=='relay' else ['binary', 'ranking'] + ([] if family=='compression' else ['attack'])})
    for seed in c17.SEEDS:
        ctx = parent['contexts'][str(seed)]
        ori = next(e for e in ctx['endpoints'] if e['method']=='Original')
        for sid in c17.s16.SCENARIOS:
            entries = [dict(e) for e in ctx['endpoints'] if e['scenario']==sid]
            users = entries[0]['users']
            entries += [materialized(workspace,auditor,f'bank_primary_s{seed}',seed,sid,users,arm,arm) for arm in ('FM0','FM64')]
            add('primary',seed,sid,users,entries,True)
        for row in ctx['sequential']:
            add('sequential',seed,row['union_hash'],row['users'],[ori,row['exact'],
                materialized(workspace,auditor,f'bank_primary_s{seed}',seed,row['union_hash'],row['users'])])
        for row in ctx['long']:
            add('horizon',seed,row['id'],row['users'],[row['original'],row['exact'],
                materialized(workspace,auditor,f'bank_long_s{seed}',seed,row['id'],row['users'])],
                order=c17.s3.build_long_schedule(ctx['registry']['order'],seed))
        screen = c17.path(workspace, f'storage_screen_s{seed}')
        auditor.bundle(screen)
        for rep in c17.s6._representation_names():
            for sid in c17.s16.SCENARIOS:
                refs = [dict(e) for e in ctx['endpoints'] if e['scenario']==sid and e['method'] in ('Original','Exact-Masked','BOTA64')]
                users = refs[0]['users']
                for e in refs:
                    if e['method']=='BOTA64': e['method']='Uncompressed-FS64'
                refs.append({'path':str(screen/'adapters'/rep/sid/'adapter'),'kind':'fixed_ab',
                             'method':'BOTA64','seed':seed,'users':users})
                add('compression',seed,rep+'_'+sid,users,refs)
        if seed==42:
            for row in ctx['stronger']:
                add('stronger',seed,row['id'],row['users'],[ori,row['exact'],
                    materialized(workspace,auditor,'bank_strong_s42',seed,row['id'],row['users'])],row['id']=='high_k4')
    # The historical relay has a different schedule; reuse its own final models
    # and original full Development cohort, never merge it with 1200-step runs.
    path = c17.path(workspace,'legacy_relay_L8'); auditor.bundle(path)
    users = next(r['users'] for r in parent['contexts']['42']['registry']['scenarios'] if r['id']=='L8')
    add('relay',42,'L8_600',users,[{'path':str(path/'adapters'/m),'kind':'fixed_ab','method':m,'seed':42,'users':users}
                                for m in ('Original','Exact-Masked','BOTA64')])
    return specs


def prepare(repo, workspace, config):
    print('[Stage26] inventory and hashes only; no model inference/training', flush=True)
    auditor = Auditor()
    protocol = c17.root(workspace)/'protocol'
    if auditor.bundle(protocol)['status']!='VERIFIED': raise ValueError('Stage17 authority missing')
    parent = read(protocol/'contract.json')
    if parent['protocol'] != c17.PROTOCOL: raise ValueError('historical protocol drift')
    specs = specifications(workspace,parent,auditor)
    benchmark = parent['contexts']['42']['benchmark']
    for ctx in parent['contexts'].values():
        if ctx['benchmark']['source'] != benchmark['source']: raise ValueError('dataset lineage differs across seeds')
    for rel in ('train_json','development_json'):
        p = workspace/benchmark['source'][rel]
        auditor.file(p, parent['hashes'].get(str(p.resolve())))
    for dep in parent['dependencies'].values():
        p = workspace/dep['path']
        if p.is_file(): auditor.file(p,dep['sha256'])
        elif p.is_dir():
            files = [(q.relative_to(p).as_posix(),auditor.file(q)) for q in sorted(p.rglob('*')) if q.is_file()]
            if canonical_hash(files)!=dep['sha256']: raise ValueError(f'model dependency changed {p}')
        else: raise ValueError(f'model dependency missing: {p}')
    for p in (workspace/benchmark['source']['raw_data']).rglob('*'):
        if p.is_file(): auditor.file(p,parent['hashes'].get(str(p.resolve())))
    rank_path = workspace/c17.s15.ROOT/'protocol/candidates.json'
    auditor.file(rank_path,parent['ranking_plan_sha256'])
    ranking = read(rank_path)
    train_users, dev_users, _ = c17.s2._source_users(workspace,benchmark)
    train = read(workspace/benchmark['source']['train_json'])
    dev = read(workspace/benchmark['source']['development_json'])
    if len(dev)!=20000 or len(dev_users)!=len(dev) or len(train_users)!=len(train): raise ValueError('cohort size/lineage changed')
    cohorts = {}
    binary = {'scope':'Development','lane':'binary','benchmark':benchmark,'indices':list(range(len(dev))),
              'users':dev_users,'content_sha256':canonical_hash(dev)}
    rank = {'scope':'Development','lane':'ranking','benchmark':benchmark,'indices':list(range(ranking['candidate_count'])),
            'plan':ranking,'content_sha256':canonical_hash(ranking)}
    for lane, cohort in (('binary',binary),('ranking',rank)): cohorts[lane]=cohort
    jobs = {}
    for spec in specs:
        spec['cohorts']={lane:lane for lane in spec['lanes'] if lane!='attack'}
        if 'attack' in spec['lanes']:
            order = spec['order'] if spec['order'] is not None else parent['contexts'][str(spec['seed'])]['registry']['order']
            positions = [i for i in order if train_users[i] in spec['users']]
            nonmember = [i for i,u in enumerate(dev_users) if u in spec['users']]
            covered = {train_users[i] for i in positions}&{dev_users[i] for i in nonmember}
            positions = [i for i in positions if train_users[i] in covered]
            nonmember = [i for i in nonmember if dev_users[i] in covered]
            members = [train_users[i] for i in positions]; nonmembers = [dev_users[i] for i in nonmember]
            key = 'attack_'+canonical_hash([positions,nonmember,members,nonmembers])[:20]
            cohorts[key] = {'scope':'Development','lane':'attack','benchmark':benchmark,
                'positions':positions,'nonmember':nonmember,'members':members,'nonmembers':nonmembers,
                'users':members+nonmembers,'indices':list(range(len(members)+len(nonmembers))),
                'content_sha256':canonical_hash([[train[i] for i in positions]+[dev[i] for i in nonmember],members,nonmembers])}
            spec['cohorts']['attack']=key
        spec['jobs']={}
        for lane, cohort_id in spec['cohorts'].items():
            spec['jobs'][lane]={}
            for entry in spec['entries']:
                # Explicit request-independent probes allow safe cache reuse for
                # the same model and cohort; no metric/request is deduplicated.
                identity=[entry['path'],entry.get('directory_sha256'),entry['status'],cohort_id]
                jid=canonical_hash(identity)[:24]
                jobs[jid]={'id':jid,'cohort':cohort_id,**entry,
                           'provenance':{'users':sorted(set(cohorts[cohort_id].get('users',[])))}}
                spec['jobs'][lane][entry['method']]=jid
    reuse=[]
    for seed in c17.SEEDS:
        for key in (f'storage_screen_s{seed}',f'latency_s{seed}'):
            item=auditor.bundle(c17.path(workspace,key));reuse.append({**item,'key':key,'action':'reuse_unmodified_no_rerun'})
    for key in ('legacy_relay_L8',*[f'legacy_position_{x}_frequency' for x in ('low','middle','high')]):
        item=auditor.bundle(c17.path(workspace,key));reuse.append({**item,'key':key,'action':'reuse_parameter_only_no_legacy_behavior'})
    # Parameter result rows are independent of inference. Preserve all 132 cells.
    parameter_records=[]
    for p in sorted((c17.root(workspace)/'cells').glob('eval_*/result.json')):
        auditor.file(p); auditor.bundle(p.parent)
        r=read(p)
        parameter_records.append({k:r[k] for k in ('family','seed','request','parameter')})
    for p in sorted((repo/'src').rglob('*.py')): auditor.file(p)
    for p in (repo/CONFIG,repo/'scripts/bota_if/run_ali_stage26_stress_mask_rescore_v1.ps1'):auditor.file(p)
    plan={'schema':SCHEMA,'config':config,'specs':specs,'cohorts':cohorts,'jobs':jobs,
          'reuse':reuse,'parameter_records':parameter_records,'sequence_cells':{
              seed:ctx['sequence_cells'] for seed,ctx in parent['contexts'].items()},
          'hashes':auditor.hashes,'missing_paths':sorted(auditor.missing),
          'tokenizer':str(workspace/parent['dependencies']['pretrained_model_directory']['path']),
          'runtime':c17.runtime(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    return plan


def check(plan):
    if c17.runtime()!=plan['runtime']: raise ValueError('frozen runtime changed')
    if (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)!=plan['gpu']: raise ValueError('GPU changed')
    for path,h in plan['hashes'].items():
        if not Path(path).is_file() or sha256_file(Path(path))!=h: raise ValueError(f'frozen input changed: {path}')
    for path in plan['missing_paths']:
        if Path(path).exists(): raise ValueError(f'previously missing input appeared; audit separately: {path}')


class CachedDataset:
    def __init__(self, data): self.data=data; self.tokenizer=data.tokenizer; self.cache={}
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        if i not in self.cache:self.cache[i]=self.data[i]
        return self.cache[i]
    def collate_fn(self,rows):return self.data.collate_fn(rows)


def dataset(workspace,plan,cohort):
    if cohort['scope']!='Development':raise PermissionError('Stage26 forbids FinalTest')
    tok=T5Tokenizer.from_pretrained(plan['tokenizer'],local_files_only=True)
    source=cohort['benchmark']['source']
    if cohort['lane']=='ranking':
        users,movies=c17.s15.lineage.load_raw_metadata(workspace/source['raw_data'])
        data=c17.s15.RankingDataset(cohort['plan'],tok,users,movies)
        if canonical_hash(cohort['plan'])!=cohort['content_sha256']:raise ValueError('ranking changed')
    else:
        data=JsonPromptDataset(workspace/source['development_json'],tok)
        if cohort['lane']=='attack':
            train=read(workspace/source['train_json'])
            data.records=[train[i] for i in cohort['positions']]+[data.records[i] for i in cohort['nonmember']]
            digest=canonical_hash([data.records,cohort['members'],cohort['nonmembers']])
        else:digest=canonical_hash(data.records)
        if digest!=cohort['content_sha256']:raise ValueError('cohort contents changed')
    if len(data)!=len(cohort['indices']):raise ValueError('cohort size changed')
    return CachedDataset(data)


def evaluate_spec(spec,plan,results):
    predictions={};statuses={}
    for lane,jobs in spec['jobs'].items():
        statuses[lane]={m:results[j]['status'] for m,j in jobs.items()}
        predictions[lane]={m:results[j].get('prediction') for m,j in jobs.items()}
    binary=predictions['binary'];users=plan['cohorts']['binary']['users'];request=set(spec['users'])
    if binary.get('Original') is None or binary.get('Exact-Masked') is None:
        return {'status':'REFERENCE_UNAVAILABLE','family':spec['family'],'seed':spec['seed'],
                'request':spec['request'],'endpoint_statuses':statuses,'behavior':None,'ranking':None,'attack':None}
    # Keep absent methods explicit, without calling them nonfinite or wins.
    available={m:p for m,p in binary.items() if p is not None}
    behavior=metrics.behavior(available,users,request)
    for m,status in statuses['binary'].items():
        if status!='OK':behavior.setdefault('rows',[]).append({'method':m,'status':status})
    ranking={m:(c17.s15.metrics_for(plan['cohorts']['ranking']['plan'],p['probability'],request)
                if p is not None else {'status':statuses['ranking'][m]})
             for m,p in predictions.get('ranking',{}).items()}
    attack=None
    if 'attack' in predictions:
        cohort=plan['cohorts'][spec['cohorts']['attack']]
        bota=next((r for r in behavior.get('rows',[]) if r['method']=='BOTA64'),{})
        attack=metrics.attacks(predictions['attack'],cohort['members'],cohort['nonmembers'],spec['formal_attack'],
                               bota.get('utility',{}).get('pass',False))
        attack['endpoint_statuses']=statuses['attack']
        attack['excluded_requested_users']=sorted(request-set(cohort['members']))
    unavailable=any(s!='OK' for values in statuses.values() for s in values.values())
    return {'status':'COMPLETED_WITH_UNAVAILABLE' if unavailable else 'COMPLETED',
            'family':spec['family'],'seed':spec['seed'],'request':spec['request'],
            'lambda':64,'behavior':behavior,'ranking':ranking,'attack':attack,'endpoint_statuses':statuses}


def selected_specs(plan,family):return [s for s in plan['specs'] if family=='All' or s['family']==family]


def scoring_cohort(cohort):
    # Stage17 ranking used the common float32 yes/no normalization, unlike
    # Stage15/25 Development ranking's float64-mass convention. Keep Stage17's
    # formula: only encoder masking changes. This internal tag is not a split.
    return {**cohort,'scope':'Stage17Development'}


def analyze(root,plan,family='All'):
    contract=canonical_hash(plan);done=[];pending=[]
    for spec in selected_specs(plan,family):
        ids={j for lane in spec['jobs'].values() for j in lane.values()}
        if any(not (root/'predictions'/j).exists() for j in ids):pending.append(spec['id']);continue
        def work(stage):
            results={j:verify(root/'predictions'/j,contract) for j in ids}
            if any(r['status']=='INVARIANCE_FAILED' for r in results.values()):raise RuntimeError('invariance gate failed')
            return evaluate_spec(spec,plan,results)
        done.append(task(root,'metrics/'+spec['id'],contract,work))
    if pending:return {'status':'INCOMPLETE','family':family,'completed':len(done),'pending':pending}
    def summary(stage):
        behavior=[];ranking=[];attacks=[]
        for r in done:
            common={k:r[k] for k in ('family','seed','request')}
            behavior.extend({**common,**row} for row in (r.get('behavior') or {}).get('rows',[]))
            for method,value in (r.get('ranking') or {}).items():
                ranking.extend({**common,'method':method,**row} for row in value.get('aggregates',[value]))
            if r.get('attack') is not None:attacks.append({**common,**r['attack']})
        sequence=[];lookup={(r['seed'],r['request']):r for r in done if r['family']=='sequential'}
        for seed,cells in plan['sequence_cells'].items():
            for row in cells:
                if (int(seed),row['union_hash']) in lookup:
                    sequence.append({'seed':int(seed),**row,'evaluation':lookup[int(seed),row['union_hash']]})
        for name,rows in [('behavior',behavior),('ranking',ranking)]:s25.s24.write_csv(stage/(name+'.csv'),rows)
        atomic_json(stage/'attacks.json',attacks);atomic_json(stage/'sequential_all_cells.json',sequence)
        atomic_json(stage/'all_results.json',done)
        atomic_text(stage/'REPORT.md','# Stage26 frozen masked stress evaluation\n\n'
            'Frozen lambda64; source-only/FM diagnostic arms preserved. No retraining, selection or FinalTest.\n'
            'All nine stronger requests reported; formal stronger attack interpretation remains high_k4 only.\n'
            'Missing weights and historical numerical failures are explicit, never comparator wins.\n'
            'Identical user unions share inference but all 60 sequential order/size cells are reported.\n'
            'Storage/SVD/parameter/timing records are verified unchanged, not measured again.\n'
            'An attack outcome is an empirical diagnostic, never a privacy certificate.\n')
        return {'status':'COMPLETED_WITH_UNAVAILABLE' if any(r['status']!='COMPLETED' for r in done) else 'COMPLETED',
                'family':family,'evaluation_cells':len(done),'sequential_order_cells':len(sequence),
                'behavior_rows':len(behavior),'ranking_rows':len(ranking),'attack_cells':len(attacks)}
    return task(root,'summary/'+family,contract,summary)


def execute(root,workspace,plan,family):
    contract=canonical_hash(plan)
    ids={j for s in selected_specs(plan,family) for lane in s['jobs'].values() for j in lane.values()}
    for cohort_id,cohort in plan['cohorts'].items():
        data=None
        try:
            for jid in sorted(ids):
                job=plan['jobs'][jid]
                if job['cohort']!=cohort_id:continue
                dest=root/'predictions'/jid
                if dest.exists():
                    if verify(dest,contract)['status']=='INVARIANCE_FAILED':raise RuntimeError('frozen invariance audit failed')
                    continue
                if job['status']!='READY' or not cohort['indices']:
                    task(root,'predictions/'+jid,contract,lambda stage:{'status':job['status'] if job['status']!='READY' else 'NO_MATCHED_USER_SUPPORT','prediction':None})
                    continue
                if data is None:data=dataset(workspace,plan,cohort)
                def work(stage):
                    def progress(shard,n,total):
                        atomic_json(stage/'progress.json',{'job':jid,'cohort':cohort_id,'method':job['method'],'done':n,'total':total,'shard':shard})
                        print(f'[Stage26] {cohort_id}/{job["method"]} shard={shard} {n}/{total}',flush=True)
                    try:return scoring.score_endpoint(workspace,scoring_cohort(cohort),job,data,plan['config'],progress,read)
                    except ValueError as error:
                        if str(error) not in ('nonfinite model logits','nonfinite scores','nonfinite prediction'):raise
                        return {'status':'NONFINITE_PREDICTIONS','error':str(error),'prediction':None}
                result=task(root,'predictions/'+jid,contract,work)
                if result['status']=='INVARIANCE_FAILED':raise RuntimeError('batch invariance failed; inspect preserved diagnostics')
        finally:del data;gc.collect()
    check(plan)
    return analyze(root,plan,family)


def estimate(workspace,plan):
    """Observed Stage25 per-lane medians plus conservative fixed overhead."""
    times=defaultdict(list)
    for p in (workspace/s25.ROOT/'cells').glob('*/result.json'):
        r=read(p);j=r.get('job',{});cohort=j.get('cohort','')
        if r['status']=='OK' and cohort.startswith('ML1M_'):
            if '_ranking_' in cohort:times['ranking',j['kind']].append(r['wall_seconds'])
            elif '_FinalTest_binary_' in cohort:times['binary',j['kind']].append(r['wall_seconds'])
    total=0.;counts=Counter();seconds_by_job={}
    for j in plan['jobs'].values():
        if j['status']!='READY':continue
        cohort=plan['cohorts'][j['cohort']];lane=cohort['lane'];counts[lane]+=1
        kind=j['kind']
        if lane=='attack':seconds=15+len(cohort['indices'])/20000*statistics.median(times['binary',kind])
        else:seconds=statistics.median(times[lane,kind])
        # Stage25 used 53,809 Development / 55,064 FinalTest candidates.
        # Their medians are an empirical approximation, not an exact throughput.
        if lane=='ranking':seconds*=len(cohort['indices'])/54436.5
        total+=seconds;seconds_by_job[j['id']]=seconds
    family_hours={}
    for family in FAMILIES:
        ids={j for spec in selected_specs(plan,family) for lane in spec['jobs'].values() for j in lane.values()}
        family_hours[family]=round(sum(seconds_by_job.get(j,0) for j in ids)/3600,2)
    # Do not claim runtime certainty: auditing, CPU bootstraps and contention vary.
    return {'unique_jobs_by_lane':dict(counts),'inference_point_hours':total/3600,
            'full_wall_hours_range':[round(total/3600*.85+1,1),round(total/3600*1.4+3,1)],
            'standalone_family_inference_hours_not_additive':family_hours,
            'basis':'Stage25 observed inference medians; includes conservative audit/CPU-bootstrap allowance; not a guarantee'}


def smoke(root,workspace,plan):
    """Small real-model inference only; never create full-scoring cache entries."""
    contract=canonical_hash(plan)
    spec=next(s for s in plan['specs'] if s['family']=='primary' and s['seed']==42 and s['request']=='L8')
    def work(stage):
        checks=[]
        for lane in ('binary','ranking','attack'):
            cohort=copy.deepcopy(plan['cohorts'][spec['cohorts'][lane]])
            data=dataset(workspace,plan,cohort)
            cohort['indices']=cohort['indices'][:64]
            if 'users' in cohort:cohort['users']=cohort['users'][:64]
            # Smoke uses a binary-tagged scorer to keep probe IDs within the
            # truncated cohort. Its data/prompts still match the selected lane.
            cohort['lane']='smoke'
            job=copy.deepcopy(plan['jobs'][spec['jobs'][lane]['BOTA64']])
            job['provenance']['users']=[]
            result=scoring.score_endpoint(workspace,scoring_cohort(cohort),job,data,plan['config'],lambda *a:None,read)
            checks.append({'lane':lane,'samples':len(cohort['indices']),'status':result['status'],
                           'audits':result['audits'],'full_cohort_gate':result.get('full_cohort_gate')})
            del data;gc.collect()
            if result['status']!='OK':raise RuntimeError(f'smoke invariance failed: {lane}')
        atomic_json(stage/'checks.json',checks)
        return {'status':'SMOKE_PASS','lanes':3,'samples_per_lane':64,'full_scoring_cache_written':False}
    return task(root,'smoke',contract,work)


def audit_output(stage,plan,workspace):
    jobs=list(plan['jobs'].values());missing=[j for j in jobs if j['status']!='READY']
    s25.s24.write_csv(stage/'endpoint_inventory.csv',jobs)
    atomic_json(stage/'unaffected_records.json',{'reuse':plan['reuse'],'parameters':plan['parameter_records']})
    atomic_json(stage/'missing_weights.json',missing)
    eta=estimate(workspace,plan);atomic_json(stage/'runtime_estimate.json',eta)
    atomic_text(stage/'REPORT.md','# Stage26 pre-run artifact audit\n\n'
        f'{len(plan["specs"])} evaluation groups; {len(jobs)} unique model/cohort jobs.\n'
        f'Endpoint statuses: {dict(Counter(j["status"] for j in jobs))}\n\n'
        'Part A: masked binary/ranking/attack rescoring of saved endpoints only.\n'
        'Part B: checksum verification and reuse of parameter/SVD/storage/latency records, not GPU reruns.\n'
        'Missing weights never trigger rebuilding. Historical lambda1 controls stay archived.\n'
        f'Estimated full wall time: {eta["full_wall_hours_range"]} hours.\n')
    return {'status':'AUDITED','groups':len(plan['specs']),'job_statuses':dict(Counter(j['status'] for j in jobs)),
            'unaffected_artifacts':len(plan['reuse']),'estimate':eta}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--mode',choices=('Preflight','Freeze','Smoke','Full','Analyze'),default='Preflight')
    parser.add_argument('--family',choices=('All',*FAMILIES),default='All')
    args=parser.parse_args();repo=args.repo.resolve();workspace=args.workspace.resolve();root=workspace/ROOT
    config=read(repo/CONFIG)
    if config['schema']!=SCHEMA or config['lambda']!=64 or config['optimizer_updates']!=0 or config['selection_performed'] or config['test_accessed']:
        raise ValueError('frozen scope changed')
    if config['audit']!=s25.load_config(repo)['audit']:raise ValueError('invariance tolerances changed')
    with s25.s24.s23.s20.lock(root):
        if (root/'protocol').exists():
            plan=verify(root/'protocol')['plan']
            if plan['config']!=config:raise ValueError('config changed after freeze')
            check(plan)
        else:plan=prepare(repo,workspace,config)
        contract=canonical_hash(plan)
        if args.mode=='Preflight':
            # Persistent audit report, but no protocol freeze or model execution.
            out=audit_output(root,plan,workspace)
        else:
            task(root,'protocol',contract,lambda stage:{'status':'FROZEN','plan':plan})
            audit=task(root,'audit',contract,lambda stage:audit_output(stage,plan,workspace))
            if args.mode=='Freeze':out=audit
            elif args.mode=='Smoke':out=smoke(root,workspace,plan)
            elif args.mode=='Analyze':out=analyze(root,plan,args.family)
            else:out=execute(root,workspace,plan,args.family)
        print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
