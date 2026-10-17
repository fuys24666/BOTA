"""Fixed-lambda MovieLens-1M refresh. Full is resumable; no search or FinalTest."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from . import ali_stage17_common as c
from . import ali_stage17_transport as t
from . import ali_stage17_evaluation as ev
from . import ali_stage17_storage as storage

def original(ctx,sid='L8'):
    return dict(next(e for e in ctx['endpoints'] if e['method']=='Original' and e['scenario']==sid))

def generated(w,contract,bank,seed,sid,users,arm='FS64',method='BOTA64'):
    e=t.materialize(w,contract,bank,seed,sid,users,arm)
    if e:e['method']=method
    return e

def run_phase(w,contract,phase,seeds):
    for seed in seeds:
        ctx=contract['contexts'][str(seed)]
        if phase=='Primary':
            bank=t.build_bank(w,contract,seed,'primary')
            for sid in c.s16.SCENARIOS:
                entries=[dict(e) for e in ctx['endpoints'] if e['scenario']==sid];request=entries[0]['users']
                for arm in ('FM0','FM64'):
                    e=generated(w,contract,bank,seed,sid,request,arm,arm)
                    if e:entries.append(e)
                ev.evaluate(w,contract,'primary',seed,sid,request,entries,True)
        elif phase=='Stronger':
            if seed!=42:continue
            bank=t.build_bank(w,contract,seed,'strong')
            for row in ctx['stronger']:
                rid=row['id'];entries=[original(ctx),row['exact'],row['old']]
                e=generated(w,contract,bank,seed,rid,row['users'])
                if e:entries.append(e)
                ev.evaluate(w,contract,'stronger',seed,rid,row['users'],entries,rid=='high_k4')
        elif phase=='Sequential':
            bank=t.build_bank(w,contract,seed,'primary')
            for row in ctx['sequential']:
                entries=[original(ctx),row['exact'],row['old']]
                e=generated(w,contract,bank,seed,row['union_hash'],row['users'])
                if e:entries.append(e)
                ev.evaluate(w,contract,'sequential',seed,row['union_hash'],row['users'],entries,False)
        elif phase=='Horizon':
            bank=t.build_bank(w,contract,seed,'long')
            for row in ctx['long']:
                entries=[row['original'],row['exact'],row['old']]
                e=generated(w,contract,bank,seed,row['id'],row['users'])
                if e:entries.append(e)
                ev.evaluate(w,contract,'horizon',seed,row['id'],row['users'],entries,False,
                            c.s3.build_long_schedule(ctx['registry']['order'],seed))
        elif phase in ('Storage','StorageBehavior','Latency'):
            for chunk in range(8):t.build_bank(w,contract,seed,'storage',chunk)
            storage.merge(w,contract,seed)
            if phase=='Latency':
                t.build_bank(w,contract,seed,'timing');storage.time_serving(w,contract,seed)
            else:
                result=storage.screen(w,contract,seed)
                if phase=='StorageBehavior':
                    for rep in c.s6._representation_names():
                        for sid in c.s16.SCENARIOS:
                            refs=[dict(e) for e in ctx['endpoints'] if e['scenario']==sid and e['method'] in ('Original','Exact-Masked','BOTA64')]
                            for e in refs:
                                if e['method']=='BOTA64':e['method']='Uncompressed-FS64'
                            request=refs[0]['users'];p=c.path(w,f'storage_screen_s{seed}')/'adapters'/rep/sid/'adapter'
                            if p.exists():refs.append(c.endpoint(w,p,'BOTA64',seed,sid,request))
                            ev.evaluate(w,contract,'compression',seed,rep+'_'+sid,request,refs,False,with_attack=False)
        elif phase in ('LegacyRelay','LegacyPosition'):
            if seed!=42:continue
            kind='relay' if phase=='LegacyRelay' else 'position'
            for request in (['L8'] if kind=='relay' else ['low_frequency','middle_frequency','high_frequency']):
                key=f'legacy_{kind}_{request}'
                def work(dest):
                    subprocess.run([sys.executable,str(Path(contract['repo'])/'scripts/bota_if/ali_stage17_legacy_worker.py'),
                        '--workspace',str(w),'--mode','Full','--kind',kind,'--request',request,
                        '--authority',contract['legacy']['authority_sha256'],'--out',str(dest)],check=True)
                    return c.read(dest/'legacy_result.json')
                c.task(w,contract,key,work)
        else:raise ValueError(phase)

def expected(contract):
    result=[]
    for seed in c.SEEDS:
        for family in ('primary','horizon'):
            result.extend(f'eval_{family}_s{seed}_'+c.canonical_hash(sid)[:12] for sid in c.s16.SCENARIOS)
        result.extend(f'eval_sequential_s{seed}_'+c.canonical_hash(r['union_hash'])[:12] for r in contract['contexts'][str(seed)]['sequential'])
        result.extend(f'eval_compression_s{seed}_'+c.canonical_hash(rep+'_'+sid)[:12] for rep in c.s6._representation_names() for sid in c.s16.SCENARIOS)
        result.extend([f'storage_screen_s{seed}',f'latency_s{seed}'])
    result.extend('eval_stronger_s42_'+c.canonical_hash(r['id'])[:12] for r in contract['contexts']['42']['stronger'])
    result.extend(['legacy_relay_L8',*[f'legacy_position_{s}_frequency' for s in ('low','middle','high')]])
    if len(set(result))!=len(result):raise ValueError('duplicate job keys')
    return result

def summarize(w,contract):
    keys=expected(contract);missing=[k for k in keys if not c.path(w,k).exists()]
    if missing:return {'status':'PARTIAL','complete':len(keys)-len(missing),'required':len(keys),'missing':missing,'test_accessed':False}
    def work(dest):
        results=[c.verify(c.path(w,k)) for k in keys];rows=[];ranking=[];attack=[];parameters=[];legacy=[]
        for r in results:
            for metric in r.get('behavior',{}).get('rows',[]):
                rows.append({'family':r['family'],'seed':r['seed'],'request':r['request'],**metric})
            for method,rank in r.get('ranking',{}).items():
                ranking.extend({'family':r['family'],'seed':r['seed'],'request':r['request'],'method':method,**g} for g in rank.get('aggregates',[]))
            if r.get('attack') is not None:attack.append({'family':r['family'],'seed':r['seed'],'request':r['request'],**r['attack']})
            for metric in r.get('parameter',[]):parameters.append({'family':r['family'],'seed':r['seed'],'request':r['request'],**metric})
            if r.get('historical_protocol_not_stage3'):
                legacy.extend({'kind':r['kind'],'request':r['request'],'seed':r['seed'],**x} for x in r['rows'])
        for end in (200,575,950,1325):
            hashes={r['canonical_sha256'] for r in legacy if r['kind']=='position' and r['end']==end}
            if len(hashes)!=1:raise RuntimeError('historical canonical endpoints differ across frequency strata')
        sequences=[]
        bykey={k:r for k,r in zip(keys,results)}
        for seed in c.SEEDS:
            for row in contract['contexts'][str(seed)]['sequence_cells']:
                value=bykey[f'eval_sequential_s{seed}_'+c.canonical_hash(row['union_hash'])[:12]]
                b=next((r for r in value['behavior'].get('rows',[]) if r['method']=='BOTA64'),{})
                sequences.append({'seed':seed,**row,'result':b,'status':value['status']})
        c.atomic_json(dest/'all_results.json',results);c.atomic_json(dest/'sequential_all_60_cells.json',sequences)
        c.atomic_json(dest/'attacks.json',attack)
        c.s16.s10._write_csv(dest/'behavior.csv',rows);c.s16.s10._write_csv(dest/'ranking.csv',ranking)
        c.s16.s10._write_csv(dest/'parameter.csv',parameters);c.s16.s10._write_csv(dest/'historical_windows_relay.csv',legacy)
        c.s16.s10._write_csv(dest/'primary_ablation.csv',[r for r in rows if r['family']=='primary' and r['method'] in ('BOTA0','BOTA1','BOTA64','FM0','FM64')])
        # All twelve compression candidates must meet the gate across all seeds/requests, not just their best row.
        compression=[]
        for rep in c.s6._representation_names():
            rr=[x for r in results for x in r.get('rows',[]) if x.get('bank_size')==256 and x['representation']==rep]
            compression.append({'representation':rep,'parameter_gate_all_six':len(rr)==6 and all(x['parameter_gate'] for x in rr),
                'minimum_compression':min((x['compression_ratio'] for x in rr),default=None),
                'worst_relative_residual_increase':max((x['relative_residual_increase'] for x in rr if x['relative_residual_increase'] is not None),default=None),
                'deployment_claim':False})
        failures=[k for k,r in zip(keys,results) if r['status']!='COMPLETED']
        c.atomic_text(dest/'report.md','# Stage 17: MovieLens-1M lambda=64 refresh\n\n'
            f'All {len(keys)} required analysis cells recorded; {len(failures)} contain explicit unavailable/numerical outcomes.\n\n'
            'No new FinalTest access, no lambda selection. See behavior.csv, ranking.csv, attacks.json, primary_ablation.csv and all_results.json.\n'
            'Sequential reports preserve all 60 order/size cells, including duplicate final unions.\n'
            'Historical position/600-step relay are refreshed in isolated historical protocols, seed42 only; do not pool with Stage3.\n'
            'No calibrated fallback, adaptive request policy, attack certificate, or population deployment guarantee is established.\n')
        return {'status':'COMPLETED' if not failures else 'COMPLETED_WITH_FAILURE_OUTCOMES','required_cells':len(keys),
            'failed_or_unavailable_cells':failures,'compression':compression,'lambda':64,
            'manuscript_numbers_require_manual_target_alignment':True,'all_historical_experiments_relabelled':False}
    return c.task(w,contract,'summary',work)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--mode',choices=('Preflight','Seal','Full','Analyze'),default='Preflight')
    p.add_argument('--phase',choices=('All',*c.PHASES),default='All');p.add_argument('--seed',choices=('All','41','42','43'),default='All')
    a=p.parse_args();repo=a.repo.resolve();w=a.workspace.resolve();old=Path.cwd()
    with c.s15.s14.execution_lock(c.root(w)):
        try:
            if a.mode in ('Preflight','Seal'):
                contract=c.prepare(repo,w)
                if a.mode=='Seal':c.seal(w,contract)
                out={'status':'SEALED' if a.mode=='Seal' else 'PREFLIGHT_PASS','analysis_cells':len(expected(contract)),
                     'lambda':64,'test_loader_built':False,'test_accessed':False}
            else:
                contract=c.load(w);os.chdir(w)
                if a.mode=='Full':
                    for phase in c.PHASES if a.phase=='All' else (a.phase,):
                        run_phase(w,contract,phase,c.SEEDS if a.seed=='All' else [int(a.seed)])
                out=summarize(w,contract)
            print(json.dumps(out,indent=2))
        finally:os.chdir(old)

if __name__=='__main__':main()
