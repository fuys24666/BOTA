"""Lambda64 banks, all compression screens, spectrum, behavior and serving costs."""
from __future__ import annotations
import gc
import math
import time
import torch
import numpy as np
from . import ali_stage17_common as c
from . import ali_stage17_transport as t
from . import ali_stage11_svd_v1 as svd
from . import online_latency_v1 as latency

def merge(w,contract,seed):
    key=f'storage_bank_s{seed}'
    def work(dest):
        banks=[c.torch_read(c.path(w,f'bank_storage_s{seed}_c{i}')/'bank.pt') for i in range(8)]
        failures=[(i,b['failures']) for i,b in enumerate(banks) if b['failures']]
        if failures:return {'status':'UNAVAILABLE_NUMERICAL_TRANSPORT','failures':failures}
        first=banks[0];names=first['names'];canonical=first['canonical']
        if any(b['names']!=names or any(not torch.equal(b['canonical'][n],canonical[n]) for n in names) for b in banks):raise ValueError('bank chunks do not share canonical')
        bank={**first,'users':[u for b in banks for u in b['users']],
              'vectors':{'FS64':torch.cat([b['vectors']['FS64'] for b in banks])}}
        if bank['users']!=contract['contexts'][str(seed)]['bank_users']:raise ValueError('bank users/order changed')
        checks={}
        for sid in c.s16.SCENARIOS:
            ref=next(e for e in contract['contexts'][str(seed)]['endpoints'] if e['method']=='BOTA64' and e['scenario']==sid)
            checks[sid]=t.alignment(t.compose(bank,'FS64',ref['users']),c.torch_read(w/ref['path']/'adapter_model.pt')['B'],canonical)
        c.atomic_torch_save(dest/'bank.pt',bank)
        timing=sum(c.read(c.path(w,f'bank_storage_s{seed}_c{i}')/'result.json')['timing']['transport_seconds'] for i in range(8))
        return {'status':'COMPLETED','users':256,'lambda':64,'reproduction':checks,'chunk_transport_seconds_sum':timing,
                'physical_optimizer_steps':1600,'dimension':bank['vectors']['FS64'].shape[1]}
    c.task(w,contract,key,work);return key

def screen(w,contract,seed):
    bank_key=f'storage_bank_s{seed}';key=f'storage_screen_s{seed}'
    def work(dest):
        if c.verify(c.path(w,bank_key))['status']!='COMPLETED':return {'status':'UNAVAILABLE_NUMERICAL_TRANSPORT'}
        bank=c.torch_read(c.path(w,bank_key)/'bank.pt');matrix=bank['vectors']['FS64'];ctx=contract['contexts'][str(seed)]
        names=bank['names'];original=c.s6._flatten_map(bank['canonical'],names);queries={}
        for sid in c.s16.SCENARIOS:
            e=next(e for e in ctx['endpoints'] if e['method']=='Exact-Masked' and e['scenario']==sid)
            exact=c.s6._flatten_map(c.torch_read(w/e['path']/'adapter_model.pt')['B'],names)-original
            queries[sid]={'rows':[bank['users'].index(u) for u in e['users']],'exact':exact,'users':e['users']}
        result=[];diagnostics=None;device=c.gpu()
        for k in c.PROTOCOL['bank_sizes']:
            x=matrix[:k].clone();gram=svd.accumulate_gram(x,16384,device)
            spec,u=svd.spectrum(gram);rank=min(64,k)
            xd=x.to(device);ud=u[:,:rank].float().to(device)
            low={'u':ud.cpu(),'basis':(ud.T@xd).cpu()}
            _,ix=torch.topk(xd.abs(),max(1,math.ceil(x.shape[1]/16)),dim=1,sorted=True)
            top={'indices':ix.to(torch.int32).cpu(),'values':torch.gather(xd,1,ix).half().cpu()}
            del xd,ud,ix;torch.cuda.empty_cache()
            if k==256:
                direction,normal,_,pairs=svd.direction_diagnostics(gram);normalized,_=svd.spectrum(normal)
                diagnostic_queries={sid:svd.query_errors(x,u,q['rows'],q['exact'],[4,8,16,32,64,128,256],16384) for sid,q in queries.items()}
                diagnostics={'raw':spec,'row_normalized':normalized,'directions':direction,'query_reconstruction':diagnostic_queries,
                    'interpretation':'Measured linear subspace/queries only; not an impossibility theorem for every compression method.'}
                c.atomic_json(dest/'pairwise_cosines.json',pairs)
            fpbytes=None
            for rep in c.s6._representation_names():
                print(f'[Stage17] storage seed={seed} K={k} {rep}',flush=True)
                begun=time.perf_counter();encoded=c.s6._encode(rep,x,low,top);encoding=time.perf_counter()-begun
                size,serialization=c.s6._measure_artifact(dest,f'k{k}_{rep}',encoded)
                if rep=='FP32':fpbytes=size
                for sid,q in queries.items():
                    begun=time.perf_counter();delta=c.s6._compose(encoded,q['rows']);seconds=time.perf_counter()-begun
                    raw=x[q['rows']].sum(0);den=float(q['exact'].double().norm())
                    finite=bool(torch.isfinite(delta).all())
                    residual=float((delta-q['exact']).double().norm())/den if den and finite else None
                    base=float((raw.double()-q['exact'].double()).norm())/den if den else None
                    increase=residual/base-1 if base and residual is not None else None
                    result.append({'seed':seed,'bank_size':k,'representation':rep,'scenario':sid,'artifact_bytes':size,
                        'compression_ratio':fpbytes/size,'parameter_residual':residual,'fp32_parameter_residual':base,
                        'relative_residual_increase':increase,'encode_seconds':encoding,'serialize_seconds':serialization,
                        'composition_seconds':seconds,'finite':finite,
                        'parameter_gate':bool(finite and fpbytes/size>=16 and increase is not None and increase<=.10)})
                    if k==256:
                        values=t.unflatten(delta,bank['canonical'],names)
                        if all(torch.isfinite(v).all() for v in values.values()):
                            c.s16.s15.runner._save_fixed_ab(dest/'adapters'/rep/sid/'adapter',names,values,bank['bases'],'Stage17-FS64-'+rep)
                del encoded
            del x,gram,u,low,top;gc.collect();torch.cuda.empty_cache()
        # Empty transient measurement directory must not appear in a sealed inventory.
        if (dest/'measure').exists() and not any((dest/'measure').iterdir()):(dest/'measure').rmdir()
        c.atomic_json(dest/'spectrum.json',diagnostics)
        c.s16.s10._write_csv(dest/'screen.csv',result)
        return {'status':'COMPLETED','seed':seed,'lambda':64,'rows':result,'screen_only':True,'deployment_claim_passed':False}
    return c.task(w,contract,key,work)

def time_serving(w,contract,seed):
    key=f'latency_s{seed}'
    def work(dest):
        if c.verify(c.path(w,f'storage_bank_s{seed}'))['status']!='COMPLETED':return {'status':'UNAVAILABLE_NUMERICAL_TRANSPORT'}
        bank=c.torch_read(c.path(w,f'storage_bank_s{seed}')/'bank.pt');x=bank['vectors']['FS64'];cfg=c.PROTOCOL['latency'];rows=[]
        ctx=contract['contexts'][str(seed)]
        # Resident CPU bank; each serialization is reloaded and tensor-checked.
        for size in cfg['bank_sizes']:
            resident=x[:size].clone();user_index={u:i for i,u in enumerate(bank['users'][:size])}
            for sid in c.s16.SCENARIOS:
                request=next(e['users'] for e in ctx['endpoints'] if e['method']=='BOTA64' and e['scenario']==sid)
                slots=[user_index[u] for u in request]
                for i in range(-cfg['warmup'],cfg['repeats']):
                    start=time.perf_counter_ns();delta=resident[slots].sum(0)
                    values=t.unflatten(delta,bank['canonical'],bank['names'])
                    state={'A':{k:v.clone() for k,v in bank['bases'].items()},'B':{k:v.clone() for k,v in values.items()},
                        'schema':'bota-short-fixed-ab-v1','rank':16,'alpha':32,'method_id':'BOTA-T2-Short'}
                    before=(time.perf_counter_ns()-start)/1e6
                    # Helper removes only its own just-created iteration publication in dest.
                    pub,reload=latency._publish_reload(dest,f'k{size}_{sid}_{i+cfg["warmup"]}',state,bank['names'])
                    if i>=0:rows.append({'bank_users':size,'scenario':sid,'iteration':i,'compose_materialize_ms':before,
                        'publication_ms':pub,'reload_verify_ms':reload,'total_ms':before+pub+reload})
                    if i>=0 and i%100==0:print(f'[Stage17] latency seed={seed} K={size} {sid} {i}/{cfg["repeats"]}',flush=True)
            del resident
        summaries=[]
        build=c.read(c.path(w,f'storage_bank_s{seed}')/'result.json')['chunk_transport_seconds_sum']
        primary_build=c.read(c.path(w,f'bank_timing_s{seed}')/'result.json')['timing']['transport_seconds']
        historical=c.read(w/'outputs/ali_stage5_sequential_v1/runs'/f'ali_stage5_sequential_seed{seed}_v1'/'timing.json')
        for size in cfg['bank_sizes']:
            for sid in c.s16.SCENARIOS:
                values=[r['total_ms'] for r in rows if r['bank_users']==size and r['scenario']==sid]
                union=next(r['union_hash'] for r in ctx['sequence_cells'] if r['scenario']==sid and r['request_size']==8)
                exact_seconds=historical['exact_replay_seconds'][union]
                offline=build if size==256 else primary_build
                saving=exact_seconds-float(np.mean(values))/1000
                summaries.append({'bank_users':size,'scenario':sid,'p50_ms':float(np.percentile(values,50)),
                    'p95_ms':float(np.percentile(values,95)),'mean_ms':float(np.mean(values)),'iterations':len(values),
                    'fp32_payload_bytes':size*x.shape[1]*4,'million_user_projection_bytes':1000000*x.shape[1]*4,
                    'bank_build_seconds':offline,
                    'build_cost_note':'Dedicated FS64 transport; 12-user single pass / 256-user eight 32-user chunks; excludes model initialization and evaluation',
                    'historical_exact_replay_seconds':exact_seconds,
                    'amortization_break_even_requests':math.ceil(offline/saving) if saving>0 else None,
                    'break_even_note':'Cost-only estimate using frozen Stage5 exact timing; assumes repeated covered requests, unchanged bank, no refresh/fallback. Not an end-to-end production break-even.',
                    'production_latency_claim':False,'fallback_included':False})
        c.s16.s10._write_csv(dest/'latency_samples.csv',rows)
        return {'status':'COMPLETED','seed':seed,'lambda':64,'summary':summaries,'online_optimizer_steps':0}
    return c.task(w,contract,key,work)
