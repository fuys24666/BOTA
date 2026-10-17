"""Isolated historical-ML1M diagnostics at lambda64; never import the newer src tree.

The old position and relay schedules are distinct from Stage3's repeated-epoch
schedule. Preserve their source authority and canonical gradient accumulation.
Only the new transport feedback multiplier changes. No old file is edited.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path


def imports(root):
    sys.path.insert(0,str(root))
    import torch
    from src.bota_short_benchmark import ml1m_relay600_pilot as relay
    from src.bota_if import p3c5f_corrected_frequency_strata_audit as position
    # Refuse an accidental import of the newer BOTA package in this subprocess.
    if not Path(relay.__file__).resolve().is_relative_to(root):raise RuntimeError('legacy source namespace mismatch')
    return torch,relay,position


def authority(root):
    torch,r,p=imports(root)
    rc=r.load_frozen_config(root/'configs/bota_if/ml1m_relay600_pilot_v1.yaml')
    rb,_,registry,request=r.validate_authority(root,rc)
    users,schedule=r.freeze_schedule(root,rb,registry,request,rc)
    pc=p.load_frozen_config(root/'configs/bota_if/p3c5f_corrected_frequency_strata_audit_v1.yaml')
    predecessor=p.validate_predecessor(root,pc);_,pa=p.p3c0.validate_authority(root,pc)
    pr,pu,lineage=p._registry(root,pc,pa);order=p.p3b.freeze_schedule(pu,pa['p3b_config'])['order'][:21200]
    if len(order)!=21200:raise ValueError('position schedule mismatch')
    public={'relay':{'config':rc,'benchmark':rb,'schedule_hash':r.canonical_hash(schedule),'registry_hash':r.canonical_hash(registry)},
            'position':{'config':pc,'predecessor':predecessor,'authority':pa,'registry_hash':pr['registry_sha256'],
                        'schedule_hash':r.canonical_hash(order),'lineage':lineage}}
    public=json.loads(json.dumps(public,default=str))
    return (torch,r,p,rc,rb,registry,request,users,schedule,pc,pa,pr,pu,order),public


def finite_json(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_json(v) for v in value]
    return value


def canonical_windows(torch,r,model,data,order,users,selected,params,names,device,pad,engine,windows,guard,out,scale):
    """Independent short-window tangent resets; one carried canonical AdamW state.

    A relay is a single 0..600 window: theta/m/v are NOT reset at 200/400.
    Failed tangents stop updating, but canonical training and later windows survive.
    """
    opt=r.p1._optimizer(params,engine);starts={a for a,b in windows};ends={b for a,b in windows}
    def anchor():
        return {'parameters':r._cpu_parameters(names,params),'optimizer':r.p3c3._cpu_tree(opt.state_dict()),'rng':r.capture_rng()}
    anchors={0:anchor()};endpoints={};states=None;failure=None;boundaries={}
    for step,start in enumerate(range(0,len(order),16),1):
        if step-1 in starts:states=r._new_t2(params,len(selected));failure=None
        chosen=order[start:start+16];batch_users=[int(users[i]) for i in chosen]
        batch=r.move_batch(r.masked_batch(data,chosen,pad),device);losses=r.p1._sample_losses(model,batch)
        if not torch.isfinite(losses).all():raise RuntimeError('nonfinite canonical reference loss')
        per=[[] for _ in params];summed=[torch.zeros_like(v) for v in params]
        for i in range(len(chosen)):
            grads=torch.autograd.grad(losses[i],params,retain_graph=i+1<len(chosen))
            for j,g in enumerate(grads):per[j].append(g.detach());summed[j].add_(g.detach()/len(chosen))
        opt.zero_grad(set_to_none=True)
        for parameter,g in zip(params,summed):parameter.grad=g
        if states is not None and failure is None:
            coefficients=[torch.stack(v) for v in per]
            sources=[torch.stack([-g[[i for i,u in enumerate(batch_users) if u==target]].sum(0)/len(chosen)
                if target in batch_users else torch.zeros_like(parameter) for target in selected]) for g,parameter in zip(coefficients,params)]
            cfg=engine['optimizer'];b1,b2=map(float,cfg['betas'])
            try:
                for state,g,source,parameter in zip(states,coefficients,sources,params):
                    old=opt.state.get(parameter,{})
                    values=r.p2b.stable_adamw_tangent_step(parameter.detach(),parameter.grad.detach(),
                        old.get('exp_avg',torch.zeros_like(parameter)).detach(),old.get('exp_avg_sq',torch.zeros_like(parameter)).detach(),
                        state['theta'],state['m'],state['v'],source+scale*r.p2b.full_fisher_product(g,state['theta']),
                        step=step,lr=float(cfg['learning_rate']),beta1=b1,beta2=b2,eps=float(cfg['eps']),
                        weight_decay=float(cfg['weight_decay']),full_v=True)
                    state.update(zip(('theta','m','v'),values))
                if not all(torch.isfinite(v).all() for s in states for v in s.values()):failure={'step':step,'reason':'nonfinite_tangent'}
            except RuntimeError as e:
                if str(e) not in {'singular_adamw_tangent_at_zero_second_moment','nonfinite_adamw_second_moment_correction','nonfinite_stable_adamw_tangent'}:raise
                failure={'step':step,'reason':str(e)}
            del coefficients,sources
        guard.step(opt,'historical_canonical')
        if step in (200,400,600):
            boundaries[str(step)]={'canonical_sha256':r.tensor_tree_hash(r._cpu_parameters(names,params)),
                'optimizer_sha256':r.tensor_tree_hash(opt.state_dict()),'transport_failure':failure,
                'state_reset':False if windows==[(0,600)] else 'see_window_starts'}
        if step in ends:
            endpoints[step]={'original':r._cpu_parameters(names,params),'failure':failure,
                'delta':None if failure else {n:s['theta'].sum(0).detach().float().cpu().clone() for n,s in zip(names,states)}}
            states=None
        if step in starts:anchors[step]=anchor()
        if step==1 or step%25==0:
            r.atomic_json(out/'progress.json',{'step':step,'total':len(order)//16,'lambda':scale,'failure':failure,'test_accessed':False})
            print(f'[Stage17 legacy] step={step}/{len(order)//16} lambda={scale} failure={failure}',flush=True)
        del batch,losses,per,summed,grads
    return anchors,endpoints,boundaries


def run(root,out,kind,request_name,expected_authority):
    deps,public=authority(root);torch,r,p,rc,rb,registry,request,users,schedule,pc,pa,pr,pu,order=deps
    if r.canonical_hash(public)!=expected_authority:raise ValueError('historical authority changed')
    device=torch.device('cuda:0')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one CUDA GPU required')
    free,total=torch.cuda.mem_get_info()
    if free/2**30<10:raise RuntimeError('10 GiB free memory required')
    torch.cuda.set_per_process_memory_fraction(min(.85,14/(total/2**30)));torch.cuda.reset_peak_memory_stats()
    r.random.seed(42);r.np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    arm_rng=r.capture_rng();started=time.perf_counter()
    if kind=='relay':
        order=schedule['order'];selected=list(map(int,request['users']));windows=[(0,600)];engine=r.runner._engine(root,rb)
        def context(fixed=None):
            r.restore_rng(arm_rng)
            return r.runner._model_context(root,rb,{**registry,'order':order},device,fixed)
        model,names,params,bases,_,tok,data,loaded=context()
        fixed={n:v.detach().cpu().clone() for n,v in bases.items()};pad=tok.pad_token_id
        if list(map(int,loaded))!=users:raise ValueError('relay training lineage changed')
        guard=r.p1.StepBudget(1200);metric_cfg={**rc,'energy_aware_gate':rc['parameter_gate']};metric_base=metric_cfg
    else:
        q=next(q for q in pr['requests'] if q['name']==request_name);selected=q['users'];users=pu
        windows=list(zip(pc['protocol']['windows']['starts'],pc['protocol']['windows']['ends']))
        coordinate=torch.load(pa['source']/'bank/coordinate.pt',map_location='cpu',weights_only=True)
        fixed=coordinate['A'];names=coordinate['module_names'];engine=p.p3b.engine_config(pa['p3b_config'],pa['base'])
        runtime=p.load_config(root/pa['base']['base_config'],root)
        tok=p.T5Tokenizer.from_pretrained(runtime['paths']['model_dir'],local_files_only=True);pad=tok.pad_token_id
        data=p.JsonPromptDataset(root/pa['p3b_config']['experiment_root']/'data/train.json',tok)
        def context(fixed=fixed):
            r.restore_rng(arm_rng)
            return p.p1._fresh_runtime(root,{'original_checkpoint':pa['base']['original_checkpoint'],
                'coordinate':{'lora_rank':16,'fixed_a_seed':42,'lora_alpha':32}},device,fixed)
        model,current,params,_,_=context()
        if list(current)!=list(names):raise ValueError('position coordinate mismatch')
        guard=r.p1.StepBudget(2125);metric_cfg=engine;metric_base=pa['base']
    anchors,endpoints,boundaries=canonical_windows(torch,r,model,data,order,users,selected,params,names,device,pad,engine,windows,guard,out,64.)
    del model,params;gc.collect();torch.cuda.empty_cache();rows=[]
    anchor_hashes={str(k):r.tensor_tree_hash(v) for k,v in anchors.items()}
    for left,right in windows:
        values=context(fixed);model,_,params=values[:3]
        exact,trace=r.p3c3.run_local_masked(model,data,order,users,selected,params,names,device,pad,engine,
            anchors[left],left,right,guard,'legacy_exact_masked')
        del model,params,values;gc.collect();torch.cuda.empty_cache()
        endpoint=endpoints[right];delta=endpoint['delta'];actual={n:exact[n]-endpoint['original'][n] for n in names}
        metrics=None if delta is None else r.p2e.compare_named(actual,delta,names,metric_cfg,metric_base)
        row={'start':left,'end':right,'lambda':64,'status':'NUMERICAL_TRANSPORT_FAILURE' if delta is None else 'COMPLETED',
             'failure':endpoint['failure'],'parameter':finite_json(metrics),'exposures':sum(int(users[i]) in selected for i in order[left*16:right*16]),
             'anchor_immutable':trace['anchor_immutable'],'canonical_sha256':r.tensor_tree_hash(endpoint['original'])}
        if kind=='relay':
            for name,values in [('Original',endpoint['original']),('Exact-Masked',exact)]+([] if delta is None else [('BOTA64',{n:endpoint['original'][n]+delta[n] for n in names})]):
                r.runner._save_fixed_ab(out/'adapters'/name,names,values,fixed,'Stage17-legacy-'+name)
            dataset=r.JsonPromptDataset(root/rb['source']['development_json'],tok)
            _,dev,_=r.reconstruct_authoritative_rows(root/rb['source']['raw_data']);devusers=[int(v.authoritative_user_id) for v in dev]
            ids=[i for i,u in enumerate(devusers) if u in selected];pred={}
            for name in ('Original','Exact-Masked','BOTA64'):
                d=out/'adapters'/name
                if d.exists():
                    v=torch.load(d/'adapter_model.pt',map_location='cpu',weights_only=True)
                    pred[name]=r._predict_values(root,rb,fixed,v['B'],dataset,list(range(len(dataset))),device)
            if any(not r.np.isfinite(pred[k]['probability']).all() for k in ('Original','Exact-Masked')):
                raise RuntimeError('nonfinite historical behavior reference')
            r.atomic_json(out/'predictions.json',finite_json(pred))
            row['behavior']=[r.evaluation._metrics(name,v,pred['Exact-Masked'],pred['Original'],ids)[0]
                             for name,v in pred.items() if r.np.isfinite(v['probability']).all()]
            row['nonfinite_prediction_methods']=[name for name,v in pred.items() if not r.np.isfinite(v['probability']).all()]
        rows.append(row)
    if anchor_hashes!={str(k):r.tensor_tree_hash(v) for k,v in anchors.items()}:raise RuntimeError('historical anchor mutated')
    peak=torch.cuda.max_memory_reserved()/2**30
    if peak>14 or guard.calls!=(1200 if kind=='relay' else 2125):raise RuntimeError('resource/step gate failed')
    # Revalidate the complete historical authority after inference/training.
    if r.canonical_hash(authority(root)[1])!=expected_authority:raise RuntimeError('historical authority mutated')
    result={'status':'COMPLETED' if all(row['status']=='COMPLETED' and not row.get('nonfinite_prediction_methods') for row in rows) else 'COMPLETED_WITH_UNAVAILABLE_OUTCOMES',
        'kind':kind,'request':request_name,'seed':42,'lambda':64,'rows':finite_json(rows),'boundaries':boundaries,
        'physical_optimizer_steps':guard.calls,'wall_seconds':time.perf_counter()-started,'peak_reserved_gib':peak,
        'historical_protocol_not_stage3':True,'test_accessed':False}
    r.atomic_json(out/'legacy_result.json',result)


def smoke(root,out):
    """Six training steps total; compare the old and new canonical/T2 interfaces."""
    deps,_=authority(root);torch,r,p,rc,rb,registry,request,users,schedule,*_=deps
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.85)
    def context(fixed=None):
        r.random.seed(42);r.np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        return r.runner._model_context(root,rb,registry,device,fixed)
    order=schedule['order'][:32];selected=list(dict.fromkeys([users[order[0]],*request['users']]))[:8]
    engine=r.runner._engine(root,rb);model,names,params,bases,_,tok,data,_=context()
    fixed={n:v.detach().cpu().clone() for n,v in bases.items()}
    anchors,expected,_=p.p3c4.run_canonical_windows(model,data,order,users,selected,params,names,device,tok.pad_token_id,
        engine,[0],[2],r.p1.StepBudget(2))
    del model,params;gc.collect();torch.cuda.empty_cache()
    for scale in (1.,64.):
        model,_,params,_,_,tok,data,_=context(fixed)
        a,e,_=canonical_windows(torch,r,model,data,order,users,selected,params,names,device,tok.pad_token_id,
            engine,[(0,2)],r.p1.StepBudget(2),out,scale)
        assert all(torch.equal(expected[2]['parameters'][n],e[2]['original'][n]) for n in names)
        assert e[2]['failure'] is None
        if scale==1:
            for n,s in zip(names,expected[2]['transport'][p.p3b.T2]):
                assert torch.equal(s['theta'].sum(0),e[2]['delta'][n])
        del model,params;gc.collect();torch.cuda.empty_cache()
    print('STAGE17_LEGACY_GPU_SMOKE_PASS: old canonical/T2 reproduced; lambda64 finite; six steps; no Test',flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--workspace',type=Path,required=True)
    a.add_argument('--mode',choices=['Preflight','Full','Smoke'],required=True);a.add_argument('--out',type=Path)
    a.add_argument('--kind',choices=['relay','position']);a.add_argument('--request',default='L8');a.add_argument('--authority')
    args=a.parse_args();root=args.workspace.resolve();os.chdir(root)
    if args.mode=='Preflight':
        deps,public=authority(root);print(json.dumps({'status':'READY','authority_sha256':deps[1].canonical_hash(public),'test_accessed':False}))
    elif args.mode=='Smoke':smoke(root,args.out.resolve())
    else:run(root,args.out.resolve(),args.kind,args.request,args.authority)


if __name__=='__main__':main()
