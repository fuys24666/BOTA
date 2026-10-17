"""Fresh lambda64 transport: scale Fisher feedback, never the deletion source."""
from __future__ import annotations
import gc
import time
import numpy as np
import torch
from src.paper_if_a2.common import seed_everything
from . import ali_stage8_bota_response_sweep_v1 as s8
from . import ali_stage14_lambda_search_v1 as s14
from . import ali_stage17_common as c

ARMS={'FS0':(0.,True),'FS1':(1.,True),'FS64':(64.,True),'FM0':(0.,False),'FM64':(64.,False)}

def advance(state,rows,sources,parameters,optimizer,step,engine,arm):
    scale,full_v=ARMS[arm];opt=engine['optimizer'];b1,b2=map(float,opt['betas'])
    for current,p,g,source in zip(state,parameters,rows,sources):
        dgradient=source if scale==0 else source+scale*s8.p2b.full_fisher_product(g,current['theta'])
        old=optimizer.state.get(p,{})
        theta,dm,dv=s8.p2b.stable_adamw_tangent_step(p.detach(),p.grad.detach(),
            old.get('exp_avg',torch.zeros_like(p)).detach(),old.get('exp_avg_sq',torch.zeros_like(p)).detach(),
            current['theta'],current['m'],current['v'],dgradient,step=step,lr=float(opt['learning_rate']),
            beta1=b1,beta2=b2,eps=float(opt['eps']),weight_decay=float(opt['weight_decay']),full_v=full_v)
        current.update(theta=theta,m=dm,v=dv)

def run(model,data,order,user_ids,selected,parameters,names,device,pad,engine,arms,progress):
    if len(order)%16 or not selected or len(set(selected))!=len(selected):raise ValueError('invalid schedule/users')
    opt=s8.p1._optimizer(parameters,engine);states={a:s8.stage1._state(parameters,len(selected)) for a in arms}
    budget=s8.StepBudget(len(order)//16);failures={};begun=time.perf_counter()
    for step,start in enumerate(range(0,len(order),16),1):
        chosen=list(order[start:start+16]);users=[int(user_ids[i]) for i in chosen]
        batch=s8.move_batch(s8.masked_batch(data,chosen,pad),device);losses=s8.p1._sample_losses(model,batch)
        if not torch.isfinite(losses).all():raise RuntimeError('canonical loss nonfinite; infrastructure/reference failure')
        per=[[] for _ in parameters]
        for sample in range(len(chosen)):
            gradients=torch.autograd.grad(losses[sample],parameters,retain_graph=True)
            for j,g in enumerate(gradients):per[j].append(g.detach())
        opt.zero_grad(set_to_none=True);loss=losses.sum()/len(chosen);loss.backward();rows=[torch.stack(r) for r in per]
        sources=[]
        for g,p in zip(rows,parameters):
            sources.append(torch.stack([-g[[i for i,u in enumerate(users) if u==target]].sum(0)/len(chosen)
                if target in users else torch.zeros_like(p) for target in selected]))
        for arm in list(states):
            try:
                advance(states[arm],rows,sources,parameters,opt,step,engine,arm)
                finite=all(torch.isfinite(v).all().item() for state in states[arm] for v in state.values())
                if not finite:failures[arm]={'status':'NUMERICAL_TRANSPORT_FAILURE','first_failed_step':step,'reason':'nonfinite_tangent'}
            except RuntimeError as error:
                if str(error) not in s14.NUMERICAL_ERRORS:raise
                failures[arm]={'status':'NUMERICAL_TRANSPORT_FAILURE','first_failed_step':step,'reason':str(error)}
            if arm in failures:del states[arm]
        budget.step(opt,'shared_canonical_reference')
        if step==1 or step%10==0 or step==len(order)//16:
            c.atomic_json(progress,{'step':step,'total':len(order)//16,'failures':failures,'test_accessed':False})
            print(f'[Stage17] transport {step}/{len(order)//16} failures={list(failures)}',flush=True)
        del batch,losses,loss,gradients,per,rows,sources
    canonical={name:p.detach().float().cpu().clone() for name,p in zip(names,parameters)}
    vectors={a:torch.cat([v['theta'].detach().cpu().reshape(len(selected),-1).float() for v in state],dim=1)
             for a,state in states.items()}
    return canonical,vectors,failures,{'optimizer_steps':budget.calls,'transport_seconds':time.perf_counter()-begun,
        'optimizer_hash':s8.tensor_tree_hash(opt.state_dict()),'rng_hash':s8.tensor_tree_hash(s8.capture_rng())}

def unflatten(flat,canonical,names):
    result={};offset=0
    for name in names:
        n=canonical[name].numel();result[name]=canonical[name]+flat[offset:offset+n].reshape(canonical[name].shape);offset+=n
    if offset!=flat.numel():raise ValueError('vector dimension mismatch')
    return result

def compose(bank,arm,users):
    if len(users)!=len(set(users)):raise ValueError('deduplicate cumulative requests before composition')
    ids=[bank['users'].index(u) for u in users]
    return unflatten(bank['vectors'][arm][ids].sum(0),bank['canonical'],bank['names'])

def alignment(candidate,reference,canonical):
    names=list(canonical)
    actual=torch.cat([(reference[k].float()-canonical[k]).reshape(-1) for k in names])
    predicted=torch.cat([(candidate[k].float()-canonical[k]).reshape(-1) for k in names])
    m=s8.p2b.vector_metrics(actual,predicted)
    if not(m['finite'] and m['cosine']>=.9999 and m['relative_l2_error']<=.01 and m['maximum_absolute_error']<=.001):
        raise RuntimeError(f'control reconstruction failed: {m}')
    return m

def build_bank(w,contract,seed,kind,chunk=None):
    ctx=contract['contexts'][str(seed)];primary=kind=='primary';long=kind=='long';strong=kind=='strong';timing_only=kind=='timing'
    if kind not in ('primary','long','strong','storage','timing'):raise ValueError('unknown bank kind')
    selected=ctx['primary_users'] if primary or long or timing_only else ctx['stronger_users'] if strong else ctx['bank_users'][chunk*32:(chunk+1)*32]
    arms=list(ARMS) if primary else ['FS1','FS64'] if long or strong else ['FS64']
    key=f'bank_{kind}_s{seed}'+(f'_c{chunk}' if chunk is not None else '')
    def work(dest):
        device=c.gpu();seed_everything(seed);benchmark=ctx['benchmark'];registry=ctx['registry']
        original=next(e for e in ctx['endpoints'] if e['method']=='Original' and e['scenario']=='L8')
        anchor=c.torch_read(w/original['path']/'adapter_model.pt')
        model=None
        try:
            model,names,parameters,bases,_,tokenizer,data,train_users=s8.runner._model_context(w,benchmark,registry,device,anchor['A'])
            order=c.s3.build_long_schedule(registry['order'],seed) if long else registry['order']
            canonical,vectors,failures,timing=run(model,data,order,train_users,selected,parameters,names,device,
                tokenizer.pad_token_id,s8.runner._engine(w,benchmark),arms,dest/'progress.json')
            bases={k:v.detach().cpu().clone() for k,v in bases.items()}
            if not c.s2._same_tensor_map(bases,anchor['A']):raise RuntimeError('LoRA A mismatch')
            expected=c.torch_read(w/ctx['long'][0]['original']['path']/'adapter_model.pt')['B'] if long else anchor['B']
            if any(not torch.equal(canonical[k],expected[k]) for k in names):raise RuntimeError('canonical endpoint mismatch')
            bank={'users':selected,'names':list(names),'canonical':canonical,'bases':bases,'vectors':vectors,'failures':failures,'lambda':64}
            checks={}
            if primary or timing_only:
                for sid in c.s16.SCENARIOS:
                    for arm,method in ([('FS0','BOTA0'),('FS1','BOTA1'),('FS64','BOTA64')] if primary else [('FS64','BOTA64')]):
                        ref=next(e for e in ctx['endpoints'] if e['method']==method and e['scenario']==sid)
                        if arm not in vectors:raise RuntimeError('short-window reproduction unexpectedly failed')
                        checks[sid+arm]=alignment(compose(bank,arm,ref['users']),c.torch_read(w/ref['path']/'adapter_model.pt')['B'],canonical)
            elif long or strong:
                for row in (ctx['long'] if long else ctx['stronger']):
                    if 'FS1' not in vectors:raise RuntimeError('lambda1 reproduction control failed')
                    checks[row['id']]=alignment(compose(bank,'FS1',row['users']),c.torch_read(w/row['old']['path']/'adapter_model.pt')['B'],canonical)
            c.atomic_torch_save(dest/'bank.pt',bank)
            return {'status':'COMPLETED' if not failures else 'COMPLETED_WITH_NUMERICAL_FAILURES',
                'kind':kind,'seed':seed,'users':len(selected),'arms':arms,'failures':failures,'timing':timing,'reproduction':checks}
        finally:
            if model is not None:del model
            gc.collect();torch.cuda.empty_cache()
    c.task(w,contract,key,work);return key

def materialize(w,contract,bank_key,seed,sid,users,arm='FS64'):
    key='adapter_'+c.canonical_hash([bank_key,sid,users,arm])[:16]
    def work(dest):
        bank=c.torch_read(c.path(w,bank_key)/'bank.pt')
        if arm in bank['failures']:return {'status':'UNAVAILABLE_NUMERICAL_TRANSPORT','failure':bank['failures'][arm]}
        candidate=compose(bank,arm,users)
        s8.runner._save_fixed_ab(dest/'adapter',bank['names'],candidate,bank['bases'],'Stage17-'+arm)
        return {'status':'COMPLETED','arm':arm,'seed':seed,'scenario':sid,'requested_users':len(users)}
    result=c.task(w,contract,key,work)
    return None if result['status']!='COMPLETED' else c.endpoint(w,c.path(w,key)/'adapter',arm,seed,sid,users)
