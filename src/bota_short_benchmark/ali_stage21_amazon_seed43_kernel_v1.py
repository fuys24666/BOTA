"""Stage21 seed43 execution kernel derived from frozen Stage20.
No calibration or selection entry point. Historical Stage20 source is untouched.
Only runtime seed, stage labels and explicit partition training seed differ.
"""
from __future__ import annotations
from contextlib import contextmanager
import math, os, shutil, time, uuid
from pathlib import Path
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, sha256_file, seed_everything
from src.paper_if_a2.artifacts import atomic_torch_save
from src.paper_baselines import partitioned
from . import ali_stage7_baseline_sweep_v1 as s7
from . import ali_stage18_amazon_lambda_search_v2 as s18
from . import evaluation, paper_v2, runner
SCHEMA = 'bota-ali-stage21-amazon-seed43-v1'
METHODS = ('IFRU', 'E2URec', 'NegGrad', 'PCGrad', 'SISA', 'RecEraser')
COMPARISON = ('A1', 'A2', 'A3')
ORDER = ('Original', 'Exact-Masked', 'BOTA', *METHODS, 'BOTA-source-only')
read = s18.read

def verify(path):
    m=read(path/'manifest.json')
    actual={p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file() and p.name!='manifest.json'}
    if m['schema']!=SCHEMA or actual!=set(m['files']) or (path/'COMPLETED').read_text(encoding='utf-8')!=SCHEMA+'\n':
        raise ValueError(f'invalid Stage21 bundle: {path}')
    for rel,h in m['files'].items():
        if sha256_file(path/rel)!=h:
            raise ValueError(f'changed Stage21 artifact: {path/rel}')
    return m


def stage_dir(root):
    path=root/'.work'/uuid.uuid4().hex[:10]
    path.mkdir(parents=True)
    return path


def publish(stage,dest):
    if dest.exists():
        raise FileExistsError(dest)
    atomic_text(stage/'COMPLETED',SCHEMA+'\n')
    atomic_json(stage/'manifest.json',{'schema':SCHEMA,
        'files':{p.relative_to(stage).as_posix():sha256_file(p) for p in sorted(stage.rglob('*')) if p.is_file()},
        'test_accessed':False})
    dest.parent.mkdir(parents=True,exist_ok=True)
    os.rename(stage,dest)


def freeze(root,key,payload):
    dest=root/key
    if dest.exists():
        verify(dest)
        if read(dest/'result.json')!=payload:
            raise ValueError(f'frozen identity changed: {key}')
    else:
        stage=stage_dir(root); atomic_json(stage/'result.json',payload); publish(stage,dest)
    return canonical_hash(payload)


@contextmanager
def lock(root):
    root.mkdir(parents=True,exist_ok=True); path=root/'execution.lock'
    handle=path.open('a+b')
    try:
        if handle.tell()==0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        handle.close()


def metric(method,candidate,request,pred,original,exact,indices,c,compute=0,hyper=None):
    if pred is None:
        return {'method':method,'candidate_id':candidate,'request':request['id'],
            'status':'NONFINITE_PREDICTIONS','E_U':None,'D_U':None,'local_residual':None,
            'utility_eligible':False,'prediction_collapse':False,'compute_count':compute,
            'hyperparameters':hyper or {},'seed':43,'test_accessed':False}
    for other in (original,exact):
        if pred['gold_label']!=other['gold_label'] or pred['sample_order_sha256']!=other['sample_order_sha256']:
            raise ValueError('prediction label/order mismatch')
    if len(pred['probability'])!=len(indices):
        raise ValueError('prediction length mismatch')
    p=np.asarray(pred['probability']);o=np.asarray(original['probability']);e=np.asarray(exact['probability'])
    y=np.asarray(pred['gold_label']);lookup={idx:j for j,idx in enumerate(indices)}
    if not set(request['local_indices'])<=set(lookup):
        raise ValueError('requested user omitted from Dev-Select')
    local=[lookup[i] for i in request['local_indices']]
    if not local:
        raise ValueError('request has no local samples')
    if not np.isfinite(p).all():
        return metric(method,candidate,request,None,original,exact,indices,c,compute,hyper)
    eu=float(np.abs(p[local]-e[local]).mean());du=float(np.abs(o[local]-e[local]).mean())
    loss=lambda a:float(np.mean(-(y*np.log(a.clip(1e-12,1-1e-12))+(1-y)*np.log(1-a.clip(1e-12,1-1e-12)))))
    auc=float(roc_auc_score(y,p));oa=float(roc_auc_score(y,o));ll=loss(p);ol=loss(o)
    damage=oa-auc;ld=ll-ol
    return {'method':method,'candidate_id':candidate,'request':request['id'],'status':'OK',
        'E_U':eu,'D_U':du,'local_residual':eu/du if du>0 else None,
        'toward_exact_rate':float(np.mean(np.abs(p[local]-e[local])<np.abs(o[local]-e[local]))),
        'overall_auc':auc,'overall_log_loss':ll,'overall_acc':float(accuracy_score(y,p>=.5)),
        'auc_damage':damage,'logloss_damage':ld,
        'utility_eligible':damage<=c['selection']['auc_damage_max'] and ld<=c['selection']['logloss_damage_max'],
        'prediction_collapse':bool(np.std(p)<1e-6),'compute_count':int(compute),
        'hyperparameters':hyper or {},'local_samples':len(local),'utility_samples':len(indices),
        'seed':43,'test_accessed':False}


def dataset_context(workspace,p):
    b=p['benchmark'];model_dir=evaluation._local_model_directory(workspace,runner._base_t5_config(workspace,b)['paths']['model_dir'])
    tok=s18.s8.T5Tokenizer.from_pretrained(model_dir,local_files_only=True)
    return s18.s8.JsonPromptDataset(workspace/b['source']['development_json'],tok)


def runtime_context(workspace,b,registry,device,fixed_a):
    # The legacy helper resolves its tokenizer relative to the experiment
    # workspace; CLI execution starts in the source repository.
    previous=Path.cwd()
    try:
        os.chdir(workspace)
        return runner._model_context(workspace,b,registry,device,fixed_a)
    finally:
        os.chdir(previous)


def fresh_endpoint(workspace,p):
    b=p['benchmark'];device=s18.gpu();seed_everything(43)
    state=torch.load(Path(p['original'])/'adapter/adapter_model.pt',map_location='cpu',weights_only=True)
    model,names,params,bases,_,tok,data,users=runtime_context(workspace,b,p['registry'],device,state['A'])
    with torch.no_grad():
        for name,param in zip(names,params):
            param.copy_(state['B'][name].to(param))
    model.train()
    return model,names,params,bases,tok,data,users,device


def prediction(root,workspace,p,path,key,dataset,component=False):
    indices=p['development_partition']['select'];dest=root/'predictions'/key
    identity={'contract':canonical_hash(p),'path':str(path.resolve()),'component':component,
        'model_sha256':sha256_file(path/'paper_model_manifest.json' if component else path/'adapter/adapter_model.pt'),
        'indices_sha256':canonical_hash(indices)}
    if dest.exists():
        verify(dest);result=read(dest/'result.json')
        if result['identity']!=identity:
            raise ValueError('prediction identity changed')
        return result['prediction']
    stage=stage_dir(root);device=s18.gpu()
    if component:
        short_id='SISA-Short-T5' if 'sisa' in key.lower() else 'RecEraser-Adapter-Short'
        manifest={'component_run':str(path.resolve()),'method_id':short_id}
        pred=evaluation._ensemble_predictions(workspace,p['benchmark'],manifest,dataset,indices,device,4)
    else:
        model=evaluation._load_fixed(workspace,p['benchmark'],path,device)
        try:
            pred=live_prediction(model,dataset,indices,device,key)
        finally:
            del model;s18.release()
    valid=s18.s14.validate_prediction(pred,indices)
    atomic_json(stage/'result.json',{'identity':identity,'prediction':pred if valid else None,'test_accessed':False})
    publish(stage,dest)
    return pred if valid else None


def live_prediction(model,dataset,indices,device,label):
    pred={'probability':[],'gold_label':[],'answer_loss':[],'sample_order_sha256':canonical_hash(indices)}
    model.eval()
    for start in range(0,len(indices),256):
        chunk=evaluation._single_predictions(model,dataset,indices[start:start+256],device,4)
        for k in ('probability','gold_label','answer_loss'):
            pred[k].extend(chunk[k])
        if start%1024==0 or start+256>=len(indices):
            print(f'[Stage21] predict {label} {min(start+256,len(indices))}/{len(indices)}',flush=True)
    model.train()
    return pred


class ProgressBudget(runner.StepBudget):
    def step(self,optimizer,arm):
        super().step(optimizer,arm)
        if self.calls==1 or self.calls%25==0:
            print(f'[Stage21] {arm} step={self.calls}/200',flush=True)


def exact(root,workspace,p,request):
    dest=root/'exact'/request['id'];identity={'contract':canonical_hash(p),'request':canonical_hash(request)}
    if dest.exists():
        verify(dest)
        if read(dest/'result.json')['identity']!=identity:
            raise ValueError('Exact identity mismatch')
        return dest
    stage=stage_dir(root);b=p['benchmark'];reg=p['registry'];device=s18.gpu();seed_everything(43)
    original=torch.load(Path(p['original'])/'adapter/adapter_model.pt',map_location='cpu',weights_only=True)
    model,names,params,bases,_,tok,data,users=runtime_context(workspace,b,reg,device,original['A'])
    try:
        budget=ProgressBudget(200);started=time.perf_counter()
        values,opt,trace=runner._train_named(model,data,reg['order'],users,request['users'],names,params,device,
            tok.pad_token_id,runner._engine(workspace,b),budget,'Exact-'+request['id'])
        if budget.calls!=200 or sum(t['masked_slots'] for t in trace)!=request['deleted_interactions']:
            raise ValueError('Exact masking/budget mismatch')
        runner._save_fixed_ab(stage/'adapter',names,values,bases,'Stage21-Exact-Masked')
        atomic_json(stage/'result.json',{'identity':identity,'compute_count':200,
            'seconds':time.perf_counter()-started,'trace_sha256':canonical_hash(trace),'test_accessed':False})
        del opt
    finally:
        del model,params,bases,data,tok;s18.release()
    publish(stage,dest)
    return dest


def references(root,workspace,p,requests,dataset):
    original=prediction(root,workspace,p,Path(p['original']),'Original',dataset)
    if original is None:
        raise ValueError('nonfinite Original')
    result={}
    for r in requests:
        ref=prediction(root,workspace,p,exact(root,workspace,p,r),'Exact_'+r['id'],dataset)
        if ref is None:
            raise ValueError('nonfinite Exact-Masked')
        result[r['id']]=(original,ref)
    return result


def unit_paths(root,identity):
    key=canonical_hash(identity)[:12]
    return root/'units'/key, root/'components'/key


def unit_result(dest,identity):
    if not dest.exists():
        return None
    verify(dest);r=read(dest/'result.json')
    if r['identity']!=identity:
        raise ValueError('unit identity changed')
    print(f"[Stage21] reused {identity['phase']}/{identity['method']}/{identity['request']}",flush=True)
    return r['rows']


def write_unit(stage,dest,identity,rows):
    atomic_json(stage/'result.json',{'identity':identity,'rows':rows,'test_accessed':False})
    publish(stage,dest)


def augmented_teacher(root,workspace,p,request):
    """One teacher per request, shared by all forget-weight candidates."""
    dest=root/'teachers'/request['id'];g=p['config']['grids']['E2URec']
    identity={'contract':canonical_hash(p),'request':canonical_hash(request),
        'steps':g['teacher_steps'],'learning_rate':g['teacher_learning_rate']}
    if dest.exists():
        verify(dest)
        if read(dest/'result.json')['identity']!=identity:raise ValueError('teacher identity changed')
        return dest
    stage=stage_dir(root);model,names,params,bases,tok,data,_,device=fresh_endpoint(workspace,p)
    started=time.perf_counter()
    opt=torch.optim.AdamW(params,lr=g['teacher_learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=.01)
    try:
        for step in range(g['teacher_steps']):
            logical=s7.e2urec._wrapped(request['forget_train_indices'],16,43,step,70000)
            opt.zero_grad(set_to_none=True)
            paper_v2._backward_mean(model,data,logical,device,tok.pad_token_id,1.,4);opt.step()
            if step==0 or (step+1)%100==0:
                print(f'[Stage21] E2URec teacher {request["id"]} {step+1}/{g["teacher_steps"]}',flush=True)
        values=runner._copy_parameters(names,params)
        if not all(torch.isfinite(v).all().item() for v in values.values()):
            raise FloatingPointError('nonfinite shared E2URec teacher')
        runner._save_fixed_ab(stage/'adapter',names,values,bases,'Stage21-E2URec-Teacher')
        atomic_json(stage/'result.json',{'identity':identity,'seconds':time.perf_counter()-started,
            'optimizer_steps':g['teacher_steps'],'test_accessed':False})
    except BaseException as error:
        atomic_json(stage/'FAILED.json',{'identity':identity,'error':repr(error),'test_accessed':False});raise
    finally:
        del model,params,bases,tok,data,opt;s18.release()
    publish(stage,dest)
    return dest


def gradient_unit(root,workspace,p,request,method,spec,phase,dataset,refs):
    c=p['config'];identity={'contract':canonical_hash(p),'phase':phase,'method':method,'request':request['id'],'spec':spec}
    dest,_=unit_paths(root,identity);done=unit_result(dest,identity)
    if done is not None:
        return done
    stage=stage_dir(root);g=c['grids'][method];outer=spec['outer'];checkpoints=set(spec['checkpoints']);rows=[]
    forget=request['forget_train_indices'];fs=set(forget);retain=[i for i in p['registry']['order'] if i not in fs]
    started=time.perf_counter();teacher_seconds=0;teachers=[]
    if method=='E2URec':
        teacher=augmented_teacher(root,workspace,p,request);device=s18.gpu()
        augmented=evaluation._load_fixed(workspace,p['benchmark'],teacher,device)
        om=evaluation._load_fixed(workspace,p['benchmark'],Path(p['original']),device)
        s7.e2urec._freeze(augmented);s7.e2urec._freeze(om)
        teachers=[augmented,om];teacher_seconds=read(teacher/'result.json')['seconds']
    model,names,params,bases,tok,data,_,device=fresh_endpoint(workspace,p)
    lr=g['learning_rate'] if method=='NegGrad' else outer if method=='PCGrad' else g['student_learning_rate']
    opt=torch.optim.AdamW(params,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    try:
        for step in range(1,max(checkpoints)+1):
            opt.zero_grad(set_to_none=True)
            if method in ('NegGrad','PCGrad'):
                ridx,fidx=s7._pair_indices(retain,forget,step-1,43)
                if method=='NegGrad':
                    paper_v2._backward_mean(model,data,ridx,device,tok.pad_token_id,1.,4)
                    paper_v2._backward_mean(model,data,fidx,device,tok.pad_token_id,-float(outer),4)
                else:
                    rg,_=paper_v2._mean_grads(model,data,ridx,params,device,tok.pad_token_id,1.,4)
                    fg,_=paper_v2._mean_grads(model,data,fidx,params,device,tok.pad_token_id,-1.,4)
                    merged,_=paper_v2._pcgrad(rg,fg)
                    for param,grad in zip(params,merged):
                        param.grad=grad
                    del rg,fg,merged
            else:
                fidx=s7.e2urec._wrapped(forget,16,43,step-1,90000)
                ridx=s7.e2urec._wrapped(retain,16,43,step-1,110000)
                for offset in range(0,16,4):
                    fb=s7.e2urec._batch(data,fidx[offset:offset+4],device,tok.pad_token_id)
                    if step<=math.ceil(len(forget)/16):
                        loss=s7.e2urec.forget_loss_light(model,om,augmented,fb,2.)
                    else:
                        rb=s7.e2urec._batch(data,ridx[offset:offset+4],device,tok.pad_token_id)
                        parts=s7.e2urec.joint_losses_light(model,om,augmented,fb,rb,2.)
                        loss=(1-float(outer))*(parts['L_sup']+parts['L_retain_KL'])+float(outer)*parts['L_forget']
                        del rb,parts
                    (loss/4).backward();del fb,loss
            opt.step()
            if step==1 or step%50==0:
                print(f'[Stage21] {phase}/{method}/{request["id"]} outer={outer} step={step}/{max(checkpoints)}',flush=True)
            if step in checkpoints:
                cid=(f'pcgrad_lr{outer:g}_s{step}' if method=='PCGrad' else
                    f'neggrad_w{outer:g}_s{step}' if method=='NegGrad' else f'e2urec_w{outer:g}_s{step}')
                values=runner._copy_parameters(names,params)
                finite=all(torch.isfinite(v).all().item() for v in values.values())
                pred=live_prediction(model,dataset,p['development_partition']['select'],device,cid) if finite else None
                hyper={'learning_rate':float(lr),'checkpoint':step}
                if method!='PCGrad':hyper['forget_weight']=float(outer)
                if method=='E2URec':hyper.update(teacher_steps=g['teacher_steps'],teacher_learning_rate=g['teacher_learning_rate'],student_learning_rate=lr)
                row=metric(method,cid,request,pred,*refs[request['id']],p['development_partition']['select'],c,
                    step+(g['teacher_steps'] if method=='E2URec' else 0),hyper)
                row.update(wall_time_seconds=time.perf_counter()-started,teacher_seconds=teacher_seconds)
                atomic_json(stage/f'{cid}_prediction.json',{'prediction':pred,'test_accessed':False})
                if finite:
                    artifact=runner._save_fixed_ab(stage/cid/'adapter',names,values,bases,cid)
                    row['model_sha256']=artifact['sha256']
                rows.append(row)
    except BaseException as error:
        atomic_json(stage/'FAILED.json',{'identity':identity,'error':repr(error),'test_accessed':False});raise
    finally:
        del model,params,opt,data,bases,tok,teachers
        if method=='E2URec':del augmented,om
        s18.release()
    write_unit(stage,dest,identity,rows)
    return rows


def checked_ifru_solve(operator,gradient,damping,max_iterations):
    """Preserve the solver's guard; only classify its explicit residual rejection."""
    try:
        return runner.conjugate_gradient(operator,gradient,damping=damping,
            relative_tolerance=.0001,absolute_tolerance=1e-10,max_iterations=max_iterations,
            residual_explosion_factor=1000.,pap_tolerance=1e-14,allow_truncated_solution=True),None
    except RuntimeError as error:
        if str(error)!='false_residual_convergence':raise
        return None,{'error':'false_residual_convergence','status':'NUMERICAL_SOLVER_FAILURE',
                     'solution_used':False,'tolerance_relaxed':False,'fallback_used':False}


def ifru_unit(root,workspace,p,request,spec,phase,dataset,refs):
    c=p['config'];identity={'contract':canonical_hash(p),'phase':phase,'method':'IFRU','request':request['id'],'spec':spec}
    dest,_=unit_paths(root,identity);done=unit_result(dest,identity)
    if done is not None:return done
    stage=stage_dir(root);rows=[];started=time.perf_counter()
    model,names,params,bases,tok,data,_,device=fresh_endpoint(workspace,p);model.eval()
    try:
        panel=[p['registry']['order'][i] for i in runner.deterministic_panel(3200,spec['panel'],43)]
        resident,_=runner.materialize_resident_panel(data,panel,8,tok.pad_token_id,device)
        counter={};operator=runner.make_sample_mean_ggn_operator(model,resident,params,counter)
        estimate=runner.estimate_lambda_max(operator,sum(x.numel() for x in params),seed=43,iterations=12,convergence_tolerance=.0001,numerical_lower_bound=1e-14)
        gradient_parts,_=runner.sample_mean_gradients(model,data,request['forget_train_indices'],params,device,8,tok.pad_token_id)
        gradient=runner.flatten_tensors(gradient_parts);flat=runner.flatten_tensors([x.detach() for x in params])
        for ratio in spec['ratios']:
            cid=f'ifru_p{spec["panel"]}_d{ratio:g}'
            print(f'[Stage21] {phase}/IFRU/{request["id"]}/{cid}',flush=True)
            cg,failure=checked_ifru_solve(operator,gradient,estimate['lambda_max_hat']*ratio,
                                         c['grids']['IFRU']['cg_max_iterations'])
            if failure is not None:
                row=metric('IFRU',cid,request,None,*refs[request['id']],p['development_partition']['select'],c,
                           hyper={'curvature_panel_samples':spec['panel'],'relative_damping_ratio':ratio,
                                  'cg_max_iterations':c['grids']['IFRU']['cg_max_iterations']})
                row.update(status='NUMERICAL_SOLVER_FAILURE',compute_count=None,cg_diagnostics=failure,
                           operator_counter=dict(counter),wall_time_seconds=time.perf_counter()-started,
                           implementation_repair='stage21-repair01')
                atomic_json(stage/f'{cid}_prediction.json',{'prediction':None,'solver_failure':failure,'test_accessed':False})
                rows.append(row)
                print(f'[Stage21] IFRU/{request["id"]}: false residual convergence; recorded failure, no endpoint used',flush=True)
                continue
            direction=cg.pop('solution');actual=runner.quantized_delta(flat,direction,request['deleted_interactions']/3200)
            values={};offset=0
            for name,param in zip(names,params):
                size=param.numel();values[name]=(param.detach().reshape(-1).cpu()+actual[offset:offset+size].cpu()).reshape_as(param);offset+=size
            artifact=runner._save_fixed_ab(stage/cid/'adapter',names,values,bases,cid)
            cm=evaluation._load_fixed(workspace,p['benchmark'],stage/cid,device)
            try:pred=live_prediction(cm,dataset,p['development_partition']['select'],device,cid)
            finally:del cm;s18.release()
            hyper={'curvature_panel_samples':spec['panel'],'relative_damping_ratio':ratio,
                'cg_max_iterations':c['grids']['IFRU']['cg_max_iterations'],'lambda_max':estimate['lambda_max_hat']}
            row=metric('IFRU',cid,request,pred,*refs[request['id']],p['development_partition']['select'],c,12+cg['hvp_calls'],hyper)
            row.update(model_sha256=artifact['sha256'],cg_diagnostics=cg,wall_time_seconds=time.perf_counter()-started)
            atomic_json(stage/f'{cid}_prediction.json',{'prediction':pred,'test_accessed':False});rows.append(row)
            del direction,actual,values
    except BaseException as error:
        atomic_json(stage/'FAILED.json',{'identity':identity,'error':repr(error),'test_accessed':False});raise
    finally:
        del model,params,bases,tok,data
        s18.release()
    write_unit(stage,dest,identity,rows)
    return rows


def partition_config(root,workspace,p,request,method,spec,component):
    """Use the existing partition engine and its manifest-bound row authority."""
    kind='sisa' if method=='SISA' else 'receraser'
    base=workspace/p['benchmark']['output_root']/'protocols'/p['config']['benchmark_name']/'data/K2'/f'{kind}.yaml'
    config=yaml.safe_load(base.read_text(encoding='utf-8'))
    config['seed']=43
    config['protocol']['partition_seed']=43
    rows=read(workspace/config['sources']['train']['path'])
    # Reject ambiguous prompt matching before training. This engine locates
    # forget examples by row identity; the selected positions must be exact.
    forget_rows=[rows[i] for i in request['forget_window_positions']]
    located=partitioned.locate_subset_rows(rows,forget_rows)
    if set(located)!=set(request['forget_window_positions']):
        raise ValueError('partition engine cannot uniquely identify this request; no silent row substitution')
    inputs=root/'partition_inputs';inputs.mkdir(parents=True,exist_ok=True)
    forget_path=inputs/f'{request["id"]}_forget.json'
    fit_path=inputs/'development_fit.json';fit_ix_path=inputs/'development_fit_indices.json'
    full_dev=read(workspace/p['benchmark']['source']['development_json'])
    def identical(path,value):
        if path.exists():
            if read(path)!=value:raise ValueError(f'frozen input changed: {path}')
        else:atomic_json(path,value)
    identical(forget_path,forget_rows)
    identical(fit_path,[full_dev[i] for i in p['development_partition']['fit']])
    identical(fit_ix_path,p['development_partition']['fit'])
    rel=lambda path:path.relative_to(workspace).as_posix()
    config['sources']['forget']={'path':rel(forget_path),'sha256':sha256_file(forget_path)}
    original_dev=config['sources']['development']
    config['sources']['development']={'path':rel(fit_path),'sha256':sha256_file(fit_path)}
    config['development_subset_authority']={'full_path':original_dev['path'],'full_sha256':original_dev['sha256'],
        'indices_path':rel(fit_ix_path),'indices_sha256':sha256_file(fit_ix_path)}
    config['data']['forget_samples']=len(forget_rows)
    config['protocol']['learning_rate']=spec['learning_rate']
    config['protocol']['epochs_per_slice']=spec['epochs_per_slice']
    config['output_root']=rel(component)
    config['synthetic_root']=rel(component.parent/'synthetic')
    if method=='SISA':
        parts=partitioned.sisa_partition(rows,4,4,43)
    else:
        assignment=partitioned.balanced_similarity_partition(partitioned.text_features(rows,32),4,43)
        parts={s:{0:[i for i,a in enumerate(assignment) if a==s]} for s in range(4)}
    forgotten=set(located);slices=4 if method=='SISA' else 1
    initial=[[len(parts[s][t]) for t in range(slices)] for s in range(4)]
    remaining=[[sum(i not in forgotten for i in parts[s][t]) for t in range(slices)] for s in range(4)]
    config['budget']['initial_partition_counts']=initial if method=='SISA' else [v[0] for v in initial]
    config['budget']['post_delete_partition_counts']=remaining if method=='SISA' else [v[0] for v in remaining]
    config['budget']['total_expected_optimizer_steps']=spec['epochs_per_slice']*sum(
        math.ceil(v/16) for group in initial+remaining for v in group)
    plan=partitioned.deletion_plan(parts,located)
    initial_steps=spec['epochs_per_slice']*sum(math.ceil(v/16) for group in initial for v in group)
    unlearn_steps=spec['epochs_per_slice']*sum(math.ceil(remaining[s][t]/16)
        for s,start in plan.items() for t in range(start,slices))
    config['stage21_executable_budget']={'initial_optimizer_steps':initial_steps,
        'unlearning_optimizer_steps':unlearn_steps,'total_optimizer_steps':initial_steps+unlearn_steps,
        'legacy_static_budget_is_upper_bound':True}
    config['budget']['official_epoch_or_step_budget']=f"Stage21 bounded candidate: {spec['epochs_per_slice']} epochs per local slice"
    partitioned.validate_partition_config(config,kind);partitioned.partitioned_budget(config,kind)
    return config


def remove_generated_component(root,component):
    base=(root/'components').resolve();target=component.resolve()
    if base not in target.parents:
        raise ValueError('component cleanup escapes Stage21 output scope')
    if target.exists():
        shutil.rmtree(target)


@contextmanager
def partition_resume_rng(run):
    """Bind completed shard boundaries to RNG state without editing the engine."""
    writer=partitioned.atomic_json;constructor=partitioned._new_local_model
    state=read(run/'run_state.json') if (run/'run_state.json').exists() else {'next_shard':0}
    last_saved=int(state['next_shard']);restored=False
    if last_saved and not (run/f'rng_{last_saved}.pt').exists() and not (run/'COMPLETED').exists():
        raise ValueError('missing RNG checkpoint for interrupted partition run')
    def save_state(path,value):
        nonlocal last_saved
        if Path(path)==run/'run_state.json' and int(value.get('next_shard',0))>last_saved:
            next_shard=int(value['next_shard'])
            atomic_torch_save(run/f'rng_{next_shard}.pt',{'next_shard':next_shard,
                'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
            last_saved=next_shard
        writer(path,value)
    def create_model(*args,**kwargs):
        nonlocal restored
        if not restored:
            if int(state['next_shard']):
                rng=torch.load(run/f'rng_{state["next_shard"]}.pt',map_location='cpu',weights_only=True)
                if rng['next_shard']!=state['next_shard']:raise ValueError('RNG/shard checkpoint mismatch')
                torch.set_rng_state(rng['cpu'])
                if rng['cuda']:torch.cuda.set_rng_state_all(rng['cuda'])
            restored=True
        return constructor(*args,**kwargs)
    partitioned.atomic_json=save_state;partitioned._new_local_model=create_model
    try:
        yield
    finally:
        partitioned.atomic_json=writer;partitioned._new_local_model=constructor


def partition_unit(root,repo,workspace,p,request,method,spec,phase,dataset,refs):
    identity={'contract':canonical_hash(p),'phase':phase,'method':method,'request':request['id'],'spec':spec}
    dest,component=unit_paths(root,identity);done=unit_result(dest,identity)
    if done is not None:
        remove_generated_component(root,component)
        return done
    kind='sisa' if method=='SISA' else 'receraser';prefix=kind
    cid=f'{prefix}_lr{spec["learning_rate"]:g}_e{spec["epochs_per_slice"]}'
    config=partition_config(root,workspace,p,request,method,spec,component)
    config_path=root/'partition_configs'/f'{canonical_hash(identity)[:12]}.yaml'
    text=yaml.safe_dump(config,sort_keys=False,allow_unicode=True)
    if config_path.exists():
        if config_path.read_text(encoding='utf-8')!=text:raise ValueError('partition config collision')
    else:atomic_text(config_path,text)
    run=component/'run';started=time.perf_counter();seed_everything(43)
    previous=Path.cwd()
    try:
        os.chdir(workspace)
        if (run/'COMPLETED').exists():
            partitioned.analyze_run(run,partitioned.MARKERS[kind])
        else:
            with partition_resume_rng(run):
                result=partitioned.execute(workspace,config_path,'run',kind,run.exists(),
                    partition_seed_override=43,formal=False,implementation_root=repo)
            if result['status']!='COMPLETED':
                raise RuntimeError(f'partition engine stopped before completion: {result}')
    finally:
        os.chdir(previous)
    pred=prediction(root,workspace,p,run,f'{kind}_{canonical_hash(identity)[:12]}',dataset,True)
    manifest=read(run/'paper_model_manifest.json')
    if manifest['optimizer_steps']!=config['stage21_executable_budget']['total_optimizer_steps']:
        raise ValueError('partition optimizer calls differ from actual affected-shard plan')
    row=metric(method,cid,request,pred,*refs[request['id']],p['development_partition']['select'],p['config'],
        manifest['optimizer_steps'],{**spec,'shards':4,'slices':4 if method=='SISA' else 1,
            'attention_fit':'Dev-Fit excluding requested users' if method=='RecEraser' else 'fixed mean'})
    row.update(wall_time_seconds=time.perf_counter()-started,
        model_sha256=sha256_file(run/'paper_model_manifest.json'),model_tensors_retained=False,
        executable_budget=config['stage21_executable_budget'])
    stage=stage_dir(root)
    for name in ('paper_model_manifest.json','budget_audit.json','partition.json','aggregation.json','timing.json'):
        shutil.copy2(run/name,stage/name)
    atomic_json(stage/'prediction.json',{'prediction':pred,'test_accessed':False})
    atomic_text(stage/'config.yaml',text)
    write_unit(stage,dest,identity,[row])
    # These are newly generated, reproducible candidate models. Keeping their
    # prediction records, tensor hashes and settings avoids hundreds of GiB.
    remove_generated_component(root,component)
    return [row]


def execute_method(root,repo,workspace,p,method,phase,requests,dataset,refs,specs):
    rows=[]
    for request in requests:
        for spec in specs:
            print(f'[Stage21] {phase}/{method}/{request["id"]} {spec}',flush=True)
            if method in ('NegGrad','PCGrad','E2URec'):
                chunk=gradient_unit(root,workspace,p,request,method,spec,phase,dataset,refs)
            elif method=='IFRU':
                chunk=ifru_unit(root,workspace,p,request,spec,phase,dataset,refs)
            else:
                chunk=partition_unit(root,repo,workspace,p,request,method,spec,phase,dataset,refs)
            rows.extend(chunk)
    return rows


def chosen_spec(method,decision):
    chosen=decision['selected'] or decision['diagnostic']
    if chosen is None:return None
    h=chosen['hyperparameters']
    if method in ('NegGrad','PCGrad','E2URec'):
        return {'outer':h['learning_rate'] if method=='PCGrad' else h['forget_weight'],
            'checkpoints':[h['checkpoint']]}
    if method=='IFRU':
        return {'panel':h['curvature_panel_samples'],'ratios':[h['relative_damping_ratio']]}
    return {'learning_rate':h['learning_rate'],'epochs_per_slice':h['epochs_per_slice']}


def build_bota(root,workspace,p):
    dest=root/'bank';identity={'contract':canonical_hash(p),'scales':[0,48]}
    if dest.exists():
        verify(dest)
        if read(dest/'result.json')['identity']!=identity:raise ValueError('BOTA bank identity changed')
        return dest
    stage=stage_dir(root);b=p['benchmark'];reg=p['registry'];device=s18.gpu();seed_everything(43)
    original=torch.load(Path(p['original'])/'adapter/adapter_model.pt',map_location='cpu',weights_only=True)
    requests=p['comparison']['requests'];users_requested=[u for r in requests for u in r['users']]
    model,names,params,bases,_,tok,data,users=runtime_context(workspace,b,reg,device,original['A'])
    started=time.perf_counter()
    try:
        canonical,vectors,failures,steps,control=s18.transport(model,data,reg['order'],users,users_requested,
            params,names,device,tok.pad_token_id,runner._engine(workspace,b),[0,48],stage/'progress.json')
        if steps!=200 or any(not torch.equal(canonical[n],original['B'][n]) for n in names):
            raise RuntimeError('Stage21 canonical endpoint differs from sealed Stage18 Original')
        if control['status']!='TENSOR_EXACT_PASS':raise RuntimeError('transport control failed')
        for request in requests:
            slots=[users_requested.index(u) for u in request['users']]
            for scale in (0,48):
                key=s18.cid(scale)
                if key in failures:continue
                values=s18.compose(canonical,names,vectors[key][slots].sum(0))
                runner._save_fixed_ab(stage/request['id']/key/'adapter',names,values,bases,f'Stage21-BOTA-{scale}')
        atomic_json(stage/'result.json',{'identity':identity,'failures':failures,'control':control,
            'compute_count':steps,'bank_users':len(users_requested),'seconds':time.perf_counter()-started,'test_accessed':False})
    except BaseException as error:
        atomic_json(stage/'FAILED.json',{'identity':identity,'error':repr(error),'test_accessed':False});raise
    finally:
        del model,params,bases,tok,data;s18.release()
    publish(stage,dest)
    return dest


def summarize_rows(rows):
    result={}
    for method in ORDER:
        cells=[r for r in rows if r['method']==method]
        if len(cells)!=3 or {r['request'] for r in cells}!=set(COMPARISON):
            raise ValueError(f'incomplete or duplicate comparison: {method}')
        ok=all(r['status']=='OK' for r in cells)
        stats={}
        for key in ('E_U','D_U','local_residual','toward_exact_rate','overall_auc','overall_log_loss'):
            values=[r.get(key) for r in cells]
            stats[key]={'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1))} if ok and all(v is not None for v in values) else None
        result[method]={'stats':stats,'utility_pass':sum(r['utility_eligible'] for r in cells),
            'valid_cells':sum(r['status']=='OK' for r in cells),'total_cells':3,
            'calibration_status':cells[0].get('calibration_status','REFERENCE'),
            'diagnostic_only':cells[0].get('diagnostic_only',False)}
    bota={r['request']:r for r in rows if r['method']=='BOTA'}
    wins={}
    for method in METHODS+('Original','BOTA-source-only'):
        cells=[r for r in rows if r['method']==method]
        wins[method]={'lower_E_U_requests':sum(r['status']=='OK' and bota[r['request']]['status']=='OK' and
            bota[r['request']]['E_U']<r['E_U'] for r in cells),'total_requests':3,
            'invalid_comparator_cells':sum(r['status']!='OK' for r in cells)}
    return {'methods':result,'BOTA_request_wins':wins,
        'dispersion':'sample standard deviation across three requests, not across seeds',
        'single_seed_exploratory':True,'seed':43,'test_accessed':False}

