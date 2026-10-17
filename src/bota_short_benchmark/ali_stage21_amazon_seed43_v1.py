"""Frozen-parameter Amazon seed43 follow-up; no calibration entry point."""
from __future__ import annotations
import argparse
import copy
import csv
import json
import shutil
import time
from pathlib import Path
import torch
import yaml
from . import ali_stage20_amazon_baselines_v1 as parent
from . import ali_stage21_amazon_seed43_kernel_v1 as k

CONFIG = 'configs/ali_stage21_amazon_seed43_v1.json'
EXPECTED = {'IFRU':'ifru_p1024_d0.1','E2URec':'e2urec_w0.4_s500',
            'NegGrad':'neggrad_w0.05_s25','PCGrad':'pcgrad_lr0.0005_s25',
            'SISA':'sisa_lr0.0001_e3','RecEraser':'receraser_lr0.002_e1'}

def load_config(repo):
    c=k.read(repo/CONFIG)
    if (c['schema']!=k.SCHEMA or c['seed']!=43 or c['calibration_seed']!=42
        or c['primary_lambda']!=48 or c['ablation_lambda']!=0 or c['retune']
        or c['requests']!=list(k.COMPARISON) or c['methods']!=list(k.METHODS)
        or c['output_root']!='outputs/ali_stage21_amazon_seed43_v1'
        or c['parent_root']!='outputs/ali_stage20_amazon_baselines_v1'
        or c['original_bank']!='outputs/ali_stage18_amazon_lambda_search_v2/banks/confirm_43'
        or c['test_access_policy']!='forbidden' or not c['preserve_parent_results']
        or not c['retain_ineligible_diagnostics']):
        raise ValueError('Stage21 frozen scope changed')
    amendment=k.read(repo/c['amendment'])
    if amendment['status']!='SPECIFIED_BEFORE_STAGE21_EXECUTION' or amendment['test_accessed'] or not amendment['final_test_forbidden']:
        raise ValueError('invalid follow-up amendment')
    return c

def check_choices(decisions):
    if set(decisions)!=set(k.METHODS):raise ValueError('missing inherited methods')
    for method in k.METHODS:
        d=decisions[method];chosen=d['selected'] or d['diagnostic']
        if chosen is None or chosen['candidate_id']!=EXPECTED[method]:
            raise ValueError(f'inherited choice changed: {method}')
        expected='NO_ELIGIBLE_CANDIDATE' if method in ('PCGrad','SISA') else 'SELECTED'
        if d['status']!=expected:raise ValueError(f'inherited eligibility changed: {method}')

def audit(repo,workspace,c):
    oldroot=workspace/c['parent_root']
    for name,sha in c['parent_hashes'].items():
        parent.verify(oldroot/name)
        if k.sha256_file(oldroot/name/'result.json')!=sha:raise ValueError(f'parent changed: {name}')
    old=k.read(oldroot/'protocol/result.json')
    parent.check_inputs(repo,old)
    decisions=k.read(oldroot/'all_selections/result.json')['decisions'];check_choices(decisions)
    bank=workspace/c['original_bank'];k.s18.verify(bank)
    original=bank/'Original'
    p=copy.deepcopy(old)
    p.update(schema=k.SCHEMA,followup_config=c,inherited_decisions=decisions,
             original=str(original.resolve()),scope='Post-Stage20 fixed-parameter seed43 follow-up on reused Development; no independent-holdout claim')
    p['config']['protocol']['seed']=43
    p['config']['output_root']=c['output_root']
    p['benchmark']['protocol']['run_seed']=43
    p['benchmark']['coordinate']['fixed_a_seed']=43
    # The registry, canonical order, cohorts, checkpoint and split deliberately remain unchanged.
    if [r['id'] for r in p['comparison']['requests']]!=list(k.COMPARISON):raise ValueError('requests changed')
    files=[repo/CONFIG,repo/c['amendment'],Path(__file__).resolve(),Path(k.__file__).resolve(),
           repo/'scripts/bota_if/run_ali_stage21_amazon_seed43_v1.ps1']
    for path in files:p['checks']['source_sha256'][path.relative_to(repo).as_posix()]=k.sha256_file(path)
    for name in c['parent_hashes']:
        for filename in ('result.json','manifest.json','COMPLETED'):
            path=oldroot/name/filename;p['checks']['input_sha256'][str(path.resolve())]=k.sha256_file(path)
    for path in [bank/'manifest.json',bank/'result.json',original/'adapter/adapter_model.pt',original/'adapter/adapter_config.json']:
        p['checks']['input_sha256'][str(path.resolve())]=k.sha256_file(path)
    return p

def preflight(workspace,p):
    free=shutil.disk_usage(workspace).free/2**30
    if free<32:raise RuntimeError(f'32 GiB free space required; available {free:.1f}')
    return {'status':'PREFLIGHT_PASS','seed':43,'calibration_seed':42,'retune':False,
            'requests':list(k.COMPARISON),'lambda':48,'ablation_lambda':0,'comparison_cells':30,
            'choices':EXPECTED,'diagnostic_only':['PCGrad','SISA'],
            'request_registry_sha256':k.canonical_hash(p['comparison']),
            'development_partition_sha256':k.canonical_hash(p['development_partition']),
            'dev_select':len(p['development_partition']['select']),'free_disk_gib':round(free,1),
            'test_accessed':False,'model_loaded':False}

def check_resume_inputs(repo,workspace,p,root):
    """Narrow, hash-bound implementation migration; frozen protocol is untouched."""
    from . import ali_stage21_repair01 as repair
    return repair.check(repo,workspace,p,root)

def compare(root,repo,workspace,p,method):
    decisions=p['inherited_decisions'];check_choices(decisions)
    k.freeze(root,'inherited_selections',{'contract':k.canonical_hash(p),'decisions':decisions,
        'source_seed':42,'evaluation_seed':43,'retuned':False,'test_accessed':False})
    data=k.dataset_context(workspace,p);requests=p['comparison']['requests']
    refs=k.references(root,workspace,p,requests,data)
    bank=k.build_bota(root,workspace,p);failures=k.read(bank/'result.json')['failures'];rows=[]
    for request in requests:
        original,exact=refs[request['id']]
        for name,pred in [('Original',original),('Exact-Masked',exact)]:
            rows.append(k.metric(name,name,request,pred,original,exact,p['development_partition']['select'],p['config']))
        for scale,name in [(48,'BOTA'),(0,'BOTA-source-only')]:
            failure=failures.get(k.s18.cid(scale))
            pred=None if failure else k.prediction(root,workspace,p,bank/request['id']/k.s18.cid(scale),name+'_'+request['id'],data)
            row=k.metric(name,name,request,pred,original,exact,p['development_partition']['select'],p['config'],0,{'lambda':scale})
            row.update(calibration_status='FIXED_FROM_STAGE18_SEED42',offline_optimizer_steps=200,online_optimizer_steps=0)
            if failure:row['transport_failure']=failure
            rows.append(row)
    k.freeze(root,'comparison/references',{'contract':k.canonical_hash(p),'rows':rows,'test_accessed':False})
    for name in (k.METHODS if method=='All' else [method]):
        dest=root/'comparison'/name
        if dest.exists():
            k.verify(dest)
            if k.read(dest/'result.json')['contract']!=k.canonical_hash(p):raise ValueError('comparison changed')
            print(f'[Stage21] reused {name}',flush=True);continue
        spec=k.chosen_spec(name,decisions[name])
        rows=k.execute_method(root,repo,workspace,p,name,'comparison',requests,data,refs,[spec])
        for row in rows:
            row.update(calibration_status=decisions[name]['status'],calibration_seed=42,
                       diagnostic_only=decisions[name]['selected'] is None)
        k.freeze(root,'comparison/'+name,{'contract':k.canonical_hash(p),'rows':rows,'test_accessed':False})
    return {'status':'COMPARISON_COMPLETED','seed':43,'method':method,'test_accessed':False}

def analyze(root,p):
    required=['comparison/references']+['comparison/'+m for m in k.METHODS]
    missing=[s for s in required if not (root/s).exists()]
    if missing:return {'status':'INCOMPLETE','missing':missing,'test_accessed':False}
    rows=[]
    for name in required:
        k.verify(root/name);r=k.read(root/name/'result.json')
        if r['contract']!=k.canonical_hash(p):raise ValueError('contract changed')
        rows+=r['rows']
    if len(rows)!=30 or any(r['seed']!=43 for r in rows):raise ValueError('missing cells or wrong seed')
    result=k.summarize_rows(rows)
    result.update(status='COMPLETED',contract=k.canonical_hash(p),rows=rows,calibration_seed=42,retuned=False,
                  scope=p['scope'],historical_results_preserved=True)
    result['implementation_repair']='stage21-repair01'
    result['solver_failures']=[{'method':r['method'],'request':r['request'],'status':r['status'],
                               'diagnostics':r.get('cg_diagnostics')} for r in rows if r['status']!='OK']
    dest=root/'summary'
    if dest.exists():
        k.verify(dest)
        if k.read(dest/'result.json')!=result:raise ValueError('summary changed')
        return result
    stage=k.stage_dir(root);k.atomic_json(stage/'result.json',result)
    fields=['method','request','seed','candidate_id','calibration_status','diagnostic_only','status',
            'E_U','D_U','local_residual','toward_exact_rate','overall_auc','overall_log_loss','utility_eligible','compute_count']
    with (stage/'comparison.csv').open('w',encoding='utf-8',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# Stage21 Amazon seed43 frozen-parameter follow-up','',p['scope'],
           'Parameters inherited from seed42; lambda48 fixed. Same three requests and samples. No retuning or FinalTest.',
           'Request dispersion is not cross-seed uncertainty. Prior seeds/results remain separate and must not be suppressed.','',
           '| Method | Mean E_U | Mean residual | Utility pass | Inherited calibration status |',
           '|---|---:|---:|---:|---|']
    for name in k.ORDER:
        r=result['methods'][name];s=r['stats']
        fmt=lambda key: 'NA' if s[key] is None else f"{s[key]['mean']:.8f}"
        lines.append(f"| {name} | {fmt('E_U')} | {fmt('local_residual')} | {r['utility_pass']}/3 | {r['calibration_status']} |")
    lines+=['','PCGrad and SISA remain ineligible diagnostics irrespective of their seed43 outcomes.']
    if result['solver_failures']:
        lines+=['','Numerical solver failures (not scored as wins; incomplete method means are NA):']
        lines += [f"- {r['method']}/{r['request']}: {r['status']} ({r['diagnostics']})" for r in result['solver_failures']]
    k.atomic_text(stage/'report.md','\n'.join(lines)+'\n');k.publish(stage,dest)
    return result

def smoke(root,repo,workspace,p):
    """One-step model/inference check, never publishes a comparison metric."""
    stage=k.stage_dir(root);device=k.s18.gpu();k.seed_everything(43)
    state=torch.load(Path(p['original'])/'adapter/adapter_model.pt',map_location='cpu',weights_only=True)
    model,names,params,bases,_,tok,data,users=k.runtime_context(workspace,p['benchmark'],p['registry'],device,state['A'])
    try:
        budget=k.runner.StepBudget(1)
        values,opt,trace=k.runner._train_named(model,data,p['registry']['order'][:16],users,[],names,params,device,
            tok.pad_token_id,k.runner._engine(workspace,p['benchmark']),budget,'Stage21-smoke')
        k.runner._save_fixed_ab(stage/'adapter',names,values,bases,'Stage21-Smoke');del opt
    finally:
        del model,params,bases,tok,data;k.s18.release()
    model=k.evaluation._load_fixed(workspace,p['benchmark'],stage,device)
    dataset=k.dataset_context(workspace,p);indices=p['development_partition']['select'][:8]
    try:pred=k.evaluation._single_predictions(model,dataset,indices,device,4)
    finally:del model;k.s18.release()
    if not k.s18.s14.validate_prediction(pred,indices):raise ValueError('nonfinite smoke prediction')
    for method in ('SISA','RecEraser'):
        _,component=k.unit_paths(root,{'smoke':method})
        cfg=k.partition_config(root,workspace,p,p['comparison']['requests'][0],method,
                               k.chosen_spec(method,p['inherited_decisions'][method]),component)
        assert cfg['seed']==cfg['protocol']['partition_seed']==43
        path=stage/f'{method}.yaml';k.atomic_text(path,yaml.safe_dump(cfg,sort_keys=False))
        k.partitioned.preflight(workspace,path,'sisa' if method=='SISA' else 'receraser',formal=False,implementation_root=repo)
    result={'status':'SMOKE_PASS','seed':43,'optimizer_steps':1,'predictions':8,
            'partition_preflight_methods':['SISA','RecEraser'],'formal_metrics_published':False,'test_accessed':False}
    k.atomic_json(stage/'result.json',result);k.publish(stage,root/'smoke_result')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--workspace',type=Path,default=Path(r'C:\python\E2URec-main'))
    parser.add_argument('--mode',choices=['Preflight','Freeze','Smoke','Full','Analyze'],default='Preflight')
    parser.add_argument('--method',choices=['All',*k.METHODS],default='All')
    args=parser.parse_args();repo=args.repo.resolve();workspace=args.workspace.resolve();c=load_config(repo)
    root=workspace/c['output_root']
    if args.mode=='Preflight':result=preflight(workspace,audit(repo,workspace,c))
    elif args.mode=='Analyze':
        if not (root/'protocol').exists():result={'status':'NOT_STARTED','test_accessed':False}
        else:
            with k.lock(root):
                k.verify(root/'protocol');p=k.read(root/'protocol/result.json');check_resume_inputs(repo,workspace,p,root)
                if p['followup_config']!=c:raise ValueError('frozen configuration changed')
                result=analyze(root,p)
    else:
        if args.mode=='Smoke':root=workspace/'outputs/ali_stage21_seed43_smoke_v1'
        with k.lock(root):
            if (root/'protocol').exists():
                k.verify(root/'protocol');p=k.read(root/'protocol/result.json');check_resume_inputs(repo,workspace,p,root)
                if p['followup_config']!=c:raise ValueError('frozen configuration changed')
            else:
                print('[Stage21] checking frozen parent artifacts and seed43 Original (no retuning)',flush=True)
                p=audit(repo,workspace,c);preflight(workspace,p);k.freeze(root,'protocol',p)
                check_resume_inputs(repo,workspace,p,root)
            if args.mode=='Freeze':result=preflight(workspace,p)
            elif args.mode=='Smoke':
                if (root/'smoke_result').exists():k.verify(root/'smoke_result');result=k.read(root/'smoke_result/result.json')
                else:result=smoke(root,repo,workspace,p)
            else:
                result=compare(root,repo,workspace,p,args.method)
                if args.method=='All':result=analyze(root,p)
    print(json.dumps({key:value for key,value in result.items() if key!='rows'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
