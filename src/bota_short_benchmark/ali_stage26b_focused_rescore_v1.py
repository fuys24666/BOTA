"""Independent, prespecified reduced Stage26 workload. No ranking or training.

Inherits exact model/cohort identities and scoring gates from the already frozen
full plan. Never changes it, selects requests by results, or uses old predictions.
"""
from __future__ import annotations
import argparse
import copy
import gc
import json
from collections import Counter
from pathlib import Path

from . import ali_stage26_stress_mask_rescore_v1 as base

SCHEMA = 'bota-ali-stage26b-focused-rescore-v1'
ROOT = 'outputs/ali_stage26b_focused_rescore_v1'
CONFIG = 'configs/ali_stage26b_focused_rescore_v1.json'
METHODS = {'Original', 'Exact-Masked', 'BOTA64', 'Uncompressed-FS64'}
ATTACK_FAMILIES = {'primary', 'stronger'}
EXPECTED_GROUPS = {'primary':6, 'stronger':9, 'sequential':39, 'horizon':6, 'compression':72, 'relay':1}


def reduce_plan(parent, config):
    plan = copy.deepcopy(parent)
    plan.update(schema=SCHEMA, config=config, parent_plan_sha256=base.canonical_hash(parent))
    for spec in plan['specs']:
        lanes = ['binary'] + (['attack'] if spec['family'] in ATTACK_FAMILIES else [])
        spec['lanes'] = lanes
        spec['cohorts'] = {lane: spec['cohorts'][lane] for lane in lanes}
        spec['jobs'] = {lane: {m:j for m,j in spec['jobs'][lane].items() if m in METHODS} for lane in lanes}
        spec['entries'] = [e for e in spec['entries'] if e['method'] in METHODS]
    ids = {j for spec in plan['specs'] for lane in spec['jobs'].values() for j in lane.values()}
    plan['jobs'] = {j:plan['jobs'][j] for j in sorted(ids)}
    cohorts = {j['cohort'] for j in plan['jobs'].values()}
    plan['cohorts'] = {k:v for k,v in plan['cohorts'].items() if k in cohorts}
    plan['scope_exclusions'] = [
        'All new ranking inference and stress-scenario ranking claims',
        'Baseline/Fisher-source-only/FM rescoring (existing separately validated main comparisons remain)',
        'Sequential/horizon attack diagnostics (primary and all nine stronger-request attacks remain)',
        'No new privacy guarantee, training, parameter search, or FinalTest access']
    return plan


def validate(plan, parent):
    if plan['schema'] != SCHEMA or plan['parent_plan_sha256'] != base.canonical_hash(parent):
        raise ValueError('parent identity changed')
    if dict(Counter(s['family'] for s in plan['specs'])) != EXPECTED_GROUPS:
        raise ValueError('request/seed coverage changed')
    original = {s['id']:s for s in parent['specs']}
    for spec in plan['specs']:
        old = original[spec['id']]
        for key in ('family','seed','request','users','formal_attack','order'):
            if spec[key] != old[key]: raise ValueError('frozen request or attack authority changed')
        expected_lanes = ['binary'] + (['attack'] if spec['family'] in ATTACK_FAMILIES else [])
        if spec['lanes'] != expected_lanes or set(spec['jobs']) != set(expected_lanes):
            raise ValueError('unexpected scoring lane')
        expected_methods = METHODS if spec['family']=='compression' else METHODS-{'Uncompressed-FS64'}
        for lane, jobs in spec['jobs'].items():
            if set(jobs) != expected_methods: raise ValueError('missing requested method/reference')
            for method, jid in jobs.items():
                if jid != old['jobs'][lane][method] or plan['jobs'][jid] != parent['jobs'][jid]:
                    raise ValueError('model identity or scoring cohort changed')
    if plan['sequence_cells'] != parent['sequence_cells']: raise ValueError('sequential expansion changed')
    for k,cohort in plan['cohorts'].items():
        if cohort != parent['cohorts'][k] or cohort['scope']!='Development' or cohort['lane']=='ranking':
            raise ValueError('unexpected cohort change or FinalTest/ranking access')
    if plan['config']['audit'] != parent['config']['audit']: raise ValueError('audit thresholds changed')


def prepare(repo, workspace, config):
    path = workspace/base.ROOT/'protocol'
    parent = base.verify(path)['plan']
    base.check(parent)
    if config['schema']!=SCHEMA or config['lambda']!=64 or config['optimizer_updates']!=0 or config['selection_performed'] or config['test_accessed']:
        raise ValueError('frozen reduced scope changed')
    plan = reduce_plan(parent, config)
    validate(plan, parent)
    for p in (path/'manifest.json',path/'result.json',repo/CONFIG,Path(__file__),
              repo/'scripts/bota_if/run_ali_stage26b_focused_rescore_v1.ps1'):
        plan['hashes'][str(p.resolve())] = base.sha256_file(p)
    return plan


def audit(stage, workspace, plan):
    jobs = list(plan['jobs'].values())
    estimate = base.estimate(workspace, plan)
    hours = estimate['inference_point_hours']
    estimate['full_wall_hours_range'] = [round(hours+1,1), round(hours*1.1+3,1)]
    estimate['basis'] += '; reduced plan: 1–3h CPU/bootstrap/check allowance; no ranking'
    base.s25.s24.write_csv(stage/'endpoint_inventory.csv', jobs)
    base.atomic_json(stage/'runtime_estimate.json', estimate)
    base.atomic_json(stage/'missing_weights.json', [j for j in jobs if j['status']!='READY'])
    base.atomic_json(stage/'scope.json', {'groups':EXPECTED_GROUPS,'exclusions':plan['scope_exclusions'],
                                        'parent_plan_sha256':plan['parent_plan_sha256']})
    base.atomic_json(stage/'unaffected_records.json', {'reuse':plan['reuse'],'parameters':plan['parameter_records']})
    base.atomic_text(stage/'REPORT.md', '# Stage26B reduced frozen rescore\n\n'
        'Same three seeds, all original request groups, full 20,000 Development binary samples.\n'
        'No ranking inference, baseline resweep, FM behavioral rescore or parameter selection.\n'
        'MIA: original primary requests + all nine stronger requests; stronger formal interpretation only high_k4.\n'
        'Sequential/horizon attack outcomes and stress ranking are not evaluated by this profile.\n'
        'Parameter/SVD/storage/timing are verified reuse, not remeasurements.\n')
    return {'status':'AUDITED','evaluation_groups':len(plan['specs']),
            'jobs':len(jobs),'job_statuses':dict(Counter(j['status'] for j in jobs)),
            'estimate':estimate,'full_plan_unchanged':True}


def analyze(root, plan, family='All'):
    contract=base.canonical_hash(plan); done=[]; pending=[]
    for spec in base.selected_specs(plan, family):
        ids={j for lane in spec['jobs'].values() for j in lane.values()}
        if any(not (root/'predictions'/j).exists() for j in ids):pending.append(spec['id']);continue
        def work(stage):
            results={j:base.verify(root/'predictions'/j,contract) for j in ids}
            if any(r['status']=='INVARIANCE_FAILED' for r in results.values()):raise RuntimeError('batch invariance failed')
            out=base.evaluate_spec(spec,plan,results)
            out['evaluation_profile']='focused_binary_and_selected_attack'
            out['ranking_evaluated']=False
            out['attack_evaluated']='attack' in spec['jobs']
            return out
        done.append(base.task(root,'metrics/'+spec['id'],contract,work))
    if pending:return {'status':'INCOMPLETE','family':family,'completed_groups':len(done),'pending_groups':len(pending)}
    def summary(stage):
        behavior=[];attacks=[]
        for r in done:
            common={k:r[k] for k in ('family','seed','request')}
            behavior.extend({**common,**row} for row in (r.get('behavior') or {}).get('rows',[]))
            if r.get('attack') is not None:attacks.append({**common,**r['attack']})
        lookup={(r['seed'],r['request']):r for r in done if r['family']=='sequential'}
        sequence=[{'seed':int(seed),**row,'evaluation':lookup[int(seed),row['union_hash']]}
                  for seed,rows in plan['sequence_cells'].items() for row in rows
                  if (int(seed),row['union_hash']) in lookup]
        base.s25.s24.write_csv(stage/'behavior.csv',behavior)
        base.atomic_json(stage/'attacks.json',attacks)
        base.atomic_json(stage/'sequential_all_cells.json',sequence)
        base.atomic_json(stage/'all_results.json',done)
        base.atomic_text(stage/'REPORT.md','# Stage26B reduced frozen stress validation\n\n'
            'BOTA lambda64 and Original/Exact-Masked references only; compression also includes uncompressed FS64.\n'
            'No new ranking measurements. Do not extend main-experiment NDCG claims to these stress settings.\n'
            'Primary and all nine stronger-request MIA records are reported; formal stronger interpretation is high_k4 only.\n'
            'Sequential/horizon attacks and FM/baseline rescoring were excluded prospectively for compute budget.\n'
            'All 60 sequential order/size records are expanded from identical-union caches.\n'
            'Unavailable references or methods are not successes. No old behavioral scores fill missing entries.\n'
            'No training, tuning, FinalTest access or privacy certification.\n')
        return {'status':'COMPLETED_WITH_UNAVAILABLE' if any(r['status']!='COMPLETED' for r in done) else 'COMPLETED',
                'family':family,'groups':len(done),'behavior_rows':len(behavior),'attack_groups':len(attacks),
                'sequential_order_cells':len(sequence),'ranking_evaluated':False,
                'scope_exclusions':plan['scope_exclusions']}
    return base.task(root,'summary/'+family,contract,summary)


def execute(root, workspace, plan, family):
    contract=base.canonical_hash(plan)
    ids={j for s in base.selected_specs(plan,family) for lane in s['jobs'].values() for j in lane.values()}
    total=len(ids); complete=0
    for cohort_id,cohort in plan['cohorts'].items():
        if cohort['lane'] not in ('binary','attack'):raise ValueError('ranking is excluded')
        data=None
        try:
            for jid in sorted(ids):
                job=plan['jobs'][jid]
                if job['cohort']!=cohort_id:continue
                dest=root/'predictions'/jid
                if dest.exists():
                    if base.verify(dest,contract)['status']=='INVARIANCE_FAILED':raise RuntimeError('frozen invariance failure')
                    complete+=1;continue
                if job['status']!='READY' or not cohort['indices']:
                    base.task(root,'predictions/'+jid,contract,lambda stage:{'status':job['status'] if job['status']!='READY' else 'NO_MATCHED_USER_SUPPORT','prediction':None})
                    complete+=1;continue
                if data is None:data=base.dataset(workspace,plan,cohort)
                print(f'[Stage26B] unit {complete+1}/{total} {cohort_id}/{job["method"]}',flush=True)
                def work(stage):
                    def progress(shard,n,count):
                        base.atomic_json(stage/'progress.json',{'job':jid,'cohort':cohort_id,'method':job['method'],
                            'done':n,'total':count,'completed_units':complete,'total_units':total})
                        print(f'[Stage26B] {cohort_id}/{job["method"]} {n}/{count}',flush=True)
                    try:return base.scoring.score_endpoint(workspace,base.scoring_cohort(cohort),job,data,plan['config'],progress,base.read)
                    except ValueError as error:
                        if str(error) not in ('nonfinite model logits','nonfinite scores','nonfinite prediction'):raise
                        return {'status':'NONFINITE_PREDICTIONS','error':str(error),'prediction':None}
                result=base.task(root,'predictions/'+jid,contract,work)
                if result['status']=='INVARIANCE_FAILED':raise RuntimeError('batch invariance failed; diagnostics retained')
                complete+=1
        finally:del data;gc.collect()
    base.check(plan)
    return analyze(root,plan,family)


def smoke(root,workspace,plan):
    contract=base.canonical_hash(plan)
    spec=next(s for s in plan['specs'] if s['family']=='primary' and s['seed']==42 and s['request']=='L8')
    def work(stage):
        checks=[]
        for lane in ('binary','attack'):
            cohort=copy.deepcopy(plan['cohorts'][spec['cohorts'][lane]])
            data=base.dataset(workspace,plan,cohort)
            cohort['indices']=cohort['indices'][:64]
            cohort['users']=cohort['users'][:64];cohort['lane']='smoke'
            job=copy.deepcopy(plan['jobs'][spec['jobs'][lane]['BOTA64']]);job['provenance']['users']=[]
            r=base.scoring.score_endpoint(workspace,base.scoring_cohort(cohort),job,data,plan['config'],lambda *a:None,base.read)
            checks.append({'lane':lane,'status':r['status'],'audits':r['audits'],'full_cohort_gate':r.get('full_cohort_gate')})
            if r['status']!='OK':raise RuntimeError(f'smoke failed: {lane}')
            del data;gc.collect()
        base.atomic_json(stage/'checks.json',checks)
        return {'status':'SMOKE_PASS','lanes':2,'samples_per_lane':64,'full_scoring_cache_written':False}
    return base.task(root,'smoke',contract,work)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--mode',choices=('Preflight','Freeze','Smoke','Full','Analyze'),default='Preflight')
    p.add_argument('--family',choices=('All',*base.FAMILIES),default='All')
    args=p.parse_args();repo=args.repo.resolve();workspace=args.workspace.resolve();root=workspace/ROOT
    config=base.read(repo/CONFIG)
    with base.s25.s24.s23.s20.lock(root):
        if (root/'protocol').exists():
            plan=base.verify(root/'protocol')['plan']
            if plan['config']!=config:raise ValueError('frozen focused config changed')
            base.check(plan)
            validate(plan,base.verify(workspace/base.ROOT/'protocol')['plan'])
        else:plan=prepare(repo,workspace,config)
        contract=base.canonical_hash(plan)
        if args.mode=='Preflight':out=audit(root,workspace,plan)
        else:
            base.task(root,'protocol',contract,lambda stage:{'status':'FROZEN','plan':plan})
            audited=base.task(root,'audit',contract,lambda stage:audit(stage,workspace,plan))
            if args.mode=='Freeze':out=audited
            elif args.mode=='Smoke':out=smoke(root,workspace,plan)
            elif args.mode=='Analyze':out=analyze(root,plan,args.family)
            else:out=execute(root,workspace,plan,args.family)
        print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
