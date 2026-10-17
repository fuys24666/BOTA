"""Common fixed Development behavior/ranking/MIA evaluation; no FinalTest loader."""
from __future__ import annotations
import copy
import gc
import math
from collections import Counter
import numpy as np
import torch
from transformers import T5Tokenizer
from . import ali_stage17_common as c
from . import ali_stage4_attack_v1 as s4
from . import ali_stage8_bota_response_sweep_v1 as s8
from src.diagnostics.t5_reconstructed_official import JsonPromptDataset

class Progress:
    def __init__(self,data,p):self.data=data;self.p=p;self.passes=0
    def __len__(self):return len(self.data)
    def __getitem__(self,i):return self.data[i]
    def collate_fn(self,rows):
        start=rows[0]['sample_id'];end=rows[-1]['sample_id']+1
        if start==0:self.passes+=1
        if start==0 or start%4000==0 or end==len(self):
            c.atomic_json(self.p,{'pass':self.passes,'done':end,'total':len(self),'test_accessed':False})
            print(f'[Stage17] pass={self.passes} {end}/{len(self)}',flush=True)
        return self.data.collate_fn(rows)

def datasets(w,contract,seed,request,order=None):
    ctx=contract['contexts'][str(seed)];benchmark=ctx['benchmark']
    tokenizer=T5Tokenizer.from_pretrained(str(w/contract['dependencies']['pretrained_model_directory']['path']),local_files_only=True)
    dev=JsonPromptDataset(w/benchmark['source']['development_json'],tokenizer)
    train_users,dev_users,_=c.s2._source_users(w,benchmark)
    plan=c.read(w/c.s15.ROOT/'protocol/candidates.json')
    users,movies=c.s15.lineage.load_raw_metadata(w/'data/ml-1m/raw_data')
    rank=c.s15.RankingDataset(plan,tokenizer,users,movies)
    # Same ordered training slots, including repeated exposures in a long trajectory.
    order=ctx['registry']['order'] if order is None else order
    records=c.read(w/benchmark['source']['train_json']);positions=[i for i in order if train_users[i] in request]
    nonmember=[i for i,u in enumerate(dev_users) if u in request]
    member_users=[train_users[i] for i in positions];nonmember_users=[dev_users[i] for i in nonmember]
    covered=set(member_users)&set(nonmember_users)
    matched_positions=[i for i in positions if train_users[i] in covered]
    nonmember=[i for i in nonmember if dev_users[i] in covered]
    attack=copy.copy(dev);attack.records=[records[i] for i in matched_positions]+[dev.records[i] for i in nonmember]
    members=[train_users[i] for i in matched_positions];nonmembers=[dev_users[i] for i in nonmember]
    return {'binary':dev,'ranking':rank,'attack':attack},dev_users,plan,members,nonmembers

def predict(w,contract,entry,data,lane,dataset_hash):
    key='predict_'+c.canonical_hash([entry['directory_sha256'],lane,dataset_hash])[:20]
    def work(dest):
        p=c.s16.s10._predict_endpoint(w,contract['contexts'][str(entry['seed'])]['benchmark'],entry,
            Progress(data,dest/'progress.json'),c.gpu(),list(range(len(data))))
        # Malformed order/length is a programming failure, nonfinite model output a reported scientific outcome.
        if p['sample_order_sha256']!=c.canonical_hash(list(range(len(data)))) or any(len(p[k])!=len(data) for k in ('probability','gold_label','answer_loss')):
            raise RuntimeError('prediction order/length mismatch')
        finite=bool(np.isfinite(p['probability']).all() and np.isfinite(p['answer_loss']).all())
        if not finite:
            c.atomic_torch_save(dest/'nonfinite_prediction.pt',p)
            return {'status':'NONFINITE_PREDICTIONS','endpoint':entry,'lane':lane}
        c.s16.validate_prediction(p,len(data))
        c.atomic_json(dest/'predictions.json',p)
        return {'status':'COMPLETED','endpoint':entry,'lane':lane,'rows':len(data)}
    result=c.task(w,contract,key,work)
    return c.read(c.path(w,key)/'predictions.json') if result['status']=='COMPLETED' else None

def behavior(pred,users,request):
    chosen=[i for i,u in enumerate(users) if u in request];support=Counter(users[i] for i in chosen)
    if not chosen:return {'status':'NO_REQUESTED_USER_SUPPORT','missing_users':sorted(request)}
    original=pred['Original'];exact=pred['Exact-Masked'];result=[]
    if original is None or exact is None:raise RuntimeError('nonfinite Original/Exact reference; cannot normalize outcomes')
    valid={k:v for k,v in pred.items() if v is not None}
    for name,p in pred.items():
        if p is None:result.append({'method':name,'status':'NONFINITE_PREDICTIONS'});continue
        if p['gold_label']!=original['gold_label'] or p['sample_order_sha256']!=original['sample_order_sha256']:raise ValueError('prediction rows mismatch')
        row=s8.behavior.behavior_metrics(name,p,exact,original,chosen,users,2000,42)
        a=np.array(p['probability']);o=np.array(original['probability']);e=np.array(exact['probability'])
        pc=c.s5._classification(p);oc=c.s5._classification(original)
        eu=float(np.abs(a[chosen]-e[chosen]).mean());du=float(np.abs(o[chosen]-e[chosen]).mean())
        guard={'auc_damage':oc['auc']-pc['auc'],'logloss_damage':pc['log_loss']-oc['log_loss'],'prediction_collapse':pc['prediction_collapse']}
        guard['pass']=guard['auc_damage']<=.005 and guard['logloss_damage']<=.01 and not guard['prediction_collapse']
        result.append({'method':name,'status':'COMPLETED','E_U':eu,'D_U':du,'residual':eu/du if du else None,
            'behavior':row,'overall':pc,'utility':guard,'mean_answer_loss':float(np.mean(p['answer_loss'])),
            'requested':c.s5._classification(p,chosen),'retained':c.s5._classification(p,[i for i,u in enumerate(users) if u not in request]),
            'rms_to_exact':float(np.sqrt(np.mean((a[chosen]-e[chosen])**2))),
            'jsd_to_exact':float(c.s16.s10.evaluation.bernoulli_jsd(a[chosen],e[chosen]).mean()),
            'P5':{**c.s16.s10._metric(name,valid,chosen,users,{'evaluation':c.s16.s10.EXPECTED_EVALUATION},42),
                  'test_accessed':False,'evaluation_split':'Development'}
                  if {'Retain-Retrain-P5','FullControl-P5'}<=set(valid) else None})
    return {'status':'COMPLETED','rows':result,'samples':len(chosen),'supported_users':len(support),
        'requested_users':len(request),'missing_users':sorted(request-set(support))}

def attacks(pred,members,nonmembers,formal,utility):
    if not members or not nonmembers:return {'status':'NO_MATCHED_USER_SUPPORT','formal_pass':False}
    caches={};rows=[];n=len(members)
    for name,p in pred.items():
        if p is None:continue
        m=[s4._row_from_prediction(p,i,u) for i,u in enumerate(members)]
        nm=[s4._row_from_prediction(p,n+i,u) for i,u in enumerate(nonmembers)];caches[name]=(m,nm)
        for score,fn in s4.ATTACKS.items():
            rows.append({'method':name,'attack':score,**s4._attack(m,nm,fn,42,2000)})
    if not {'Original','Exact-Masked','BOTA64'}<=set(caches):return {'status':'BOTA_OR_REFERENCE_UNAVAILABLE','rows':rows,'formal_pass':False}
    paired=s4._paired_bootstrap({'Original':caches['Original'],'Exact-Masked':caches['Exact-Masked'],'BOTA':caches['BOTA64']},'negative_answer_loss',42,2000)
    ref=paired['reference_signal'];gain=paired['bota_gap_improvement']
    signal=ref['point_estimate']>=.02 and ref['percentile_95_ci'][0]>0
    improves=gain['point_estimate']>0 and gain['percentile_95_ci'][0]>0
    return {'status':'COMPLETED','rows':rows,'paired':paired,'reference_signal_pass':signal,'improvement_pass':improves,
        'formal_interpretation_permitted':formal,'formal_pass':bool(formal and signal and improves and utility),
        'member_rows':len(members),'nonmember_rows':len(nonmembers),'matched_users':len(set(members)),
        'privacy_guarantee':False,'threshold_metrics':'Development diagnostic; no separately trained LiRA classifier'}

def parameter_metrics(w,entries):
    """Only compare shared fixed-A coordinates; ensembles have no such delta."""
    refs={e['method']:e for e in entries}
    original=c.torch_read(w/refs['Original']['path']/'adapter_model.pt')
    exact=c.torch_read(w/refs['Exact-Masked']['path']/'adapter_model.pt')
    names=list(original['B']);delta=torch.cat([(exact['B'][n]-original['B'][n]).reshape(-1) for n in names])
    rows=[]
    for e in entries:
        if e['kind']!='fixed_ab':
            rows.append({'method':e['method'],'status':'NOT_A_SHARED_PARAMETER_COORDINATE'});continue
        state=c.torch_read(w/e['path']/'adapter_model.pt')
        if set(state['B'])!=set(names) or not c.s2._same_tensor_map(state['A'],original['A']):
            rows.append({'method':e['method'],'status':'NOT_A_SHARED_PARAMETER_COORDINATE'});continue
        predicted=torch.cat([(state['B'][n]-original['B'][n]).reshape(-1) for n in names])
        metric=s8.p2b.vector_metrics(delta,predicted)
        rows.append({'method':e['method'],'status':'COMPLETED',
            **{k:None if isinstance(v,float) and not math.isfinite(v) else v for k,v in metric.items()}})
    return rows


def evaluate(w,contract,family,seed,sid,request,entries,formal_attack=False,order=None,with_attack=True):
    key=f'eval_{family}_s{seed}_'+c.canonical_hash(sid)[:12]
    def work(dest):
        data,users,plan,members,nonmembers=datasets(w,contract,seed,set(request),order)
        hashes={'binary':c.canonical_hash(data['binary'].records),'ranking':c.canonical_hash(plan),
                'attack':c.canonical_hash([data['attack'].records,members,nonmembers])}
        lane_predictions={}
        # Token count check is independent of model scores, and catches incompatible long prompts.
        data['ranking'].validate_tokens()
        for lane in ('binary','ranking','attack') if with_attack else ('binary','ranking'):
            lane_predictions[lane]={e['method']:predict(w,contract,e,data[lane],lane,hashes[lane])
                if len(data[lane]) else None for e in entries}
        bp=lane_predictions['binary'];behavior_result=behavior(bp,users,set(request))
        ranking={name:c.s15.metrics_for(plan,p['probability'],set(request)) if p is not None else {'status':'NONFINITE_PREDICTIONS'}
                 for name,p in lane_predictions['ranking'].items()}
        bota=next((r for r in behavior_result.get('rows',[]) if r['method']=='BOTA64'),{})
        attack=attacks(lane_predictions['attack'],members,nonmembers,formal_attack,bota.get('utility',{}).get('pass',False)) if with_attack else None
        # Explicit absence, not a dropped scenario, if lambda64 diverged in transport.
        required={'BOTA64','FM0','FM64'} if family=='primary' else {'BOTA64'}
        missing=sorted(required-set(bp));nonfinite=[f'{lane}/{name}' for lane,values in lane_predictions.items()
            for name,p in values.items() if p is None and len(data[lane])]
        unavailable='BOTA64' not in bp or bp['BOTA64'] is None
        return {'status':'COMPLETED_WITH_UNAVAILABLE_OUTCOMES' if missing or nonfinite else 'COMPLETED','family':family,'seed':seed,'request':sid,
                'lambda':64,'behavior':behavior_result,'ranking':ranking,'attack':attack,'parameter':parameter_metrics(w,entries),
                'all_endpoint_methods':[e['method'] for e in entries],'BOTA64_unavailable':unavailable,
                'missing_required_methods':missing,'nonfinite_predictions':nonfinite,
                'attack_requested_users':len(set(request)),'attack_matched_users':len(set(members)),
                'attack_excluded_users':sorted(set(request)-set(members)),
                'ranking_candidates':plan['candidate_count'],'ranking_user_count':len(plan['users'])}
    return c.task(w,contract,key,work)
