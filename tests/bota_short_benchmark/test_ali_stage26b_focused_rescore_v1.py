import ast
import copy
from pathlib import Path

import pytest
from src.bota_short_benchmark import ali_stage26b_focused_rescore_v1 as s


def parent_fixture():
    parent={'specs':[],'jobs':{},'cohorts':{},'sequence_cells':{},'config':{'audit':{'samples':32}},'hashes':{}}
    for family,count in s.EXPECTED_GROUPS.items():
        for i in range(count):
            key=f'{family}_{i}';methods=['Original','Exact-Masked','BOTA64','IFRU','FM64']
            if family=='compression':methods.append('Uncompressed-FS64')
            spec={'id':key,'family':family,'seed':41+i%3,'request':str(i),'users':[i],
                  'order':None,'formal_attack':family=='primary','lanes':['binary','ranking','attack'],
                  'cohorts':{},'jobs':{},'entries':[{'method':m} for m in methods]}
            for lane in spec['lanes']:
                ck=key+'_'+lane;spec['cohorts'][lane]=ck;spec['jobs'][lane]={}
                parent['cohorts'][ck]={'scope':'Development','lane':lane,'indices':[0,1],'users':[i,i]}
                for method in methods:
                    jid=ck+'_'+method;spec['jobs'][lane][method]=jid
                    parent['jobs'][jid]={'id':jid,'cohort':ck,'method':method,'path':'fixed_'+method,'status':'READY'}
            parent['specs'].append(spec)
    return parent


def test_reduction_keeps_requests_seeds_and_parent_immutable():
    parent=parent_fixture();before=s.base.canonical_hash(parent)
    plan=s.reduce_plan(parent,{'audit':parent['config']['audit']})
    s.validate(plan,parent)
    assert before==s.base.canonical_hash(parent)
    assert len(plan['specs'])==133
    assert all(c['lane']!='ranking' for c in plan['cohorts'].values())
    assert sum('attack' in x['jobs'] for x in plan['specs'])==15
    assert all(j['method'] not in ('IFRU','FM64') for j in plan['jobs'].values())


@pytest.mark.parametrize('key,value',[('users',[999]),('seed',999),('request','chosen_by_result'),('formal_attack',False)])
def test_request_and_gate_changes_rejected(key,value):
    parent=parent_fixture();plan=s.reduce_plan(parent,{'audit':parent['config']['audit']})
    plan['specs'][0][key]=value
    with pytest.raises(ValueError):s.validate(plan,parent)


def test_no_extra_lane_or_method():
    parent=parent_fixture();plan=s.reduce_plan(parent,{'audit':parent['config']['audit']})
    plan['specs'][0]['lanes'].append('ranking')
    with pytest.raises(ValueError,match='lane'):s.validate(plan,parent)


def test_original_full_plan_cannot_be_reused_as_focused():
    with pytest.raises((ValueError,KeyError)):s.validate(parent_fixture(),parent_fixture())


def test_no_training_or_full_executor_calls():
    tree=ast.parse(Path(s.__file__).read_text(encoding='utf-8'))
    forbidden={'step','backward','build_bank','run_phase','materialize','screen','time_serving'}
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not [n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr in forbidden]
    assert not [n for n in calls if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
                and n.func.value.id=='base' and n.func.attr in ('execute','analyze','smoke')]


def test_missing_prediction_does_not_publish_complete_summary(tmp_path):
    spec={'id':'one','family':'primary','jobs':{'binary':{'Original':'absent'}}}
    assert s.analyze(tmp_path,{'specs':[spec]})['status']=='INCOMPLETE'
    assert not (tmp_path/'summary').exists()


def test_summary_is_explicitly_nonranking_and_no_fm_claim(tmp_path,monkeypatch):
    spec={'id':'one','family':'sequential','seed':42,'request':'u','jobs':{'binary':{'Original':'j'}}}
    plan={'specs':[spec],'sequence_cells':{'42':[{'union_hash':'u','order':i} for i in range(4)]},
          'scope_exclusions':['ranking','FM']}
    contract=s.base.canonical_hash(plan)
    s.base.task(tmp_path,'predictions/j',contract,lambda p:{'status':'OK','prediction':{}})
    monkeypatch.setattr(s.base,'evaluate_spec',lambda *a:{'status':'COMPLETED','family':'sequential','seed':42,
        'request':'u','behavior':{'rows':[{'method':'Original','status':'COMPLETED'}]},'attack':None,'ranking':{}})
    result=s.analyze(tmp_path,plan)
    assert result['sequential_order_cells']==4 and not result['ranking_evaluated']
    assert not (tmp_path/'summary/All/ranking.csv').exists()
    report=(tmp_path/'summary/All/REPORT.md').read_text()
    assert 'No new ranking measurements' in report
    assert s.analyze(tmp_path,plan)==result


def test_invariance_failure_blocks_summary(tmp_path):
    plan={'specs':[{'id':'one','family':'primary','jobs':{'binary':{'BOTA64':'j'}}}]}
    contract=s.base.canonical_hash(plan)
    s.base.task(tmp_path,'predictions/j',contract,lambda p:{'status':'INVARIANCE_FAILED'})
    with pytest.raises(RuntimeError,match='invariance'):s.analyze(tmp_path,plan)
