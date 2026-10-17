from pathlib import Path
import ast
import copy
import inspect
import re
import pytest
from src.bota_short_benchmark import ali_stage20_amazon_baselines_v1 as old
from src.bota_short_benchmark import ali_stage21_amazon_seed43_kernel_v1 as k
from src.bota_short_benchmark import ali_stage21_amazon_seed43_v1 as s

REPO=Path(__file__).resolve().parents[2]

def decisions():
    result={}
    for name,cid in s.EXPECTED.items():
        bad=name in ('PCGrad','SISA')
        result[name]={'selected':None if bad else {'candidate_id':cid},
                      'diagnostic':{'candidate_id':cid} if bad else None,
                      'status':'NO_ELIGIBLE_CANDIDATE' if bad else 'SELECTED'}
    return result

def test_config_and_no_calibration_entry():
    c=s.load_config(REPO)
    assert c['seed']==43 and c['calibration_seed']==42 and not c['retune']
    assert c['output_root']!=c['parent_root']
    assert not hasattr(k,'calibrate') and not hasattr(s,'calibrate')

def test_choices_keep_ineligible_methods():
    d=decisions();s.check_choices(d)
    d['SISA']['status']='SELECTED'
    with pytest.raises(ValueError,match='eligibility'):s.check_choices(d)

def test_refuse_changed_candidate():
    d=decisions();d['IFRU']['selected']['candidate_id']='different'
    with pytest.raises(ValueError,match='choice'):s.check_choices(d)

@pytest.mark.parametrize('name',[n for n,v in vars(k).items()
    if (inspect.isfunction(v) or inspect.isclass(v)) and getattr(v,'__module__',None)==k.__name__
    and n not in ('checked_ifru_solve','ifru_unit')])
def test_execution_kernel_is_only_explicit_seed_migration(name):
    expected=inspect.getsource(getattr(old,name))
    expected=re.sub(r'\b42\b','43',expected).replace('Stage20','Stage21').replace('stage20_executable_budget','stage21_executable_budget')
    if name=='partition_config':
        expected=expected.replace("config=yaml.safe_load(base.read_text(encoding='utf-8'))",
            "config=yaml.safe_load(base.read_text(encoding='utf-8'))\n    config['seed']=43\n    config['protocol']['partition_seed']=43")
    # Decorator wrapping preserves __wrapped__; inspect retrieves the original source.
    assert ast.dump(ast.parse(inspect.getsource(getattr(k,name))))==ast.dump(ast.parse(expected))

def test_metric_seed_and_identical_samples():
    c=old.load_config(REPO)[0];c=copy.deepcopy(c);c['protocol']['seed']=43
    pred={'probability':[.2,.8],'gold_label':[0,1],'sample_order_sha256':'same'}
    r=k.metric('BOTA','BOTA',{'id':'A1','local_indices':[0,1]},pred,pred,pred,[0,1],c)
    assert r['seed']==43 and r['E_U']==0 and r['utility_eligible']
    bad=copy.deepcopy(pred);bad['sample_order_sha256']='different'
    with pytest.raises(ValueError,match='order'):k.metric('BOTA','BOTA',{'id':'A1'},bad,pred,pred,[0,1],c)

def test_manifest_refuses_changed_result(tmp_path):
    k.freeze(tmp_path,'unit',{'seed':43,'test_accessed':False})
    k.verify(tmp_path/'unit')
    (tmp_path/'unit/result.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):k.verify(tmp_path/'unit')

def test_cleanup_cannot_leave_component_root(tmp_path):
    with pytest.raises(ValueError,match='escapes'):k.remove_generated_component(tmp_path,tmp_path/'unrelated')

def test_analyze_is_incomplete_without_all_methods(tmp_path):
    r=s.analyze(tmp_path,{})
    assert r['status']=='INCOMPLETE' and len(r['missing'])==7
