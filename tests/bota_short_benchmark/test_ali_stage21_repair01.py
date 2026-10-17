from pathlib import Path
from types import SimpleNamespace
import copy
import pytest
import torch
from src.bota_short_benchmark import ali_stage21_amazon_seed43_kernel_v1 as k
from src.bota_short_benchmark import ali_stage21_amazon_seed43_v1 as s
from src.bota_short_benchmark import ali_stage21_repair01 as repair

def test_false_residual_is_failure_without_relaxation(monkeypatch):
    seen={}
    def fail(*args,**kwargs):
        seen.update(kwargs);raise RuntimeError('false_residual_convergence')
    monkeypatch.setattr(k.runner,'conjugate_gradient',fail)
    value,diag=k.checked_ifru_solve(None,None,.1,40)
    assert value is None and diag['status']=='NUMERICAL_SOLVER_FAILURE'
    assert not diag['solution_used'] and not diag['fallback_used']
    assert seen==dict(damping=.1,relative_tolerance=.0001,absolute_tolerance=1e-10,
        max_iterations=40,residual_explosion_factor=1000.,pap_tolerance=1e-14,allow_truncated_solution=True)

@pytest.mark.parametrize('message',['CUDA out of memory','other unexpected failure'])
def test_unexpected_error_is_not_swallowed(monkeypatch,message):
    def fail(*args,**kwargs):raise RuntimeError(message)
    monkeypatch.setattr(k.runner,'conjugate_gradient',fail)
    with pytest.raises(RuntimeError,match=message):k.checked_ifru_solve(None,None,.1,40)

def test_success_unchanged(monkeypatch):
    result={'solution':torch.tensor([1.]),'converged':True}
    monkeypatch.setattr(k.runner,'conjugate_gradient',lambda *a,**kw:result)
    actual,diag=k.checked_ifru_solve(None,None,.1,40)
    assert actual is result and diag is None

def test_failed_unit_published_reused_and_not_a_win(tmp_path,monkeypatch):
    model=SimpleNamespace(eval=lambda:None)
    monkeypatch.setattr(k,'fresh_endpoint',lambda *a:(model,['x'],[torch.tensor([1.])],{},SimpleNamespace(pad_token_id=0),None,None,'cpu'))
    monkeypatch.setattr(k.runner,'deterministic_panel',lambda *a:[0])
    monkeypatch.setattr(k.runner,'materialize_resident_panel',lambda *a:(None,None))
    monkeypatch.setattr(k.runner,'make_sample_mean_ggn_operator',lambda *a:None)
    monkeypatch.setattr(k.runner,'estimate_lambda_max',lambda *a,**kw:{'lambda_max_hat':1.})
    monkeypatch.setattr(k.runner,'sample_mean_gradients',lambda *a:([torch.tensor([1.])],None))
    def fail(*a,**kw):raise RuntimeError('false_residual_convergence')
    monkeypatch.setattr(k.runner,'conjugate_gradient',fail)
    monkeypatch.setattr(k.s18,'release',lambda:None)
    p={'config':{'grids':{'IFRU':{'cg_max_iterations':40}}},'registry':{'order':[0]},
       'development_partition':{'select':[0]}}
    request={'id':'A1','forget_train_indices':[0]};spec={'panel':1024,'ratios':[.1]}
    rows=k.ifru_unit(tmp_path,tmp_path,p,request,spec,'comparison',None,{'A1':(None,None)})
    assert len(rows)==1 and rows[0]['status']=='NUMERICAL_SOLVER_FAILURE'
    assert rows[0]['E_U'] is None and not rows[0]['utility_eligible']
    assert rows[0]['seed']==43 and rows[0]['compute_count'] is None
    monkeypatch.setattr(k,'fresh_endpoint',lambda *a:pytest.fail('completed failure must be reused'))
    assert k.ifru_unit(tmp_path,tmp_path,p,request,spec,'comparison',None,{'A1':(None,None)})==rows
    all_rows=[]
    for method in k.ORDER:
        for req in k.COMPARISON:
            r={'method':method,'request':req,'status':'OK','seed':43,'utility_eligible':True,
               'E_U':.1,'D_U':.2,'local_residual':.5,'toward_exact_rate':.5,'overall_auc':.8,'overall_log_loss':.2}
            if method=='IFRU' and req=='A1':r.update(rows[0])
            all_rows.append(r)
    summary=k.summarize_rows(all_rows)
    assert summary['methods']['IFRU']['valid_cells']==2
    assert summary['methods']['IFRU']['stats']['E_U'] is None
    assert summary['BOTA_request_wins']['IFRU']['invalid_comparator_cells']==1
    assert summary['BOTA_request_wins']['IFRU']['lower_E_U_requests']==0

def test_migration_only_allows_exact_bound_files(tmp_path,monkeypatch):
    # The verifier rejects arbitrary predecessors before any ordinary input check.
    record={'id':'stage21-repair01','migrations':{name:{'before':'old','after':'new'} for name in repair.ALLOWED}}
    monkeypatch.setattr(repair.k,'read',lambda path:record)
    p={'checks':{'source_sha256':{name:'unknown' for name in repair.ALLOWED}}}
    with pytest.raises(ValueError,match='predecessor'):repair.check(tmp_path,tmp_path,p,tmp_path)
