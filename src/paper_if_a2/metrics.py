from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, matthews_corrcoef, roc_auc_score
from scipy.stats import spearmanr


def bernoulli_jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p=np.clip(np.asarray(p,float),1e-12,1-1e-12); q=np.clip(np.asarray(q,float),1e-12,1-1e-12); m=(p+q)/2
    kl=lambda a,b: a*np.log(a/b)+(1-a)*np.log((1-a)/(1-b))
    return .5*kl(p,m)+.5*kl(q,m)


def full_vocab_jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p=np.clip(np.asarray(p,float),1e-30,None); q=np.clip(np.asarray(q,float),1e-30,None); p/=p.sum(-1,keepdims=True); q/=q.sum(-1,keepdims=True); m=(p+q)/2
    return .5*np.sum(p*np.log(p/m),axis=-1)+.5*np.sum(q*np.log(q/m),axis=-1)


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if abs(denominator) <= 1e-15 else numerator/denominator


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str,float]:
    y=np.asarray(labels,int); p=np.clip(np.asarray(probabilities,float),1e-12,1-1e-12); pred=(p>=.5).astype(int)
    return {"auc":float(roc_auc_score(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),"accuracy":float(accuracy_score(y,pred)),"positive_rate":float(pred.mean()),"confidence_mean":float(np.maximum(p,1-p).mean()),"probability_mean":float(p.mean()),"probability_std":float(p.std()),"prediction_collapse":bool(p.std()<1e-6)}


def clustered_bootstrap(values: Sequence[float], users: Sequence[int], *, resamples: int=2000, seed: int=42, statistic: Callable[[np.ndarray],float]=np.mean) -> dict[str,Any]:
    x=np.asarray(values,float); u=np.asarray(users); unique=np.unique(u); rng=np.random.default_rng(seed); estimates=[]
    groups={user:np.flatnonzero(u==user) for user in unique}
    for _ in range(resamples):
        chosen=rng.choice(unique,size=len(unique),replace=True); idx=np.concatenate([groups[user] for user in chosen]); estimates.append(float(statistic(x[idx])))
    return {"point":float(statistic(x)),"ci_lower":float(np.percentile(estimates,2.5)),"ci_upper":float(np.percentile(estimates,97.5)),"samples":len(x),"users":len(unique),"resamples":resamples}


def clustered_ratio(numerator: Sequence[float], denominator: Sequence[float], users: Sequence[int], *, resamples:int=2000, seed:int=42) -> dict[str,Any]:
    n=np.asarray(numerator,float);d=np.asarray(denominator,float);u=np.asarray(users);unique=np.unique(u);groups={user:np.flatnonzero(u==user) for user in unique};rng=np.random.default_rng(seed);draws=[]
    for _ in range(resamples):
        chosen=rng.choice(unique,size=len(unique),replace=True);idx=np.concatenate([groups[user] for user in chosen]);value=safe_ratio(float(n[idx].mean()),float(d[idx].mean()))
        if value is not None:draws.append(value)
    point=safe_ratio(float(n.mean()),float(d.mean()))
    return {"point":point,"ci_lower":None if not draws else float(np.percentile(draws,2.5)),"ci_upper":None if not draws else float(np.percentile(draws,97.5)),"samples":len(n),"users":len(unique),"resamples":resamples}


def membership_metrics(member_scores:Sequence[float],member_users:Sequence[int],nonmember_scores:Sequence[float],nonmember_users:Sequence[int],*,resamples:int=2000,seed:int=42)->dict[str,Any]:
    scores=np.asarray(list(member_scores)+list(nonmember_scores),float);labels=np.asarray([1]*len(member_scores)+[0]*len(nonmember_scores));users=np.asarray(list(member_users)+list(nonmember_users));prediction=(scores>=np.median(scores)).astype(int);point={"auc":float(roc_auc_score(labels,scores)),"accuracy":float(accuracy_score(labels,prediction)),"log_loss":float(log_loss(labels,1/(1+np.exp(-np.clip(scores,-30,30))),labels=[0,1])),"score_mean":float(scores.mean()),"score_std":float(scores.std())}
    unique=np.unique(users);groups={user:np.flatnonzero(users==user) for user in unique};rng=np.random.default_rng(seed);draws=[]
    for _ in range(resamples):
        chosen=rng.choice(unique,size=len(unique),replace=True);idx=np.concatenate([groups[user] for user in chosen])
        if len(np.unique(labels[idx]))==2:draws.append(float(roc_auc_score(labels[idx],scores[idx])))
    return {**point,"auc_ci_lower":float(np.percentile(draws,2.5)),"auc_ci_upper":float(np.percentile(draws,97.5)),"samples":len(scores),"users":len(unique),"resamples":resamples}


def direction_metrics(original: Sequence[float], method: Sequence[float], retrain: Sequence[float], epsilon: float=.001) -> dict[str,Any]:
    o=np.asarray(original); m=np.asarray(method); r=np.asarray(retrain); target=r-o; actual=m-o; product=target*actual
    labels=np.where(np.abs(target)<=epsilon,"equivalent",np.where(product>0,"toward","away")); truth=np.sign(target); pred=np.sign(actual)
    nonzero=(truth!=0); spearman=None if np.std(actual)==0 or np.std(target)==0 else float(spearmanr(actual,target).statistic)
    return {"toward":int(np.sum(labels=="toward")),"away":int(np.sum(labels=="away")),"equivalent":int(np.sum(labels=="equivalent")),"sign_accuracy":float(np.mean(pred[nonzero]==truth[nonzero])) if nonzero.any() else None,"balanced_accuracy":float(np.mean(pred[nonzero]==truth[nonzero])) if nonzero.any() else None,"mcc":float(matthews_corrcoef(truth[nonzero],pred[nonzero])) if nonzero.any() else None,"spearman":spearman,"prediction_agreement":float(np.mean((m>=.5)==(r>=.5)))}


def classify_paper(*, valid:bool, forgetting:bool, utility:bool, mia:bool, efficient:bool) -> dict[str,str]:
    if not valid:return {"category":"PAPER-D","next_action":"stop_invalid_or_conflicted"}
    if not forgetting or not utility:return {"category":"PAPER-C","next_action":"do_not_access_test_stop_primary_claim"}
    if not mia or not efficient:return {"category":"PAPER-B","next_action":"report_tradeoff_and_complete_ablation_without_changing_primary_method"}
    return {"category":"PAPER-A","next_action":"freeze_method_and_run_additional_seeds_datasets_before_final_test"}
