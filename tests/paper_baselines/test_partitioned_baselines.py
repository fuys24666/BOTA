from __future__ import annotations
import numpy as np
import pytest
from src.paper_baselines.partitioned import aggregate_probabilities,assert_interaction_disjoint,balanced_similarity_partition,deletion_plan,fit_attention,locate_subset_rows,sisa_partition,validate_partition_config

def rows(n=40):return [{"input":f"user history {i%7} item {i}","output":"Yes." if i%2 else "No."} for i in range(n)]

def test_sisa_is_sharded_isolated_sliced_and_complete()->None:
 p=sisa_partition(rows(),4,3,42);flat=[i for shard in p.values() for part in shard.values() for i in part];assert sorted(flat)==list(range(40));assert len(flat)==len(set(flat))

def test_sisa_deletion_retrains_from_earliest_affected_slice()->None:
 p=sisa_partition(rows(),4,3,42);chosen=[]
 for shard,parts in p.items():
  for slice_id,indices in parts.items():
   if indices:chosen.append(indices[0])
 plan=deletion_plan(p,chosen)
 for index in chosen:
  location=next((s,t) for s,parts in p.items() for t,indices in parts.items() if index in indices);assert plan[location[0]]<=location[1]

def test_subset_matching_rejects_extra_duplicates()->None:
 full=rows(3);assert locate_subset_rows(full,[full[1]])==[1]
 with pytest.raises(ValueError):locate_subset_rows(full,[full[1],full[1]])

def test_identical_interaction_cannot_cross_training_and_development()->None:
 with pytest.raises(ValueError,match="identical interactions"):assert_interaction_disjoint(rows(3),[rows(3)[1]])

def test_equal_content_for_distinct_manifest_bound_entities_is_not_identity_leakage()->None:
 row=rows(1)[0];evidence=assert_interaction_disjoint([row],[row],[10],[20]);assert evidence=={"semantic_overlap_keys":1,"cross_entity_equal_rows":1,"identity_overlap":False}
 with pytest.raises(ValueError,match="same entity"):assert_interaction_disjoint([row],[row],[10],[10])
 with pytest.raises(ValueError,match="provided together"):assert_interaction_disjoint([row],[row],[10],None)

def test_receraser_partition_is_balanced_and_deterministic()->None:
 features=np.eye(20,32,dtype=np.float32);left=balanced_similarity_partition(features,4,42);right=balanced_similarity_partition(features,4,42);assert left==right;counts=[left.count(i) for i in range(4)];assert max(counts)-min(counts)<=1

def test_attention_aggregation_is_finite_and_normalized()->None:
 p=np.array([[.1,.9],[.2,.8],[.8,.2],[.9,.1]]);y=np.array([1,1,0,0]);w=fit_attention(p,y,50,.1);assert np.isclose(w.sum(),1);assert np.isfinite(aggregate_probabilities(p,w)).all()

def test_invalid_aggregation_weights_rejected()->None:
 with pytest.raises(ValueError):aggregate_probabilities(np.ones((2,2)),np.array([1.,1.]))

def test_sisa_cannot_claim_unavailable_user_partition()->None:
 value={"protocol":{"sharded":True,"isolated":True,"sliced":True,"aggregation":"mean_yes_no_probability","partition_unit":"user","shards":4}}
 with pytest.raises(ValueError,match="interaction-level"):validate_partition_config(value,"sisa")

def test_receraser_must_be_named_adapter()->None:
 value={"protocol":{"method_name":"RecEraser","partition":"balanced_text_similarity","aggregation":"learned_development_probability_attention","shards":4}}
 with pytest.raises(ValueError):validate_partition_config(value,"receraser")
