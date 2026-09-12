from src.bota_short_benchmark.ali_stage5_sequential_v1 import PRIMARY_ORDERS, SIZES, build_registry, synthetic


def _registry():
    people=[]
    for i in range(8):
        people.append({"development_samples":20,"global_train_frequency":i+1,"role":"low" if i<4 else "middle","user_hash":f"{i:064x}","window_exposure":1})
    scenario=lambda name:{"id":name,"forget_train_indices":list(range(8)),"public_members":[{**row,"role":"low" if name=="L8" else row["role"]} for row in people]}
    return {"scenarios":[scenario("L8"),scenario("L4M4")]}


def test_registry_is_complete_and_k8_is_order_invariant():
    public,cells,users=build_registry(_registry(),list(range(8)))
    assert len(cells)==20 and len(users)==8 and public["cell_count"]==20
    assert len({row["union_hash"] for row in cells if row["scenario"]=="L4M4" and row["request_size"]==8})==1
    assert all("users" not in row for row in public["cells"])


def test_synthetic_contract():
    result=synthetic()
    assert result["total_cells"]==60
    assert result["primary_orders"]==list(PRIMARY_ORDERS)
    assert result["sizes"]==list(SIZES)
    assert result["example_pass"] is True
    assert result["chain_from_published_adapter"] is False
    assert result["test_accessed"] is False
