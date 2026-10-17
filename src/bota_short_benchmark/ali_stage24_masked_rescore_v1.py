"""Stage24: evaluation-only repair of Stage23. No optimizer or transport calls."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from pathlib import Path

import torch
from src.paper_if_a2.common import atomic_json, atomic_text, canonical_hash, sha256_file
from . import ali_stage23_pair_decomposition_v1 as s23
from . import ali_stage24_masked_scoring as scoring

SCHEMA = "bota-ali-stage24-masked-rescore-v1"
CONFIG = "configs/ali_stage24_masked_rescore_v1.json"
ROOT = "outputs/ali_stage24_masked_rescore_v1"
read = s23.read


def verify(path, contract=None):
    manifest = read(path/"manifest.json")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if manifest["schema"] != SCHEMA or actual != set(manifest["files"]) or (path/"COMPLETED").read_text(encoding="utf-8") != SCHEMA+"\n":
        raise ValueError(f"invalid Stage24 bundle: {path}")
    for rel, digest in manifest["files"].items():
        if sha256_file(path/rel) != digest:
            raise ValueError(f"changed artifact: {path/rel}")
    result = read(path/"result.json")
    if contract is not None and result["contract"] != contract:
        raise ValueError(f"contract changed: {path}")
    return result


def task(root, key, contract, fn):
    dest = root/key
    if dest.exists():
        return verify(dest, contract)
    stage = root/".work"/uuid.uuid4().hex[:10]
    stage.mkdir(parents=True)
    started = time.perf_counter()
    print(f"[Stage24] {key}; work={stage}", flush=True)
    try:
        result = {**fn(stage), "contract": contract, "test_accessed": False,
                  "optimizer_step_calls": 0, "wall_seconds": time.perf_counter()-started}
        atomic_json(stage/"result.json", result)
        atomic_text(stage/"COMPLETED", SCHEMA+"\n")
        atomic_json(stage/"manifest.json", {"schema": SCHEMA, "files": {
            p.relative_to(stage).as_posix(): sha256_file(p) for p in sorted(stage.rglob("*")) if p.is_file()}})
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.rename(stage, dest)
        return result
    except BaseException as error:
        atomic_json(stage/"FAILED.json", {"key": key, "error": repr(error),
                    "interrupted": isinstance(error, KeyboardInterrupt), "contract": contract})
        print(f"[Stage24] incomplete unit retained at {stage}", flush=True)
        raise


def load_config(repo):
    config = read(repo/CONFIG)
    required = {"schema": SCHEMA, "output_root": ROOT, "parent_root": s23.ROOT,
                "split": "Development", "inference_batch_size": 4, "chunk_size": 256,
                "selection_performed": False, "test_accessed": False, "final_test_forbidden": True,
                "optimizer_step_calls": 0, "mask": "encoder_input_ids_ne_tokenizer_pad_token_id",
                "decoder_protocol": "unchanged_teacher_forced_targets"}
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("Stage24 scope changed")
    if config["audit"] != {"samples": 32, "local_samples": 16, "batch_sizes": [1, 4, 8],
                            "orders": ["length_sorted", "short_long_interleaved"],
                            "probability_atol": 1e-5, "probability_rtol": 1e-5,
                            "loss_atol": 1e-4, "loss_rtol": 1e-5}:
        raise ValueError("frozen audit rules changed")
    return config


def freeze_plan(repo, workspace, config):
    parent = workspace/config["parent_root"]
    for key, rel in (("parent_protocol_sha256", "protocol/result.json"),
                     ("parent_summary_sha256", "summary/result.json")):
        if sha256_file(parent/rel) != config[key]:
            raise ValueError(f"Stage23 parent changed: {rel}")
    protocol = s23.verify(parent/"protocol")["protocol"]
    s23.check_inputs(protocol)
    parent_contract = canonical_hash(protocol)
    summary = s23.verify(parent/"summary", parent_contract)
    if summary["status"] != "COMPLETED" or len(summary["parameters"]) != 36 or len(summary["behavior"]) != 180:
        raise ValueError("Stage23 is not fully completed")
    files = {**protocol["input_sha256"], **protocol["source_sha256"]}
    # Verify every published parent unit and bind its manifest as well as contents.
    for marker in sorted(parent.rglob("COMPLETED")):
        if ".work" in marker.parts:
            continue
        bundle = marker.parent
        s23.verify(bundle, parent_contract)
        for p in bundle.rglob("*"):
            if p.is_file():
                files[str(p.resolve())] = sha256_file(p)
    for path in (repo/CONFIG, Path(__file__).resolve(), Path(scoring.__file__).resolve(),
                 repo/"scripts/bota_if/run_ali_stage24_masked_rescore_v1.ps1"):
        files[str(path.resolve())] = sha256_file(path)
    return {"schema": SCHEMA, "config": config, "parent_protocol": protocol,
            "hashes": files, "runtime": {**s23.c17.runtime(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "test_accessed": False, "selection_performed": False}


def check_plan(plan, config):
    if plan["config"] != config:
        raise ValueError("frozen Stage24 configuration changed")
    runtime = {**s23.c17.runtime(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if plan["runtime"] != runtime:
        raise ValueError("Stage24 runtime changed")
    for path, digest in plan["hashes"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"frozen source/input changed: {path}")


def load_endpoints(parent, ctx, parent_contract):
    """Read and compose CPU tensors only; never invoke replay/build_bank."""
    prefix = f"{ctx['dataset']}_s{ctx['seed']}"
    control = s23.verify(parent/prefix/"control", parent_contract)
    if not control["original_reproduction"]:
        raise ValueError("missing Original reproduction control")
    s23.verify(parent/prefix/"bank", parent_contract)
    bank = s23.state(parent/prefix/"bank/bank.pt")
    original = s23.state(ctx["original"])
    singles, pairs = {}, {}
    def load(path):
        value = s23.state(path)
        if not s23.same_map(value["A"], original["A"]):
            raise ValueError("fixed-A mismatch")
        return value["B"]
    for request in ctx["pairs"]:
        for user in request["users"]:
            bundle = parent/prefix/f"exact/u{user}"
            s23.verify(bundle, parent_contract)
            singles[user] = load(bundle/"adapter/adapter_model.pt")
        rid = request["id"]
        if rid in ctx["historical_joint"]:
            path = Path(ctx["historical_joint"][rid])
        else:
            bundle = parent/prefix/"exact"/rid
            s23.verify(bundle, parent_contract)
            path = bundle/"adapter/adapter_model.pt"
        pairs[rid] = load(path)
    endpoints = s23.make_endpoints(original, bank["names"], bank, singles, pairs, ctx["pairs"])
    for name, values in endpoints.items():
        old = s23.verify(parent/prefix/"predictions"/name, parent_contract)
        if old.get("endpoint_sha256") != s23.s18.s8.tensor_tree_hash(values):
            raise ValueError(f"endpoint reconstruction changed: {prefix}/{name}")
    return endpoints, bank


def rescore_context(root, parent, workspace, ctx, contract, parent_contract, config):
    prefix = f"{ctx['dataset']}_s{ctx['seed']}"
    endpoints, bank = load_endpoints(parent, ctx, parent_contract)
    remaining = []
    for name in endpoints:
        dest = root/prefix/"predictions"/name
        if dest.exists():
            verify(dest, contract)
            if verify(root/prefix/"audits"/name, contract)["status"] != "PASS":
                raise RuntimeError("cached predictions lack passing invariance gate")
        else:
            remaining.append(name)
    if remaining:
        # Model construction only; fresh() performs no training/transport.
        model, names, params, bases, tok, data, users, device, original = s23.fresh(workspace, ctx)
        dev = s23.s18.s8.JsonPromptDataset(workspace/ctx["benchmark"]["source"]["development_json"], tok)
        try:
            model.eval()
            if list(names) != bank["names"]:
                raise ValueError("parameter ordering changed")
            for name in remaining:
                values = endpoints[name]
                if any(not bool(torch.isfinite(v).all()) for v in values.values()):
                    raise ValueError(f"nonfinite saved endpoint: {name}")
                with torch.no_grad():
                    for key, param in zip(names, params):
                        param.copy_(values[key].to(param))
                result = task(root, prefix+"/audits/"+name, contract,
                    lambda stage: scoring.invariance_audit(model, dev, ctx, device, tok.pad_token_id, config["audit"]))
                if result["status"] != "PASS":
                    raise RuntimeError(f"batch invariance failed: {prefix}/{name}; diagnostics retained; do not widen tolerances after inspecting outcomes")
                def work(stage):
                    pred = {"probability": [], "gold_label": [], "answer_loss": [],
                            "sample_order_sha256": canonical_hash(ctx["indices"])}
                    for start in range(0, len(ctx["indices"]), config["chunk_size"]):
                        piece = scoring.predict(model, dev, ctx["indices"][start:start+config["chunk_size"]],
                            device, config["inference_batch_size"], tok.pad_token_id)
                        for field in ("probability", "gold_label", "answer_loss"):
                            pred[field].extend(piece[field])
                        print(f"[Stage24] {prefix}/{name} samples={len(pred['probability'])}/{len(ctx['indices'])}", flush=True)
                    # Validate audit probes against their *full-cohort* neighbors too.
                    positions = {v: i for i, v in enumerate(ctx["indices"])}
                    ids = result["sample_ids"]
                    probe = {field: [pred[field][positions[i]] for i in ids]
                             for field in ("probability", "gold_label", "answer_loss")}
                    probe["sample_order_sha256"] = canonical_hash(ids)
                    full_gate = scoring.difference(result["reference_prediction"], probe, ids, ids, config["audit"])
                    if not full_gate["passed"]:
                        atomic_json(stage/"full_cohort_gate.json", full_gate)
                        raise RuntimeError("full-cohort neighbor invariance failed")
                    old = s23.verify(parent/prefix/"predictions"/name, parent_contract)["prediction"]
                    if old is None:
                        legacy = {"status": "OLD_PREDICTION_UNAVAILABLE"}
                    else:
                        legacy = scoring.difference(old, pred, ctx["indices"], ctx["indices"], config["audit"])
                    return {"status": "OK", "prediction": pred,
                            "endpoint_sha256": s23.s18.s8.tensor_tree_hash(values),
                            "full_cohort_gate": full_gate, "legacy_to_masked": legacy}
                task(root, prefix+"/predictions/"+name, contract, work)
        finally:
            del model, params, bases, tok, data, dev
            s23.s18.release()
    return behavior(root, prefix, ctx, contract, endpoints, bank)


def behavior(root, prefix, ctx, contract, endpoints, bank):
    def work(stage):
        preds = {key: verify(root/prefix/"predictions"/key, contract)["prediction"] for key in endpoints}
        original = preds["Original"]
        lookup = {i: j for j, i in enumerate(ctx["indices"])}
        rows = []
        def score(key, exactkey, groups, unit):
            pred, exact = preds.get(key), preds[exactkey]
            if pred is None:
                return {**unit, "method": key, "status": "UNAVAILABLE_OR_NONFINITE"}
            for other in (pred, exact):
                if other["gold_label"] != original["gold_label"] or other["sample_order_sha256"] != original["sample_order_sha256"]:
                    raise ValueError("labels/order mismatch")
            groups = {**groups, "diagnostic_control": ctx["control_indices"]}
            metrics = {label: s23.math23.probability_metrics(pred["probability"], original["probability"],
                       exact["probability"], [lookup[i] for i in ix]) for label, ix in groups.items()}
            return {**unit, "method": key, "status": "OK", "metrics": metrics}
        for request in ctx["pairs"]:
            rid = request["id"]
            groups = {"all_local": request["local_indices"], "matched_5_per_user": request["matched_indices"]}
            for member in request["members"]:
                groups[f"user_{member['user']}"] = member["indices"]
            for key in ("Original", f"OracleSum_{rid}", f"B0_{rid}", f"B{ctx['lambda']}_{rid}"):
                rows.append(score(key, f"Exact_{rid}", groups, {"request": rid, "kind": "pair"}))
            for member in request["members"]:
                user = member["user"]
                groups = {"all_local": member["indices"], "matched_5_per_user": member["matched_indices"]}
                for key in ("Original", f"B0_u{user}", f"B{ctx['lambda']}_u{user}"):
                    rows.append(score(key, f"Exact_u{user}", groups, {"request": rid, "kind": "single", "user": user}))
        return {"status": "COMPLETED", "rows": rows, "failures": bank["failures"],
                "dataset": ctx["dataset"], "seed": ctx["seed"], "control_is_not_full_utility_evaluation": True}
    return task(root, prefix+"/behavior", contract, work)


def flat_rows(rows):
    for row in rows:
        base = {k: v for k, v in row.items() if k != "metrics"}
        for scope, metrics in row.get("metrics", {"unavailable": {}}).items():
            yield {**base, "scope": scope, **metrics}


def write_csv(path, rows):
    fields = sorted({k for row in rows for k, v in row.items() if not isinstance(v, (dict, list))})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, parent, plan):
    contract = canonical_hash(plan)
    parent_contract = canonical_hash(plan["parent_protocol"])
    rows, oldrows, audits, changes, missing = [], [], [], [], []
    for prefix, ctx in plan["parent_protocol"]["contexts"].items():
        path = root/prefix/"behavior"
        if not path.exists():
            missing.append(prefix)
            continue
        rows.extend({**r, "dataset": ctx["dataset"], "seed": ctx["seed"]} for r in verify(path, contract)["rows"])
        oldrows.extend({**r, "dataset": ctx["dataset"], "seed": ctx["seed"]}
                       for r in s23.verify(parent/prefix/"behavior", parent_contract)["rows"])
        expected = [p.name for p in (parent/prefix/"predictions").iterdir() if p.is_dir()]
        for name in expected:
            audit = verify(root/prefix/"audits"/name, contract)
            prediction = verify(root/prefix/"predictions"/name, contract)
            if audit["status"] != "PASS" or not prediction["full_cohort_gate"]["passed"]:
                raise ValueError("summary blocked by invariance failure")
            for mode, field in (("masked", "rows"), ("legacy_unmasked", "legacy_negative_control")):
                audits.extend({"context": prefix, "endpoint": name, "mode": mode, **r} for r in audit[field])
            changes.append({"context": prefix, "endpoint": name, **prediction["legacy_to_masked"]})
    if missing:
        return {"status": "INCOMPLETE", "missing_contexts": missing, "behavior_rows": len(rows)}
    def work(stage):
        flat, oldflat = list(flat_rows(rows)), list(flat_rows(oldrows))
        keys = ("dataset", "seed", "request", "kind", "user", "method", "scope")
        oldmap = {tuple(r.get(k) for k in keys): r for r in oldflat}
        if len(oldmap) != len(oldflat) or len(flat) != len(oldflat):
            raise ValueError("old/new metric row count mismatch")
        paired = []
        for row in flat:
            old = oldmap[tuple(row.get(k) for k in keys)]
            out = {k: row.get(k) for k in keys}
            out.update(old_status=old["status"], masked_status=row["status"])
            for metric in ("E_U", "D_U", "residual", "toward_exact_rate"):
                a, b = old.get(metric), row.get(metric)
                out.update({"legacy_"+metric: a, "masked_"+metric: b,
                            "delta_"+metric: b-a if a is not None and b is not None else None})
            paired.append(out)
        write_csv(stage/"behavior_metrics.csv", flat)
        write_csv(stage/"legacy_vs_masked_metrics.csv", paired)
        write_csv(stage/"batch_invariance.csv", audits)
        write_csv(stage/"prediction_changes.csv", changes)
        parameters = s23.verify(parent/"summary", parent_contract)["parameters"]
        write_csv(stage/"parameter_decomposition_unchanged.csv", parameters)
        lines = ["# Stage24 masked evaluation repair", "", "All models and cohorts inherited unchanged from Stage23; zero optimizer steps.",
                 "All endpoint probe and full-cohort neighbor invariance gates passed.",
                 "Legacy-to-masked deltas change the encoder masking protocol only; no retuning.", "",
                 "| Dataset | Seed | Request | Method | Legacy E_U | Masked E_U | Legacy residual | Masked residual |",
                 "|---|---:|---|---|---:|---:|---:|---:|"]
        def fmt(x):
            return "NA" if x is None else f"{x:.6g}"
        for r in paired:
            if r["kind"] == "pair" and r["scope"] == "all_local":
                lines.append(f"| {r['dataset']} | {r['seed']} | {r['request']} | {r['method']} | {fmt(r['legacy_E_U'])} | {fmt(r['masked_E_U'])} | {fmt(r['legacy_residual'])} | {fmt(r['masked_residual'])} |")
        lines.extend(["", "Parameter decomposition is inherited, not recalculated from behavioral scores.",
                      "Diagnostic control is not full utility, NDCG, or privacy validation.",
                      "No IFRU/baseline rerun; no historical main-result or lambda-selection revalidation.",
                      "A padding defect cannot explain parameter-space error. Dataset/seed causal explanations remain provisional.",
                      "Review both absolute errors and reference signal; small denominators amplify residual changes."])
        atomic_text(stage/"REPORT.md", "\n".join(lines)+"\n")
        return {"status": "COMPLETED", "behavior": rows, "endpoint_count": len(changes),
                "invariance_gates_passed": True, "selection_performed": False}
    return task(root, "summary", contract, work)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path, default=Path("C:/python/E2URec-main"))
    parser.add_argument("--mode", choices=["Preflight", "Freeze", "Full", "Analyze"], default="Preflight")
    parser.add_argument("--dataset", choices=["All", "Amazon", "ML1M"], default="All")
    parser.add_argument("--seed", choices=["All", "41", "42", "43"], default="All")
    args = parser.parse_args()
    repo, workspace = args.repo.resolve(), args.workspace.resolve()
    config = load_config(repo)
    root, parent = workspace/ROOT, workspace/config["parent_root"]
    if (root/"protocol").exists():
        plan = verify(root/"protocol")["plan"]
        check_plan(plan, config)
    else:
        plan = freeze_plan(repo, workspace, config)
    contract = canonical_hash(plan)
    if args.mode == "Preflight":
        result = {"status": "PREFLIGHT_PASS", "contexts": list(plan["parent_protocol"]["contexts"]),
                  "model_loaded": False, "test_accessed": False, "optimizer_step_calls": 0}
    else:
        with s23.s20.lock(root):
            task(root, "protocol", contract, lambda stage: {"status": "FROZEN", "plan": plan})
            if args.mode == "Freeze":
                result = {"status": "FROZEN", "contract": contract}
            else:
                if args.mode == "Full":
                    parent_contract = canonical_hash(plan["parent_protocol"])
                    for ctx in plan["parent_protocol"]["contexts"].values():
                        if args.dataset not in ("All", ctx["dataset"]) or args.seed not in ("All", str(ctx["seed"])):
                            continue
                        rescore_context(root, parent, workspace, ctx, contract, parent_contract, config)
                        check_plan(plan, config)
                result = analyze(root, parent, plan)
    if result.get("status") == "COMPLETED":
        result = {"status": "COMPLETED", "behavior_rows": len(result["behavior"]),
                  "endpoints": result["endpoint_count"], "report": str(root/"summary/REPORT.md"),
                  "optimizer_step_calls": 0, "test_accessed": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
