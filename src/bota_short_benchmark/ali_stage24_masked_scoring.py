"""Isolated, explicit encoder-padding fix. Legacy scoring is never overwritten."""
from __future__ import annotations

import numpy as np
import torch
from src.paper_if_a2.common import canonical_hash


def predict(model, dataset, indices, device, batch_size, pad_token_id, *, masked=True):
    if batch_size < 1 or len(set(indices)) != len(indices):
        raise ValueError("invalid batch size or duplicate sample IDs")
    if model.training:
        raise ValueError("scoring requires model.eval()")
    out = {"probability": [], "gold_label": [], "answer_loss": [],
           "sample_order_sha256": canonical_hash(list(indices))}
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start+batch_size]
            cpu = dataset.collate_fn([dataset[i] for i in chosen])
            inputs = cpu["input_ids"].to(device)
            targets = cpu["target_ids"].to(device)
            mask = inputs.ne(pad_token_id)
            if not bool(mask.any(1).all()) or not bool(targets.ne(-100).any(1).all()):
                raise ValueError("empty source or target")
            if not bool(((targets[:, 0] == 465) | (targets[:, 0] == 2163)).all()):
                raise ValueError("unexpected yes/no target token; do not silently relabel")
            args = dict(input_ids=inputs, labels=targets)
            if masked:
                args["attention_mask"] = mask
            logits = model(**args).logits
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite logits")
            # Keep the legacy score/loss formula: only encoder masking changes.
            probability = torch.softmax(logits, -1)
            yn = probability[:, 0, [465, 2163]]
            yes = yn[:, 1] / yn.sum(-1)
            target_mask = targets.ne(-100)
            gathered = torch.gather(probability, 2, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            loss = (-(gathered.clamp_min(1e-30).log()) * target_mask).sum(-1) / target_mask.sum(-1)
            if not bool(torch.isfinite(yes).all() and torch.isfinite(loss).all()):
                raise ValueError("nonfinite prediction")
            out["probability"].extend(yes.cpu().tolist())
            out["gold_label"].extend((targets[:, 0] == 2163).int().cpu().tolist())
            out["answer_loss"].extend(loss.cpu().tolist())
    return out


def difference(reference, candidate, indices, candidate_indices, config):
    if len(set(indices)) != len(indices) or set(indices) != set(candidate_indices) or len(indices) != len(candidate_indices):
        raise ValueError("audit sample IDs differ")
    for pred, order in ((reference, indices), (candidate, candidate_indices)):
        if pred["sample_order_sha256"] != canonical_hash(list(order)):
            raise ValueError("prediction order hash mismatch")
        if any(len(pred[k]) != len(order) for k in ("probability", "gold_label", "answer_loss")):
            raise ValueError("prediction length mismatch")
    positions = {v: i for i, v in enumerate(candidate_indices)}
    reorder = [positions[v] for v in indices]
    if reference["gold_label"] != [candidate["gold_label"][i] for i in reorder]:
        raise ValueError("prediction labels differ")
    result = {"passed": True}
    for field, prefix in (("probability", "probability"), ("answer_loss", "loss")):
        a = np.asarray(reference[field], dtype=float)
        b = np.asarray(candidate[field], dtype=float)[reorder]
        if not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("empty/nonfinite audit predictions")
        error = np.abs(a-b)
        passed = bool(np.all(error <= config[prefix+"_atol"] + config[prefix+"_rtol"]*np.abs(a)))
        result[prefix+"_max_abs_delta"] = float(error.max())
        result[prefix+"_mean_abs_delta"] = float(error.mean())
        result["passed"] &= passed
    return result


def audit_orders(dataset, ctx, config):
    """Choose probes from input lengths and membership only, never model scores."""
    lengths = {i: len(dataset[i]["input_ids"]) for i in ctx["indices"]}
    local = sorted({i for r in ctx["pairs"] for i in r["local_indices"]}, key=lambda i: (lengths[i], i))
    ordered = sorted(lengths, key=lambda i: (lengths[i], i))
    def spread(values, count):
        return [values[i] for i in np.linspace(0, len(values)-1, min(count, len(values)), dtype=int)]
    selected = set(spread(local, config["local_samples"]))
    selected.update(spread(ordered, config["samples"]-len(selected)))
    for i in ordered:
        if len(selected) >= min(config["samples"], len(ordered)):
            break
        selected.add(i)
    selected = sorted(selected, key=lambda i: (lengths[i], i))
    interleaved = []
    left, right = 0, len(selected)-1
    while left <= right:
        interleaved.append(selected[left])
        if left != right:
            interleaved.append(selected[right])
        left += 1
        right -= 1
    return {"length_sorted": selected, "short_long_interleaved": interleaved}, lengths


def invariance_audit(model, dataset, ctx, device, pad, config):
    orders, lengths = audit_orders(dataset, ctx, config)
    ids = orders["length_sorted"]
    reference = predict(model, dataset, ids, device, 1, pad)
    rows = []
    for order_name in config["orders"]:
        for batch in config["batch_sizes"]:
            if order_name == "length_sorted" and batch == 1:
                candidate = reference
            else:
                candidate = predict(model, dataset, orders[order_name], device, batch, pad)
            rows.append({"order": order_name, "batch_size": batch,
                         **difference(reference, candidate, ids, orders[order_name], config)})
    # Negative control: deliberately omit mask on the same probes only.
    legacy = []
    legacy_ref = predict(model, dataset, ids, device, 1, pad, masked=False)
    for order_name in config["orders"]:
        candidate = predict(model, dataset, orders[order_name], device, 4, pad, masked=False)
        legacy.append({"order": order_name, "batch_size": 4,
                       **difference(legacy_ref, candidate, ids, orders[order_name], config)})
    return {"status": "PASS" if all(r["passed"] for r in rows) else "FAIL",
            "rows": rows, "legacy_negative_control": legacy,
            "sample_ids": ids, "input_lengths": [lengths[i] for i in ids],
            "reference_prediction": reference,
            "legacy_failure_is_not_required": True}
