"""Exact vector accounting and metadata-only diagnostic cohorts."""
from __future__ import annotations
import hashlib
from collections import Counter
import numpy as np
import torch


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def ml_pairs(registry, train_users, development_users):
    row = next(r for r in registry["scenarios"] if r["id"] == "L4M4")
    pools = {"low": [], "middle": []}
    for member in row["public_members"]:
        role = member["role"]
        found = [int(u) for u in row["users"] if digest(f"bota-short:{registry['seed']}:{role}:{u}") == member["user_hash"]]
        if len(found) != 1: raise ValueError("cannot resolve public metadata to actual user")
        user = found[0]
        if Counter(map(int, train_users))[user] != member["global_train_frequency"]:
            raise ValueError("user frequency metadata mismatch")
        if Counter(map(int, development_users))[user] != member["development_samples"]:
            raise ValueError("user Development metadata mismatch")
        pools[role].append((member["user_hash"], user))
    if any(len(v) != 4 for v in pools.values()): raise ValueError("expected four low and four middle users")
    low, mid = sorted(pools["low"]), sorted(pools["middle"])
    return [{"id": f"M{i+1}", "users": [a[1], b[1]], "roles": ["low", "middle"]}
            for i, (a, b) in enumerate(zip(low[:3], mid[:3]))]


def enrich_pairs(pairs, train_users, dev_users, order, evaluation):
    tr = Counter(map(int, train_users)); dv = Counter(map(int, dev_users))
    win = Counter(int(train_users[i]) for i in order)
    requested = [int(u) for r in pairs for u in r["users"]]
    if len(requested) != 6 or len(set(requested)) != 6: raise ValueError("three disjoint pairs required")
    salt = evaluation["selection_salt"]; rows = []
    for r in pairs:
        per = []
        for u, role in zip(r["users"], r["roles"]):
            indices = [i for i, v in enumerate(dev_users) if int(v) == u]
            ordered = sorted(indices, key=lambda i: digest(f"{salt}:{u}:{i}"))
            if win[u] == 0 or len(indices) < evaluation["matched_samples_per_user"]:
                raise ValueError("insufficient fixed user support; no replacement")
            per.append({"user": u, "role": role, "train_frequency": tr[u], "window_exposure": win[u],
                        "development_samples": dv[u], "indices": indices,
                        "matched_indices": sorted(ordered[:evaluation["matched_samples_per_user"]])})
        slots = [j for j, idx in enumerate(order) if int(train_users[idx]) in r["users"]]
        rows.append({**r, "members": per, "forget_window_positions": slots,
                     "forget_train_indices": [order[j] for j in slots], "deleted_interactions": len(slots),
                     "window_ratio": len(slots)/len(order),
                     "local_indices": sorted(i for m in per for i in m["indices"]),
                     "matched_indices": sorted(i for m in per for i in m["matched_indices"])})
    control = [i for i, u in enumerate(dev_users) if int(u) not in requested]
    control.sort(key=lambda i: digest(f"{salt}:control:{i}"))
    count = evaluation["nonrequested_control_samples"]
    if len(control) < count: raise ValueError("insufficient diagnostic control samples")
    control = sorted(control[:count])
    indices = sorted(set(control) | {i for r in rows for i in r["local_indices"]})
    return rows, indices, control


def cosine(a, b):
    den = float(torch.linalg.vector_norm(a))*float(torch.linalg.vector_norm(b))
    return max(-1., min(1., float(torch.dot(a, b))/den)) if den else None


def decompose(original, exact_u, exact_v, exact_pair, approximate_u, approximate_v, approximate_pair, floor=1e-12):
    values = [v.detach().double().cpu() for v in
              (original, exact_u, exact_v, exact_pair, approximate_u, approximate_v, approximate_pair)]
    if len({v.shape for v in values}) != 1 or any(v.ndim != 1 or not torch.isfinite(v).all() for v in values):
        raise ValueError("incompatible/nonfinite coordinate")
    o, u, v, uv, au, av, ap = values
    du, dv, dp = u-o, v-o, uv-o
    interaction = dp-du-dv
    eu, ev = au-u, av-v
    rounding = ap-au-av+o
    total = ap-uv
    closure = total-(eu+ev-interaction+rounding)
    magnitude = lambda x: float(torch.linalg.vector_norm(x))
    dn = magnitude(dp)
    def ratio(n): return n/dn if dn > floor else None
    pieces = {"single_u_error": eu, "single_v_error": ev, "negative_interaction": -interaction,
              "composition_rounding": rounding}
    cross = {}
    keys = list(pieces)
    for i, key in enumerate(keys):
        for other in keys[i+1:]: cross[f"{key}__{other}"] = 2*float(torch.dot(pieces[key], pieces[other]))
    norm_terms = {key: float(torch.dot(vec, vec)) for key, vec in pieces.items()}
    if magnitude(closure) > 1e-10*max(1., magnitude(total)):
        raise RuntimeError("decomposition identity failed")
    return {"exact_u_norm": magnitude(du), "exact_v_norm": magnitude(dv), "exact_pair_norm": dn,
            "interaction_norm": magnitude(interaction), "interaction_over_pair_norm": ratio(magnitude(interaction)),
            "interaction_over_sum_individual_norms": magnitude(interaction)/(magnitude(du)+magnitude(dv)) if magnitude(du)+magnitude(dv) > floor else None,
            "single_u_error_norm": magnitude(eu), "single_v_error_norm": magnitude(ev),
            "single_u_relative_error": magnitude(eu)/magnitude(du) if magnitude(du)>floor else None,
            "single_v_relative_error": magnitude(ev)/magnitude(dv) if magnitude(dv)>floor else None,
            "summed_single_error_norm": magnitude(eu+ev), "pair_error_norm": magnitude(total),
            "pair_relative_error": ratio(magnitude(total)), "pair_signal_below_floor": dn <= floor,
            "single_error_interaction_cosine": cosine(eu+ev, interaction),
            "exact_user_effect_cosine": cosine(du, dv), "approximate_user_effect_cosine": cosine(au-o, av-o),
            "composition_rounding_norm": magnitude(rounding), "identity_closure_norm": magnitude(closure),
            "squared_error_terms": norm_terms, "signed_cross_terms": cross,
            "squared_identity_gap": abs(float(torch.dot(total, total))-sum(norm_terms.values())-sum(cross.values())),
            "warning": "Norm ratios are not causal contribution percentages; cross terms can cancel."}


def probability_metrics(prediction, original, exact, positions):
    if not positions: raise ValueError("empty diagnostic evaluation slice")
    p, o, e = [np.asarray(x, dtype=float)[positions] for x in (prediction, original, exact)]
    if any(not np.isfinite(x).all() for x in (p, o, e)): raise ValueError("nonfinite predictions")
    eu, du = float(np.abs(p-e).mean()), float(np.abs(o-e).mean())
    return {"samples": len(positions), "E_U": eu, "D_U": du, "residual": eu/du if du>1e-12 else None,
            "toward_exact_rate": float(np.mean(np.abs(p-e)<np.abs(o-e))),
            "relative_error_is_not_privacy": True}
