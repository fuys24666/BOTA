"""Small CPU-testable diagnostics; no model loading, tuning, or endpoint updates."""
from __future__ import annotations

import math
import time
import torch


def norm(x):
    return float(torch.linalg.vector_norm(x))


def comparison(value, reference):
    left, right = norm(value), norm(reference)
    return {
        "absolute_l2": norm(value-reference),
        "reference_norm": right,
        "relative_l2": norm(value-reference)/right if right else None,
        "cosine": float(torch.dot(value, reference))/(left*right) if left and right else None,
        "exact_equal": bool(torch.equal(value, reference)),
    }


def operator_probes(operator, gradient, seed, scales, progress=None):
    """Raw G probes. No damping hides defects in G; no PSD certification."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random = torch.randn(gradient.numel(), dtype=torch.float64, generator=generator)
    v = gradient / norm(gradient) if norm(gradient) else random / norm(random)
    w = torch.randn(gradient.numel(), dtype=torch.float64, generator=generator)
    w = w / norm(w)
    calls = 0
    def call(x):
        nonlocal calls
        y = operator(x).detach().double().cpu()
        if y.shape != x.shape or not torch.isfinite(y).all():
            raise FloatingPointError("invalid operator probe")
        calls += 1
        if progress: progress({"probe_operator_calls": calls})
        return y
    gv, gv2, gw = call(v), call(v), call(w)
    repeat = comparison(gv2, gv)
    add = comparison(call(v+w), gv+gw)
    vw, wv = float(torch.dot(v, gw)), float(torch.dot(w, gv))
    denom = norm(v)*norm(gw)+norm(w)*norm(gv)
    rows = [{"scale": scale, **comparison(call(scale*v)/scale, gv)} for scale in scales]
    # The unnormalised request gradient matters: CG directions can be tiny.
    raw = comparison(call(gradient), gv*norm(gradient))
    return {"repeatability": repeat, "additivity": add,
            "symmetry": {"v_Gw": vw, "w_Gv": wv, "absolute_gap": abs(vw-wv),
                         "normalized_gap": abs(vw-wv)/denom if denom else None},
            "rayleigh_probes": [float(torch.dot(v, gv)), float(torch.dot(w, gw))],
            "homogeneity": rows, "raw_gradient_homogeneity": raw,
            "operator_calls": calls, "probe_seed": seed,
            "interpretation": "Limited directional probes, not a proof of linearity, symmetry or PSD."}


def diagnostic_cg(operator, gradient, damping, config, variant="legacy", restart_interval=5,
                  progress=None):
    """Legacy recurrence/guards, plus separately labelled monitoring/restarts.

    legacy: same matvec sequence and recurrence as historical CG; retain the
    failure trace that historical CG discarded. monitor: extra true residuals,
    NO recurrence modification. restart5: replace r and reset p every five
    steps or whenever recursive convergence is false. No tolerance relaxation.
    The total iteration budget includes restarts. No solution is applied.
    """
    if variant not in ("legacy", "monitor", "restart5"):
        raise ValueError("unknown CG diagnostic variant")
    g = gradient.detach().double().cpu()
    if (not torch.isfinite(g).all() or damping < 0 or not math.isfinite(damping)
            or config["max_iterations"] < 1 or restart_interval < 1):
        raise ValueError("invalid diagnostic CG input")
    x = torch.zeros_like(g); r = g.clone(); p = r.clone()
    initial = norm(r); target = max(config["absolute_tolerance"], config["relative_tolerance"]*initial)
    calls = 0; history = []; restarts = []; converged = initial <= target; error = None
    started = time.perf_counter()

    def A(v):
        nonlocal calls
        y = operator(v).detach().double().cpu()+damping*v
        calls += 1
        if y.shape != v.shape or not torch.isfinite(y).all():
            raise FloatingPointError("nonfinite/invalid HVP")
        return y

    def relative(n):
        return n/initial if initial else (0.0 if n == 0 else None)

    for iteration in range(1, config["max_iterations"]+1):
        if converged: break
        ap = A(p); pap = float(torch.dot(p, ap))
        negative_tol = config["pap_tolerance"]+1e-15*norm(p)*norm(ap)
        row = {"iteration": iteration, "pAp": pap, "negative_pAp_tolerance": negative_tol,
               "direction_norm": norm(p), "operator_product_norm": norm(ap)}
        if pap <= 0:
            error = "significantly_negative_pAp" if pap < -negative_tol else "numerically_indeterminate_pAp"
            row["error"] = error; history.append(row); break
        rr = torch.dot(r, r); alpha = rr/pap
        x = x+alpha*p; r_new = r-alpha*ap; rn = norm(r_new)
        row.update(recursive_residual_norm=rn, recursive_relative_residual=relative(rn))
        if not math.isfinite(rn) or rn > config["residual_explosion_factor"]*max(initial, config["absolute_tolerance"]):
            error = "residual_explosion"; row["error"] = error; history.append(row); break
        recursive_pass = rn <= target
        actual = None
        if variant != "legacy":
            actual = g-A(x); actual_norm = norm(actual)
            row.update(true_residual_norm=actual_norm, true_relative_residual=relative(actual_norm),
                       residual_gap_norm=norm(actual-r_new), true_over_threshold=actual_norm/target,
                       true_tolerance_satisfied=actual_norm <= target*(1+1e-8))
        restart = (variant == "restart5" and not row["true_tolerance_satisfied"]
                   and (iteration % restart_interval == 0 or recursive_pass))
        row.update(restarted=restart, operator_calls=calls)
        history.append(row)
        if progress: progress({"variant": variant, "last": row, "iterations": len(history), "target": target})
        if variant == "restart5" and row["true_tolerance_satisfied"]:
            r = actual; converged = True; break
        if restart:
            r = actual; p = r.clone(); restarts.append(iteration)
            continue
        if recursive_pass:
            r = r_new; converged = True; break
        beta = torch.dot(r_new, r_new)/rr
        p = r_new+beta*p; r = r_new
    # This is the exact final check responsible for Stage21's failure flag.
    actual = g-A(x); actual_norm = norm(actual)
    true_pass = actual_norm <= target*(1+1e-8)
    if error is None and converged and not true_pass: error = "false_residual_convergence"
    if error is None and (not math.isfinite(actual_norm) or actual_norm > config["residual_explosion_factor"]*max(initial, config["absolute_tolerance"])):
        error = "residual_explosion"
    gx = float(torch.dot(g, x))
    if error is None and (not math.isfinite(gx) or gx < -1e-12): error = "negative_influence_sign"
    status = error or ("converged" if converged else "max_iterations_reached")
    return {"solution": x, "status": status, "variant": variant, "error": error,
            "converged_by_stopping_rule": converged, "true_tolerance_satisfied": bool(true_pass),
            "historical_acceptance": error is None if variant == "legacy" else None,
            "historical_truncated_acceptance": variant == "legacy" and error is None and not converged,
            "solution_applied": False, "tolerance_relaxed": False,
            "initial_residual": initial, "target": target, "final_residual": actual_norm,
            "final_relative_residual": relative(actual_norm), "true_over_threshold": actual_norm/target,
            "final_recursive_residual": norm(r), "g_dot_solution": gx,
            "iterations": len(history), "operator_calls": calls, "restarts": restarts,
            "history": history, "wall_time_seconds": time.perf_counter()-started}
