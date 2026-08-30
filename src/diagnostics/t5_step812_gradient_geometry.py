from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import T5Tokenizer

from src.diagnostics.git_provenance import git_provenance, implementation_provenance, require_clean_git
from src.diagnostics.t5_full_runner import _batch, _restore_rng_payload, batch_index_hash, batch_indices
from src.diagnostics.t5_reconstructed_official import (
    JsonPromptDataset, build_current_model, compute_components, freeze_teacher,
    load_config, load_legacy_model, move_batch, sha256_file,
)
from src.diagnostics.t5_trajectory_diagnostics import capture_rng, restore_rng, rng_hashes
from src.diagnostics.t5_zero_training_audit import _data_lineage

SCHEMA = "t5-step812-gradient-geometry-audit-v1"
CACHE_SCHEMA = "t5-step812-gradient-geometry-cache-v2"
UNIT_MANIFEST_SCHEMA = "t5-step812-gradient-geometry-unit-manifest-v1"
ANALYSIS_SCHEMA = "t5-step812-gradient-geometry-analysis-v1"
OUTPUT_NAME = "t5_step812_gradient_geometry_audit_v1"
LOSSES = ("L_forget", "L_retain_KL", "L_sup")
IMPLEMENTATION_FILES = (
    "src/diagnostics/t5_step812_gradient_geometry.py",
    "src/diagnostics/git_provenance.py",
    "configs/t5_step812_gradient_geometry_audit_v1.yaml",
    "scripts/diagnostics/t5_step812_gradient_geometry_v1.ps1",
    "docs/t5_step812_gradient_geometry_audit_v1.md",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:10]}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _write_text_fsync_direct(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text); handle.flush(); os.fsync(handle.fileno())


def _safe_name(value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}: raise ValueError("RunName must be one path component")
    return value


def _resolve(root: Path, value: str | Path, *, output: bool = False) -> Path:
    path = (root / value).resolve()
    if path != root.resolve() and root.resolve() not in path.parents: raise ValueError("path escapes project root")
    if not output and any("test" in part.lower() for part in path.parts): raise ValueError(f"test path forbidden: {path}")
    return path


def load_audit_config(path: Path, root: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    fixed = {
        "schema": SCHEMA, "cache_schema": CACHE_SCHEMA, "analysis_schema": ANALYSIS_SCHEMA,
        "development_only": True, "alpha": 2.0, "batch_size": 16,
        "seed": 42, "test_access_policy": "forbidden",
    }
    if any(value.get(key) != expected for key, expected in fixed.items()): raise ValueError("gradient audit config differs from preregistration")
    if value["classification"] != {"eta_f_min": 0.10, "rho_equivalent_min": 0.31622776601683794}: raise ValueError("classification threshold changed")
    if value["projection"] != {"algorithm":"direct_float64_retain_matrix_svd","relative_singular_tolerance":1.0e-10,"normalized_residual_tolerance":1.0e-8,"rho_zero_tolerance":1.0e-8}: raise ValueError("projection protocol changed")
    if value["secondary"] != {"paired_batches": 2939, "retain_epoch": 0, "retain_positions": "sequential_complete_epoch", "forget_start_epoch": 1, "forget_positions": "cyclic_complete_epochs", "dropout_mode": "train", "rng_strategy": "standardized_per_pair_seed", "rng_seed_base": 420000}: raise ValueError("secondary catalog/RNG changed")
    value["_sha256"] = sha256_file(path); value["_path"] = str(path.resolve())
    return value


def catalog_pair(index: int, forget_count: int, retain_count: int, batch_size: int, seed: int) -> dict[str, Any]:
    forget_batches = math.ceil(forget_count / batch_size)
    retain_batches = math.ceil(retain_count / batch_size)
    if index < 0 or index >= retain_batches: raise IndexError(index)
    forget_epoch = 1 + index // forget_batches; forget_position = index % forget_batches
    forget = batch_indices(forget_count, batch_size, seed, forget_epoch, forget_position, 0)
    retain = batch_indices(retain_count, batch_size, seed, 0, index, 10_000)
    return {"catalog_index": index, "forget_epoch": forget_epoch, "forget_position": forget_position, "retain_epoch": 0, "retain_position": index, "forget_indices": forget, "retain_indices": retain, "batch_hash": batch_index_hash(forget, retain)}


def _checkpoint_metadata(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path, state_path = checkpoint / "manifest.json", checkpoint / "state.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("step") != 812 or manifest.get("published_atomically") is not True or manifest.get("state_sha256") != sha256_file(state_path): raise ValueError("step812 checkpoint publication invalid")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "t5-e2urec-full-runner-v1" or payload["state"].get("step") != 812 or payload["state"].get("next_optimizer_step") != 813: raise ValueError("step812 checkpoint state invalid")
    adapter = payload.get("adapter_state")
    if not isinstance(adapter, dict) or not adapter or any(not torch.is_tensor(value) for value in adapter.values()): raise ValueError("step812 adapter state invalid")
    return manifest, payload


def _tree_sha256(path: Path) -> str:
    files=sorted(item for item in path.rglob("*") if item.is_file())
    if not files: raise ValueError("tokenizer definition directory is empty")
    return canonical_hash({item.relative_to(path).as_posix():sha256_file(item) for item in files})


def preflight(root: Path, config_path: Path, *, git_function=git_provenance, implementation_function=implementation_provenance) -> dict[str, Any]:
    audit = load_audit_config(config_path, root); base_path = _resolve(root, audit["base_config"]); base = load_config(base_path, root)
    checkpoint = _resolve(root, audit["step812_checkpoint"]); manifest, payload = _checkpoint_metadata(checkpoint)
    if base["training"]["alpha"] != audit["alpha"] or base["training"]["per_device_batch_size"] != audit["batch_size"] or base["training"]["seed"] != audit["seed"]: raise ValueError("base protocol differs")
    forget_path, retain_path = Path(base["paths"]["forget"]), Path(base["paths"]["retain"])
    forget_count = len(json.loads(forget_path.read_text(encoding="utf-8"))); retain_count = len(json.loads(retain_path.read_text(encoding="utf-8")))
    lineage,_,_=_data_lineage(root,base,_resolve(root,audit["protocol_root"]))
    if lineage["data"]["forget_train"]["sha256"]!=sha256_file(forget_path) or lineage["data"]["retain_train"]["sha256"]!=sha256_file(retain_path): raise ValueError("authoritative lineage data hash mismatch")
    primary = catalog_pair(0, forget_count, retain_count, audit["batch_size"], audit["seed"])
    if primary["batch_hash"] != payload["state"].get("next_batch_hash"): raise ValueError("planned step813 batch differs from step812 checkpoint anchor")
    adapter = payload["adapter_state"]
    git = git_function(root); implementation = implementation_function(root, IMPLEMENTATION_FILES)
    return {
        "schema": SCHEMA, "mode": "Preflight", "config_sha256": audit["_sha256"], "base_config_sha256": sha256_file(base_path),
        "checkpoint": {"directory": str(checkpoint), "state_path": str(checkpoint / "state.pt"), "state_sha256": manifest["state_sha256"], "manifest_sha256": sha256_file(checkpoint / "manifest.json"), "step": 812, "next_optimizer_step": 813},
        "adapter": {"tensor_count": len(adapter), "parameter_count": sum(value.numel() for value in adapter.values()), "key_sha256": canonical_hash(sorted(adapter)), "shapes_sha256": canonical_hash({key: list(value.shape) for key, value in adapter.items()})},
        "serialized_optimizer_state_ignored": True,
        "teachers": {role: {"path": base["paths"][key], "sha256": sha256_file(Path(base["paths"][key]))} for role, key in (("original", "original"), ("augmented", "augmented_teacher"))},
        "retrain_loaded": False, "formula": "forced_logits and compute_components from reconstructed-official implementation", "alpha": 2.0,
        "branch_loss_audit": {"J0": "authoritative compute_components; all three components", "J2": "same components, supervised excluded only from total objective", "J4": "same ungated components with calibrated supervised coefficient", "J5": "retain components unchanged; Forget component is separately gated and is not used by this ungated forced-target audit", "source_sha256":{name:sha256_file(root/path) for name,path in {"authority":"src/diagnostics/t5_reconstructed_official.py","J2":"src/diagnostics/t5_joint_ablation.py","J4":"src/diagnostics/t5_joint_j4.py","J5":"src/diagnostics/t5_teacher_gate.py"}.items()}},
        "data": {"forget": {"path": str(forget_path), "sha256": sha256_file(forget_path), "samples": forget_count}, "retain": {"path": str(retain_path), "sha256": sha256_file(retain_path), "samples": retain_count}},
        "lineage": {"manifest":lineage["manifest"],"forget_user_manifest":lineage["forget_user_manifest"],"partition_checks":lineage["partition_checks"],"processed_test_split_read":False,"test_accessed":False},
        "tokenizer": {"definition_path":base["paths"]["model_dir"],"definition_tree_sha256":_tree_sha256(Path(base["paths"]["model_dir"])),"loaded":False},
        "primary_exact": {key: value for key, value in primary.items() if not key.endswith("indices")},
        "secondary": {"paired_batches": math.ceil(retain_count / audit["batch_size"]), "covers_retain_once": True, "covers_forget_with_deterministic_cycles": True, "dropout_mode": "train", "rng_strategy": "standardized_per_pair_seed", "rng_seed_base": 420000, "historical_replay_claimed": False},
        "rng": {"checkpoint_rng_hash": payload["rng_hash"], "primary_exact_restores_checkpoint_rng": True},
        "resource_estimate": {"geometry_units": 1 + math.ceil(retain_count / audit["batch_size"]), "model_forwards_per_unit": 5, "autograd_grad_calls_per_unit": 3, "gradient_vectors_persisted": False},
        "git": git, "implementation": implementation, "model_loaded": False, "gradient_computed": False,
        "optimizer_constructed": False, "optimizer_steps_executed": 0, "parameters_updated": False, "test_loader_built": False, "test_accessed": False,
    }


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    a, b = left.double().reshape(-1), right.double().reshape(-1); na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    if float(na) == 0.0 or float(nb) == 0.0: return None
    return float(torch.dot(a, b) / (na * nb))


def project_forget(forget: torch.Tensor, kl: torch.Tensor, sup: torch.Tensor, *, relative_tolerance: float = 1e-10) -> dict[str, Any]:
    vectors = [value.double().reshape(-1) for value in (forget, kl, sup)]
    if any(not bool(torch.isfinite(value).all()) for value in vectors): raise FloatingPointError("non-finite gradient")
    f, k, s = vectors; norm_f = torch.linalg.vector_norm(f)
    retain = torch.stack((k, s), dim=1); gram = retain.T @ retain
    left, singular, _ = torch.linalg.svd(retain, full_matrices=False)
    largest = float(singular.max()) if singular.numel() else 0.0; tolerance = largest * relative_tolerance
    active = singular > tolerance; rank = int(active.sum())
    basis = left[:, active]
    projection = basis @ (basis.T @ f) if rank else torch.zeros_like(f)
    perpendicular = f - projection
    norm_perp = torch.linalg.vector_norm(perpendicular)
    rho = None if float(norm_f) == 0.0 else float(norm_perp / norm_f)
    eta = None if float(norm_f) == 0.0 else float(torch.dot(f, perpendicular) / torch.dot(f, f))
    residuals = [float(torch.dot(perpendicular, value)) for value in (k, s)]
    normalized = [None if float(norm_perp) == 0.0 or float(torch.linalg.vector_norm(value)) == 0.0 else abs(dot) / float(norm_perp * torch.linalg.vector_norm(value)) for dot, value in zip(residuals, (k, s))]
    positive = singular[active]
    condition = None if rank == 0 else (float(positive.max() / positive.min()) if rank > 1 else 1.0)
    return {"algorithm": "direct_float64_retain_matrix_svd", "rank": rank, "gram_matrix": [[float(item) for item in row] for row in gram], "singular_values": [float(item) for item in singular.tolist()], "condition_number": condition, "relative_singular_tolerance": relative_tolerance, "absolute_singular_tolerance": tolerance, "g_forget_perp": perpendicular, "norm_forget": float(norm_f), "norm_forget_perp": float(norm_perp), "rho": rho, "eta_F": eta, "eta_rho_squared_error": None if eta is None else abs(eta-rho*rho), "retain_dots_after": {"L_retain_KL": residuals[0], "L_sup": residuals[1]}, "normalized_residuals": {"L_retain_KL": normalized[0], "L_sup": normalized[1]}}


def first_order_predictions(f: torch.Tensor, k: torch.Tensor, s: torch.Tensor, perpendicular: torch.Tensor) -> dict[str, dict[str, float]]:
    def protected(retain:torch.Tensor)->torch.Tensor:
        denominator=torch.dot(retain.double(),retain.double())
        return f.double() if float(denominator)==0.0 else f.double()-retain.double()*(torch.dot(retain.double(),f.double())/denominator)
    directions = {"raw_forget": -f, "J0_fixed_weight": -(0.4*f+0.6*k+0.6*s), "KL_protected_only": -protected(k), "supervised_protected_only": -protected(s), "dual_retain_projection": -perpendicular}
    return {name: {loss: float(torch.dot(gradient.double(), direction.double())) for loss, gradient in zip(LOSSES, (f, k, s))} for name, direction in directions.items()}


def parameter_group(name: str) -> dict[str, str]:
    stack = "encoder" if ".encoder." in name else ("decoder" if ".decoder." in name else "other")
    layer = re.search(r"\.block\.(\d+)\.", name); layer_name = f"{stack}.layer.{layer.group(1)}" if layer else f"{stack}.layer.unknown"
    functional = "attention" if any(value in name for value in ("Attention", ".q.", ".k.", ".v.", ".o.")) else ("ffn" if any(value in name for value in ("DenseReluDense", ".wi", ".wo", "up_proj", "down_proj", "gate_proj")) else "other")
    projection = next((value for value in ("q", "k", "v", "o", "up", "down", "gate") if re.search(rf"\.{value}(?:\.|_proj)", name)), "other")
    module = re.split(r"\.lora_[AB](?:\.default)?\.weight$", name)[0]
    return {"stack": stack, "functional": functional, "projection": projection, "layer": layer_name, "module": module}


def _geometry_scalars(f: torch.Tensor, k: torch.Tensor, s: torch.Tensor, tolerance: float) -> dict[str, Any]:
    projection = project_forget(f, k, s, relative_tolerance=tolerance); perpendicular = projection.pop("g_forget_perp")
    norms = {loss: float(torch.linalg.vector_norm(value.double())) for loss, value in zip(LOSSES, (f, k, s))}
    dots = {"forget_KL": float(torch.dot(f.double(), k.double())), "forget_sup": float(torch.dot(f.double(), s.double())), "KL_sup": float(torch.dot(k.double(), s.double()))}
    return {"norms": norms, "zero_gradient": {key: value == 0.0 for key, value in norms.items()}, "cosines": {"forget_KL": safe_cosine(f,k), "forget_sup": safe_cosine(f,s), "KL_sup": safe_cosine(k,s)}, "dots": dots, "conflict": {"KL": dots["forget_KL"] < 0, "supervised": dots["forget_sup"] < 0, "any": dots["forget_KL"] < 0 or dots["forget_sup"] < 0, "both": dots["forget_KL"] < 0 and dots["forget_sup"] < 0}, "projection": projection, "first_order_predictions": first_order_predictions(f,k,s,perpendicular)}


def geometry_from_gradients(names: list[str], gradients: dict[str, list[torch.Tensor | None]], *, tolerance: float = 1e-10) -> dict[str, Any]:
    if set(gradients) != set(LOSSES) or any(len(gradients[key]) != len(names) for key in LOSSES): raise ValueError("gradient structure mismatch")
    shapes = [next((gradients[key][i].shape for key in LOSSES if gradients[key][i] is not None), None) for i in range(len(names))]
    if any(shape is None for shape in shapes): raise ValueError("parameter missing from all component gradients")
    materialized: dict[str, list[torch.Tensor]] = {}
    missing = {}
    for loss in LOSSES:
        missing[loss] = [names[i] for i,value in enumerate(gradients[loss]) if value is None]
        materialized[loss] = [torch.zeros(shapes[i]) if value is None else value.detach().cpu() for i,value in enumerate(gradients[loss])]
    flat = {loss: torch.cat([value.reshape(-1) for value in materialized[loss]]) for loss in LOSSES}
    overall = _geometry_scalars(flat[LOSSES[0]], flat[LOSSES[1]], flat[LOSSES[2]], tolerance)
    total_f_sq = max(overall["norms"]["L_forget"]**2, torch.finfo(torch.float64).tiny)
    groups: dict[str, list[int]] = {"all": list(range(len(names)))}
    parameter_metadata = []
    for index,name in enumerate(names):
        metadata = parameter_group(name); parameter_metadata.append({"name": name, **metadata})
        for kind,value in metadata.items(): groups.setdefault(f"{kind}:{value}", []).append(index)
    summaries = {}
    for group, indices in groups.items():
        local = {loss: torch.cat([materialized[loss][i].reshape(-1) for i in indices]) for loss in LOSSES}
        value = _geometry_scalars(local[LOSSES[0]], local[LOSSES[1]], local[LOSSES[2]], tolerance)
        value["forget_gradient_norm_fraction"] = value["norms"]["L_forget"]**2 / total_f_sq
        value["parameters"] = len(indices); summaries[group] = value
    return {"overall": overall, "groups": summaries, "parameter_metadata": parameter_metadata, "missing_gradients": missing, "gradient_vectors_persisted": False}


def classify_geometry(primary: dict[str, Any], aggregate: dict[str, Any], *, eta_threshold: float = .10, rho_zero: float = 1e-8, residual_tolerance: float = 1e-8) -> str:
    try:
        p = primary["overall"]["projection"]; lower = aggregate["bootstrap"]["median_eta_F"]["percentile_95_ci"][0]
        residual_ok = aggregate["projection_residual_pass"] and all(value is None or value <= residual_tolerance for value in p["normalized_residuals"].values())
        median_rho = aggregate["quantiles"]["rho"]["P50"]
        numeric = (p["eta_F"], p["rho"], lower, median_rho)
        if any(value is None or not math.isfinite(float(value)) for value in numeric) or not aggregate["all_numeric_valid"] or not residual_ok: return "G-D"
        primary_nonzero, secondary_nonzero = p["rho"] > rho_zero, median_rho > rho_zero
        if primary_nonzero != secondary_nonzero: return "G-D"
        if not primary_nonzero: return "G-C"
        if p["eta_F"] >= eta_threshold and lower >= eta_threshold: return "G-A"
        return "G-B"
    except (KeyError, TypeError, ValueError, OverflowError): return "G-D"


def aggregate_units(units: list[dict[str, Any]], *, seed: int = 42, resamples: int = 2000, residual_tolerance: float = 1e-8) -> dict[str, Any]:
    if not units: raise ValueError("secondary catalog empty")
    rho = np.asarray([unit["overall"]["projection"]["rho"] for unit in units], dtype=np.float64); eta = np.asarray([unit["overall"]["projection"]["eta_F"] for unit in units], dtype=np.float64); conflict = np.asarray([unit["overall"]["conflict"]["any"] for unit in units], dtype=np.float64)
    if not np.isfinite(rho).all() or not np.isfinite(eta).all(): raise FloatingPointError("non-finite aggregate")
    rng=np.random.default_rng(seed); boot={key:np.empty(resamples) for key in ("median_rho","mean_rho","median_eta_F","conflict_proportion")}
    for i in range(resamples):
        sample=rng.integers(0,len(units),len(units)); boot["median_rho"][i]=np.median(rho[sample]); boot["mean_rho"][i]=np.mean(rho[sample]); boot["median_eta_F"][i]=np.median(eta[sample]); boot["conflict_proportion"][i]=np.mean(conflict[sample])
    point={"median_rho":float(np.median(rho)),"mean_rho":float(np.mean(rho)),"median_eta_F":float(np.median(eta)),"conflict_proportion":float(np.mean(conflict))}
    bootstrap={key:{"point_estimate":point[key],"standard_error":float(np.std(values,ddof=1)),"percentile_95_ci":[float(x) for x in np.percentile(values,[2.5,97.5])]} for key,values in boot.items()}
    quant=lambda values:{key:float(value) for key,value in zip(("P05","P25","P50","P75","P95"),np.percentile(values,[5,25,50,75,95]))}
    residuals=[value for unit in units for value in unit["overall"]["projection"]["normalized_residuals"].values() if value is not None]
    return {"bootstrap":bootstrap,"bootstrap_unit":"paired_batch","seed":seed,"resamples":resamples,"quantiles":{"rho":quant(rho),"eta_F":quant(eta)},"conflict_proportions":{"KL":float(np.mean([u["overall"]["conflict"]["KL"] for u in units])),"supervised":float(np.mean([u["overall"]["conflict"]["supervised"] for u in units])),"any":float(np.mean(conflict)),"both":float(np.mean([u["overall"]["conflict"]["both"] for u in units]))},"projection_residual_pass":max(residuals,default=0)<=residual_tolerance,"all_numeric_valid":True}


def _tensor_state_hash(module: torch.nn.Module) -> str:
    digest=hashlib.sha256()
    for name,value in module.state_dict().items():
        tensor=value.detach().cpu().contiguous(); digest.update(name.encode()); digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _trainable_tensor_hash(module: torch.nn.Module) -> str:
    digest=hashlib.sha256()
    for name,value in module.named_parameters():
        if value.requires_grad:
            tensor=value.detach().cpu().contiguous(); digest.update(name.encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_sets(model: torch.nn.Module) -> dict[str, Any]:
    trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]; frozen=[(name,p) for name,p in model.named_parameters() if not p.requires_grad]
    return {"trainable_names":[name for name,_ in trainable],"trainable_name_sha256":canonical_hash([name for name,_ in trainable]),"trainable_tensors":len(trainable),"trainable_parameters":sum(p.numel() for _,p in trainable),"frozen_name_sha256":canonical_hash([name for name,_ in frozen]),"frozen_tensors":len(frozen),"frozen_parameters":sum(p.numel() for _,p in frozen)}


def _load_runtime(root: Path, config_path: Path, device: torch.device) -> dict[str, Any]:
    audit=load_audit_config(config_path,root); base=load_config(_resolve(root,audit["base_config"]),root); checkpoint=_resolve(root,audit["step812_checkpoint"]); _,payload=_checkpoint_metadata(checkpoint)
    tokenizer=T5Tokenizer.from_pretrained(base["paths"]["model_dir"]); forget=JsonPromptDataset(Path(base["paths"]["forget"]),tokenizer); retain=JsonPromptDataset(Path(base["paths"]["retain"]),tokenizer)
    current=build_current_model(Path(base["paths"]["original"]),base["lora"]); result=set_peft_model_state_dict(current,payload["adapter_state"])
    if getattr(result,"unexpected_keys",[]): raise ValueError("step812 adapter unexpected keys")
    reloaded=get_peft_model_state_dict(current)
    if reloaded.keys()!=payload["adapter_state"].keys() or any(not torch.equal(reloaded[key].cpu(),payload["adapter_state"][key]) for key in reloaded): raise ValueError("step812 adapter strict tensor reload mismatch")
    current=current.to(device); original=freeze_teacher(load_legacy_model(Path(base["paths"]["original"]))).to(device); augmented=freeze_teacher(load_legacy_model(Path(base["paths"]["augmented_teacher"]))).to(device)
    if original.training or augmented.training or any(p.requires_grad for teacher in (original,augmented) for p in teacher.parameters()): raise RuntimeError("teachers are not frozen/eval")
    parameters=_parameter_sets(current)
    if parameters["trainable_parameters"] != sum(value.numel() for value in payload["adapter_state"].values()): raise ValueError("runtime LoRA parameter set differs from checkpoint")
    return {"audit":audit,"base":base,"payload":payload,"tokenizer":tokenizer,"forget":forget,"retain":retain,"current":current,"original":original,"augmented":augmented,"parameter_sets":parameters}


def _standardized_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def compute_geometry_unit(runtime: dict[str, Any], pair: dict[str, Any], *, unit_id: str, rng_strategy: str, device: torch.device) -> dict[str, Any]:
    current,original,augmented=runtime["current"],runtime["original"],runtime["augmented"]; audit=runtime["audit"]
    outer_rng=capture_rng(); current_mode=current.training; student_before=_trainable_tensor_hash(current)
    try:
        if rng_strategy=="exact_step812_checkpoint_rng": _restore_rng_payload(runtime["payload"]["rng"])
        elif rng_strategy=="standardized_per_pair_seed": _standardized_seed(audit["secondary"]["rng_seed_base"]+pair["catalog_index"])
        else: raise ValueError("unknown RNG strategy")
        rng_used=rng_hashes(capture_rng()); current.train(); forget_batch=move_batch(_batch(runtime["forget"],pair["forget_indices"]),device); retain_batch=move_batch(_batch(runtime["retain"],pair["retain_indices"]),device)
        components=compute_components(current,original,augmented,forget_batch,retain_batch,audit["alpha"]); named=[(name,p) for name,p in current.named_parameters() if p.requires_grad]; names=[name for name,_ in named]; parameters=[p for _,p in named]
        gradients={}
        for index,loss in enumerate(LOSSES): gradients[loss]=list(torch.autograd.grad(components[loss],parameters,retain_graph=index<2,allow_unused=True,create_graph=False))
        if any(parameter.grad is not None for parameter in parameters): raise RuntimeError("autograd.grad populated parameter.grad")
        geometry=geometry_from_gradients(names,gradients,tolerance=audit["projection"]["relative_singular_tolerance"])
        losses={loss:float(components[loss].detach().cpu()) for loss in LOSSES}
        del components,gradients,forget_batch,retain_batch
    finally:
        current.train(current_mode); restore_rng(outer_rng)
    if _trainable_tensor_hash(current)!=student_before: raise RuntimeError("trainable LoRA tensor changed during zero-update unit")
    return {"schema":CACHE_SCHEMA,"unit_id":unit_id,"kind":"primary_exact" if unit_id=="primary_exact" else "secondary","catalog_index":pair["catalog_index"],"batch_hash":pair["batch_hash"],"forget_epoch":pair["forget_epoch"],"forget_position":pair["forget_position"],"retain_epoch":pair["retain_epoch"],"retain_position":pair["retain_position"],"rng_strategy":rng_strategy,"rng_hash":rng_used,"losses":losses,**geometry,"gradient_computed":True,"backward_method":"torch.autograd.grad","loss_backward_called":False,"optimizer_constructed":False,"optimizer_steps_executed":0,"parameters_updated":False,"student_parameters_unchanged":True,"teacher_parameters_unchanged":True,"gradient_vectors_persisted":False,"full_vocabulary_logits_persisted":False,"raw_samples_persisted":False,"test_loader_built":False,"test_accessed":False}


def _finite(value: Any) -> bool:
    if value is None or isinstance(value,(str,bool)): return True
    if isinstance(value,(int,float)): return math.isfinite(float(value))
    if isinstance(value,dict): return all(_finite(item) for item in value.values())
    if isinstance(value,list): return all(_finite(item) for item in value)
    return False


def _validate_unit(value: dict[str, Any], expected: dict[str, Any]) -> None:
    if any(value.get(key)!=item for key,item in expected.items()): raise ValueError("geometry cache contract mismatch")
    safety={"gradient_computed":True,"backward_method":"torch.autograd.grad","loss_backward_called":False,"optimizer_constructed":False,"optimizer_steps_executed":0,"parameters_updated":False,"gradient_vectors_persisted":False,"full_vocabulary_logits_persisted":False,"raw_samples_persisted":False,"test_loader_built":False,"test_accessed":False}
    if any(value.get(key)!=item for key,item in safety.items()): raise ValueError("geometry cache safety mismatch")
    if not {"overall","groups","parameter_metadata","missing_gradients"} <= set(value) or not _finite(value): raise ValueError("geometry cache structure/numeric mismatch")
    forbidden={"gradient_vector","gradient_vectors","full_vocabulary_logits","input_ids","target_ids","raw_sample","raw_samples"}
    def keys(item:Any)->set[str]:
        if isinstance(item,dict): return set(item).union(*(keys(child) for child in item.values()),set())
        if isinstance(item,list): return set().union(*(keys(child) for child in item),set())
        return set()
    if keys(value)&forbidden: raise ValueError("forbidden tensor/sample payload persisted")


def _unit_path(stage: Path, unit_id: str) -> Path: return stage/"units"/unit_id


def _unit_binding(contract_sha256: str, git: dict[str, Any], implementation: dict[str, Any]) -> dict[str, str]:
    return {
        "contract_sha256": contract_sha256,
        "git_commit": git["git_commit"],
        "implementation_canonical_sha256": implementation["canonical_sha256"],
    }


def _publish_unit(path: Path, value: dict[str, Any], binding: dict[str, str]) -> None:
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex[:10]}.stage"
    if temporary.exists(): raise FileExistsError(temporary)
    temporary.mkdir()
    try:
        unit_path = temporary / "unit.json"
        _write_text_fsync_direct(unit_path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        manifest = {
            "schema": UNIT_MANIFEST_SCHEMA,
            "unit_schema": value["schema"],
            "unit_id": value["unit_id"],
            "kind": value["kind"],
            "catalog_index": value["catalog_index"],
            "batch_hash": value["batch_hash"],
            "rng_strategy": value["rng_strategy"],
            "rng_hash": value["rng_hash"],
            **binding,
            "unit_json_sha256": sha256_file(unit_path),
            "optimizer_constructed": False,
            "optimizer_steps_executed": 0,
            "parameters_updated": False,
            "published_atomically": True,
            "test_loader_built": False,
            "test_accessed": False,
        }
        _write_text_fsync_direct(temporary / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        _write_text_fsync_direct(temporary / "COMPLETED", "UNIT_COMPLETED\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists(): shutil.rmtree(temporary)
        raise


def _load_unit(path: Path, expected: dict[str, Any], binding: dict[str, str]) -> tuple[dict[str, Any], dict[str, str]]:
    if not path.is_dir(): raise FileNotFoundError(path)
    required = {"unit.json", "manifest.json", "COMPLETED"}
    if {item.name for item in path.iterdir()} != required: raise ValueError("geometry unit publication incomplete or contains unexpected files")
    if (path / "COMPLETED").read_text(encoding="utf-8") != "UNIT_COMPLETED\n": raise ValueError("geometry unit completion marker mismatch")
    parse = lambda item: (_ for _ in ()).throw(ValueError(item))
    value=json.loads((path/"unit.json").read_text(encoding="utf-8"),parse_constant=parse); _validate_unit(value,expected)
    manifest=json.loads((path/"manifest.json").read_text(encoding="utf-8"),parse_constant=parse)
    required_manifest = {
        "schema": UNIT_MANIFEST_SCHEMA,
        "unit_schema": value["schema"],
        "unit_id": value["unit_id"],
        "kind": value["kind"],
        "catalog_index": value["catalog_index"],
        "batch_hash": value["batch_hash"],
        "rng_strategy": value["rng_strategy"],
        "rng_hash": value.get("rng_hash"),
        **binding,
        "unit_json_sha256": sha256_file(path/"unit.json"),
        "optimizer_constructed": False,
        "optimizer_steps_executed": 0,
        "parameters_updated": False,
        "published_atomically": True,
        "test_loader_built": False,
        "test_accessed": False,
    }
    if manifest != required_manifest: raise ValueError("geometry unit manifest/SHA/provenance mismatch")
    return value, {"data_sha256": required_manifest["unit_json_sha256"], "manifest_sha256": sha256_file(path/"manifest.json")}


def _resume_unit(path: Path, expected: dict[str, Any], binding: dict[str, str]) -> tuple[dict[str, Any], dict[str, str]] | None:
    if not path.exists(): return None
    return _load_unit(path, expected, binding)


def _validate_unit_inventory(source: Path, manifest: dict[str, Any], pairs: list[tuple[str, dict[str, Any], str]], binding: dict[str, str]) -> list[dict[str, Any]]:
    units, inventories = [], {}
    for unit_id,pair,rng_strategy in pairs:
        expected={"schema":CACHE_SCHEMA,"unit_id":unit_id,"kind":"primary_exact" if unit_id=="primary_exact" else "secondary","catalog_index":pair["catalog_index"],"batch_hash":pair["batch_hash"],"rng_strategy":rng_strategy}
        value,inventory=_load_unit(_unit_path(source,unit_id),expected,binding); units.append(value); inventories[unit_id]=inventory
    expected_order=[unit_id for unit_id,_,_ in pairs]
    actual_unit_names=sorted(item.name for item in (source/"units").iterdir()) if (source/"units").is_dir() else []
    if manifest.get("unit_count")!=len(units) or manifest.get("unit_order")!=expected_order or actual_unit_names!=sorted(expected_order) or manifest.get("unit_data_sha256")!={key:value["data_sha256"] for key,value in inventories.items()} or manifest.get("unit_manifest_sha256")!={key:value["manifest_sha256"] for key,value in inventories.items()}: raise ValueError("Full unit inventory mismatch")
    return units


class RunLock:
    def __init__(self,path:Path): self.path=path
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        try: descriptor=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError as error: raise RuntimeError(f"RunName locked: {self.path}") from error
        with os.fdopen(descriptor,"w",encoding="utf-8") as handle: json.dump({"pid":os.getpid()},handle); handle.flush(); os.fsync(handle.fileno())
        return self
    def __exit__(self,*args): self.path.unlink(missing_ok=True)


def _contract(pre: dict[str,Any],audit:dict[str,Any],run_name:str,mode:str,secondary_limit:int|None)->dict[str,Any]:
    return {"schema":SCHEMA,"run_name":run_name,"mode":mode,"config_sha256":audit["_sha256"],"checkpoint":pre["checkpoint"],"adapter":pre["adapter"],"teachers":pre["teachers"],"data":pre["data"],"lineage":pre["lineage"],"tokenizer":pre["tokenizer"],"branch_loss_audit":pre["branch_loss_audit"],"primary_exact":pre["primary_exact"],"secondary":pre["secondary"],"projection":audit["projection"],"classification":audit["classification"],"bootstrap":audit["bootstrap"],"secondary_limit":secondary_limit,"git":pre["git"],"implementation":pre["implementation"],"optimizer_constructed":False,"optimizer_steps_executed":0,"parameters_updated":False,"test_accessed":False}


def run_geometry(root:Path,config_path:Path,run_name:str,*,mode:str)->dict[str,Any]:
    run_name=_safe_name(run_name); pre=preflight(root,config_path); audit=load_audit_config(config_path,root)
    if mode in {"Full","Resume"}: require_clean_git(pre["git"],f"gradient geometry {mode}")
    dry=mode=="DryRun"; output=_resolve(root,audit["output_root"],output=True); category="dry_runs" if dry else "full_runs"; final=output/category/run_name; stage=output/category/f".{run_name}.stage"; limit=2 if dry else None; contract=_contract(pre,audit,run_name,"Full" if mode=="Resume" else mode,limit)
    with RunLock(output/"locks"/f"{run_name}.lock"):
        if mode!="Resume":
            if final.exists() or stage.exists(): raise FileExistsError("refusing existing run/stage")
            stage.mkdir(parents=True); atomic_json(stage/"contract.json",contract)
        else:
            if final.exists() or not stage.is_dir(): raise ValueError("Resume requires incomplete verified stage")
            if json.loads((stage/"contract.json").read_text())!=contract: raise ValueError("Resume contract mismatch")
        binding=_unit_binding(sha256_file(stage/"contract.json"),pre["git"],pre["implementation"])
        pairs=[("primary_exact",catalog_pair(0,pre["data"]["forget"]["samples"],pre["data"]["retain"]["samples"],16,42),"exact_step812_checkpoint_rng")]
        count=pre["secondary"]["paired_batches"] if limit is None else limit
        pairs += [(f"secondary_{index:05d}",catalog_pair(index,pre["data"]["forget"]["samples"],pre["data"]["retain"]["samples"],16,42),"standardized_per_pair_seed") for index in range(count)]
        cached={}
        for unit_id,pair,rng_strategy in pairs:
            expected={"schema":CACHE_SCHEMA,"unit_id":unit_id,"kind":"primary_exact" if unit_id=="primary_exact" else "secondary","catalog_index":pair["catalog_index"],"batch_hash":pair["batch_hash"],"rng_strategy":rng_strategy}; path=_unit_path(stage,unit_id)
            stale=list(path.parent.glob(f".{unit_id}.*.stage")) if path.parent.exists() else []
            if stale: raise ValueError(f"unpublished geometry unit staging evidence exists: {stale}")
            resumed=_resume_unit(path,expected,binding)
            if resumed is not None: cached[unit_id]=resumed[0]
        atomic_json(stage/"run_state.json",{"status":"RUNNING","optimizer_steps_executed":0,"test_accessed":False})
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); runtime=_load_runtime(root,config_path,device); source_before={pre["checkpoint"]["state_path"]:pre["checkpoint"]["state_sha256"],**{value["path"]:value["sha256"] for value in pre["teachers"].values()}}; student_before=_tensor_state_hash(runtime["current"]); teachers_before={role:_tensor_state_hash(runtime[role]) for role in ("original","augmented")}
        units=[]; completed=[]; skipped=[]
        try:
            for unit_id,pair,rng_strategy in pairs:
                expected={"schema":CACHE_SCHEMA,"unit_id":unit_id,"kind":"primary_exact" if unit_id=="primary_exact" else "secondary","catalog_index":pair["catalog_index"],"batch_hash":pair["batch_hash"],"rng_strategy":rng_strategy}
                path=_unit_path(stage,unit_id)
                if unit_id in cached: units.append(cached[unit_id]); skipped.append(unit_id); continue
                value=compute_geometry_unit(runtime,pair,unit_id=unit_id,rng_strategy=rng_strategy,device=device); _validate_unit(value,expected); _publish_unit(path,value,binding); units.append(value); completed.append(unit_id)
            if _tensor_state_hash(runtime["current"])!=student_before or {role:_tensor_state_hash(runtime[role]) for role in ("original","augmented")}!=teachers_before: raise RuntimeError("student/teacher changed across audit")
            source_after={path:sha256_file(Path(path)) for path in source_before}
            if source_after!=source_before: raise RuntimeError("checkpoint/teacher source changed")
            state={"status":"DRY_RUN_COMPLETED" if dry else "INFERENCE_COMPLETED","units":len(units),"completed":completed,"skipped":skipped,"checkpoint_sha256_before":source_before,"checkpoint_sha256_after":source_after,"gradient_computed":True,"backward_method":"torch.autograd.grad","loss_backward_called":False,"optimizer_constructed":False,"optimizer_steps_executed":0,"parameters_updated":False,"student_parameters_unchanged":True,"teacher_parameters_unchanged":True,"gradient_vectors_persisted":False,"full_vocabulary_logits_persisted":False,"test_accessed":False}
            atomic_json(stage/"run_state.json",state); atomic_json(stage/"provenance.json",{**pre,"runtime_parameter_sets":runtime["parameter_sets"],**state}); unit_order=[unit_id for unit_id,_,_ in pairs]; inventories={unit_id:_load_unit(_unit_path(stage,unit_id),{"schema":CACHE_SCHEMA,"unit_id":unit_id,"kind":"primary_exact" if unit_id=="primary_exact" else "secondary","catalog_index":pair["catalog_index"],"batch_hash":pair["batch_hash"],"rng_strategy":rng_strategy},binding)[1] for unit_id,pair,rng_strategy in pairs}; atomic_json(stage/"manifest.json",{"schema":SCHEMA,"contract_sha256":sha256_file(stage/"contract.json"),"run_state_sha256":sha256_file(stage/"run_state.json"),"provenance_sha256":sha256_file(stage/"provenance.json"),"unit_order":unit_order,"unit_data_sha256":{key:value["data_sha256"] for key,value in inventories.items()},"unit_manifest_sha256":{key:value["manifest_sha256"] for key,value in inventories.items()},"unit_count":len(inventories),"test_accessed":False}); atomic_text(stage/("DRY_RUN_COMPLETED" if dry else "INFERENCE_COMPLETED"),state["status"]+"\n"); final.parent.mkdir(parents=True,exist_ok=True); os.replace(stage,final); return {**state,"run_dir":str(final)}
        except BaseException as error:
            atomic_json(stage/"run_state.json",{"status":"FAILED","error":f"{type(error).__name__}: {error}","optimizer_steps_executed":0,"parameters_updated":False,"test_accessed":False}); raise
        finally:
            del runtime
            if device.type=="cuda": torch.cuda.empty_cache()


def verify_full(root:Path,config_path:Path,run_name:str,*,preflight_function=preflight)->dict[str,Any]:
    run_name=_safe_name(run_name); pre=preflight_function(root,config_path); require_clean_git(pre["git"],"gradient geometry Analyze"); audit=load_audit_config(config_path,root); output=_resolve(root,audit["output_root"],output=True); source=output/"full_runs"/run_name
    required=("contract.json","run_state.json","provenance.json","manifest.json","INFERENCE_COMPLETED")
    if not source.is_dir() or any(not (source/name).is_file() for name in required): raise ValueError("Full publication incomplete")
    contract=json.loads((source/"contract.json").read_text()); expected_contract=_contract(pre,audit,run_name,"Full",None)
    if contract!=expected_contract: raise ValueError("Full contract/HEAD/implementation mismatch")
    state=json.loads((source/"run_state.json").read_text()); safety={"status":"INFERENCE_COMPLETED","loss_backward_called":False,"optimizer_constructed":False,"optimizer_steps_executed":0,"parameters_updated":False,"gradient_vectors_persisted":False,"full_vocabulary_logits_persisted":False,"test_accessed":False}
    if any(state.get(key)!=value for key,value in safety.items()) or (source/"INFERENCE_COMPLETED").read_text()!="INFERENCE_COMPLETED\n": raise ValueError("Full state/completion safety mismatch")
    provenance=json.loads((source/"provenance.json").read_text());
    if provenance.get("git")!=pre["git"] or provenance.get("implementation")!=pre["implementation"]: raise ValueError("Full provenance mismatch")
    manifest=json.loads((source/"manifest.json").read_text()); hashes={"contract_sha256":sha256_file(source/"contract.json"),"run_state_sha256":sha256_file(source/"run_state.json"),"provenance_sha256":sha256_file(source/"provenance.json")}
    if manifest.get("schema")!=SCHEMA or manifest.get("test_accessed") is not False or any(manifest.get(key)!=value for key,value in hashes.items()): raise ValueError("Full artifact SHA/safety mismatch")
    pairs=[("primary_exact",catalog_pair(0,pre["data"]["forget"]["samples"],pre["data"]["retain"]["samples"],16,42),"exact_step812_checkpoint_rng")]+[(f"secondary_{i:05d}",catalog_pair(i,pre["data"]["forget"]["samples"],pre["data"]["retain"]["samples"],16,42),"standardized_per_pair_seed") for i in range(pre["secondary"]["paired_batches"])]
    binding=_unit_binding(hashes["contract_sha256"],pre["git"],pre["implementation"])
    units=_validate_unit_inventory(source,manifest,pairs,binding)
    for record in contract["teachers"].values():
        if sha256_file(Path(record["path"]))!=record["sha256"]: raise ValueError("teacher checkpoint changed")
    if sha256_file(Path(contract["checkpoint"]["state_path"]))!=contract["checkpoint"]["state_sha256"]: raise ValueError("step812 checkpoint changed")
    return {"source":source,"preflight":pre,"contract":contract,"state":state,"manifest":manifest,"units":units}


def analyze_full(root:Path,config_path:Path,run_name:str,*,verify_function=verify_full)->dict[str,Any]:
    verified=verify_function(root,config_path,run_name); audit=load_audit_config(config_path,root); output=_resolve(root,audit["output_root"],output=True); final=output/"analysis_runs"/run_name; stage=output/"analysis_runs"/f".{run_name}.stage"
    with RunLock(output/"locks"/f"analyze-{run_name}.lock"):
        if final.exists() or stage.exists(): raise FileExistsError("refusing existing analysis")
        stage.mkdir(parents=True); primary=next(unit for unit in verified["units"] if unit["unit_id"]=="primary_exact"); secondary=[unit for unit in verified["units"] if unit["kind"]=="secondary"]
        aggregate=aggregate_units(secondary,seed=audit["bootstrap"]["seed"],resamples=audit["bootstrap"]["resamples"],residual_tolerance=audit["projection"]["normalized_residual_tolerance"]); category=classify_geometry(primary,aggregate,eta_threshold=audit["classification"]["eta_f_min"],rho_zero=audit["projection"]["rho_zero_tolerance"],residual_tolerance=audit["projection"]["normalized_residual_tolerance"])
        result={"schema":ANALYSIS_SCHEMA,"run_name":run_name,"category":category,"primary_exact":primary,"secondary_aggregate":aggregate,"scope":"standardized frozen-step812 geometry; not step813-1200 trajectory replay","eta_threshold":0.10,"threshold_semantics":"engineering preregistration, not universal theory","gradient_computed":True,"backward_method":"torch.autograd.grad","loss_backward_called":False,"optimizer_constructed":False,"optimizer_steps_executed":0,"parameters_updated":False,"test_accessed":False,"provenance":{"source_run":str(verified["source"]),"source_manifest_sha256":sha256_file(verified["source"]/"manifest.json"),"git":verified["preflight"]["git"],"implementation":verified["preflight"]["implementation"]}}
        atomic_json(stage/"analysis.json",result); atomic_json(stage/"manifest.json",{"schema":ANALYSIS_SCHEMA,"analysis_sha256":sha256_file(stage/"analysis.json"),"test_accessed":False}); atomic_text(stage/"COMPLETED","ANALYSIS_COMPLETED\n"); final.parent.mkdir(parents=True,exist_ok=True); os.replace(stage,final); return {**result,"analysis_dir":str(final)}


def synthetic_dry_run(root:Path,config_path:Path,run_name:str)->dict[str,Any]:
    audit=load_audit_config(config_path,root); output=_resolve(root,audit["output_root"],output=True); final=output/"dry_runs"/_safe_name(run_name)
    if final.exists(): raise FileExistsError(final)
    cases={
        "rank2":(torch.tensor([1.,2.,3.]),torch.tensor([1.,0.,0.]),torch.tensor([0.,1.,0.])),
        "rank1":(torch.tensor([1.,2.,3.]),torch.tensor([1.,0.,0.]),torch.tensor([2.,0.,0.])),
        "rank0":(torch.tensor([1.,2.,3.]),torch.zeros(3),torch.zeros(3)),
        "orthogonal":(torch.tensor([0.,0.,1.]),torch.tensor([1.,0.,0.]),torch.tensor([0.,1.,0.])),
        "collinear":(torch.tensor([1.,0.,0.]),torch.tensor([1.,0.,0.]),torch.tensor([2.,0.,0.])),
        "conflict":(torch.tensor([1.,0.,0.]),torch.tensor([-1.,0.,0.]),torch.tensor([0.,1.,0.])),
        "zero_forget":(torch.zeros(3),torch.tensor([1.,0.,0.]),torch.tensor([0.,1.,0.])),
    }
    results={name:_geometry_scalars(f,k,s,audit["projection"]["relative_singular_tolerance"]) for name,(f,k,s) in cases.items()}; git=git_provenance(root); implementation=implementation_provenance(root,IMPLEMENTATION_FILES)
    state={"schema":SCHEMA,"status":"SYNTHETIC_DRY_RUN_COMPLETED","cases":results,"git":git,"implementation":implementation,"model_loaded":False,"gradient_computed":False,"loss_backward_called":False,"optimizer_constructed":False,"optimizer_steps_executed":0,"parameters_updated":False,"test_accessed":False}; final.parent.mkdir(parents=True,exist_ok=True); stage=final.parent/f".{run_name}.stage"; stage.mkdir(); atomic_json(stage/"synthetic_geometry.json",state); atomic_json(stage/"manifest.json",{"schema":SCHEMA,"artifact_sha256":sha256_file(stage/"synthetic_geometry.json"),"test_accessed":False}); os.replace(stage,final); return {**state,"run_dir":str(final)}


def main()->None:
    parser=argparse.ArgumentParser(description="T5 step812 zero-update gradient geometry audit"); parser.add_argument("--mode",choices=("Preflight","SyntheticDryRun","DryRun","Full","Resume","Analyze"),default="Preflight"); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--project-root",type=Path,default=Path.cwd()); parser.add_argument("--run-name"); args=parser.parse_args(); root=args.project_root.resolve(); config=args.config.resolve()
    if args.mode=="Preflight": result=preflight(root,config)
    else:
        if not args.run_name: raise ValueError(f"{args.mode} requires --run-name")
        if args.mode=="SyntheticDryRun": result=synthetic_dry_run(root,config,args.run_name)
        elif args.mode=="Analyze": result=analyze_full(root,config,args.run_name)
        else: result=run_geometry(root,config,args.run_name,mode=args.mode)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=="__main__": main()
