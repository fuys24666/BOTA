from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

PROTOCOL_NAME = "t5_reconstructed_official_code"
SCHEMA_VERSION = "t5-e2urec-diagnostics-v1"
ALLOWED_DEVELOPMENT_SPLITS = frozenset({"validation"})
FORBIDDEN_PATH_PARTS = frozenset({"test", "rella_test", "final_test", "final-test"})
LEGACY_TRANSFORMERS_VERSION = "4.28.1"
RUNTIME_TRANSFORMERS_VERSION = transformers.__version__
ARCHITECTURE_FIELDS = (
    "vocab_size",
    "d_model",
    "d_ff",
    "num_layers",
    "num_heads",
    "dropout_rate",
    "decoder_start_token_id",
    "eos_token_id",
    "pad_token_id",
)


@dataclass(frozen=True)
class Budget:
    forget_samples: int
    retain_samples: int
    batch_size: int
    forget_batches: int
    retain_batches: int
    joint_batches_per_epoch: int
    epochs: int
    forget_epochs: int
    warmup_steps: int
    joint_steps: int
    total_steps: int
    forget_sample_visits: int
    retain_sample_visits: int


def derive_budget(
    forget_samples: int,
    retain_samples: int,
    batch_size: int,
    epochs: int,
    forget_epochs: int,
) -> Budget:
    if min(forget_samples, retain_samples, batch_size, epochs) <= 0:
        raise ValueError("sample counts, batch size, and epochs must be positive")
    if not 0 <= forget_epochs <= epochs:
        raise ValueError("forget_epochs must be within total epochs")
    forget_batches = math.ceil(forget_samples / batch_size)
    retain_batches = math.ceil(retain_samples / batch_size)
    joint_batches = min(forget_batches, retain_batches)
    warmup_steps = forget_batches * forget_epochs
    joint_steps = joint_batches * (epochs - forget_epochs)
    return Budget(
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        batch_size=batch_size,
        forget_batches=forget_batches,
        retain_batches=retain_batches,
        joint_batches_per_epoch=joint_batches,
        epochs=epochs,
        forget_epochs=forget_epochs,
        warmup_steps=warmup_steps,
        joint_steps=joint_steps,
        total_steps=warmup_steps + joint_steps,
        forget_sample_visits=forget_samples * epochs,
        retain_sample_visits=min(retain_samples, forget_batches * batch_size)
        * (epochs - forget_epochs),
    )


def phase_for_step(step: int, budget: Budget) -> str:
    if not 1 <= step <= budget.total_steps:
        raise ValueError(f"step must be in [1, {budget.total_steps}]")
    return "forget_only" if step <= budget.warmup_steps else "joint"


def checkpoint_plan(budget: Budget) -> list[int]:
    required = {0, 200, 400, 600, 800, budget.warmup_steps}
    if budget.warmup_steps < budget.total_steps:
        required.add(budget.warmup_steps + 1)
    required.update({1000, 1200, budget.total_steps})
    return sorted(step for step in required if 0 <= step <= budget.total_steps)


def _resolved(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def reject_test_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    lowered_parts = {part.lower() for part in resolved.parts}
    if lowered_parts & FORBIDDEN_PATH_PARTS or "test" in resolved.name.lower():
        raise ValueError(f"test paths are forbidden in reconstructed diagnostics: {resolved}")
    return resolved


def require_development_split(name: str) -> str:
    if name not in ALLOWED_DEVELOPMENT_SPLITS:
        raise ValueError(
            f"only overall validation is available; forbidden development split: {name!r}"
        )
    return name


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _local_rng_snapshot() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _local_rng_restore(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _local_rng_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    numpy_equal = (
        left["numpy"][0] == right["numpy"][0]
        and np.array_equal(left["numpy"][1], right["numpy"][1])
        and left["numpy"][2:] == right["numpy"][2:]
    )
    return (
        left["python"] == right["python"]
        and numpy_equal
        and torch.equal(left["torch_cpu"], right["torch_cpu"])
        and len(left["torch_cuda"]) == len(right["torch_cuda"])
        and all(
            torch.equal(a, b)
            for a, b in zip(left["torch_cuda"], right["torch_cuda"])
        )
    )


def build_clean_t5_config_from_legacy(
    legacy_config: T5Config,
) -> tuple[T5Config, dict[str, Any]]:
    if not isinstance(legacy_config, T5Config):
        raise TypeError(f"expected T5Config, got {type(legacy_config)!r}")
    architecture = {}
    for field in ARCHITECTURE_FIELDS:
        if not hasattr(legacy_config, field):
            raise ValueError(f"missing structure-related T5 config field: {field}")
        architecture[field] = getattr(legacy_config, field)
    public_values = {
        key: copy.deepcopy(value)
        for key, value in legacy_config.__dict__.items()
        if not key.startswith("_")
    }
    public_values["output_attentions"] = False
    public_values["use_cache"] = False
    public_values["transformers_version"] = RUNTIME_TRANSFORMERS_VERSION
    clean = T5Config(**public_values)
    clean.output_attentions = False
    clean.use_cache = False
    clean._attn_implementation = "eager"
    if clean._attn_implementation_internal != "eager":
        raise RuntimeError("clean T5 config failed to pin eager attention")
    clean_architecture = {field: getattr(clean, field) for field in ARCHITECTURE_FIELDS}
    if clean_architecture != architecture:
        raise RuntimeError("clean T5 config changed architecture fields")
    return clean, {
        "source_transformers_version": LEGACY_TRANSFORMERS_VERSION,
        "runtime_transformers_version": RUNTIME_TRANSFORMERS_VERSION,
        "source": "legacy_public_config_fields",
        "output_attentions": False,
        "use_cache": False,
        "attention_implementation": "eager",
        "attention_authority": (
            "repository requirements plus legacy T5 manual-attention implementation"
        ),
        "current_default_rejected": "sdpa auto-selection may differ from legacy",
        "architecture_before": architecture,
        "architecture_after": clean_architecture,
        "architecture_unchanged": True,
        "old_config_attached_to_runtime_model": False,
    }


def validate_clean_layer_indices(model: T5ForConditionalGeneration) -> dict[str, Any]:
    encoder = [
        block.layer[0].SelfAttention.layer_idx for block in model.encoder.block
    ]
    decoder_self = [
        block.layer[0].SelfAttention.layer_idx for block in model.decoder.block
    ]
    decoder_cross = [
        block.layer[1].EncDecAttention.layer_idx for block in model.decoder.block
    ]
    expected_encoder = list(range(len(model.encoder.block)))
    expected_decoder = list(range(len(model.decoder.block)))
    if encoder != expected_encoder:
        raise RuntimeError(f"encoder layer_idx mismatch: {encoder}")
    if decoder_self != expected_decoder or decoder_cross != expected_decoder:
        raise RuntimeError(
            "decoder layer_idx mismatch: "
            f"self={decoder_self}, cross={decoder_cross}"
        )
    return {
        "encoder_self_attention": encoder,
        "decoder_self_attention": decoder_self,
        "decoder_cross_attention": decoder_cross,
        "continuous_and_correct": True,
        "source": "current_transformers_standard_constructor",
    }


def _state_tensor_audit(
    legacy_state: dict[str, torch.Tensor], rebuilt_state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    if legacy_state.keys() != rebuilt_state.keys():
        raise RuntimeError("state_dict key set changed during clean reconstruction")
    max_absolute_error = 0.0
    tensor_records = {}
    for key in legacy_state:
        left = legacy_state[key]
        right = rebuilt_state[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"state tensor metadata changed: {key}")
        if not torch.equal(left, right):
            raise RuntimeError(f"state tensor value changed: {key}")
        tensor_records[key] = {
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "max_absolute_error": 0.0,
        }
    return {
        "key_count": len(legacy_state),
        "key_set_unchanged": True,
        "all_shapes_unchanged": True,
        "all_dtypes_unchanged": True,
        "all_values_unchanged": True,
        "max_absolute_error": max_absolute_error,
        "tensors": tensor_records,
    }


def load_legacy_t5_for_reconstructed_diagnostics(
    path: Path,
) -> tuple[T5ForConditionalGeneration, dict[str, Any]]:
    resolved = path.resolve()
    checkpoint_before = sha256_file(resolved)
    rng_before = _local_rng_snapshot()
    try:
        with resolved.open("rb") as handle:
            legacy = torch.load(handle, map_location="cpu", weights_only=False)
        if not isinstance(legacy, T5ForConditionalGeneration):
            raise TypeError(
                f"diagnostic T5 loader refuses non-T5 checkpoint: {type(legacy)!r}"
            )
        clean_config, config_report = build_clean_t5_config_from_legacy(legacy.config)
        legacy_state = legacy.state_dict()
        clean = T5ForConditionalGeneration(clean_config)
        layer_idx_report = validate_clean_layer_indices(clean)
        incompatible = clean.load_state_dict(legacy_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "strict state_dict load returned missing/unexpected keys: "
                f"{incompatible}"
            )
        state_report = _state_tensor_audit(legacy_state, clean.state_dict())
        legacy_parameter_count = sum(value.numel() for value in legacy.parameters())
        clean_parameter_count = sum(value.numel() for value in clean.parameters())
        if legacy_parameter_count != clean_parameter_count:
            raise RuntimeError("parameter count changed during clean reconstruction")
        checkpoint_after = sha256_file(resolved)
        if checkpoint_after != checkpoint_before:
            raise RuntimeError("legacy checkpoint bytes changed during compatibility load")
        report = {
            "path": str(resolved),
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "checkpoint_sha256_unchanged": True,
            "strict_state_dict_load": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "parameter_count_before": legacy_parameter_count,
            "parameter_count_after": clean_parameter_count,
            "parameter_count_unchanged": True,
            "config": config_report,
            "layer_idx": layer_idx_report,
            "state_dict": state_report,
            "attention_implementation": "eager",
            "sdpa_or_flash_path_enabled": False,
            "legacy_object_forward_attempted": False,
            "legacy_attention_layer_idx_patched": False,
            "runtime_model": "clean_state_dict_reconstruction",
            "runtime_attention_implementation": "eager",
            "runtime_layer_idx_source": "current_transformers_standard_constructor",
            "weight_reconstruction_exact": True,
            "legacy_object_runtime_compatible": False,
            "historical_runtime_equivalence_claimed": False,
            "historical_bitwise_equivalence": False,
        }
        del legacy, legacy_state
        return clean, report
    finally:
        _local_rng_restore(rng_before)
        if not _local_rng_equal(_local_rng_snapshot(), rng_before):
            raise RuntimeError("compatibility loader failed to restore RNG")


def _gradient_comparison(
    left: T5ForConditionalGeneration, right: T5ForConditionalGeneration
) -> dict[str, Any]:
    left_parameters = dict(left.named_parameters())
    right_parameters = dict(right.named_parameters())
    if left_parameters.keys() != right_parameters.keys():
        raise RuntimeError("gradient parameter key set mismatch")
    max_error = 0.0
    for name in left_parameters:
        left_gradient = left_parameters[name].grad
        right_gradient = right_parameters[name].grad
        if (left_gradient is None) != (right_gradient is None):
            raise RuntimeError(f"gradient presence mismatch: {name}")
        if left_gradient is not None:
            if not torch.equal(left_gradient, right_gradient):
                raise RuntimeError(f"train gradient mismatch: {name}")
            if left_gradient.numel():
                max_error = max(
                    max_error,
                    float(
                        (left_gradient - right_gradient)
                        .detach()
                        .abs()
                        .max()
                        .cpu()
                    ),
                )
    return {
        "tensorwise_equal": True,
        "max_absolute_error": max_error,
        "parameter_keys_equal": True,
    }


def verify_clean_reconstruction_determinism(
    path: Path, device: torch.device | None = None
) -> dict[str, Any]:
    resolved = path.resolve()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    function_rng = _local_rng_snapshot()
    checkpoint_before = sha256_file(resolved)
    try:
        left, left_report = load_legacy_t5_for_reconstructed_diagnostics(resolved)
        right, right_report = load_legacy_t5_for_reconstructed_diagnostics(resolved)
        left.to(device)
        right.to(device)
        input_ids = torch.tensor([[10, 20, 30, 1]], dtype=torch.long, device=device)
        labels = torch.tensor([[2163, 5, 1]], dtype=torch.long, device=device)

        left.eval()
        right.eval()
        with torch.no_grad():
            left_eval = left(input_ids=input_ids, labels=labels)
            right_eval = right(input_ids=input_ids, labels=labels)
        eval_logits_equal = torch.equal(left_eval.logits, right_eval.logits)
        eval_loss_equal = torch.equal(left_eval.loss, right_eval.loss)
        if not eval_logits_equal or not eval_loss_equal:
            raise RuntimeError("two clean reconstructions differ in eval")
        eval_logits_error = float(
            (left_eval.logits - right_eval.logits).detach().abs().max().cpu()
        )
        eval_loss_error = float(
            (left_eval.loss - right_eval.loss).detach().abs().cpu()
        )
        if any(
            key.endswith("attentions")
            for key in set(left_eval.keys()) | set(right_eval.keys())
        ):
            raise RuntimeError("output_attentions=False unexpectedly returned attentions")

        left.train()
        right.train()
        left.zero_grad(set_to_none=True)
        right.zero_grad(set_to_none=True)
        train_rng = _local_rng_snapshot()
        left_train = left(input_ids=input_ids, labels=labels)
        left_train.loss.backward()
        left_post_rng = _local_rng_snapshot()
        _local_rng_restore(train_rng)
        right_train = right(input_ids=input_ids, labels=labels)
        right_train.loss.backward()
        right_post_rng = _local_rng_snapshot()
        if not torch.equal(left_train.logits, right_train.logits):
            raise RuntimeError("two clean reconstructions differ in train logits")
        if not torch.equal(left_train.loss, right_train.loss):
            raise RuntimeError("two clean reconstructions differ in train loss")
        if not _local_rng_equal(left_post_rng, right_post_rng):
            raise RuntimeError("two clean reconstructions differ in train RNG")
        gradient_report = _gradient_comparison(left, right)
        checkpoint_after = sha256_file(resolved)
        if checkpoint_after != checkpoint_before:
            raise RuntimeError("checkpoint hash changed during equivalence verification")
        return {
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "checkpoint_sha256_unchanged": True,
            "device": str(device),
            "attention_implementation": "eager",
            "left_clean_rebuild": left_report,
            "right_clean_rebuild": right_report,
            "eval": {
                "logits_equal": True,
                "loss_equal": True,
                "logits_max_absolute_error": eval_logits_error,
                "loss_absolute_error": eval_loss_error,
                "attentions_returned": False,
            },
            "train": {
                "logits_equal": True,
                "loss_equal": True,
                "logits_max_absolute_error": float(
                    (left_train.logits - right_train.logits)
                    .detach()
                    .abs()
                    .max()
                    .cpu()
                ),
                "loss_absolute_error": float(
                    (left_train.loss - right_train.loss).detach().abs().cpu()
                ),
                "gradient": gradient_report,
                "post_forward_rng_equal": True,
            },
            "weight_reconstruction_exact": True,
            "runtime_reconstruction_deterministic": True,
            "legacy_object_runtime_compatible": False,
            "historical_bitwise_equivalence": False,
            "legacy_object_forward_attempted": False,
            "exact": True,
            "test_loader_built": False,
            "test_accessed": False,
        }
    finally:
        _local_rng_restore(function_rng)
        if not _local_rng_equal(_local_rng_snapshot(), function_rng):
            raise RuntimeError("equivalence verifier failed to restore RNG")


def artifact_provenance(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "role": role,
        "path": str(resolved),
        "artifact_exists": True,
        "artifact_bytes": resolved.stat().st_size,
        "artifact_sha256": sha256_file(resolved),
        "historical_training_command_confirmed": False,
        "historical_selection_rule_confirmed": False,
        "historical_test_selection_unknown": True,
        "lineage_status": "legacy_artifact_unverified",
        "reconstructed_protocol_proves_historical_training": False,
    }


def load_config(path: Path, project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or Path.cwd()).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_name": PROTOCOL_NAME,
        "schema_version": SCHEMA_VERSION,
        "development_only": True,
        "test_access_policy": "forbidden",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must equal {value!r}")
    fixed = config["training"]
    required_training = {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "per_device_batch_size": 16,
        "gradient_accumulation": 1,
        "effective_batch_size": 16,
        "epochs": 20,
        "forget_epoch": 1,
        "alpha": 2.0,
        "code_weight": 0.6,
        "weight_semantics": "remembering_loss_weight",
        "sampler": "python_zip_legacy",
        "seed": 42,
        "scheduler": "none",
        "finite_gradient_clipping": "none",
    }
    for key, value in required_training.items():
        if fixed.get(key) != value:
            raise ValueError(f"training.{key} must equal {value!r}")
    lora = config["lora"]
    if lora != {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q", "v"],
    }:
        raise ValueError("LoRA configuration differs from reconstructed protocol")
    paths = config["paths"]
    for key in ("forget", "retain", "validation"):
        paths[key] = str(reject_test_path(_resolved(paths[key], root)))
    for key in ("original", "augmented_teacher", "retrain_reference"):
        paths[key] = str(_resolved(paths[key], root))
    output_root = _resolved(paths["output_root"], root)
    expected_output = (root / "outputs" / "t5_e2urec_diagnostics_v1").resolve()
    if output_root != expected_output:
        raise ValueError(f"output_root must be {expected_output}")
    paths["output_root"] = str(output_root)
    require_development_split(config["development"]["split"])
    budget = derive_budget(
        config["data"]["forget_samples"],
        config["data"]["retain_samples"],
        fixed["per_device_batch_size"],
        fixed["epochs"],
        fixed["forget_epoch"],
    )
    config["derived_budget"] = asdict(budget)
    config["checkpoint_plan"] = checkpoint_plan(budget)
    return config


def protocol_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def resume_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_name": PROTOCOL_NAME,
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_hash(config),
        "paths": {
            key: config["paths"][key]
            for key in (
                "original",
                "augmented_teacher",
                "retrain_reference",
                "forget",
                "retain",
                "validation",
            )
        },
        "training": config["training"],
        "lora": config["lora"],
        "derived_budget": config["derived_budget"],
        "checkpoint_plan": config["checkpoint_plan"],
        "diagnostic_schema": SCHEMA_VERSION,
        "test_access_policy": "forbidden",
    }


def validate_resume_contract(
    config: dict[str, Any], saved_contract: dict[str, Any]
) -> None:
    expected = resume_contract(config)
    if saved_contract != expected:
        differing = sorted(
            key
            for key in set(expected) | set(saved_contract)
            if expected.get(key) != saved_contract.get(key)
        )
        raise ValueError(f"resume contract mismatch: {differing}")


def prepare_run_directory(output_root: Path, run_name: str, dry_run: bool) -> Path:
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one non-empty path component")
    base = output_root / ("dry_runs" if dry_run else "")
    destination = (base / run_name).resolve()
    if output_root.resolve() not in destination.parents:
        raise ValueError("run output escaped the isolated output root")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


class JsonPromptDataset(Dataset):
    def __init__(self, path: Path, tokenizer: T5Tokenizer):
        reject_test_path(path)
        self.records = json.loads(path.read_text(encoding="utf-8"))
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        input_ids = self.tokenizer.encode(
            record["input"], padding=True, truncation=True, max_length=512
        )
        target_ids = self.tokenizer.encode(
            record["output"], padding=True, truncation=True, max_length=512
        )
        return {
            "sample_id": index,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        source_length = max(len(row["input_ids"]) for row in batch)
        target_length = max(len(row["target_ids"]) for row in batch)
        inputs = torch.full(
            (batch_size, source_length), self.tokenizer.pad_token_id, dtype=torch.long
        )
        targets = torch.full(
            (batch_size, target_length), self.tokenizer.pad_token_id, dtype=torch.long
        )
        sample_ids = torch.empty(batch_size, dtype=torch.long)
        for offset, row in enumerate(batch):
            inputs[offset, : len(row["input_ids"])] = row["input_ids"]
            targets[offset, : len(row["target_ids"])] = row["target_ids"]
            sample_ids[offset] = row["sample_id"]
        targets[targets == self.tokenizer.pad_token_id] = -100
        return {"sample_id": sample_ids, "input_ids": inputs, "target_ids": targets}


def make_loader(
    path: Path,
    tokenizer: T5Tokenizer,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = JsonPromptDataset(path, tokenizer)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        generator=generator,
        drop_last=False,
    )


def legacy_zip(
    retain_loader: Iterable[dict[str, torch.Tensor]],
    forget_loader: Iterable[dict[str, torch.Tensor]],
) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
    return zip(retain_loader, forget_loader)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_legacy_model(path: Path) -> torch.nn.Module:
    if path.is_dir():
        return T5ForConditionalGeneration.from_pretrained(path, attn_implementation="eager").float()
    model, _ = load_legacy_t5_for_reconstructed_diagnostics(path)
    return model


def build_current_model(original_path: Path, lora: dict[str, Any]) -> torch.nn.Module:
    model = load_legacy_model(original_path)
    return get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora["r"],
            lora_alpha=lora["lora_alpha"],
            target_modules=lora["target_modules"],
            lora_dropout=lora["lora_dropout"],
        ),
    )


def freeze_teacher(model: torch.nn.Module) -> torch.nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def batch_order_hash(*batches: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(batch["sample_id"].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def forced_logits(
    original_logits: torch.Tensor, augmented_logits: torch.Tensor, alpha: float
) -> torch.Tensor:
    return original_logits - alpha * F.relu(augmented_logits - original_logits)


def teacher_cross_entropy(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    teacher = F.softmax(teacher_logits, dim=-1)
    student = F.softmax(student_logits, dim=-1)
    token_loss = -(teacher * torch.log(student + 1e-12)).sum(-1)
    if reduction == "mean":
        return token_loss.mean()
    if reduction == "none":
        return token_loss.mean(dim=-1)
    raise ValueError(f"unsupported teacher loss reduction: {reduction}")


def compute_components(
    current: torch.nn.Module,
    original: torch.nn.Module,
    augmented: torch.nn.Module,
    forget_batch: dict[str, torch.Tensor],
    retain_batch: dict[str, torch.Tensor] | None,
    alpha: float,
) -> dict[str, torch.Tensor]:
    forget_student = current(
        input_ids=forget_batch["input_ids"], labels=forget_batch["target_ids"]
    ).logits
    with torch.no_grad():
        forget_original = original(
            input_ids=forget_batch["input_ids"], labels=forget_batch["target_ids"]
        ).logits
        forget_augmented = augmented(
            input_ids=forget_batch["input_ids"], labels=forget_batch["target_ids"]
        ).logits
    forced = forced_logits(forget_original, forget_augmented, alpha)
    result = {
        "L_forget": teacher_cross_entropy(forced, forget_student),
        "forced_logits": forced,
        "forget_original_logits": forget_original,
        "forget_augmented_logits": forget_augmented,
        "forget_student_logits": forget_student,
    }
    if retain_batch is not None:
        retain_output = current(
            input_ids=retain_batch["input_ids"], labels=retain_batch["target_ids"]
        )
        with torch.no_grad():
            retain_original = original(
                input_ids=retain_batch["input_ids"], labels=retain_batch["target_ids"]
            ).logits
        result.update(
            {
                "L_sup": retain_output.loss,
                "L_retain_KL": teacher_cross_entropy(
                    retain_original, retain_output.logits
                ),
                "retain_original_logits": retain_original,
                "retain_student_logits": retain_output.logits,
            }
        )
    return result


def total_loss(
    components: dict[str, torch.Tensor], phase: str, remembering_weight: float
) -> torch.Tensor:
    if phase == "forget_only":
        return components["L_forget"]
    if phase != "joint":
        raise ValueError(f"unknown phase: {phase}")
    return remembering_weight * (
        components["L_sup"] + components["L_retain_KL"]
    ) + (1.0 - remembering_weight) * components["L_forget"]


def make_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )


def beta_weight_semantics() -> dict[str, str]:
    return {
        "paper_beta": "forgetting_loss_weight",
        "code_weight": "remembering_loss_weight",
        "equivalence": "paper_beta = 1 - code_weight",
    }


def preflight(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_config(config_path, project_root)
    paths = config["paths"]
    artifacts = [
        artifact_provenance(Path(paths["original"]), "original_source"),
        artifact_provenance(Path(paths["augmented_teacher"]), "forgetting_teacher"),
        artifact_provenance(Path(paths["retrain_reference"]), "retrain_development_reference"),
    ]
    actual_counts = {}
    for split in ("forget", "retain", "validation"):
        actual_counts[split] = len(
            json.loads(Path(paths[split]).read_text(encoding="utf-8"))
        )
    expected_counts = {
        "forget": config["data"]["forget_samples"],
        "retain": config["data"]["retain_samples"],
        "validation": config["data"]["validation_samples"],
    }
    if actual_counts != expected_counts:
        raise ValueError(f"data count mismatch: {actual_counts} != {expected_counts}")
    return {
        "protocol_name": PROTOCOL_NAME,
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": protocol_hash(config),
        "config": config,
        "actual_counts": actual_counts,
        "artifacts": artifacts,
        "teacher_selection": {
            "augmented_step": 1200,
            "public_shell_specified": True,
            "training_endpoint": False,
            "selection_rationale_confirmed": False,
            "reselected_this_run": False,
            "test_used_to_validate_teacher_this_run": False,
        },
        "test_loader_built": False,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only reconstructed-official T5 E2URec preflight"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = preflight(arguments.config, arguments.project_root)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
