from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
import numpy as np
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from transformers import T5Tokenizer

from src.diagnostics.t5_full_runner import _batch
from src.diagnostics.t5_reconstructed_official import (
    JsonPromptDataset,
    build_current_model,
    freeze_teacher,
    load_legacy_model,
    move_batch,
)


def tokenizer_and_dataset(model_dir: Path, data_path: Path) -> tuple[T5Tokenizer, JsonPromptDataset]:
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    return tokenizer, JsonPromptDataset(data_path, tokenizer)


def batch(dataset: JsonPromptDataset, indices: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    return move_batch(_batch(dataset, indices), device)


def answer_loss(model: torch.nn.Module, value: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(input_ids=value["input_ids"], labels=value["target_ids"]).loss


def clean_full_model(checkpoint: Path, device: torch.device, trainable: bool = True) -> torch.nn.Module:
    model = load_legacy_model(checkpoint)
    for parameter in model.parameters(): parameter.requires_grad_(trainable)
    model.train(trainable)
    return model.to(device)


def clean_lora_model(checkpoint: Path, lora: dict[str, Any], device: torch.device) -> torch.nn.Module:
    return build_current_model(checkpoint, lora).to(device)


def frozen_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    return freeze_teacher(load_legacy_model(checkpoint)).to(device)


def restore_full_from_state(checkpoint: Path, state: dict[str, torch.Tensor], device: torch.device, trainable: bool = False) -> torch.nn.Module:
    model = load_legacy_model(checkpoint)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("strict full-model reconstruction failed")
    for parameter in model.parameters(): parameter.requires_grad_(trainable)
    model.train(trainable)
    return model.to(device)


def trainable_state(model: torch.nn.Module, model_type: str) -> dict[str, torch.Tensor]:
    source = get_peft_model_state_dict(model) if model_type == "lora_adapter" else model.state_dict()
    return {name: value.detach().cpu().clone() for name, value in source.items()}


def load_trainable_state(model: torch.nn.Module, state: dict[str, torch.Tensor], model_type: str) -> None:
    if model_type == "lora_adapter":
        set_peft_model_state_dict(model, state)
    else:
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys: raise ValueError("strict model state load failed")


def model_counts(model: torch.nn.Module) -> tuple[int, int]:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), sum(parameter.numel() for parameter in model.parameters())


def development_smoke(model: torch.nn.Module, dataset: JsonPromptDataset, device: torch.device, count: int = 16) -> dict[str, Any]:
    previous = model.training; model.eval(); value = batch(dataset, list(range(min(count, len(dataset)))), device)
    with torch.inference_mode():
        output = model(input_ids=value["input_ids"], labels=value["target_ids"])
        loss = float(output.loss.detach().cpu())
        finite = bool(torch.isfinite(output.logits).all().item()) and bool(torch.isfinite(output.loss).item())
    model.train(previous)
    return {"samples": len(value["input_ids"]), "loss": loss, "finite": finite, "test_accessed": False}


def development_binary_metrics(model: torch.nn.Module, dataset: JsonPromptDataset, device: torch.device, batch_size: int = 4) -> dict[str, Any]:
    probabilities, labels, losses = development_binary_predictions(model, dataset, device, batch_size)
    p = np.asarray(probabilities); y = np.asarray(labels)
    return {"samples": len(labels), "auc": float(roc_auc_score(y, p)), "accuracy": float(accuracy_score(y, p >= .5)), "log_loss": float(log_loss(y, p, labels=[0, 1])), "answer_loss": float(np.mean(losses)), "prediction_collapse": bool(np.std(p) < 1e-8), "test_accessed": False}


def development_binary_predictions(model: torch.nn.Module, dataset: JsonPromptDataset, device: torch.device, batch_size: int = 4) -> tuple[list[float], list[int], list[float]]:
    previous = model.training; model.eval(); probabilities: list[float] = []; labels: list[int] = []; losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            value = batch(dataset, list(range(start, min(start + batch_size, len(dataset)))), device)
            output = model(input_ids=value["input_ids"], labels=value["target_ids"])
            targets = value["target_ids"][:, 0]; yes = torch.softmax(output.logits[:, 0, [465, 2163]], -1)[:, 1]
            probabilities.extend(yes.detach().cpu().tolist()); labels.extend((targets == 2163).long().cpu().tolist()); losses.append(float(output.loss.detach().cpu()))
    model.train(previous); return probabilities, labels, losses


def release(*objects: Any) -> None:
    for obj in objects: del obj
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()
