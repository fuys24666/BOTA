from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _teacher_logits(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode(): value = model(input_ids=batch["input_ids"], labels=batch["target_ids"]).logits
    result = value.clone(); del value
    if result.requires_grad or torch.is_inference(result): raise RuntimeError("teacher output must be ordinary gradient-free tensor")
    return result


def _teacher_ce_from_logits(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    teacher = F.softmax(teacher_logits, dim=-1); loss = -(teacher * torch.log(F.softmax(student_logits, dim=-1) + 1e-12)).sum(-1).mean(); del teacher
    return loss


def forget_loss_light(current: torch.nn.Module, original: torch.nn.Module, augmented: torch.nn.Module, batch: dict[str, torch.Tensor], alpha: float) -> torch.Tensor:
    original_logits = _teacher_logits(original, batch); augmented_logits = _teacher_logits(augmented, batch); forced = original_logits - alpha * F.relu(augmented_logits - original_logits); del original_logits, augmented_logits
    student_logits = current(input_ids=batch["input_ids"], labels=batch["target_ids"]).logits; loss = _teacher_ce_from_logits(forced, student_logits); del forced, student_logits
    return loss


def joint_losses_light(current: torch.nn.Module, original: torch.nn.Module, augmented: torch.nn.Module, forget: dict[str, torch.Tensor], retain: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    original_forget = _teacher_logits(original, forget); augmented_forget = _teacher_logits(augmented, forget); forced = original_forget - alpha * F.relu(augmented_forget - original_forget); del original_forget, augmented_forget
    retain_original = _teacher_logits(original, retain)
    forget_student = current(input_ids=forget["input_ids"], labels=forget["target_ids"]).logits; retain_output = current(input_ids=retain["input_ids"], labels=retain["target_ids"])
    result = {"L_forget": _teacher_ce_from_logits(forced, forget_student), "L_sup": retain_output.loss, "L_retain_KL": _teacher_ce_from_logits(retain_original, retain_output.logits)}; del forced, retain_original, forget_student, retain_output
    return result


def teachers_are_frozen(*models: torch.nn.Module) -> bool:
    return all(not parameter.requires_grad and parameter.grad is None for model in models for parameter in model.parameters())

