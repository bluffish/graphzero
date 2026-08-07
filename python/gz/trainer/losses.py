from __future__ import annotations

import math
from typing import TYPE_CHECKING

from gz.trainer.data import TrainingBatch

if TYPE_CHECKING:
    from gz.trainer.loop import LoopConfig


def policy_ce_loss(
    logits: object,
    policy: object,
    action_count: object,
    row_count: int,
    *,
    action_kind: object | None = None,
    mask_stop: bool = False,
) -> object:
    torch = _torch()
    action_index = torch.arange(logits.shape[1], device=logits.device)
    action_mask = action_index.unsqueeze(0) < action_count.unsqueeze(1)
    if mask_stop:
        if action_kind is None:
            raise ValueError("mask_stop requires action_kind")
        action_mask = action_mask & (action_kind != 1)
    row_mask = _row_mask(torch, row_count, logits.shape[0], logits.device)
    masked_logits = logits.masked_fill(~action_mask, -1.0e9)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    policy_masked = torch.where(action_mask, policy, torch.zeros_like(policy))
    per_row = -(policy_masked * log_probs).sum(dim=1)
    valid = row_mask & (policy_masked.sum(dim=1) > 0)
    weight = valid.to(per_row.dtype)
    return (per_row * weight).sum() / weight.sum().clamp(min=1.0)


def policy_diagnostic_metrics(
    logits: object,
    policy: object,
    action_count: object,
    row_count: int,
    *,
    action_kind: object | None = None,
    mask_stop: bool = False,
) -> tuple[object, object, object, object, object]:
    torch = _torch()
    logits = logits.float()
    policy = policy.float()
    action_index = torch.arange(logits.shape[1], device=logits.device)
    action_mask = action_index.unsqueeze(0) < action_count.unsqueeze(1)
    if mask_stop:
        if action_kind is None:
            raise ValueError("mask_stop requires action_kind")
        action_mask = action_mask & (action_kind != 1)
    row_mask = _row_mask(torch, row_count, logits.shape[0], logits.device)
    target = torch.where(
        action_mask,
        policy.clamp_min(0.0),
        torch.zeros_like(policy),
    )
    target_mass = target.sum(dim=1, keepdim=True)
    target = torch.where(
        target_mass > 0.0,
        target / target_mass.clamp_min(torch.finfo(target.dtype).tiny),
        torch.zeros_like(target),
    )
    valid_actions = action_mask.sum(dim=1)
    valid = row_mask & (target_mass.squeeze(1) > 0.0) & (valid_actions > 0)
    weight = valid.to(logits.dtype)
    denominator = weight.sum().clamp(min=1.0)

    log_probs = torch.log_softmax(logits.masked_fill(~action_mask, -1.0e9), dim=-1)
    target_entropy = -(
        target * target.clamp_min(torch.finfo(target.dtype).tiny).log()
    ).sum(dim=1)
    cross_entropy = -(target * log_probs).sum(dim=1)
    uniform_cross_entropy = valid_actions.clamp_min(1).float().log()

    action_weight = action_mask.to(logits.dtype)
    action_denominator = valid_actions.clamp_min(1).to(logits.dtype)
    logit_mean = (logits * action_weight).sum(dim=1) / action_denominator
    logit_variance = (
        (logits - logit_mean.unsqueeze(1)).square() * action_weight
    ).sum(dim=1) / action_denominator
    logit_std = logit_variance.sqrt()
    logit_abs_mean = (logits.abs() * action_weight).sum(dim=1) / action_denominator

    def mean(value: object) -> object:
        return (value * weight).sum() / denominator

    entropy = mean(target_entropy)
    return (
        entropy,
        (mean(cross_entropy) - entropy).clamp_min(0.0),
        mean(uniform_cross_entropy - cross_entropy),
        mean(logit_std),
        mean(logit_abs_mean),
    )


def softened_policy_target(
    policy: object,
    action_count: object,
    *,
    temperature: float,
    action_kind: object | None = None,
    mask_stop: bool = False,
) -> object:
    if not math.isfinite(temperature) or temperature <= 1.0:
        raise ValueError("soft-policy temperature must be finite and greater than one")
    torch = _torch()
    action_index = torch.arange(policy.shape[1], device=policy.device)
    action_mask = action_index.unsqueeze(0) < action_count.unsqueeze(1)
    if mask_stop:
        if action_kind is None:
            raise ValueError("mask_stop requires action_kind")
        action_mask = action_mask & (action_kind != 1)
    masked = torch.where(
        action_mask,
        policy.float().clamp_min(0.0),
        torch.zeros_like(policy, dtype=torch.float32),
    )
    softened = masked.pow(1.0 / temperature)
    normalizer = softened.sum(dim=1, keepdim=True)
    return torch.where(
        normalizer > 0.0,
        softened / normalizer.clamp_min(torch.finfo(softened.dtype).tiny),
        torch.zeros_like(softened),
    )


def soft_policy_ce_loss(
    logits: object,
    policy: object,
    action_count: object,
    row_count: int,
    *,
    temperature: float,
    action_kind: object | None = None,
    mask_stop: bool = False,
) -> tuple[object, object, object]:
    torch = _torch()
    target = softened_policy_target(
        policy,
        action_count,
        temperature=temperature,
        action_kind=action_kind,
        mask_stop=mask_stop,
    )
    loss = policy_ce_loss(
        logits,
        target,
        action_count,
        row_count,
        action_kind=action_kind,
        mask_stop=mask_stop,
    )
    row_mask = _row_mask(torch, row_count, logits.shape[0], logits.device)
    valid = row_mask & (target.sum(dim=1) > 0.0)
    per_row_entropy = -(
        target * target.clamp_min(torch.finfo(target.dtype).tiny).log()
    ).sum(dim=1)
    weight = valid.to(per_row_entropy.dtype)
    entropy = (per_row_entropy * weight).sum() / weight.sum().clamp(min=1.0)
    return loss, (loss - entropy).clamp_min(0.0), entropy


def value_mse_loss(value_raw: object, value: object, value_valid: object, row_count: int) -> object:
    # Fully tensorized masked MSE for bounded sign and auxiliary labels.
    torch = _torch()
    row_mask = _row_mask(torch, row_count, value_raw.shape[0], value_raw.device)
    valid = row_mask & (value_valid > 0)
    weight = valid.to(value_raw.dtype)
    target = torch.where(valid, value, torch.zeros_like(value))
    per_row = (value_raw - target) ** 2
    return (per_row * weight).sum() / weight.sum().clamp(min=1.0)


def value_smooth_l1_loss(
    value_raw: object,
    value: object,
    value_valid: object,
    row_count: int,
) -> object:
    torch = _torch()
    row_mask = _row_mask(torch, row_count, value_raw.shape[0], value_raw.device)
    valid = row_mask & (value_valid > 0)
    weight = valid.to(value_raw.dtype)
    target = torch.where(valid, value, torch.zeros_like(value))
    per_row = torch.nn.functional.smooth_l1_loss(
        value_raw,
        target,
        reduction="none",
        beta=1.0,
    )
    return (per_row * weight).sum() / weight.sum().clamp(min=1.0)


def value_head_loss(
    value_raw: object,
    value: object,
    value_valid: object,
    row_count: int,
) -> object:
    return value_mse_loss(value_raw, value, value_valid, row_count)


def auxiliary_task_loss(
    model: object,
    value_raw: object,
    horizon_raw: object,
    score_raw: object,
    batch: TrainingBatch,
    config: LoopConfig,
) -> tuple[object, object, object, object, object, object]:
    torch = _torch()
    final_loss = value_head_loss(
        value_raw,
        batch.value,
        batch.value_valid,
        batch.row_count,
    )
    v8_loss = value_mse_loss(
        horizon_raw[:, 0],
        batch.horizon_value[:, 0],
        batch.horizon_value_valid,
        batch.row_count,
    )
    v32_loss = value_mse_loss(
        horizon_raw[:, 1],
        batch.horizon_value[:, 1],
        batch.horizon_value_valid,
        batch.row_count,
    )
    terminal_cost = -batch.reward
    if config.terminal_score_target == "node-count":
        normalized_prediction = torch.sigmoid(score_raw)
        normalized_target = terminal_cost / float(model.schema.max_nodes)
        score_loss = value_mse_loss(
            normalized_prediction,
            normalized_target,
            batch.horizon_value_valid,
            batch.row_count,
        )
        score_prediction = normalized_prediction * float(model.schema.max_nodes)
    elif config.terminal_score_target == "log1p-cost-ms":
        log_prediction = torch.nn.functional.softplus(score_raw)
        log_target = torch.log1p(terminal_cost.clamp_min(0.0))
        score_loss = value_smooth_l1_loss(
            log_prediction,
            log_target,
            batch.horizon_value_valid,
            batch.row_count,
        )
        score_prediction = torch.expm1(log_prediction)
    else:
        raise ValueError(f"unknown terminal score target: {config.terminal_score_target}")
    weighted_losses = (
        (config.value_final_weight, final_loss),
        (config.value_v8_weight, v8_loss),
        (config.value_v32_weight, v32_loss),
        (config.terminal_score_weight, score_loss),
    )
    combined = sum(weight * loss for weight, loss in weighted_losses if weight > 0.0)
    return combined, final_loss, v8_loss, v32_loss, score_loss, score_prediction


def _validate_task_weights(config: LoopConfig) -> None:
    weights = (
        config.value_final_weight,
        config.value_v8_weight,
        config.value_v32_weight,
        config.terminal_score_weight,
    )
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("value task weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("value task weights must sum to one")
    if config.terminal_score_target not in ("node-count", "log1p-cost-ms"):
        raise ValueError(f"unknown terminal score target: {config.terminal_score_target}")
    if not math.isfinite(config.soft_policy_weight) or config.soft_policy_weight < 0.0:
        raise ValueError("soft-policy weight must be finite and non-negative")
    if (
        not math.isfinite(config.soft_policy_temperature)
        or config.soft_policy_temperature <= 1.0
    ):
        raise ValueError("soft-policy temperature must be finite and greater than one")
    if not math.isfinite(config.soft_policy_trunk_grad_scale) or not (
        0.0 <= config.soft_policy_trunk_grad_scale <= 1.0
    ):
        raise ValueError("soft-policy trunk gradient scale must be finite and in [0, 1]")


def lr_at_step(
    base_lr: float,
    step: int,
    warmup_steps: int,
    total_steps: int,
    schedule: str = "cosine",
    min_lr_ratio: float = 0.0,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if schedule == "constant":
        return base_lr
    if total_steps <= warmup_steps:
        return base_lr
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _row_mask(torch: object, row_count: int, capacity: int, device: object) -> object:
    return torch.arange(capacity, device=device) < row_count


def _trim_to_row_count(batch: TrainingBatch) -> TrainingBatch:
    row_count = batch.row_count
    if batch.value.shape[0] == row_count:
        return batch
    features = type(batch.features)(*(tensor[:row_count] for tensor in batch.features))
    return TrainingBatch(
        features=features,
        policy=batch.policy[:row_count],
        value=batch.value[:row_count],
        value_valid=batch.value_valid[:row_count],
        horizon_value=batch.horizon_value[:row_count],
        horizon_value_valid=batch.horizon_value_valid[:row_count],
        reward=batch.reward[:row_count],
        row_count=row_count,
    )


def _torch():
    import torch

    return torch
