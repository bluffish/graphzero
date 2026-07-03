from __future__ import annotations

import math
from dataclasses import dataclass

from gz.trainer.data import TrainingBatch


@dataclass(frozen=True, slots=True)
class LoopConfig:
    lr: float = 3e-4
    warmup_steps: int = 200
    total_steps: int = 1000
    value_weight: float = 1.0
    grad_clip: float = 1.0
    weight_decay: float = 0.01


@dataclass(frozen=True, slots=True)
class StepMetrics:
    step: int
    policy_loss: float
    value_loss: float
    loss: float
    grad_norm: float
    lr: float
    value_accuracy: float
    fraction_valid: float
    label_mean: float
    terminal_cost_mean: float
    terminal_cost_best: float


class TrainerLoop:
    def __init__(self, model: object, config: LoopConfig) -> None:
        torch = _torch()
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.step_index = 0

    def train_step(self, batch: TrainingBatch) -> StepMetrics:
        torch = _torch()
        functional = torch.nn.functional
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        value_raw, logits = self.model(batch.features)
        policy_loss = policy_ce_loss(logits, batch.policy, batch.features.action_count, batch.row_count)
        value_loss = value_bce_loss(value_raw, batch.value, batch.value_valid, batch.row_count)
        loss = policy_loss + self.config.value_weight * value_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        lr = lr_at_step(self.config.lr, self.step_index + 1, self.config.warmup_steps, self.config.total_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.step_index += 1

        with torch.no_grad():
            row_mask = _row_mask(torch, batch.row_count, value_raw.shape[0], value_raw.device)
            valid = row_mask & (batch.value_valid > 0)
            valid_count = valid.sum()
            if bool(valid_count.item()):
                prediction = torch.where(value_raw[valid] >= 0, 1.0, -1.0)
                label = torch.where(batch.value[valid] >= 0, 1.0, -1.0)
                value_accuracy = (prediction == label).float().mean()
                label_mean = batch.value[valid].mean()
            else:
                value_accuracy = value_raw.new_tensor(0.0)
                label_mean = value_raw.new_tensor(0.0)
            fraction_valid = valid.float().mean()
            # Whittle reward is -(measured cost); report the cost directly.
            # Row-weighted: long episodes contribute more rows to the batch.
            costs = -batch.reward[row_mask]
            terminal_cost_mean = costs.mean() if batch.row_count else costs.new_tensor(0.0)
            terminal_cost_best = costs.min() if batch.row_count else costs.new_tensor(0.0)

        return StepMetrics(
            step=self.step_index,
            policy_loss=float(policy_loss.detach().cpu()),
            value_loss=float(value_loss.detach().cpu()),
            loss=float(loss.detach().cpu()),
            grad_norm=float(grad_norm.detach().cpu()),
            lr=lr,
            value_accuracy=float(value_accuracy.detach().cpu()),
            fraction_valid=float(fraction_valid.detach().cpu()),
            label_mean=float(label_mean.detach().cpu()),
            terminal_cost_mean=float(terminal_cost_mean.detach().cpu()),
            terminal_cost_best=float(terminal_cost_best.detach().cpu()),
        )


def policy_ce_loss(logits: object, policy: object, action_count: object, row_count: int) -> object:
    torch = _torch()
    action_index = torch.arange(logits.shape[1], device=logits.device)
    action_mask = action_index.unsqueeze(0) < action_count.unsqueeze(1)
    row_mask = _row_mask(torch, row_count, logits.shape[0], logits.device)
    masked_logits = logits.masked_fill(~action_mask, -1.0e9)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    policy_masked = torch.where(action_mask, policy, torch.zeros_like(policy))
    per_row = -(policy_masked * log_probs).sum(dim=1)
    denom = max(row_count, 1)
    return (per_row * row_mask.to(per_row.dtype)).sum() / denom


def value_bce_loss(value_raw: object, value: object, value_valid: object, row_count: int) -> object:
    torch = _torch()
    functional = torch.nn.functional
    row_mask = _row_mask(torch, row_count, value_raw.shape[0], value_raw.device)
    valid = row_mask & (value_valid > 0)
    if not bool(valid.any().item()):
        return value_raw.sum() * 0.0
    target = (value[valid] + 1.0) * 0.5
    return functional.binary_cross_entropy_with_logits(2.0 * value_raw[valid], target, reduction="mean")


def lr_at_step(base_lr: float, step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _row_mask(torch: object, row_count: int, capacity: int, device: object) -> object:
    return torch.arange(capacity, device=device) < row_count


def _torch():
    import torch

    return torch
