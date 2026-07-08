from __future__ import annotations

import math
from dataclasses import dataclass

from gz.trainer.data import TrainingBatch
from gz.trainer.sampler import step_seed


@dataclass(frozen=True, slots=True)
class LoopConfig:
    lr: float = 3e-4
    warmup_steps: int = 200
    total_steps: int = 1000
    lr_schedule: str = "cosine"
    value_weight: float = 1.0
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    run_seed: int = 0
    # Train both orientations of every pair (targets z and -z) instead of
    # a random per-step flip: whittlezero's mirrored value stream.
    value_mirror: bool = False


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
        # bf16 autocast on CUDA, matching the evaluator's serving numerics.
        # Params and optimizer state stay f32; no GradScaler is needed for
        # bf16 (full f32 exponent range).
        self.device_type = next(model.parameters()).device.type

    def train_step(self, batch: TrainingBatch, with_metrics: bool = True) -> StepMetrics | None:
        """One optimizer step. With `with_metrics=False` the step enqueues no
        host-device synchronization at all (no `.item()`/`.cpu()`), so
        back-to-back steps pipeline on the GPU; callers request metrics only
        on the steps they log."""
        torch = _torch()
        functional = torch.nn.functional
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=self.device_type == "cuda",
        ):
            pair_mode = getattr(getattr(self.model, "arch", None), "value_input", None) == "pair"
            mirror = self.config.value_mirror and pair_mode
            value_flip = None
            if pair_mode and not mirror:
                value_flip = pair_value_flip(torch, batch, self.config.run_seed, self.step_index)
            value_raw, logits = self.model(
                batch.features, value_flip=value_flip, value_mirror=mirror
            )
            policy_loss = policy_ce_loss(
                logits, batch.policy, batch.features.action_count, batch.row_count
            )
            tanh_head = (
                getattr(getattr(self.model, "arch", None), "value_activation", "logit") == "tanh"
            )
            value_loss_fn = value_mse_loss if tanh_head else value_bce_loss
            if mirror:
                # whittlezero's mirrored stream: every pair trains both
                # orientations (targets z and -z); the swapped example is
                # masked to rows that actually carry an opponent state.
                canonical, mirrored = value_raw[0], value_raw[1]
                value_loss = value_loss_fn(
                    canonical, batch.value, batch.value_valid, batch.row_count
                )
                present = getattr(batch.features, "opponent_state_present", None)
                if present is not None:
                    mirrored_valid = batch.value_valid * (present > 0).to(batch.value_valid.dtype)
                    value_loss = 0.5 * value_loss + 0.5 * value_loss_fn(
                        mirrored, -batch.value, mirrored_valid, batch.row_count
                    )
                value_raw = canonical
                value = batch.value
            else:
                value = flipped_value_targets(torch, batch.value, value_flip)
                value_loss = value_loss_fn(value_raw, value, batch.value_valid, batch.row_count)
            loss = policy_loss + self.config.value_weight * value_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        lr = lr_at_step(
            self.config.lr,
            self.step_index + 1,
            self.config.warmup_steps,
            self.config.total_steps,
            self.config.lr_schedule,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.step_index += 1

        if not with_metrics:
            return None

        with torch.no_grad():
            row_mask = _row_mask(torch, batch.row_count, value_raw.shape[0], value_raw.device)
            valid = row_mask & (batch.value_valid > 0)
            valid_count = valid.sum()
            if bool(valid_count.item()):
                prediction = torch.where(value_raw[valid] >= 0, 1.0, -1.0)
                # Accuracy against the same flipped targets the loss saw:
                # value_raw came from flipped pair inputs, so the unflipped
                # batch.value counts correct flipped predictions as wrong.
                # label_mean stays unflipped -- it reports the stored data.
                label = torch.where(value[valid] >= 0, 1.0, -1.0)
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
    weight = valid.to(value_raw.dtype)
    # Fully tensorized: a data-dependent host branch here would synchronize
    # the CUDA stream between the forward pass and backward, stalling the
    # GPU mid-step. Invalid rows may carry arbitrary label bytes, so their
    # targets are zeroed before the pointwise loss and their terms weighted
    # out; zero valid rows yields loss 0 with finite gradients.
    target = torch.where(valid, (value + 1.0) * 0.5, torch.zeros_like(value))
    per_row = functional.binary_cross_entropy_with_logits(
        2.0 * value_raw, target, reduction="none"
    )
    return (per_row * weight).sum() / weight.sum().clamp(min=1.0)


def value_mse_loss(value_raw: object, value: object, value_valid: object, row_count: int) -> object:
    # whittlezero's value loss: MSE against the +/-1 target on the
    # tanh-bounded head. Masking mirrors value_bce_loss -- fully
    # tensorized, invalid rows zeroed and weighted out.
    torch = _torch()
    row_mask = _row_mask(torch, row_count, value_raw.shape[0], value_raw.device)
    valid = row_mask & (value_valid > 0)
    weight = valid.to(value_raw.dtype)
    target = torch.where(valid, value, torch.zeros_like(value))
    per_row = (value_raw - target) ** 2
    return (per_row * weight).sum() / weight.sum().clamp(min=1.0)


def pair_value_flip(torch: object, batch: TrainingBatch, run_seed: int, step: int) -> object:
    if getattr(batch.features, "opponent_state_present", None) is None:
        return None
    device = batch.value.device
    generator = torch.Generator(device=device)
    generator.manual_seed(step_seed(run_seed, step))
    row_mask = _row_mask(torch, batch.row_count, batch.value.shape[0], device)
    present = batch.features.opponent_state_present > 0
    return (torch.rand(batch.value.shape, generator=generator, device=device) < 0.5) & row_mask & present


def flipped_value_targets(torch: object, value: object, value_flip: object) -> object:
    if value_flip is None:
        return value
    return torch.where(value_flip, -value, value)


def lr_at_step(
    base_lr: float,
    step: int,
    warmup_steps: int,
    total_steps: int,
    schedule: str = "cosine",
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if schedule == "constant":
        return base_lr
    if total_steps <= warmup_steps:
        return base_lr
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _row_mask(torch: object, row_count: int, capacity: int, device: object) -> object:
    return torch.arange(capacity, device=device) < row_count


def _torch():
    import torch

    return torch
