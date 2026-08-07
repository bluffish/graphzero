from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gz.trainer.data import TrainingBatch
from gz.trainer.diagnostics import (
    AuxiliarySignalMetrics,
    ParameterUpdateMetrics,
    ReadoutGradientMetrics,
    auxiliary_signal_metrics,
    diagnostic_logging_fields,
    finish_parameter_metrics,
    finish_readout_gradient_metrics,
    readout_gradient_tensors,
    snapshot_parameter_metrics,
)
from gz.trainer.losses import (
    _row_mask,
    _torch,
    _trim_to_row_count,
    auxiliary_task_loss,
    lr_at_step,
    policy_ce_loss,
    policy_diagnostic_metrics,
    soft_policy_ce_loss,
    value_head_loss,
)

if TYPE_CHECKING:
    from gz.trainer.loop import TrainerLoop


@dataclass(frozen=True, slots=True)
class StepMetrics:
    step: int
    policy_loss: float
    policy_target_entropy: float
    policy_kl: float
    policy_uniform_advantage: float
    policy_logit_std: float
    policy_logit_abs_mean: float
    soft_policy_loss: float
    soft_policy_kl: float
    soft_policy_target_entropy: float
    value_loss: float
    value_final_loss: float
    value_v8_loss: float
    value_v32_loss: float
    terminal_score_loss: float
    terminal_score_mae: float
    terminal_score_bias: float
    loss: float
    grad_norm: float
    grad_clip_scale: float
    lr: float
    value_accuracy: float
    value_mae: float
    value_rmse: float
    fraction_valid: float
    label_mean: float
    learner_win_rate: float
    auxiliary_signals: AuxiliarySignalMetrics | None
    readout_gradients: ReadoutGradientMetrics | None
    parameter_updates: ParameterUpdateMetrics | None

    def logging_fields(self) -> dict[str, float | int]:
        return diagnostic_logging_fields(
            self.grad_clip_scale,
            self.auxiliary_signals,
            self.readout_gradients,
            self.parameter_updates,
        )


@dataclass(slots=True)
class _StepComputation:
    policy_loss: object
    policy_target_entropy: object
    policy_kl: object
    policy_uniform_advantage: object
    policy_logit_std: object
    policy_logit_abs_mean: object
    soft_policy_loss: object
    soft_policy_kl: object
    soft_policy_target_entropy: object
    value_loss: object
    value_final_loss: object
    value_v8_loss: object
    value_v32_loss: object
    terminal_score_loss: object
    loss: object
    value_prediction: object
    value_labels: object
    value_batch: TrainingBatch
    score_prediction: object | None = None
    horizon_prediction: object | None = None
    readout_gradient_tensors: object | None = None


def execute_training_step(
    loop: TrainerLoop,
    batch: TrainingBatch,
    value_batch: TrainingBatch | None,
    with_metrics: bool,
) -> StepMetrics | None:
    torch = _torch()
    loop.model.train()
    loop.optimizer.zero_grad(set_to_none=True)
    if loop.config.value_weight != 0.0 and value_batch is not None:
        computation = _separate_batch_step(loop, torch, batch, value_batch, with_metrics)
    else:
        computation = _joint_batch_step(loop, torch, batch, with_metrics)

    if with_metrics and loop._auxiliary_layout:
        parameter_snapshot, parameter_totals = snapshot_parameter_metrics(loop.model)
    else:
        parameter_snapshot = None
        parameter_totals = None
    grad_norm = torch.nn.utils.clip_grad_norm_(
        loop.model.parameters(),
        loop.config.grad_clip,
        error_if_nonfinite=True,
    )
    grad_clip_scale = torch.clamp(
        loop.config.grad_clip / (grad_norm.float() + 1.0e-6),
        max=1.0,
    )
    lr = lr_at_step(
        loop.config.lr,
        loop.step_index + 1,
        loop.config.warmup_steps,
        loop.config.lr_decay_steps or loop.config.total_steps,
        loop.config.lr_schedule,
        loop.config.min_lr_ratio,
    )
    for group in loop.optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)
    loop.optimizer.step()
    loop.step_index += 1
    if not with_metrics:
        return None

    parameter_updates = (
        finish_parameter_metrics(
            loop.model,
            parameter_snapshot,
            parameter_totals,
        )
        if parameter_snapshot is not None and parameter_totals is not None
        else None
    )
    readout_gradients = (
        finish_readout_gradient_metrics(computation.readout_gradient_tensors)
        if computation.readout_gradient_tensors is not None
        else None
    )
    return _finish_metrics(
        loop,
        torch,
        computation,
        grad_norm,
        grad_clip_scale,
        lr,
        readout_gradients,
        parameter_updates,
    )


def _separate_batch_step(
    loop: TrainerLoop,
    torch: object,
    policy_batch: TrainingBatch,
    value_batch: TrainingBatch,
    with_metrics: bool,
) -> _StepComputation:
    value_batch = _trim_to_row_count(value_batch)
    zero = policy_batch.value.new_zeros(())
    with _autocast(torch, loop.device_type):
        logits, soft_logits = _policy_forward(loop, policy_batch)
        policy_loss, soft_loss, soft_kl, soft_entropy = _policy_losses(
            loop,
            policy_batch,
            logits,
            soft_logits,
            zero,
        )
        policy_metrics = _policy_diagnostics(
            loop,
            policy_batch,
            logits,
            zero,
            with_metrics,
        )
        policy_objective = policy_loss + loop.config.soft_policy_weight * soft_loss
    policy_objective.backward()
    del logits

    with _autocast(torch, loop.device_type):
        if loop._auxiliary_tasks:
            assert loop._training_values is not None
            value_raw, horizon_raw, score_raw, value_readout = loop._training_values(
                value_batch.features,
                value_trunk_grad_scale=loop.config.value_trunk_grad_scale,
            )
        else:
            value_raw = loop._value_only(
                value_batch.features,
                value_trunk_grad_scale=loop.config.value_trunk_grad_scale,
            )
            horizon_raw = score_raw = value_readout = None
        value_terms = _value_losses(
            loop,
            value_batch,
            value_raw,
            horizon_raw,
            score_raw,
            zero,
        )
    gradients = None
    horizon_prediction = None
    if loop._auxiliary_tasks and with_metrics:
        gradients = readout_gradient_tensors(
            torch,
            value_readout,
            final_loss=value_terms[1],
            v8_loss=value_terms[2],
            v32_loss=value_terms[3],
            terminal_score_loss=value_terms[4],
            policy_loss=None,
            config=loop.config,
        )
        horizon_prediction = horizon_raw.detach()
    (loop.config.value_weight * value_terms[0]).backward()
    loss = policy_objective.detach() + loop.config.value_weight * value_terms[0].detach()
    return _StepComputation(
        policy_loss=policy_loss.detach(),
        policy_target_entropy=policy_metrics[0],
        policy_kl=policy_metrics[1],
        policy_uniform_advantage=policy_metrics[2],
        policy_logit_std=policy_metrics[3],
        policy_logit_abs_mean=policy_metrics[4],
        soft_policy_loss=soft_loss.detach(),
        soft_policy_kl=soft_kl.detach(),
        soft_policy_target_entropy=soft_entropy.detach(),
        value_loss=value_terms[0].detach(),
        value_final_loss=value_terms[1].detach(),
        value_v8_loss=value_terms[2].detach(),
        value_v32_loss=value_terms[3].detach(),
        terminal_score_loss=value_terms[4].detach(),
        loss=loss,
        value_prediction=value_raw.detach(),
        value_labels=value_batch.value,
        value_batch=value_batch,
        score_prediction=(
            value_terms[5].detach() if value_terms[5] is not None else None
        ),
        horizon_prediction=horizon_prediction,
        readout_gradient_tensors=gradients,
    )


def _joint_batch_step(
    loop: TrainerLoop,
    torch: object,
    batch: TrainingBatch,
    with_metrics: bool,
) -> _StepComputation:
    zero = batch.value.new_zeros(())
    with _autocast(torch, loop.device_type):
        if loop.config.value_weight == 0.0:
            logits, soft_logits = _policy_forward(loop, batch)
            value_raw = batch.value.new_zeros(batch.value.shape)
            horizon_raw = score_raw = value_readout = None
            value_terms = (logits.sum() * 0.0, zero, zero, zero, zero, None)
        else:
            (
                value_raw,
                horizon_raw,
                score_raw,
                logits,
                soft_logits,
                value_readout,
            ) = _joint_forward(loop, batch)
            value_terms = _value_losses(
                loop,
                batch,
                value_raw,
                horizon_raw,
                score_raw,
                zero,
            )
        policy_loss, soft_loss, soft_kl, soft_entropy = _policy_losses(
            loop,
            batch,
            logits,
            soft_logits,
            zero,
        )
        policy_metrics = _policy_diagnostics(
            loop,
            batch,
            logits,
            zero,
            with_metrics,
        )
        loss = (
            policy_loss
            + loop.config.soft_policy_weight * soft_loss
            + loop.config.value_weight * value_terms[0]
        )

    gradients = None
    horizon_prediction = None
    if loop._auxiliary_tasks and loop.config.value_weight != 0.0 and with_metrics:
        gradients = readout_gradient_tensors(
            torch,
            value_readout,
            final_loss=value_terms[1],
            v8_loss=value_terms[2],
            v32_loss=value_terms[3],
            terminal_score_loss=value_terms[4],
            policy_loss=policy_loss,
            config=loop.config,
        )
        horizon_prediction = horizon_raw.detach()
    loss.backward()
    return _StepComputation(
        policy_loss=policy_loss.detach(),
        policy_target_entropy=policy_metrics[0],
        policy_kl=policy_metrics[1],
        policy_uniform_advantage=policy_metrics[2],
        policy_logit_std=policy_metrics[3],
        policy_logit_abs_mean=policy_metrics[4],
        soft_policy_loss=soft_loss.detach(),
        soft_policy_kl=soft_kl.detach(),
        soft_policy_target_entropy=soft_entropy.detach(),
        value_loss=value_terms[0].detach(),
        value_final_loss=value_terms[1].detach(),
        value_v8_loss=value_terms[2].detach(),
        value_v32_loss=value_terms[3].detach(),
        terminal_score_loss=value_terms[4].detach(),
        loss=loss.detach(),
        value_prediction=value_raw.detach(),
        value_labels=batch.value,
        value_batch=batch,
        score_prediction=(
            value_terms[5].detach() if value_terms[5] is not None else None
        ),
        horizon_prediction=horizon_prediction,
        readout_gradient_tensors=gradients,
    )


def _policy_forward(
    loop: TrainerLoop,
    batch: TrainingBatch,
) -> tuple[object, object | None]:
    if loop._soft_policy_task:
        assert loop._training_policy_logits is not None
        return loop._training_policy_logits(
            batch.features,
            loop.config.soft_policy_trunk_grad_scale,
        )
    return loop._policy_logits(batch.features), None


def _joint_forward(
    loop: TrainerLoop,
    batch: TrainingBatch,
) -> tuple[object, object | None, object | None, object, object | None, object | None]:
    if loop._soft_policy_task:
        assert loop._training_forward_with_soft_policy is not None
        return loop._training_forward_with_soft_policy(
            batch.features,
            value_trunk_grad_scale=loop.config.value_trunk_grad_scale,
            soft_policy_trunk_grad_scale=loop.config.soft_policy_trunk_grad_scale,
        )
    if loop._auxiliary_tasks:
        assert loop._training_forward is not None
        value_raw, horizon_raw, score_raw, logits, value_readout = loop._training_forward(
            batch.features,
            value_trunk_grad_scale=loop.config.value_trunk_grad_scale,
        )
        return value_raw, horizon_raw, score_raw, logits, None, value_readout
    value_raw, logits = loop._model_forward(
        batch.features,
        value_trunk_grad_scale=loop.config.value_trunk_grad_scale,
    )
    return value_raw, None, None, logits, None, None


def _policy_losses(
    loop: TrainerLoop,
    batch: TrainingBatch,
    logits: object,
    soft_logits: object | None,
    zero: object,
) -> tuple[object, object, object, object]:
    policy_loss = policy_ce_loss(
        logits,
        batch.policy,
        batch.features.action_count,
        batch.row_count,
        action_kind=getattr(batch.features, "action_kind", None),
        mask_stop=loop.config.mask_stop_loss,
    )
    if not loop._soft_policy_task:
        return policy_loss, zero, zero, zero
    assert soft_logits is not None
    soft_loss, soft_kl, soft_entropy = soft_policy_ce_loss(
        soft_logits,
        batch.policy,
        batch.features.action_count,
        batch.row_count,
        temperature=loop.config.soft_policy_temperature,
        action_kind=getattr(batch.features, "action_kind", None),
        mask_stop=loop.config.mask_stop_loss,
    )
    return policy_loss, soft_loss, soft_kl, soft_entropy


def _policy_diagnostics(
    loop: TrainerLoop,
    batch: TrainingBatch,
    logits: object,
    zero: object,
    with_metrics: bool,
) -> tuple[object, object, object, object, object]:
    if not with_metrics:
        return zero, zero, zero, zero, zero
    with _torch().no_grad():
        return policy_diagnostic_metrics(
            logits,
            batch.policy,
            batch.features.action_count,
            batch.row_count,
            action_kind=getattr(batch.features, "action_kind", None),
            mask_stop=loop.config.mask_stop_loss,
        )


def _value_losses(
    loop: TrainerLoop,
    batch: TrainingBatch,
    value_raw: object,
    horizon_raw: object | None,
    score_raw: object | None,
    zero: object,
) -> tuple[object, object, object, object, object, object | None]:
    if loop._auxiliary_tasks:
        assert horizon_raw is not None and score_raw is not None
        return auxiliary_task_loss(
            loop.model,
            value_raw,
            horizon_raw,
            score_raw,
            batch,
            loop.config,
        )
    value_loss = value_head_loss(
        value_raw,
        batch.value,
        batch.value_valid,
        batch.row_count,
    )
    return value_loss, value_loss, zero, zero, zero, None


def _finish_metrics(
    loop: TrainerLoop,
    torch: object,
    computation: _StepComputation,
    grad_norm: object,
    grad_clip_scale: object,
    lr: float,
    readout_gradients: ReadoutGradientMetrics | None,
    parameter_updates: ParameterUpdateMetrics | None,
) -> StepMetrics:
    batch = computation.value_batch
    prediction = computation.value_prediction
    value = computation.value_labels
    with torch.no_grad():
        row_mask = _row_mask(torch, batch.row_count, prediction.shape[0], prediction.device)
        valid = row_mask & (batch.value_valid > 0)
        valid_count = valid.sum()
        if bool(valid_count.item()):
            sign_valid = valid & (value != 0)
            if bool(sign_valid.sum().item()):
                predicted_sign = torch.where(prediction[sign_valid] >= 0, 1.0, -1.0)
                label_sign = torch.where(value[sign_valid] > 0, 1.0, -1.0)
                value_accuracy = (predicted_sign == label_sign).float().mean()
            else:
                value_accuracy = prediction.new_tensor(0.0)
            label_mean = value[valid].mean()
            learner_win_rate = (batch.value[valid] > 0).float().mean()
            value_error = prediction[valid].float() - value[valid].float()
            value_mae = value_error.abs().mean()
            value_rmse = value_error.square().mean().sqrt()
        else:
            value_accuracy = prediction.new_tensor(0.0)
            label_mean = prediction.new_tensor(0.0)
            learner_win_rate = prediction.new_tensor(0.0)
            value_mae = prediction.new_tensor(0.0)
            value_rmse = prediction.new_tensor(0.0)
        fraction_valid = valid.float().mean()
        if computation.score_prediction is not None:
            score_valid = row_mask & (batch.horizon_value_valid > 0)
            terminal_cost = -batch.reward.float()
            score_error = computation.score_prediction.float() - terminal_cost
            score_weight = score_valid.to(score_error.dtype)
            score_count = score_weight.sum().clamp(min=1.0)
            terminal_score_mae = (score_error.abs() * score_weight).sum() / score_count
            terminal_score_bias = (score_error * score_weight).sum() / score_count
        else:
            terminal_score_mae = prediction.new_tensor(0.0)
            terminal_score_bias = prediction.new_tensor(0.0)
        auxiliary_signals = (
            auxiliary_signal_metrics(
                torch,
                batch,
                computation.score_prediction,
            )
            if computation.horizon_prediction is not None
            and computation.score_prediction is not None
            else None
        )
    return StepMetrics(
        step=loop.step_index,
        policy_loss=_scalar(computation.policy_loss),
        policy_target_entropy=_scalar(computation.policy_target_entropy),
        policy_kl=_scalar(computation.policy_kl),
        policy_uniform_advantage=_scalar(computation.policy_uniform_advantage),
        policy_logit_std=_scalar(computation.policy_logit_std),
        policy_logit_abs_mean=_scalar(computation.policy_logit_abs_mean),
        soft_policy_loss=_scalar(computation.soft_policy_loss),
        soft_policy_kl=_scalar(computation.soft_policy_kl),
        soft_policy_target_entropy=_scalar(computation.soft_policy_target_entropy),
        value_loss=_scalar(computation.value_loss),
        value_final_loss=_scalar(computation.value_final_loss),
        value_v8_loss=_scalar(computation.value_v8_loss),
        value_v32_loss=_scalar(computation.value_v32_loss),
        terminal_score_loss=_scalar(computation.terminal_score_loss),
        terminal_score_mae=_scalar(terminal_score_mae),
        terminal_score_bias=_scalar(terminal_score_bias),
        loss=_scalar(computation.loss),
        grad_norm=_scalar(grad_norm),
        grad_clip_scale=_scalar(grad_clip_scale),
        lr=lr,
        value_accuracy=_scalar(value_accuracy),
        value_mae=_scalar(value_mae),
        value_rmse=_scalar(value_rmse),
        fraction_valid=_scalar(fraction_valid),
        label_mean=_scalar(label_mean),
        learner_win_rate=_scalar(learner_win_rate),
        auxiliary_signals=auxiliary_signals,
        readout_gradients=readout_gradients,
        parameter_updates=parameter_updates,
    )


def _autocast(torch: object, device_type: str):
    return torch.autocast(
        device_type=device_type,
        dtype=torch.bfloat16,
        enabled=device_type == "cuda",
    )


def _scalar(tensor: object) -> float:
    return float(tensor.detach().cpu())
