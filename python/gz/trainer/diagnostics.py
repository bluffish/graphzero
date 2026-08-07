"""Logging-step diagnostics for auxiliary value tasks.

Readout attribution stops at the shared fanout so it does not repeat the graph
trunk backward. Parameter snapshots bracket the optimizer step and therefore
measure the exact combined update. Callers keep both operations off the
non-logging path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from gz.trainer.data import TrainingBatch


@dataclass(frozen=True, slots=True)
class AuxiliarySignalMetrics:
    v8_final_target_correlation: float
    v32_final_target_correlation: float
    v8_v32_target_correlation: float
    terminal_score_correlation: float
    early_v8_final_target_correlation: float
    early_v32_final_target_correlation: float
    early_v8_target_std: float
    early_v32_target_std: float


@dataclass(frozen=True, slots=True)
class ReadoutGradientMetrics:
    effective_auxiliary_norm: float
    auxiliary_to_final_norm_ratio: float
    auxiliary_alignment_ratio: float
    final_auxiliary_cosine: float
    policy_auxiliary_cosine: float | None = None


@dataclass(frozen=True, slots=True)
class ParameterUpdateMetrics:
    trunk_gradient_norm: float
    trunk_update_to_parameter: float
    value_final_update_to_parameter: float
    value_horizons_update_to_parameter: float
    terminal_score_update_to_parameter: float


def diagnostic_logging_fields(
    grad_clip_scale: float,
    auxiliary_signals: AuxiliarySignalMetrics | None,
    readout_gradients: ReadoutGradientMetrics | None,
    parameter_updates: ParameterUpdateMetrics | None,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {"grad_clip_scale": grad_clip_scale}
    for prefix, metrics in (
        ("aux_signal", auxiliary_signals),
        ("aux_gradient", readout_gradients),
        ("parameter", parameter_updates),
    ):
        if metrics is None:
            continue
        result.update(
            {
                f"{prefix}_{field}": value
                for field, value in asdict(metrics).items()
                if value is not None
            }
        )
    return result


def readout_gradient_tensors(
    torch: object,
    readout: object,
    *,
    final_loss: object,
    v8_loss: object,
    v32_loss: object,
    terminal_score_loss: object,
    policy_loss: object | None,
    config: object,
) -> dict[str, object]:
    """Build the small set of device-resident task-interaction scalars."""

    def task_gradient(loss: object) -> object:
        gradient = torch.autograd.grad(
            loss,
            readout,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is None:
            gradient = torch.zeros_like(readout)
        return gradient.detach().float()

    final = task_gradient(final_loss)
    v8 = task_gradient(v8_loss)
    v32 = task_gradient(v32_loss)
    terminal_score = task_gradient(terminal_score_loss)

    effective_final = final * (config.value_weight * config.value_final_weight)
    effective_v8 = v8 * (config.value_weight * config.value_v8_weight)
    effective_v32 = v32 * (config.value_weight * config.value_v32_weight)
    effective_terminal_score = terminal_score * (
        config.value_weight * config.terminal_score_weight
    )
    effective_auxiliary = effective_v8 + effective_v32 + effective_terminal_score
    effective_auxiliary_norm = _vector_norm(torch, effective_auxiliary)
    effective_final_norm = _vector_norm(torch, effective_final)
    auxiliary_norm_sum = (
        _vector_norm(torch, effective_v8)
        + _vector_norm(torch, effective_v32)
        + _vector_norm(torch, effective_terminal_score)
    )

    tensors = {
        "effective_auxiliary_norm": effective_auxiliary_norm,
        "auxiliary_to_final_norm_ratio": _safe_ratio(
            torch, effective_auxiliary_norm, effective_final_norm
        ),
        "auxiliary_alignment_ratio": _safe_ratio(
            torch, effective_auxiliary_norm, auxiliary_norm_sum
        ),
        "final_auxiliary_cosine": _cosine(torch, effective_final, effective_auxiliary),
    }
    if policy_loss is not None:
        tensors["policy_auxiliary_cosine"] = _cosine(
            torch, task_gradient(policy_loss), effective_auxiliary
        )
    return tensors


def finish_readout_gradient_metrics(
    tensors: dict[str, object],
) -> ReadoutGradientMetrics:
    return ReadoutGradientMetrics(**_host_scalar_dict(tensors))


_PARAMETER_METRIC_GROUPS = (
    "trunk",
    "value_final",
    "value_horizons",
    "terminal_score",
)


def _parameter_metric_group(parameter_name: str) -> str | None:
    root = parameter_name.split(".", 1)[0]
    if root == "value":
        return "value_final"
    if root == "horizon_value":
        return "value_horizons"
    if root == "terminal_score":
        return "terminal_score"
    if root in {"kind_embedding", "match", "policy", "stop"}:
        return None
    return "trunk"


def snapshot_parameter_metrics(
    model: object,
) -> tuple[dict[str, object], dict[str, dict[str, object | None]]]:
    """Capture only parameters needed for the retained update diagnostics."""
    snapshot = {}
    totals: dict[str, dict[str, object | None]] = {
        group: {"parameter_squared": None, "gradient_squared": None}
        for group in _PARAMETER_METRIC_GROUPS
    }
    for name, parameter in model.named_parameters():
        group = _parameter_metric_group(name)
        if group is None or not parameter.requires_grad:
            continue
        before = parameter.detach().clone()
        snapshot[name] = before
        totals[group]["parameter_squared"] = _sum_scalar_tensor(
            totals[group]["parameter_squared"], before.float().square().sum()
        )
        if group == "trunk" and parameter.grad is not None:
            totals[group]["gradient_squared"] = _sum_scalar_tensor(
                totals[group]["gradient_squared"],
                parameter.grad.detach().float().square().sum(),
            )
    return snapshot, totals


def finish_parameter_metrics(
    model: object,
    snapshot: dict[str, object],
    totals: dict[str, dict[str, object | None]],
) -> ParameterUpdateMetrics:
    """Compare the live post-step model with its exact pre-step snapshot."""
    torch = _torch()
    update_squared: dict[str, object | None] = {
        group: None for group in _PARAMETER_METRIC_GROUPS
    }
    for name, parameter in model.named_parameters():
        before = snapshot.get(name)
        if before is None:
            continue
        group = _parameter_metric_group(name)
        assert group is not None
        update_squared[group] = _sum_scalar_tensor(
            update_squared[group],
            (parameter.detach().float() - before.float()).square().sum(),
        )

    first = next(iter(snapshot.values()))
    zero = first.float().new_zeros(())
    parameter_norms = {
        group: (
            totals[group]["parameter_squared"]
            if totals[group]["parameter_squared"] is not None
            else zero
        ).sqrt()
        for group in _PARAMETER_METRIC_GROUPS
    }
    update_norms = {
        group: (
            update_squared[group] if update_squared[group] is not None else zero
        ).sqrt()
        for group in _PARAMETER_METRIC_GROUPS
    }
    trunk_gradient_squared = totals["trunk"]["gradient_squared"]
    tensors = {
        "trunk_gradient_norm": (
            trunk_gradient_squared if trunk_gradient_squared is not None else zero
        ).sqrt(),
        **{
            f"{group}_update_to_parameter": _safe_ratio(
                torch, update_norms[group], parameter_norms[group]
            )
            for group in _PARAMETER_METRIC_GROUPS
        },
    }
    return ParameterUpdateMetrics(**_host_scalar_dict(tensors))


def auxiliary_signal_metrics(
    torch: object,
    batch: TrainingBatch,
    score_prediction: object,
) -> AuxiliarySignalMetrics:
    """Summarize target novelty, early-state noise, and score learnability."""
    row_mask = _row_mask(
        torch,
        batch.row_count,
        batch.value.shape[0],
        batch.value.device,
    )
    final_valid = row_mask & (batch.value_valid > 0)
    auxiliary_valid = row_mask & (batch.horizon_value_valid > 0)
    shared_valid = final_valid & auxiliary_valid

    position = getattr(batch.features, "position", None)
    if position is None:
        progress = batch.value.new_zeros(batch.value.shape)
    else:
        progress = (1.0 - position[:, 2].float()).clamp(0.0, 1.0)
    early_auxiliary_valid = auxiliary_valid & (progress < (1.0 / 3.0))
    early_shared_valid = final_valid & early_auxiliary_valid

    v8_final = _masked_pair_stats(
        torch, batch.horizon_value[:, 0], batch.value, shared_valid
    )
    v32_final = _masked_pair_stats(
        torch, batch.horizon_value[:, 1], batch.value, shared_valid
    )
    v8_v32 = _masked_pair_stats(
        torch,
        batch.horizon_value[:, 0],
        batch.horizon_value[:, 1],
        auxiliary_valid,
    )
    score = _masked_pair_stats(
        torch,
        score_prediction,
        -batch.reward,
        auxiliary_valid,
    )
    early_v8_final = _masked_pair_stats(
        torch, batch.horizon_value[:, 0], batch.value, early_shared_valid
    )
    early_v32_final = _masked_pair_stats(
        torch, batch.horizon_value[:, 1], batch.value, early_shared_valid
    )

    return AuxiliarySignalMetrics(
        **_host_scalar_dict(
            {
                "v8_final_target_correlation": v8_final["correlation"],
                "v32_final_target_correlation": v32_final["correlation"],
                "v8_v32_target_correlation": v8_v32["correlation"],
                "terminal_score_correlation": score["correlation"],
                "early_v8_final_target_correlation": early_v8_final["correlation"],
                "early_v32_final_target_correlation": early_v32_final["correlation"],
                "early_v8_target_std": _masked_standard_deviation(
                    torch, batch.horizon_value[:, 0], early_auxiliary_valid
                ),
                "early_v32_target_std": _masked_standard_deviation(
                    torch, batch.horizon_value[:, 1], early_auxiliary_valid
                ),
            }
        )
    )


def _masked_pair_stats(
    torch: object,
    left: object,
    right: object,
    mask: object,
) -> dict[str, object]:
    left = left.float()
    right = right.float()
    weight = mask.to(left.dtype)
    count = weight.sum().clamp(min=1.0)
    safe_left = torch.where(mask, left, torch.zeros_like(left))
    safe_right = torch.where(mask, right, torch.zeros_like(right))
    left_mean = safe_left.sum() / count
    right_mean = safe_right.sum() / count
    centered_left = torch.where(mask, left - left_mean, torch.zeros_like(left))
    centered_right = torch.where(mask, right - right_mean, torch.zeros_like(right))
    left_variance = centered_left.square().sum() / count
    right_variance = centered_right.square().sum() / count
    covariance = (centered_left * centered_right).sum() / count
    return {
        "correlation": _safe_ratio(
            torch,
            covariance,
            (left_variance * right_variance).sqrt(),
        )
    }


def _masked_standard_deviation(torch: object, value: object, mask: object) -> object:
    value = value.float()
    count = mask.to(value.dtype).sum().clamp(min=1.0)
    safe_value = torch.where(mask, value, torch.zeros_like(value))
    mean = safe_value.sum() / count
    centered = torch.where(mask, value - mean, torch.zeros_like(value))
    return (centered.square().sum() / count).sqrt()


def _vector_norm(torch: object, value: object) -> object:
    return torch.linalg.vector_norm(value.reshape(-1))


def _cosine(torch: object, left: object, right: object) -> object:
    left = left.reshape(-1)
    right = right.reshape(-1)
    denominator = _vector_norm(torch, left) * _vector_norm(torch, right)
    return _safe_ratio(torch, torch.dot(left, right), denominator)


def _safe_ratio(torch: object, numerator: object, denominator: object) -> object:
    nonzero = denominator.abs() > 0.0
    safe_denominator = torch.where(nonzero, denominator, torch.ones_like(denominator))
    return torch.where(
        nonzero,
        numerator / safe_denominator,
        numerator.new_zeros(()),
    )


def _sum_scalar_tensor(current: object | None, value: object) -> object:
    return value if current is None else current + value


def _host_scalar_dict(tensors: dict[str, object]) -> dict[str, float]:
    if not tensors:
        return {}
    names = list(tensors)
    values = _torch().stack([tensors[name].detach().float() for name in names])
    return dict(zip(names, values.cpu().tolist(), strict=True))


def _row_mask(torch: object, row_count: int, capacity: int, device: object) -> object:
    return torch.arange(capacity, device=device) < row_count


def _torch():
    import torch

    return torch
