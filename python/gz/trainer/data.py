from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from gz.codec import BatchView, FeatureSchemaConfig, TargetsView
from gz.model.exphormer import BatchStager, GraphBatchTensors


class TrainingBatch(NamedTuple):
    features: GraphBatchTensors
    policy: object
    value: object
    value_valid: object
    horizon_value: object
    horizon_value_valid: object
    reward: object
    row_count: int


class TrainingStager:
    """Double-buffered, event-guarded staging.

    The sync-free fast path lets the CPU enqueue several steps ahead of
    the GPU, and np.copyto into pinned memory is not stream-ordered: a
    single staging buffer gets overwritten before the GPU executes the
    previous batch's H2D copy, silently pairing one batch's features
    with another's targets. Two full staging sets alternate, and each
    set's CUDA event is synchronized before its pinned buffers are
    touched again -- the same hazard the evaluator's ping-pong stagers
    close, with the event standing in for its bounded pipeline.
    """

    def __init__(
        self,
        schema: FeatureSchemaConfig,
        capacity: int,
        device: str | object,
        pinned_staging: bool = True,
        terminal_score_target: str | None = None,
    ) -> None:
        torch = _torch()
        self.schema = schema
        self.capacity = capacity
        self.device = torch.device(device)
        self.terminal_score_target = terminal_score_target
        self._sets = [
            _StagerSet(schema, capacity, self.device, pinned_staging) for _ in range(2)
        ]
        self._index = 0

    def copy(self, batch: BatchView, targets: TargetsView) -> TrainingBatch:
        if batch.batch_capacity != self.capacity or targets.capacity != self.capacity:
            raise ValueError("capacity mismatch")
        if batch.row_count != targets.row_count:
            raise ValueError("row count mismatch")
        if batch.max_actions != targets.max_actions:
            raise ValueError("max action mismatch")
        if self.terminal_score_target == "node-count":
            _validate_terminal_scores(targets, self.schema.max_nodes)
        elif self.terminal_score_target == "log1p-cost-ms":
            _validate_terminal_costs(targets)
        elif self.terminal_score_target is not None:
            raise ValueError(
                f"unknown terminal score target: {self.terminal_score_target}"
            )
        staging = self._sets[self._index]
        self._index = 1 - self._index
        return staging.copy(batch, targets)


class _StagerSet:
    def __init__(self, schema: FeatureSchemaConfig, capacity: int, device: object, pinned_staging: bool) -> None:
        torch = _torch()
        self.features = BatchStager(schema, capacity, device, pinned_staging)
        pin = bool(pinned_staging and device.type == "cuda")
        self.policy = _StagedTensor((capacity, schema.max_actions), torch.float32, device, pin)
        self.value = _StagedTensor((capacity,), torch.float32, device, pin)
        self.value_valid = _StagedTensor((capacity,), torch.float32, device, pin)
        self.horizon_value = _StagedTensor((capacity, 2), torch.float32, device, pin)
        self.horizon_value_valid = _StagedTensor(
            (capacity,), torch.float32, device, pin
        )
        self.reward = _StagedTensor((capacity,), torch.float32, device, pin)
        self.event = torch.cuda.Event() if device.type == "cuda" else None
        self._recorded = False

    def copy(self, batch: BatchView, targets: TargetsView) -> TrainingBatch:
        # The GPU must be done reading this set's pinned buffers (the H2D
        # copies enqueued two steps ago) before the host rewrites them.
        if self.event is not None and self._recorded:
            self.event.synchronize()
        self.policy.copy(targets.policy)
        self.value.copy(targets.value)
        self.value_valid.copy(targets.value_valid.astype(np.float32, copy=False))
        self.horizon_value.copy(targets.horizon_value)
        self.horizon_value_valid.copy(
            targets.horizon_value_valid.astype(np.float32, copy=False)
        )
        self.reward.copy(targets.reward)
        features = self.features.copy(batch)
        if self.event is not None:
            self.event.record()
            self._recorded = True
        return TrainingBatch(
            features=features,
            policy=self.policy.device_tensor,
            value=self.value.device_tensor,
            value_valid=self.value_valid.device_tensor,
            horizon_value=self.horizon_value.device_tensor,
            horizon_value_valid=self.horizon_value_valid.device_tensor,
            reward=self.reward.device_tensor,
            row_count=targets.row_count,
        )


@dataclass(slots=True)
class _StagedTensor:
    cpu: object
    device_tensor: object
    non_blocking: bool

    def __init__(self, shape: tuple[int, ...], dtype: object, device: object, pin: bool) -> None:
        torch = _torch()
        self.cpu = torch.empty(shape, dtype=dtype, pin_memory=pin)
        self.device_tensor = torch.empty(shape, dtype=dtype, device=device)
        self.non_blocking = pin

    def copy(self, array: np.ndarray) -> None:
        np.copyto(self.cpu.numpy(), array, casting="unsafe")
        self.device_tensor.copy_(self.cpu, non_blocking=self.non_blocking)


def _validate_terminal_scores(targets: TargetsView, max_nodes: int) -> None:
    valid = targets.horizon_value_valid[: targets.row_count] > 0
    terminal_nodes = -targets.reward[: targets.row_count][valid]
    if not np.isfinite(terminal_nodes).all():
        raise ValueError("terminal score targets must be finite")
    if not np.equal(terminal_nodes, np.rint(terminal_nodes)).all():
        raise ValueError("terminal score targets must be integral node counts")
    if ((terminal_nodes < 0.0) | (terminal_nodes > float(max_nodes))).any():
        raise ValueError("terminal score targets exceed feature schema bounds")


def _validate_terminal_costs(targets: TargetsView) -> None:
    valid = targets.horizon_value_valid[: targets.row_count] > 0
    terminal_costs = -targets.reward[: targets.row_count][valid]
    if not np.isfinite(terminal_costs).all():
        raise ValueError("terminal cost targets must be finite")
    if (terminal_costs < 0.0).any():
        raise ValueError("terminal cost targets must be non-negative")


def _torch():
    import torch

    return torch
