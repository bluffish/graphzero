from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gz.codec import BatchView, FeatureSchemaConfig
from gz.model.network import model_class
from gz.model.tensors import GraphBatchTensors, GraphStateTensors


@dataclass(frozen=True, slots=True)
class ArchConfig:
    dim: int = 128
    layers: int = 4
    heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    auxiliary_heads: str = "none"
    policy_head: str = "pointer"

    name: ClassVar[str] = "gz-graph-v2"
    activation: ClassVar[str] = "gelu"
    aggregation: ClassVar[str] = "attention"
    global_tokens: ClassVar[int] = 1
    state_input: ClassVar[str] = "joint-board"
    value_input: ClassVar[str] = "single"
    trunk: ClassVar[str] = "exphormer"
    sage_layers: ClassVar[int] = 3
    value_activation: ClassVar[str] = "tanh"
    subject_encoding: ClassVar[str] = "mean"
    position_encoding: ClassVar[str] = "remaining_budget"
    action_encoding: ClassVar[str] = "kind_prior"
    profile: ClassVar[str] = "graphzero"
    value_hidden: ClassVar[int] = 0
    value_head: ClassVar[str] = "scalar"
    value_bins: ClassVar[int] = 101
    value_min: ClassVar[float] = -1.0
    value_max: ClassVar[float] = 1.0
    value_sigma_ratio: ClassVar[float] = 0.75

    _RUNTIME_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "dim",
            "layers",
            "heads",
            "ffn_dim",
            "dropout",
            "auxiliary_heads",
            "policy_head",
        }
    )
    _FIXED_FIELDS: ClassVar[dict[str, object]] = {
        "name": name,
        "activation": activation,
        "aggregation": aggregation,
        "global_tokens": global_tokens,
        "state_input": state_input,
        "value_input": value_input,
        "trunk": trunk,
        "sage_layers": sage_layers,
        "value_activation": value_activation,
        "subject_encoding": subject_encoding,
        "position_encoding": position_encoding,
        "action_encoding": action_encoding,
        "profile": profile,
        "value_hidden": value_hidden,
        "value_head": value_head,
        "value_bins": value_bins,
        "value_min": value_min,
        "value_max": value_max,
        "value_sigma_ratio": value_sigma_ratio,
    }

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.layers <= 0 or self.heads <= 0 or self.ffn_dim <= 0:
            raise ValueError("arch dimensions must be positive")
        if self.dim % self.heads != 0:
            raise ValueError("dim must be divisible by heads")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout out of range")
        if self.auxiliary_heads not in {
            "none",
            "v8-v32-score",
            "v8-v32-score-soft-policy",
            "v8-v32-score-soft-policy-v2",
        }:
            raise ValueError("unsupported auxiliary_heads")
        if self.policy_head not in {"pointer", "factorized-pointer"}:
            raise ValueError("unsupported policy_head")

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dim": self.dim,
            "layers": self.layers,
            "heads": self.heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "activation": self.activation,
            "aggregation": self.aggregation,
            "global_tokens": self.global_tokens,
            "state_input": self.state_input,
            "value_input": self.value_input,
            "policy_head": self.policy_head,
            "trunk": self.trunk,
            "sage_layers": self.sage_layers,
            "value_activation": self.value_activation,
            "subject_encoding": self.subject_encoding,
            "position_encoding": self.position_encoding,
            "action_encoding": self.action_encoding,
            "profile": self.profile,
            "value_hidden": self.value_hidden,
            "value_head": self.value_head,
            "value_bins": self.value_bins,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "value_sigma_ratio": self.value_sigma_ratio,
            "auxiliary_heads": self.auxiliary_heads,
        }

    def encode(self) -> bytes:
        return json.dumps(
            self.to_manifest_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def hash(self) -> bytes:
        hasher = hashlib.blake2b(digest_size=32)
        _update_chunk(hasher, b"gz-arch-config-v1")
        _update_chunk(hasher, self.encode())
        return hasher.digest()

    @classmethod
    def from_manifest_dict(cls, value: dict[str, object]) -> ArchConfig:
        if set(value) != cls._RUNTIME_FIELDS | cls._FIXED_FIELDS.keys():
            raise ValueError("arch config fields mismatch")
        cls._validate_fixed_fields(value)
        return cls._runtime_from_dict(value)

    @classmethod
    def from_config_dict(cls, value: object) -> ArchConfig:
        if not isinstance(value, dict):
            raise ValueError("[arch] must be a table")
        unknown = set(value) - cls._RUNTIME_FIELDS - cls._FIXED_FIELDS.keys()
        if unknown:
            raise ValueError(f"unknown config fields for ArchConfig: {sorted(unknown)}")
        cls._validate_fixed_fields(value)
        defaults = cls()
        return cls(
            dim=_int(value, "dim", defaults.dim),
            layers=_int(value, "layers", defaults.layers),
            heads=_int(value, "heads", defaults.heads),
            ffn_dim=_int(value, "ffn_dim", defaults.ffn_dim),
            dropout=_float(value, "dropout", defaults.dropout),
            auxiliary_heads=_str(
                value,
                "auxiliary_heads",
                defaults.auxiliary_heads,
            ),
            policy_head=_str(value, "policy_head", defaults.policy_head),
        )

    @classmethod
    def _runtime_from_dict(cls, value: dict[str, object]) -> ArchConfig:
        return cls(
            dim=_int(value, "dim"),
            layers=_int(value, "layers"),
            heads=_int(value, "heads"),
            ffn_dim=_int(value, "ffn_dim"),
            dropout=_float(value, "dropout"),
            auxiliary_heads=_str(value, "auxiliary_heads"),
            policy_head=_str(value, "policy_head"),
        )

    @classmethod
    def _validate_fixed_fields(cls, value: dict[str, object]) -> None:
        for name, expected in cls._FIXED_FIELDS.items():
            if name in value and value[name] != expected:
                raise ValueError(f"{name} must be {expected!r}")


def build_model(schema: FeatureSchemaConfig, arch: ArchConfig):
    return model_class()(schema, arch)


def initialize_policy(model: object, mode: str) -> None:
    if mode == "default":
        return
    if mode != "neutral":
        raise ValueError(f"unsupported policy initializer: {mode}")
    _torch().nn.init.zeros_(model.policy.pointer_key.weight)
    family_key = getattr(model.policy, "family_key", None)
    if family_key is not None:
        _torch().nn.init.zeros_(family_key.weight)


def initialize_value(model: object, mode: str) -> None:
    if mode == "default":
        return
    if mode != "zero":
        raise ValueError(f"unsupported value initializer: {mode}")
    output = model.value[-1]
    for parameter in output.parameters():
        _torch().nn.init.zeros_(parameter)


def tensors_from_batch(view: BatchView, device: str | object, pinned_staging: bool = True) -> GraphBatchTensors:
    return BatchStager.from_view(view, device=device, pinned_staging=pinned_staging).copy(view)


class BatchStager:
    def __init__(
        self,
        schema: FeatureSchemaConfig,
        capacity: int,
        device: str | object,
        pinned_staging: bool = True,
        transfer_stream: object = None,
    ) -> None:
        torch = _torch()
        self.schema = schema
        self.capacity = capacity
        self.device = torch.device(device)
        self.pin = bool(pinned_staging and self.device.type == "cuda")
        if transfer_stream is not None and self.device.type != "cuda":
            raise ValueError("transfer stream requires a CUDA device")
        self.transfer_stream = transfer_stream
        self.ready_event = torch.cuda.Event() if transfer_stream is not None else None
        b = capacity
        n = schema.max_nodes
        e = schema.max_edges
        a = schema.max_actions
        s = schema.max_subjects
        d = schema.node_attr_dim
        index = torch.int32
        self.node_count = _StagedTensor((b,), index, self.device, self.pin, transfer_stream)
        self.node_tokens = _StagedTensor((b, n), index, self.device, self.pin, transfer_stream)
        self.node_attrs = _StagedTensor((b, n, d), torch.float32, self.device, self.pin, transfer_stream)
        self.edge_count = _StagedTensor((b,), index, self.device, self.pin, transfer_stream)
        self.edge_src = _StagedTensor((b, e), index, self.device, self.pin, transfer_stream)
        self.edge_dst = _StagedTensor((b, e), index, self.device, self.pin, transfer_stream)
        self.edge_type = _StagedTensor((b, e), index, self.device, self.pin, transfer_stream)
        self.action_count = _StagedTensor((b,), index, self.device, self.pin, transfer_stream)
        self.action_kind = _StagedTensor((b, a), index, self.device, self.pin, transfer_stream)
        self.action_prior = _StagedTensor((b, a), torch.float32, self.device, self.pin, transfer_stream)
        self.subject_count = _StagedTensor((b, a), index, self.device, self.pin, transfer_stream)
        self.action_subjects = _StagedTensor((b, a, s), index, self.device, self.pin, transfer_stream)
        self.position = _StagedTensor((b, 4), torch.float32, self.device, self.pin, transfer_stream)
        self.opponent_state_present = _StagedTensor((b,), torch.float32, self.device, self.pin, transfer_stream)
        self.opponent_node_count = _StagedTensor((b,), index, self.device, self.pin, transfer_stream)
        self.opponent_node_tokens = _StagedTensor((b, n), index, self.device, self.pin, transfer_stream)
        self.opponent_node_attrs = _StagedTensor((b, n, d), torch.float32, self.device, self.pin, transfer_stream)
        self.opponent_edge_count = _StagedTensor((b,), index, self.device, self.pin, transfer_stream)
        self.opponent_edge_src = _StagedTensor((b, e), index, self.device, self.pin, transfer_stream)
        self.opponent_edge_dst = _StagedTensor((b, e), index, self.device, self.pin, transfer_stream)
        self.opponent_edge_type = _StagedTensor((b, e), index, self.device, self.pin, transfer_stream)
        self.opponent_position = _StagedTensor((b, 4), torch.float32, self.device, self.pin, transfer_stream)

    @classmethod
    def from_view(cls, view: BatchView, device: str | object, pinned_staging: bool = True) -> BatchStager:
        schema = FeatureSchemaConfig(
            name="batch-view",
            node_vocab_size=max(2, int(view.node_tokens.max(initial=0)) + 1),
            node_attr_dim=view.dims.node_attr_dim,
            edge_type_count=max(1, int(view.edge_type.max(initial=0)) + 1),
            action_kind_vocab_size=max(3, int(view.action_kind.max(initial=0)) + 1),
            max_nodes=view.dims.max_nodes,
            max_edges=view.dims.max_edges,
            max_actions=view.dims.max_actions,
            max_subjects=view.dims.max_subjects,
            opponent_reward_scale=256.0,
            expander_degree=0,
            expander_seed=0,
        )
        return cls(schema, view.batch_capacity, device, pinned_staging)

    def copy(self, view: BatchView) -> GraphBatchTensors:
        self._check_view(view)
        self.node_count.copy(view.node_count)
        self.node_tokens.copy(view.node_tokens)
        if view.node_attrs is None:
            self.node_attrs.zero_()
        else:
            self.node_attrs.copy(view.node_attrs)
        self.edge_count.copy(view.edge_count)
        self.edge_src.copy(view.edge_src)
        self.edge_dst.copy(view.edge_dst)
        self.edge_type.copy(view.edge_type)
        self.action_count.copy(view.action_count)
        self.action_kind.copy(view.action_kind)
        self.action_prior.copy(view.action_prior)
        self.subject_count.copy(view.subject_count)
        self.action_subjects.copy(view.action_subjects)
        self.position.copy(view.position)
        self.opponent_state_present.copy(view.opponent_state_present)
        self.opponent_node_count.copy(view.opponent_node_count)
        self.opponent_node_tokens.copy(view.opponent_node_tokens)
        if view.opponent_node_attrs is None:
            self.opponent_node_attrs.zero_()
        else:
            self.opponent_node_attrs.copy(view.opponent_node_attrs)
        self.opponent_edge_count.copy(view.opponent_edge_count)
        self.opponent_edge_src.copy(view.opponent_edge_src)
        self.opponent_edge_dst.copy(view.opponent_edge_dst)
        self.opponent_edge_type.copy(view.opponent_edge_type)
        self.opponent_position.copy(view.opponent_position)
        self._record_ready()
        return self.tensors()

    def dummy(self) -> GraphBatchTensors:
        self.node_count.fill_(1)
        self.node_tokens.zero_()
        self.node_tokens.cpu[..., 0] = 1
        self.node_attrs.zero_()
        self.edge_count.zero_()
        self.edge_src.zero_()
        self.edge_dst.zero_()
        self.edge_type.zero_()
        self.action_count.fill_(1)
        self.action_kind.zero_()
        self.action_kind.cpu[..., 0] = 1
        self.action_prior.zero_()
        self.subject_count.zero_()
        self.action_subjects.fill_(-1)
        self.position.zero_()
        self.opponent_state_present.zero_()
        self.opponent_node_count.zero_()
        self.opponent_node_tokens.zero_()
        self.opponent_node_attrs.zero_()
        self.opponent_edge_count.zero_()
        self.opponent_edge_src.zero_()
        self.opponent_edge_dst.zero_()
        self.opponent_edge_type.zero_()
        self.opponent_position.zero_()
        for tensor in self._all():
            tensor.sync()
        self._record_ready()
        return self.tensors()

    def _record_ready(self) -> None:
        if self.ready_event is not None:
            self.ready_event.record(self.transfer_stream)

    def tensors(self) -> GraphBatchTensors:
        return GraphBatchTensors(
            node_count=self.node_count.device_tensor,
            node_tokens=self.node_tokens.device_tensor,
            node_attrs=self.node_attrs.device_tensor,
            edge_count=self.edge_count.device_tensor,
            edge_src=self.edge_src.device_tensor,
            edge_dst=self.edge_dst.device_tensor,
            edge_type=self.edge_type.device_tensor,
            action_count=self.action_count.device_tensor,
            action_kind=self.action_kind.device_tensor,
            action_prior=self.action_prior.device_tensor,
            subject_count=self.subject_count.device_tensor,
            action_subjects=self.action_subjects.device_tensor,
            position=self.position.device_tensor,
            opponent_state_present=self.opponent_state_present.device_tensor,
            opponent_node_count=self.opponent_node_count.device_tensor,
            opponent_node_tokens=self.opponent_node_tokens.device_tensor,
            opponent_node_attrs=self.opponent_node_attrs.device_tensor,
            opponent_edge_count=self.opponent_edge_count.device_tensor,
            opponent_edge_src=self.opponent_edge_src.device_tensor,
            opponent_edge_dst=self.opponent_edge_dst.device_tensor,
            opponent_edge_type=self.opponent_edge_type.device_tensor,
            opponent_position=self.opponent_position.device_tensor,
        )

    def _check_view(self, view: BatchView) -> None:
        dims = view.dims
        if view.batch_capacity != self.capacity:
            raise ValueError("batch capacity mismatch")
        if dims.max_nodes != self.schema.max_nodes:
            raise ValueError("max_nodes mismatch")
        if dims.max_edges != self.schema.max_edges:
            raise ValueError("max_edges mismatch")
        if dims.max_actions != self.schema.max_actions:
            raise ValueError("max_actions mismatch")
        if dims.max_subjects != self.schema.max_subjects:
            raise ValueError("max_subjects mismatch")
        if dims.node_attr_dim != self.schema.node_attr_dim:
            raise ValueError("node_attr_dim mismatch")

    def _all(self) -> tuple[_StagedTensor, ...]:
        return (
            self.node_count,
            self.node_tokens,
            self.node_attrs,
            self.edge_count,
            self.edge_src,
            self.edge_dst,
            self.edge_type,
            self.action_count,
            self.action_kind,
            self.action_prior,
            self.subject_count,
            self.action_subjects,
            self.position,
            self.opponent_state_present,
            self.opponent_node_count,
            self.opponent_node_tokens,
            self.opponent_node_attrs,
            self.opponent_edge_count,
            self.opponent_edge_src,
            self.opponent_edge_dst,
            self.opponent_edge_type,
            self.opponent_position,
        )


class _StagedTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: object,
        device: object,
        pin: bool,
        transfer_stream: object = None,
    ) -> None:
        torch = _torch()
        self.cpu = torch.empty(shape, dtype=dtype, pin_memory=pin)
        self.device_tensor = torch.empty(shape, dtype=dtype, device=device)
        self.non_blocking = pin
        self.transfer_stream = transfer_stream

    def copy(self, array: np.ndarray) -> None:
        np.copyto(self.cpu.numpy(), array, casting="unsafe")
        self.sync()

    def zero_(self) -> None:
        self.cpu.zero_()
        self.sync()

    def fill_(self, value: int | float) -> None:
        self.cpu.fill_(value)
        self.sync()

    def sync(self) -> None:
        if self.transfer_stream is None:
            self.device_tensor.copy_(self.cpu, non_blocking=self.non_blocking)
            return
        torch = _torch()
        with torch.cuda.stream(self.transfer_stream):
            self.device_tensor.copy_(self.cpu, non_blocking=self.non_blocking)


def _torch():
    return importlib.import_module("torch")


def _update_chunk(hasher: object, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "little"))
    hasher.update(value)


def _int(value: dict[str, object], name: str, default: int | None = None) -> int:
    field = value[name] if default is None else value.get(name, default)
    if not isinstance(field, int):
        raise ValueError(f"{name} must be an integer")
    return field


def _float(value: dict[str, object], name: str, default: float | None = None) -> float:
    field = value[name] if default is None else value.get(name, default)
    if not isinstance(field, (float, int)):
        raise ValueError(f"{name} must be numeric")
    return float(field)


def _str(value: dict[str, object], name: str, default: str | None = None) -> str:
    field = value.get(name, default)
    if not isinstance(field, str):
        raise ValueError(f"{name} must be a string")
    return field
