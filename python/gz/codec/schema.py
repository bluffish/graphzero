from __future__ import annotations

import struct
from dataclasses import dataclass
from math import isfinite

from gz.common.tags import FeatureSchemaHash


class SchemaConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FeatureSchemaConfig:
    name: str
    node_vocab_size: int
    node_attr_dim: int
    edge_type_count: int
    action_kind_vocab_size: int
    max_nodes: int
    max_edges: int
    max_actions: int
    max_subjects: int
    opponent_reward_scale: float = 256.0
    expander_degree: int = 0
    expander_seed: int = 0

    def __post_init__(self) -> None:
        _validate_schema_config(self)

    def encode(self) -> bytes:
        name = self.name.encode("utf-8")
        if len(name) > 0xFFFF:
            raise SchemaConfigError("name too long")
        return (
            struct.pack("<H", len(name))
            + name
            + struct.pack(
                "<HHBIIIIIfBQ",
                self.node_vocab_size,
                self.node_attr_dim,
                self.edge_type_count,
                self.action_kind_vocab_size,
                self.max_nodes,
                self.max_edges,
                self.max_actions,
                self.max_subjects,
                self.opponent_reward_scale,
                self.expander_degree,
                self.expander_seed,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "node_vocab_size": self.node_vocab_size,
            "node_attr_dim": self.node_attr_dim,
            "edge_type_count": self.edge_type_count,
            "action_kind_vocab_size": self.action_kind_vocab_size,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "max_actions": self.max_actions,
            "max_subjects": self.max_subjects,
            "opponent_reward_scale": self.opponent_reward_scale,
            "expander_degree": self.expander_degree,
            "expander_seed": self.expander_seed,
        }

    @classmethod
    def from_dict(cls, value: object) -> FeatureSchemaConfig:
        if not isinstance(value, dict):
            raise SchemaConfigError("feature_schema must be an object")
        fields = {
            "name",
            "node_vocab_size",
            "node_attr_dim",
            "edge_type_count",
            "action_kind_vocab_size",
            "max_nodes",
            "max_edges",
            "max_actions",
            "max_subjects",
            "opponent_reward_scale",
            "expander_degree",
            "expander_seed",
        }
        optional = {"opponent_reward_scale"}
        keys = set(value)
        if keys != fields and keys != fields - optional:
            raise SchemaConfigError("feature_schema fields mismatch")
        return cls(
            name=_str_field(value, "name"),
            node_vocab_size=_int_field(value, "node_vocab_size"),
            node_attr_dim=_int_field(value, "node_attr_dim"),
            edge_type_count=_int_field(value, "edge_type_count"),
            action_kind_vocab_size=_int_field(value, "action_kind_vocab_size"),
            max_nodes=_int_field(value, "max_nodes"),
            max_edges=_int_field(value, "max_edges"),
            max_actions=_int_field(value, "max_actions"),
            max_subjects=_int_field(value, "max_subjects"),
            opponent_reward_scale=_float_field(value, "opponent_reward_scale", 256.0),
            expander_degree=_int_field(value, "expander_degree"),
            expander_seed=_int_field(value, "expander_seed"),
        )

    @classmethod
    def decode(cls, buf: bytes | bytearray | memoryview) -> FeatureSchemaConfig:
        view = memoryview(buf)
        if len(view) < 2:
            raise SchemaConfigError("schema config truncated")
        name_len = struct.unpack_from("<H", view, 0)[0]
        cursor = 2
        end = cursor + name_len
        if len(view) < end:
            raise SchemaConfigError("schema name truncated")
        try:
            name = bytes(view[cursor:end]).decode("utf-8")
        except UnicodeDecodeError as error:
            raise SchemaConfigError("invalid schema name utf8") from error
        cursor = end
        tail = struct.calcsize("<HHBIIIIIfBQ")
        if len(view) != cursor + tail:
            raise SchemaConfigError("bad schema config length")
        (
            node_vocab_size,
            node_attr_dim,
            edge_type_count,
            action_kind_vocab_size,
            max_nodes,
            max_edges,
            max_actions,
            max_subjects,
            opponent_reward_scale,
            expander_degree,
            expander_seed,
        ) = struct.unpack_from("<HHBIIIIIfBQ", view, cursor)
        return cls(
            name=name,
            node_vocab_size=node_vocab_size,
            node_attr_dim=node_attr_dim,
            edge_type_count=edge_type_count,
            action_kind_vocab_size=action_kind_vocab_size,
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_actions=max_actions,
            max_subjects=max_subjects,
            opponent_reward_scale=opponent_reward_scale,
            expander_degree=expander_degree,
            expander_seed=expander_seed,
        )


@dataclass(frozen=True, slots=True)
class SchemaDims:
    feature_schema_hash: FeatureSchemaHash
    batch_capacity: int
    row_count: int
    max_nodes: int
    max_edges: int
    max_actions: int
    max_subjects: int
    node_attr_dim: int


def _validate_schema_config(config: FeatureSchemaConfig) -> None:
    if not config.name:
        raise SchemaConfigError("name must be non-empty")
    _check_range("node_vocab_size", config.node_vocab_size, 2, 0xFFFF)
    _check_range("node_attr_dim", config.node_attr_dim, 0, 0xFFFF)
    _check_range("edge_type_count", config.edge_type_count, 0, 0xFF)
    _check_range("action_kind_vocab_size", config.action_kind_vocab_size, 3, 0xFFFF_FFFF)
    _check_range("max_nodes", config.max_nodes, 1, 0xFFFF_FFFF)
    _check_range("max_edges", config.max_edges, 1, 0xFFFF_FFFF)
    _check_range("max_actions", config.max_actions, 1, 0xFFFF_FFFF)
    _check_range("max_subjects", config.max_subjects, 1, 0xFFFF_FFFF)
    if not isinstance(config.opponent_reward_scale, (float, int)):
        raise SchemaConfigError("opponent_reward_scale must be numeric")
    if not isfinite(config.opponent_reward_scale) or config.opponent_reward_scale <= 0.0:
        raise SchemaConfigError("opponent_reward_scale must be finite and positive")
    _check_range("expander_degree", config.expander_degree, 0, 0xFF)
    _check_range("expander_seed", config.expander_seed, 0, 0xFFFF_FFFF_FFFF_FFFF)
    if config.expander_degree > 0:
        if config.edge_type_count == 0:
            raise SchemaConfigError("edge_type_count must include expander type")
        required_edges = config.expander_degree * config.max_nodes + 1
        if config.max_edges < required_edges:
            raise SchemaConfigError("max_edges too small for expander_degree")


def _check_range(name: str, value: int, low: int, high: int) -> None:
    if not isinstance(value, int):
        raise SchemaConfigError(f"{name} must be an integer")
    if value < low or value > high:
        raise SchemaConfigError(f"{name} out of range")


def _int_field(value: dict[str, object], name: str) -> int:
    field = value[name]
    if not isinstance(field, int):
        raise SchemaConfigError(f"{name} must be an integer")
    return field


def _float_field(value: dict[str, object], name: str, default: float) -> float:
    field = value.get(name, default)
    if not isinstance(field, (float, int)):
        raise SchemaConfigError(f"{name} must be numeric")
    return float(field)


def _str_field(value: dict[str, object], name: str) -> str:
    field = value[name]
    if not isinstance(field, str):
        raise SchemaConfigError(f"{name} must be a string")
    return field
