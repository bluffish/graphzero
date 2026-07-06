from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from gz.codec.schema import SchemaDims
from gz.common.tags import FeatureSchemaHash
from gz.proto.frames import BATCH_ENCODING_VERSION

BATCH_MAGIC = b"GZFB"
BATCH_HEADER_LEN = 68


class EncodingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _Layout:
    b: int
    n: int
    e: int
    a: int
    s: int
    d: int
    node_count: int
    node_tokens: int
    node_attrs: int
    edge_count: int
    edge_src: int
    edge_dst: int
    edge_type: int
    action_count: int
    action_kind: int
    action_prior: int
    subject_count: int
    action_subjects: int
    position: int
    opponent_reward: int
    opponent_present: int
    total_len: int


@dataclass(frozen=True, slots=True)
class BatchView:
    dims: SchemaDims
    node_count: np.ndarray
    node_tokens: np.ndarray
    node_attrs: np.ndarray | None
    edge_count: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_type: np.ndarray
    action_count: np.ndarray
    action_kind: np.ndarray
    action_prior: np.ndarray
    subject_count: np.ndarray
    action_subjects: np.ndarray
    position: np.ndarray
    opponent_reward: np.ndarray
    opponent_present: np.ndarray

    @property
    def feature_schema_hash(self) -> FeatureSchemaHash:
        return self.dims.feature_schema_hash

    @property
    def batch_capacity(self) -> int:
        return self.dims.batch_capacity

    @property
    def row_count(self) -> int:
        return self.dims.row_count

    @property
    def max_actions(self) -> int:
        return self.dims.max_actions

    @classmethod
    def parse(cls, buf: bytes | bytearray | memoryview) -> BatchView:
        view = memoryview(buf)
        if len(view) < BATCH_HEADER_LEN:
            raise EncodingError("batch header truncated")
        if bytes(view[0:4]) != BATCH_MAGIC:
            raise EncodingError("bad batch magic")
        version = _u32(view, 4)
        if version != BATCH_ENCODING_VERSION:
            raise EncodingError("unsupported batch version")
        dims = SchemaDims(
            feature_schema_hash=FeatureSchemaHash.from_bytes(view[8:40]),
            batch_capacity=_u32(view, 40),
            row_count=_u32(view, 44),
            max_nodes=_u32(view, 48),
            max_edges=_u32(view, 52),
            max_actions=_u32(view, 56),
            max_subjects=_u32(view, 60),
            node_attr_dim=_u32(view, 64),
        )
        _validate_dims(dims)
        layout = _layout(
            dims.batch_capacity,
            dims.max_nodes,
            dims.max_edges,
            dims.max_actions,
            dims.max_subjects,
            dims.node_attr_dim,
        )
        if len(view) != layout.total_len:
            raise EncodingError("bad batch length")

        node_attrs = None
        if layout.d != 0:
            node_attrs = _bf16_array(view, layout.node_attrs, (layout.b, layout.n, layout.d))

        return cls(
            dims=dims,
            node_count=_array(view, layout.node_count, "<u4", (layout.b,)),
            node_tokens=_array(view, layout.node_tokens, "<u2", (layout.b, layout.n)),
            node_attrs=node_attrs,
            edge_count=_array(view, layout.edge_count, "<u4", (layout.b,)),
            edge_src=_array(view, layout.edge_src, "<u2", (layout.b, layout.e)),
            edge_dst=_array(view, layout.edge_dst, "<u2", (layout.b, layout.e)),
            edge_type=_array(view, layout.edge_type, "u1", (layout.b, layout.e)),
            action_count=_array(view, layout.action_count, "<u4", (layout.b,)),
            action_kind=_array(view, layout.action_kind, "<u2", (layout.b, layout.a)),
            action_prior=_bf16_array(view, layout.action_prior, (layout.b, layout.a)),
            subject_count=_array(view, layout.subject_count, "u1", (layout.b, layout.a)),
            action_subjects=_array(
                view,
                layout.action_subjects,
                "<u2",
                (layout.b, layout.a, layout.s),
            ),
            position=_bf16_array(view, layout.position, (layout.b, 4)),
            opponent_reward=_bf16_array(view, layout.opponent_reward, (layout.b,)),
            opponent_present=_array(view, layout.opponent_present, "u1", (layout.b,)),
        )


def _validate_dims(dims: SchemaDims) -> None:
    if dims.batch_capacity <= 0:
        raise EncodingError("zero batch capacity")
    if dims.row_count > dims.batch_capacity:
        raise EncodingError("row count exceeds capacity")
    if dims.max_nodes <= 0:
        raise EncodingError("zero max_nodes")
    if dims.max_edges <= 0:
        raise EncodingError("zero max_edges")
    if dims.max_actions <= 0:
        raise EncodingError("zero max_actions")
    if dims.max_subjects <= 0:
        raise EncodingError("zero max_subjects")
    if dims.node_attr_dim < 0:
        raise EncodingError("negative node_attr_dim")


def _layout(b: int, n: int, e: int, a: int, s: int, d: int) -> _Layout:
    cursor = BATCH_HEADER_LEN
    node_count, cursor = _section(cursor, b * 4)
    node_tokens, cursor = _section(cursor, b * n * 2)
    node_attrs, cursor = _section(cursor, b * n * d * 2)
    edge_count, cursor = _section(cursor, b * 4)
    edge_src, cursor = _section(cursor, b * e * 2)
    edge_dst, cursor = _section(cursor, b * e * 2)
    edge_type, cursor = _section(cursor, b * e)
    action_count, cursor = _section(cursor, b * 4)
    action_kind, cursor = _section(cursor, b * a * 2)
    action_prior, cursor = _section(cursor, b * a * 2)
    subject_count, cursor = _section(cursor, b * a)
    action_subjects, cursor = _section(cursor, b * a * s * 2)
    position, cursor = _section(cursor, b * 4 * 2)
    opponent_reward, cursor = _section(cursor, b * 2)
    opponent_present, cursor = _section(cursor, b)
    return _Layout(
        b=b,
        n=n,
        e=e,
        a=a,
        s=s,
        d=d,
        node_count=node_count,
        node_tokens=node_tokens,
        node_attrs=node_attrs,
        edge_count=edge_count,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_type=edge_type,
        action_count=action_count,
        action_kind=action_kind,
        action_prior=action_prior,
        subject_count=subject_count,
        action_subjects=action_subjects,
        position=position,
        opponent_reward=opponent_reward,
        opponent_present=opponent_present,
        total_len=_align4(cursor),
    )


def _section(cursor: int, length: int) -> tuple[int, int]:
    offset = _align4(cursor)
    return offset, offset + length


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _u32(buf: memoryview, offset: int) -> int:
    return struct.unpack_from("<I", buf, offset)[0]


def _bf16_array(buf: memoryview, offset: int, shape: tuple[int, ...]) -> np.ndarray:
    """Widens wire bfloat16 into a float32 copy (numpy has no bf16), so
    downstream consumers keep seeing float32 arrays."""
    raw = _array(buf, offset, "<u2", shape)
    return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _array(buf: memoryview, offset: int, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    return np.frombuffer(buf, dtype=np.dtype(dtype), count=count, offset=offset).reshape(shape)
