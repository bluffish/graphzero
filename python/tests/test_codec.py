from __future__ import annotations

import struct

import numpy as np
import pytest

from gz.codec import BatchView, FeatureSchemaConfig, OutputEncoder
from gz.codec.batch import EncodingError
from gz.codec.schema import SchemaConfigError
from gz.proto.frames import BATCH_ENCODING_VERSION

SCHEMA_HASH = b"f" * 32


def test_header_validation_rejects_bad_inputs() -> None:
    valid = make_batch(attr_dim=1)

    with pytest.raises(EncodingError, match="bad batch magic"):
        BatchView.parse(b"BAD!" + valid[4:])
    with pytest.raises(EncodingError, match="unsupported batch version"):
        bad = bytearray(valid)
        struct.pack_into("<I", bad, 4, BATCH_ENCODING_VERSION + 1)
        BatchView.parse(bad)
    with pytest.raises(EncodingError, match="zero max_nodes"):
        bad = bytearray(valid)
        struct.pack_into("<I", bad, 48, 0)
        BatchView.parse(bad)
    with pytest.raises(EncodingError, match="bad batch length"):
        BatchView.parse(valid[:-4])


def test_offset_arithmetic_and_zero_copy_with_attrs() -> None:
    buf = bytearray(make_batch(attr_dim=1))
    view = BatchView.parse(buf)

    assert view.batch_capacity == 2
    assert view.row_count == 2
    assert view.node_count.tolist() == [2, 1]
    assert view.node_tokens.tolist() == [[1, 2, 0], [3, 0, 0]]
    assert view.node_attrs is not None
    assert view.node_attrs[:, :, 0].tolist() == [[0.5, -1.0, 0.0], [2.0, 0.0, 0.0]]
    assert view.edge_count.tolist() == [1, 0]
    assert view.edge_src.tolist() == [[0, 0], [0, 0]]
    assert view.edge_dst.tolist() == [[1, 0], [0, 0]]
    assert view.edge_type.tolist() == [[1, 0], [0, 0]]
    assert view.action_count.tolist() == [2, 1]
    assert view.action_kind.tolist() == [[4, 1, 0], [1, 0, 0]]
    assert view.action_prior.tolist() == [[0.25, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert view.subject_count.tolist() == [[1, 0, 0], [0, 0, 0]]
    assert view.action_subjects[0, 0].tolist() == [1, 0xFFFF]
    assert view.position.tolist() == [[2.0, 3.0, 0.75, 0.125], [1.0, 0.0, 1.0, 0.5]]
    assert view.opponent_reward.tolist() == [0.5, -0.25]
    assert view.opponent_present.tolist() == [1, 1]

    token_offset = _layout(2, 3, 2, 3, 2, 1)["node_tokens"]
    struct.pack_into("<H", buf, token_offset, 6)
    assert view.node_tokens[0, 0] == 6


def test_absent_attr_section_when_attr_dim_is_zero() -> None:
    view = BatchView.parse(make_batch(attr_dim=0))

    assert view.node_attrs is None
    assert view.node_count.tolist() == [2, 1]
    assert view.node_tokens.tolist() == [[1, 2, 0], [3, 0, 0]]


def test_output_encoder_exact_bytes_and_reuse() -> None:
    encoder = OutputEncoder(capacity=2, max_actions=3)
    values = np.array([0.5, -0.25], dtype=np.float32)
    logits = np.array([[1.0, 2.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)

    first = bytes(encoder.encode(values, logits, row_count=2))
    second = bytes(encoder.encode(values, logits, row_count=2))

    expected = bytearray()
    expected.extend(b"GZFO")
    expected.extend(struct.pack("<III", BATCH_ENCODING_VERSION, 2, 3))
    expected.extend(np.array([0.5, -0.25], dtype="<f4").tobytes())
    expected.extend(np.array([1.0, 2.0, 0.0, -1.0, 0.0, 0.0], dtype="<f4").tobytes())
    assert first == bytes(expected)
    assert second == first


def test_feature_schema_config_codec_roundtrip_and_golden() -> None:
    config = FeatureSchemaConfig(
        name="whittle-v1",
        node_vocab_size=7,
        node_attr_dim=0,
        edge_type_count=3,
        action_kind_vocab_size=10,
        max_nodes=64,
        max_edges=448,
        max_actions=256,
        max_subjects=8,
        opponent_reward_scale=256.0,
        expander_degree=5,
        expander_seed=42,
    )
    expected = bytes.fromhex(
        "0a00"
        "77686974746c652d7631"
        "0700"
        "0000"
        "03"
        "0a000000"
        "40000000"
        "c0010000"
        "00010000"
        "08000000"
        "00008043"
        "05"
        "2a00000000000000"
    )

    encoded = config.encode()

    assert encoded == expected
    assert FeatureSchemaConfig.decode(encoded) == config


def test_feature_schema_config_validation() -> None:
    with pytest.raises(SchemaConfigError, match="max_edges too small"):
        FeatureSchemaConfig(
            name="bad",
            node_vocab_size=2,
            node_attr_dim=0,
            edge_type_count=1,
            action_kind_vocab_size=3,
            max_nodes=4,
            max_edges=4,
            max_actions=1,
            max_subjects=1,
            expander_degree=1,
            expander_seed=0,
        )

    with pytest.raises(SchemaConfigError, match="bad schema config length"):
        FeatureSchemaConfig.decode(b"\x00\x00trailing")


def make_batch(attr_dim: int, schema_hash: bytes = SCHEMA_HASH, capacity: int = 2) -> bytes:
    b, n, e, a, s, d = capacity, 3, 2, 3, 2, attr_dim
    layout = _layout(b, n, e, a, s, d)
    out = bytearray(layout["total_len"])
    struct.pack_into("<4sI32sIIIIIII", out, 0, b"GZFB", BATCH_ENCODING_VERSION, schema_hash, b, 2, n, e, a, s, d)
    _fill_subject_padding(out, layout, b, a, s)

    _u32(out, layout["node_count"], [2, 1])
    _u16(out, layout["node_tokens"], [1, 2, 0, 3, 0, 0])
    if d:
        _bf16(out, layout["node_attrs"], [0.5, -1.0, 0.0, 2.0, 0.0, 0.0])
    _u32(out, layout["edge_count"], [1, 0])
    _u16(out, layout["edge_src"], [0, 0, 0, 0])
    _u16(out, layout["edge_dst"], [1, 0, 0, 0])
    out[layout["edge_type"]] = 1
    _u32(out, layout["action_count"], [2, 1])
    _u16(out, layout["action_kind"], [4, 1, 0, 1, 0, 0])
    _bf16(out, layout["action_prior"], [0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
    out[layout["subject_count"]] = 1
    _u16(out, layout["action_subjects"], [1, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF])
    _bf16(out, layout["position"], [2.0, 3.0, 0.75, 0.125, 1.0, 0.0, 1.0, 0.5])
    _bf16(out, layout["opponent_reward"], [0.5, -0.25])
    out[layout["opponent_present"]] = 1
    out[layout["opponent_present"] + 1] = 1
    return bytes(out)


def _layout(b: int, n: int, e: int, a: int, s: int, d: int) -> dict[str, int]:
    cursor = 68
    out = {}
    for name, size in [
        ("node_count", b * 4),
        ("node_tokens", b * n * 2),
        ("node_attrs", b * n * d * 2),
        ("edge_count", b * 4),
        ("edge_src", b * e * 2),
        ("edge_dst", b * e * 2),
        ("edge_type", b * e),
        ("action_count", b * 4),
        ("action_kind", b * a * 2),
        ("action_prior", b * a * 2),
        ("subject_count", b * a),
        ("action_subjects", b * a * s * 2),
        ("position", b * 4 * 2),
        ("opponent_reward", b * 2),
        ("opponent_present", b),
    ]:
        cursor = _align4(cursor)
        out[name] = cursor
        cursor += size
    out["total_len"] = _align4(cursor)
    return out


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _fill_subject_padding(out: bytearray, layout: dict[str, int], b: int, a: int, s: int) -> None:
    start = layout["action_subjects"]
    out[start : start + b * a * s * 2] = b"\xff" * (b * a * s * 2)


def _bf16(out: bytearray, offset: int, values: list[float]) -> None:
    for index, value in enumerate(values):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        rounding = 0x7FFF + ((bits >> 16) & 1)
        struct.pack_into("<H", out, offset + index * 2, ((bits + rounding) >> 16) & 0xFFFF)


def _u16(out: bytearray, offset: int, values: list[int]) -> None:
    for index, value in enumerate(values):
        struct.pack_into("<H", out, offset + index * 2, value)


def _u32(out: bytearray, offset: int, values: list[int]) -> None:
    for index, value in enumerate(values):
        struct.pack_into("<I", out, offset + index * 4, value)


def _f32(out: bytearray, offset: int, values: list[float]) -> None:
    for index, value in enumerate(values):
        struct.pack_into("<f", out, offset + index * 4, value)
