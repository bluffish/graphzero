from __future__ import annotations

import struct
from pathlib import Path

import pytest

from gz.codec import BatchView, FeatureSchemaConfig, TargetsView
from gz.codec.batch import EncodingError
from gz.proto.frames import ENCODING_VERSION
from gz.trainer.data import (
    TrainingStager,
    _validate_terminal_costs,
    _validate_terminal_scores,
)
from python.tests.test_codec import make_batch

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_targets_view_parse_and_zero_copy() -> None:
    raw = bytearray(make_targets())
    view = TargetsView.parse(raw)

    assert view.capacity == 2
    assert view.row_count == 2
    assert view.max_actions == 3
    assert view.policy.tolist() == [[0.75, 0.25, 0.0], [1.0, 0.0, 0.0]]
    assert view.value.tolist() == [1.0, -1.0]
    assert view.value_valid.tolist() == [1, 1]
    assert view.horizon_value.tolist() == [[0.5, 0.25], [-0.5, -0.25]]
    assert view.horizon_value_valid.tolist() == [1, 1]
    assert view.reward.tolist() == [2.5, -3.0]

    struct.pack_into("<f", raw, _layout(2, 3)["policy"], 0.25)
    assert view.policy[0, 0] == pytest.approx(0.25)


def test_targets_view_rejects_bad_headers() -> None:
    valid = make_targets()
    with pytest.raises(EncodingError, match="bad target magic"):
        TargetsView.parse(b"BAD!" + valid[4:])
    with pytest.raises(EncodingError, match="unsupported target version"):
        bad = bytearray(valid)
        struct.pack_into("<I", bad, 4, ENCODING_VERSION + 1)
        TargetsView.parse(bad)
    with pytest.raises(EncodingError, match="zero target capacity"):
        bad = bytearray(valid)
        struct.pack_into("<I", bad, 8, 0)
        TargetsView.parse(bad)
    with pytest.raises(EncodingError, match="bad target length"):
        TargetsView.parse(valid[:-4])


def test_committed_gzft_fixture() -> None:
    view = TargetsView.parse((FIXTURES / "targets.gzft").read_bytes())

    assert view.capacity == 2
    assert view.row_count == 2
    assert view.max_actions == 3
    assert view.policy.tolist() == [[0.75, 0.25, 0.0], [1.0, 0.0, 0.0]]
    assert view.value.tolist() == [1.0, -1.0]
    assert view.value_valid.tolist() == [1, 1]
    assert view.horizon_value.tolist() == [[0.5, 0.25], [-0.5, -0.25]]
    assert view.horizon_value_valid.tolist() == [1, 1]
    assert view.reward.tolist() == [2.5, -3.0]


def test_terminal_score_validation_requires_bounded_integral_node_counts() -> None:
    valid = bytearray(make_targets())
    reward_offset = _layout(2, 3)["reward"]
    _f32(valid, reward_offset, [-2.0, -3.0])
    _validate_terminal_scores(TargetsView.parse(valid), max_nodes=3)

    for reward, message in [
        (-2.5, "integral"),
        (1.0, "bounds"),
        (-4.0, "bounds"),
        (float("nan"), "finite"),
    ]:
        invalid = bytearray(valid)
        struct.pack_into("<f", invalid, reward_offset, reward)
        with pytest.raises(ValueError, match=message):
            _validate_terminal_scores(TargetsView.parse(invalid), max_nodes=3)


def test_terminal_cost_validation_accepts_latency_and_rejects_positive_reward() -> None:
    valid = bytearray(make_targets())
    reward_offset = _layout(2, 3)["reward"]
    _f32(valid, reward_offset, [-5.25, -10_000.0])
    _validate_terminal_costs(TargetsView.parse(valid))

    for reward, message in [
        (1.0, "non-negative"),
        (float("nan"), "finite"),
    ]:
        invalid = bytearray(valid)
        struct.pack_into("<f", invalid, reward_offset, reward)
        with pytest.raises(ValueError, match=message):
            _validate_terminal_costs(TargetsView.parse(invalid))


def test_training_stager_copies_horizon_targets_and_validity() -> None:
    batch = BatchView.parse(make_batch(attr_dim=1))
    targets = TargetsView.parse(make_targets())
    schema = FeatureSchemaConfig(
        name="test",
        node_vocab_size=8,
        node_attr_dim=1,
        edge_type_count=2,
        action_kind_vocab_size=8,
        max_nodes=3,
        max_edges=2,
        max_actions=3,
        max_subjects=2,
    )

    staged = TrainingStager(schema, capacity=2, device="cpu").copy(batch, targets)

    assert staged.horizon_value.tolist() == [[0.5, 0.25], [-0.5, -0.25]]
    assert staged.horizon_value_valid.tolist() == [1.0, 1.0]


def make_targets() -> bytes:
    b, a = 2, 3
    layout = _layout(b, a)
    out = bytearray(layout["total_len"])
    struct.pack_into("<4sIIII", out, 0, b"GZFT", ENCODING_VERSION, b, 2, a)
    _f32(out, layout["policy"], [0.75, 0.25, 0.0, 1.0, 0.0, 0.0])
    _f32(out, layout["value"], [1.0, -1.0])
    out[layout["value_valid"] : layout["value_valid"] + 2] = b"\x01\x01"
    _f32(out, layout["horizon_value"], [0.5, 0.25, -0.5, -0.25])
    out[
        layout["horizon_value_valid"] : layout["horizon_value_valid"] + 2
    ] = b"\x01\x01"
    _f32(out, layout["reward"], [2.5, -3.0])
    return bytes(out)


def _layout(b: int, a: int) -> dict[str, int]:
    cursor = 20
    out = {}
    for name, size in [
        ("policy", b * a * 4),
        ("value", b * 4),
        ("value_valid", b),
        ("horizon_value", b * 2 * 4),
        ("horizon_value_valid", b),
        ("reward", b * 4),
    ]:
        cursor = _align4(cursor)
        out[name] = cursor
        cursor += size
    out["total_len"] = _align4(cursor)
    return out


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _f32(out: bytearray, offset: int, values: list[float]) -> None:
    for index, value in enumerate(values):
        struct.pack_into("<f", out, offset + index * 4, value)
