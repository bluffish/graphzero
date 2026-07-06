from __future__ import annotations

import struct

import numpy as np

from gz.codec import BatchView
from gz.common import FeatureSchemaHash
from gz.model import build
from gz.model.stub import STUB_MODEL_VERSION, stub
from gz.proto.frames import BATCH_ENCODING_VERSION
from python.tests.test_codec import make_batch


def test_registry_builds_stub() -> None:
    batch = BatchView.parse(make_batch(attr_dim=1))
    model = build("stub", batch.dims, {})

    values, logits = model(batch)

    assert values.shape == (2,)
    assert logits.shape == (2, 3)


def test_stub_matches_scalar_reference_on_batch() -> None:
    batch = BatchView.parse(make_batch(attr_dim=1))
    values, logits = stub(batch)
    expected_values, expected_logits = scalar_stub(
        batch.node_count.tolist(),
        batch.action_count.tolist(),
        batch.row_count,
        batch.max_actions,
    )

    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(logits, expected_logits)
    assert bytes(STUB_MODEL_VERSION) == b"gz-stub-v1" + b"\x00" * 6


def test_stub_matches_scalar_reference_for_seeded_counts() -> None:
    rng = np.random.default_rng(7)
    node_counts = rng.integers(1, 8, size=4, dtype=np.uint32).tolist()
    action_counts = rng.integers(1, 6, size=4, dtype=np.uint32).tolist()
    batch = count_batch(node_counts, action_counts, row_count=3, max_actions=6)

    values, logits = stub(batch)
    expected_values, expected_logits = scalar_stub(node_counts, action_counts, 3, 6)

    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(logits, expected_logits)


def scalar_stub(
    node_counts: list[int],
    action_counts: list[int],
    row_count: int,
    max_actions: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(node_counts), dtype=np.float32)
    logits = np.zeros((len(node_counts), max_actions), dtype=np.float32)
    for row, (nodes, actions) in enumerate(zip(node_counts, action_counts, strict=True)):
        if row >= row_count:
            continue
        values[row] = np.float32((((nodes * 2_654_435_761 + actions * 40_503) % 4096) - 2048) / 2048.0)
        for action in range(actions):
            logits[row, action] = np.float32((((nodes + 31 * action + 7 * actions) % 64) - 32) / 32.0)
    return values, logits


def count_batch(node_counts: list[int], action_counts: list[int], row_count: int, max_actions: int) -> BatchView:
    capacity = len(node_counts)
    max_nodes = max(node_counts)
    max_edges = 1
    max_subjects = 1
    total_len = 68
    sections = []
    for size in [
        capacity * 4,
        capacity * max_nodes * 2,
        0,
        capacity * 4,
        capacity * max_edges * 2,
        capacity * max_edges * 2,
        capacity * max_edges,
        capacity * 4,
        capacity * max_actions * 2,
        capacity * max_actions * 2,
        capacity * max_actions,
        capacity * max_actions * max_subjects * 2,
        capacity * 8,
        capacity * 2,
        capacity,
    ]:
        total_len = (total_len + 3) & ~3
        sections.append(total_len)
        total_len += size
    total_len = (total_len + 3) & ~3

    out = bytearray(total_len)
    struct.pack_into(
        "<4sI32sIIIIIII",
        out,
        0,
        b"GZFB",
        BATCH_ENCODING_VERSION,
        bytes(FeatureSchemaHash.from_bytes(b"r" * 32)),
        capacity,
        row_count,
        max_nodes,
        max_edges,
        max_actions,
        max_subjects,
        0,
    )
    out[sections[11] : sections[11] + capacity * max_actions * max_subjects * 2] = b"\xff" * (
        capacity * max_actions * max_subjects * 2
    )
    for index, value in enumerate(node_counts):
        struct.pack_into("<I", out, sections[0] + index * 4, value)
    for index, value in enumerate(action_counts):
        struct.pack_into("<I", out, sections[7] + index * 4, value)
    return BatchView.parse(out)
