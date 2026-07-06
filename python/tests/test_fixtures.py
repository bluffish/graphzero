from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from gz.codec import BatchView
from gz.model.stub import stub
from gz.proto.frames import BATCH_ENCODING_VERSION
from test_stub import scalar_stub

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_attr1_fixture_matches_spec_table() -> None:
    raw = (FIXTURES / "batch_attr1.gzfb").read_bytes()
    view = BatchView.parse(raw)

    assert struct.unpack_from("<I", raw, 4)[0] == BATCH_ENCODING_VERSION
    assert view.batch_capacity == 4
    assert view.row_count == 3
    assert view.dims.max_nodes == 8
    assert view.dims.max_edges == 4
    assert view.max_actions == 6
    assert view.dims.max_subjects == 2
    assert view.dims.node_attr_dim == 1

    assert view.node_count.tolist() == [3, 1, 5, 0]
    assert view.node_tokens.tolist() == [
        [1, 2, 3, 0, 0, 0, 0, 0],
        [6, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 4, 5, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert view.node_attrs is not None
    assert view.node_attrs[:, :, 0].tolist() == [
        [0.5, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert view.edge_count.tolist() == [2, 0, 4, 0]
    assert view.edge_src.tolist()[0] == [0, 1, 0, 0]
    assert view.edge_dst.tolist()[0] == [2, 2, 0, 0]
    assert view.edge_type.tolist()[0] == [0, 1, 0, 0]
    assert view.edge_src.tolist()[2] == [0, 1, 2, 3]
    assert view.edge_dst.tolist()[2] == [2, 2, 4, 4]
    assert view.edge_type.tolist()[2] == [0, 1, 0, 1]

    assert view.action_count.tolist() == [2, 1, 6, 0]
    assert view.action_kind.tolist()[0] == [4, 1, 0, 0, 0, 0]
    assert view.action_kind.tolist()[1] == [1, 0, 0, 0, 0, 0]
    assert view.action_kind.tolist()[2] == [2, 3, 4, 5, 6, 1]
    assert view.action_prior.tolist()[0] == [0.25, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert view.action_prior.tolist()[2] == [-0.5, 0.0, 1.0, 0.125, -1.0, 0.0]
    assert view.subject_count.tolist()[0] == [1, 0, 0, 0, 0, 0]
    assert view.subject_count.tolist()[2] == [2, 0, 1, 2, 1, 0]
    assert view.action_subjects[0, 0].tolist() == [2, 0xFFFF]
    assert view.action_subjects[2, 0].tolist() == [0, 1]
    assert view.action_subjects[2, 3].tolist() == [2, 3]
    assert view.position.tolist() == [
        [0.0, 0.0, 1.0, 0.125],
        [1.0, 2.0, 0.75, 0.125],
        [3.0, 1.0, 0.5, 0.25],
        [0.0, 0.0, 0.0, 0.0],
    ]
    assert view.opponent_reward.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert view.opponent_present.tolist() == [0, 0, 0, 0]


def test_attr0_fixture_omits_attrs() -> None:
    view = BatchView.parse((FIXTURES / "batch_attr0.gzfb").read_bytes())

    assert view.batch_capacity == 2
    assert view.row_count == 1
    assert view.dims.node_attr_dim == 0
    assert view.node_attrs is None
    assert view.node_count.tolist() == [2, 0]
    assert view.node_tokens.tolist() == [[1, 2, 0, 0], [0, 0, 0, 0]]
    assert view.edge_count.tolist() == [1, 0]
    assert view.edge_src.tolist()[0] == [0, 0]
    assert view.edge_dst.tolist()[0] == [1, 0]
    assert view.edge_type.tolist()[0] == [1, 0]
    assert view.action_count.tolist() == [2, 0]
    assert view.action_kind.tolist()[0] == [4, 1, 0]
    assert view.action_prior.tolist()[0] == [0.5, 0.0, 0.0]
    assert view.subject_count.tolist()[0] == [1, 0, 0]
    assert view.action_subjects[0, 0].tolist() == [1, 0xFFFF]
    assert view.position.tolist()[0] == [0.0, 1.0, 0.25, 0.25]
    assert view.opponent_reward.tolist() == [0.0, 0.0]
    assert view.opponent_present.tolist() == [0, 0]


def test_expander_fixture_contains_expander_typed_edges() -> None:
    view = BatchView.parse((FIXTURES / "batch_expander.gzfb").read_bytes())

    assert view.batch_capacity == 2
    assert view.row_count == 1
    assert view.dims.node_attr_dim == 0
    assert view.dims.max_edges == 10
    assert view.edge_count.tolist() == [4, 0]
    assert view.edge_src.tolist()[0] == [0, 0, 1, 2, 0, 0, 0, 0, 0, 0]
    assert view.edge_dst.tolist()[0] == [2, 1, 2, 0, 0, 0, 0, 0, 0, 0]
    assert view.edge_type.tolist()[0] == [0, 2, 2, 2, 0, 0, 0, 0, 0, 0]


def test_stub_on_fixture_matches_scalar_reference() -> None:
    view = BatchView.parse((FIXTURES / "batch_attr1.gzfb").read_bytes())
    values, logits = stub(view)
    expected_values, expected_logits = scalar_stub(
        view.node_count.tolist(),
        view.action_count.tolist(),
        view.row_count,
        view.max_actions,
    )

    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(logits, expected_logits)
