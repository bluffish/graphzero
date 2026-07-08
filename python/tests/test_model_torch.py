from __future__ import annotations

import struct
from pathlib import Path

import pytest

from gz.codec import BatchView, FeatureSchemaConfig
from gz.model.exphormer import ArchConfig, BatchStager, build_model
from python.tests.test_codec import _bf16, _layout, _u16, make_batch

torch = pytest.importorskip("torch")

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize("aggregation", ["attention", "gine", "sage", "match"])
def test_padding_invariance(aggregation: str) -> None:
    small = BatchView.parse(make_batch(attr_dim=1, capacity=2))
    padded = BatchView.parse(make_batch(attr_dim=1, capacity=3))
    schema = schema_for_view(padded, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = make_arch(aggregation)
    model = build_model(schema, arch).eval()

    small_values, small_logits = run_model(model, schema, small)
    padded_values, padded_logits = run_model(model, schema, padded)

    torch.testing.assert_close(padded_values[:2], small_values, rtol=0, atol=1e-7)
    torch.testing.assert_close(padded_logits[:2], small_logits, rtol=0, atol=1e-7)


@pytest.mark.parametrize("aggregation", ["attention", "gine", "sage", "match"])
def test_batch_independence(aggregation: str) -> None:
    original = BatchView.parse(make_batch(attr_dim=1))
    mutated_bytes = bytearray(make_batch(attr_dim=1))
    layout = _layout(2, 3, 2, 3, 2, 1)
    struct.pack_into("<I", mutated_bytes, layout["node_count"] + 4, 3)
    struct.pack_into("<H", mutated_bytes, layout["node_tokens"] + 3 * 2, 6)
    struct.pack_into("<H", mutated_bytes, layout["node_tokens"] + 4 * 2, 5)
    struct.pack_into("<H", mutated_bytes, layout["node_tokens"] + 5 * 2, 4)
    mutated = BatchView.parse(mutated_bytes)
    schema = schema_for_view(original, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = make_arch(aggregation)
    model = build_model(schema, arch).eval()

    values, logits = run_model(model, schema, original)
    mutated_values, mutated_logits = run_model(model, schema, mutated)

    torch.testing.assert_close(mutated_values[:1], values[:1], rtol=0, atol=0)
    torch.testing.assert_close(mutated_logits[:1], logits[:1], rtol=0, atol=0)


@pytest.mark.parametrize("aggregation", ["attention", "gine", "sage", "match"])
def test_masks_reject_padding_edges_and_subjects(aggregation: str) -> None:
    baseline = BatchView.parse(make_batch(attr_dim=0))
    mutated_bytes = bytearray(make_batch(attr_dim=0))
    layout = _layout(2, 3, 2, 3, 2, 0)
    struct.pack_into("<I", mutated_bytes, layout["edge_count"], 2)
    struct.pack_into("<H", mutated_bytes, layout["edge_src"] + 2, 2)
    struct.pack_into("<H", mutated_bytes, layout["edge_dst"] + 2, 1)
    mutated_bytes[layout["edge_type"] + 1] = 1
    mutated_bytes[layout["subject_count"]] = 2
    struct.pack_into("<H", mutated_bytes, layout["action_subjects"] + 2, 2)
    mutated = BatchView.parse(mutated_bytes)
    schema = schema_for_view(baseline, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = make_arch(aggregation)
    model = build_model(schema, arch).eval()

    values, logits = run_model(model, schema, baseline)
    mutated_values, mutated_logits = run_model(model, schema, mutated)

    torch.testing.assert_close(mutated_values[:1], values[:1], rtol=0, atol=0)
    torch.testing.assert_close(mutated_logits[:1, :1], logits[:1, :1], rtol=0, atol=0)
    assert torch.isfinite(logits[0, 1])


@pytest.mark.parametrize("aggregation", ["attention", "gine", "sage", "match"])
def test_compile_fullgraph_and_no_recompile_for_row_count_change(aggregation: str) -> None:
    # Each variant compiles a fresh model; without a reset the process-wide
    # dynamo cache fills up and later in-process compiles (the serving
    # backend tests) fall over.
    torch._dynamo.reset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    view = BatchView.parse(make_batch(attr_dim=1))
    changed = bytearray(make_batch(attr_dim=1))
    struct.pack_into("<I", changed, 44, 1)
    changed_view = BatchView.parse(changed)
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = make_arch(aggregation)
    model = build_model(schema, arch).to(device).eval()
    stager = BatchStager(schema, view.batch_capacity, device)
    tensors = stager.copy(view)
    eager = model(tensors)
    compiled = torch.compile(model, fullgraph=True)
    actual = compiled(tensors)

    torch.testing.assert_close(actual[0], eager[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual[1], eager[1], rtol=1e-2, atol=1e-2)

    counter = torch._dynamo.testing.CompileCounter()
    counted = torch.compile(model, backend=counter, fullgraph=True)
    counted(stager.copy(view))
    counted(stager.copy(changed_view))
    assert counter.frame_count == 1


@pytest.mark.parametrize("aggregation", ["attention", "gine", "sage", "match"])
def test_expander_fixture_flows_through_model(aggregation: str) -> None:
    view = BatchView.parse((FIXTURES / "batch_expander.gzfb").read_bytes())
    schema = schema_for_view(view, node_vocab_size=8, edge_type_count=3, action_kind_vocab_size=8)
    arch = make_arch(aggregation)
    model = build_model(schema, arch).eval()

    values, logits = run_model(model, schema, view)

    assert values.shape == (2,)
    assert logits.shape == (2, 4)
    assert torch.isfinite(values[: view.row_count]).all()
    assert torch.isfinite(logits[: view.row_count, : view.action_count[0]]).all()
    assert view.edge_type[0, : view.edge_count[0]].tolist().count(2) == 3


def test_scalar_value_head_uses_opponent_features() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    changed_bytes = bytearray(make_batch(attr_dim=1))
    layout = _layout(2, 3, 2, 3, 2, 1)
    changed_bytes[layout["opponent_present"]] = 0
    changed_bytes[layout["opponent_present"] + 1] = 0
    changed = BatchView.parse(changed_bytes)
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        aggregation="attention",
        value_input="scalar",
    )
    model = build_model(schema, arch).eval()
    with torch.no_grad():
        for param in model.value.parameters():
            param.zero_()
        model.value[0].weight[0, arch.dim + 1] = 1.0
        model.value[3].weight[0, 0] = 1.0

    values, logits = run_model(model, schema, view)
    changed_values, changed_logits = run_model(model, schema, changed)

    assert values.shape == (2,)
    assert logits.shape == (2, 3)
    torch.testing.assert_close(changed_logits, logits, rtol=0, atol=0)
    assert not torch.equal(changed_values, values)


def test_pair_value_head_uses_opponent_state() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    changed_bytes = bytearray(make_batch(attr_dim=1))
    layout = _layout(2, 3, 2, 3, 2, 1)
    changed_bytes[layout["opponent_state_present"]] = 0
    changed_bytes[layout["opponent_state_present"] + 1] = 0
    changed = BatchView.parse(changed_bytes)
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        aggregation="attention",
        value_input="pair",
    )
    model = build_model(schema, arch).eval()

    values, logits = run_model(model, schema, view)
    changed_values, changed_logits = run_model(model, schema, changed)

    assert values.shape == (2,)
    assert logits.shape == (2, 3)
    torch.testing.assert_close(changed_logits, logits, rtol=0, atol=0)
    assert not torch.equal(changed_values, values)


def test_pointer_policy_head_bounded_and_masks_padded_actions() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    changed_bytes = bytearray(make_batch(attr_dim=1))
    layout = _layout(2, 3, 2, 3, 2, 1)
    # Mutate action slots past each row's action_count (row 0 counts 2,
    # row 1 counts 1): padded slots must not leak through the glimpse.
    _u16(changed_bytes, layout["action_kind"] + 2 * 2, [3])
    _bf16(changed_bytes, layout["action_prior"] + 2 * 2, [0.75])
    _u16(changed_bytes, layout["action_kind"] + 4 * 2, [5])
    _bf16(changed_bytes, layout["action_prior"] + 4 * 2, [0.5])
    changed = BatchView.parse(changed_bytes)
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(
        dim=16,
        layers=1,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        aggregation="attention",
        policy_head="pointer",
    )
    model = build_model(schema, arch).eval()

    values, logits = run_model(model, schema, view)
    changed_values, changed_logits = run_model(model, schema, changed)

    assert values.shape == (2,)
    assert logits.shape == (2, 3)
    assert logits.abs().max() <= 10.0
    torch.testing.assert_close(changed_values, values, rtol=0, atol=0)
    torch.testing.assert_close(changed_logits[0, :2], logits[0, :2], rtol=0, atol=0)
    torch.testing.assert_close(changed_logits[1, :1], logits[1, :1], rtol=0, atol=0)
    assert ArchConfig.from_dict(make_arch("attention").to_dict()).policy_head == "mlp"
    legacy = {k: v for k, v in make_arch("attention").to_dict().items() if k != "policy_head"}
    assert ArchConfig.from_dict(legacy).policy_head == "mlp"


def test_tanh_value_activation_bounds_values() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0, value_activation="tanh")
    model = build_model(schema, arch).eval()

    values, _ = run_model(model, schema, view)

    assert values.abs().max() < 1.0
    assert ArchConfig.from_dict(arch.to_dict()) == arch
    legacy = {k: v for k, v in arch.to_dict().items() if k != "value_activation"}
    assert ArchConfig.from_dict(legacy).value_activation == "logit"


def test_match_encoding_distinguishes_subject_order() -> None:
    # Two candidates over the same node set in different roles must score
    # differently under match encoding; the mean pool aliases them.
    base = bytearray(make_batch(attr_dim=1))
    layout = _layout(2, 3, 2, 3, 2, 1)
    base[layout["subject_count"]] = 2
    _u16(base, layout["action_subjects"], [0, 1])
    swapped = bytearray(base)
    _u16(swapped, layout["action_subjects"], [1, 0])
    schema = schema_for_view(BatchView.parse(base), node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)

    match_model = build_model(schema, make_arch("match")).eval()
    mean_model = build_model(schema, make_arch("attention")).eval()
    with torch.no_grad():
        _, match_a = match_model(tensors_of(schema, BatchView.parse(base)))
        _, match_b = match_model(tensors_of(schema, BatchView.parse(swapped)))
        _, mean_a = mean_model(tensors_of(schema, BatchView.parse(base)))
        _, mean_b = mean_model(tensors_of(schema, BatchView.parse(swapped)))

    assert not torch.equal(match_a[0, :1], match_b[0, :1]), "match encoding sees role order"
    torch.testing.assert_close(mean_a[0, :1], mean_b[0, :1], rtol=0, atol=0)


def tensors_of(schema: FeatureSchemaConfig, view: BatchView):
    return BatchStager(schema, view.batch_capacity, "cpu").copy(view)


def test_value_mirror_returns_both_orientations() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view, node_vocab_size=7, edge_type_count=2, action_kind_vocab_size=8)
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0, value_input="pair")
    model = build_model(schema, arch).eval()
    tensors = tensors_of(schema, view)
    with torch.no_grad():
        mirrored, _ = model(tensors, value_mirror=True)
        canonical, _ = model(tensors)

    assert mirrored.shape[0] == 2
    torch.testing.assert_close(mirrored[0], canonical, rtol=0, atol=0)
    # The swapped orientation equals canonical exactly when self and
    # opponent readouts coincide; on real batches they differ per row
    # only where an opponent state is present.
    present = tensors.opponent_state_present > 0
    if bool(present.any()):
        assert not torch.equal(mirrored[1][present], mirrored[0][present])


def test_sage_trunk_arch_round_trip_and_legacy_defaults() -> None:
    arch = make_arch("sage")
    assert ArchConfig.from_dict(arch.to_dict()) == arch
    legacy = {k: v for k, v in make_arch("attention").to_dict().items() if k not in {"trunk", "sage_layers"}}
    parsed = ArchConfig.from_dict(legacy)
    assert parsed.trunk == "exphormer"
    assert parsed.sage_layers == 3


def run_model(model: object, schema: FeatureSchemaConfig, view: BatchView):
    stager = BatchStager(schema, view.batch_capacity, "cpu")
    return model(stager.copy(view))


def make_arch(aggregation: str) -> ArchConfig:
    # "sage" is the whittlezero SAGE+transformer trunk and "match" its
    # role-preserving subject encoding; the exphormer trunk variants
    # select the edge aggregation instead.
    if aggregation == "sage":
        return ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0, trunk="sage", sage_layers=2)
    if aggregation == "match":
        return ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0, subject_encoding="match")
    return ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0, aggregation=aggregation)


def schema_for_view(
    view: BatchView,
    *,
    node_vocab_size: int,
    edge_type_count: int,
    action_kind_vocab_size: int,
) -> FeatureSchemaConfig:
    return FeatureSchemaConfig(
        name="test",
        node_vocab_size=node_vocab_size,
        node_attr_dim=view.dims.node_attr_dim,
        edge_type_count=edge_type_count,
        action_kind_vocab_size=action_kind_vocab_size,
        max_nodes=view.dims.max_nodes,
        max_edges=view.dims.max_edges,
        max_actions=view.dims.max_actions,
        max_subjects=view.dims.max_subjects,
        expander_degree=0,
        expander_seed=0,
    )
