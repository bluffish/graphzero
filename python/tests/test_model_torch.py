from __future__ import annotations

import struct
from dataclasses import replace

import pytest

from gz.codec import BatchView, FeatureSchemaConfig
from gz.model.exphormer import (
    ArchConfig,
    BatchStager,
    build_model,
    initialize_policy,
    initialize_value,
)
from python.tests.test_codec import _bf16, _layout, _u16, make_batch

torch = pytest.importorskip("torch")


def test_arch_round_trip_and_fixed_runtime_contract() -> None:
    arch = ArchConfig(dim=16, layers=2, heads=4, ffn_dim=32, dropout=0.0)
    assert ArchConfig.from_manifest_dict(arch.to_manifest_dict()) == arch
    soft_policy_arch = ArchConfig(
        dim=16,
        layers=2,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        auxiliary_heads="v8-v32-score-soft-policy-v2",
    )
    assert ArchConfig.from_manifest_dict(soft_policy_arch.to_manifest_dict()) == soft_policy_arch
    factorized_arch = replace(arch, policy_head="factorized-pointer")
    assert ArchConfig.from_manifest_dict(factorized_arch.to_manifest_dict()) == factorized_arch
    legacy_soft_policy_arch = replace(
        soft_policy_arch,
        auxiliary_heads="v8-v32-score-soft-policy",
    )
    legacy_model = build_model(
        schema_for_view(BatchView.parse(make_batch(attr_dim=1))),
        legacy_soft_policy_arch,
    )
    assert legacy_model.policy.soft_pointer_key is not None

    for field, value in (
        ("name", "gz-graph-v1"),
        ("state_input", "single-graph"),
        ("value_input", "pair"),
        ("trunk", "sage"),
        ("policy_head", "mlp"),
        ("position_encoding", "shared"),
        ("profile", "whittlezero"),
        ("value_head", "hl-gauss"),
    ):
        values = arch.to_manifest_dict()
        values[field] = value
        with pytest.raises(ValueError, match=field):
            ArchConfig.from_manifest_dict(values)


def test_forward_is_finite_bounded_and_split_entrypoints_match() -> None:
    view, schema, model, tensors = model_fixture()
    model.eval()
    with torch.no_grad():
        values, logits = model(tensors)
        policy = model.policy_logits(tensors)
        value = model.value_only(tensors)

    assert values.shape == (view.batch_capacity,)
    assert logits.shape == (view.batch_capacity, view.max_actions)
    assert torch.isfinite(values).all()
    assert torch.isfinite(logits).all()
    assert values.abs().max() < 1.0
    assert logits.abs().max() <= 10.0
    torch.testing.assert_close(policy, logits, rtol=0, atol=0)
    torch.testing.assert_close(value, values, rtol=0, atol=0)


def test_pointer_policy_ignores_common_raw_score_offset() -> None:
    _, _, model, _ = model_fixture()
    dim = model.arch.dim
    board = torch.zeros((1, dim))
    board[0, 0] = dim**0.5
    tokens = torch.zeros((1, 3, dim))
    tokens[0, :, 0] = torch.tensor([-0.25, 0.0, 0.5])
    action_mask = torch.ones((1, 3), dtype=torch.bool)
    identity = torch.nn.Identity()

    logits = model.policy._pointer_logits(board, tokens, identity, action_mask)
    shifted = tokens.clone()
    shifted[0, :, 0] += 5.0
    shifted_logits = model.policy._pointer_logits(
        board,
        shifted,
        identity,
        action_mask,
    )
    torch.testing.assert_close(
        torch.softmax(shifted_logits, dim=-1),
        torch.softmax(logits, dim=-1),
        rtol=1e-6,
        atol=1e-7,
    )


def test_factorized_policy_does_not_reward_a_family_for_duplicate_candidates() -> None:
    _, schema, _, _ = model_fixture()
    model = build_model(schema, replace(small_arch(), policy_head="factorized-pointer"))
    head = model.policy.eval()
    dim = model.arch.dim
    feat_dim = head.token_proj.in_features
    readout = torch.randn((1, dim))
    family_action = torch.randn((feat_dim,))
    other_action = torch.randn((feat_dim,))

    two_actions = torch.stack((family_action, other_action)).unsqueeze(0)
    two_kinds = torch.tensor([[2, 3]])
    two_mask = torch.ones((1, 2), dtype=torch.bool)
    duplicated = torch.stack(
        (family_action, family_action, family_action, other_action)
    ).unsqueeze(0)
    duplicated_kinds = torch.tensor([[2, 2, 2, 3]])
    duplicated_mask = torch.ones((1, 4), dtype=torch.bool)

    with torch.no_grad():
        two_policy = torch.softmax(
            head(readout, two_actions, two_kinds, two_mask),
            dim=-1,
        )
        duplicated_policy = torch.softmax(
            head(readout, duplicated, duplicated_kinds, duplicated_mask),
            dim=-1,
        )

    torch.testing.assert_close(
        duplicated_policy[0, :3].sum(),
        two_policy[0, 0],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        duplicated_policy[0, 3],
        two_policy[0, 1],
        rtol=1e-5,
        atol=1e-6,
    )


def test_factorized_policy_trains_family_and_conditional_scores() -> None:
    _, schema, _, _ = model_fixture()
    model = build_model(schema, replace(small_arch(), policy_head="factorized-pointer"))
    head = model.policy
    dim = model.arch.dim
    feat_dim = head.token_proj.in_features
    logits = head(
        torch.randn((1, dim)),
        torch.randn((1, 3, feat_dim)),
        torch.tensor([[2, 2, 3]]),
        torch.ones((1, 3), dtype=torch.bool),
    )
    torch.nn.functional.cross_entropy(logits, torch.tensor([0])).backward()

    assert head.family_key.weight.grad is not None
    assert head.family_key.weight.grad.abs().sum() > 0
    assert head.pointer_key.weight.grad is not None
    assert head.pointer_key.weight.grad.abs().sum() > 0


def test_factorized_conditional_scores_accept_mixed_precision_family_counts() -> None:
    _, schema, _, _ = model_fixture()
    model = build_model(schema, replace(small_arch(), policy_head="factorized-pointer"))
    head = model.policy
    dim = model.arch.dim
    logits = head._conditional_logits(
        torch.randn((1, dim)),
        torch.randn((1, 3, dim)),
        torch.tensor([[2, 2, 3]]),
        torch.ones((1, 3), dtype=torch.bool),
        torch.tensor(
            [[0, 0, 2, 1, 0, 0, 0, 0]],
            dtype=torch.bfloat16,
        ),
    )

    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_factorized_policy_supports_cuda_bfloat16_autocast() -> None:
    _, schema, _, _ = model_fixture()
    model = build_model(
        schema,
        replace(small_arch(), policy_head="factorized-pointer"),
    ).cuda()
    initialize_policy(model, "neutral")
    head = torch.compile(model.policy, fullgraph=True, dynamic=False)
    dim = model.arch.dim
    feat_dim = head.token_proj.in_features
    action_count = 271
    kinds = torch.full((1, action_count), 2, device="cuda", dtype=torch.long)
    kinds[:, -1] = 3
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = head(
            torch.randn((1, dim), device="cuda"),
            torch.randn((1, action_count, feat_dim), device="cuda"),
            kinds,
            torch.ones((1, action_count), dtype=torch.bool, device="cuda"),
        )
        loss = torch.nn.functional.cross_entropy(
            logits,
            torch.tensor([0], device="cuda"),
        )
    loss.backward()

    assert torch.isfinite(logits).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )


def test_joint_board_conditions_both_heads_on_opponent_state() -> None:
    raw = make_batch(attr_dim=1)
    view = BatchView.parse(raw)
    schema = schema_for_view(view)
    model = build_model(schema, small_arch(layers=2)).eval()
    changed = bytearray(raw)
    layout = _layout(2, 3, 2, 3, 2, 1)
    struct.pack_into("<H", changed, layout["opponent_node_tokens"], 6)

    with torch.no_grad():
        values, logits = model(tensors_of(schema, view))
        changed_values, changed_logits = model(
            tensors_of(schema, BatchView.parse(changed))
        )

    assert not torch.equal(changed_values, values)
    assert not torch.equal(changed_logits, logits)


def test_padded_actions_do_not_change_valid_outputs() -> None:
    raw = make_batch(attr_dim=1)
    view = BatchView.parse(raw)
    changed = bytearray(raw)
    layout = _layout(2, 3, 2, 3, 2, 1)
    _u16(changed, layout["action_kind"] + 2 * 2, [7])
    _bf16(changed, layout["action_prior"] + 2 * 2, [0.75])
    _u16(changed, layout["action_kind"] + 4 * 2, [6])
    _bf16(changed, layout["action_prior"] + 4 * 2, [-0.5])
    schema = schema_for_view(view)
    model = build_model(schema, small_arch()).eval()

    with torch.no_grad():
        values, logits = model(tensors_of(schema, view))
        changed_values, changed_logits = model(
            tensors_of(schema, BatchView.parse(changed))
        )

    torch.testing.assert_close(changed_values, values, rtol=0, atol=0)
    torch.testing.assert_close(changed_logits[0, :2], logits[0, :2], rtol=0, atol=0)
    torch.testing.assert_close(changed_logits[1, :1], logits[1, :1], rtol=0, atol=0)


def test_remaining_budget_ignores_raw_root_step() -> None:
    raw = make_batch(attr_dim=1)
    changed = bytearray(raw)
    layout = _layout(2, 3, 2, 3, 2, 1)
    _bf16(changed, layout["position"], [63.0])
    view = BatchView.parse(raw)
    schema = schema_for_view(view)
    model = build_model(schema, small_arch()).eval()

    with torch.no_grad():
        expected = model(tensors_of(schema, view))
        actual = model(tensors_of(schema, BatchView.parse(changed)))

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_batch_stager_copies_joint_board_indexes_as_int32() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    tensors = tensors_of(schema_for_view(view), view)
    for name in (
        "node_count",
        "node_tokens",
        "edge_count",
        "edge_src",
        "edge_dst",
        "edge_type",
        "action_count",
        "action_kind",
        "subject_count",
        "action_subjects",
        "opponent_node_count",
        "opponent_node_tokens",
        "opponent_edge_count",
        "opponent_edge_src",
        "opponent_edge_dst",
        "opponent_edge_type",
    ):
        assert getattr(tensors, name).dtype == torch.int32
    assert tensors.opponent_node_tokens[0, 0].item() == 5


def test_neutral_initializers_preserve_other_modules_and_remain_trainable() -> None:
    _, _, model, tensors = model_fixture()
    value_before = {name: tensor.clone() for name, tensor in model.value.state_dict().items()}
    initialize_policy(model, "neutral")
    logits = model.policy_logits(tensors)
    torch.testing.assert_close(logits[0, :2], torch.zeros(2), rtol=0, atol=0)
    assert all(
        torch.equal(model.value.state_dict()[name], tensor)
        for name, tensor in value_before.items()
    )
    torch.nn.functional.cross_entropy(
        logits[0, :2].unsqueeze(0), torch.tensor([0])
    ).backward()
    assert model.policy.pointer_key.weight.grad is not None
    assert model.policy.pointer_key.weight.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    initialize_value(model, "zero")
    assert all(torch.count_nonzero(tensor) == 0 for tensor in model.value[-1].state_dict().values())
    model.value_only(tensors).sum().backward()
    assert model.value[-1].weight.grad is not None
    assert model.value[-1].weight.grad.abs().sum() > 0


def test_value_trunk_gradient_scale_leaves_forward_and_head_gradient_unchanged() -> None:
    _, schema, full, tensors = model_fixture()
    scaled = build_model(schema, full.arch)
    scaled.load_state_dict(full.state_dict())

    full_value = full.value_only(tensors, value_trunk_grad_scale=1.0)
    scaled_value = scaled.value_only(tensors, value_trunk_grad_scale=0.1)
    torch.testing.assert_close(scaled_value, full_value, rtol=0, atol=0)
    full_value.sum().backward()
    scaled_value.sum().backward()

    torch.testing.assert_close(
        scaled.value[-1].weight.grad,
        full.value[-1].weight.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        scaled.node_embedding.weight.grad,
        full.node_embedding.weight.grad * 0.1,
        rtol=1e-5,
        atol=1e-6,
    )


def test_auxiliary_heads_share_trunk_without_changing_serving_outputs() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view)
    model = build_model(
        schema,
        small_arch(auxiliary_heads="v8-v32-score-soft-policy-v2"),
    ).eval()
    tensors = tensors_of(schema, view)
    with torch.no_grad():
        serving_before = model(tensors)
        value, horizons, score, logits, soft_logits, readout = (
            model.training_forward_with_soft_policy(tensors)
        )
        for parameter in (*model.horizon_value.parameters(), *model.terminal_score.parameters()):
            parameter.add_(torch.randn_like(parameter))
        for parameter in (
            *model.soft_policy_kind_embedding.parameters(),
            *model.soft_policy.parameters(),
        ):
            parameter.add_(torch.randn_like(parameter))
        serving_after = model(tensors)

    assert value.shape == (view.batch_capacity,)
    assert horizons.shape == (view.batch_capacity, 2)
    assert score.shape == (view.batch_capacity,)
    assert logits.shape == (view.batch_capacity, view.max_actions)
    assert soft_logits.shape == (view.batch_capacity, view.max_actions)
    assert readout.shape == (view.batch_capacity, model.arch.dim)
    torch.testing.assert_close(serving_after[0], serving_before[0], rtol=0, atol=0)
    torch.testing.assert_close(serving_after[1], serving_before[1], rtol=0, atol=0)


def test_soft_policy_gradient_isolated_from_main_head_and_scaled_at_trunk() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view)
    full = build_model(
        schema,
        small_arch(auxiliary_heads="v8-v32-score-soft-policy-v2"),
    )
    scaled = build_model(schema, full.arch)
    scaled.load_state_dict(full.state_dict())
    tensors = tensors_of(schema, view)

    _, full_soft = full.training_policy_logits(
        tensors,
        soft_policy_trunk_grad_scale=1.0,
    )
    _, scaled_soft = scaled.training_policy_logits(
        tensors,
        soft_policy_trunk_grad_scale=0.1,
    )
    full_soft.sum().backward()
    scaled_soft.sum().backward()

    assert all(parameter.grad is None for parameter in full.policy.parameters())
    assert all(parameter.grad is None for parameter in scaled.policy.parameters())
    assert full.kind_embedding.weight.grad is None
    assert scaled.kind_embedding.weight.grad is None
    torch.testing.assert_close(
        scaled.soft_policy.pointer_key.weight.grad,
        full.soft_policy.pointer_key.weight.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        scaled.soft_policy_kind_embedding.weight.grad,
        full.soft_policy_kind_embedding.weight.grad,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        scaled.node_embedding.weight.grad,
        full.node_embedding.weight.grad * 0.1,
        rtol=1e-5,
        atol=1e-6,
    )


def test_soft_policy_v2_preserves_preexisting_initialization() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view)
    torch.manual_seed(5)
    baseline = build_model(
        schema,
        small_arch(auxiliary_heads="v8-v32-score"),
    )
    torch.manual_seed(5)
    auxiliary = build_model(
        schema,
        small_arch(auxiliary_heads="v8-v32-score-soft-policy-v2"),
    )

    baseline_state = baseline.state_dict()
    auxiliary_state = auxiliary.state_dict()
    assert all(
        torch.equal(tensor, auxiliary_state[name])
        for name, tensor in baseline_state.items()
    )


def model_fixture():
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = schema_for_view(view)
    model = build_model(schema, small_arch())
    return view, schema, model, tensors_of(schema, view)


def small_arch(*, layers: int = 1, auxiliary_heads: str = "none") -> ArchConfig:
    return ArchConfig(
        dim=16,
        layers=layers,
        heads=4,
        ffn_dim=32,
        dropout=0.0,
        auxiliary_heads=auxiliary_heads,
    )


def schema_for_view(view: BatchView) -> FeatureSchemaConfig:
    return FeatureSchemaConfig(
        name="test",
        node_vocab_size=8,
        node_attr_dim=view.dims.node_attr_dim,
        edge_type_count=3,
        action_kind_vocab_size=8,
        max_nodes=view.dims.max_nodes,
        max_edges=view.dims.max_edges,
        max_actions=view.dims.max_actions,
        max_subjects=view.dims.max_subjects,
        expander_degree=0,
        expander_seed=0,
    )


def tensors_of(schema: FeatureSchemaConfig, view: BatchView):
    return BatchStager(schema, view.batch_capacity, "cpu").copy(view)
