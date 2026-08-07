from __future__ import annotations

import math
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from gz.codec import BatchView, FeatureSchemaConfig
from gz.model.exphormer import ArchConfig, BatchStager, build_model
from gz.trainer.data import TrainingBatch
from gz.trainer.loop import (
    LoopConfig,
    TrainerLoop,
    auxiliary_task_loss,
    lr_at_step,
    policy_ce_loss,
    soft_policy_ce_loss,
    softened_policy_target,
    value_head_loss,
    value_mse_loss,
)
from python.tests.test_codec import make_batch

torch = pytest.importorskip("torch")


def test_policy_ce_loss_masks_padding_and_reserved_stop() -> None:
    logits = torch.tensor(
        [[math.log(3.0), math.log(1.0), 100.0]],
        requires_grad=True,
    )
    policy = torch.tensor([[0.5, 0.5, 999.0]])
    action_count = torch.tensor([3])
    action_kind = torch.tensor([[2, 3, 1]])

    loss = policy_ce_loss(
        logits,
        policy,
        action_count,
        row_count=1,
        action_kind=action_kind,
        mask_stop=True,
    )
    expected = 0.5 * -math.log(3.0 / 4.0) + 0.5 * -math.log(1.0 / 4.0)
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 2].item() == 0.0


def test_policy_ce_loss_ignores_value_only_zero_action_rows() -> None:
    logits = torch.tensor([[0.0, 0.0], [50.0, -50.0]], requires_grad=True)
    policy = torch.tensor([[0.25, 0.75], [0.0, 0.0]])
    action_count = torch.tensor([2, 0])

    loss = policy_ce_loss(logits, policy, action_count, row_count=2)

    assert float(loss.detach()) == pytest.approx(math.log(2.0))
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[1].tolist() == [0.0, 0.0]


def test_t4_soft_policy_target_preserves_zeros_and_renormalizes() -> None:
    policy = torch.tensor(
        [
            [0.60, 0.36, 0.03, 0.01, 999.0],
            [0.0, 0.0, 0.0, 0.0, 999.0],
        ]
    )
    action_count = torch.tensor([4, 4])
    target = softened_policy_target(
        policy,
        action_count,
        temperature=4.0,
    )
    expected = policy[0, :4].pow(0.25)
    expected = expected / expected.sum()

    torch.testing.assert_close(target[0, :4], expected)
    assert target[0, 4].item() == 0.0
    assert torch.count_nonzero(target[1]).item() == 0

    logits = torch.zeros_like(policy, requires_grad=True)
    loss, kl, entropy = soft_policy_ce_loss(
        logits,
        policy,
        action_count,
        row_count=2,
        temperature=4.0,
    )
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert float(loss.detach()) == pytest.approx(math.log(4.0))
    assert float(kl.detach()) == pytest.approx(math.log(4.0) - float(entropy))


def test_value_mse_masks_invalid_and_zero_valid_has_finite_gradient() -> None:
    prediction = torch.tensor([0.5, -0.25, 0.9], requires_grad=True)
    target = torch.tensor([1.0, -1.0, 1.0])
    valid = torch.tensor([1.0, 1.0, 0.0])
    loss = value_mse_loss(prediction, target, valid, row_count=3)
    assert float(loss.detach()) == pytest.approx((0.5**2 + 0.75**2) / 2.0)

    empty = torch.tensor([1.0, -1.0], requires_grad=True)
    empty_loss = value_head_loss(
        empty,
        torch.tensor([1.0, -1.0]),
        torch.zeros(2),
        row_count=2,
    )
    empty_loss.backward()
    assert float(empty_loss) == 0.0
    assert empty.grad is not None
    assert empty.grad.tolist() == [0.0, 0.0]


def test_auxiliary_task_loss_uses_configured_weights_and_node_scale() -> None:
    model = SimpleNamespace(schema=SimpleNamespace(max_nodes=100))
    batch = training_batch(2)
    batch = batch._replace(
        value=torch.tensor([1.0, -1.0]),
        horizon_value=torch.tensor([[0.5, -0.5], [1.0, -1.0]]),
        horizon_value_valid=torch.ones(2),
        reward=torch.tensor([-73.0, -20.0]),
    )
    score_raw = torch.logit(torch.tensor([0.73, 0.20]))
    config = LoopConfig(
        value_final_weight=0.5,
        value_v8_weight=0.2,
        value_v32_weight=0.2,
        terminal_score_weight=0.1,
    )

    combined, final, v8, v32, score, prediction = auxiliary_task_loss(
        model,
        torch.zeros(2),
        torch.zeros((2, 2)),
        score_raw,
        batch,
        config,
    )

    assert float(final) == pytest.approx(1.0)
    assert float(v8) == pytest.approx(0.625)
    assert float(v32) == pytest.approx(0.625)
    assert float(score) == pytest.approx(0.0, abs=1e-12)
    assert float(combined) == pytest.approx(0.75)
    torch.testing.assert_close(prediction, torch.tensor([73.0, 20.0]))


def test_auxiliary_latency_loss_uses_log1p_cost_and_handles_penalty() -> None:
    model = SimpleNamespace(schema=SimpleNamespace(max_nodes=100))
    costs = torch.tensor([5.25, 10_000.0])
    batch = training_batch(2)._replace(
        value=torch.tensor([1.0, -1.0]),
        horizon_value=torch.tensor([[0.5, -0.5], [1.0, -1.0]]),
        horizon_value_valid=torch.ones(2),
        reward=-costs,
    )
    log_costs = torch.log1p(costs)
    score_raw = torch.log(torch.expm1(log_costs))
    config = LoopConfig(
        value_final_weight=0.5,
        value_v8_weight=0.2,
        value_v32_weight=0.2,
        terminal_score_weight=0.1,
        terminal_score_target="log1p-cost-ms",
    )

    combined, _, _, _, score, prediction = auxiliary_task_loss(
        model,
        torch.zeros(2),
        torch.zeros((2, 2)),
        score_raw,
        batch,
        config,
    )

    assert float(score) == pytest.approx(0.0, abs=1e-12)
    assert torch.isfinite(combined)
    torch.testing.assert_close(prediction, costs, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "config",
    [
        LoopConfig(value_final_weight=0.5),
        LoopConfig(value_final_weight=1.1, value_v8_weight=-0.1),
        LoopConfig(value_final_weight=float("nan")),
    ],
)
def test_trainer_rejects_invalid_value_task_weights(config: LoopConfig) -> None:
    with pytest.raises(ValueError, match="value task weights"):
        TrainerLoop(object(), config)


def test_shared_batch_uses_combined_forward() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
            self.arch = SimpleNamespace(auxiliary_heads="none")
            self.forward_calls = 0

        def forward(self, features, value_trunk_grad_scale=1.0):
            self.forward_calls += 1
            assert value_trunk_grad_scale == 0.25
            rows = features.action_count.shape[0]
            value = torch.tanh(self.weight).expand(rows)
            logits = torch.stack((self.weight.expand(rows), -self.weight.expand(rows)), dim=1)
            return value, logits

        def policy_logits(self, *_args):
            raise AssertionError("split policy path must not run")

        def value_only(self, *_args, **_kwargs):
            raise AssertionError("split value path must not run")

    model = Model()
    loop = TrainerLoop(
        model,
        LoopConfig(
            lr=1e-3,
            warmup_steps=0,
            total_steps=1,
            value_trunk_grad_scale=0.25,
        ),
    )
    metrics = loop.train_step(training_batch(2))
    assert metrics is not None
    assert model.forward_calls == 1
    assert model.weight.grad is not None
    assert metrics.fraction_valid == 1.0


def test_independent_value_batch_uses_split_paths_and_trims_capacity() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
            self.arch = SimpleNamespace(auxiliary_heads="none")
            self.policy_calls = 0
            self.value_rows = 0

        def forward(self, *_args, **_kwargs):
            raise AssertionError("combined path must not run")

        def policy_logits(self, features):
            self.policy_calls += 1
            rows = features.action_count.shape[0]
            return torch.stack((self.weight.expand(rows), -self.weight.expand(rows)), dim=1)

        def value_only(self, features, value_trunk_grad_scale=1.0):
            assert self.weight.grad is not None
            assert value_trunk_grad_scale == 0.1
            self.value_rows = features.action_count.shape[0]
            return torch.tanh(self.weight).expand(self.value_rows)

    model = Model()
    loop = TrainerLoop(
        model,
        LoopConfig(
            lr=1e-3,
            warmup_steps=0,
            total_steps=1,
            value_trunk_grad_scale=0.1,
        ),
    )
    value_batch = training_batch(3)._replace(row_count=1)
    metrics = loop.train_step(training_batch(2), value_batch)
    assert metrics is not None
    assert model.policy_calls == 1
    assert model.value_rows == 1
    assert metrics.fraction_valid == 1.0


def test_zero_value_weight_uses_policy_only_path() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy_weight = torch.nn.Parameter(torch.tensor(0.1))
            self.value_weight = torch.nn.Parameter(torch.tensor(0.0))
            self.arch = SimpleNamespace(auxiliary_heads="none")

        def forward(self, *_args, **_kwargs):
            raise AssertionError("combined path must not run")

        def value_only(self, *_args, **_kwargs):
            raise AssertionError("value path must not run")

        def policy_logits(self, features):
            rows = features.action_count.shape[0]
            return torch.stack(
                (self.policy_weight.expand(rows), -self.policy_weight.expand(rows)),
                dim=1,
            )

    model = Model()
    loop = TrainerLoop(model, LoopConfig(value_weight=0.0, warmup_steps=0))
    metrics = loop.train_step(training_batch(2))
    assert metrics is not None
    assert metrics.value_loss == 0.0
    assert model.policy_weight.grad is not None
    assert model.value_weight.grad is None


def test_auxiliary_model_trains_through_shared_and_independent_batches() -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = FeatureSchemaConfig(
        name="test",
        node_vocab_size=8,
        node_attr_dim=1,
        edge_type_count=3,
        action_kind_vocab_size=8,
        max_nodes=view.dims.max_nodes,
        max_edges=view.dims.max_edges,
        max_actions=view.dims.max_actions,
        max_subjects=view.dims.max_subjects,
        expander_degree=0,
        expander_seed=0,
    )
    model = build_model(
        schema,
        ArchConfig(
            dim=16,
            layers=1,
            heads=4,
            ffn_dim=32,
            dropout=0.0,
            auxiliary_heads="v8-v32-score-soft-policy-v2",
        ),
    )
    features = BatchStager(schema, view.batch_capacity, "cpu").copy(view)
    policy = torch.zeros((view.batch_capacity, view.max_actions))
    policy[:, 0] = 1.0
    batch = TrainingBatch(
        features=features,
        policy=policy,
        value=torch.tensor([1.0, -1.0]),
        value_valid=torch.ones(2),
        horizon_value=torch.tensor([[1.0, 1.0], [-1.0, -1.0]]),
        horizon_value_valid=torch.ones(2),
        reward=torch.tensor([-20.0, -30.0]),
        row_count=2,
    )
    config = LoopConfig(
        warmup_steps=0,
        value_final_weight=0.5,
        value_v8_weight=0.2,
        value_v32_weight=0.2,
        terminal_score_weight=0.1,
        soft_policy_weight=8.0,
        soft_policy_temperature=4.0,
        soft_policy_trunk_grad_scale=0.1,
    )
    soft_before = model.soft_policy.pointer_key.weight.detach().clone()
    metrics = TrainerLoop(model, config).train_step(batch)
    assert metrics is not None
    assert metrics.policy_target_entropy == pytest.approx(0.0)
    assert metrics.policy_kl == pytest.approx(metrics.policy_loss)
    assert metrics.policy_uniform_advantage == pytest.approx(
        0.5 * torch.log(torch.tensor(2.0)).item() - metrics.policy_loss
    )
    assert metrics.policy_logit_std >= 0.0
    assert metrics.policy_logit_abs_mean >= 0.0
    assert metrics.soft_policy_loss > metrics.soft_policy_target_entropy
    assert metrics.soft_policy_kl == pytest.approx(
        metrics.soft_policy_loss - metrics.soft_policy_target_entropy
    )
    assert not torch.equal(model.soft_policy.pointer_key.weight, soft_before)

    split_model = build_model(schema, model.arch)
    assert (
        TrainerLoop(split_model, config).train_step(
            batch,
            batch,
            with_metrics=False,
        )
        is None
    )


def test_compile_wraps_soft_policy_training_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = BatchView.parse(make_batch(attr_dim=1))
    schema = FeatureSchemaConfig(
        name="test",
        node_vocab_size=8,
        node_attr_dim=1,
        edge_type_count=3,
        action_kind_vocab_size=8,
        max_nodes=view.dims.max_nodes,
        max_edges=view.dims.max_edges,
        max_actions=view.dims.max_actions,
        max_subjects=view.dims.max_subjects,
        expander_degree=0,
        expander_seed=0,
    )
    model = build_model(
        schema,
        ArchConfig(
            dim=16,
            layers=1,
            heads=4,
            ffn_dim=32,
            dropout=0.0,
            auxiliary_heads="v8-v32-score-soft-policy-v2",
        ),
    )
    features = BatchStager(schema, view.batch_capacity, "cpu").copy(view)
    policy = torch.zeros((view.batch_capacity, view.max_actions))
    policy[:, 0] = 1.0
    batch = TrainingBatch(
        features=features,
        policy=policy,
        value=torch.tensor([1.0, -1.0]),
        value_valid=torch.ones(2),
        horizon_value=torch.tensor([[1.0, 1.0], [-1.0, -1.0]]),
        horizon_value_valid=torch.ones(2),
        reward=torch.tensor([-20.0, -30.0]),
        row_count=2,
    )
    compiled = []

    def compile_path(path, **_options):
        compiled.append(path)
        return path

    monkeypatch.setattr(torch, "compile", compile_path)
    loop = TrainerLoop(
        model,
        LoopConfig(
            warmup_steps=0,
            value_final_weight=0.5,
            value_v8_weight=0.2,
            value_v32_weight=0.2,
            terminal_score_weight=0.1,
            soft_policy_weight=8.0,
            soft_policy_temperature=4.0,
            soft_policy_trunk_grad_scale=0.1,
            compile_model=True,
        ),
    )

    assert loop.train_step(batch, with_metrics=False) is None
    assert len(compiled) == 6


def test_compile_wraps_current_static_training_entrypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
            self.arch = SimpleNamespace(auxiliary_heads="none")

        def forward(self, features, value_trunk_grad_scale=1.0):
            del value_trunk_grad_scale
            rows = features.action_count.shape[0]
            return self.weight.expand(rows), self.weight.expand(rows, 2)

        def policy_logits(self, features):
            return self.weight.expand(features.action_count.shape[0], 2)

        def value_only(self, features, value_trunk_grad_scale=1.0):
            del value_trunk_grad_scale
            return self.weight.expand(features.action_count.shape[0])

        def _model_graph(self, features):
            return features, None

        def _encode_graph(self, graph, node_roles):
            del node_roles
            rows = graph.action_count.shape[0]
            return self.weight.expand(rows, 1, 1), self.weight.expand(rows, 1), None

        def _policy_logits(self, features, h, readout, node_mask):
            del h, readout, node_mask
            return self.weight.expand(features.action_count.shape[0], 2)

    calls = []

    def compile_path(path, **options):
        calls.append((path, options))
        return path

    monkeypatch.setattr(torch, "compile", compile_path)
    model = Model()
    loop = TrainerLoop(
        model,
        LoopConfig(compile_model=True, compile_mode="reduce-overhead"),
    )
    assert len(calls) == 4
    assert calls[0][0] is model
    assert calls[2][0] == model._policy_logits
    assert calls[3][0] == model.value_only
    assert all(
        options
        == {"fullgraph": True, "dynamic": False, "mode": "reduce-overhead"}
        for _, options in calls
    )
    features = Features(action_count=torch.ones(2, dtype=torch.int64), action_kind=torch.zeros((2, 2), dtype=torch.int64))
    assert loop._policy_logits(features).shape == (2, 2)


def test_nonfinite_gradient_aborts_before_optimizer_step() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))
            self.arch = SimpleNamespace(auxiliary_heads="none")

        def forward(self, *_args, **_kwargs):
            raise AssertionError("combined path must not run")

        def value_only(self, *_args, **_kwargs):
            raise AssertionError("value path must not run")

        def policy_logits(self, features):
            value = self.weight * torch.tensor(float("nan"))
            return value.expand(features.action_count.shape[0], 2)

    model = Model()
    loop = TrainerLoop(model, LoopConfig(value_weight=0.0, warmup_steps=0))
    with pytest.raises(RuntimeError, match="non-finite"):
        loop.train_step(training_batch(2))
    assert model.weight.item() == pytest.approx(0.1)


def test_learning_rate_schedules() -> None:
    assert lr_at_step(3e-4, 5, 10, 1000, "constant") == pytest.approx(1.5e-4)
    assert lr_at_step(3e-4, 500, 10, 1000, "constant") == pytest.approx(3e-4)
    assert lr_at_step(3e-4, 1000, 10, 1000, "cosine", 0.1) == pytest.approx(3e-5)


class Features(NamedTuple):
    action_count: object
    action_kind: object


def training_batch(capacity: int) -> TrainingBatch:
    return TrainingBatch(
        features=Features(
            action_count=torch.full((capacity,), 2, dtype=torch.int64),
            action_kind=torch.zeros((capacity, 2), dtype=torch.int64),
        ),
        policy=torch.full((capacity, 2), 0.5),
        value=torch.tensor(([1.0, -1.0] * ((capacity + 1) // 2))[:capacity]),
        value_valid=torch.ones(capacity),
        horizon_value=torch.zeros((capacity, 2)),
        horizon_value_valid=torch.zeros(capacity),
        reward=-torch.full((capacity,), 100.0),
        row_count=capacity,
    )
