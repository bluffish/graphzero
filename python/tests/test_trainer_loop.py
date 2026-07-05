from __future__ import annotations

import math

import pytest

from gz.trainer.loop import policy_ce_loss, value_bce_loss

torch = pytest.importorskip("torch")


def test_policy_ce_loss_matches_literal_and_ignores_padded_slots() -> None:
    logits = torch.tensor([[math.log(2.0), math.log(1.0), 100.0]], dtype=torch.float32)
    policy = torch.tensor([[0.25, 0.75, 1000.0]], dtype=torch.float32)
    action_count = torch.tensor([2], dtype=torch.int64)

    loss = policy_ce_loss(logits, policy, action_count, row_count=1)

    assert float(loss) == pytest.approx(0.25 * -math.log(2.0 / 3.0) + 0.75 * -math.log(1.0 / 3.0))

    changed = policy.clone()
    changed[0, 2] = -5000.0
    assert float(policy_ce_loss(logits, changed, action_count, row_count=1)) == pytest.approx(float(loss))


def test_value_bce_loss_matches_literal_with_tie() -> None:
    value_raw = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    value = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    valid = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)

    loss = value_bce_loss(value_raw, value, valid, row_count=3)

    expected = (math.log(2.0) + math.log1p(math.exp(-2.0)) + math.log1p(math.exp(-2.0))) / 3.0
    assert float(loss) == pytest.approx(expected)


def test_value_bce_loss_zero_valid_has_finite_gradient() -> None:
    value_raw = torch.tensor([1.0, -1.0], dtype=torch.float32, requires_grad=True)
    value = torch.tensor([1.0, -1.0], dtype=torch.float32)
    valid = torch.tensor([0.0, 0.0], dtype=torch.float32)

    loss = value_bce_loss(value_raw, value, valid, row_count=2)
    loss.backward()

    assert float(loss) == 0.0
    assert value_raw.grad is not None
    assert value_raw.grad.tolist() == [0.0, 0.0]


def test_constant_schedule_holds_base_lr_after_warmup() -> None:
    from gz.trainer.loop import lr_at_step

    assert lr_at_step(3e-4, 5, 10, 1000, "constant") == pytest.approx(1.5e-4)  # warmup ramp
    assert lr_at_step(3e-4, 500, 10, 1000, "constant") == pytest.approx(3e-4)
    assert lr_at_step(3e-4, 999999, 10, 1000, "constant") == pytest.approx(3e-4)
    # cosine still anneals
    assert lr_at_step(3e-4, 1000, 10, 1000, "cosine") < 1e-8
