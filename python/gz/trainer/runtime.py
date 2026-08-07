from __future__ import annotations

from gz.trainer.config import TrainerConfig
from gz.trainer.loop import LoopConfig


def seed_model(seed: int) -> None:
    import torch

    torch.manual_seed(seed)


def set_matmul_precision(precision: str) -> None:
    import torch

    torch.set_float32_matmul_precision(precision)


def trainer_loop_config(
    config: TrainerConfig,
    *,
    symmetric_mask_stop: bool = True,
) -> LoopConfig:
    return LoopConfig(
        lr=config.lr,
        lr_schedule=config.lr_schedule,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps,
        lr_decay_steps=config.lr_decay_steps,
        min_lr_ratio=config.min_lr_ratio,
        value_weight=config.value_weight,
        value_trunk_grad_scale=config.value_trunk_grad_scale,
        value_final_weight=config.value_final_weight,
        value_v8_weight=config.value_v8_weight,
        value_v32_weight=config.value_v32_weight,
        terminal_score_weight=config.terminal_score_weight,
        terminal_score_target=config.terminal_score_target,
        soft_policy_weight=config.soft_policy_weight,
        soft_policy_temperature=config.soft_policy_temperature,
        soft_policy_trunk_grad_scale=config.soft_policy_trunk_grad_scale,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        adamw_lr=config.adamw_lr,
        momentum=config.momentum,
        nesterov=config.nesterov,
        ns_steps=config.ns_steps,
        grad_clip=config.grad_clip,
        mask_stop_loss=symmetric_mask_stop,
        compile_model=config.compile_model,
        compile_mode=config.compile_mode,
    )
