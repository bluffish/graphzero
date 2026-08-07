from __future__ import annotations

from dataclasses import dataclass

from gz.trainer.data import TrainingBatch
from gz.trainer.optim import build_optimizer
from gz.trainer.step import StepMetrics, execute_training_step
from gz.trainer.losses import (
    _torch,
    _validate_task_weights,
    auxiliary_task_loss,
    lr_at_step,
    policy_ce_loss,
    soft_policy_ce_loss,
    softened_policy_target,
    value_head_loss,
    value_mse_loss,
)


@dataclass(frozen=True, slots=True)
class LoopConfig:
    lr: float = 3e-4
    warmup_steps: int = 200
    total_steps: int = 1000
    lr_decay_steps: int | None = None
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.0
    value_weight: float = 1.0
    value_trunk_grad_scale: float = 1.0
    value_final_weight: float = 1.0
    value_v8_weight: float = 0.0
    value_v32_weight: float = 0.0
    terminal_score_weight: float = 0.0
    terminal_score_target: str = "node-count"
    soft_policy_weight: float = 0.0
    soft_policy_temperature: float = 4.0
    soft_policy_trunk_grad_scale: float = 1.0
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    adamw_lr: float | None = None
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    mask_stop_loss: bool = False
    compile_model: bool = False
    compile_mode: str = "default"


class TrainerLoop:
    def __init__(self, model: object, config: LoopConfig) -> None:
        torch = _torch()
        self.model = model
        self.config = config
        _validate_task_weights(config)
        auxiliary_heads = getattr(
            getattr(model, "arch", None),
            "auxiliary_heads",
            "none",
        )
        self._auxiliary_layout = auxiliary_heads in {
            "v8-v32-score",
            "v8-v32-score-soft-policy-v2",
        }
        self._soft_policy_layout = auxiliary_heads == "v8-v32-score-soft-policy-v2"
        self._auxiliary_tasks = any(
            weight > 0.0
            for weight in (
                config.value_v8_weight,
                config.value_v32_weight,
                config.terminal_score_weight,
            )
        )
        if self._auxiliary_tasks and not self._auxiliary_layout:
            raise ValueError("auxiliary value task weights require v8-v32-score model heads")
        self._soft_policy_task = config.soft_policy_weight > 0.0
        if self._soft_policy_task and not self._soft_policy_layout:
            raise ValueError("soft-policy weight requires a soft-policy model head")
        self.optimizer = build_optimizer(
            model,
            name=config.optimizer,
            lr=config.lr,
            adamw_lr=config.adamw_lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=config.nesterov,
            ns_steps=config.ns_steps,
        )
        self._model_forward = model
        self._policy_logits = model.policy_logits
        self._value_only = model.value_only
        self._training_forward = getattr(model, "training_forward", None)
        self._training_values = getattr(model, "training_values", None)
        self._training_policy_logits = getattr(model, "training_policy_logits", None)
        self._training_forward_with_soft_policy = getattr(
            model,
            "training_forward_with_soft_policy",
            None,
        )
        if self._auxiliary_tasks and (
            self._training_forward is None or self._training_values is None
        ):
            raise ValueError("model does not expose auxiliary training outputs")
        if self._soft_policy_task and (
            self._training_policy_logits is None
            or self._training_forward_with_soft_policy is None
        ):
            raise ValueError("model does not expose soft-policy training outputs")
        if config.compile_model:
            if self._auxiliary_tasks:
                # Logging steps take several retained VJPs at the shared
                # readout before the combined backward. AOTAutograd's donated
                # backward buffers permit only one backward invocation.
                from torch._functorch import config as functorch_config

                functorch_config.donated_buffer = False
            compile_options = {
                "fullgraph": True,
                "dynamic": False,
                "mode": config.compile_mode,
            }
            self._model_forward = torch.compile(model, **compile_options)
            # Keep a graph boundary between the trunk and policy head. A single
            # bf16 Inductor graph spanning both produces incorrect backward
            # gradients on the 10M joint-board model.
            def policy_trunk(batch: object) -> object:
                graph, node_roles = model._model_graph(batch)
                return model._encode_graph(graph, node_roles)

            compiled_policy_trunk = torch.compile(policy_trunk, **compile_options)
            compiled_policy_head = torch.compile(model._policy_logits, **compile_options)

            def split_policy_logits(batch: object) -> object:
                h, g_readout, node_mask = compiled_policy_trunk(batch)
                return compiled_policy_head(batch, h, g_readout, node_mask)

            self._policy_logits = split_policy_logits
            self._value_only = torch.compile(model.value_only, **compile_options)
            if self._soft_policy_task:
                compiled_training_policy_head = torch.compile(
                    model._training_policy_logits,
                    **compile_options,
                )

                def split_training_policy_logits(
                    batch: object,
                    soft_policy_trunk_grad_scale: float = 1.0,
                ) -> tuple[object, object]:
                    h, g_readout, node_mask = compiled_policy_trunk(batch)
                    return compiled_training_policy_head(
                        batch,
                        h,
                        g_readout,
                        node_mask,
                        soft_policy_trunk_grad_scale,
                    )

                self._training_policy_logits = split_training_policy_logits
            if self._auxiliary_tasks:
                compiled_auxiliary_heads = torch.compile(
                    model._training_value_outputs,
                    **compile_options,
                )

                def split_training_forward(
                    batch: object,
                    value_trunk_grad_scale: float = 1.0,
                ) -> tuple[object, object, object, object, object]:
                    h, g_readout, node_mask = compiled_policy_trunk(batch)
                    value_outputs = compiled_auxiliary_heads(
                        g_readout,
                        value_trunk_grad_scale,
                    )
                    logits = compiled_policy_head(batch, h, g_readout, node_mask)
                    return (*value_outputs, logits, g_readout)

                def split_training_values(
                    batch: object,
                    value_trunk_grad_scale: float = 1.0,
                ) -> tuple[object, object, object, object]:
                    _, g_readout, _ = compiled_policy_trunk(batch)
                    return (
                        *compiled_auxiliary_heads(
                            g_readout,
                            value_trunk_grad_scale,
                        ),
                        g_readout,
                    )

                self._training_forward = split_training_forward
                self._training_values = split_training_values
                if self._soft_policy_task:

                    def split_training_forward_with_soft_policy(
                        batch: object,
                        value_trunk_grad_scale: float = 1.0,
                        soft_policy_trunk_grad_scale: float = 1.0,
                    ) -> tuple[object, object, object, object, object, object]:
                        h, g_readout, node_mask = compiled_policy_trunk(batch)
                        value_outputs = compiled_auxiliary_heads(
                            g_readout,
                            value_trunk_grad_scale,
                        )
                        logits, soft_policy_logits = compiled_training_policy_head(
                            batch,
                            h,
                            g_readout,
                            node_mask,
                            soft_policy_trunk_grad_scale,
                        )
                        return (
                            *value_outputs,
                            logits,
                            soft_policy_logits,
                            g_readout,
                        )

                    self._training_forward_with_soft_policy = (
                        split_training_forward_with_soft_policy
                    )
        self.step_index = 0
        # bf16 autocast on CUDA, matching the evaluator's serving numerics.
        # Params and optimizer state stay f32; no GradScaler is needed for
        # bf16 (full f32 exponent range).
        self.device_type = next(model.parameters()).device.type

    def train_step(
        self,
        batch: TrainingBatch,
        value_batch: TrainingBatch | None = None,
        with_metrics: bool = True,
    ) -> StepMetrics | None:
        """Run one optimizer step without synchronizing unless metrics are requested."""
        return execute_training_step(self, batch, value_batch, with_metrics)
