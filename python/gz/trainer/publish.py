from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gz.checkpoints import CheckpointManifest, publish_checkpoint
from gz.codec import FeatureSchemaConfig
from gz.common import ActionSetHash, EngineId, EngineVersion, FeatureSchemaHash
from gz.model.exphormer import ArchConfig


class EmaWeights:
    def __init__(self, model: object, decay: float) -> None:
        if decay < 0.0 or decay >= 1.0:
            raise ValueError("ema decay must be in [0, 1)")
        self.decay = decay
        self.shadow = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    def update(self, model: object) -> None:
        import torch

        # Fused multi-tensor update: the per-tensor loop launched two tiny
        # kernels per parameter every step and was the trainer's single
        # largest line (~19% of step wall). _foreach_ batches the same
        # mul/add arithmetic into a handful of launches, bit-identically.
        float_shadows = []
        float_lives = []
        for name, tensor in model.state_dict().items():
            live = tensor.detach()
            shadow = self.shadow[name]
            if live.is_floating_point():
                float_shadows.append(shadow)
                float_lives.append(live)
            else:
                shadow.copy_(live)
        if float_shadows:
            torch._foreach_mul_(float_shadows, self.decay)
            torch._foreach_add_(float_shadows, float_lives, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {name: tensor.detach().clone() for name, tensor in self.shadow.items()}

    def norms(self, previous: dict[str, object] | None) -> tuple[float, float]:
        """(L2 norm of the EMA weights, L2 norm of the delta vs `previous`).
        The update norm is 0.0 when there is no previous snapshot."""
        import torch

        with torch.no_grad():
            param_sq = 0.0
            delta_sq = 0.0
            for name, tensor in self.shadow.items():
                if not tensor.is_floating_point():
                    continue
                param_sq += float(tensor.float().pow(2).sum())
                if previous is not None:
                    delta_sq += float((tensor.float() - previous[name].float()).pow(2).sum())
            return param_sq**0.5, delta_sq**0.5


@dataclass(frozen=True, slots=True)
class PublishTags:
    engine_id: EngineId
    engine_version: EngineVersion
    action_set_hash: ActionSetHash

    @classmethod
    def zeros(cls) -> PublishTags:
        return cls(
            engine_id=EngineId.from_bytes(b"\x00" * 16),
            engine_version=EngineVersion.from_bytes(b"\x00" * 16),
            action_set_hash=ActionSetHash.from_bytes(b"\x00" * 32),
        )


def publish_ema(
    checkpoint_dir: str | Path,
    ema: EmaWeights,
    *,
    schema: FeatureSchemaConfig,
    schema_hash: FeatureSchemaHash,
    arch: ArchConfig,
    training_step: int,
    run_id: str,
    tags: PublishTags | None = None,
) -> CheckpointManifest:
    tags = tags or PublishTags.zeros()
    return publish_checkpoint(
        checkpoint_dir,
        ema.state_dict(),
        arch_name=arch.name,
        arch_config=arch.to_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=schema_hash,
        engine_id=tags.engine_id,
        engine_version=tags.engine_version,
        action_set_hash=tags.action_set_hash,
        training_step=training_step,
        run_id=run_id,
    )
