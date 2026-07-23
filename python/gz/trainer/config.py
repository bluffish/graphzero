from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

from gz.model.exphormer import ArchConfig


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    lr: float = 3e-4
    lr_schedule: str = "cosine"
    warmup_steps: int = 200
    lr_decay_steps: int | None = None
    min_lr_ratio: float = 0.0
    batch: int = 256
    # Zero shares the policy batch, preserving the historical trainer path.
    # A positive value samples and stages an independent value batch.
    value_batch: int = 0
    window_rows: int = 200_000
    # Zero inherits window_rows. This counts source rows; paired value
    # training evaluates both orientations of every sampled source row.
    value_window_rows: int = 0
    total_steps: int = 1000
    publish_interval: int = 500
    # Newest ordinary actor checkpoints to retain. Named checkpoint pointers
    # and in-flight arena challengers are retained in addition; zero disables.
    checkpoint_retain: int = 0
    # Publish and permanently pin an exact checkpoint at each positive
    # multiple of this many optimizer steps. Zero disables milestone pins.
    permanent_checkpoint_interval: int = 1000
    # Hold each periodic checkpoint until the next training gate. A one-block
    # lag matches whittlezero's overlapped actor snapshot schedule.
    publish_lag_blocks: int = 0
    value_weight: float = 1.0
    # Scale value gradients entering the shared trunk while leaving the
    # private value head's gradients unchanged.
    value_trunk_grad_scale: float = 1.0
    value_final_weight: float = 1.0
    value_v8_weight: float = 0.0
    value_v32_weight: float = 0.0
    terminal_score_weight: float = 0.0
    soft_policy_weight: float = 0.0
    soft_policy_temperature: float = 4.0
    # Scale the auxiliary policy gradient entering its shared policy/trunk
    # representation without weakening its private readout update.
    soft_policy_trunk_grad_scale: float = 1.0
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    adamw_lr: float | None = None
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    policy_init: str = "default"
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    min_startup_rows: int = 256
    # Optional experiment controls. By default the legacy seed drives both
    # Torch (initialization/dropout) and replay/value-orientation sampling.
    seed: int = 0
    model_seed: int | None = None
    data_seed: int | None = None
    device: str = "cuda:1"
    startup_timeout: float = 60.0
    reconnect_limit: int = 5
    log_interval: int = 1
    step_sleep: float = 0.0
    min_available_gb: float = 40.0
    # Sample batch N+1 on a background thread while the GPU trains batch N,
    # taking the socket read/decode off the step critical path. Off = the
    # historical strictly-serial loop, kept for A/B comparison.
    prefetch: bool = True
    # Sample policy and value batches on separate replay connections. Off
    # keeps GPU prefetching but issues both requests sequentially on one client.
    parallel_value_sampling: bool = True
    # Compile static-shape model forward/backward graphs with TorchInductor.
    # Optimizer, EMA, and checkpoints continue to own the original module.
    compile_model: bool = False
    compile_mode: str = "default"
    matmul_precision: str = "highest"
    # Pace the trainer against fresh production: each gate waits until enough
    # source rows exist for its cumulative policy samples. Zero disables.
    max_reuse: float = 0.0
    # Number of optimizer steps admitted together by max_reuse. One is the
    # historical streaming gate; whittlezero admits eight after each wave.
    reuse_gate_interval: int = 1
    # Completed episodes required per admitted block. Zero disables the
    # episode-count gate; this preserves fixed actor-wave cadence when
    # episode lengths change.
    reuse_gate_episodes: int = 0
    # Continue an interrupted run in place: load the latest
    # published checkpoint (EMA weights seed both the live model and the
    # EMA -- an approximate resume; optimizer moments restart), and start
    # the step counter at the checkpoint's training_step.
    resume: bool = False
    # Seed a new run from another checkpoint directory. Only model weights
    # transfer; optimizer, EMA, counters, and training step restart at zero.
    init_checkpoint: str = ""
    # "all" restores every model tensor. "policy" transfers the trunk and
    # policy while preserving this run's freshly initialized value module.
    init_checkpoint_scope: str = "all"


@dataclass(frozen=True, slots=True)
class SelfplayConfig:
    lanes: int = 2
    workers_per_lane: int = 8
    simulations: int = 8
    max_considered: int = 8
    gumbel_scale: float = 1.0
    c_visit: float = 50.0
    c_scale: float = 1.0
    max_steps: int = 8
    max_candidates: int = 255
    max_row_backlog: int = 200_000
    replay_retain: int = 0
    eval_device: str = "cuda:0"
    eval_poll_interval: float = 10.0
    actor_checkpoint_pointer: str = "latest.json"
    seed: int = 0
    max_batch: int = 16
    python_dir: str = "python"
    position_features: bool = True
    no_backtrack: bool = True
    gumbel_noise_overlap: float = 0.5
    mask_stop: bool = False
    tree_reuse: bool = True
    eval_processes: int = 1
    admission_stagger_ms: int = 0
    admission_smoothing: bool = False


@dataclass(frozen=True, slots=True)
class MeasurementConfig:
    enabled: bool = False
    listen: str = "0.0.0.0:50051"
    server_cert: str = ""
    server_key: str = ""
    client_ca: str = ""
    agents: tuple[str, ...] = ()
    profile: str = ""
    receipt_dir: str = ""
    startup_timeout_ms: int = 60_000


@dataclass(frozen=True, slots=True)
class WandbConfig:
    project: str = ""
    entity: str = ""
    run_name: str = ""
    mode: str = ""
    # Resume this wandb run id in place (wandb.init(resume="must")).
    run_id: str = ""


@dataclass(frozen=True, slots=True)
class PathsConfig:
    replay_dir: Path
    checkpoint_dir: Path
    actor_checkpoint_dir: Path
    run_dir: Path
    sample_socket: Path
    graphzero_bin: str


@dataclass(frozen=True, slots=True)
class RunConfig:
    trainer: TrainerConfig
    selfplay: SelfplayConfig
    paths: PathsConfig
    wandb: WandbConfig
    arch: ArchConfig
    measurement: MeasurementConfig


def resolved_trainer_seeds(config: TrainerConfig) -> tuple[int, int]:
    model_seed = config.seed if config.model_seed is None else config.model_seed
    data_seed = config.seed if config.data_seed is None else config.data_seed
    return model_seed, data_seed


def _validate(config: RunConfig) -> RunConfig:
    if config.trainer.lr_schedule not in ("cosine", "constant"):
        raise ValueError(f"unknown lr_schedule: {config.trainer.lr_schedule}")
    if config.trainer.min_startup_rows < 1:
        raise ValueError("min_startup_rows must be at least 1")
    if config.trainer.publish_interval < 1:
        raise ValueError("publish_interval must be positive")
    if config.trainer.checkpoint_retain < 0:
        raise ValueError("checkpoint_retain must be non-negative")
    if config.trainer.permanent_checkpoint_interval < 0:
        raise ValueError("permanent_checkpoint_interval must be non-negative")
    if config.trainer.publish_lag_blocks not in (0, 1):
        raise ValueError("publish_lag_blocks must be 0 or 1")
    if config.trainer.batch < 1:
        raise ValueError("batch must be positive")
    if config.trainer.value_batch < 0:
        raise ValueError("value_batch must be non-negative")
    if config.trainer.window_rows < 1:
        raise ValueError("window_rows must be positive")
    if config.trainer.value_window_rows < 0:
        raise ValueError("value_window_rows must be non-negative")
    if not math.isfinite(config.trainer.value_trunk_grad_scale) or not (
        0.0 <= config.trainer.value_trunk_grad_scale <= 1.0
    ):
        raise ValueError("value_trunk_grad_scale must be finite and in [0, 1]")
    task_weights = (
        config.trainer.value_final_weight,
        config.trainer.value_v8_weight,
        config.trainer.value_v32_weight,
        config.trainer.terminal_score_weight,
    )
    if any(not math.isfinite(weight) or weight < 0.0 for weight in task_weights):
        raise ValueError("value task weights must be finite and non-negative")
    if not math.isclose(sum(task_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("value task weights must sum to one")
    auxiliary_weight = any(weight > 0.0 for weight in task_weights[1:])
    if auxiliary_weight and config.arch.auxiliary_heads not in {
        "v8-v32-score",
        "v8-v32-score-soft-policy-v2",
    }:
        raise ValueError("auxiliary task weights require v8-v32-score model heads")
    if (
        not math.isfinite(config.trainer.soft_policy_weight)
        or config.trainer.soft_policy_weight < 0.0
    ):
        raise ValueError("soft_policy_weight must be finite and non-negative")
    if (
        not math.isfinite(config.trainer.soft_policy_temperature)
        or config.trainer.soft_policy_temperature <= 1.0
    ):
        raise ValueError("soft_policy_temperature must be finite and greater than one")
    if not math.isfinite(config.trainer.soft_policy_trunk_grad_scale) or not (
        0.0 <= config.trainer.soft_policy_trunk_grad_scale <= 1.0
    ):
        raise ValueError("soft_policy_trunk_grad_scale must be finite and in [0, 1]")
    if (
        config.trainer.soft_policy_weight > 0.0
        and config.arch.auxiliary_heads != "v8-v32-score-soft-policy-v2"
    ):
        raise ValueError("soft_policy_weight requires a soft-policy model head")
    if config.trainer.compile_mode not in (
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    ):
        raise ValueError(f"unknown compile_mode: {config.trainer.compile_mode}")
    if config.trainer.matmul_precision not in ("highest", "high", "medium"):
        raise ValueError(
            f"unknown matmul_precision: {config.trainer.matmul_precision}"
        )
    if config.trainer.reuse_gate_interval < 1:
        raise ValueError("reuse_gate_interval must be positive")
    if config.trainer.reuse_gate_episodes < 0:
        raise ValueError("reuse_gate_episodes must be non-negative")
    if config.trainer.publish_lag_blocks and (
        config.trainer.reuse_gate_interval != config.trainer.publish_interval
        or (
            config.trainer.max_reuse == 0.0
            and config.trainer.reuse_gate_episodes == 0
        )
    ):
        raise ValueError(
            "publish_lag_blocks requires a publish-aligned reuse gate"
        )
    if not math.isfinite(config.trainer.max_reuse) or config.trainer.max_reuse < 0.0:
        raise ValueError("max_reuse must be finite and non-negative")
    if config.trainer.lr_decay_steps is not None and config.trainer.lr_decay_steps < 1:
        raise ValueError("lr_decay_steps must be positive")
    if not 0.0 <= config.trainer.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if config.trainer.optimizer not in ("adamw", "muon_mixed"):
        raise ValueError(f"unknown optimizer: {config.trainer.optimizer}")
    if config.trainer.adamw_lr is not None and config.trainer.adamw_lr <= 0.0:
        raise ValueError("adamw_lr must be positive")
    if not 0.0 <= config.trainer.momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    if config.trainer.ns_steps < 1:
        raise ValueError("ns_steps must be positive")
    if config.trainer.policy_init not in ("default", "neutral"):
        raise ValueError(f"unsupported policy_init: {config.trainer.policy_init}")
    if config.trainer.policy_init == "neutral" and config.arch.policy_head != "pointer":
        raise ValueError("policy_init = 'neutral' requires policy_head = 'pointer'")
    if config.trainer.resume and config.trainer.init_checkpoint:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    if config.trainer.init_checkpoint_scope not in ("all", "policy"):
        raise ValueError("init_checkpoint_scope must be 'all' or 'policy'")
    if not config.trainer.init_checkpoint and config.trainer.init_checkpoint_scope != "all":
        raise ValueError("init_checkpoint_scope requires init_checkpoint")
    for name, seed in (
        ("seed", config.trainer.seed),
        ("model_seed", config.trainer.model_seed),
        ("data_seed", config.trainer.data_seed),
    ):
        if seed is not None and not 0 <= seed < 2**64:
            raise ValueError(f"{name} must fit an unsigned 64-bit integer")
    if (
        not math.isfinite(config.selfplay.c_visit)
        or config.selfplay.c_visit < 0.0
        or not math.isfinite(config.selfplay.c_scale)
        or config.selfplay.c_scale < 0.0
    ):
        raise ValueError("c_visit and c_scale must be finite and non-negative")
    if (
        config.arch.position_encoding == "policy_budget"
        and not config.selfplay.position_features
    ):
        raise ValueError(
            f"position_encoding = '{config.arch.position_encoding}' requires position_features = true"
        )
    for name, value in (
        ("lanes", config.selfplay.lanes),
        ("workers_per_lane", config.selfplay.workers_per_lane),
        ("simulations", config.selfplay.simulations),
        ("max_considered", config.selfplay.max_considered),
        ("max_steps", config.selfplay.max_steps),
        ("max_candidates", config.selfplay.max_candidates),
        ("max_batch", config.selfplay.max_batch),
        ("eval_processes", config.selfplay.eval_processes),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if config.selfplay.eval_processes > config.selfplay.lanes:
        raise ValueError("eval_processes cannot exceed lanes")
    if config.selfplay.admission_stagger_ms < 0:
        raise ValueError("admission_stagger_ms must be non-negative")
    if config.selfplay.admission_smoothing and config.selfplay.admission_stagger_ms:
        raise ValueError(
            "admission_smoothing and admission_stagger_ms are mutually exclusive"
        )
    if config.selfplay.max_row_backlog < 1:
        raise ValueError("max_row_backlog must be positive")
    if config.selfplay.replay_retain < 0:
        raise ValueError("replay_retain must be non-negative")
    if (
        not math.isfinite(config.selfplay.gumbel_scale)
        or config.selfplay.gumbel_scale < 0.0
    ):
        raise ValueError("gumbel_scale must be finite and non-negative")
    if (
        not math.isfinite(config.selfplay.gumbel_noise_overlap)
        or config.selfplay.gumbel_noise_overlap >= 1.0
    ):
        raise ValueError("gumbel_noise_overlap must be < 1")
    if (
        not math.isfinite(config.selfplay.eval_poll_interval)
        or config.selfplay.eval_poll_interval < 0.0
    ):
        raise ValueError("eval_poll_interval must be finite and non-negative")
    actor_pointer = config.selfplay.actor_checkpoint_pointer
    if not actor_pointer or Path(actor_pointer).name != actor_pointer:
        raise ValueError("actor_checkpoint_pointer must be a checkpoint file name")
    if not 0 <= config.selfplay.seed < 2**64:
        raise ValueError("selfplay seed must fit an unsigned 64-bit integer")
    if not config.selfplay.mask_stop and not config.selfplay.position_features:
        raise ValueError("STOP-enabled symmetric selfplay requires position_features = true")
    if config.arch.state_input != "joint-board":
        raise ValueError("symmetric selfplay requires state_input = 'joint-board'")
    if config.arch.value_input != "single":
        raise ValueError("symmetric selfplay requires value_input = 'single'")
    if config.measurement.enabled:
        if not config.measurement.listen:
            raise ValueError("measurement.listen is required")
        for name, value in (
            ("server_cert", config.measurement.server_cert),
            ("server_key", config.measurement.server_key),
            ("client_ca", config.measurement.client_ca),
            ("profile", config.measurement.profile),
            ("receipt_dir", config.measurement.receipt_dir),
        ):
            if not value:
                raise ValueError(f"measurement.{name} is required")
        if not config.measurement.agents:
            raise ValueError("measurement.agents must contain at least one device")
        for agent in config.measurement.agents:
            device_id, separator, certificate = agent.partition("=")
            if (
                not separator
                or len(device_id) != 32
                or any(character not in "0123456789abcdefABCDEF" for character in device_id)
                or not certificate
            ):
                raise ValueError(
                    "measurement.agents entries must be DEVICE_ID=CERTIFICATE_PATH"
                )
        if config.measurement.startup_timeout_ms < 1:
            raise ValueError("measurement.startup_timeout_ms must be positive")
    return config


def load_config(
    path: str | Path,
    *,
    extension_sections: frozenset[str] = frozenset(),
) -> RunConfig:
    data = load_config_table(Path(path))
    unknown_sections = (
        set(data)
        - {"trainer", "selfplay", "wandb", "arch", "paths", "measurement"}
        - extension_sections
    )
    if unknown_sections:
        raise ValueError(f"unknown config sections: {sorted(unknown_sections)}")
    trainer = dataclass_from_dict(
        TrainerConfig,
        data.get("trainer", {}),
    )
    selfplay = dataclass_from_dict(
        SelfplayConfig,
        data.get("selfplay", {}),
    )
    wandb = dataclass_from_dict(WandbConfig, data.get("wandb", {}))
    arch = ArchConfig.from_config_dict(data.get("arch", {}))
    measurement = dataclass_from_dict(
        MeasurementConfig,
        data.get("measurement", {}),
    )
    raw_paths = data.get("paths", {})
    if not isinstance(raw_paths, dict):
        raise ValueError("[paths] must be a table")
    known_paths = {
        "replay_dir",
        "checkpoint_dir",
        "actor_checkpoint_dir",
        "run_dir",
        "sample_socket",
        "graphzero_bin",
    }
    unknown_paths = set(raw_paths) - known_paths
    if unknown_paths:
        raise ValueError(f"unknown config fields for PathsConfig: {sorted(unknown_paths)}")
    run_dir = Path(str(raw_paths.get("run_dir", "runs/train-whittle")))
    replay_dir = Path(str(raw_paths.get("replay_dir", run_dir / "replay")))
    checkpoint_dir = Path(str(raw_paths.get("checkpoint_dir", run_dir / "checkpoints")))
    actor_checkpoint_dir = Path(
        str(raw_paths.get("actor_checkpoint_dir", checkpoint_dir))
    )
    sample_socket = Path(str(raw_paths.get("sample_socket", run_dir / "sample.sock")))
    graphzero_bin = str(
        raw_paths.get("graphzero_bin", os.environ.get("GRAPHZERO_BIN", "graphzero"))
    )
    # Children run in their own working directories (the evaluator runs in
    # python_dir), so relative config paths must be pinned to the trainer's
    # cwd before they cross a process boundary.
    run_dir = run_dir.absolute()
    replay_dir = replay_dir.absolute()
    checkpoint_dir = checkpoint_dir.absolute()
    actor_checkpoint_dir = actor_checkpoint_dir.absolute()
    sample_socket = sample_socket.absolute()
    if measurement.enabled:
        agents = []
        for agent in measurement.agents:
            device_id, separator, certificate = agent.partition("=")
            agents.append(
                f"{device_id}{separator}{Path(certificate).absolute()}"
                if separator
                else agent
            )
        measurement = replace(
            measurement,
            server_cert=str(Path(measurement.server_cert).absolute()),
            server_key=str(Path(measurement.server_key).absolute()),
            client_ca=str(Path(measurement.client_ca).absolute()),
            agents=tuple(agents),
            receipt_dir=str(Path(measurement.receipt_dir).absolute()),
        )
    return _validate(
        RunConfig(
            trainer=trainer,
            selfplay=selfplay,
            paths=PathsConfig(
                replay_dir=replay_dir,
                checkpoint_dir=checkpoint_dir,
                actor_checkpoint_dir=actor_checkpoint_dir,
                run_dir=run_dir,
                sample_socket=sample_socket,
                graphzero_bin=graphzero_bin,
            ),
            wandb=wandb,
            arch=arch,
            measurement=measurement,
        )
    )


def load_config_table(path: Path) -> dict[str, object]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    extends = data.pop("extends", None)
    if extends is None:
        return data
    if not isinstance(extends, str):
        raise ValueError("extends must be a string")

    base_path = (path.parent / extends).resolve()
    base = tomllib.loads(base_path.read_text(encoding="utf-8"))
    if "extends" in base:
        raise ValueError("config inheritance is limited to one layer")
    return _merge_config_tables(base, data)


def _merge_config_tables(base: dict[str, object], child: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in child.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_config_tables(base_value, value)
        else:
            merged[key] = value
    return merged


def dataclass_from_dict[T](cls: type[T], data: object) -> T:
    if not isinstance(data, dict):
        raise ValueError("config section must be a table")
    known = {field.name for field in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config fields for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)
