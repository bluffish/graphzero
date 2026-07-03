from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

from gz.model.exphormer import ArchConfig, build_model
from gz.trainer.data import TrainingStager
from gz.trainer.loop import LoopConfig, TrainerLoop
from gz.trainer.publish import EmaWeights, publish_ema
from gz.trainer.sampler import SampleClient, step_seed


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    lr: float = 3e-4
    warmup_steps: int = 200
    batch: int = 256
    window_rows: int = 200_000
    total_steps: int = 1000
    publish_interval: int = 500
    value_weight: float = 1.0
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    min_startup_rows: int = 256
    seed: int = 0
    device: str = "cuda:1"
    startup_timeout: float = 60.0
    reconnect_limit: int = 5
    log_interval: int = 1
    step_sleep: float = 0.0
    bootstrap_episodes: int = 64


@dataclass(frozen=True, slots=True)
class SelfplayConfig:
    lanes: int = 2
    workers_per_lane: int = 8
    simulations: int = 8
    max_steps: int = 8
    reference: str = "self-average"
    reference_ema_decay: float = 0.99
    max_row_backlog: int = 200_000
    eval_device: str = "cuda:0"
    eval_poll_interval: float = 10.0
    seed: int = 0
    max_batch: int = 16
    python_dir: str = "python"


@dataclass(frozen=True, slots=True)
class PathsConfig:
    replay_dir: Path
    checkpoint_dir: Path
    run_dir: Path
    sample_socket: Path
    graphzero_bin: str


@dataclass(frozen=True, slots=True)
class RunConfig:
    trainer: TrainerConfig
    selfplay: SelfplayConfig
    paths: PathsConfig


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    for path in (config.paths.replay_dir, config.paths.checkpoint_dir, config.paths.run_dir):
        path.mkdir(parents=True, exist_ok=True)
    metrics = MetricsWriter(config.paths.run_dir / "metrics.jsonl")

    bootstrap_selfplay(config)
    serve = spawn_replay_serve(config)
    try:
        sampler = SampleClient(
            config.paths.sample_socket,
            startup_timeout=config.trainer.startup_timeout,
            reconnect_limit=config.trainer.reconnect_limit,
        )
        sampler.wait_until_ready(
            config.trainer.min_startup_rows,
            alive_check=lambda: check_child(serve, "replay-serve"),
        )
        arch = ArchConfig()
        model = build_model(sampler.feature_schema, arch).to(config.trainer.device)
        ema = EmaWeights(model, config.trainer.ema_decay)
        first = publish_ema(
            config.paths.checkpoint_dir,
            ema,
            schema=sampler.feature_schema,
            schema_hash=sampler.feature_schema_hash,
            arch=arch,
            training_step=0,
            run_id=config.paths.run_dir.name,
        )
        metrics.write({"event": "publish", "training_step": 0, "model_version": first.model_version.hex()})
    finally:
        stop_child(serve)

    selfplay = spawn_torch_selfplay(config)
    try:
        sampler = SampleClient(
            config.paths.sample_socket,
            startup_timeout=config.trainer.startup_timeout,
            reconnect_limit=config.trainer.reconnect_limit,
        )
        sampler.wait_until_ready(
            config.trainer.min_startup_rows,
            alive_check=lambda: check_child(selfplay, "selfplay"),
        )
        stager = TrainingStager(sampler.feature_schema, config.trainer.batch, config.trainer.device)
        loop = TrainerLoop(
            model,
            LoopConfig(
                lr=config.trainer.lr,
                warmup_steps=config.trainer.warmup_steps,
                total_steps=config.trainer.total_steps,
                value_weight=config.trainer.value_weight,
                grad_clip=config.trainer.grad_clip,
            ),
        )
        for step in range(config.trainer.total_steps):
            check_child(selfplay, "selfplay")
            result = sampler.sample(
                config.trainer.batch,
                config.trainer.window_rows,
                step_seed(config.trainer.seed, step),
            )
            metrics_record = loop.train_step(stager.copy(result.batch, result.targets))
            ema.update(model)
            if step % config.trainer.log_interval == 0:
                produced = sampler.refresh().produced_rows
                metrics.write(
                    {
                        "event": "step",
                        "timestamp": time.time(),
                        "step": metrics_record.step,
                        "policy_loss": metrics_record.policy_loss,
                        "value_loss": metrics_record.value_loss,
                        "loss": metrics_record.loss,
                        "grad_norm": metrics_record.grad_norm,
                        "lr": metrics_record.lr,
                        "value_accuracy": metrics_record.value_accuracy,
                        "fraction_valid": metrics_record.fraction_valid,
                        "label_mean": metrics_record.label_mean,
                        "max_reward": metrics_record.max_reward,
                        "produced_rows": produced,
                        "samples_per_row": ((step + 1) * config.trainer.batch / produced) if produced else 0.0,
                    }
                )
            if (step + 1) % config.trainer.publish_interval == 0:
                manifest = publish_ema(
                    config.paths.checkpoint_dir,
                    ema,
                    schema=sampler.feature_schema,
                    schema_hash=sampler.feature_schema_hash,
                    arch=arch,
                    training_step=step + 1,
                    run_id=config.paths.run_dir.name,
                )
                metrics.write(
                    {
                        "event": "publish",
                        "training_step": step + 1,
                        "model_version": manifest.model_version.hex(),
                    }
                )
            if config.trainer.step_sleep:
                time.sleep(config.trainer.step_sleep)
        if config.trainer.total_steps % config.trainer.publish_interval != 0:
            final = publish_ema(
                config.paths.checkpoint_dir,
                ema,
                schema=sampler.feature_schema,
                schema_hash=sampler.feature_schema_hash,
                arch=arch,
                training_step=config.trainer.total_steps,
                run_id=config.paths.run_dir.name,
            )
            metrics.write(
                {
                    "event": "publish",
                    "training_step": config.trainer.total_steps,
                    "model_version": final.model_version.hex(),
                }
            )
    except BaseException:
        kill_child(selfplay)
        raise
    else:
        kill_child(selfplay)


class MetricsWriter:
    def __init__(self, path: Path) -> None:
        self.handle = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, object]) -> None:
        self.handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self.handle.flush()


def load_config(path: str | Path) -> RunConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    trainer = _dataclass_from_dict(TrainerConfig, data.get("trainer", {}))
    selfplay = _dataclass_from_dict(SelfplayConfig, data.get("selfplay", {}))
    raw_paths = data.get("paths", {})
    if not isinstance(raw_paths, dict):
        raise ValueError("[paths] must be a table")
    run_dir = Path(str(raw_paths.get("run_dir", "runs/train-whittle")))
    replay_dir = Path(str(raw_paths.get("replay_dir", run_dir / "replay")))
    checkpoint_dir = Path(str(raw_paths.get("checkpoint_dir", run_dir / "checkpoints")))
    sample_socket = Path(str(raw_paths.get("sample_socket", run_dir / "sample.sock")))
    graphzero_bin = str(raw_paths.get("graphzero_bin", os.environ.get("GRAPHZERO_BIN", "graphzero")))
    return RunConfig(
        trainer=trainer,
        selfplay=selfplay,
        paths=PathsConfig(
            replay_dir=replay_dir,
            checkpoint_dir=checkpoint_dir,
            run_dir=run_dir,
            sample_socket=sample_socket,
            graphzero_bin=graphzero_bin,
        ),
    )


def bootstrap_selfplay(config: RunConfig) -> None:
    command = [
        config.paths.graphzero_bin,
        "selfplay",
        "--replay-dir",
        str(config.paths.replay_dir),
        "--episodes",
        str(config.trainer.bootstrap_episodes),
        "--lanes",
        str(config.selfplay.lanes),
        "--workers-per-lane",
        str(config.selfplay.workers_per_lane),
        "--reference",
        "root",
        "--evaluator",
        "stub",
        "--seed",
        str(config.selfplay.seed),
        "--max-steps",
        str(config.selfplay.max_steps),
        "--simulations",
        str(config.selfplay.simulations),
        "--max-batch",
        str(config.selfplay.max_batch),
    ]
    subprocess.run(command, check=True)


def spawn_replay_serve(config: RunConfig) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            config.paths.graphzero_bin,
            "replay-serve",
            "--replay-dir",
            str(config.paths.replay_dir),
            "--socket",
            str(config.paths.sample_socket),
            "--max-batch",
            str(config.trainer.batch),
        ]
    )


def spawn_torch_selfplay(config: RunConfig) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            config.paths.graphzero_bin,
            "selfplay",
            "--replay-dir",
            str(config.paths.replay_dir),
            "--episodes",
            "0",
            "--lanes",
            str(config.selfplay.lanes),
            "--workers-per-lane",
            str(config.selfplay.workers_per_lane),
            "--reference",
            config.selfplay.reference,
            "--reference-ema-decay",
            str(config.selfplay.reference_ema_decay),
            "--evaluator",
            "torch",
            "--python-dir",
            config.selfplay.python_dir,
            "--checkpoint-dir",
            str(config.paths.checkpoint_dir),
            "--eval-device",
            config.selfplay.eval_device,
            "--eval-poll-interval",
            str(config.selfplay.eval_poll_interval),
            "--seed",
            str(config.selfplay.seed),
            "--max-steps",
            str(config.selfplay.max_steps),
            "--simulations",
            str(config.selfplay.simulations),
            "--max-batch",
            str(config.selfplay.max_batch),
            "--serve-socket",
            str(config.paths.sample_socket),
            # Sampled GZFB/GZFT batches are encoded at the serve capacity, and
            # the trainer stages at trainer.batch — they must be one knob.
            "--serve-max-batch",
            str(config.trainer.batch),
            "--replay-backlog",
            str(config.selfplay.max_row_backlog),
        ],
        # Selfplay spawns the evaluator child; a new session lets kill_child
        # take down the whole group instead of orphaning the evaluator (and
        # its GPU memory) when selfplay is SIGKILLed.
        start_new_session=True,
    )


def check_child(child: subprocess.Popen[bytes], name: str) -> None:
    status = child.poll()
    if status is not None:
        raise RuntimeError(f"{name} exited with status {status}")


def stop_child(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        kill_child(child)


def kill_child(child: subprocess.Popen[bytes]) -> None:
    try:
        # Children spawned with start_new_session lead their own group;
        # kill the group so their own children (the evaluator) die too.
        if os.getpgid(child.pid) == child.pid:
            os.killpg(child.pid, signal.SIGKILL)
        elif child.poll() is None:
            child.send_signal(signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def _dataclass_from_dict(cls: object, data: object) -> object:
    if not isinstance(data, dict):
        raise ValueError("config section must be a table")
    fields = cls.__dataclass_fields__
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"unknown config fields for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)
