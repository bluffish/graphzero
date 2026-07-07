from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from gz.model.exphormer import ArchConfig, build_model
from gz.trainer.data import TrainingStager
from gz.trainer.loop import LoopConfig, TrainerLoop
from gz.trainer.publish import EmaWeights, publish_ema
from gz.trainer.sampler import SampleClient, step_seed


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    lr: float = 3e-4
    lr_schedule: str = "cosine"
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
    min_available_gb: float = 40.0
    # Sample batch N+1 on a background thread while the GPU trains batch N,
    # taking the socket read/decode off the step critical path. Off = the
    # historical strictly-serial loop, kept for A/B comparison.
    prefetch: bool = True
    # Continue an interrupted run in place: skip bootstrap, load the latest
    # published checkpoint (EMA weights seed both the live model and the
    # EMA -- an approximate resume; optimizer moments restart), and start
    # the step counter at the checkpoint's training_step.
    resume: bool = False


@dataclass(frozen=True, slots=True)
class SelfplayConfig:
    lanes: int = 2
    workers_per_lane: int = 8
    simulations: int = 8
    max_considered: int = 16
    gumbel_scale: float = 0.0
    max_steps: int = 8
    max_candidates: int = 255
    reference: str = "self-average"
    root_mode: str = "generated"
    reference_ema_decay: float = 0.99
    max_row_backlog: int = 200_000
    replay_retain: int = 0
    eval_device: str = "cuda:0"
    eval_poll_interval: float = 10.0
    seed: int = 0
    max_batch: int = 16
    python_dir: str = "python"
    # Export real position features to evals/rows; off = graph + opponent only.
    position_features: bool = True
    # Evaluator server processes; lanes stripe across them (torch only).
    eval_processes: int = 1


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


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    for path in (config.paths.replay_dir, config.paths.checkpoint_dir, config.paths.run_dir):
        path.mkdir(parents=True, exist_ok=True)
    metrics = MetricsWriter(config.paths.run_dir / "metrics.jsonl", WandbRun.start(config))

    arch = config.arch
    model = None
    ema = None
    published_snapshot = None
    resume_start = 0
    if not config.trainer.resume:
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
            param_norm, _ = ema.norms(None)
            published_snapshot = ema.state_dict()
            metrics.write(
                {
                    "event": "publish",
                    "training_step": 0,
                    "model_version": first.model_version.hex(),
                    "param_norm": param_norm,
                    "update_norm": 0.0,
                }
            )
        finally:
            stop_child(serve)

    selfplay = spawn_torch_selfplay(config)
    try:
        sampler = SampleClient(
            config.paths.sample_socket,
            startup_timeout=config.trainer.startup_timeout,
            reconnect_limit=config.trainer.reconnect_limit,
        )
        ready_ack = sampler.wait_until_ready(
            config.trainer.min_startup_rows,
            alive_check=lambda: check_child(selfplay, "selfplay"),
        )
        if ready_ack.root is not None and not config.trainer.resume:
            metrics.write(
                {
                    "event": "graph",
                    "root_cost": ready_ack.root.cost,
                    "root_nodes": ready_ack.root.node_count,
                    "root_edges": ready_ack.root.edge_count,
                    "root_candidates": ready_ack.root.candidate_count,
                }
            )
        if config.trainer.resume:
            from gz.checkpoints import DirectorySource
            from gz.checkpoints.weights import load_state_dict

            resolved = DirectorySource(str(config.paths.checkpoint_dir)).resolve_latest()
            if resolved.manifest.feature_schema_hash != sampler.feature_schema_hash:
                raise RuntimeError("resume checkpoint feature schema does not match the store")
            if ArchConfig.from_dict(resolved.manifest.arch_config) != arch:
                raise RuntimeError("resume checkpoint arch does not match [arch] config")
            model = build_model(sampler.feature_schema, arch).to(config.trainer.device)
            model.load_state_dict(load_state_dict(resolved.weights_path))
            ema = EmaWeights(model, config.trainer.ema_decay)
            published_snapshot = ema.state_dict()
            resume_start = resolved.manifest.training_step
            if resume_start >= config.trainer.total_steps:
                raise RuntimeError("resume checkpoint is at or past total_steps")
            metrics.write(
                {
                    "event": "resume",
                    "training_step": resume_start,
                    "model_version": resolved.manifest.model_version.hex(),
                }
            )
        stager = TrainingStager(sampler.feature_schema, config.trainer.batch, config.trainer.device)
        loop = TrainerLoop(
            model,
            LoopConfig(
                lr=config.trainer.lr,
                lr_schedule=config.trainer.lr_schedule,
                warmup_steps=config.trainer.warmup_steps,
                total_steps=config.trainer.total_steps,
                value_weight=config.trainer.value_weight,
                grad_clip=config.trainer.grad_clip,
                run_seed=config.trainer.seed,
            ),
        )
        loop.step_index = resume_start
        window = PerfWindow()
        prefetcher = None
        if config.trainer.prefetch:
            prefetcher = SamplePrefetcher(
                sampler,
                config.trainer.batch,
                config.trainer.window_rows,
                config.trainer.seed,
                config.trainer.total_steps,
                start_step=resume_start,
            )
            prefetcher.start()
        for step in range(resume_start, config.trainer.total_steps):
            check_child(selfplay, "selfplay")
            if step % 50 == 0:
                check_memory(config.trainer.min_available_gb)
            # With prefetch, sample_ms measures the wait for the queued
            # batch: ~0 while sampling keeps up, the residual stall when it
            # does not.
            sample_started = time.perf_counter()
            if prefetcher is not None:
                result = prefetcher.next()
            else:
                result = sampler.sample(
                    config.trainer.batch,
                    config.trainer.window_rows,
                    step_seed(config.trainer.seed, step),
                )
            train_started = time.perf_counter()
            # Metrics force a host-device sync; off-interval steps skip them
            # entirely so consecutive steps pipeline on the GPU.
            metrics_step = step % config.trainer.log_interval == 0
            metrics_record = loop.train_step(
                stager.copy(result.batch, result.targets), with_metrics=metrics_step
            )
            ema.update(model)
            window.record(sample_started, train_started, time.perf_counter())
            if metrics_step:
                assert metrics_record is not None
                ack = prefetcher.refresh() if prefetcher is not None else sampler.refresh()
                produced = ack.produced_rows
                stop_rate = ack.episodes_stopped / ack.episodes if ack.episodes else 0.0
                record = {
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
                    "terminal_cost_mean": metrics_record.terminal_cost_mean,
                    "terminal_cost_best": metrics_record.terminal_cost_best,
                    "produced_rows": produced,
                    "samples_per_row": ((step + 1) * config.trainer.batch / produced) if produced else 0.0,
                    "stop_rate": stop_rate,
                    "episode_cost_ema": ack.episode_cost_ema,
                    "episode_len_ema": ack.episode_len_ema,
                    "stop_rate_ema": ack.stop_rate_ema,
                    "best_cost": ack.best_cost,
                }
                # Outcome gauges are per-store-open; a zero means unseeded
                # (no episode appended by this selfplay process yet).
                if ack.root is not None and ack.episode_cost_ema > 0.0:
                    record["reduction_ema"] = ack.root.cost - ack.episode_cost_ema
                if ack.root is not None and ack.best_cost > 0.0:
                    record["reduction_best"] = ack.root.cost - ack.best_cost
                record.update(window.drain(produced, ack.episodes))
                metrics.write(record)
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
                param_norm, update_norm = ema.norms(published_snapshot)
                published_snapshot = ema.state_dict()
                metrics.write(
                    {
                        "event": "publish",
                        "training_step": step + 1,
                        "model_version": manifest.model_version.hex(),
                        "param_norm": param_norm,
                        "update_norm": update_norm,
                    }
                )
            if config.trainer.step_sleep:
                time.sleep(config.trainer.step_sleep)
        if prefetcher is not None:
            prefetcher.stop()
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
            param_norm, update_norm = ema.norms(published_snapshot)
            metrics.write(
                {
                    "event": "publish",
                    "training_step": config.trainer.total_steps,
                    "model_version": final.model_version.hex(),
                    "param_norm": param_norm,
                    "update_norm": update_norm,
                }
            )
    except BaseException:
        # wandb's atexit hook marks the run crashed; only the clean path
        # finishes it explicitly.
        kill_child(selfplay)
        raise
    else:
        kill_child(selfplay)
        metrics.finish()


class MetricsWriter:
    def __init__(self, path: Path, wandb_run: WandbRun | None = None) -> None:
        self.handle = path.open("a", encoding="utf-8")
        self.wandb_run = wandb_run

    def write(self, record: dict[str, object]) -> None:
        self.handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self.handle.flush()
        if self.wandb_run is not None:
            self.wandb_run.write(record)

    def finish(self) -> None:
        self.handle.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


class SamplePrefetcher:
    """Keeps one sample batch in flight on a background thread so the socket
    read and decode overlap GPU training. Owns all sampler socket use after
    start(): the internal lock serializes the sample loop against refresh(),
    which would otherwise interleave protocol frames on the shared stream.
    Errors raised while sampling surface on the consumer's next()."""

    def __init__(
        self,
        sampler: SampleClient,
        batch: int,
        window_rows: int,
        seed: int,
        total_steps: int,
        start_step: int = 0,
    ) -> None:
        self._sampler = sampler
        self._batch = batch
        self._window_rows = window_rows
        self._seed = seed
        self._total_steps = total_steps
        self._start_step = start_step
        # Depth 2 rides out replay-store read spikes (compaction bursts)
        # without letting sample timing drift more than two steps.
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sample-prefetch", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # Unblock a full queue so the thread can observe the stop flag.
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

    def next(self) -> object:
        result, error = self._queue.get()
        if error is not None:
            raise error
        return result

    def refresh(self) -> object:
        with self._lock:
            return self._sampler.refresh()

    def _run(self) -> None:
        for step in range(self._start_step, self._total_steps):
            if self._stop.is_set():
                return
            try:
                with self._lock:
                    result = self._sampler.sample(
                        self._batch,
                        self._window_rows,
                        step_seed(self._seed, step),
                    )
            except BaseException as error:  # surfaced on next()
                self._queue.put((None, error))
                return
            while not self._stop.is_set():
                try:
                    self._queue.put((result, None), timeout=1.0)
                    break
                except queue.Full:
                    continue


class PerfWindow:
    """Accumulates per-step timings between metric writes."""

    def __init__(self) -> None:
        self.window_started = time.perf_counter()
        self.last_produced = 0
        self.last_episodes = 0
        self.steps = 0
        self.sample_seconds = 0.0
        self.train_seconds = 0.0

    def record(self, sample_started: float, train_started: float, finished: float) -> None:
        self.steps += 1
        self.sample_seconds += train_started - sample_started
        self.train_seconds += finished - train_started

    def drain(self, produced: int, episodes: int) -> dict[str, float]:
        now = time.perf_counter()
        elapsed = max(now - self.window_started, 1e-9)
        steps = max(self.steps, 1)
        perf = {
            "steps_per_s": self.steps / elapsed,
            "rows_per_s": max(produced - self.last_produced, 0) / elapsed if self.last_produced else 0.0,
            "episodes_per_s": max(episodes - self.last_episodes, 0) / elapsed if self.last_episodes else 0.0,
            "sample_ms": 1000.0 * self.sample_seconds / steps,
            "train_ms": 1000.0 * self.train_seconds / steps,
        }
        self.window_started = now
        self.last_produced = produced
        self.last_episodes = episodes
        self.steps = 0
        self.sample_seconds = 0.0
        self.train_seconds = 0.0
        return perf


# JSONL keys -> grouped wandb keys. Anything unlisted stays out of wandb,
# which is the over-logging guard: extending the JSONL never widens wandb.
WANDB_KEYS = {
    "policy_loss": "train/policy_loss",
    "value_loss": "train/value_loss",
    "loss": "train/loss",
    "grad_norm": "train/grad_norm",
    "lr": "train/lr",
    "value_accuracy": "train/value_accuracy",
    "fraction_valid": "train/fraction_valid",
    "label_mean": "train/label_mean",
    "terminal_cost_mean": "selfplay/terminal_cost_mean",
    "terminal_cost_best": "selfplay/terminal_cost_best",
    "stop_rate": "selfplay/stop_rate",
    "episode_cost_ema": "selfplay/episode_cost_ema",
    "episode_len_ema": "selfplay/episode_len_ema",
    "stop_rate_ema": "selfplay/stop_rate_ema",
    "best_cost": "selfplay/best_cost",
    "reduction_ema": "graph/reduction_ema",
    "reduction_best": "graph/reduction_best",
    "steps_per_s": "perf/steps_per_s",
    "rows_per_s": "perf/rows_per_s",
    "episodes_per_s": "perf/episodes_per_s",
    "sample_ms": "perf/sample_ms",
    "train_ms": "perf/train_ms",
    "produced_rows": "perf/produced_rows",
    "samples_per_row": "perf/samples_per_row",
}


class WandbRun:
    """Optional wandb mirror of the metrics JSONL. Never load-bearing:
    init failure logs one line and the run proceeds without it."""

    def __init__(self, run: object) -> None:
        self.run = run
        self.publishes = 0

    @classmethod
    def start(cls, config: RunConfig) -> WandbRun | None:
        if not config.wandb.project:
            return None
        try:
            import wandb

            run = wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity or None,
                name=config.wandb.run_name or config.paths.run_dir.name,
                mode=config.wandb.mode or None,
                id=config.wandb.run_id or None,
                resume="must" if config.wandb.run_id else None,
                # A resumed run keeps its original config; re-sending it
                # would conflict on any knob the resume changed.
                config=None
                if config.wandb.run_id
                else {
                    "trainer": asdict(config.trainer),
                    "selfplay": asdict(config.selfplay),
                    "arch": asdict(config.arch),
                    "run_dir": str(config.paths.run_dir),
                },
            )
        except Exception as error:
            print(f"event=wandb_disabled error={error}", file=sys.stderr, flush=True)
            return None
        return cls(run)

    def write(self, record: dict[str, object]) -> None:
        if record.get("event") == "step":
            payload = {WANDB_KEYS[k]: v for k, v in record.items() if k in WANDB_KEYS}
            self.run.log(payload, step=record["step"])
        elif record.get("event") == "graph":
            facts = {k: v for k, v in record.items() if k != "event"}
            self.run.config.update({"graph": facts}, allow_val_change=True)
            self.run.log({f"graph/{k}": v for k, v in facts.items()}, step=0)
        elif record.get("event") == "publish":
            self.publishes += 1
            payload = {
                "publish/count": self.publishes,
                "publish/training_step": record["training_step"],
            }
            for key in ("param_norm", "update_norm"):
                if key in record:
                    payload[f"publish/{key}"] = record[key]
            self.run.log(payload, step=record["training_step"])

    def finish(self) -> None:
        self.run.finish()


def _validate(config: RunConfig) -> RunConfig:
    if config.trainer.lr_schedule not in ("cosine", "constant"):
        raise ValueError(f"unknown lr_schedule: {config.trainer.lr_schedule}")
    ceiling = config.trainer.bootstrap_episodes * config.selfplay.max_steps
    if ceiling < config.trainer.min_startup_rows:
        raise ValueError(
            f"bootstrap_episodes x max_steps ({ceiling}) cannot reach "
            f"min_startup_rows ({config.trainer.min_startup_rows}); startup would hang"
        )
    return config


def load_config(path: str | Path) -> RunConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    trainer = _dataclass_from_dict(TrainerConfig, data.get("trainer", {}))
    selfplay = _dataclass_from_dict(SelfplayConfig, data.get("selfplay", {}))
    wandb = _dataclass_from_dict(WandbConfig, data.get("wandb", {}))
    arch = _dataclass_from_dict(ArchConfig, data.get("arch", {}))
    raw_paths = data.get("paths", {})
    if not isinstance(raw_paths, dict):
        raise ValueError("[paths] must be a table")
    run_dir = Path(str(raw_paths.get("run_dir", "runs/train-whittle")))
    replay_dir = Path(str(raw_paths.get("replay_dir", run_dir / "replay")))
    checkpoint_dir = Path(str(raw_paths.get("checkpoint_dir", run_dir / "checkpoints")))
    sample_socket = Path(str(raw_paths.get("sample_socket", run_dir / "sample.sock")))
    graphzero_bin = str(raw_paths.get("graphzero_bin", os.environ.get("GRAPHZERO_BIN", "graphzero")))
    # Children run in their own working directories (the evaluator runs in
    # python_dir), so relative config paths must be pinned to the trainer's
    # cwd before they cross a process boundary.
    run_dir = run_dir.absolute()
    replay_dir = replay_dir.absolute()
    checkpoint_dir = checkpoint_dir.absolute()
    sample_socket = sample_socket.absolute()
    return _validate(RunConfig(
        trainer=trainer,
        selfplay=selfplay,
        paths=PathsConfig(
            replay_dir=replay_dir,
            checkpoint_dir=checkpoint_dir,
            run_dir=run_dir,
            sample_socket=sample_socket,
            graphzero_bin=graphzero_bin,
        ),
        wandb=wandb,
        arch=arch,
    ))


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
        "--root-mode",
        config.selfplay.root_mode,
        "--evaluator",
        "stub",
        "--seed",
        str(config.selfplay.seed),
        "--max-steps",
        str(config.selfplay.max_steps),
        "--simulations",
        str(config.selfplay.simulations),
        "--max-considered",
        str(config.selfplay.max_considered),
        "--gumbel-scale",
        str(config.selfplay.gumbel_scale),
        "--max-candidates",
        str(config.selfplay.max_candidates),
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
            "--root-mode",
            config.selfplay.root_mode,
            "--reference-ema-decay",
            str(config.selfplay.reference_ema_decay),
            "--position-features",
            "true" if config.selfplay.position_features else "false",
            "--eval-processes",
            str(config.selfplay.eval_processes),
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
            "--max-considered",
            str(config.selfplay.max_considered),
            "--gumbel-scale",
            str(config.selfplay.gumbel_scale),
            "--max-candidates",
            str(config.selfplay.max_candidates),
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
            *(
                ["--replay-retain", str(config.selfplay.replay_retain)]
                if config.selfplay.replay_retain
                else []
            ),
        ],
        # Selfplay spawns the evaluator child; a new session lets kill_child
        # take down the whole group instead of orphaning the evaluator (and
        # its GPU memory) when selfplay is SIGKILLed.
        start_new_session=True,
    )


def check_memory(min_available_gb: float) -> None:
    """Aborts the run before memory pressure can freeze a swapless box:
    the kernel thrashes long before the OOM killer fires."""
    if min_available_gb <= 0:
        return
    available = _mem_available_gb()
    if available is not None and available < min_available_gb:
        raise RuntimeError(
            f"aborting: {available:.1f} GiB available < {min_available_gb} GiB floor"
        )


def _mem_available_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        return None
    return None


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
