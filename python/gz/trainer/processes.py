from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass

from gz.trainer.config import RunConfig
from gz.trainer.sampler import SampleClient
from gz.trainer.telemetry import SelfplayStatsTracker, pump_selfplay_stderr


def init_replay(config: RunConfig) -> None:
    subprocess.run(
        [
            config.paths.graphzero_bin,
            "replay-init",
            "--replay-dir",
            str(config.paths.replay_dir),
            "--max-candidates",
            str(config.selfplay.max_candidates),
            "--mask-stop",
            "true" if config.selfplay.mask_stop else "false",
        ],
        check=True,
    )


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
            str(max(config.trainer.batch, config.trainer.value_batch)),
        ]
    )


def spawn_torch_selfplay(config: RunConfig) -> subprocess.Popen[bytes]:
    measurement_args: list[str] = []
    if config.measurement.enabled:
        measurement_args = [
            "--measure-listen",
            config.measurement.listen,
            "--measure-server-cert",
            config.measurement.server_cert,
            "--measure-server-key",
            config.measurement.server_key,
            "--measure-client-ca",
            config.measurement.client_ca,
            "--measure-profile",
            config.measurement.profile,
            "--measure-receipt-dir",
            config.measurement.receipt_dir,
            "--measure-startup-timeout-ms",
            str(config.measurement.startup_timeout_ms),
        ]
        for agent in config.measurement.agents:
            measurement_args.extend(("--measure-agent", agent))
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
            "--position-features",
            "true" if config.selfplay.position_features else "false",
            "--no-backtrack",
            "true" if config.selfplay.no_backtrack else "false",
            "--gumbel-noise-overlap",
            str(config.selfplay.gumbel_noise_overlap),
            "--tree-reuse",
            "true" if config.selfplay.tree_reuse else "false",
            "--mask-stop",
            "true" if config.selfplay.mask_stop else "false",
            "--eval-processes",
            str(config.selfplay.eval_processes),
            "--admission-stagger-ms",
            str(config.selfplay.admission_stagger_ms),
            "--admission-smoothing",
            "true" if config.selfplay.admission_smoothing else "false",
            "--evaluator",
            "torch",
            "--python-dir",
            config.selfplay.python_dir,
            "--checkpoint-dir",
            str(config.paths.actor_checkpoint_dir),
            "--checkpoint-pointer",
            config.selfplay.actor_checkpoint_pointer,
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
            "--c-visit",
            str(config.selfplay.c_visit),
            "--c-scale",
            str(config.selfplay.c_scale),
            "--max-candidates",
            str(config.selfplay.max_candidates),
            "--max-batch",
            str(config.selfplay.max_batch),
            "--serve-socket",
            str(config.paths.sample_socket),
            # Sampled GZFB/GZFT batches are encoded at the serve capacity, and
            # the trainer stages at trainer.batch - they must be one knob.
            "--serve-max-batch",
            str(config.trainer.batch),
            "--replay-backlog",
            str(config.selfplay.max_row_backlog),
            *(
                ["--replay-retain", str(config.selfplay.replay_retain)]
                if config.selfplay.replay_retain
                else []
            ),
            *measurement_args,
        ],
        # Selfplay spawns the evaluator child; a new session lets kill_child
        # take down the whole group instead of orphaning the evaluator.
        start_new_session=True,
        stderr=subprocess.PIPE,
    )


@dataclass(slots=True)
class SelfplayStage:
    child: subprocess.Popen[bytes]
    pump: threading.Thread
    stats: SelfplayStatsTracker
    sampler: SampleClient

    @classmethod
    def start(cls, config: RunConfig) -> SelfplayStage:
        child = spawn_torch_selfplay(config)
        stats = SelfplayStatsTracker()
        pump = threading.Thread(
            target=pump_selfplay_stderr,
            args=(child, stats),
            daemon=True,
        )
        pump.start()
        sampler = SampleClient(
            config.paths.sample_socket,
            startup_timeout=config.trainer.startup_timeout,
            reconnect_limit=config.trainer.reconnect_limit,
        )
        try:
            sampler.wait_until_ready(
                config.trainer.min_startup_rows,
                alive_check=lambda: check_child(child, "selfplay"),
            )
        except BaseException:
            sampler.close()
            kill_child(child)
            pump.join(timeout=5.0)
            raise
        return cls(child=child, pump=pump, stats=stats, sampler=sampler)

    def terminate(self) -> None:
        kill_child(self.child)

    def close(self) -> None:
        self.sampler.close()
        self.pump.join(timeout=5.0)


def check_memory(min_available_gb: float) -> None:
    """Abort before memory pressure can freeze a swapless host."""
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
        # Children spawned with start_new_session lead their own group; kill
        # the group so their evaluator children die too.
        if os.getpgid(child.pid) == child.pid:
            os.killpg(child.pid, signal.SIGKILL)
        elif child.poll() is None:
            child.send_signal(signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()
