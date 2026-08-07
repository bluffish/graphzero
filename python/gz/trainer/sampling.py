from __future__ import annotations

import math
import queue
import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

from gz.trainer.sampler import SampleClient, step_seed


def sample_window_rows(window_rows: int, produced_rows: int) -> int:
    return max(1, min(int(window_rows), int(produced_rows)))


def cumulative_reuse(step: int, batch: int, produced_rows: int) -> float:
    if batch <= 0 or produced_rows <= 0:
        return 0.0
    return (step + 1) * batch / produced_rows


def required_produced_rows(
    step: int,
    batch: int,
    max_reuse: float,
    gate_interval: int,
) -> int:
    if max_reuse <= 0.0:
        return 0
    block_end = (step // gate_interval + 1) * gate_interval
    return math.ceil(block_end * batch / max_reuse)


def required_episodes(step: int, gate_interval: int, episodes_per_gate: int) -> int:
    if episodes_per_gate <= 0:
        return 0
    block = step // gate_interval + 1
    return block * episodes_per_gate


@dataclass(frozen=True, slots=True)
class SampledBatches:
    policy: object
    value: object | None


def sample_training_batches(
    sampler: SampleClient,
    *,
    policy_batch: int,
    policy_window_rows: int,
    value_batch: int,
    value_window_rows: int,
    value_terminal_fraction: float,
    run_seed: int,
    step: int,
    produced_rows: int,
    value_sampler: SampleClient | None = None,
    value_executor: Executor | None = None,
) -> SampledBatches:
    if value_executor is not None and value_sampler is None:
        raise ValueError("parallel value sampling requires a separate sample client")
    value_future = None
    terminal_value_batch = _select_terminal_value_batch(
        run_seed, step, value_terminal_fraction
    )
    if value_batch > 0 and value_executor is not None:
        assert value_sampler is not None
        value_future = value_executor.submit(
            partial(
                value_sampler.sample,
                value_batch,
                sample_window_rows(value_window_rows, produced_rows),
                step_seed(run_seed, step, "value-sample"),
                terminal_only=terminal_value_batch,
            )
        )
    policy = sampler.sample(
        policy_batch,
        sample_window_rows(policy_window_rows, produced_rows),
        step_seed(run_seed, step),
    )
    value = None
    if value_batch > 0:
        if value_future is not None:
            value = value_future.result()
        else:
            value = sampler.sample(
                value_batch,
                sample_window_rows(value_window_rows, produced_rows),
                step_seed(run_seed, step, "value-sample"),
                terminal_only=terminal_value_batch,
            )
    return SampledBatches(policy=policy, value=value)


def _select_terminal_value_batch(run_seed: int, step: int, fraction: float) -> bool:
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    draw = step_seed(run_seed, step, "value-terminal") / float(1 << 64)
    return draw < fraction


class SamplePrefetcher:
    """Keeps one training pair in flight while the GPU trains.

    The policy client remains shared with refresh() under a lock. A distinct
    value client lets the replay service sample and collate both roles in
    parallel without interleaving frames on either socket.
    """

    def __init__(
        self,
        sampler: SampleClient,
        batch: int,
        window_rows: int,
        value_batch: int,
        value_window_rows: int,
        value_terminal_fraction: float,
        seed: int,
        total_steps: int,
        max_reuse: float,
        reuse_gate_interval: int,
        reuse_gate_episodes: int,
        start_step: int = 0,
        *,
        value_sampler: SampleClient | None = None,
    ) -> None:
        self._sampler = sampler
        self._batch = batch
        self._window_rows = window_rows
        self._value_batch = value_batch
        self._value_window_rows = value_window_rows
        self._value_terminal_fraction = value_terminal_fraction
        self._seed = seed
        self._total_steps = total_steps
        self._max_reuse = max_reuse
        self._reuse_gate_interval = reuse_gate_interval
        self._reuse_gate_episodes = reuse_gate_episodes
        self._start_step = start_step
        self._value_sampler = value_sampler
        # Depth 2 rides out replay-store read spikes (compaction bursts)
        # without letting sample timing drift more than two steps.
        self._queue: queue.Queue[
            tuple[SampledBatches | None, BaseException | None]
        ] = queue.Queue(maxsize=2)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="sample-prefetch",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # Unblock a full queue so the thread can observe the stop flag.
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

    def join(self, timeout: float = 10.0) -> None:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise RuntimeError("sample prefetcher did not stop")

    def next(self) -> SampledBatches:
        result, error = self._queue.get()
        if error is not None:
            raise error
        if result is None:
            raise RuntimeError("sample prefetcher returned no result")
        return result

    def refresh(self) -> object:
        with self._lock:
            return self._sampler.refresh()

    def _run(self) -> None:
        value_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="value-sample")
            if self._value_sampler is not None
            else None
        )
        try:
            for step in range(self._start_step, self._total_steps):
                if self._stop.is_set():
                    return
                try:
                    needed_rows = max(
                        required_produced_rows(
                            step,
                            self._batch,
                            self._max_reuse,
                            self._reuse_gate_interval,
                        ),
                        required_produced_rows(
                            step,
                            self._value_batch,
                            self._max_reuse,
                            self._reuse_gate_interval,
                        ),
                    )
                    needed_episodes = required_episodes(
                        step,
                        self._reuse_gate_interval,
                        self._reuse_gate_episodes,
                    )
                    while True:
                        with self._lock:
                            ack = self._sampler.refresh()
                        if (
                            ack.produced_rows >= needed_rows
                            and ack.episodes >= needed_episodes
                        ):
                            break
                        if self._stop.wait(0.1):
                            return
                    with self._lock:
                        result = sample_training_batches(
                            self._sampler,
                            policy_batch=self._batch,
                            policy_window_rows=self._window_rows,
                            value_batch=self._value_batch,
                            value_window_rows=self._value_window_rows,
                            value_terminal_fraction=self._value_terminal_fraction,
                            run_seed=self._seed,
                            step=step,
                            produced_rows=ack.produced_rows,
                            value_sampler=self._value_sampler,
                            value_executor=value_executor,
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
        finally:
            if value_executor is not None:
                value_executor.shutdown(wait=True, cancel_futures=True)
            if self._value_sampler is not None:
                self._value_sampler.close()
