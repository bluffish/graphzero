from __future__ import annotations

import importlib
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from gz.codec import BatchView, OutputEncoder
from gz.checkpoints import CheckpointSource, DirectorySource, ResolvedCheckpoint
from gz.common.tags import EngineIdentity, ModelVersion
from gz.model.exphormer import ArchConfig, BatchStager, build_model
from gz.model.stub import STUB_MODEL_VERSION, stub
from gz.proto import ERROR_CAPACITY, ERROR_PROTOCOL, ERROR_SCHEMA, Hello, ProtocolError

WARMUP_RUNS = 3
INITIAL_MODEL_GENERATION = 1
DEFAULT_MAX_RESIDENT_MODEL_GENERATIONS = 2


@dataclass(frozen=True, slots=True)
class EvalResult:
    model_version: ModelVersion
    payload: memoryview


class EvalTimingStats:
    def __init__(self, interval: int = 100) -> None:
        self.interval = interval
        self.batches = 0
        self.rows = 0
        self.payload_bytes = 0
        self.stage_s = 0.0
        self.launch_s = 0.0
        self.sync_s = 0.0
        self.encode_s = 0.0

    def record_stage(self, seconds: float) -> None:
        self.stage_s += seconds

    def record_launch(self, seconds: float) -> None:
        self.launch_s += seconds

    def record_finish(
        self,
        *,
        rows: int,
        payload_bytes: int,
        sync_s: float,
        encode_s: float,
    ) -> None:
        self.batches += 1
        self.rows += rows
        self.payload_bytes += payload_bytes
        self.sync_s += sync_s
        self.encode_s += encode_s
        if self.batches % self.interval == 0:
            denom = float(self.interval)
            print(
                "event=eval_timing"
                f" batches={self.batches}"
                f" rows={self.rows}"
                f" stage_ms={self.stage_s * 1000.0 / denom:.3f}"
                f" launch_ms={self.launch_s * 1000.0 / denom:.3f}"
                f" sync_ms={self.sync_s * 1000.0 / denom:.3f}"
                f" encode_ms={self.encode_s * 1000.0 / denom:.3f}"
                f" payload_kb={self.payload_bytes / 1024.0 / denom:.1f}",
                file=sys.stderr,
                flush=True,
            )
            self.stage_s = 0.0
            self.launch_s = 0.0
            self.sync_s = 0.0
            self.encode_s = 0.0
            self.payload_bytes = 0


# The pipelined serving contract: stage(view) copies everything it needs
# out of the request buffer (the view dies when the next frame is read),
# launch(staged) enqueues compute AND the device-to-host copy of its
# outputs (into per-slot pinned buffers, stream-ordered behind the
# replay), finish(pending) waits on the slot's event and encodes. Because
# outputs leave the CUDA-graph static buffers at launch time, up to
# HOST_SLOTS launches may be outstanding before a finish -- the server
# runs the loop at the same depth:
#   stage(N+1) -> launch(N+1) -> ... -> finish(N) -> write reply(N)
# finish(N) waits only on N's event, never on replay(N+1) queued behind
# it.


class StubBackend:
    def __init__(self) -> None:
        self._encoder: OutputEncoder | None = None

    def handshake(self, hello: Hello) -> ModelVersion:
        _ = hello
        return STUB_MODEL_VERSION

    def apply_pending_swap(self) -> None:
        return None

    def model_generation(self) -> tuple[int, ModelVersion]:
        return INITIAL_MODEL_GENERATION, STUB_MODEL_VERSION

    def release_model_generation(self, generation: int, version: ModelVersion) -> None:
        if generation != INITIAL_MODEL_GENERATION or version != STUB_MODEL_VERSION:
            raise ProtocolError(ERROR_PROTOCOL, "unknown model generation")
        raise ProtocolError(ERROR_PROTOCOL, "cannot release the active model generation")

    def eval(self, view: BatchView, model_version: ModelVersion | None = None) -> EvalResult:
        if model_version is not None and model_version != STUB_MODEL_VERSION:
            raise ProtocolError(ERROR_PROTOCOL, "requested model version is unavailable")
        if (
            self._encoder is None
            or self._encoder.capacity != view.batch_capacity
            or self._encoder.max_actions != view.max_actions
        ):
            self._encoder = OutputEncoder(view.batch_capacity, view.max_actions)
        values, logits = stub(view)
        return EvalResult(
            model_version=STUB_MODEL_VERSION,
            payload=self._encoder.encode(
                values,
                logits,
                view.row_count,
                view.action_count[: view.row_count],
            ),
        )

    # The stub computes eagerly at stage; the payload is copied because the
    # encoder buffer would otherwise be clobbered by the next stage before
    # the pipelined server writes this reply.
    def stage(
        self, view: BatchView, model_version: ModelVersion | None = None
    ) -> EvalResult:
        result = self.eval(view, model_version)
        return EvalResult(
            model_version=result.model_version,
            payload=memoryview(bytes(result.payload)),
        )

    def launch(self, staged: EvalResult) -> EvalResult:
        return staged

    def finish(self, pending: EvalResult) -> EvalResult:
        return pending


@dataclass(frozen=True, slots=True)
class _ServingSlot:
    manifest: object
    runner: object
    model_version: ModelVersion


@dataclass(frozen=True, slots=True)
class StagedEval:
    serving: _ServingSlot
    tensors: object
    ready_event: object
    row_count: int
    action_counts: object
    encoder: OutputEncoder


@dataclass(frozen=True, slots=True)
class _HostSlot:
    values: object
    logits: object
    event: object


@dataclass(frozen=True, slots=True)
class PendingEval:
    model_version: ModelVersion
    # CUDA path: outputs already copied to slot's pinned buffers.
    slot: _HostSlot | None
    # CPU path: raw tensors, finished synchronously.
    value_raw: object
    logits: object
    row_count: int
    action_counts: object
    encoder: OutputEncoder


PIPELINE_DEPTH = 3
HOST_SLOTS = PIPELINE_DEPTH


class TorchBackend:
    def __init__(
        self,
        source: CheckpointSource | str,
        *,
        device: str | None = None,
        compile_model: bool = True,
        compile_mode: str = "reduce-overhead",
        max_batch: int = 1024,
        max_resident_model_generations: int = DEFAULT_MAX_RESIDENT_MODEL_GENERATIONS,
        poll_interval: float = 10.0,
    ) -> None:
        torch = _torch()
        self.source = DirectorySource(source) if isinstance(source, (str, Path)) else source
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.max_batch = max_batch
        if max_resident_model_generations < 2:
            raise ValueError("max_resident_model_generations must be at least two")
        self.max_resident_model_generations = max_resident_model_generations
        self.poll_interval = poll_interval
        self.resolved = self.source.resolve_latest()
        self._active = self._build_slot(self.resolved)
        self._active_generation = INITIAL_MODEL_GENERATION
        self._next_generation = INITIAL_MODEL_GENERATION + 1
        self._slots_by_version: dict[str, tuple[int, _ServingSlot]] = {
            self._active.model_version.hex(): (self._active_generation, self._active)
        }
        self._versions_by_generation: dict[int, str] = {
            self._active_generation: self._active.model_version.hex()
        }
        self.manifest = self._active.manifest
        self.stager: BatchStager | None = None
        self._stagers: tuple[BatchStager, ...] = ()
        self._stage_index = 0
        self._host_slots: tuple[_HostSlot, ...] = ()
        self._transfer_stream = None
        self._slot_index = 0
        self._encoder: OutputEncoder | None = None
        self._pending: _ServingSlot | None = None
        self._pending_lock = threading.Lock()
        self._resident_versions = {self._active.model_version.hex()}
        self._loading_version: str | None = None
        self._adopting_version: str | None = None
        self._logged_rejections: set[str] = set()
        self._loader_started = False
        self._stop_polling = threading.Event()
        self._timings = EvalTimingStats()
        self._engine_identity: EngineIdentity | None = None

    def handshake(self, hello: Hello) -> ModelVersion:
        if hello.feature_schema_hash != self._active.manifest.feature_schema_hash:
            raise ProtocolError(ERROR_SCHEMA, "feature schema hash mismatch")
        engine_identity = hello.engine_identity
        if self._engine_identity is not None and engine_identity != self._engine_identity:
            raise ProtocolError(ERROR_SCHEMA, "engine identity changed across handshakes")
        if self._active.manifest.engine_identity != engine_identity:
            raise ProtocolError(ERROR_SCHEMA, "engine identity mismatch")
        if hello.batch_capacity > self.max_batch:
            raise ProtocolError(ERROR_CAPACITY, "batch capacity exceeds backend maximum")
        self._engine_identity = engine_identity
        torch = _torch()
        self._transfer_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self.stager = BatchStager(
            self._active.manifest.feature_schema,
            hello.batch_capacity,
            self.device,
            transfer_stream=self._transfer_stream,
        )
        # Runtime staging matches the pending depth. Warmup owns a separate
        # stager because swaps can happen with old launches still queued.
        self._stagers = tuple(
            BatchStager(
                self._active.manifest.feature_schema,
                hello.batch_capacity,
                self.device,
                transfer_stream=self._transfer_stream,
            )
            for _ in range(PIPELINE_DEPTH)
        )
        self._stage_index = 0
        if self.device.type == "cuda":
            schema = self._active.manifest.feature_schema
            self._host_slots = tuple(
                _HostSlot(
                    values=torch.empty(hello.batch_capacity, dtype=torch.float32, pin_memory=True),
                    logits=torch.empty(
                        (hello.batch_capacity, schema.max_actions),
                        dtype=torch.float32,
                        pin_memory=True,
                    ),
                    event=torch.cuda.Event(),
                )
            for _ in range(HOST_SLOTS)
            )
            self._slot_index = 0
        self._warm_slot(self._active, self.stager, WARMUP_RUNS)
        if self.poll_interval > 0.0:
            self._start_loader()
        return self._active.model_version

    def apply_pending_swap(self) -> None:
        if self.stager is None:
            return
        with self._pending_lock:
            pending = self._pending
            self._pending = None
            self._adopting_version = None if pending is None else pending.model_version.hex()
        if pending is None:
            return
        import time as _time

        warm_started = _time.perf_counter()
        try:
            # CUDA graph capture (reduce-overhead) only works on the serving
            # thread, so the loader publishes the slot unwarmed and warmup
            # happens here, between frames. A same-arch checkpoint hits the
            # inductor cache and warms in well under a second; a cold compile
            # pauses serving for its duration while workers park.
            self._warm_slot(pending, self.stager, WARMUP_RUNS)
        except Exception as error:
            with self._pending_lock:
                self._adopting_version = None
            self._log_rejection(pending.model_version.hex(), pending.model_version, error)
            return
        generation = self._next_generation
        self._next_generation += 1
        self._active = pending
        self._active_generation = generation
        version_key = pending.model_version.hex()
        self._slots_by_version[version_key] = (generation, pending)
        self._versions_by_generation[generation] = version_key
        self.manifest = pending.manifest
        with self._pending_lock:
            self._resident_versions.add(version_key)
            self._adopting_version = None
        torch = _torch()
        allocated = (
            torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
        )
        print(
            f"event=checkpoint_swapped model_version={pending.model_version.hex()}"
            f" generation={generation}"
            f" training_step={pending.manifest.training_step}"
            f" warm_s={_time.perf_counter() - warm_started:.2f}"
            f" gpu_alloc_mb={allocated / 1e6:.0f}",
            file=sys.stderr,
            flush=True,
        )

    def stop_polling(self) -> None:
        self._stop_polling.set()

    def model_generation(self) -> tuple[int, ModelVersion]:
        return self._active_generation, self._active.model_version

    def release_model_generation(self, generation: int, version: ModelVersion) -> None:
        if generation == self._active_generation:
            raise ProtocolError(ERROR_PROTOCOL, "cannot release the active model generation")
        version_key = version.hex()
        if self._versions_by_generation.get(generation) != version_key:
            raise ProtocolError(ERROR_PROTOCOL, "unknown model generation")
        stored = self._slots_by_version.get(version_key)
        if stored is None or stored[0] != generation:
            raise ProtocolError(ERROR_PROTOCOL, "model generation mismatch")
        del self._versions_by_generation[generation]
        del self._slots_by_version[version_key]
        with self._pending_lock:
            self._resident_versions.discard(version_key)

    def eval(self, view: BatchView, model_version: ModelVersion | None = None) -> EvalResult:
        return self.finish(self.launch(self.stage(view, model_version)))

    def stage(
        self, view: BatchView, model_version: ModelVersion | None = None
    ) -> StagedEval:
        if self.stager is None:
            raise RuntimeError("torch backend used before handshake")
        if (
            self._encoder is None
            or self._encoder.capacity != view.batch_capacity
            or self._encoder.max_actions != view.max_actions
        ):
            self._encoder = OutputEncoder(view.batch_capacity, view.max_actions)
        started = time.perf_counter()
        stager = self._stagers[self._stage_index]
        self._stage_index = (self._stage_index + 1) % len(self._stagers)
        active = self._serving_slot(model_version)
        tensors = stager.copy(view)
        action_counts = view.action_count[: view.row_count].copy()
        self._timings.record_stage(time.perf_counter() - started)
        # The encoder rides with the batch: finish() runs after the NEXT
        # batch was staged, which may have re-keyed self._encoder.
        return StagedEval(
            serving=active,
            tensors=tensors,
            ready_event=stager.ready_event,
            row_count=view.row_count,
            action_counts=action_counts,
            encoder=self._encoder,
        )

    def launch(self, staged: StagedEval) -> PendingEval:
        active = staged.serving
        started = time.perf_counter()
        self._wait_for_stage(staged.ready_event)
        value_raw, logits = self._run_runner(active.runner, staged.tensors)
        if not self._host_slots:
            self._timings.record_launch(time.perf_counter() - started)
            return PendingEval(
                model_version=active.model_version,
                slot=None,
                value_raw=value_raw,
                logits=logits,
                row_count=staged.row_count,
                action_counts=staged.action_counts,
                encoder=staged.encoder,
            )
        torch = _torch()
        slot = self._host_slots[self._slot_index]
        self._slot_index = (self._slot_index + 1) % len(self._host_slots)
        # Enqueued on the current stream, so these read the static
        # CUDA-graph outputs BEFORE any later replay overwrites them;
        # the event marks when the pinned copies are complete.
        with torch.inference_mode():
            rows = staged.row_count
            slot.values[:rows].copy_(value_raw[:rows].float(), non_blocking=True)
            slot.logits[:rows].copy_(logits[:rows].float(), non_blocking=True)
        slot.event.record()
        self._timings.record_launch(time.perf_counter() - started)
        return PendingEval(
            model_version=active.model_version,
            slot=slot,
            value_raw=None,
            logits=None,
            row_count=staged.row_count,
            action_counts=staged.action_counts,
            encoder=staged.encoder,
        )

    def finish(self, pending: PendingEval) -> EvalResult:
        if pending.slot is not None:
            sync_started = time.perf_counter()
            pending.slot.event.synchronize()
            sync_s = time.perf_counter() - sync_started
            encode_started = time.perf_counter()
            payload = pending.encoder.encode(
                pending.slot.values.numpy(),
                pending.slot.logits.numpy(),
                pending.row_count,
                pending.action_counts,
            )
            encode_s = time.perf_counter() - encode_started
            self._timings.record_finish(
                rows=pending.row_count,
                payload_bytes=len(payload),
                sync_s=sync_s,
                encode_s=encode_s,
            )
            return EvalResult(
                model_version=pending.model_version,
                payload=payload,
            )
        sync_started = time.perf_counter()
        values = pending.value_raw.detach().float().cpu().numpy()
        logits = pending.logits.detach().float().cpu().numpy()
        sync_s = time.perf_counter() - sync_started
        encode_started = time.perf_counter()
        payload = pending.encoder.encode(
            values,
            logits,
            pending.row_count,
            pending.action_counts,
        )
        encode_s = time.perf_counter() - encode_started
        self._timings.record_finish(
            rows=pending.row_count,
            payload_bytes=len(payload),
            sync_s=sync_s,
            encode_s=encode_s,
        )
        return EvalResult(
            model_version=pending.model_version,
            payload=payload,
        )

    def _serving_slot(self, model_version: ModelVersion | None) -> _ServingSlot:
        if model_version is None:
            return self._active
        stored = self._slots_by_version.get(model_version.hex())
        if stored is None:
            raise ProtocolError(ERROR_PROTOCOL, "requested model version is unavailable")
        return stored[1]

    def _wait_for_stage(self, ready_event: object) -> None:
        if ready_event is None:
            return
        torch = _torch()
        torch.cuda.current_stream(self.device).wait_event(ready_event)

    def _start_loader(self) -> None:
        if self._loader_started:
            return
        self._loader_started = True
        thread = threading.Thread(target=self._loader_loop, name="gz-evaluator-hotswap", daemon=True)
        thread.start()

    def _loader_loop(self) -> None:
        while not self._stop_polling.wait(self.poll_interval):
            try:
                self._poll_once()
            except Exception as error:
                self._log_rejection(f"loader:{type(error).__name__}:{error}", None, error)

    def _poll_once(self) -> None:
        resolved = self.source.resolve_latest()
        version = resolved.manifest.model_version
        version_key = version.hex()
        if version_key in self._logged_rejections:
            return
        with self._pending_lock:
            pending_version = self._pending.model_version if self._pending is not None else None
            occupied = (
                len(self._resident_versions)
                + int(self._pending is not None)
                + int(self._loading_version is not None)
                + int(self._adopting_version is not None)
            )
            if (
                version_key in self._resident_versions
                or version == pending_version
                or version_key == self._loading_version
                or version_key == self._adopting_version
                or occupied >= self.max_resident_model_generations
            ):
                return
            self._loading_version = version_key
        if resolved.manifest.feature_schema_hash != self._active.manifest.feature_schema_hash:
            with self._pending_lock:
                self._loading_version = None
            self._log_rejection(version.hex(), version, "feature schema hash mismatch")
            return
        if (
            self._engine_identity is not None
            and resolved.manifest.engine_identity != self._engine_identity
        ):
            with self._pending_lock:
                self._loading_version = None
            self._log_rejection(version.hex(), version, "engine identity mismatch")
            return
        try:
            slot = self._build_slot(resolved)
        except Exception as error:
            with self._pending_lock:
                self._loading_version = None
            self._log_rejection(version.hex(), version, error)
            return
        with self._pending_lock:
            self._loading_version = None
            self._pending = slot

    def _build_slot(self, resolved: ResolvedCheckpoint) -> _ServingSlot:
        arch = ArchConfig.from_manifest_dict(resolved.manifest.arch_config)
        if arch.name != resolved.manifest.arch_name:
            raise ValueError("manifest arch name mismatch")
        model = build_model(resolved.manifest.feature_schema, arch)
        from gz.checkpoints.weights import load_state_dict

        model.load_state_dict(load_state_dict(resolved.weights_path))
        model.to(self.device)
        model.eval()
        runner = (
            _torch().compile(model, fullgraph=True, mode=self.compile_mode)
            if self.compile_model
            else model
        )
        return _ServingSlot(
            manifest=resolved.manifest,
            runner=runner,
            model_version=resolved.manifest.model_version,
        )

    def _warm_slot(self, slot: _ServingSlot, stager: BatchStager, count: int) -> None:
        for _ in range(count):
            tensors = stager.dummy()
            self._wait_for_stage(stager.ready_event)
            self._run_runner(slot.runner, tensors)
            if self.device.type == "cuda":
                _torch().cuda.current_stream(self.device).synchronize()

    def _run_runner(
        self,
        runner: object,
        tensors: object,
    ) -> tuple[object, object]:
        torch = _torch()
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return runner(tensors)
            return runner(tensors)

    def _log_rejection(self, key: str, version: ModelVersion | None, error: object) -> None:
        if key in self._logged_rejections:
            return
        self._logged_rejections.add(key)
        version_text = version.hex() if version is not None else "unknown"
        print(f"event=checkpoint_rejected model_version={version_text} error={error}", file=sys.stderr, flush=True)


def _torch():
    return importlib.import_module("torch")
