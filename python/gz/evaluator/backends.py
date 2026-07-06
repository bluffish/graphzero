from __future__ import annotations

import importlib
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from gz.codec import BatchView, OutputEncoder
from gz.checkpoints import CheckpointSource, DirectorySource, ResolvedCheckpoint
from gz.common.tags import ModelVersion
from gz.model import build
from gz.model.exphormer import ArchConfig, BatchStager
from gz.model.stub import STUB_MODEL_VERSION, stub
from gz.proto import ERROR_CAPACITY, ERROR_SCHEMA, Hello, ProtocolError

WARMUP_RUNS = 3


@dataclass(frozen=True, slots=True)
class EvalResult:
    model_version: ModelVersion
    payload: memoryview


# The pipelined serving contract: stage(view) copies everything it needs
# out of the request buffer (the view dies when the next frame is read),
# launch(staged) enqueues compute AND the device-to-host copy of its
# outputs (into per-slot pinned buffers, stream-ordered behind the
# replay), finish(pending) waits on the slot's event and encodes. Because
# outputs leave the CUDA-graph static buffers at launch time, up to
# HOST_SLOTS launches may be outstanding before a finish -- the server
# runs the loop at depth 2:
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

    def eval(self, view: BatchView) -> EvalResult:
        if (
            self._encoder is None
            or self._encoder.capacity != view.batch_capacity
            or self._encoder.max_actions != view.max_actions
        ):
            self._encoder = OutputEncoder(view.batch_capacity, view.max_actions)
        values, logits = stub(view)
        return EvalResult(
            model_version=STUB_MODEL_VERSION,
            payload=self._encoder.encode(values, logits, view.row_count),
        )

    # The stub computes eagerly at stage; the payload is copied because the
    # encoder buffer would otherwise be clobbered by the next stage before
    # the pipelined server writes this reply.
    def stage(self, view: BatchView) -> EvalResult:
        result = self.eval(view)
        return EvalResult(model_version=result.model_version, payload=memoryview(bytes(result.payload)))

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
    tensors: object
    row_count: int
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
    encoder: OutputEncoder


HOST_SLOTS = 2


class TorchBackend:
    def __init__(
        self,
        source: CheckpointSource | str,
        *,
        device: str | None = None,
        compile_model: bool = True,
        compile_mode: str = "reduce-overhead",
        max_batch: int = 1024,
        poll_interval: float = 10.0,
    ) -> None:
        torch = _torch()
        self.source = DirectorySource(source) if isinstance(source, (str, Path)) else source
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.max_batch = max_batch
        self.poll_interval = poll_interval
        self.resolved = self.source.resolve_latest()
        self._active = self._build_slot(self.resolved)
        self.manifest = self._active.manifest
        self.stager: BatchStager | None = None
        self._stagers: tuple[BatchStager, ...] = ()
        self._stage_index = 0
        self._host_slots: tuple[_HostSlot, ...] = ()
        self._slot_index = 0
        self._encoder: OutputEncoder | None = None
        self._pending: _ServingSlot | None = None
        self._pending_lock = threading.Lock()
        self._logged_rejections: set[str] = set()
        self._loader_started = False
        self._stop_polling = threading.Event()

    def handshake(self, hello: Hello) -> ModelVersion:
        if hello.feature_schema_hash != self._active.manifest.feature_schema_hash:
            raise ProtocolError(ERROR_SCHEMA, "feature schema hash mismatch")
        if hello.batch_capacity > self.max_batch:
            raise ProtocolError(ERROR_CAPACITY, "batch capacity exceeds backend maximum")
        self.stager = BatchStager(self._active.manifest.feature_schema, hello.batch_capacity, self.device)
        # Ping-pong staging: stage(N+1) runs while N's H2D copy may still
        # be in flight from the same buffers, so batches alternate between
        # two independent staging sets.
        self._stagers = (
            self.stager,
            BatchStager(self._active.manifest.feature_schema, hello.batch_capacity, self.device),
        )
        self._stage_index = 0
        if self.device.type == "cuda":
            torch = _torch()
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
        self._warm_runner(self._active.runner, self.stager, WARMUP_RUNS)
        if self.poll_interval > 0.0:
            self._start_loader()
        return self._active.model_version

    def apply_pending_swap(self) -> None:
        if self.stager is None:
            return
        with self._pending_lock:
            pending = self._pending
            self._pending = None
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
            self._warm_runner(pending.runner, self.stager, WARMUP_RUNS)
        except Exception as error:
            self._log_rejection(pending.model_version.hex(), pending.model_version, error)
            return
        self._active = pending
        self.manifest = pending.manifest
        torch = _torch()
        allocated = (
            torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
        )
        print(
            f"event=checkpoint_swapped model_version={pending.model_version.hex()}"
            f" warm_s={_time.perf_counter() - warm_started:.2f}"
            f" gpu_alloc_mb={allocated / 1e6:.0f}",
            file=sys.stderr,
            flush=True,
        )

    def stop_polling(self) -> None:
        self._stop_polling.set()

    def eval(self, view: BatchView) -> EvalResult:
        return self.finish(self.launch(self.stage(view)))

    def stage(self, view: BatchView) -> StagedEval:
        if self.stager is None:
            raise RuntimeError("torch backend used before handshake")
        if (
            self._encoder is None
            or self._encoder.capacity != view.batch_capacity
            or self._encoder.max_actions != view.max_actions
        ):
            self._encoder = OutputEncoder(view.batch_capacity, view.max_actions)
        stager = self._stagers[self._stage_index]
        self._stage_index = 1 - self._stage_index
        # The encoder rides with the batch: finish() runs after the NEXT
        # batch was staged, which may have re-keyed self._encoder.
        return StagedEval(tensors=stager.copy(view), row_count=view.row_count, encoder=self._encoder)

    def launch(self, staged: StagedEval) -> PendingEval:
        active = self._active
        value_raw, logits = self._run_runner(active.runner, staged.tensors)
        if not self._host_slots:
            return PendingEval(
                model_version=active.model_version,
                slot=None,
                value_raw=value_raw,
                logits=logits,
                row_count=staged.row_count,
                encoder=staged.encoder,
            )
        torch = _torch()
        slot = self._host_slots[self._slot_index]
        self._slot_index = (self._slot_index + 1) % len(self._host_slots)
        # Enqueued on the current stream, so these read the static
        # CUDA-graph outputs BEFORE any later replay overwrites them;
        # the event marks when the pinned copies are complete.
        with torch.inference_mode():
            slot.values.copy_(torch.tanh(value_raw).float(), non_blocking=True)
            slot.logits.copy_(logits.float(), non_blocking=True)
        slot.event.record()
        return PendingEval(
            model_version=active.model_version,
            slot=slot,
            value_raw=None,
            logits=None,
            row_count=staged.row_count,
            encoder=staged.encoder,
        )

    def finish(self, pending: PendingEval) -> EvalResult:
        if pending.slot is not None:
            pending.slot.event.synchronize()
            return EvalResult(
                model_version=pending.model_version,
                payload=pending.encoder.encode(
                    pending.slot.values.numpy(),
                    pending.slot.logits.numpy(),
                    pending.row_count,
                ),
            )
        torch = _torch()
        return EvalResult(
            model_version=pending.model_version,
            payload=pending.encoder.encode(
                torch.tanh(pending.value_raw).detach().float().cpu().numpy(),
                pending.logits.detach().float().cpu().numpy(),
                pending.row_count,
            ),
        )

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
        if version.hex() in self._logged_rejections:
            return
        with self._pending_lock:
            pending_version = self._pending.model_version if self._pending is not None else None
        if version == self._active.model_version or version == pending_version:
            return
        if resolved.manifest.feature_schema_hash != self._active.manifest.feature_schema_hash:
            self._log_rejection(version.hex(), version, "feature schema hash mismatch")
            return
        try:
            slot = self._build_slot(resolved)
        except Exception as error:
            self._log_rejection(version.hex(), version, error)
            return
        with self._pending_lock:
            self._pending = slot

    def _build_slot(self, resolved: ResolvedCheckpoint) -> _ServingSlot:
        arch = ArchConfig.from_dict(resolved.manifest.arch_config)
        if arch.name != resolved.manifest.arch_name:
            raise ValueError("manifest arch name mismatch")
        model = build(resolved.manifest.feature_schema, arch)
        from gz.checkpoints.weights import load_state_dict

        model.load_state_dict(load_state_dict(resolved.weights_path))
        model.to(self.device)
        model.eval()
        torch = _torch()
        runner = torch.compile(model, fullgraph=True, mode=self.compile_mode) if self.compile_model else model
        return _ServingSlot(
            manifest=resolved.manifest,
            runner=runner,
            model_version=resolved.manifest.model_version,
        )

    def _warm_runner(self, runner: object, stager: BatchStager, count: int) -> None:
        for _ in range(count):
            self._run_runner(runner, stager.dummy())

    def _run_runner(self, runner: object, tensors: object) -> tuple[object, object]:
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
