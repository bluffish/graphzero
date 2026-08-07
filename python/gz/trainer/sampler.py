from __future__ import annotations

import hashlib
import socket
import time
from dataclasses import dataclass
from pathlib import Path

from gz.codec import BatchView, FeatureSchemaConfig, TargetsView
from gz.common import ActionSetHash, EngineIdentity, EngineId, EngineVersion, FeatureSchemaHash
from gz.proto import (
    ProtocolError,
    decode_error,
    read_frame,
    write_frame,
)
from gz.trainer.sample_protocol import (
    FRAME_ERROR,
    FRAME_HELLO,
    FRAME_HELLO_ACK,
    FRAME_SAMPLE,
    FRAME_SAMPLE_RESULT,
    SAMPLE_PROTOCOL_VERSION,
    SampleAck,
    SampleError,
    decode_ack,
    encode_hello,
    encode_sample_request,
    split_sample_result,
)


@dataclass(frozen=True, slots=True)
class SampleResult:
    batch: BatchView
    targets: TargetsView
    produced_rows: int


class SampleClient:
    def __init__(
        self,
        socket_path: str | Path,
        *,
        startup_timeout: float = 60.0,
        reconnect_limit: int = 5,
        backoff: float = 0.5,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.startup_timeout = startup_timeout
        self.reconnect_limit = reconnect_limit
        self.backoff = backoff
        self.sock: socket.socket | None = None
        self.read_buf = bytearray()
        self.ack: SampleAck | None = None

    @property
    def feature_schema(self) -> FeatureSchemaConfig:
        return self._ack().feature_schema

    @property
    def feature_schema_hash(self) -> FeatureSchemaHash:
        return self._ack().feature_schema_hash

    @property
    def engine_id(self) -> EngineId:
        return self._ack().engine_id

    @property
    def engine_version(self) -> EngineVersion:
        return self._ack().engine_version

    @property
    def action_set_hash(self) -> ActionSetHash:
        return self._ack().action_set_hash

    @property
    def engine_identity(self) -> EngineIdentity:
        return self._ack().engine_identity

    @property
    def max_batch(self) -> int:
        return self._ack().max_batch

    @property
    def produced_rows(self) -> int:
        return self._ack().produced_rows

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def fork(self) -> SampleClient:
        """Return an independent connection with the same retry policy."""
        return SampleClient(
            self.socket_path,
            startup_timeout=self.startup_timeout,
            reconnect_limit=self.reconnect_limit,
            backoff=self.backoff,
        )

    def connect(self) -> SampleAck:
        self.close()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(self.socket_path))
        self.sock = sock
        write_frame(sock, FRAME_HELLO, encode_hello())
        frame_type, payload = read_frame(sock, self.read_buf)
        if frame_type == FRAME_ERROR:
            code, message = decode_error(payload)
            raise SampleError(f"sample hello failed: {code} {message}")
        if frame_type != FRAME_HELLO_ACK:
            raise SampleError("expected sample HELLO_ACK")
        self.ack = decode_ack(payload)
        return self.ack

    def wait_until_ready(
        self,
        min_startup_rows: int,
        alive_check: object = None,
    ) -> SampleAck:
        deadline = time.monotonic() + self.startup_timeout
        while True:
            if alive_check is not None:
                alive_check()
            try:
                ack = self.connect()
                if ack.produced_rows >= min_startup_rows:
                    return ack
            except (OSError, ProtocolError, SampleError):
                self.close()
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for replay sample service")
            time.sleep(self.backoff)

    def sample(
        self, batch: int, window: int, seed: int, *, terminal_only: bool = False
    ) -> SampleResult:
        return self._with_reconnect(
            lambda: self._sample_connected(batch, window, seed, terminal_only)
        )

    def refresh(self) -> SampleAck:
        """Re-acks on the live connection for fresh produced_rows."""
        return self._with_reconnect(self._refresh_connected)

    def _with_reconnect(self, request: object) -> object:
        failures = 0
        while True:
            try:
                return request()
            except (OSError, ProtocolError, SampleError):
                self.close()
                failures += 1
                if failures > self.reconnect_limit:
                    raise
                time.sleep(self.backoff)
                self.connect()

    def _refresh_connected(self) -> SampleAck:
        if self.sock is None:
            return self.connect()
        write_frame(self.sock, FRAME_HELLO, encode_hello())
        frame_type, payload = read_frame(self.sock, self.read_buf)
        if frame_type == FRAME_ERROR:
            code, message = decode_error(payload)
            raise SampleError(f"sample hello failed: {code} {message}")
        if frame_type != FRAME_HELLO_ACK:
            raise SampleError("expected sample HELLO_ACK")
        self.ack = decode_ack(payload)
        return self.ack

    def _sample_connected(
        self, batch: int, window: int, seed: int, terminal_only: bool
    ) -> SampleResult:
        if self.sock is None:
            self.connect()
        assert self.sock is not None
        write_frame(
            self.sock,
            FRAME_SAMPLE,
            encode_sample_request(batch, window, seed, terminal_only=terminal_only),
        )
        frame_type, payload = read_frame(self.sock, self.read_buf)
        if frame_type == FRAME_ERROR:
            code, message = decode_error(payload)
            raise SampleError(f"sample failed: {code} {message}")
        if frame_type != FRAME_SAMPLE_RESULT:
            raise SampleError("expected SAMPLE_RESULT")
        batch_payload, targets_payload = split_sample_result(payload)
        batch_view = BatchView.parse(batch_payload)
        targets = TargetsView.parse(targets_payload)
        if batch_view.batch_capacity != targets.capacity:
            raise SampleError("sample batch/target capacity mismatch")
        if batch_view.row_count != targets.row_count:
            raise SampleError("sample batch/target row count mismatch")
        if batch_view.max_actions != targets.max_actions:
            raise SampleError("sample batch/target action width mismatch")
        # BatchView and TargetsView are zero-copy views into read_buf. Hand the
        # backing allocation to this result before another request reuses the
        # client, otherwise a prefetched value sample overwrites the policy
        # sample that precedes it.
        self.read_buf = bytearray()
        return SampleResult(batch=batch_view, targets=targets, produced_rows=self.produced_rows)

    def _ack(self) -> SampleAck:
        if self.ack is None:
            raise RuntimeError("sample client is not connected")
        return self.ack

def step_seed(run_seed: int, step: int, stream: str = "") -> int:
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(b"gz-trainer-step-seed-v1")
    hasher.update(run_seed.to_bytes(8, "little", signed=False))
    hasher.update(step.to_bytes(8, "little", signed=False))
    if stream:
        hasher.update(b"\0")
        hasher.update(stream.encode("ascii"))
    return int.from_bytes(hasher.digest(), "little")
