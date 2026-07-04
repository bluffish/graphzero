from __future__ import annotations

import hashlib
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from gz.codec import BatchView, FeatureSchemaConfig, TargetsView
from gz.common import FeatureSchemaHash
from gz.proto import (
    ENCODING_VERSION,
    ProtocolError,
    decode_error,
    read_frame,
    write_frame,
)

SAMPLE_PROTOCOL_VERSION = 4

FRAME_HELLO = 1
FRAME_HELLO_ACK = 2
FRAME_SAMPLE = 3
FRAME_SAMPLE_RESULT = 4
FRAME_ERROR = 5


class SampleError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SampleResult:
    batch: BatchView
    targets: TargetsView
    produced_rows: int


@dataclass(frozen=True, slots=True)
class SampleAck:
    feature_schema_hash: FeatureSchemaHash
    max_batch: int
    produced_rows: int
    episodes: int
    episodes_stopped: int
    episode_cost_ema: float
    episode_len_ema: float
    stop_rate_ema: float
    best_cost: float
    root: RootInfo | None
    feature_schema: FeatureSchemaConfig


@dataclass(frozen=True, slots=True)
class RootInfo:
    cost: float
    node_count: int
    edge_count: int
    candidate_count: int


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
    def max_batch(self) -> int:
        return self._ack().max_batch

    @property
    def produced_rows(self) -> int:
        return self._ack().produced_rows

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def connect(self) -> SampleAck:
        self.close()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(self.socket_path))
        self.sock = sock
        write_frame(sock, FRAME_HELLO, struct.pack("<II", SAMPLE_PROTOCOL_VERSION, ENCODING_VERSION))
        frame_type, payload = read_frame(sock, self.read_buf)
        if frame_type == FRAME_ERROR:
            code, message = decode_error(payload)
            raise SampleError(f"sample hello failed: {code} {message}")
        if frame_type != FRAME_HELLO_ACK:
            raise SampleError("expected sample HELLO_ACK")
        self.ack = decode_ack(payload)
        return self.ack

    def wait_until_ready(self, min_startup_rows: int, alive_check: object = None) -> SampleAck:
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

    def sample(self, batch: int, window: int, seed: int) -> SampleResult:
        return self._with_reconnect(lambda: self._sample_connected(batch, window, seed))

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
        write_frame(self.sock, FRAME_HELLO, struct.pack("<II", SAMPLE_PROTOCOL_VERSION, ENCODING_VERSION))
        frame_type, payload = read_frame(self.sock, self.read_buf)
        if frame_type == FRAME_ERROR:
            code, message = decode_error(payload)
            raise SampleError(f"sample hello failed: {code} {message}")
        if frame_type != FRAME_HELLO_ACK:
            raise SampleError("expected sample HELLO_ACK")
        self.ack = decode_ack(payload)
        return self.ack

    def _sample_connected(self, batch: int, window: int, seed: int) -> SampleResult:
        if self.sock is None:
            self.connect()
        if batch <= 0 or window <= 0:
            raise ValueError("batch and window must be positive")
        assert self.sock is not None
        write_frame(self.sock, FRAME_SAMPLE, struct.pack("<IQQ", batch, window, seed))
        frame_type, payload = read_frame(self.sock, self.read_buf)
        if frame_type == FRAME_ERROR:
            code, message = decode_error(payload)
            raise SampleError(f"sample failed: {code} {message}")
        if frame_type != FRAME_SAMPLE_RESULT:
            raise SampleError("expected SAMPLE_RESULT")
        if len(payload) < 4:
            raise SampleError("sample result truncated")
        gzfb_len = struct.unpack_from("<I", payload, 0)[0]
        start = 4
        end = start + gzfb_len
        if len(payload) < end:
            raise SampleError("sample gzfb truncated")
        batch_view = BatchView.parse(payload[start:end])
        targets = TargetsView.parse(payload[end:])
        if batch_view.batch_capacity != targets.capacity:
            raise SampleError("sample batch/target capacity mismatch")
        if batch_view.row_count != targets.row_count:
            raise SampleError("sample batch/target row count mismatch")
        if batch_view.max_actions != targets.max_actions:
            raise SampleError("sample batch/target action width mismatch")
        return SampleResult(batch=batch_view, targets=targets, produced_rows=self.produced_rows)

    def _ack(self) -> SampleAck:
        if self.ack is None:
            raise RuntimeError("sample client is not connected")
        return self.ack


def decode_ack(payload: memoryview) -> SampleAck:
    if len(payload) < 100:
        raise SampleError("sample HELLO_ACK truncated")
    protocol_version = struct.unpack_from("<I", payload, 0)[0]
    if protocol_version != SAMPLE_PROTOCOL_VERSION:
        raise SampleError("sample protocol version mismatch")
    max_batch = struct.unpack_from("<I", payload, 36)[0]
    produced_rows = struct.unpack_from("<Q", payload, 40)[0]
    episodes = struct.unpack_from("<Q", payload, 48)[0]
    episodes_stopped = struct.unpack_from("<Q", payload, 56)[0]
    cost_ema, len_ema, stop_ema = struct.unpack_from("<fff", payload, 64)
    best_cost = struct.unpack_from("<f", payload, 76)[0]
    root_present = struct.unpack_from("<I", payload, 80)[0]
    root_cost = struct.unpack_from("<f", payload, 84)[0]
    root_nodes, root_edges, root_candidates = struct.unpack_from("<III", payload, 88)
    root = (
        RootInfo(
            cost=root_cost,
            node_count=root_nodes,
            edge_count=root_edges,
            candidate_count=root_candidates,
        )
        if root_present
        else None
    )
    return SampleAck(
        feature_schema_hash=FeatureSchemaHash.from_bytes(payload[4:36]),
        max_batch=max_batch,
        produced_rows=produced_rows,
        episodes=episodes,
        episodes_stopped=episodes_stopped,
        episode_cost_ema=cost_ema,
        episode_len_ema=len_ema,
        stop_rate_ema=stop_ema,
        best_cost=best_cost,
        root=root,
        feature_schema=FeatureSchemaConfig.decode(payload[100:]),
    )


def step_seed(run_seed: int, step: int) -> int:
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(b"gz-trainer-step-seed-v1")
    hasher.update(run_seed.to_bytes(8, "little", signed=False))
    hasher.update(step.to_bytes(8, "little", signed=False))
    return int.from_bytes(hasher.digest(), "little")
