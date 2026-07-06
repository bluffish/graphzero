from __future__ import annotations

import os
import select
import socket
import struct
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from gz.codec import BatchView
from gz.codec.batch import EncodingError
from gz.common.tags import ActionSetHash, EngineId, EngineVersion, FeatureSchemaHash
from gz.evaluator.backends import StubBackend
from gz.proto import (
    BATCH_ENCODING_VERSION,
    ERROR_CAPACITY,
    ERROR_ENCODING,
    ERROR_MALFORMED,
    ERROR_PROTOCOL,
    ERROR_SCHEMA,
    FRAME_ERROR,
    FRAME_EVAL,
    FRAME_EVAL_RESULT,
    FRAME_HELLO,
    FRAME_HELLO_ACK,
    FRAME_PING,
    FRAME_PONG,
    Hello,
    HelloAck,
    PROTOCOL_VERSION,
    ProtocolError,
    encode_error,
    read_frame,
    write_frame_into,
)


@dataclass(frozen=True, slots=True)
class _ConnectionState:
    feature_schema_hash: FeatureSchemaHash
    batch_capacity: int
    engine_id: EngineId
    engine_version: EngineVersion
    action_set_hash: ActionSetHash


PIPELINE_DEPTH = 2


def serve(socket_path: str | Path, backend: StubBackend, *, ready_event: Event | None = None) -> None:
    path = Path(socket_path)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path))
        listener.listen(1)
        if ready_event is not None:
            ready_event.set()
        conn, _ = listener.accept()
        with conn:
            _serve_connection(conn, backend)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _serve_connection(conn: socket.socket, backend: StubBackend) -> None:
    read_buf = bytearray()
    write_buf = bytearray()
    # Launched-but-unfinished evals, oldest first, at most PIPELINE_DEPTH.
    # Per EVAL frame the order is stage(new) -> launch(new) -> queue; the
    # oldest entry is finished and its reply written when the queue is
    # full or the client has nothing further on the wire. Backends move
    # outputs off CUDA-graph static buffers at launch time, so multiple
    # launches may be outstanding; replies stay FIFO.
    pending: deque[tuple[int, object]] = deque()
    try:
        state = _handshake(conn, read_buf, write_buf, backend)
        while True:
            while pending:
                # Hold replies open only while the next request is already
                # on the wire and there is queue room to keep pipelining
                # (read_frame never over-reads, so socket readability is
                # the complete signal that the client is not blocked).
                readable, _, _ = select.select([conn], [], [], 0)
                if readable and len(pending) < PIPELINE_DEPTH:
                    break
                _flush_oldest(conn, write_buf, backend, pending)
            frame_type, payload = read_frame(conn, read_buf)
            try:
                if frame_type == FRAME_PING:
                    while pending:
                        _flush_oldest(conn, write_buf, backend, pending)
                    _handle_ping(conn, write_buf, payload)
                elif frame_type == FRAME_EVAL:
                    batch_id, view = _parse_eval(state, payload)
                    staged = backend.stage(view)
                    del view
                    if len(pending) >= PIPELINE_DEPTH:
                        _flush_oldest(conn, write_buf, backend, pending)
                    backend.apply_pending_swap()
                    pending.append((batch_id, backend.launch(staged)))
                else:
                    raise ProtocolError(ERROR_PROTOCOL, "unexpected frame type")
            finally:
                # read_buf cannot grow while a memoryview references it
                # (bytearray.extend raises BufferError), so the payload view
                # must be dropped before the next read_frame.
                del payload
    except ProtocolError as error:
        _send_error(conn, write_buf, error.code, error.message)
    except EncodingError as error:
        _send_error(conn, write_buf, ERROR_ENCODING, str(error))


def _handshake(
    conn: socket.socket,
    read_buf: bytearray,
    write_buf: bytearray,
    backend: StubBackend,
) -> _ConnectionState:
    frame_type, payload = read_frame(conn, read_buf)
    if frame_type != FRAME_HELLO:
        raise ProtocolError(ERROR_PROTOCOL, "expected HELLO")
    hello = Hello.decode(payload)
    if hello.protocol_version != PROTOCOL_VERSION:
        raise ProtocolError(ERROR_PROTOCOL, "protocol version mismatch")
    if hello.encoding_version != BATCH_ENCODING_VERSION:
        raise ProtocolError(ERROR_ENCODING, "encoding version mismatch")
    if hello.batch_capacity == 0:
        raise ProtocolError(ERROR_CAPACITY, "zero batch capacity")
    model_version = backend.handshake(hello)
    _ensure_reply_send_buffer(conn, backend, hello.batch_capacity)
    write_frame_into(
        conn,
        write_buf,
        FRAME_HELLO_ACK,
        HelloAck(PROTOCOL_VERSION, model_version).encode(),
    )
    return _ConnectionState(
        feature_schema_hash=hello.feature_schema_hash,
        batch_capacity=hello.batch_capacity,
        engine_id=hello.engine_id,
        engine_version=hello.engine_version,
        action_set_hash=hello.action_set_hash,
    )


def _ensure_reply_send_buffer(conn: socket.socket, backend: StubBackend, capacity: int) -> None:
    """The pipelined loop's deadlock-freedom invariant: a reply write must
    never block (the client may be mid-write of its next request and not
    reading). The kernel guarantees that when the send buffer holds one
    full reply frame. Verified loudly at handshake instead of deadlocking
    at the first full-size batch."""
    manifest = getattr(backend, "manifest", None)
    schema = getattr(manifest, "feature_schema", None)
    max_actions = getattr(schema, "max_actions", None)
    if max_actions is None:
        # Schema-less backends (the stub) get best-effort sizing only.
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
        return
    # PIPELINE_DEPTH replies can be queued back-to-back while the client
    # is mid-write of its next request; the send buffer must hold all of
    # them for the reply writes to never block.
    reply_frame = (45 + capacity * 4 + capacity * int(max_actions) * 4) * PIPELINE_DEPTH
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, reply_frame)
    achieved = conn.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    if achieved < reply_frame:
        raise ProtocolError(
            ERROR_CAPACITY,
            f"send buffer {achieved} cannot hold {PIPELINE_DEPTH} reply frames ({reply_frame}); "
            "raise net.core.wmem_max or lower the eval batch capacity",
        )


def _handle_ping(conn: socket.socket, write_buf: bytearray, payload: memoryview) -> None:
    if len(payload) != 8:
        raise ProtocolError(ERROR_MALFORMED, "bad PING length")
    write_frame_into(conn, write_buf, FRAME_PONG, payload)


def _parse_eval(state: _ConnectionState, payload: memoryview) -> tuple[int, BatchView]:
    if len(payload) < 8:
        raise ProtocolError(ERROR_MALFORMED, "EVAL frame truncated")
    batch_id = struct.unpack_from("<Q", payload, 0)[0]
    try:
        batch = BatchView.parse(payload[8:])
    except EncodingError as error:
        raise ProtocolError(ERROR_ENCODING, str(error)) from error
    if batch.feature_schema_hash != state.feature_schema_hash:
        raise ProtocolError(ERROR_SCHEMA, "feature schema hash mismatch")
    if batch.batch_capacity != state.batch_capacity:
        raise ProtocolError(ERROR_CAPACITY, "batch capacity mismatch")
    return batch_id, batch


def _flush_oldest(
    conn: socket.socket,
    write_buf: bytearray,
    backend: StubBackend,
    pending: deque[tuple[int, object]],
) -> None:
    batch_id, handle = pending.popleft()
    result = backend.finish(handle)
    write_frame_into(
        conn,
        write_buf,
        FRAME_EVAL_RESULT,
        struct.pack("<Q", batch_id),
        bytes(result.model_version),
        result.payload,
    )


def _send_error(conn: socket.socket, write_buf: bytearray, code: int, message: str) -> None:
    try:
        write_frame_into(conn, write_buf, FRAME_ERROR, encode_error(code, message))
    except OSError:
        pass
