from __future__ import annotations

import socket
import struct

from gz.proto.errors import ERROR_MALFORMED, ERROR_PROTOCOL, ProtocolError

PROTOCOL_VERSION = 1
# Row/targets encoding version: pairs with the Rust row codec.
ENCODING_VERSION = 3
# Eval-wire batch/output encoding version. Transient bytes only, moves
# independently of the rows.
BATCH_ENCODING_VERSION = 3
MAX_FRAME = 256 * 1024 * 1024

FRAME_HELLO = 1
FRAME_HELLO_ACK = 2
FRAME_EVAL = 3
FRAME_EVAL_RESULT = 4
FRAME_PING = 5
FRAME_PONG = 6
FRAME_ERROR = 7

_KNOWN_TYPES = {
    FRAME_HELLO,
    FRAME_HELLO_ACK,
    FRAME_EVAL,
    FRAME_EVAL_RESULT,
    FRAME_PING,
    FRAME_PONG,
    FRAME_ERROR,
}


def read_frame(sock: socket.socket, buf: bytearray) -> tuple[int, memoryview]:
    _read_exact(sock, buf, 4)
    body_len = struct.unpack_from("<I", buf, 0)[0]
    if body_len == 0:
        raise ProtocolError(ERROR_MALFORMED, "empty frame")
    if body_len > MAX_FRAME:
        raise ProtocolError(ERROR_PROTOCOL, "frame exceeds maximum size")
    _read_exact(sock, buf, body_len)
    frame_type = buf[0]
    if frame_type not in _KNOWN_TYPES:
        raise ProtocolError(ERROR_PROTOCOL, "unknown frame type")
    return frame_type, memoryview(buf)[:body_len][1:]


def write_frame(sock: socket.socket, frame_type: int, *parts: bytes | memoryview) -> None:
    out = bytearray()
    write_frame_into(sock, out, frame_type, *parts)


def write_frame_into(
    sock: socket.socket,
    out: bytearray,
    frame_type: int,
    *parts: bytes | memoryview,
) -> None:
    if frame_type not in _KNOWN_TYPES:
        raise ProtocolError(ERROR_PROTOCOL, "unknown frame type")
    body_len = 1 + sum(len(part) for part in parts)
    if body_len > MAX_FRAME:
        raise ProtocolError(ERROR_PROTOCOL, "frame exceeds maximum size")
    frame_len = 4 + body_len
    if len(out) < frame_len:
        out.extend(b"\x00" * (frame_len - len(out)))
    struct.pack_into("<I", out, 0, body_len)
    out[4] = frame_type
    cursor = 5
    for part in parts:
        part_len = len(part)
        out[cursor : cursor + part_len] = part
        cursor += part_len
    sock.sendall(memoryview(out)[:frame_len])


def _read_exact(sock: socket.socket, buf: bytearray, size: int) -> None:
    if len(buf) < size:
        buf.extend(b"\x00" * (size - len(buf)))
    view = memoryview(buf)
    offset = 0
    while offset < size:
        received = sock.recv_into(view[offset:size])
        if received == 0:
            raise ProtocolError(ERROR_MALFORMED, "unexpected eof")
        offset += received
