from __future__ import annotations

import struct
from dataclasses import dataclass

from gz.codec import FeatureSchemaConfig
from gz.common import (
    ActionSetHash,
    EngineIdentity,
    EngineId,
    EngineVersion,
    FeatureSchemaHash,
)
from gz.proto import ENCODING_VERSION

SAMPLE_PROTOCOL_VERSION = 13
HELLO_ACK_FIXED_LEN = 224

FRAME_HELLO = 1
FRAME_HELLO_ACK = 2
FRAME_SAMPLE = 3
FRAME_SAMPLE_RESULT = 4
FRAME_ERROR = 5

_HELLO = struct.Struct("<II")
_SAMPLE_REQUEST = struct.Struct("<IIQQ")
_SAMPLE_RESULT_PREFIX = struct.Struct("<I")
_HELLO_ACK = struct.Struct("<I32sIQQQffffff20sffI10f16s16s32s")
assert _HELLO_ACK.size == HELLO_ACK_FIXED_LEN


class SampleError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SymmetricSelfplayMetrics:
    p1_win_rate_ema: float
    p2_win_rate_ema: float
    draw_rate_ema: float
    seat_advantage_ema: float
    p1_terminal_cost_ema: float
    p2_terminal_cost_ema: float
    mean_terminal_cost_ema: float
    terminal_cost_margin_ema: float
    terminal_cost_best: float
    p1_episode_len_ema: float
    p2_episode_len_ema: float
    game_len_ema: float
    episode_len_margin_ema: float


@dataclass(frozen=True, slots=True)
class SampleAck:
    feature_schema_hash: FeatureSchemaHash
    engine_identity: EngineIdentity
    max_batch: int
    produced_rows: int
    episodes: int
    episodes_stopped: int
    episode_cost_ema: float
    episode_len_ema: float
    stop_rate_ema: float
    learner_win_rate_ema: float
    value_sign_accuracy_early_ema: float
    value_sign_accuracy_late_ema: float
    episode_latency_ema: float
    best_cost: float
    symmetric_selfplay: SymmetricSelfplayMetrics | None
    feature_schema: FeatureSchemaConfig

    @property
    def engine_id(self) -> EngineId:
        return self.engine_identity.engine_id

    @property
    def engine_version(self) -> EngineVersion:
        return self.engine_identity.engine_version

    @property
    def action_set_hash(self) -> ActionSetHash:
        return self.engine_identity.action_set_hash


def encode_hello() -> bytes:
    return _HELLO.pack(SAMPLE_PROTOCOL_VERSION, ENCODING_VERSION)


def encode_sample_request(
    batch: int, window_rows: int, seed: int, *, terminal_only: bool = False
) -> bytes:
    if batch <= 0 or window_rows <= 0:
        raise ValueError("batch and window must be positive")
    return _SAMPLE_REQUEST.pack(batch, int(terminal_only), window_rows, seed)


def split_sample_result(payload: memoryview) -> tuple[memoryview, memoryview]:
    if len(payload) < _SAMPLE_RESULT_PREFIX.size:
        raise SampleError("sample result truncated")
    gzfb_len = _SAMPLE_RESULT_PREFIX.unpack_from(payload)[0]
    end = _SAMPLE_RESULT_PREFIX.size + gzfb_len
    if len(payload) < end:
        raise SampleError("sample gzfb truncated")
    return payload[_SAMPLE_RESULT_PREFIX.size:end], payload[end:]


def decode_ack(payload: memoryview) -> SampleAck:
    if len(payload) < HELLO_ACK_FIXED_LEN:
        raise SampleError("sample HELLO_ACK truncated")
    (
        protocol_version,
        feature_schema_hash,
        max_batch,
        produced_rows,
        episodes,
        episodes_stopped,
        cost_ema,
        len_ema,
        stop_ema,
        win_ema,
        latency_ema,
        best_cost,
        retired_fixed_root,
        value_sign_early_ema,
        value_sign_late_ema,
        symmetric_present,
        *tail,
    ) = _HELLO_ACK.unpack_from(payload)
    if protocol_version != SAMPLE_PROTOCOL_VERSION:
        raise SampleError("sample protocol version mismatch")
    if any(retired_fixed_root):
        raise SampleError("sample HELLO_ACK uses retired fixed-root telemetry")
    if symmetric_present not in (0, 1):
        raise SampleError("sample HELLO_ACK has invalid symmetric metrics flag")
    symmetric_values = tail[:10]
    engine_id, engine_version, action_set_hash = tail[10:]
    symmetric = _symmetric_metrics(symmetric_values) if symmetric_present else None
    try:
        engine_identity = EngineIdentity.from_parts(
            EngineId.from_bytes(engine_id),
            EngineVersion.from_bytes(engine_version),
            ActionSetHash.from_bytes(action_set_hash),
        )
    except ValueError as error:
        raise SampleError(str(error)) from error
    return SampleAck(
        feature_schema_hash=FeatureSchemaHash.from_bytes(feature_schema_hash),
        engine_identity=engine_identity,
        max_batch=max_batch,
        produced_rows=produced_rows,
        episodes=episodes,
        episodes_stopped=episodes_stopped,
        episode_cost_ema=cost_ema,
        episode_len_ema=len_ema,
        stop_rate_ema=stop_ema,
        learner_win_rate_ema=win_ema,
        value_sign_accuracy_early_ema=value_sign_early_ema,
        value_sign_accuracy_late_ema=value_sign_late_ema,
        episode_latency_ema=latency_ema,
        best_cost=best_cost,
        symmetric_selfplay=symmetric,
        feature_schema=FeatureSchemaConfig.decode(payload[HELLO_ACK_FIXED_LEN:]),
    )


def _symmetric_metrics(values: list[float]) -> SymmetricSelfplayMetrics:
    (
        p1_win_rate_ema,
        p2_win_rate_ema,
        draw_rate_ema,
        p1_terminal_cost_ema,
        p2_terminal_cost_ema,
        terminal_cost_margin_ema,
        terminal_cost_best,
        p1_episode_len_ema,
        p2_episode_len_ema,
        episode_len_margin_ema,
    ) = values
    return SymmetricSelfplayMetrics(
        p1_win_rate_ema=p1_win_rate_ema,
        p2_win_rate_ema=p2_win_rate_ema,
        draw_rate_ema=draw_rate_ema,
        seat_advantage_ema=p1_win_rate_ema - p2_win_rate_ema,
        p1_terminal_cost_ema=p1_terminal_cost_ema,
        p2_terminal_cost_ema=p2_terminal_cost_ema,
        mean_terminal_cost_ema=0.5 * (p1_terminal_cost_ema + p2_terminal_cost_ema),
        terminal_cost_margin_ema=terminal_cost_margin_ema,
        terminal_cost_best=terminal_cost_best,
        p1_episode_len_ema=p1_episode_len_ema,
        p2_episode_len_ema=p2_episode_len_ema,
        game_len_ema=p1_episode_len_ema + p2_episode_len_ema,
        episode_len_margin_ema=episode_len_margin_ema,
    )
