from __future__ import annotations

import socket
import struct
import threading
from pathlib import Path

from gz.codec import FeatureSchemaConfig
from gz.common import FeatureSchemaHash
from gz.proto import read_frame, write_frame
from gz.trainer.sampler import SampleClient, decode_ack, step_seed
from python.tests.test_codec import SCHEMA_HASH, make_batch
from python.tests.test_targets import make_targets


def test_sample_client_handshake_and_deterministic_sample(tmp_path: Path) -> None:
    socket_path = tmp_path / "sample.sock"
    raw_batch = make_batch(attr_dim=1)
    raw_targets = make_targets()
    thread = serve_samples(socket_path, produced_rows=[2], responses=[(raw_batch, raw_targets), (raw_batch, raw_targets)])
    client = SampleClient(socket_path, startup_timeout=1.0, backoff=0.01)
    try:
        ack = client.wait_until_ready(1)
        first = client.sample(1, 2, 99)
        second = client.sample(1, 2, 99)

        assert ack.feature_schema == schema_config()
        assert ack.feature_schema_hash == FeatureSchemaHash.from_bytes(SCHEMA_HASH)
        assert first.produced_rows == 2
        assert first.batch.node_count.tolist() == second.batch.node_count.tolist()
        assert first.targets.policy.tolist() == second.targets.policy.tolist()
    finally:
        client.close()
        thread.join(timeout=1)


def test_sample_client_startup_wait_reconnects_until_enough_rows(tmp_path: Path) -> None:
    socket_path = tmp_path / "sample.sock"
    thread = serve_samples(socket_path, produced_rows=[0, 4], responses=[])
    client = SampleClient(socket_path, startup_timeout=1.0, backoff=0.01)
    try:
        ack = client.wait_until_ready(4)

        assert ack.produced_rows == 4
    finally:
        client.close()
        thread.join(timeout=1)


def test_step_seed_is_deterministic_and_step_sensitive() -> None:
    assert step_seed(7, 3) == step_seed(7, 3)
    assert step_seed(7, 3) != step_seed(7, 4)


def test_decode_ack_rejects_truncated() -> None:
    try:
        decode_ack(memoryview(b"short"))
    except Exception as error:
        assert "truncated" in str(error)
    else:
        raise AssertionError("decode_ack accepted truncated payload")


def serve_samples(
    socket_path: Path,
    *,
    produced_rows: list[int],
    responses: list[tuple[bytes, bytes]],
) -> threading.Thread:
    ready = threading.Event()

    def run() -> None:
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(socket_path))
            listener.listen(1)
            ready.set()
            response_index = 0
            for produced in produced_rows:
                conn, _ = listener.accept()
                with conn:
                    frame_type, _payload = read_frame(conn, bytearray())
                    assert frame_type == 1
                    write_frame(conn, 2, ack_payload(produced))
                    if produced == 0:
                        continue
                    while response_index < len(responses):
                        frame_type, payload = read_frame(conn, bytearray())
                        assert frame_type == 3
                        assert struct.unpack_from("<I", payload, 0)[0] > 0
                        batch, targets = responses[response_index]
                        response_index += 1
                        write_frame(conn, 4, struct.pack("<I", len(batch)), batch, targets)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(timeout=1)
    return thread


def ack_payload(produced_rows: int) -> bytes:
    return (
        struct.pack("<I", 4)
        + SCHEMA_HASH
        + struct.pack("<I", 2)
        + struct.pack("<Q", produced_rows)
        + struct.pack("<Q", 6)
        + struct.pack("<Q", 2)
        + struct.pack("<fff", 87.5, 12.0, 0.25)
        + struct.pack("<f", 61.0)
        + struct.pack("<I", 1)
        + struct.pack("<f", 150.0)
        + struct.pack("<III", 200, 400, 900)
        + schema_config().encode()
    )


def schema_config() -> FeatureSchemaConfig:
    return FeatureSchemaConfig(
        name="sample-test",
        node_vocab_size=7,
        node_attr_dim=1,
        edge_type_count=2,
        action_kind_vocab_size=8,
        max_nodes=3,
        max_edges=2,
        max_actions=3,
        max_subjects=2,
        expander_degree=0,
        expander_seed=0,
    )
