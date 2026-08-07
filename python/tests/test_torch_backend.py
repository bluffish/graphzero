from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from gz.checkpoints import DirectorySource, publish_checkpoint
from gz.codec import BatchView, FeatureSchemaConfig
from gz.common import ActionSetHash, EngineIdentity, EngineId, EngineVersion, FeatureSchemaHash
from gz.evaluator import TorchBackend
from gz.evaluator.backends import PIPELINE_DEPTH
from gz.model.exphormer import ArchConfig, build_model
from gz.proto import BATCH_ENCODING_VERSION, Hello, PROTOCOL_VERSION, ProtocolError
from python.tests.test_codec import _layout

torch = pytest.importorskip("torch")

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_backend_serves_bounded_joint_board_checkpoint(tmp_path: Path) -> None:
    view = fixture_view()
    manifest = publish_random_checkpoint(tmp_path, view, seed=11)
    backend = make_backend(tmp_path)
    assert backend.handshake(make_hello(view, manifest.feature_schema_hash)) == manifest.model_version

    result = backend.eval(view, manifest.model_version)
    values, logits = decode_output(result.payload, view)

    assert result.model_version == manifest.model_version
    assert np.isfinite(values).all()
    assert np.isfinite(logits).all()
    assert np.abs(values).max() < 1.0
    assert np.abs(logits).max() <= 10.0


def test_backend_rejects_wrong_schema(tmp_path: Path) -> None:
    view = fixture_view()
    publish_random_checkpoint(tmp_path, view)
    backend = make_backend(tmp_path)
    with pytest.raises(ProtocolError, match="feature schema hash mismatch"):
        backend.handshake(make_hello(view, FeatureSchemaHash.from_bytes(b"x" * 32)))


def test_backend_rejects_wrong_engine_identity(tmp_path: Path) -> None:
    view = fixture_view()
    manifest = publish_random_checkpoint(tmp_path, view)
    backend = make_backend(tmp_path)
    hello = make_hello(view, manifest.feature_schema_hash)
    hello = Hello(
        protocol_version=hello.protocol_version,
        encoding_version=hello.encoding_version,
        feature_schema_hash=hello.feature_schema_hash,
        batch_capacity=hello.batch_capacity,
        engine_identity=EngineIdentity.from_parts(
            EngineId.from_bytes(b"x" * 16),
            hello.engine_version,
            hello.action_set_hash,
        ),
    )

    with pytest.raises(ProtocolError, match="engine identity mismatch"):
        backend.handshake(hello)


def test_backend_hot_swap_keeps_leased_generation_until_release(tmp_path: Path) -> None:
    view = fixture_view()
    first = publish_random_checkpoint(tmp_path, view, seed=11)
    backend = make_backend(tmp_path)
    backend.handshake(make_hello(view, first.feature_schema_hash))
    first_generation, first_version = backend.model_generation()

    second = publish_random_checkpoint(tmp_path, view, seed=12)
    backend._poll_once()
    backend.apply_pending_swap()
    second_generation, second_version = backend.model_generation()

    assert second_generation > first_generation
    assert second_version == second.model_version
    assert backend.eval(view, first_version).model_version == first_version
    backend.release_model_generation(first_generation, first_version)
    with pytest.raises(ProtocolError, match="unavailable"):
        backend.eval(view, first_version)


def test_backend_resident_cap_defers_third_checkpoint_until_release(tmp_path: Path) -> None:
    view = fixture_view()
    first = publish_random_checkpoint(tmp_path, view, seed=11)
    backend = make_backend(tmp_path)
    backend.handshake(make_hello(view, first.feature_schema_hash))
    first_generation, first_version = backend.model_generation()

    second = publish_random_checkpoint(tmp_path, view, seed=12)
    backend._poll_once()
    backend.apply_pending_swap()
    third = publish_random_checkpoint(tmp_path, view, seed=13)
    backend._poll_once()
    backend.apply_pending_swap()
    assert backend.model_generation()[1] == second.model_version

    backend.release_model_generation(first_generation, first_version)
    backend._poll_once()
    backend.apply_pending_swap()
    assert backend.model_generation()[1] == third.model_version


def test_larger_resident_cap_adopts_third_leased_checkpoint(tmp_path: Path) -> None:
    view = fixture_view()
    first = publish_random_checkpoint(tmp_path, view, seed=11)
    backend = make_backend(tmp_path, max_resident_model_generations=3)
    backend.handshake(make_hello(view, first.feature_schema_hash))

    publish_random_checkpoint(tmp_path, view, seed=12)
    backend._poll_once()
    backend.apply_pending_swap()
    third = publish_random_checkpoint(tmp_path, view, seed=13)
    backend._poll_once()
    backend.apply_pending_swap()

    assert backend.model_generation()[1] == third.model_version


def test_backend_rejects_incompatible_latest_and_keeps_serving(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    view = fixture_view()
    first = publish_random_checkpoint(tmp_path, view, seed=11)
    backend = make_backend(tmp_path)
    backend.handshake(make_hello(view, first.feature_schema_hash))
    publish_random_checkpoint(
        tmp_path,
        view,
        seed=12,
        feature_schema_hash=FeatureSchemaHash.from_bytes(b"z" * 32),
    )

    backend._poll_once()
    backend.apply_pending_swap()

    assert backend.model_generation()[1] == first.model_version
    assert backend.eval(view, first.model_version).model_version == first.model_version
    assert "feature schema hash mismatch" in capfd.readouterr().err


def test_backend_rejects_hot_swap_for_another_action_domain(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    view = fixture_view()
    first = publish_random_checkpoint(tmp_path, view, seed=11)
    backend = make_backend(tmp_path)
    backend.handshake(make_hello(view, first.feature_schema_hash))
    publish_random_checkpoint(
        tmp_path,
        view,
        seed=12,
        action_set_hash=ActionSetHash.from_bytes(b"x" * 32),
    )

    backend._poll_once()
    backend.apply_pending_swap()

    assert backend.model_generation()[1] == first.model_version
    assert "engine identity mismatch" in capfd.readouterr().err


def test_stage_always_copies_current_opponent_board(tmp_path: Path) -> None:
    raw = bytearray((FIXTURES / "batch_expander.gzfb").read_bytes())
    view = BatchView.parse(raw)
    manifest = publish_random_checkpoint(tmp_path, view)
    backend = make_backend(tmp_path)
    backend.handshake(make_hello(view, manifest.feature_schema_hash))

    first = backend.stage(view)
    original = int(first.tensors.opponent_node_tokens[0, 0])
    layout = _layout(
        view.batch_capacity,
        view.dims.max_nodes,
        view.dims.max_edges,
        view.dims.max_actions,
        view.dims.max_subjects,
        view.dims.node_attr_dim,
    )
    struct.pack_into("<H", raw, layout["opponent_node_tokens"], original + 1)
    changed = backend.stage(BatchView.parse(raw))

    assert int(changed.tensors.opponent_node_tokens[0, 0]) == original + 1


def test_handshake_warms_each_serving_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    view = fixture_view()
    manifest = publish_random_checkpoint(tmp_path, view)
    backend = make_backend(tmp_path)
    calls = 0
    run = backend._run_runner

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return run(*args, **kwargs)

    monkeypatch.setattr(backend, "_run_runner", counted)
    backend.handshake(make_hello(view, manifest.feature_schema_hash))
    assert calls == PIPELINE_DEPTH


def fixture_view() -> BatchView:
    return BatchView.parse((FIXTURES / "batch_expander.gzfb").read_bytes())


def make_backend(root: Path, max_resident_model_generations: int = 2) -> TorchBackend:
    return TorchBackend(
        DirectorySource(root),
        device="cpu",
        compile_model=False,
        max_resident_model_generations=max_resident_model_generations,
        poll_interval=0.0,
    )


def publish_random_checkpoint(
    root: Path,
    view: BatchView,
    *,
    seed: int = 11,
    feature_schema_hash: FeatureSchemaHash | None = None,
    action_set_hash: ActionSetHash | None = None,
):
    schema = FeatureSchemaConfig(
        name="expander-test",
        node_vocab_size=8,
        node_attr_dim=view.dims.node_attr_dim,
        edge_type_count=3,
        action_kind_vocab_size=8,
        max_nodes=view.dims.max_nodes,
        max_edges=view.dims.max_edges,
        max_actions=view.dims.max_actions,
        max_subjects=view.dims.max_subjects,
        expander_degree=2,
        expander_seed=0,
    )
    arch = ArchConfig(dim=16, layers=1, heads=4, ffn_dim=32, dropout=0.0)
    torch.manual_seed(seed)
    model = build_model(schema, arch)
    return publish_checkpoint(
        root,
        model.state_dict(),
        arch_name=arch.name,
        arch_config=arch.to_manifest_dict(),
        arch_config_hash=arch.hash(),
        feature_schema=schema,
        feature_schema_hash=feature_schema_hash or view.feature_schema_hash,
        engine_identity=EngineIdentity.from_parts(
            EngineId.from_bytes(b"e" * 16),
            EngineVersion.from_bytes(b"v" * 16),
            action_set_hash or ActionSetHash.from_bytes(b"a" * 32),
        ),
        training_step=0,
        run_id="run",
    )


def make_hello(view: BatchView, schema_hash: FeatureSchemaHash) -> Hello:
    return Hello(
        protocol_version=PROTOCOL_VERSION,
        encoding_version=BATCH_ENCODING_VERSION,
        feature_schema_hash=schema_hash,
        batch_capacity=view.batch_capacity,
        engine_identity=EngineIdentity.from_parts(
            EngineId.from_bytes(b"e" * 16),
            EngineVersion.from_bytes(b"v" * 16),
            ActionSetHash.from_bytes(b"a" * 32),
        ),
    )


def decode_output(payload: memoryview, view: BatchView) -> tuple[np.ndarray, np.ndarray]:
    raw = bytes(payload)
    assert raw[:4] == b"GZFO"
    version, rows, max_actions = struct.unpack_from("<III", raw, 4)
    assert (version, rows, max_actions) == (
        BATCH_ENCODING_VERSION,
        view.row_count,
        view.max_actions,
    )
    values = np.frombuffer(raw, dtype="<f4", count=rows, offset=16)
    logits = np.frombuffer(
        raw,
        dtype="<f4",
        count=int(view.action_count[:rows].sum()),
        offset=16 + rows * 4,
    )
    return values, logits
