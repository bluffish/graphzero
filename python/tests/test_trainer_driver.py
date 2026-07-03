from __future__ import annotations

from pathlib import Path

import pytest

from gz.trainer.driver import WandbRun, load_config


def test_load_config_defaults_and_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[trainer]
batch = 4
total_steps = 3

[selfplay]
lanes = 1
workers_per_lane = 1

[paths]
run_dir = "run"
graphzero_bin = "graphzero-test"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.trainer.batch == 4
    assert config.trainer.total_steps == 3
    assert config.selfplay.lanes == 1
    # Paths are pinned to the trainer's cwd: children (the evaluator) run in
    # their own working directories, so relative paths must not cross over.
    assert config.paths.run_dir == Path.cwd() / "run"
    assert config.paths.replay_dir == Path.cwd() / "run/replay"
    assert config.paths.checkpoint_dir == Path.cwd() / "run/checkpoints"
    assert config.paths.sample_socket == Path.cwd() / "run/sample.sock"
    assert config.paths.graphzero_bin == "graphzero-test"


def test_load_config_rejects_unknown_field(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text("[trainer]\nunknown = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown config fields"):
        load_config(config_path)


def test_load_config_parses_wandb_table(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[wandb]
project = "graphzero"
run_name = "curve-2"

[paths]
run_dir = "run"
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.wandb.project == "graphzero"
    assert config.wandb.run_name == "curve-2"
    assert config.wandb.entity == ""


def test_wandb_run_disabled_without_project(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text('[paths]\nrun_dir = "run"\n', encoding="utf-8")

    assert WandbRun.start(load_config(config_path)) is None


def test_wandb_run_maps_step_records_to_grouped_keys() -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.logged: list[tuple[dict, int]] = []
            self.finished = False

        def log(self, payload: dict, step: int) -> None:
            self.logged.append((payload, step))

        def finish(self) -> None:
            self.finished = True

    fake = FakeRun()
    run = WandbRun(fake)
    run.write(
        {
            "event": "step",
            "step": 7,
            "timestamp": 123.0,
            "policy_loss": 4.5,
            "rows_per_s": 200.0,
            "produced_rows": 4096,
        }
    )
    run.write({"event": "publish", "training_step": 10, "model_version": "ab"})
    run.finish()

    payload, step = run.run.logged[0]
    assert step == 7
    assert payload == {
        "train/policy_loss": 4.5,
        "perf/rows_per_s": 200.0,
        "perf/produced_rows": 4096,
    }
    assert "timestamp" not in payload
    publish_payload, publish_step = run.run.logged[1]
    assert publish_step == 10
    assert publish_payload == {"publish/count": 1, "publish/training_step": 10}
    assert fake.finished


def test_load_config_parses_arch_table(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        '[arch]\ndim = 64\nlayers = 2\n\n[paths]\nrun_dir = "run"\n',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.arch.dim == 64
    assert config.arch.layers == 2
    assert config.arch.heads == 4


def test_load_config_rejects_unreachable_startup_rows(tmp_path: Path) -> None:
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        '[trainer]\nbootstrap_episodes = 4\nmin_startup_rows = 512\n\n'
        '[selfplay]\nmax_steps = 8\n\n[paths]\nrun_dir = "run"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot reach"):
        load_config(config_path)
