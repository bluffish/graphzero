from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from gz.checkpoints import CheckpointManifest, DirectorySource, ResolvedCheckpoint
from gz.model.exphormer import (
    ArchConfig,
    build_model,
    initialize_policy,
    initialize_value,
)
from gz.trainer.checkpointing import (
    checkpoint_due as _checkpoint_due,
    load_initial_checkpoint as _load_initial_checkpoint,
    permanent_checkpoint_pointers as _permanent_checkpoint_pointers,
    prune_training_checkpoints as _prune_training_checkpoints,
    resolve_actor_checkpoint as _resolve_actor_checkpoint,
    validate_checkpoint_engine_identity as _validate_checkpoint_engine_identity,
)
from gz.trainer.config import (
    RunConfig,
    load_config,
    resolved_trainer_seeds,
)
from gz.trainer.data import TrainingStager
from gz.trainer.loop import TrainerLoop
from gz.trainer.processes import (
    SelfplayStage,
    check_child,
    check_memory,
    init_replay,
    spawn_replay_serve,
    stop_child,
)
from gz.trainer.publish import EmaWeights, publish_ema
from gz.trainer.runtime import (
    seed_model as _seed_model,
    set_matmul_precision as _set_matmul_precision,
    trainer_loop_config,
)
from gz.trainer.sampler import SampleClient
from gz.trainer.sample_protocol import SampleAck
from gz.trainer.sampling import (
    SamplePrefetcher,
    SampledBatches,
    cumulative_reuse as _cumulative_reuse,
    required_episodes as _required_episodes,
    required_produced_rows as _required_produced_rows,
    sample_training_batches as _sample_training_batches,
)
from gz.trainer.step import StepMetrics
from gz.trainer.telemetry import (
    MetricsWriter,
    PerfWindow,
    WandbRun,
    symmetric_step_fields,
)


@dataclass(slots=True)
class _PreparedRun:
    learner_manifest: CheckpointManifest
    resume_resolved: ResolvedCheckpoint | None
    start_step: int
    model: object | None = None
    ema: EmaWeights | None = None
    published_snapshot: dict[str, object] | None = None


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    _set_matmul_precision(config.trainer.matmul_precision)
    model_seed, data_seed = resolved_trainer_seeds(config.trainer)
    for path in (config.paths.replay_dir, config.paths.checkpoint_dir, config.paths.run_dir):
        path.mkdir(parents=True, exist_ok=True)
    metrics = MetricsWriter(config.paths.run_dir / "metrics.jsonl", WandbRun.start(config))
    prepared = _prepare_run(config, metrics, model_seed)
    _record_actor_checkpoint(config, prepared.learner_manifest, metrics)
    _prune_training_checkpoints(config)

    stage = SelfplayStage.start(config)
    session = None
    failed = True
    try:
        session = _TrainingSession(config, prepared, stage, metrics, data_seed)
        session.train()
        failed = False
    finally:
        if session is not None:
            session.stop_prefetch()
        stage.terminate()
        try:
            if session is not None:
                session.join_prefetch(suppress_errors=failed)
        finally:
            stage.close()
    metrics.finish()


def _prepare_run(
    config: RunConfig,
    metrics: MetricsWriter,
    model_seed: int,
) -> _PreparedRun:
    if config.trainer.resume:
        resolved = DirectorySource(str(config.paths.checkpoint_dir)).resolve_latest()
        start_step = resolved.manifest.training_step
        if start_step >= config.trainer.total_steps:
            raise RuntimeError("resume checkpoint is at or past total_steps")
        return _PreparedRun(
            learner_manifest=resolved.manifest,
            resume_resolved=resolved,
            start_step=start_step,
        )

    init_replay(config)
    serve = spawn_replay_serve(config)
    sampler = None
    try:
        sampler = SampleClient(
            config.paths.sample_socket,
            startup_timeout=config.trainer.startup_timeout,
            reconnect_limit=config.trainer.reconnect_limit,
        )
        sampler.wait_until_ready(
            0,
            alive_check=lambda: check_child(serve, "replay-serve"),
        )
        _seed_model(model_seed)
        model = build_model(sampler.feature_schema, config.arch)
        initialize_policy(model, config.trainer.policy_init)
        initialize_value(model, config.trainer.value_init)
        if config.trainer.init_checkpoint:
            resolved = _load_initial_checkpoint(
                model,
                config.trainer.init_checkpoint,
                sampler.feature_schema_hash,
                config.arch,
                scope=config.trainer.init_checkpoint_scope,
            )
            _validate_checkpoint_engine_identity(
                resolved.manifest,
                sampler.engine_identity,
            )
            metrics.write(
                {
                    "event": "init_checkpoint",
                    "source_model_version": resolved.manifest.model_version.hex(),
                    "source_training_step": resolved.manifest.training_step,
                    "scope": config.trainer.init_checkpoint_scope,
                }
            )
        model = model.to(config.trainer.device)
        ema = EmaWeights(model, config.trainer.ema_decay)
        manifest = publish_ema(
            config.paths.checkpoint_dir,
            ema,
            schema=sampler.feature_schema,
            schema_hash=sampler.feature_schema_hash,
            arch=config.arch,
            training_step=0,
            run_id=config.paths.run_dir.name,
            engine_identity=sampler.engine_identity,
        )
        param_norm, _ = ema.norms(None)
        published_snapshot = ema.state_dict()
        metrics.write(
            {
                "event": "publish",
                "training_step": 0,
                "model_version": manifest.model_version.hex(),
                "param_norm": param_norm,
                "update_norm": 0.0,
            }
        )
        return _PreparedRun(
            learner_manifest=manifest,
            resume_resolved=None,
            start_step=0,
            model=model,
            ema=ema,
            published_snapshot=published_snapshot,
        )
    finally:
        if sampler is not None:
            sampler.close()
        stop_child(serve)


def _record_actor_checkpoint(
    config: RunConfig,
    learner_manifest: CheckpointManifest,
    metrics: MetricsWriter,
) -> None:
    actor_checkpoint = _resolve_actor_checkpoint(
        config,
        learner_manifest.feature_schema_hash,
    )
    metrics.write(
        {
            "event": "actor_checkpoint",
            "checkpoint_dir": str(config.paths.actor_checkpoint_dir),
            "checkpoint_pointer": config.selfplay.actor_checkpoint_pointer,
            "model_version": actor_checkpoint.manifest.model_version.hex(),
            "training_step": actor_checkpoint.manifest.training_step,
            "arch_config_hash": actor_checkpoint.manifest.arch_config_hash,
            "frozen": config.selfplay.eval_poll_interval == 0.0,
        }
    )


class _TrainingSession:
    def __init__(
        self,
        config: RunConfig,
        prepared: _PreparedRun,
        stage: SelfplayStage,
        metrics: MetricsWriter,
        data_seed: int,
    ) -> None:
        self.config = config
        self.stage = stage
        self.sampler = stage.sampler
        self.metrics = metrics
        self.data_seed = data_seed
        self.model, self.ema, self.published_snapshot = self._load_model(prepared)
        self.last_published_step = prepared.start_step
        self.pending_publish_step: int | None = None
        self.produced_floor = 0
        self.episodes_floor = 0
        self.window = PerfWindow(0, 0)

        schema = self.sampler.feature_schema
        score_target = (
            config.trainer.terminal_score_target
            if config.trainer.terminal_score_weight > 0.0
            else None
        )
        self.policy_stager = TrainingStager(
            schema,
            self.sampler.max_batch,
            config.trainer.device,
            terminal_score_target=score_target,
        )
        self.value_stager = (
            TrainingStager(
                schema,
                self.sampler.max_batch,
                config.trainer.device,
                terminal_score_target=score_target,
            )
            if config.trainer.value_batch > 0
            else None
        )
        self.loop = TrainerLoop(
            self.model,
            trainer_loop_config(
                config.trainer,
                symmetric_mask_stop=config.selfplay.mask_stop,
            ),
        )
        self.loop.step_index = prepared.start_step
        self._record_value_tasks()
        self.prefetcher = self._start_prefetcher(prepared.start_step)

    def _load_model(
        self,
        prepared: _PreparedRun,
    ) -> tuple[object, EmaWeights, dict[str, object]]:
        _validate_checkpoint_engine_identity(
            prepared.learner_manifest,
            self.sampler.engine_identity,
        )
        if prepared.resume_resolved is None:
            if (
                prepared.model is None
                or prepared.ema is None
                or prepared.published_snapshot is None
            ):
                raise RuntimeError("fresh training state is incomplete")
            return prepared.model, prepared.ema, prepared.published_snapshot

        from gz.checkpoints.weights import load_state_dict

        resolved = prepared.resume_resolved
        if resolved.manifest.feature_schema_hash != self.sampler.feature_schema_hash:
            raise RuntimeError("resume checkpoint feature schema does not match the store")
        if ArchConfig.from_manifest_dict(resolved.manifest.arch_config) != self.config.arch:
            raise RuntimeError("resume checkpoint arch does not match [arch] config")
        model = build_model(self.sampler.feature_schema, self.config.arch).to(
            self.config.trainer.device
        )
        model.load_state_dict(load_state_dict(resolved.weights_path))
        ema = EmaWeights(model, self.config.trainer.ema_decay)
        self.metrics.write(
            {
                "event": "resume",
                "training_step": prepared.start_step,
                "model_version": resolved.manifest.model_version.hex(),
            }
        )
        return model, ema, ema.state_dict()

    def _record_value_tasks(self) -> None:
        trainer = self.config.trainer
        self.metrics.write(
            {
                "event": "value_tasks",
                "auxiliary_heads": self.config.arch.auxiliary_heads,
                "horizons": [8, 32],
                "score_scale": (
                    self.sampler.feature_schema.max_nodes
                    if trainer.terminal_score_target == "node-count"
                    else None
                ),
                "terminal_score_unit": (
                    "nodes"
                    if trainer.terminal_score_target == "node-count"
                    else "milliseconds"
                ),
                "value_final_weight": trainer.value_final_weight,
                "value_v8_weight": trainer.value_v8_weight,
                "value_v32_weight": trainer.value_v32_weight,
                "terminal_score_weight": trainer.terminal_score_weight,
                "terminal_score_target": trainer.terminal_score_target,
                "soft_policy_weight": trainer.soft_policy_weight,
                "soft_policy_temperature": trainer.soft_policy_temperature,
                "soft_policy_trunk_grad_scale": trainer.soft_policy_trunk_grad_scale,
            }
        )

    def _start_prefetcher(self, start_step: int) -> SamplePrefetcher | None:
        trainer = self.config.trainer
        if not trainer.prefetch:
            return None
        prefetcher = SamplePrefetcher(
            self.sampler,
            trainer.batch,
            trainer.window_rows,
            trainer.value_batch,
            trainer.value_window_rows or trainer.window_rows,
            trainer.value_terminal_fraction,
            self.data_seed,
            trainer.total_steps,
            trainer.max_reuse,
            trainer.reuse_gate_interval,
            trainer.reuse_gate_episodes,
            start_step=start_step,
            value_sampler=(
                self.sampler.fork()
                if trainer.value_batch > 0 and trainer.parallel_value_sampling
                else None
            ),
        )
        prefetcher.start()
        return prefetcher

    def train(self) -> None:
        trainer = self.config.trainer
        for step in range(self.loop.step_index, trainer.total_steps):
            check_child(self.stage.child, "selfplay")
            if step % 50 == 0:
                check_memory(trainer.min_available_gb)
            sample_started = time.perf_counter()
            self._wait_for_reuse_gate(step)
            if self.pending_publish_step is not None:
                self._publish(self.pending_publish_step)
                self.pending_publish_step = None
            samples = self._next_samples(step)
            train_started = time.perf_counter()
            with_metrics = step % trainer.log_interval == 0
            step_metrics = self._train_batch(samples, with_metrics)
            self.ema.update(self.model)
            self.window.record(sample_started, train_started, time.perf_counter())
            if with_metrics:
                assert step_metrics is not None
                self._write_step_metrics(step, step_metrics)
            if _checkpoint_due(trainer, step + 1):
                if trainer.publish_lag_blocks:
                    self.pending_publish_step = step + 1
                else:
                    self._publish(step + 1)
            if trainer.step_sleep:
                time.sleep(trainer.step_sleep)

        self.stop_prefetch()
        if self.pending_publish_step is not None:
            self._publish(self.pending_publish_step)
        elif not _checkpoint_due(trainer, trainer.total_steps):
            self._publish(trainer.total_steps)

    def _wait_for_reuse_gate(self, step: int) -> None:
        trainer = self.config.trainer
        if trainer.max_reuse <= 0 and trainer.reuse_gate_episodes <= 0:
            return
        needed_rows = _required_produced_rows(
            step,
            trainer.batch,
            trainer.max_reuse,
            trainer.reuse_gate_interval,
        )
        needed_episodes = _required_episodes(
            step,
            trainer.reuse_gate_interval,
            trainer.reuse_gate_episodes,
        )
        while (
            self.produced_floor < needed_rows
            or self.episodes_floor < needed_episodes
        ):
            ack = self._refresh()
            self.produced_floor = ack.produced_rows
            self.episodes_floor = ack.episodes
            if (
                self.produced_floor >= needed_rows
                and self.episodes_floor >= needed_episodes
            ):
                return
            check_child(self.stage.child, "selfplay")
            time.sleep(0.1)

    def _next_samples(self, step: int) -> SampledBatches:
        if self.prefetcher is not None:
            return self.prefetcher.next()
        ack = self.sampler.refresh()
        trainer = self.config.trainer
        return _sample_training_batches(
            self.sampler,
            policy_batch=trainer.batch,
            policy_window_rows=trainer.window_rows,
            value_batch=trainer.value_batch,
            value_window_rows=trainer.value_window_rows or trainer.window_rows,
            value_terminal_fraction=trainer.value_terminal_fraction,
            run_seed=self.data_seed,
            step=step,
            produced_rows=ack.produced_rows,
        )

    def _train_batch(
        self,
        samples: SampledBatches,
        with_metrics: bool,
    ) -> StepMetrics | None:
        policy_batch = self.policy_stager.copy(
            samples.policy.batch,
            samples.policy.targets,
        )
        value_batch = None
        if self.value_stager is not None:
            if samples.value is None:
                raise RuntimeError("value sampler returned no value batch")
            value_batch = self.value_stager.copy(
                samples.value.batch,
                samples.value.targets,
            )
        return self.loop.train_step(
            policy_batch,
            value_batch,
            with_metrics=with_metrics,
        )

    def _write_step_metrics(self, step: int, step_metrics: StepMetrics) -> None:
        ack = self._refresh()
        produced = ack.produced_rows
        episodes = ack.episodes
        trainer = self.config.trainer
        value_batch = trainer.value_batch
        if value_batch == 0 and trainer.value_weight != 0.0:
            value_batch = trainer.batch
        stop_rate = ack.episodes_stopped / episodes if episodes else 0.0
        record = {
            "event": "step",
            "timestamp": time.time(),
            "step": step_metrics.step,
            "policy_loss": step_metrics.policy_loss,
            "policy_target_entropy": step_metrics.policy_target_entropy,
            "policy_kl": step_metrics.policy_kl,
            "policy_uniform_advantage": step_metrics.policy_uniform_advantage,
            "policy_logit_std": step_metrics.policy_logit_std,
            "policy_logit_abs_mean": step_metrics.policy_logit_abs_mean,
            "soft_policy_loss": step_metrics.soft_policy_loss,
            "soft_policy_kl": step_metrics.soft_policy_kl,
            "soft_policy_target_entropy": step_metrics.soft_policy_target_entropy,
            "value_loss": step_metrics.value_loss,
            "value_final_loss": step_metrics.value_final_loss,
            "value_v8_loss": step_metrics.value_v8_loss,
            "value_v32_loss": step_metrics.value_v32_loss,
            "terminal_score_loss": step_metrics.terminal_score_loss,
            "terminal_score_mae": step_metrics.terminal_score_mae,
            "terminal_score_bias": step_metrics.terminal_score_bias,
            "loss": step_metrics.loss,
            "grad_norm": step_metrics.grad_norm,
            "lr": step_metrics.lr,
            "fraction_valid": step_metrics.fraction_valid,
            "label_mean": step_metrics.label_mean,
            "terminal_cost_ema": ack.episode_cost_ema,
            "terminal_cost_best": ack.best_cost,
            "produced_rows": produced,
            "policy_reuse": _cumulative_reuse(step, trainer.batch, produced),
            "value_reuse": _cumulative_reuse(step, value_batch, produced),
            "stop_rate": stop_rate,
            "episode_len_ema": ack.episode_len_ema,
            "stop_rate_ema": ack.stop_rate_ema,
            **self.stage.stats.step_fields(),
        }
        record["value_accuracy"] = step_metrics.value_accuracy
        if "actor_training_step" in record:
            record["actor_training_step_lag"] = max(
                step_metrics.step - int(record["actor_training_step"]), 0
            )
        record["learner_win_rate"] = step_metrics.learner_win_rate
        if ack.learner_win_rate_ema >= 0.0:
            record["learner_win_rate_ema"] = ack.learner_win_rate_ema
        if ack.value_sign_accuracy_early_ema >= 0.0:
            record["value_sign_accuracy_early_ema"] = ack.value_sign_accuracy_early_ema
        if ack.value_sign_accuracy_late_ema >= 0.0:
            record["value_sign_accuracy_late_ema"] = ack.value_sign_accuracy_late_ema
        if ack.episode_latency_ema >= 0.0:
            record["episode_latency_s"] = ack.episode_latency_ema
        record.update(symmetric_step_fields(ack, episodes))
        record.update(self.window.drain(produced, episodes))
        record.update(step_metrics.logging_fields())
        if trainer.terminal_score_target == "log1p-cost-ms":
            record["terminal_latency_mae_ms"] = record.pop("terminal_score_mae")
            record["terminal_latency_bias_ms"] = record.pop("terminal_score_bias")
        self.metrics.write(record)

    def _publish(self, training_step: int) -> None:
        if training_step == self.last_published_step:
            return
        if training_step < self.last_published_step:
            raise RuntimeError("training checkpoints must publish in step order")
        manifest = publish_ema(
            self.config.paths.checkpoint_dir,
            self.ema,
            schema=self.sampler.feature_schema,
            schema_hash=self.sampler.feature_schema_hash,
            arch=self.config.arch,
            training_step=training_step,
            run_id=self.config.paths.run_dir.name,
            engine_identity=self.sampler.engine_identity,
            checkpoint_pointers=_permanent_checkpoint_pointers(
                self.config.trainer,
                training_step,
            ),
        )
        pruned = _prune_training_checkpoints(self.config)
        param_norm, update_norm = self.ema.norms(self.published_snapshot)
        self.published_snapshot = self.ema.state_dict()
        self.metrics.write(
            {
                "event": "publish",
                "training_step": training_step,
                "model_version": manifest.model_version.hex(),
                "param_norm": param_norm,
                "update_norm": update_norm,
                "checkpoints_pruned": len(pruned),
            }
        )
        self.last_published_step = training_step

    def _refresh(self) -> SampleAck:
        if self.prefetcher is not None:
            return self.prefetcher.refresh()
        return self.sampler.refresh()

    def stop_prefetch(self) -> None:
        if self.prefetcher is not None:
            self.prefetcher.stop()

    def join_prefetch(self, *, suppress_errors: bool) -> None:
        if self.prefetcher is None:
            return
        try:
            self.prefetcher.join()
        except BaseException:
            if not suppress_errors:
                raise
        finally:
            self.prefetcher = None
