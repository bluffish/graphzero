from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

from gz.trainer.config import RunConfig
from gz.trainer.sampler import SampleAck


class SelfplayStatsTracker:
    """Parses eval_stats / measure_stats heartbeats off the selfplay
    stderr pump. The selfplay side emits cumulative counters every 30s;
    step_fields() reports window rates (delta since the last fold) plus
    the cumulative ledger, so batch fill and the measure repeat rate
    are live in wandb instead of dying with the killed process's exit
    summary."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.eval_batches = None
        self.eval_rows = None
        self.eval_at = None
        self.folded = None
        self.measure: dict[str, int] = {}
        self.admission: dict[str, int] = {}

    def observe_eval(self, fields: dict[str, str]) -> None:
        if fields.get("role", "current") != "current":
            return
        try:
            batches = int(fields["batches"])
            rows = int(fields["rows"])
        except (KeyError, ValueError):
            return
        with self.lock:
            self.eval_batches = batches
            self.eval_rows = rows
            self.eval_at = time.time()

    def observe_measure(self, fields: dict[str, str]) -> None:
        try:
            parsed = {
                key: int(fields[key])
                for key in ("appended", "dropped", "finals", "distinct")
            }
        except (KeyError, ValueError):
            return
        with self.lock:
            self.measure = parsed

    def observe_admission(self, fields: dict[str, str]) -> None:
        try:
            parsed = {
                key: int(fields[key])
                for key in (
                    "outstanding",
                    "reserved",
                    "waiting",
                    "max_waiting",
                    "bootstrap_grants",
                    "paced_grants",
                    "eval_capacity_milli",
                    "episode_work_milli",
                    "pressure_gain_milli",
                    "gap_us",
                )
            }
        except (KeyError, ValueError):
            return
        with self.lock:
            self.admission = parsed

    def step_fields(self) -> dict[str, object]:
        with self.lock:
            out: dict[str, object] = {}
            if self.eval_batches is not None:
                out["eval_batches_total"] = self.eval_batches
                out["eval_rows_total"] = self.eval_rows
                if self.folded is not None:
                    prev_batches, prev_rows, prev_at = self.folded
                    d_batches = self.eval_batches - prev_batches
                    d_rows = self.eval_rows - prev_rows
                    dt = self.eval_at - prev_at
                    if d_batches > 0 and dt > 0:
                        out["eval_mean_batch"] = d_rows / d_batches
                        out["eval_batches_per_s"] = d_batches / dt
                        out["eval_evals_per_s"] = d_rows / dt
                if self.folded is None or self.folded[0] != self.eval_batches:
                    self.folded = (self.eval_batches, self.eval_rows, self.eval_at)
            if self.measure:
                out["measure_finals"] = self.measure["finals"]
                out["measure_distinct_finals"] = self.measure["distinct"]
                if self.measure["finals"] > 0:
                    out["measure_repeat_rate"] = (
                        self.measure["finals"] - self.measure["distinct"]
                    ) / self.measure["finals"]
            if self.admission:
                for key in (
                    "outstanding",
                    "reserved",
                    "waiting",
                    "max_waiting",
                    "bootstrap_grants",
                    "paced_grants",
                ):
                    out[f"admission_{key}"] = self.admission[key]
                out["admission_eval_capacity"] = (
                    self.admission["eval_capacity_milli"] / 1_000
                )
                out["admission_episode_work"] = (
                    self.admission["episode_work_milli"] / 1_000
                )
                out["admission_pressure_gain"] = (
                    self.admission["pressure_gain_milli"] / 1_000
                )
                out["admission_gap_ms"] = self.admission["gap_us"] / 1_000
            return out


def parse_stat_fields(line: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in line.strip().split() if "=" in token)


def pump_selfplay_stderr(
    process: subprocess.Popen[bytes],
    stats: SelfplayStatsTracker,
) -> None:
    """Relays stderr and folds selfplay heartbeat counters."""
    assert process.stderr is not None
    for raw in iter(process.stderr.readline, b""):
        sys.stderr.buffer.write(raw)
        sys.stderr.buffer.flush()
        if raw.startswith(b"event=eval_stats "):
            stats.observe_eval(parse_stat_fields(raw.decode("utf-8", "replace")))
        elif raw.startswith(b"event=measure_stats "):
            stats.observe_measure(parse_stat_fields(raw.decode("utf-8", "replace")))
        elif raw.startswith(b"event=admission_stats "):
            stats.observe_admission(parse_stat_fields(raw.decode("utf-8", "replace")))


class MetricsWriter:
    def __init__(self, path: Path, wandb_run: WandbRun | None = None) -> None:
        self.handle = path.open("a", encoding="utf-8")
        self.wandb_run = wandb_run

    def write(self, record: dict[str, object]) -> None:
        self.handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self.handle.flush()
        if self.wandb_run is not None:
            self.wandb_run.write(record)

    def finish(self) -> None:
        self.handle.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


class PerfWindow:
    """Accumulates per-step timings between metric writes."""

    def __init__(self, produced_rows: int = 0, episodes: int = 0) -> None:
        self.window_started = time.perf_counter()
        self.last_produced = produced_rows
        self.last_episodes = episodes
        self.has_counter_baseline = False
        self.steps = 0
        self.sample_seconds = 0.0
        self.train_seconds = 0.0

    def record(self, sample_started: float, train_started: float, finished: float) -> None:
        self.steps += 1
        self.sample_seconds += train_started - sample_started
        self.train_seconds += finished - train_started

    def drain(self, produced: int, episodes: int) -> dict[str, float]:
        now = time.perf_counter()
        elapsed = max(now - self.window_started, 1e-9)
        steps = max(self.steps, 1)
        perf = {
            "steps_per_s": self.steps / elapsed,
            "rows_per_s": (
                max(produced - self.last_produced, 0) / elapsed
                if self.has_counter_baseline
                else 0.0
            ),
            "episodes_per_s": (
                max(episodes - self.last_episodes, 0) / elapsed
                if self.has_counter_baseline
                else 0.0
            ),
            "sample_ms": 1000.0 * self.sample_seconds / steps,
            "train_ms": 1000.0 * self.train_seconds / steps,
        }
        self.window_started = now
        self.last_produced = produced
        self.last_episodes = episodes
        self.has_counter_baseline = True
        self.steps = 0
        self.sample_seconds = 0.0
        self.train_seconds = 0.0
        return perf


# JSONL keys -> grouped wandb keys. Keeping diagnostics explicit here prevents
# experimental fields from silently flooding the human-facing dashboard.
WANDB_KEYS = {
    "policy_loss": "train/policy_loss",
    "soft_policy_loss": "train/soft_policy_loss",
    "soft_policy_kl": "train/soft_policy_kl",
    "soft_policy_target_entropy": "train/soft_policy_target_entropy",
    "value_loss": "train/value_loss",
    "value_final_loss": "train/value_final_loss",
    "value_v8_loss": "train/value_v8_loss",
    "value_v32_loss": "train/value_v32_loss",
    "terminal_score_loss": "train/terminal_score_loss",
    "terminal_score_mae": "train/terminal_score_mae_nodes",
    "terminal_score_bias": "train/terminal_score_bias_nodes",
    "loss": "train/loss",
    "grad_norm": "train/grad_norm",
    "grad_clip_scale": "train/grad_clip_scale",
    "lr": "train/lr",
    "value_accuracy": "train/value_accuracy",
    "value_mae": "train/value_mae",
    "value_rmse": "train/value_rmse",
    "fraction_valid": "train/fraction_valid",
    "label_mean": "train/label_mean",
    "learner_win_rate": "train/learner_win_rate",
    "aux_signal_v8_final_target_correlation": (
        "auxiliary/signal/v8_final_target_correlation"
    ),
    "aux_signal_v32_final_target_correlation": (
        "auxiliary/signal/v32_final_target_correlation"
    ),
    "aux_signal_v8_v32_target_correlation": (
        "auxiliary/signal/v8_v32_target_correlation"
    ),
    "aux_signal_terminal_score_correlation": (
        "auxiliary/signal/terminal_score_correlation"
    ),
    "aux_signal_early_v8_final_target_correlation": (
        "auxiliary/signal/early_v8_final_target_correlation"
    ),
    "aux_signal_early_v32_final_target_correlation": (
        "auxiliary/signal/early_v32_final_target_correlation"
    ),
    "aux_signal_early_v8_target_std": "auxiliary/signal/early_v8_target_std",
    "aux_signal_early_v32_target_std": "auxiliary/signal/early_v32_target_std",
    "aux_gradient_effective_auxiliary_norm": (
        "auxiliary/readout_gradient/effective_auxiliary_norm"
    ),
    "aux_gradient_auxiliary_to_final_norm_ratio": (
        "auxiliary/readout_gradient/auxiliary_to_final_norm_ratio"
    ),
    "aux_gradient_auxiliary_alignment_ratio": (
        "auxiliary/readout_gradient/auxiliary_alignment_ratio"
    ),
    "aux_gradient_final_auxiliary_cosine": (
        "auxiliary/readout_gradient/final_auxiliary_cosine"
    ),
    "aux_gradient_policy_auxiliary_cosine": (
        "auxiliary/readout_gradient/policy_auxiliary_cosine"
    ),
    "parameter_trunk_gradient_norm": "optimizer/parameter/trunk_gradient_norm",
    "parameter_trunk_update_to_parameter": (
        "optimizer/parameter/trunk_update_to_parameter"
    ),
    "parameter_value_final_update_to_parameter": (
        "optimizer/parameter/value_final_update_to_parameter"
    ),
    "parameter_value_horizons_update_to_parameter": (
        "optimizer/parameter/value_horizons_update_to_parameter"
    ),
    "parameter_terminal_score_update_to_parameter": (
        "optimizer/parameter/terminal_score_update_to_parameter"
    ),
    "learner_win_rate_ema": "selfplay/learner_win_rate_ema",
    "value_sign_accuracy_early_ema": "selfplay/value_sign_accuracy_early_ema",
    "value_sign_accuracy_late_ema": "selfplay/value_sign_accuracy_late_ema",
    "episode_latency_s": "lag/episode_latency_s",
    "eval_mean_batch": "eval/mean_batch",
    "eval_batches_per_s": "eval/batches_per_s",
    "eval_evals_per_s": "eval/evals_per_s",
    "measure_finals": "measure/finals",
    "measure_distinct_finals": "measure/distinct_finals",
    "measure_repeat_rate": "measure/repeat_rate",
    "admission_outstanding": "admission/outstanding_evals",
    "admission_reserved": "admission/reserved_evals",
    "admission_waiting": "admission/waiting_workers",
    "admission_max_waiting": "admission/max_waiting_workers",
    "admission_bootstrap_grants": "admission/bootstrap_grants",
    "admission_paced_grants": "admission/paced_grants",
    "admission_eval_capacity": "admission/eval_capacity",
    "admission_episode_work": "admission/evals_per_episode",
    "admission_pressure_gain": "admission/pressure_gain",
    "admission_gap_ms": "admission/gap_ms",
    "terminal_cost_ema": "selfplay/terminal_cost_ema",
    "terminal_cost_best": "selfplay/terminal_cost_best",
    "stop_rate": "selfplay/stop_rate",
    "episode_len_ema": "selfplay/episode_len_ema",
    "stop_rate_ema": "selfplay/stop_rate_ema",
    "symmetric_games_completed": "symmetric/games_completed",
    "symmetric_p1_win_rate_ema": "symmetric/p1_win_rate_ema",
    "symmetric_p2_win_rate_ema": "symmetric/p2_win_rate_ema",
    "symmetric_draw_rate_ema": "symmetric/draw_rate_ema",
    "symmetric_decisive_rate_ema": "symmetric/decisive_rate_ema",
    "symmetric_seat_advantage_ema": "symmetric/seat_advantage_ema",
    "symmetric_p1_terminal_cost_ema": "symmetric/p1_terminal_cost_ema",
    "symmetric_p2_terminal_cost_ema": "symmetric/p2_terminal_cost_ema",
    "symmetric_mean_terminal_cost_ema": "symmetric/mean_terminal_cost_ema",
    "symmetric_best_of_two_terminal_cost_ema": "symmetric/best_of_two_terminal_cost_ema",
    "symmetric_terminal_cost_margin_ema": "symmetric/terminal_cost_margin_ema",
    "symmetric_terminal_cost_best": "symmetric/terminal_cost_best",
    "symmetric_p1_rewrites_ema": "symmetric/p1_rewrites_ema",
    "symmetric_p2_rewrites_ema": "symmetric/p2_rewrites_ema",
    "symmetric_game_rewrites_ema": "symmetric/game_rewrites_ema",
    "symmetric_rewrite_margin_ema": "symmetric/rewrite_margin_ema",
    "symmetric_value_sign_accuracy_early_ema": "symmetric/value_sign_accuracy_early_ema",
    "symmetric_value_sign_accuracy_late_ema": "symmetric/value_sign_accuracy_late_ema",
    "symmetric_game_latency_s": "symmetric/game_latency_s",
    "reduction_ema": "graph/reduction_ema",
    "reduction_best": "graph/reduction_best",
    "steps_per_s": "perf/steps_per_s",
    "rows_per_s": "perf/rows_per_s",
    "episodes_per_s": "perf/episodes_per_s",
    "sample_ms": "perf/sample_ms",
    "train_ms": "perf/train_ms",
    "produced_rows": "perf/produced_rows",
    "policy_reuse": "perf/policy_reuse",
    "value_reuse": "perf/value_reuse",
}


class WandbRun:
    """Optional wandb mirror of the metrics JSONL. Never load-bearing:
    init failure logs one line and the run proceeds without it."""

    def __init__(self, run: object) -> None:
        self.run = run
        self.publishes = 0

    @classmethod
    def start(
        cls,
        config: RunConfig,
        extra_config: dict[str, object] | None = None,
    ) -> WandbRun | None:
        if not config.wandb.project:
            return None
        try:
            import wandb

            run_config = {
                "trainer": asdict(config.trainer),
                "selfplay": asdict(config.selfplay),
                "measurement": asdict(config.measurement),
                "arch": asdict(config.arch),
                "run_dir": str(config.paths.run_dir),
            }
            if extra_config:
                run_config.update(extra_config)
            run = wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity or None,
                name=config.wandb.run_name or config.paths.run_dir.name,
                mode=config.wandb.mode or None,
                id=config.wandb.run_id or None,
                resume="must" if config.wandb.run_id else None,
                # A resumed run keeps its original config; re-sending it
                # would conflict on any knob the resume changed.
                config=None if config.wandb.run_id else run_config,
            )
        except Exception as error:
            print(f"event=wandb_disabled error={error}", file=sys.stderr, flush=True)
            return None
        return cls(run)

    def write(self, record: dict[str, object]) -> None:
        if record.get("event") == "step":
            payload = {
                WANDB_KEYS[key]: value
                for key, value in record.items()
                if key in WANDB_KEYS
            }
            self.run.log(payload, step=record["step"])
        elif record.get("event") == "graph":
            facts = {k: v for k, v in record.items() if k != "event"}
            self.run.config.update({"graph": facts}, allow_val_change=True)
            self.run.log({f"graph/{k}": v for k, v in facts.items()}, step=0)
        elif record.get("event") == "publish":
            self.publishes += 1
            payload = {
                "publish/count": self.publishes,
                "publish/training_step": record["training_step"],
            }
            for key in ("param_norm", "update_norm", "checkpoints_pruned"):
                if key in record:
                    payload[f"publish/{key}"] = record[key]
            self.run.log(payload, step=record["training_step"])

    def finish(self) -> None:
        self.run.finish()

def symmetric_step_fields(ack: SampleAck, completed_games: int) -> dict[str, float | int]:
    fields: dict[str, float | int] = {
        "symmetric_games_completed": completed_games,
    }
    metrics = ack.symmetric_selfplay
    if metrics is None:
        return fields
    fields.update(
        {
            "symmetric_p1_win_rate_ema": metrics.p1_win_rate_ema,
            "symmetric_p2_win_rate_ema": metrics.p2_win_rate_ema,
            "symmetric_draw_rate_ema": metrics.draw_rate_ema,
            "symmetric_decisive_rate_ema": max(0.0, 1.0 - metrics.draw_rate_ema),
            "symmetric_seat_advantage_ema": metrics.seat_advantage_ema,
            "symmetric_p1_terminal_cost_ema": metrics.p1_terminal_cost_ema,
            "symmetric_p2_terminal_cost_ema": metrics.p2_terminal_cost_ema,
            "symmetric_mean_terminal_cost_ema": metrics.mean_terminal_cost_ema,
            "symmetric_best_of_two_terminal_cost_ema": metrics.mean_terminal_cost_ema
            - 0.5 * metrics.terminal_cost_margin_ema,
            "symmetric_terminal_cost_margin_ema": metrics.terminal_cost_margin_ema,
            "symmetric_terminal_cost_best": metrics.terminal_cost_best,
            "symmetric_p1_rewrites_ema": metrics.p1_episode_len_ema,
            "symmetric_p2_rewrites_ema": metrics.p2_episode_len_ema,
            "symmetric_game_rewrites_ema": metrics.game_len_ema,
            "symmetric_rewrite_margin_ema": metrics.episode_len_margin_ema,
        }
    )
    if ack.value_sign_accuracy_early_ema >= 0.0:
        fields["symmetric_value_sign_accuracy_early_ema"] = (
            ack.value_sign_accuracy_early_ema
        )
    if ack.value_sign_accuracy_late_ema >= 0.0:
        fields["symmetric_value_sign_accuracy_late_ema"] = (
            ack.value_sign_accuracy_late_ema
        )
    if ack.episode_latency_ema >= 0.0:
        fields["symmetric_game_latency_s"] = ack.episode_latency_ema
    return fields
