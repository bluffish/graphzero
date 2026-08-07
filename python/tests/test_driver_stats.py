from __future__ import annotations

from gz.trainer.telemetry import SelfplayStatsTracker, parse_stat_fields


def test_parse_stat_fields_extracts_pairs() -> None:
    fields = parse_stat_fields("event=eval_stats batches=120 rows=7300\n")
    assert fields["batches"] == "120"
    assert fields["rows"] == "7300"


def test_stats_tracker_reports_window_rates_from_cumulative() -> None:
    tracker = SelfplayStatsTracker()
    assert tracker.step_fields() == {}

    tracker.observe_eval({"batches": "100", "rows": "5000"})
    first = tracker.step_fields()
    # First fold has no window yet: totals only.
    assert first["eval_batches_total"] == 100
    assert first["eval_rows_total"] == 5000
    assert "eval_mean_batch" not in first

    tracker.observe_eval({"batches": "200", "rows": "25000"})
    second = tracker.step_fields()
    # Window: 100 batches carrying 20000 rows -> mean batch 200.
    assert second["eval_mean_batch"] == 200.0
    assert second["eval_evals_per_s"] > 0

    # No new heartbeat: rates drop out, totals stay.
    third = tracker.step_fields()
    assert "eval_mean_batch" not in third
    assert third["eval_batches_total"] == 200


def test_stats_tracker_measure_ledger_fields() -> None:
    tracker = SelfplayStatsTracker()
    tracker.observe_measure(
        {"appended": "900", "dropped": "3", "finals": "1000", "distinct": "250"}
    )
    fields = tracker.step_fields()
    assert fields["measure_finals"] == 1000
    assert fields["measure_distinct_finals"] == 250
    assert fields["measure_repeat_rate"] == 0.75


def test_stats_tracker_ignores_malformed_lines() -> None:
    tracker = SelfplayStatsTracker()
    tracker.observe_eval({"batches": "x"})
    tracker.observe_measure({"appended": "1"})
    assert tracker.step_fields() == {}


def test_stats_tracker_reports_actor_checkpoint_step() -> None:
    tracker = SelfplayStatsTracker()
    tracker.observe_checkpoint_swap({"training_step": "320"})
    assert tracker.step_fields()["actor_training_step"] == 320
