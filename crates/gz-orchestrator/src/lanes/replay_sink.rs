use super::{ReplayJob, map_replay_error};
use gz_engine::EngineResult;
use gz_measurer::{MeasurerRunSummary, ReplayMeasurer};
use gz_replay::{ReplayStore, SymmetricRewardTargetScale, SymmetricRewardTieEpsilon};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

pub(super) fn run_replay_sink(
    store: &ReplayStore,
    replay_rx: Receiver<ReplayJob>,
    reward_tie_epsilon: SymmetricRewardTieEpsilon,
    reward_target_scale: SymmetricRewardTargetScale,
) -> EngineResult<MeasurerRunSummary> {
    let mut measurer = ReplayMeasurer::new(store, reward_tie_epsilon, reward_target_scale);
    // Machine-parsed by the trainer driver (measure ledger metrics);
    // field changes must update its parser. Counters are cumulative.
    const STATS_INTERVAL: Duration = Duration::from_secs(30);
    let mut last_stats = Instant::now();

    while let Ok(job) = replay_rx.recv() {
        let (result, ack) = match job {
            ReplayJob::Symmetric { game, ack } => (
                measurer.admit_symmetric(*game).map_err(map_replay_error),
                ack,
            ),
        };
        let failed = result.as_ref().err().cloned();
        let _ = ack.send(result);
        if let Some(error) = failed {
            return Err(error);
        }
        if last_stats.elapsed() >= STATS_INTERVAL {
            last_stats = Instant::now();
            let stats = measurer.stats();
            eprintln!(
                "event=measure_stats appended={} dropped={} finals={} distinct={}",
                stats.episodes_appended,
                stats.episodes_dropped,
                stats.finals,
                stats.distinct_finals,
            );
        }
    }

    Ok(measurer.finish())
}
