#![forbid(unsafe_code)]

//! Execution drivers for GraphZero search workers.

mod batch;
mod bench;
mod ids;
mod lanes;
mod pool;
mod root;
mod serial;
mod service;

pub use batch::{BatchedGumbelOrchestrator, BatchedRun};
pub use bench::{
    SelfplayBenchConfig, SelfplayBenchReport, SelfplayEpisodeStats, SelfplayRunStats,
    run_selfplay_benchmark, run_serial_selfplay_benchmark,
};
pub use ids::{EpisodeId, WorkerId};
pub use lanes::{
    LaneEpisodes, ThreadedGumbelOrchestrator, ThreadedOrchestratorConfig, ThreadedRun,
};
pub use root::{CountedRoots, RootSource};
pub use serial::{OrchestratedEpisode, SerialEpisode, SerialGumbelOrchestrator};
