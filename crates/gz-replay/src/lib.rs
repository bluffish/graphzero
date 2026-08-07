#![forbid(unsafe_code)]

//! Durable replay storage for GraphZero selfplay rows.

mod append;
mod database;
mod error;
mod keys;
mod records;
mod sample;
mod store;

pub use error::{ReplayError, ReplayResult};
pub use records::{ReplayEpisodeId, ReplayEpisodeRecord, ReplayOutcome, ReplayRow, ReplayRowKind};
pub use sample::SampleConfig;
pub use store::{
    ReplayContract, ReplayCounters, ReplayDataMode, ReplayStore, SymmetricRewardTargetScale,
    SymmetricRewardTieEpsilon, SymmetricSelfplayMetrics,
};
