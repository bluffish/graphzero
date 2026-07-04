#![forbid(unsafe_code)]

//! Durable replay storage for GraphZero selfplay rows.

mod error;
mod keys;
mod records;
mod sample;
mod store;

pub use error::{ReplayError, ReplayResult};
pub use records::{
    ReplayEpisodeId, ReplayEpisodeRecord, ReplayOutcome, ReplayReference, ReplayReferenceKind,
    ReplayRootInfo, ReplayRow,
};
pub use sample::SampleConfig;
pub use store::{ReplayCounters, ReplayStore};
