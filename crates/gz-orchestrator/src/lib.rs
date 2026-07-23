#![forbid(unsafe_code)]

//! Execution drivers for GraphZero search workers.

pub mod admission;
mod ids;
mod lanes;
mod leases;
mod measurement;
mod pool;
mod root;

pub use admission::{AdaptiveAdmissionSchedule, AdmissionDecision, AdmissionSmoothingConfig};
pub use ids::EpisodeId;
pub use lanes::{
    FeaturizedRuntime, ReplayBackpressure, ReplayRuntime, ThreadedGumbelOrchestrator,
    ThreadedOrchestratorConfig, ThreadedReplayRun,
};
pub use measurement::{MeasureSubmissionEncoder, RemoteMeasurementRuntime};
pub use root::RootSource;

pub(crate) fn internal(message: &'static str) -> gz_engine::EngineError {
    gz_engine::EngineError::Internal {
        code: gz_engine::ErrorCode::new(1),
        message: gz_engine::ErrorMessage::new(message)
            .expect("internal orchestrator message is short"),
    }
}
