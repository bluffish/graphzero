#![forbid(unsafe_code)]

//! Distributed terminal measurement transport.
//!
//! Networking and scheduling live here so `gz-engine` remains independent of
//! async runtimes and transport choices.

mod agent;
mod coordinator;
mod error;
mod ids;
mod protocol;
mod receipt;
mod transport;

pub use agent::{
    AgentBackend, AgentBackendError, AgentConfig, AgentMeasurement, AgentRuntime, FetchedSubject,
};
pub use coordinator::{
    Coordinator, CoordinatorConfig, CoordinatorHandle, CoordinatorSnapshot, Enrollment,
};
pub use error::{ServiceError, ServiceResult};
pub use ids::{
    ArtifactDigest, CertificateFingerprint, DeviceId, DeviceProfileHash, JobId, LeaseId,
    MeasurementKey, RequestNonce, SessionId, certificate_fingerprint,
};
pub use protocol::{
    ArtifactDescriptor, CommittedMeasurement, EncodedMeasureConfig, MEASUREMENT_PROTOCOL_VERSION,
    MeasureSubmission, PROTOCOL_MAJOR, PROTOCOL_MINOR, artifact_descriptor, artifact_format,
    device_profile_hash, engine_identity_from_wire, engine_identity_to_wire, job_id,
    measurement_key, wire,
};
pub use receipt::ReceiptLedgerConfig;
pub use transport::{AgentTlsConfig, CoordinatorServer, CoordinatorTlsConfig, InsecureTransport};
