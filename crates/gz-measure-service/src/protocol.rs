use crate::{
    ArtifactDigest, DeviceProfileHash, JobId, MeasurementKey, RequestNonce, ServiceError,
    ServiceResult,
};
use gz_engine::{
    EngineError, EngineIdentity as CoreEngineIdentity, ErrorCode, ErrorMessage, GraphArtifact,
    GraphArtifactFormat, GraphEngine, LatencyStats, MeasureFailure, MeasureMetadata,
    MeasureOptions, MeasureResult,
};
use prost::Message;

pub const PROTOCOL_MAJOR: u32 = 1;
pub const PROTOCOL_MINOR: u32 = 1;
pub const MEASUREMENT_PROTOCOL_VERSION: u32 = 1;

#[allow(clippy::large_enum_variant)]
pub mod wire {
    tonic::include_proto!("graphzero.measure.v1");
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedMeasureConfig {
    pub encoding: u32,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct MeasureSubmission {
    pub kind: MeasureSubmissionKind,
    pub engine: CoreEngineIdentity,
    /// Logical subject order is significant. Symmetric player one is index 0.
    pub subjects: Vec<GraphArtifact>,
    pub options: MeasureOptions,
    pub measure_config: EncodedMeasureConfig,
    pub target_device_profile_hash: DeviceProfileHash,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeasureSubmissionKind {
    Single,
    SymmetricPair,
}

impl MeasureSubmissionKind {
    #[must_use]
    pub const fn wire(self) -> wire::MeasureKind {
        match self {
            Self::Single => wire::MeasureKind::Single,
            Self::SymmetricPair => wire::MeasureKind::SymmetricPair,
        }
    }

    #[must_use]
    pub const fn subject_count(self) -> usize {
        match self {
            Self::Single => 1,
            Self::SymmetricPair => 2,
        }
    }
}

impl MeasureSubmission {
    pub fn from_engine<E: GraphEngine>(
        engine: &E,
        graph: E::Graph,
        options: MeasureOptions,
        measure_config: EncodedMeasureConfig,
        target_device_profile_hash: DeviceProfileHash,
    ) -> Result<Self, EngineError> {
        Ok(Self {
            kind: MeasureSubmissionKind::Single,
            engine: CoreEngineIdentity::from_engine(engine),
            subjects: vec![engine.export_graph(graph)?],
            options,
            measure_config,
            target_device_profile_hash,
        })
    }

    pub fn from_engine_pair<E: GraphEngine>(
        engine: &E,
        graphs: [E::Graph; 2],
        options: MeasureOptions,
        measure_config: EncodedMeasureConfig,
        target_device_profile_hash: DeviceProfileHash,
    ) -> Result<Self, EngineError> {
        Ok(Self {
            kind: MeasureSubmissionKind::SymmetricPair,
            engine: CoreEngineIdentity::from_engine(engine),
            subjects: vec![
                engine.export_graph(graphs[0])?,
                engine.export_graph(graphs[1])?,
            ],
            options,
            measure_config,
            target_device_profile_hash,
        })
    }

    pub fn into_symmetric_pair<E: GraphEngine>(
        mut self,
        engine: &E,
        second: E::Graph,
    ) -> Result<Self, EngineError> {
        if self.kind != MeasureSubmissionKind::Single
            || self.subjects.len() != 1
            || self.engine != CoreEngineIdentity::from_engine(engine)
        {
            return Err(remote_engine_error(
                9,
                "single measurement cannot be promoted to this engine pair",
            ));
        }
        self.kind = MeasureSubmissionKind::SymmetricPair;
        self.subjects.push(engine.export_graph(second)?);
        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub struct CommittedMeasurement {
    pub job_id: JobId,
    pub report: wire::MeasureReport,
}

impl CommittedMeasurement {
    pub fn into_measure_result<G>(
        self,
        graph: G,
        options: MeasureOptions,
    ) -> Result<MeasureResult<G>, EngineError> {
        if self.report.kind != wire::MeasureKind::Single as i32 || self.report.subjects.len() != 1 {
            return Err(remote_engine_error(
                1,
                "remote report is not a single measurement",
            ));
        }
        measure_result_for(&self.report, 0, graph, options)
    }

    pub fn into_measure_pair_results<G>(
        self,
        graphs: [G; 2],
        options: MeasureOptions,
    ) -> Result<[MeasureResult<G>; 2], EngineError> {
        if self.report.kind != wire::MeasureKind::SymmetricPair as i32
            || self.report.subjects.len() != 2
        {
            return Err(remote_engine_error(
                10,
                "remote report is not a symmetric pair",
            ));
        }
        let [first, second] = graphs;
        Ok([
            measure_result_for(&self.report, 0, first, options)?,
            measure_result_for(&self.report, 1, second, options)?,
        ])
    }
}

fn measure_result_for<G>(
    report: &wire::MeasureReport,
    logical_index: u32,
    graph: G,
    options: MeasureOptions,
) -> Result<MeasureResult<G>, EngineError> {
    let subject = report
        .subjects
        .iter()
        .find(|subject| subject.logical_index == logical_index)
        .ok_or_else(|| remote_engine_error(2, "remote report omitted a subject"))?;
    let graph_hash = gz_engine::GraphHash::from_bytes(
        subject
            .graph_hash
            .as_slice()
            .try_into()
            .map_err(|_| remote_engine_error(3, "remote graph hash has the wrong width"))?,
    );
    let outcome = wire::MeasureAttemptOutcome::try_from(report.outcome)
        .map_err(|_| remote_engine_error(4, "remote report has an unknown outcome"))?;

    let samples_ms = report
        .samples
        .iter()
        .filter(|sample| sample.logical_index == logical_index)
        .map(|sample| sample.gpu_elapsed_ns as f32 / 1_000_000.0)
        .collect::<Vec<_>>();
    let latency = if samples_ms.is_empty() {
        None
    } else {
        Some(
            LatencyStats::from_samples(samples_ms)
                .map_err(|_| remote_engine_error(5, "remote latency samples are invalid"))?,
        )
    };

    let (measured, valid, scalar_reward, failure) = match outcome {
        wire::MeasureAttemptOutcome::MeasureOutcomeSucceeded => {
            let reward = subject.scalar_reward.ok_or_else(|| {
                remote_engine_error(6, "successful remote report omitted scalar reward")
            })? as f32;
            if !reward.is_finite() {
                return Err(remote_engine_error(
                    7,
                    "successful remote report has non-finite reward",
                ));
            }
            (true, true, Some(reward), None)
        }
        _ => {
            let detail = report.failure.as_ref().ok_or_else(|| {
                remote_engine_error(8, "failed remote report omitted failure detail")
            })?;
            let message = ErrorMessage::new(detail.bounded_message.clone())
                .unwrap_or_else(|_| ErrorMessage::new("remote measurement failed").unwrap());
            (
                true,
                false,
                None,
                Some(MeasureFailure {
                    code: ErrorCode::new(detail.engine_error_code),
                    message,
                }),
            )
        }
    };

    MeasureResult {
        graph,
        graph_hash,
        config_hash: options.config_hash,
        measured,
        valid,
        latency,
        scalar_reward,
        failure,
        metadata: MeasureMetadata {
            bytes: subject.engine_metadata.clone(),
        },
    }
    .validate()
    .map_err(|_| remote_engine_error(9, "remote measurement result is invalid"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArtifactDescriptor {
    pub digest: ArtifactDigest,
    pub size: u64,
    pub format_kind: wire::GraphArtifactFormatKind,
    pub adapter_format_id: u32,
}

pub fn artifact_descriptor(artifact: &GraphArtifact) -> ServiceResult<ArtifactDescriptor> {
    let (format_kind, adapter_format_id) = artifact_format(artifact.format);
    let size = u64::try_from(artifact.bytes.len())
        .map_err(|_| ServiceError::capacity("artifact length does not fit u64"))?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"gz-graph-artifact-v1\0");
    hasher.update(&(format_kind as u32).to_le_bytes());
    hasher.update(&adapter_format_id.to_le_bytes());
    hasher.update(&artifact.bytes);

    Ok(ArtifactDescriptor {
        digest: ArtifactDigest::from_bytes(*hasher.finalize().as_bytes()),
        size,
        format_kind,
        adapter_format_id,
    })
}

#[must_use]
pub fn device_profile_hash(profile: &wire::DeviceProfile) -> DeviceProfileHash {
    let encoded = profile.encode_to_vec();
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"gz-device-profile-v1\0");
    hasher.update(&encoded);
    DeviceProfileHash::from_bytes(*hasher.finalize().as_bytes())
}

pub fn measurement_key(
    submission: &MeasureSubmission,
    descriptors: &[ArtifactDescriptor],
) -> MeasurementKey {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"gz-measure-key-v1\0");
    hasher.update(&(submission.kind.wire() as u32).to_le_bytes());
    hasher.update(submission.engine.engine_id.as_bytes());
    hasher.update(submission.engine.engine_version.as_bytes());
    hasher.update(submission.engine.action_set_hash.as_bytes());
    hasher.update(submission.options.config_hash.as_bytes());
    hasher.update(submission.target_device_profile_hash.as_bytes());
    hasher.update(&(submission.subjects.len() as u32).to_le_bytes());
    for (logical_index, (subject, descriptor)) in
        submission.subjects.iter().zip(descriptors).enumerate()
    {
        hasher.update(&(logical_index as u32).to_le_bytes());
        hasher.update(subject.graph_hash.as_bytes());
        hasher.update(&(descriptor.format_kind as u32).to_le_bytes());
        hasher.update(&descriptor.adapter_format_id.to_le_bytes());
        hasher.update(descriptor.digest.as_bytes());
        hasher.update(&descriptor.size.to_le_bytes());
    }
    MeasurementKey::from_bytes(*hasher.finalize().as_bytes())
}

#[must_use]
pub fn job_id(measurement_key: MeasurementKey, request_nonce: RequestNonce) -> JobId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"gz-measure-job-v1\0");
    hasher.update(measurement_key.as_bytes());
    hasher.update(request_nonce.as_bytes());
    JobId::from_bytes(*hasher.finalize().as_bytes())
}

#[must_use]
pub fn engine_identity_to_wire(identity: CoreEngineIdentity) -> wire::EngineIdentity {
    wire::EngineIdentity {
        engine_id: identity.engine_id.as_bytes().to_vec(),
        engine_version: identity.engine_version.as_bytes().to_vec(),
        action_set_hash: identity.action_set_hash.as_bytes().to_vec(),
    }
}

pub fn engine_identity_from_wire(
    identity: &wire::EngineIdentity,
) -> ServiceResult<CoreEngineIdentity> {
    Ok(CoreEngineIdentity {
        engine_id: gz_engine::EngineId::from_bytes(fixed_bytes("engine_id", &identity.engine_id)?),
        engine_version: gz_engine::EngineVersion::from_bytes(fixed_bytes(
            "engine_version",
            &identity.engine_version,
        )?),
        action_set_hash: gz_engine::ActionSetHash::from_bytes(fixed_bytes(
            "action_set_hash",
            &identity.action_set_hash,
        )?),
    })
}

#[must_use]
pub fn artifact_format(format: GraphArtifactFormat) -> (wire::GraphArtifactFormatKind, u32) {
    match format {
        GraphArtifactFormat::Text => (wire::GraphArtifactFormatKind::GraphArtifactFormatText, 0),
        GraphArtifactFormat::Json => (wire::GraphArtifactFormatKind::GraphArtifactFormatJson, 0),
        GraphArtifactFormat::Dot => (wire::GraphArtifactFormatKind::GraphArtifactFormatDot, 0),
        GraphArtifactFormat::Binary => {
            (wire::GraphArtifactFormatKind::GraphArtifactFormatBinary, 0)
        }
        GraphArtifactFormat::AdapterSpecific(id) => (
            wire::GraphArtifactFormatKind::GraphArtifactFormatAdapterSpecific,
            id,
        ),
    }
}

pub(crate) fn fixed_bytes<const N: usize>(name: &str, bytes: &[u8]) -> ServiceResult<[u8; N]> {
    bytes.try_into().map_err(|_| {
        ServiceError::protocol(format!("{name} must be {N} bytes, got {}", bytes.len()))
    })
}

fn remote_engine_error(code: u32, message: &'static str) -> EngineError {
    EngineError::Internal {
        code: ErrorCode::new(10_000 + code),
        message: ErrorMessage::new(message).expect("remote engine errors are bounded"),
    }
}
