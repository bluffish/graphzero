#![forbid(unsafe_code)]

//! Whittle backend used to validate the distributed measurement transport.

use gz_engine::{EngineIdentity, GraphEngine, MeasureConfigHash, MeasureOptions};
use gz_engine_whittle::{WhittleEngine, WhittleEngineConfig, WhittleRoot};
use gz_measure_service::{
    AgentBackend, AgentBackendError, AgentMeasurement, DeviceProfileHash, EncodedMeasureConfig,
    FetchedSubject, MEASUREMENT_PROTOCOL_VERSION, MeasureSubmission, ServiceError, ServiceResult,
    device_profile_hash, engine_identity_to_wire, wire,
};

pub const WHITTLE_MEASURE_CONFIG_ENCODING: u32 = 1;
const WHITTLE_CONFIG_MAGIC: &[u8; 4] = b"WMO1";
const WHITTLE_CONFIG_BYTES: usize = 17;

#[derive(Clone, Debug, Default)]
pub struct WhittleBackend;

impl WhittleBackend {
    #[must_use]
    pub fn capability_for(engine: &WhittleEngine) -> wire::EngineCapability {
        wire::EngineCapability {
            engine: Some(engine_identity_to_wire(EngineIdentity::from_engine(engine))),
            artifact_formats: vec![wire::ArtifactFormatCapability {
                format_kind: wire::GraphArtifactFormatKind::GraphArtifactFormatBinary as i32,
                adapter_format_id: 0,
            }],
            measure_config_encodings: vec![WHITTLE_MEASURE_CONFIG_ENCODING],
            measurement_protocol_versions: vec![MEASUREMENT_PROTOCOL_VERSION],
        }
    }
}

impl AgentBackend for WhittleBackend {
    fn capability(&self) -> wire::EngineCapability {
        Self::capability_for(&WhittleEngine::default())
    }

    fn measure(
        &self,
        lease: &wire::MeasureLease,
        mut subjects: Vec<FetchedSubject>,
    ) -> Result<AgentMeasurement, AgentBackendError> {
        if subjects.len() != 1 {
            return Err(backend_error(1, "Whittle requires exactly one subject"));
        }
        let subject = subjects.pop().unwrap();
        if subject.logical_index != 0
            || subject.format_kind != wire::GraphArtifactFormatKind::GraphArtifactFormatBinary
            || subject.adapter_format_id != 0
        {
            return Err(backend_error(2, "unsupported Whittle subject format"));
        }
        if lease.measure_config_encoding != WHITTLE_MEASURE_CONFIG_ENCODING {
            return Err(backend_error(3, "unsupported Whittle measure config"));
        }
        let config_hash = MeasureConfigHash::from_bytes(
            lease
                .measure_config_hash
                .as_slice()
                .try_into()
                .map_err(|_| backend_error(4, "Whittle config hash has the wrong width"))?,
        );
        let options = decode_measure_options(config_hash, &lease.measure_config_payload)
            .map_err(|error| backend_error(5, error.to_string()))?;
        let mut engine = WhittleEngine::new(WhittleEngineConfig {
            root: WhittleRoot::Artifact(subject.bytes),
            ..WhittleEngineConfig::default()
        })
        .map_err(|error| backend_error(6, error.to_string()))?;
        if engine_identity_to_wire(EngineIdentity::from_engine(&engine))
            != lease.engine.clone().unwrap_or_default()
        {
            return Err(backend_error(7, "Whittle engine identity mismatch"));
        }
        if engine.measure_config_hash() != config_hash {
            return Err(backend_error(8, "Whittle measure config hash mismatch"));
        }
        let root = engine.root();
        if engine
            .hash(root)
            .map_err(|error| backend_error(9, error.to_string()))?
            .as_bytes()
            != &subject.graph_hash
        {
            return Err(backend_error(10, "Whittle artifact graph hash mismatch"));
        }
        let result = engine
            .measure(root, options)
            .map_err(|error| backend_error(11, error.to_string()))?;
        let scalar_reward = result
            .scalar_reward
            .ok_or_else(|| backend_error(12, "Whittle omitted scalar reward"))?;
        if !result.measured || !result.valid || !scalar_reward.is_finite() {
            return Err(backend_error(13, "Whittle returned an invalid measurement"));
        }

        Ok(AgentMeasurement {
            subjects: vec![wire::SubjectMeasurement {
                logical_index: 0,
                graph_hash: result.graph_hash.as_bytes().to_vec(),
                compile_elapsed_ns: None,
                capture_elapsed_ns: None,
                scalar_reward: Some(f64::from(scalar_reward)),
                engine_metadata: result.metadata.bytes,
            }],
            samples: Vec::new(),
            telemetry: Vec::new(),
        })
    }
}

#[must_use]
pub fn encode_measure_options(options: MeasureOptions) -> EncodedMeasureConfig {
    let mut payload = Vec::with_capacity(WHITTLE_CONFIG_BYTES);
    payload.extend_from_slice(WHITTLE_CONFIG_MAGIC);
    payload.extend_from_slice(&options.samples.to_le_bytes());
    payload.extend_from_slice(&options.timeout_ms.unwrap_or(0).to_le_bytes());
    payload.push(u8::from(options.deterministic));
    EncodedMeasureConfig {
        encoding: WHITTLE_MEASURE_CONFIG_ENCODING,
        payload,
    }
}

pub fn decode_measure_options(
    config_hash: MeasureConfigHash,
    payload: &[u8],
) -> ServiceResult<MeasureOptions> {
    if payload.len() != WHITTLE_CONFIG_BYTES || &payload[..4] != WHITTLE_CONFIG_MAGIC {
        return Err(ServiceError::protocol(
            "invalid Whittle measure config payload",
        ));
    }
    let samples = u32::from_le_bytes(payload[4..8].try_into().unwrap());
    let timeout = u64::from_le_bytes(payload[8..16].try_into().unwrap());
    let deterministic = match payload[16] {
        0 => false,
        1 => true,
        _ => return Err(ServiceError::protocol("invalid Whittle deterministic flag")),
    };
    MeasureOptions::new(
        config_hash,
        samples,
        (timeout != 0).then_some(timeout),
        deterministic,
    )
    .map_err(|error| ServiceError::protocol(error.to_string()))
}

pub fn submission(
    engine: &WhittleEngine,
    graph: <WhittleEngine as GraphEngine>::Graph,
    options: MeasureOptions,
    target_device_profile_hash: DeviceProfileHash,
) -> Result<MeasureSubmission, gz_engine::EngineError> {
    MeasureSubmission::from_engine(
        engine,
        graph,
        options,
        encode_measure_options(options),
        target_device_profile_hash,
    )
}

#[must_use]
pub fn named_test_profile(name: &str) -> wire::DeviceProfile {
    let os_digest = named_digest(b"gz-whittle-test-os-v1", name);
    let agent_digest = named_digest(b"gz-whittle-agent-v1", name);
    wire::DeviceProfile {
        platform_family: "graphzero-test".to_owned(),
        board_model: name.to_owned(),
        soc: "whittle-cpu".to_owned(),
        gpu_architecture: "not-used".to_owned(),
        usable_memory_bytes: 1,
        operating_system_image_digest: os_digest.to_vec(),
        platform_release: "whittle-network-v1".to_owned(),
        cuda_version: "not-used".to_owned(),
        gpu_driver_version: "not-used".to_owned(),
        compiler_version: "not-used".to_owned(),
        compiler_runtime_version: "not-used".to_owned(),
        agent_image_digest: agent_digest.to_vec(),
        power_profile: "not-used".to_owned(),
        clock_policy: "not-used".to_owned(),
        cooling_policy: "not-used".to_owned(),
        measurement_protocol_version: MEASUREMENT_PROTOCOL_VERSION,
    }
}

#[must_use]
pub fn named_test_profile_hash(name: &str) -> DeviceProfileHash {
    device_profile_hash(&named_test_profile(name))
}

fn named_digest(domain: &[u8], name: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(name.as_bytes());
    *hasher.finalize().as_bytes()
}

fn backend_error(code: u32, message: impl Into<String>) -> AgentBackendError {
    AgentBackendError::new(
        wire::MeasureAttemptOutcome::MeasureOutcomeAgentInternal,
        code,
        message,
        false,
    )
}
