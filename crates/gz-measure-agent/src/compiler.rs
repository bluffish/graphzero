use artifact::ComponentSequenceId;
#[cfg(feature = "intel")]
use artifact::IntelReusableCommandApi as RuntimeIntelReusableCommandApi;
use gz_engine::{
    EngineIdentity, GraphArtifact, GraphArtifactFormat, GraphEngine, GraphHash, MeasureConfigHash,
    MeasureOptions,
};
use gz_engine_compiler::{
    ACCURACY_FAILURE_REWARD, COMPILE_FAILURE_CODE, COMPILER_STATE_ARTIFACT_FORMAT,
    CompilerCandidateMode, CompilerEngine, CompilerEngineConfig, CompilerEngineDomain,
    CompilerEngineTarget, CompilerEvaluationWorkload, CompilerMeasurementEvidence,
    CompilerMeasurementPolicy, accuracy_failure_evidence, decode_measurement_evidence, native,
};
use gz_measure_service::{
    AgentBackend, AgentBackendError, AgentMeasurement, DeviceProfileHash, EncodedMeasureConfig,
    FetchedSubject, MEASUREMENT_PROTOCOL_VERSION, MeasureSubmission, ServiceError, ServiceResult,
    device_profile_hash, engine_identity_to_wire, wire,
};
#[cfg(feature = "intel")]
use native::backends::intel::lowering::ExternalIntelPayloadCompiler;
use native::backends::intel::{
    lowering::{IntelCompiledKernel, IntelPayloadCompileRequest, IntelPayloadCompiler},
    target::{IntelReusableCommandApi, IntelTargetRequest},
};
use native::{
    backends::{
        BackendResult, LoweringPolicy, TargetDeviceRequest, internal_backend_error,
        nvidia::{
            lowering::{ExternalNvidiaPayloadCompiler, NvidiaPayloadCompiler},
            providers::native_cuda::compile::{NativeCudaCompileRequest, NativeCudaCompileResult},
            target::NvidiaTargetRequest,
        },
    },
    bufferization::BufferizationPolicy,
    evaluators::{
        AccuracyCase, AccuracyEvaluationRequest, AccuracyReport, AccuracySuite, BuildFingerprint,
        CaseId, CaseStatePolicy, EvaluationBindingSet, EvaluationExecutionError,
        EvaluationExecutionTarget, EvaluationExecutor, EvaluationRuntimePolicy,
        EvaluationSessionPolicy, PerformanceCase, PerformanceEvaluationRequest, PerformanceReport,
        PerformanceSuite, ReferenceCase, ReferenceOutputBundle, Tolerance, TolerancePolicy,
        ToleranceValues, load_tensor_bundle,
    },
    importer::{ImportError, import},
};
#[cfg(feature = "intel")]
use runtime_intel::{IntelRuntimeAdapter, IntelRuntimeConfig, NativeIntelDriver};
use runtime_nvidia::{NvidiaRuntimeAdapter, NvidiaRuntimeConfig};
use serde::Deserialize;
use std::{
    collections::VecDeque,
    fs::File,
    io::Read,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

pub const COMPILER_MEASURE_CONFIG_ENCODING: u32 = 2;
const DEPLOYMENT_SCHEMA: &str = "graphzero.compiler-deployment.v1";
const COMPILER_CONFIG_MAGIC: &[u8; 4] = b"CMO1";
const COMPILER_CONFIG_BYTES: usize = 17;
const DEFAULT_COMPILER_MEASUREMENT_PIPELINE_SLOTS: usize = 2;
const MAX_COMPILER_MEASUREMENT_PIPELINE_SLOTS: usize = 8;
const COMPILER_MEASUREMENT_ENGINES_PER_SLOT: usize = 2;
const DEFAULT_PREPARED_STATE_CACHE_CAPACITY: usize = 8;
const MAX_PREPARED_STATE_CACHE_CAPACITY: usize = 256;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompilerDeploymentFile {
    schema: String,
    #[serde(default)]
    backend: CompilerDeploymentBackend,
    source_dir: PathBuf,
    inputs: PathBuf,
    references: PathBuf,
    cuda_python: Option<PathBuf>,
    cuda_helper: Option<PathBuf>,
    intel_python: Option<PathBuf>,
    intel_helper: Option<PathBuf>,
    #[serde(default)]
    intel_driver_version: u32,
    device_ordinal: u32,
    compute_capability_major: u16,
    compute_capability_minor: u16,
    performance_warmup_iterations: u32,
    #[serde(default = "default_runtime_accuracy_verification")]
    runtime_accuracy_verification: bool,
    #[serde(default = "default_measurement_pipeline_slots")]
    measurement_pipeline_slots: usize,
    #[serde(default = "default_prepared_state_cache_capacity")]
    prepared_state_cache_capacity: usize,
    evaluator_queue_capacity: usize,
    evaluator_cache_capacity: usize,
    candidate_mode: String,
    case_id: String,
    execution_sequence: String,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    compiler_build: String,
    runtime_build: String,
    device_profile: DeviceProfileFile,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
enum CompilerDeploymentBackend {
    #[default]
    Nvidia,
    IntelArcB580,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeviceProfileFile {
    platform_family: String,
    board_model: String,
    soc: String,
    gpu_architecture: String,
    usable_memory_bytes: u64,
    operating_system_image: String,
    platform_release: String,
    cuda_version: String,
    gpu_driver_version: String,
    power_profile: String,
    clock_policy: String,
    cooling_policy: String,
    agent_image: String,
}

#[derive(Clone, Debug)]
pub struct CompilerDeployment {
    file: CompilerDeploymentFile,
    candidate_mode: CompilerCandidateMode,
    measure_config_hash: MeasureConfigHash,
    profile: wire::DeviceProfile,
}

impl CompilerDeployment {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let contents = std::str::from_utf8(&bytes)
            .map_err(|error| format!("{} is not UTF-8: {error}", path.display()))?;
        let file: CompilerDeploymentFile = toml::from_str(contents)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
        validate_deployment(&file)?;
        let candidate_mode = parse_candidate_mode(&file.candidate_mode)?;
        let measure_config_hash = measurement_config_hash(&file, candidate_mode)?;
        let profile = device_profile(&file);
        Ok(Self {
            file,
            candidate_mode,
            measure_config_hash,
            profile,
        })
    }

    #[must_use]
    pub const fn measure_config_hash(&self) -> MeasureConfigHash {
        self.measure_config_hash
    }

    #[must_use]
    pub fn device_profile(&self) -> &wire::DeviceProfile {
        &self.profile
    }

    #[must_use]
    pub fn device_profile_hash(&self) -> DeviceProfileHash {
        device_profile_hash(&self.profile)
    }

    pub fn measure_options(&self, samples: u32) -> Result<MeasureOptions, String> {
        MeasureOptions::new(self.measure_config_hash, samples, None, true)
            .map_err(|error| error.to_string())
    }

    #[must_use]
    pub const fn measurement_pipeline_slots(&self) -> usize {
        self.file.measurement_pipeline_slots
    }

    pub fn build_search_engine(&self) -> Result<CompilerEngine, String> {
        let workload = self.load_workload()?;
        let root = import(&self.file.source_dir).map_err(format_import_error)?;
        self.build_engine(
            self.disabled_target()?,
            Arc::new(DisabledEvaluator),
            workload,
            root,
        )
    }

    pub fn build_search_engines(&self, count: usize) -> Result<Vec<CompilerEngine>, String> {
        if count == 0 {
            return Err("compiler search engine count must be greater than zero".to_owned());
        }
        let workload = self.load_workload()?;
        let root = import(&self.file.source_dir).map_err(format_import_error)?;
        (0..count)
            .map(|_| {
                self.build_engine(
                    self.disabled_target()?,
                    Arc::new(DisabledEvaluator),
                    workload.clone(),
                    root.clone(),
                )
            })
            .collect()
    }

    pub fn build_measure_engine(&self) -> Result<CompilerEngine, String> {
        let mut engines = self.build_measure_engines(1)?;
        Ok(engines.pop().expect("one requested engine was built"))
    }

    fn build_measure_engines(&self, count: usize) -> Result<Vec<CompilerEngine>, String> {
        if count == 0 {
            return Err("compiler measurement engine count must be greater than zero".to_owned());
        }
        match self.file.backend {
            CompilerDeploymentBackend::Nvidia => self.build_nvidia_measure_engines(count),
            CompilerDeploymentBackend::IntelArcB580 => self.build_intel_measure_engines(count),
        }
    }

    fn build_nvidia_measure_engines(&self, count: usize) -> Result<Vec<CompilerEngine>, String> {
        let runtime_config = NvidiaRuntimeConfig {
            device_ordinal: self.file.device_ordinal,
        };
        let environment = NvidiaRuntimeAdapter::new(runtime_config)
            .map_err(|error| error.to_string())?
            .environment()
            .map_err(|error| error.to_string())?;
        let expected = [
            u32::from(self.file.compute_capability_major),
            u32::from(self.file.compute_capability_minor),
        ];
        if environment.compute_capability != expected {
            return Err(format!(
                "deployment requires compute capability {}.{}, device reports {}.{}",
                expected[0],
                expected[1],
                environment.compute_capability[0],
                environment.compute_capability[1]
            ));
        }
        // Every compiler slot shares this single-worker evaluator. Preparation
        // may overlap, but accuracy, warm-up, and timed GPU execution cannot.
        let evaluator: Arc<dyn EvaluationExecutor> = Arc::new(
            evaluator_nvidia::start(evaluator_nvidia::NvidiaEvaluatorConfig {
                runtime: runtime_config,
                queue_capacity: NonZeroUsize::new(self.file.evaluator_queue_capacity)
                    .expect("validated evaluator queue capacity"),
                cache_capacity: NonZeroUsize::new(self.file.evaluator_cache_capacity)
                    .expect("validated evaluator cache capacity"),
                cache_results: false,
                evaluator_build: build_fingerprint(&self.file.compiler_build),
                runtime_build: build_fingerprint(&self.file.runtime_build),
            })
            .map_err(|error| error.to_string())?,
        );
        let python = self
            .file
            .cuda_python
            .as_ref()
            .expect("validated CUDA Python");
        let helper = self
            .file
            .cuda_helper
            .as_ref()
            .expect("validated CUDA helper");
        let compiler: Arc<dyn NvidiaPayloadCompiler> = Arc::new(
            ExternalNvidiaPayloadCompiler::new(python).with_argument(helper.as_os_str().to_owned()),
        );
        let target = CompilerEngineTarget::Nvidia {
            request: NvidiaTargetRequest::sm(
                TargetDeviceRequest::BackendDefault,
                self.file.compute_capability_major,
                self.file.compute_capability_minor,
            )
            .map_err(|error| error.to_string())?,
            payload_compiler: compiler,
        };
        let workload = self.load_workload()?;
        let root = import(&self.file.source_dir).map_err(format_import_error)?;
        (0..count)
            .map(|_| {
                self.build_engine(
                    target.clone(),
                    Arc::clone(&evaluator),
                    workload.clone(),
                    root.clone(),
                )
            })
            .collect()
    }

    #[cfg(feature = "intel")]
    fn build_intel_measure_engines(&self, count: usize) -> Result<Vec<CompilerEngine>, String> {
        let runtime_config = IntelRuntimeConfig::for_device_ordinal(self.file.device_ordinal)
            .with_reusable_command_api(RuntimeIntelReusableCommandApi::Software);
        let driver = NativeIntelDriver::new(&runtime_config).map_err(|error| error.to_string())?;
        IntelRuntimeAdapter::new(driver, runtime_config.clone())
            .map_err(|error| error.to_string())?;
        let evaluator: Arc<dyn EvaluationExecutor> = Arc::new(
            evaluator_intel::start(evaluator_intel::IntelEvaluatorConfig {
                runtime: runtime_config,
                queue_capacity: NonZeroUsize::new(self.file.evaluator_queue_capacity)
                    .expect("validated evaluator queue capacity"),
                cache_capacity: NonZeroUsize::new(self.file.evaluator_cache_capacity)
                    .expect("validated evaluator cache capacity"),
                cache_results: false,
                evaluator_build: build_fingerprint(&self.file.compiler_build),
                runtime_build: build_fingerprint(&self.file.runtime_build),
                driver_version: self.file.intel_driver_version,
            })
            .map_err(|error| error.to_string())?,
        );
        let python = self
            .file
            .intel_python
            .as_ref()
            .expect("validated Intel Python");
        let helper = self
            .file
            .intel_helper
            .as_ref()
            .expect("validated Intel helper");
        let compiler: Arc<dyn IntelPayloadCompiler> = Arc::new(
            ExternalIntelPayloadCompiler::new(python)
                .with_fixed_argument(helper.as_os_str().to_owned()),
        );
        let target = CompilerEngineTarget::Intel {
            request: IntelTargetRequest::arc_b580(TargetDeviceRequest::BackendDefault)
                .with_reusable_command_api(IntelReusableCommandApi::Software),
            payload_compiler: compiler,
        };
        let workload = self.load_workload()?;
        let root = import(&self.file.source_dir).map_err(format_import_error)?;
        (0..count)
            .map(|_| {
                self.build_engine(
                    target.clone(),
                    Arc::clone(&evaluator),
                    workload.clone(),
                    root.clone(),
                )
            })
            .collect()
    }

    #[cfg(not(feature = "intel"))]
    fn build_intel_measure_engines(&self, _count: usize) -> Result<Vec<CompilerEngine>, String> {
        Err("Intel measurement support was not enabled at build time".to_owned())
    }

    fn disabled_target(&self) -> Result<CompilerEngineTarget, String> {
        Ok(match self.file.backend {
            CompilerDeploymentBackend::Nvidia => CompilerEngineTarget::Nvidia {
                request: NvidiaTargetRequest::sm(
                    TargetDeviceRequest::BackendDefault,
                    self.file.compute_capability_major,
                    self.file.compute_capability_minor,
                )
                .map_err(|error| error.to_string())?,
                payload_compiler: Arc::new(DisabledPayloadCompiler),
            },
            CompilerDeploymentBackend::IntelArcB580 => CompilerEngineTarget::Intel {
                request: IntelTargetRequest::arc_b580(TargetDeviceRequest::BackendDefault)
                    .with_reusable_command_api(IntelReusableCommandApi::Software),
                payload_compiler: Arc::new(DisabledPayloadCompiler),
            },
        })
    }

    fn load_workload(&self) -> Result<CompilerEvaluationWorkload, String> {
        let input_bundle =
            load_tensor_bundle(&self.file.inputs).map_err(|error| error.to_string())?;
        let case = CaseId::from(self.file.case_id.clone());
        Ok(CompilerEvaluationWorkload {
            execution: EvaluationExecutionTarget::ComponentSequence(ComponentSequenceId::from(
                self.file.execution_sequence.clone(),
            )),
            bindings: EvaluationBindingSet::default(),
            runtime: EvaluationRuntimePolicy {
                cases: CaseStatePolicy::ResetBeforeEachCase,
                session: EvaluationSessionPolicy::SteadyState,
            },
            accuracy_suite: AccuracySuite::try_from(vec![AccuracyCase {
                id: case.clone(),
                inputs: input_bundle.clone(),
            }])
            .map_err(|error| error.to_string())?,
            reference_outputs: ReferenceOutputBundle::try_from(vec![ReferenceCase {
                id: case.clone(),
                outputs: load_tensor_bundle(&self.file.references)
                    .map_err(|error| error.to_string())?,
            }])
            .map_err(|error| error.to_string())?,
            tolerance: TolerancePolicy::Uniform(
                Tolerance::try_from(ToleranceValues {
                    absolute: self.file.absolute_tolerance,
                    relative: self.file.relative_tolerance,
                })
                .map_err(|error| error.to_string())?,
            ),
            performance_suite: PerformanceSuite::try_from(vec![PerformanceCase {
                id: case,
                inputs: input_bundle,
            }])
            .map_err(|error| error.to_string())?,
        })
    }

    fn build_engine(
        &self,
        target: CompilerEngineTarget,
        evaluator: Arc<dyn EvaluationExecutor>,
        workload: CompilerEvaluationWorkload,
        root: native::importer::ImportedProgram,
    ) -> Result<CompilerEngine, String> {
        CompilerEngine::new(CompilerEngineConfig {
            domain: CompilerEngineDomain {
                target,
                bufferization: BufferizationPolicy::default(),
                lowering: LoweringPolicy::default(),
                evaluator,
                performance_warmup_iterations: self.file.performance_warmup_iterations,
            },
            root,
            candidate_mode: self.candidate_mode,
            measure_config_hash: self.measure_config_hash,
            measurement_policy: if self.file.runtime_accuracy_verification {
                CompilerMeasurementPolicy::VerifyAccuracy
            } else {
                CompilerMeasurementPolicy::PerformanceOnly
            },
            workload,
        })
        .map_err(|error| error.to_string())
    }
}

pub struct CompilerBackend {
    capability: wire::EngineCapability,
    engine_pool: CompilerEnginePool,
}

impl CompilerBackend {
    pub fn new(engine: CompilerEngine) -> Self {
        Self::with_engines(vec![engine]).expect("one compiler engine is a valid backend")
    }

    pub fn with_engines(engines: Vec<CompilerEngine>) -> Result<Self, String> {
        Self::with_engine_slots(vec![engines], DEFAULT_PREPARED_STATE_CACHE_CAPACITY)
    }

    fn with_engine_slots(
        engine_slots: Vec<Vec<CompilerEngine>>,
        prepared_state_cache_capacity: usize,
    ) -> Result<Self, String> {
        let Some(first) = engine_slots.iter().find_map(|slot| slot.first()) else {
            return Err("compiler backend requires at least one engine".to_owned());
        };
        if engine_slots.iter().any(Vec::is_empty) {
            return Err("compiler backend engine slots must not be empty".to_owned());
        }
        let identity = EngineIdentity::from_engine(first);
        if engine_slots
            .iter()
            .flatten()
            .any(|engine| EngineIdentity::from_engine(engine) != identity)
        {
            return Err("compiler backend engines have different identities".to_owned());
        }
        let capability = wire::EngineCapability {
            engine: Some(engine_identity_to_wire(identity)),
            artifact_formats: vec![wire::ArtifactFormatCapability {
                format_kind: wire::GraphArtifactFormatKind::GraphArtifactFormatAdapterSpecific
                    as i32,
                adapter_format_id: COMPILER_STATE_ARTIFACT_FORMAT,
            }],
            measure_config_encodings: vec![COMPILER_MEASURE_CONFIG_ENCODING],
            measurement_protocol_versions: vec![MEASUREMENT_PROTOCOL_VERSION],
        };
        Ok(Self {
            capability,
            engine_pool: CompilerEnginePool::new(engine_slots, prepared_state_cache_capacity),
        })
    }

    pub fn from_deployment(deployment: &CompilerDeployment) -> Result<Self, String> {
        let engines = deployment.build_measure_engines(
            deployment.file.measurement_pipeline_slots * COMPILER_MEASUREMENT_ENGINES_PER_SLOT,
        )?;
        let mut engines = engines.into_iter();
        let slots = (0..deployment.file.measurement_pipeline_slots)
            .map(|_| {
                engines
                    .by_ref()
                    .take(COMPILER_MEASUREMENT_ENGINES_PER_SLOT)
                    .collect()
            })
            .collect();
        Self::with_engine_slots(slots, deployment.file.prepared_state_cache_capacity)
    }
}

impl AgentBackend for CompilerBackend {
    fn capability(&self) -> wire::EngineCapability {
        self.capability.clone()
    }

    fn concurrent_job_slots(&self) -> u32 {
        self.engine_pool.capacity()
    }

    fn measure(
        &self,
        lease: &wire::MeasureLease,
        subjects: Vec<FetchedSubject>,
    ) -> Result<AgentMeasurement, AgentBackendError> {
        if lease.measure_config_encoding != COMPILER_MEASURE_CONFIG_ENCODING {
            return Err(backend_error(
                wire::MeasureAttemptOutcome::MeasureOutcomeUnsupported,
                1,
                "unsupported compiler measure config encoding",
                false,
            ));
        }
        let config_hash = MeasureConfigHash::from_bytes(
            lease
                .measure_config_hash
                .as_slice()
                .try_into()
                .map_err(|_| {
                    backend_error(
                        wire::MeasureAttemptOutcome::MeasureOutcomeUnsupported,
                        2,
                        "compiler config hash has the wrong width",
                        false,
                    )
                })?,
        );
        let options = decode_compiler_measure_options(config_hash, &lease.measure_config_payload)
            .map_err(|error| {
            backend_error(
                wire::MeasureAttemptOutcome::MeasureOutcomeUnsupported,
                3,
                error.to_string(),
                false,
            )
        })?;
        let kind = wire::MeasureKind::try_from(lease.kind).map_err(|_| {
            backend_error(
                wire::MeasureAttemptOutcome::MeasureOutcomeUnsupported,
                4,
                "unknown compiler measurement kind",
                false,
            )
        })?;
        let expected_subjects = match kind {
            wire::MeasureKind::Single => 1,
            wire::MeasureKind::SymmetricPair => 2,
            wire::MeasureKind::Unspecified => {
                return Err(backend_error(
                    wire::MeasureAttemptOutcome::MeasureOutcomeUnsupported,
                    5,
                    "compiler measurement kind is unspecified",
                    false,
                ));
            }
        };
        if subjects.len() != expected_subjects || subjects.len() != lease.subjects.len() {
            return Err(backend_error(
                wire::MeasureAttemptOutcome::MeasureOutcomeUnsupported,
                6,
                "compiler subject count does not match measurement kind",
                false,
            ));
        }

        let mut engine_slot = self.engine_pool.checkout().map_err(|message| {
            backend_error(
                wire::MeasureAttemptOutcome::MeasureOutcomeAgentInternal,
                7,
                message,
                false,
            )
        })?;
        let lanes = engine_slot.lanes();
        if lanes.len() < subjects.len() {
            return Err(backend_error(
                wire::MeasureAttemptOutcome::MeasureOutcomeAgentInternal,
                8,
                "compiler engine pool is smaller than the measurement subject count",
                false,
            ));
        }
        let lanes = &mut lanes[..subjects.len()];
        if engine_identity_to_wire(EngineIdentity::from_engine(&lanes[0].engine))
            != lease.engine.clone().unwrap_or_default()
        {
            return Err(backend_error(
                wire::MeasureAttemptOutcome::MeasureOutcomeUnsupported,
                9,
                "compiler engine identity mismatch",
                false,
            ));
        }
        if lanes
            .iter()
            .any(|lane| lane.engine.measure_config_hash() != config_hash)
        {
            return Err(backend_error(
                wire::MeasureAttemptOutcome::MeasureOutcomeUnsupported,
                10,
                "compiler measure config hash mismatch",
                false,
            ));
        }

        let started = Instant::now();
        let mut restored = Vec::with_capacity(subjects.len());
        for (lane, subject) in lanes.iter_mut().zip(&subjects) {
            if subject.format_kind
                != wire::GraphArtifactFormatKind::GraphArtifactFormatAdapterSpecific
                || subject.adapter_format_id != COMPILER_STATE_ARTIFACT_FORMAT
            {
                return Err(backend_error(
                    wire::MeasureAttemptOutcome::MeasureOutcomeUnsupported,
                    11,
                    "unsupported compiler state artifact format",
                    false,
                ));
            }
            let graph_hash = GraphHash::from_bytes(subject.graph_hash);
            let artifact = GraphArtifact {
                graph_hash,
                format: GraphArtifactFormat::AdapterSpecific(COMPILER_STATE_ARTIFACT_FORMAT),
                bytes: subject.bytes.clone(),
            };
            let key = PreparedStateCacheKey {
                graph_hash,
                engine: EngineIdentity::from_engine(&lane.engine),
                measure_config_hash: config_hash,
            };
            match lane.restore_or_reuse(key, &artifact) {
                Ok(graph) => restored.push(graph),
                Err(error) => {
                    return Err(backend_error(
                        wire::MeasureAttemptOutcome::MeasureOutcomeUnsupported,
                        12,
                        error.to_string(),
                        false,
                    ));
                }
            }
        }

        let mut telemetry = vec![phase_sample(
            wire::JobPhase::VerifyArtifact,
            started.elapsed(),
        )];
        let compile_elapsed = prepare_measurements(lanes, &restored, options)?;
        telemetry.push(phase_sample(wire::JobPhase::Compile, started.elapsed()));
        measure_subjects(
            lanes,
            MeasurementJob {
                lease,
                subjects: &subjects,
                graphs: &restored,
                compile_elapsed: &compile_elapsed,
                options,
                kind,
                started,
            },
            telemetry,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreparedStateCacheKey {
    graph_hash: GraphHash,
    engine: EngineIdentity,
    measure_config_hash: MeasureConfigHash,
}

struct CompilerEngineLane {
    engine: CompilerEngine,
    prepared_states: VecDeque<PreparedStateCacheEntry>,
    prepared_state_cache_capacity: usize,
}

struct PreparedStateCacheEntry {
    key: PreparedStateCacheKey,
    graph: <CompilerEngine as GraphEngine>::Graph,
    prepared: bool,
}

impl CompilerEngineLane {
    fn new(engine: CompilerEngine, prepared_state_cache_capacity: usize) -> Self {
        Self {
            engine,
            prepared_states: VecDeque::with_capacity(prepared_state_cache_capacity),
            prepared_state_cache_capacity,
        }
    }

    fn restore_or_reuse(
        &mut self,
        key: PreparedStateCacheKey,
        artifact: &GraphArtifact,
    ) -> Result<<CompilerEngine as GraphEngine>::Graph, gz_engine::EngineError> {
        if let Some(index) = self
            .prepared_states
            .iter()
            .position(|entry| entry.key == key)
        {
            let entry = self
                .prepared_states
                .remove(index)
                .expect("located prepared state is present");
            let graph = entry.graph;
            self.prepared_states.push_back(entry);
            return Ok(graph);
        }

        let graph = self.engine.restore_graph(artifact)?;
        if self.prepared_states.len() == self.prepared_state_cache_capacity {
            let evicted = self
                .prepared_states
                .pop_front()
                .expect("full prepared-state cache is non-empty");
            let _ = self
                .engine
                .release(std::slice::from_ref(&evicted.graph), &[]);
        }
        self.prepared_states.push_back(PreparedStateCacheEntry {
            key,
            graph,
            prepared: false,
        });
        Ok(graph)
    }

    fn prepare(
        &mut self,
        graph: <CompilerEngine as GraphEngine>::Graph,
        options: MeasureOptions,
    ) -> Result<Duration, AgentBackendError> {
        let Some(index) = self
            .prepared_states
            .iter()
            .position(|entry| entry.graph == graph)
        else {
            return Err(backend_error(
                wire::MeasureAttemptOutcome::MeasureOutcomeAgentInternal,
                24,
                "restored compiler state is absent from its engine-lane cache",
                false,
            ));
        };
        if self.prepared_states[index].prepared {
            return Ok(Duration::ZERO);
        }
        let elapsed = prepare_one(&mut self.engine, graph, options)?;
        self.prepared_states[index].prepared = true;
        Ok(elapsed)
    }
}

impl Drop for CompilerEngineLane {
    fn drop(&mut self) {
        let graphs = self
            .prepared_states
            .drain(..)
            .map(|entry| entry.graph)
            .collect::<Vec<_>>();
        let _ = self.engine.release(&graphs, &[]);
    }
}

struct CompilerEnginePool {
    capacity: u32,
    available: Mutex<Vec<Vec<CompilerEngineLane>>>,
    available_changed: Condvar,
}

impl CompilerEnginePool {
    fn new(slots: Vec<Vec<CompilerEngine>>, prepared_state_cache_capacity: usize) -> Self {
        Self {
            capacity: u32::try_from(slots.len()).expect("compiler engine slot count fits u32"),
            available: Mutex::new(
                slots
                    .into_iter()
                    .map(|slot| {
                        slot.into_iter()
                            .map(|engine| {
                                CompilerEngineLane::new(engine, prepared_state_cache_capacity)
                            })
                            .collect()
                    })
                    .collect(),
            ),
            available_changed: Condvar::new(),
        }
    }

    const fn capacity(&self) -> u32 {
        self.capacity
    }

    fn checkout(&self) -> Result<CompilerEngineSlot<'_>, &'static str> {
        let mut available = self
            .available
            .lock()
            .map_err(|_| "compiler engine pool lock is poisoned")?;
        while available.is_empty() {
            available = self
                .available_changed
                .wait(available)
                .map_err(|_| "compiler engine pool lock is poisoned")?;
        }
        let lanes = available
            .pop()
            .expect("non-empty compiler engine pool has a slot");
        Ok(CompilerEngineSlot {
            pool: self,
            lanes: Some(lanes),
        })
    }
}

struct CompilerEngineSlot<'a> {
    pool: &'a CompilerEnginePool,
    lanes: Option<Vec<CompilerEngineLane>>,
}

impl CompilerEngineSlot<'_> {
    fn lanes(&mut self) -> &mut Vec<CompilerEngineLane> {
        self.lanes
            .as_mut()
            .expect("checked-out compiler engine slot is present")
    }
}

impl Drop for CompilerEngineSlot<'_> {
    fn drop(&mut self) {
        let Some(lanes) = self.lanes.take() else {
            return;
        };
        if let Ok(mut available) = self.pool.available.lock() {
            available.push(lanes);
            self.pool.available_changed.notify_one();
        }
    }
}

#[must_use]
pub fn encode_compiler_measure_options(options: MeasureOptions) -> EncodedMeasureConfig {
    let mut payload = Vec::with_capacity(COMPILER_CONFIG_BYTES);
    payload.extend_from_slice(COMPILER_CONFIG_MAGIC);
    payload.extend_from_slice(&options.samples.to_le_bytes());
    payload.extend_from_slice(&options.timeout_ms.unwrap_or(0).to_le_bytes());
    payload.push(u8::from(options.deterministic));
    EncodedMeasureConfig {
        encoding: COMPILER_MEASURE_CONFIG_ENCODING,
        payload,
    }
}

pub fn decode_compiler_measure_options(
    config_hash: MeasureConfigHash,
    payload: &[u8],
) -> ServiceResult<MeasureOptions> {
    if payload.len() != COMPILER_CONFIG_BYTES || &payload[..4] != COMPILER_CONFIG_MAGIC {
        return Err(ServiceError::protocol(
            "invalid compiler measure config payload",
        ));
    }
    let samples = u32::from_le_bytes(payload[4..8].try_into().unwrap());
    let timeout = u64::from_le_bytes(payload[8..16].try_into().unwrap());
    if timeout != 0 {
        return Err(ServiceError::protocol(
            "compiler measurement timeout belongs to the coordinator, not MeasureOptions",
        ));
    }
    let deterministic = match payload[16] {
        0 => false,
        1 => true,
        _ => {
            return Err(ServiceError::protocol(
                "invalid compiler deterministic flag",
            ));
        }
    };
    if !deterministic {
        return Err(ServiceError::protocol(
            "compiler measurement must be deterministic",
        ));
    }
    MeasureOptions::new(config_hash, samples, None, true)
        .map_err(|error| ServiceError::protocol(error.to_string()))
}

pub fn compiler_measure_submission(
    engine: &CompilerEngine,
    graph: <CompilerEngine as GraphEngine>::Graph,
    options: MeasureOptions,
    target_device_profile_hash: DeviceProfileHash,
) -> Result<MeasureSubmission, gz_engine::EngineError> {
    MeasureSubmission::from_engine(
        engine,
        graph,
        options,
        encode_compiler_measure_options(options),
        target_device_profile_hash,
    )
}

pub fn compiler_measure_submission_pair(
    engine: &CompilerEngine,
    graphs: [<CompilerEngine as GraphEngine>::Graph; 2],
    options: MeasureOptions,
    target_device_profile_hash: DeviceProfileHash,
) -> Result<MeasureSubmission, gz_engine::EngineError> {
    MeasureSubmission::from_engine_pair(
        engine,
        graphs,
        options,
        encode_compiler_measure_options(options),
        target_device_profile_hash,
    )
}

fn prepare_measurements(
    lanes: &mut [CompilerEngineLane],
    graphs: &[<CompilerEngine as GraphEngine>::Graph],
    options: MeasureOptions,
) -> Result<Vec<Duration>, AgentBackendError> {
    debug_assert_eq!(lanes.len(), graphs.len());
    match (lanes, graphs) {
        ([lane], [graph]) => lane.prepare(*graph, options).map(|elapsed| vec![elapsed]),
        ([first_lane, second_lane], [first_graph, second_graph]) => {
            let prepared = std::thread::scope(|scope| {
                let first = scope.spawn(|| first_lane.prepare(*first_graph, options));
                let second = scope.spawn(|| second_lane.prepare(*second_graph, options));
                [first.join(), second.join()]
            });
            prepared
                .into_iter()
                .map(|result| {
                    result.map_err(|_| {
                        backend_error(
                            wire::MeasureAttemptOutcome::MeasureOutcomeAgentInternal,
                            13,
                            "compiler preparation worker panicked",
                            false,
                        )
                    })?
                })
                .collect()
        }
        _ => Err(backend_error(
            wire::MeasureAttemptOutcome::MeasureOutcomeAgentInternal,
            14,
            "compiler measurement supports one subject or one symmetric pair",
            false,
        )),
    }
}

fn prepare_one(
    engine: &mut CompilerEngine,
    graph: <CompilerEngine as GraphEngine>::Graph,
    options: MeasureOptions,
) -> Result<Duration, AgentBackendError> {
    let started = Instant::now();
    engine
        .prepare_measurement(graph, options)
        .map_err(|error| {
            backend_error(
                wire::MeasureAttemptOutcome::MeasureOutcomeCompileFailed,
                15,
                error.to_string(),
                false,
            )
        })?;
    Ok(started.elapsed())
}

struct MeasurementJob<'a> {
    lease: &'a wire::MeasureLease,
    subjects: &'a [FetchedSubject],
    graphs: &'a [<CompilerEngine as GraphEngine>::Graph],
    compile_elapsed: &'a [Duration],
    options: MeasureOptions,
    kind: wire::MeasureKind,
    started: Instant,
}

fn measure_subjects(
    lanes: &mut [CompilerEngineLane],
    job: MeasurementJob<'_>,
    mut telemetry: Vec<wire::TelemetrySample>,
) -> Result<AgentMeasurement, AgentBackendError> {
    let MeasurementJob {
        lease,
        subjects,
        graphs,
        compile_elapsed,
        options,
        kind,
        started,
    } = job;
    let runs = measurement_runs(
        kind,
        options.samples,
        lease.job_id.first().copied().unwrap_or(0),
    );
    let mut accumulators = subjects
        .iter()
        .map(|subject| SubjectAccumulator {
            logical_index: subject.logical_index,
            graph_hash: subject.graph_hash,
            execution_ns: Vec::with_capacity(options.samples as usize),
            end_to_end_ns: Vec::with_capacity(options.samples as usize),
            fixed_reward: None,
        })
        .collect::<Vec<_>>();
    let mut warmed = vec![false; subjects.len()];

    for (subject_index, samples) in runs {
        if accumulators[subject_index].fixed_reward.is_some() {
            continue;
        }
        let run_options = MeasureOptions::new(options.config_hash, samples, None, true)
            .expect("validated compiler run options");
        let warmup = if warmed[subject_index] {
            0
        } else {
            lanes[subject_index].engine.performance_warmup_iterations()
        };
        let measured = lanes[subject_index]
            .engine
            .measure_with_warmup(graphs[subject_index], run_options, warmup)
            .map_err(|error| {
                backend_error(
                    wire::MeasureAttemptOutcome::MeasureOutcomeAgentInternal,
                    16,
                    error.to_string(),
                    false,
                )
            })?;
        warmed[subject_index] = true;
        if !measured.measured || !measured.valid {
            let failure = measured.failure.as_ref();
            let outcome = if failure.is_some_and(|failure| failure.code == COMPILE_FAILURE_CODE) {
                wire::MeasureAttemptOutcome::MeasureOutcomeCompileFailed
            } else {
                wire::MeasureAttemptOutcome::MeasureOutcomeExecutionFailed
            };
            return Err(backend_error(
                outcome,
                failure.map_or(17, |failure| failure.code.get()),
                failure.map_or_else(
                    || "compiler returned an invalid measurement".to_owned(),
                    |failure| failure.message.to_string(),
                ),
                false,
            ));
        }
        let reward = measured.scalar_reward.ok_or_else(|| {
            backend_error(
                wire::MeasureAttemptOutcome::MeasureOutcomeAgentInternal,
                18,
                "compiler omitted scalar reward",
                false,
            )
        })?;
        let evidence = decode_measurement_evidence(&measured.metadata).map_err(|error| {
            backend_error(
                wire::MeasureAttemptOutcome::MeasureOutcomeAgentInternal,
                19,
                error.to_string(),
                false,
            )
        })?;
        let accumulator = &mut accumulators[subject_index];
        match evidence {
            Some(evidence) => {
                if evidence.execution_ns.len() != samples as usize
                    || evidence.end_to_end_ns.len() != samples as usize
                {
                    return Err(backend_error(
                        wire::MeasureAttemptOutcome::MeasureOutcomeAgentInternal,
                        20,
                        "compiler returned the wrong number of timing samples",
                        false,
                    ));
                }
                accumulator.execution_ns.extend(evidence.execution_ns);
                accumulator.end_to_end_ns.extend(evidence.end_to_end_ns);
            }
            None if reward == ACCURACY_FAILURE_REWARD => {
                accumulator.fixed_reward = Some(f64::from(reward));
            }
            None => {
                return Err(backend_error(
                    wire::MeasureAttemptOutcome::MeasureOutcomeAgentInternal,
                    21,
                    "compiler timing evidence is absent",
                    false,
                ));
            }
        }
    }

    let mut reports = Vec::with_capacity(accumulators.len());
    for (accumulator, compile_elapsed) in accumulators.iter().zip(compile_elapsed) {
        let (scalar_reward, engine_metadata) = match accumulator.fixed_reward {
            Some(reward) => (reward, accuracy_failure_evidence().bytes),
            None => {
                if accumulator.execution_ns.len() != options.samples as usize {
                    return Err(backend_error(
                        wire::MeasureAttemptOutcome::MeasureOutcomeAgentInternal,
                        22,
                        "compiler paired schedule omitted timing samples",
                        false,
                    ));
                }
                let metadata = CompilerMeasurementEvidence {
                    execution_ns: accumulator.execution_ns.clone(),
                    end_to_end_ns: accumulator.end_to_end_ns.clone(),
                }
                .encode()
                .map_err(|error| {
                    backend_error(
                        wire::MeasureAttemptOutcome::MeasureOutcomeAgentInternal,
                        23,
                        error.to_string(),
                        false,
                    )
                })?;
                (
                    -(median_ns(&accumulator.execution_ns) as f64) / 1_000_000.0,
                    metadata.bytes,
                )
            }
        };
        reports.push(wire::SubjectMeasurement {
            logical_index: accumulator.logical_index,
            graph_hash: accumulator.graph_hash.to_vec(),
            compile_elapsed_ns: Some(duration_ns(*compile_elapsed)),
            capture_elapsed_ns: None,
            scalar_reward: Some(scalar_reward),
            engine_metadata,
        });
    }

    let mut schedule_index = 0_u32;
    let mut subject_sample_indices = vec![0_u32; accumulators.len()];
    let mut samples = Vec::with_capacity(
        accumulators
            .iter()
            .map(|accumulator| accumulator.execution_ns.len())
            .sum(),
    );
    for (subject_index, count) in measurement_runs(
        kind,
        options.samples,
        lease.job_id.first().copied().unwrap_or(0),
    ) {
        let accumulator = &accumulators[subject_index];
        if accumulator.fixed_reward.is_some() {
            continue;
        }
        let start = subject_sample_indices[subject_index] as usize;
        let end = start + count as usize;
        for sample_index in start..end {
            samples.push(wire::MeasurementSample {
                schedule_index,
                logical_index: accumulator.logical_index,
                subject_sample_index: subject_sample_indices[subject_index],
                gpu_elapsed_ns: accumulator.execution_ns[sample_index],
                end_to_end_elapsed_ns: accumulator.end_to_end_ns[sample_index],
            });
            schedule_index += 1;
            subject_sample_indices[subject_index] += 1;
        }
    }

    telemetry.push(phase_sample(wire::JobPhase::Measure, started.elapsed()));
    Ok(AgentMeasurement {
        subjects: reports,
        samples,
        telemetry,
    })
}

fn measurement_runs(kind: wire::MeasureKind, samples: u32, parity: u8) -> Vec<(usize, u32)> {
    if kind == wire::MeasureKind::Single {
        return vec![(0, samples)];
    }
    let mut sequence = Vec::with_capacity(samples as usize * 2);
    let swap = parity & 1 != 0;
    for sample in 0..samples {
        let first = usize::from((sample & 1 != 0) ^ swap);
        sequence.push(first);
        sequence.push(first ^ 1);
    }
    let mut runs = Vec::<(usize, u32)>::new();
    for subject in sequence {
        if let Some((previous, count)) = runs.last_mut()
            && *previous == subject
        {
            *count += 1;
        } else {
            runs.push((subject, 1));
        }
    }
    runs
}

fn median_ns(samples: &[u64]) -> u64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = sorted.len().div_ceil(2);
    sorted[rank.saturating_sub(1)]
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn phase_sample(phase: wire::JobPhase, elapsed: Duration) -> wire::TelemetrySample {
    wire::TelemetrySample {
        attempt_elapsed_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
        phase: phase as i32,
        telemetry: None,
    }
}

struct SubjectAccumulator {
    logical_index: u32,
    graph_hash: [u8; 32],
    execution_ns: Vec<u64>,
    end_to_end_ns: Vec<u64>,
    fixed_reward: Option<f64>,
}

fn validate_deployment(file: &CompilerDeploymentFile) -> Result<(), String> {
    if file.schema != DEPLOYMENT_SCHEMA {
        return Err(format!(
            "unsupported compiler deployment schema {:?}",
            file.schema
        ));
    }
    for (path, name) in [
        (&file.source_dir, "source_dir"),
        (&file.inputs, "inputs"),
        (&file.references, "references"),
    ] {
        if !path.exists() {
            return Err(format!("{name} does not exist: {}", path.display()));
        }
    }
    let backend_paths = match file.backend {
        CompilerDeploymentBackend::Nvidia => [
            (file.cuda_python.as_ref(), "cuda_python"),
            (file.cuda_helper.as_ref(), "cuda_helper"),
        ],
        CompilerDeploymentBackend::IntelArcB580 => [
            (file.intel_python.as_ref(), "intel_python"),
            (file.intel_helper.as_ref(), "intel_helper"),
        ],
    };
    for (path, name) in backend_paths {
        let path = path.ok_or_else(|| format!("{name} is required for the selected backend"))?;
        if !path.exists() {
            return Err(format!("{name} does not exist: {}", path.display()));
        }
    }
    if file.evaluator_queue_capacity == 0 || file.evaluator_cache_capacity == 0 {
        return Err("evaluator capacities must be greater than zero".to_owned());
    }
    if file.measurement_pipeline_slots == 0
        || file.measurement_pipeline_slots > MAX_COMPILER_MEASUREMENT_PIPELINE_SLOTS
    {
        return Err(format!(
            "measurement_pipeline_slots must be in 1..={MAX_COMPILER_MEASUREMENT_PIPELINE_SLOTS}"
        ));
    }
    if file.prepared_state_cache_capacity == 0
        || file.prepared_state_cache_capacity > MAX_PREPARED_STATE_CACHE_CAPACITY
    {
        return Err(format!(
            "prepared_state_cache_capacity must be in 1..={MAX_PREPARED_STATE_CACHE_CAPACITY}"
        ));
    }
    if file.case_id.is_empty()
        || file.execution_sequence.is_empty()
        || file.compiler_build.is_empty()
        || file.runtime_build.is_empty()
    {
        return Err("compiler workload identities must not be empty".to_owned());
    }
    for (value, name) in [
        (file.absolute_tolerance, "absolute_tolerance"),
        (file.relative_tolerance, "relative_tolerance"),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{name} must be finite and non-negative"));
        }
    }
    if file.device_profile.usable_memory_bytes == 0 {
        return Err("device profile usable_memory_bytes must be greater than zero".to_owned());
    }
    for (value, name) in [
        (&file.device_profile.platform_family, "platform_family"),
        (&file.device_profile.board_model, "board_model"),
        (&file.device_profile.soc, "soc"),
        (&file.device_profile.gpu_architecture, "gpu_architecture"),
        (
            &file.device_profile.operating_system_image,
            "operating_system_image",
        ),
        (&file.device_profile.platform_release, "platform_release"),
        (&file.device_profile.cuda_version, "cuda_version"),
        (
            &file.device_profile.gpu_driver_version,
            "gpu_driver_version",
        ),
        (&file.device_profile.power_profile, "power_profile"),
        (&file.device_profile.clock_policy, "clock_policy"),
        (&file.device_profile.cooling_policy, "cooling_policy"),
        (&file.device_profile.agent_image, "agent_image"),
    ] {
        if value.is_empty() {
            return Err(format!("device profile {name} must not be empty"));
        }
    }
    parse_candidate_mode(&file.candidate_mode)?;
    Ok(())
}

fn format_import_error(error: ImportError) -> String {
    match error {
        ImportError::Invalid(diagnostics) => diagnostics
            .as_ref()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; "),
        error => error.to_string(),
    }
}

fn parse_candidate_mode(value: &str) -> Result<CompilerCandidateMode, String> {
    match value {
        "sitewise" => Ok(CompilerCandidateMode::Sitewise),
        "hybrid-v1" => Ok(CompilerCandidateMode::HybridV1),
        "hybrid-provider-v1" => Ok(CompilerCandidateMode::HybridProviderV1),
        "hybrid-nvidia-v1" => Ok(CompilerCandidateMode::HybridNvidiaV1),
        "hybrid-nvidia-v2" => Ok(CompilerCandidateMode::HybridNvidiaV2),
        "end-to-end-v1" => Ok(CompilerCandidateMode::EndToEndV1),
        "end-to-end-nvidia-v1" => Ok(CompilerCandidateMode::EndToEndNvidiaV1),
        "end-to-end-nvidia-v2" => Ok(CompilerCandidateMode::EndToEndNvidiaV2),
        "end-to-end-nvidia-v3" => Ok(CompilerCandidateMode::EndToEndNvidiaV3),
        "end-to-end-nvidia-v4" => Ok(CompilerCandidateMode::EndToEndNvidiaV4),
        "end-to-end-nvidia-v5" => Ok(CompilerCandidateMode::EndToEndNvidiaV5),
        _ => Err(format!("unknown compiler candidate mode: {value}")),
    }
}

fn measurement_config_hash(
    file: &CompilerDeploymentFile,
    candidate_mode: CompilerCandidateMode,
) -> Result<MeasureConfigHash, String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"gz-compiler-measure-config-v1\0");
    update_string(&mut hasher, DEPLOYMENT_SCHEMA);
    update_file(&mut hasher, &file.source_dir.join("graphs.json"))?;
    update_file(&mut hasher, &file.inputs)?;
    update_file(&mut hasher, &file.references)?;
    hasher.update(&[match file.backend {
        CompilerDeploymentBackend::Nvidia => 0,
        CompilerDeploymentBackend::IntelArcB580 => 1,
    }]);
    match file.backend {
        CompilerDeploymentBackend::Nvidia => {
            update_file(
                &mut hasher,
                file.cuda_helper.as_ref().expect("validated CUDA helper"),
            )?;
        }
        CompilerDeploymentBackend::IntelArcB580 => {
            update_file(
                &mut hasher,
                file.intel_helper.as_ref().expect("validated Intel helper"),
            )?;
            hasher.update(&file.intel_driver_version.to_le_bytes());
        }
    }
    hasher.update(&file.compute_capability_major.to_le_bytes());
    hasher.update(&file.compute_capability_minor.to_le_bytes());
    hasher.update(&file.performance_warmup_iterations.to_le_bytes());
    hasher.update(&[u8::from(file.runtime_accuracy_verification)]);
    hasher.update(&[match candidate_mode {
        CompilerCandidateMode::Sitewise => 0,
        CompilerCandidateMode::HybridV1 => 1,
        CompilerCandidateMode::HybridProviderV1 => 2,
        CompilerCandidateMode::HybridNvidiaV1 => 3,
        CompilerCandidateMode::HybridNvidiaV2 => 4,
        CompilerCandidateMode::EndToEndV1 => 5,
        CompilerCandidateMode::EndToEndNvidiaV1 => 6,
        CompilerCandidateMode::EndToEndNvidiaV2 => 7,
        CompilerCandidateMode::EndToEndNvidiaV3 => 8,
        CompilerCandidateMode::EndToEndNvidiaV4 => 9,
        CompilerCandidateMode::EndToEndNvidiaV5 => 10,
    }]);
    update_string(&mut hasher, "bufferization-default-v1");
    update_string(&mut hasher, "lowering-default-v1");
    update_string(&mut hasher, &file.case_id);
    update_string(&mut hasher, &file.execution_sequence);
    hasher.update(&file.absolute_tolerance.to_bits().to_le_bytes());
    hasher.update(&file.relative_tolerance.to_bits().to_le_bytes());
    update_string(&mut hasher, "reset-before-each-case");
    update_string(&mut hasher, "steady-state");
    update_string(&mut hasher, "reward-negative-execution-median-ms");
    update_string(
        &mut hasher,
        "symmetric-pair-parallel-prepare-abba-job-parity-v2",
    );
    update_string(
        &mut hasher,
        "accuracy-once-per-restored-state-fresh-performance-results",
    );
    update_string(
        &mut hasher,
        "bounded-compile-overlap-single-serialized-evaluator-v1",
    );
    hasher.update(&(file.measurement_pipeline_slots as u64).to_le_bytes());
    hasher.update(&(file.prepared_state_cache_capacity as u64).to_le_bytes());
    Ok(MeasureConfigHash::from_bytes(*hasher.finalize().as_bytes()))
}

fn update_file(hasher: &mut blake3::Hasher, path: &Path) -> Result<(), String> {
    let mut file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let size = file
        .metadata()
        .map_err(|error| format!("failed to stat {}: {error}", path.display()))?
        .len();
    hasher.update(&size.to_le_bytes());
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(())
}

fn update_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

const fn default_measurement_pipeline_slots() -> usize {
    DEFAULT_COMPILER_MEASUREMENT_PIPELINE_SLOTS
}

const fn default_prepared_state_cache_capacity() -> usize {
    DEFAULT_PREPARED_STATE_CACHE_CAPACITY
}

const fn default_runtime_accuracy_verification() -> bool {
    true
}

fn device_profile(file: &CompilerDeploymentFile) -> wire::DeviceProfile {
    let profile = &file.device_profile;
    wire::DeviceProfile {
        platform_family: profile.platform_family.clone(),
        board_model: profile.board_model.clone(),
        soc: profile.soc.clone(),
        gpu_architecture: profile.gpu_architecture.clone(),
        usable_memory_bytes: profile.usable_memory_bytes,
        operating_system_image_digest: manifest_digest(
            b"gz-os-image-v1",
            &profile.operating_system_image,
        )
        .to_vec(),
        platform_release: profile.platform_release.clone(),
        cuda_version: profile.cuda_version.clone(),
        gpu_driver_version: profile.gpu_driver_version.clone(),
        compiler_version: file.compiler_build.clone(),
        compiler_runtime_version: file.runtime_build.clone(),
        agent_image_digest: manifest_digest(b"gz-agent-image-v1", &profile.agent_image).to_vec(),
        power_profile: profile.power_profile.clone(),
        clock_policy: profile.clock_policy.clone(),
        cooling_policy: profile.cooling_policy.clone(),
        measurement_protocol_version: MEASUREMENT_PROTOCOL_VERSION,
    }
}

fn manifest_digest(domain: &[u8], value: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    update_string(&mut hasher, value);
    *hasher.finalize().as_bytes()
}

fn build_fingerprint(value: &str) -> BuildFingerprint {
    BuildFingerprint::from(manifest_digest(b"gz-compiler-build-v1", value))
}

fn backend_error(
    outcome: wire::MeasureAttemptOutcome,
    code: u32,
    message: impl Into<String>,
    retriable: bool,
) -> AgentBackendError {
    AgentBackendError::new(outcome, code, message, retriable)
}

struct DisabledPayloadCompiler;

impl NvidiaPayloadCompiler for DisabledPayloadCompiler {
    fn compile_native_cuda(
        &self,
        _request: &NativeCudaCompileRequest,
    ) -> BackendResult<NativeCudaCompileResult> {
        Err(internal_backend_error(
            "local compiler measurement is disabled; use the remote measurement gateway",
        ))
    }
}

impl IntelPayloadCompiler for DisabledPayloadCompiler {
    fn compile(
        &self,
        _request: &IntelPayloadCompileRequest<'_>,
    ) -> BackendResult<IntelCompiledKernel> {
        Err(internal_backend_error(
            "local compiler measurement is disabled; use the remote measurement gateway",
        ))
    }
}

struct DisabledEvaluator;

impl EvaluationExecutor for DisabledEvaluator {
    fn evaluate_accuracy(
        &self,
        _request: AccuracyEvaluationRequest,
    ) -> Result<AccuracyReport, EvaluationExecutionError> {
        Err(EvaluationExecutionError::executor(DisabledEvaluation))
    }

    fn evaluate_performance(
        &self,
        _request: PerformanceEvaluationRequest,
    ) -> Result<PerformanceReport, EvaluationExecutionError> {
        Err(EvaluationExecutionError::executor(DisabledEvaluation))
    }
}

#[derive(Debug)]
struct DisabledEvaluation;

impl std::fmt::Display for DisabledEvaluation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("local compiler evaluation is disabled")
    }
}

impl std::error::Error for DisabledEvaluation {}
