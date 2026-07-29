use artifact::{ByteSize, Device, NvidiaTarget, RuntimeTarget};
use graphs_compiler::{
    backends::{
        BackendResult, LoweringPolicy, TargetDeviceRequest, internal_backend_error,
        nvidia::{
            lowering::NvidiaPayloadCompiler,
            providers::native_cuda::{
                compile::{
                    NATIVE_CUDA_COMPILE_SCHEMA, NativeCudaCompileRequest, NativeCudaCompileResult,
                    NativeCudaCompiledPayload, NativeCudaCompilerMetadata, NativeCudaPayloadFormat,
                },
                ir::CudaKernelSymbol,
            },
            target::NvidiaTargetRequest,
        },
    },
    bufferization::BufferizationPolicy,
    evaluators::{
        AccuracyCase, AccuracyEvaluationRequest, AccuracyReport, AccuracySuite, AccuracySummary,
        AccuracyVerdict, BuildFingerprint, CaseId, CaseStatePolicy, EvaluationBindingSet,
        EvaluationExecutionError, EvaluationExecutionTarget, EvaluationExecutor,
        EvaluationRuntimePolicy, EvaluationSessionPolicy, LatencyDistribution,
        MeasurementEnvironment, PerformanceCase, PerformanceEvaluationRequest, PerformanceReport,
        PerformanceSuite, ReferenceCase, ReferenceOutputBundle, RuntimeMemoryMetrics,
        TensorBindings, ThroughputMetrics, Tolerance, TolerancePolicy, ToleranceValues,
        accuracy_identity, performance_identity,
    },
    importer::import,
};
use gz_engine::{
    CandidateOptions, EngineError, EngineIdentity, GraphArtifactFormat, GraphEngine,
    MeasureConfigHash, MeasureOptions,
};
use gz_engine_compiler::{
    ACCURACY_EVALUATION_FAILURE_CODE, ACCURACY_FAILURE_REWARD, COMPILE_FAILURE_CODE,
    COMPILER_STATE_ARTIFACT_FORMAT, CompilerEngine, CompilerEngineConfig, CompilerEngineDomain,
    CompilerEvaluationWorkload, PERFORMANCE_EVALUATION_FAILURE_CODE,
};
use std::{
    fmt, fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

const MEASURE_CONFIG_HASH: MeasureConfigHash = MeasureConfigHash::from_bytes([7; 32]);

#[test]
fn identity_enumeration_apply_and_artifact_are_stable() {
    let (mut first, _, _) = engine(EvaluatorBehavior::Pass, false);
    let (mut second, _, _) = engine(EvaluatorBehavior::Pass, false);
    let first_root = first.root();
    let second_root = second.root();

    assert_eq!(
        EngineIdentity::from_engine(&first),
        EngineIdentity::from_engine(&second)
    );
    assert_eq!(
        first.hash(first_root).unwrap(),
        second.hash(second_root).unwrap()
    );

    let mut first_candidates = Vec::new();
    first
        .candidates(
            first_root,
            CandidateOptions::default(),
            &mut first_candidates,
        )
        .unwrap();
    assert!(!first_candidates.is_empty());
    let first_info = first_candidates
        .iter()
        .map(|candidate| first.candidate_info(first_root, *candidate).unwrap())
        .collect::<Vec<_>>();

    let mut limited = Vec::new();
    second
        .candidates(
            second_root,
            CandidateOptions {
                max_candidates: Some(1),
                deterministic_order: true,
            },
            &mut limited,
        )
        .unwrap();
    assert_eq!(limited.len(), 1);
    assert_eq!(
        second
            .candidate_info(second_root, limited[0])
            .unwrap()
            .candidate_hash,
        first_info[0].candidate_hash
    );
    second.release(&[], &limited).unwrap();

    let mut second_candidates = Vec::new();
    second
        .candidates(
            second_root,
            CandidateOptions::default(),
            &mut second_candidates,
        )
        .unwrap();
    let second_info = second_candidates
        .iter()
        .map(|candidate| second.candidate_info(second_root, *candidate).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        first_info
            .iter()
            .map(|info| (info.candidate_hash, info.kind, info.display_name.as_str()))
            .collect::<Vec<_>>(),
        second_info
            .iter()
            .map(|info| (info.candidate_hash, info.kind, info.display_name.as_str()))
            .collect::<Vec<_>>()
    );

    let applied = first.apply(first_root, first_candidates[0]).unwrap();
    assert_eq!(applied.before_hash, first.hash(first_root).unwrap());
    assert_eq!(applied.after_hash, first.hash(applied.after).unwrap());
    assert!(applied.changed);
    assert!(matches!(
        first.candidate_info(applied.after, first_candidates[0]),
        Err(EngineError::StaleCandidate {
            expected_graph_hash,
            actual_graph_hash,
            candidate_hash,
        }) if expected_graph_hash == applied.before_hash
            && actual_graph_hash == applied.after_hash
            && candidate_hash == applied.candidate_hash
    ));

    let artifact = first.export_graph(applied.after).unwrap();
    assert_eq!(artifact.graph_hash, applied.after_hash);
    assert_eq!(
        artifact.format,
        GraphArtifactFormat::AdapterSpecific(COMPILER_STATE_ARTIFACT_FORMAT)
    );
    assert!(!artifact.bytes.is_empty());
    let restored = second.restore_graph(&artifact).unwrap();
    assert_eq!(second.hash(restored).unwrap(), applied.after_hash);

    first.release(&[applied.after], &first_candidates).unwrap();
    second.release(&[restored], &second_candidates).unwrap();
}

#[test]
fn foreign_and_released_handles_map_to_boundary_errors() {
    let (mut first, _, _) = engine(EvaluatorBehavior::Pass, false);
    let (mut second, _, _) = engine(EvaluatorBehavior::Pass, false);
    let first_root = first.root();
    let second_root = second.root();
    let mut second_candidates = Vec::new();
    second
        .candidates(
            second_root,
            CandidateOptions {
                max_candidates: Some(1),
                deterministic_order: true,
            },
            &mut second_candidates,
        )
        .unwrap();

    assert!(matches!(
        first.hash(second_root),
        Err(EngineError::UnknownGraph { .. })
    ));
    assert!(matches!(
        first.candidate_info(first_root, second_candidates[0]),
        Err(EngineError::UnknownCandidate { .. })
    ));

    let mut first_candidates = Vec::new();
    first
        .candidates(
            first_root,
            CandidateOptions {
                max_candidates: Some(1),
                deterministic_order: true,
            },
            &mut first_candidates,
        )
        .unwrap();
    let candidate = first_candidates[0];
    let child = first.apply(first_root, candidate).unwrap().after;
    first.release(&[child], &[candidate]).unwrap();
    assert!(matches!(
        first.hash(child),
        Err(EngineError::UnknownGraph { .. })
    ));
    assert!(matches!(
        first.candidate_info(first_root, candidate),
        Err(EngineError::UnknownCandidate { .. })
    ));

    second.release(&[], &second_candidates).unwrap();
}

#[test]
fn successful_measurement_projects_execution_median_to_reward() {
    let (mut engine, payload, evaluator) = engine(EvaluatorBehavior::Pass, false);
    let root = engine.root();
    let options = engine.measure_options(3).unwrap();

    let first = engine.measure(root, options).unwrap().validate().unwrap();
    assert!(first.measured);
    assert!(first.valid);
    assert_eq!(first.scalar_reward, Some(-2.0));
    let latency = first.latency.unwrap();
    assert_eq!(latency.median_ms, 2.0);
    assert_eq!(latency.p95_ms, 4.0);
    assert_eq!(latency.samples_ms, vec![1.0, 2.0, 4.0]);
    assert_eq!(first.failure, None);

    let compile_calls = payload.calls.load(Ordering::Relaxed);
    assert!(compile_calls > 0);
    let second = engine.measure(root, options).unwrap();
    assert!(second.valid);
    assert_eq!(payload.calls.load(Ordering::Relaxed), compile_calls);
    assert_eq!(evaluator.accuracy_calls.load(Ordering::Relaxed), 2);
    assert_eq!(evaluator.performance_calls.load(Ordering::Relaxed), 2);
}

#[test]
fn accuracy_failure_is_a_valid_penalized_measurement() {
    let (mut engine, _, evaluator) = engine(EvaluatorBehavior::AccuracyFail, false);
    let result = engine
        .measure(engine.root(), engine.measure_options(1).unwrap())
        .unwrap()
        .validate()
        .unwrap();

    assert!(result.measured);
    assert!(result.valid);
    assert_eq!(result.scalar_reward, Some(ACCURACY_FAILURE_REWARD));
    assert_eq!(result.latency, None);
    assert_eq!(result.failure, None);
    assert_eq!(evaluator.accuracy_calls.load(Ordering::Relaxed), 1);
    assert_eq!(evaluator.performance_calls.load(Ordering::Relaxed), 0);
}

#[test]
fn compiler_and_evaluator_errors_are_distinct_invalid_measurements() {
    let cases = [
        (EvaluatorBehavior::Pass, true, COMPILE_FAILURE_CODE),
        (
            EvaluatorBehavior::AccuracyError,
            false,
            ACCURACY_EVALUATION_FAILURE_CODE,
        ),
        (
            EvaluatorBehavior::PerformanceError,
            false,
            PERFORMANCE_EVALUATION_FAILURE_CODE,
        ),
    ];

    for (behavior, payload_fails, expected_code) in cases {
        let (mut engine, _, _) = engine(behavior, payload_fails);
        let result = engine
            .measure(engine.root(), engine.measure_options(1).unwrap())
            .unwrap()
            .validate()
            .unwrap();

        assert!(result.measured);
        assert!(!result.valid);
        assert_eq!(result.scalar_reward, None);
        assert_eq!(result.latency, None);
        assert_eq!(result.failure.unwrap().code, expected_code);
    }
}

#[test]
fn unsupported_measure_options_fail_before_measurement() {
    let (mut engine, payload, evaluator) = engine(EvaluatorBehavior::Pass, false);
    let root = engine.root();
    let wrong_config =
        MeasureOptions::new(MeasureConfigHash::from_bytes([8; 32]), 1, None, true).unwrap();
    assert!(matches!(
        engine.measure(root, wrong_config),
        Err(EngineError::Internal { .. })
    ));

    let timeout = MeasureOptions::new(MEASURE_CONFIG_HASH, 1, Some(1), true).unwrap();
    assert!(matches!(
        engine.measure(root, timeout),
        Err(EngineError::Internal { .. })
    ));
    assert_eq!(payload.calls.load(Ordering::Relaxed), 0);
    assert_eq!(evaluator.accuracy_calls.load(Ordering::Relaxed), 0);
}

fn engine(
    evaluator_behavior: EvaluatorBehavior,
    payload_fails: bool,
) -> (CompilerEngine, Arc<FakePayloadCompiler>, Arc<FakeEvaluator>) {
    let payload = Arc::new(FakePayloadCompiler::new(payload_fails));
    let evaluator = Arc::new(FakeEvaluator::new(evaluator_behavior));
    let engine = CompilerEngine::new(CompilerEngineConfig {
        domain: CompilerEngineDomain {
            target: NvidiaTargetRequest::sm(TargetDeviceRequest::BackendDefault, 11, 0).unwrap(),
            bufferization: BufferizationPolicy::default(),
            lowering: LoweringPolicy::default(),
            payload_compiler: payload.clone(),
            evaluator: evaluator.clone(),
            performance_warmup_iterations: 1,
        },
        root: imported_multiply(),
        measure_config_hash: MEASURE_CONFIG_HASH,
        workload: workload(),
    })
    .unwrap();
    (engine, payload, evaluator)
}

fn workload() -> CompilerEvaluationWorkload {
    let id = CaseId::from(String::from("case"));
    let accuracy_suite = AccuracySuite::try_from(vec![AccuracyCase {
        id: id.clone(),
        inputs: TensorBindings::default(),
    }])
    .unwrap();
    let reference_outputs = ReferenceOutputBundle::try_from(vec![ReferenceCase {
        id: id.clone(),
        outputs: TensorBindings::default(),
    }])
    .unwrap();
    let performance_suite = PerformanceSuite::try_from(vec![PerformanceCase {
        id,
        inputs: TensorBindings::default(),
    }])
    .unwrap();

    CompilerEvaluationWorkload {
        execution: EvaluationExecutionTarget::Component(artifact::ComponentId::from(0)),
        bindings: EvaluationBindingSet::default(),
        runtime: EvaluationRuntimePolicy {
            cases: CaseStatePolicy::ResetBeforeEachCase,
            session: EvaluationSessionPolicy::SteadyState,
        },
        accuracy_suite,
        reference_outputs,
        tolerance: TolerancePolicy::Uniform(
            Tolerance::try_from(ToleranceValues {
                absolute: 0.0,
                relative: 0.0,
            })
            .unwrap(),
        ),
        performance_suite,
    }
}

fn imported_multiply() -> graphs_compiler::importer::ImportedProgram {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../graphs/rust/compiler/tests/fixtures/importer/tuple_max");
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("tensors")).unwrap();
    fs::copy(
        fixture.join("tensors/parameters.safetensors"),
        root.path().join("tensors/parameters.safetensors"),
    )
    .unwrap();
    let mut document: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture.join("graphs.json")).unwrap()).unwrap();
    let graph = &mut document["graphs"][0];
    graph["nodes"].as_array_mut().unwrap().truncate(3);
    graph["nodes"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({
            "fx_op": "output",
            "id": "node.3",
            "inputs": ["value.2"]
        }));
    graph["values"].as_array_mut().unwrap().truncate(3);
    graph["signature"]["outputs"] = serde_json::json!([{
        "id": "output.0",
        "role": "user_output",
        "value_id": "value.2"
    }]);
    fs::write(
        root.path().join("graphs.json"),
        serde_json_canonicalizer::to_vec(&document).unwrap(),
    )
    .unwrap();
    import(root.path()).unwrap()
}

struct FakePayloadCompiler {
    calls: AtomicUsize,
    fails: bool,
}

impl FakePayloadCompiler {
    fn new(fails: bool) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            fails,
        }
    }
}

impl NvidiaPayloadCompiler for FakePayloadCompiler {
    fn compile_native_cuda(
        &self,
        request: &NativeCudaCompileRequest,
    ) -> BackendResult<NativeCudaCompileResult> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.fails {
            return Err(internal_backend_error("fake payload compiler failure"));
        }
        let payloads = request
            .expected_payloads
            .iter()
            .map(|expectation| NativeCudaCompiledPayload {
                format: expectation.format,
                bytes: fake_payload(expectation.format, &request.kernel.kernel),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(NativeCudaCompileResult {
            schema: NATIVE_CUDA_COMPILE_SCHEMA.into(),
            request_fingerprint: request.fingerprint(),
            kernel: request.kernel.kernel.clone(),
            compiler: NativeCudaCompilerMetadata {
                implementation: "fake".into(),
                version: "1".into(),
            },
            shared_memory_bytes: u32::try_from(request.kernel.launch.static_shared_memory_bytes)
                .unwrap(),
            payloads,
        })
    }
}

fn fake_payload(format: NativeCudaPayloadFormat, kernel: &CudaKernelSymbol) -> Vec<u8> {
    match format {
        NativeCudaPayloadFormat::NvidiaCubin => {
            let mut bytes = b"\x7fELF".to_vec();
            bytes.extend_from_slice(kernel.as_str().as_bytes());
            bytes
        }
        NativeCudaPayloadFormat::NvidiaPtx => format!(
            ".version 8.0\n.target sm_110\n.visible .entry {}() {{}}\n",
            kernel.as_str()
        )
        .into_bytes(),
    }
}

#[derive(Clone, Copy)]
enum EvaluatorBehavior {
    Pass,
    AccuracyFail,
    AccuracyError,
    PerformanceError,
}

struct FakeEvaluator {
    behavior: EvaluatorBehavior,
    accuracy_calls: AtomicUsize,
    performance_calls: AtomicUsize,
}

impl FakeEvaluator {
    fn new(behavior: EvaluatorBehavior) -> Self {
        Self {
            behavior,
            accuracy_calls: AtomicUsize::new(0),
            performance_calls: AtomicUsize::new(0),
        }
    }
}

#[derive(Debug)]
struct FakeEvaluatorFailure;

impl fmt::Display for FakeEvaluatorFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("fake evaluator failure")
    }
}

impl std::error::Error for FakeEvaluatorFailure {}

impl EvaluationExecutor for FakeEvaluator {
    fn evaluate_accuracy(
        &self,
        request: AccuracyEvaluationRequest,
    ) -> Result<AccuracyReport, EvaluationExecutionError> {
        self.accuracy_calls.fetch_add(1, Ordering::Relaxed);
        if matches!(self.behavior, EvaluatorBehavior::AccuracyError) {
            return Err(EvaluationExecutionError::executor(FakeEvaluatorFailure));
        }
        let environment = measurement_environment_from_target(&request.target);
        Ok(AccuracyReport {
            verdict: if matches!(self.behavior, EvaluatorBehavior::AccuracyFail) {
                AccuracyVerdict::Failed
            } else {
                AccuracyVerdict::Passed
            },
            cases: Box::new([]),
            summary: AccuracySummary {
                cases: 0,
                failed_cases: 0,
                outputs: 0,
                failed_outputs: 0,
                mismatched_elements: 0,
            },
            identity: accuracy_identity(&request, BuildFingerprint::from([1; 32]), &environment),
        })
    }

    fn evaluate_performance(
        &self,
        request: PerformanceEvaluationRequest,
    ) -> Result<PerformanceReport, EvaluationExecutionError> {
        self.performance_calls.fetch_add(1, Ordering::Relaxed);
        if matches!(self.behavior, EvaluatorBehavior::PerformanceError) {
            return Err(EvaluationExecutionError::executor(FakeEvaluatorFailure));
        }
        let environment = measurement_environment_from_target(&request.target);
        let identity =
            performance_identity(&request, BuildFingerprint::from([1; 32]), &environment);
        let execution = LatencyDistribution::try_from(vec![
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(4),
        ])
        .unwrap();
        let overhead = LatencyDistribution::try_from(vec![Duration::from_millis(1)]).unwrap();
        Ok(PerformanceReport {
            setup: overhead.clone(),
            execution,
            observed: overhead,
            throughput: ThroughputMetrics {
                executions: 3,
                elapsed: Duration::from_millis(7),
            },
            memory: RuntimeMemoryMetrics {
                runtime_owned: ByteSize(0),
                external: ByteSize(0),
            },
            cases: Box::new([]),
            environment,
            identity,
        })
    }
}

fn measurement_environment_from_target(
    _target: &graphs_compiler::evaluators::ExecutableEvaluationTarget,
) -> MeasurementEnvironment {
    MeasurementEnvironment {
        device: Device::Nvidia { ordinal: 0 },
        target: RuntimeTarget::Nvidia(NvidiaTarget {
            compute_capability: [11, 0],
        }),
        driver_version: 1,
        runtime_build: BuildFingerprint::from([2; 32]),
    }
}
