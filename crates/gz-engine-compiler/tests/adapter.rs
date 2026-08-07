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
            selection::DEFAULT_OPTIMIZATION_IDS,
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
    COMPILER_STATE_ARTIFACT_FORMAT, CompilerCandidateId, CompilerCandidateMode, CompilerEngine,
    CompilerEngineConfig, CompilerEngineDomain, CompilerEngineTarget, CompilerEvaluationWorkload,
    CompilerFeatureExtractor, CompilerFeatureExtractorConfig, CompilerFeatureLayout,
    CompilerMeasurementPolicy, CompilerStateId, PERFORMANCE_EVALUATION_FAILURE_CODE,
    decode_measurement_evidence,
};
use gz_features::{FeatureError, FeatureExtractor, PositionFeatures, STOP_ACTION_KIND_TOKEN};
use std::{
    collections::BTreeMap,
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
fn hybrid_mode_groups_before_limiting_and_versions_candidate_semantics() {
    let (mut sitewise, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::Sitewise,
        imported_repeated_multiply(),
    );
    let (mut hybrid, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::HybridV1,
        imported_repeated_multiply(),
    );
    let (mut repeated_hybrid, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::HybridV1,
        imported_repeated_multiply(),
    );
    let sitewise_root = sitewise.root();
    let hybrid_root = hybrid.root();
    let repeated_hybrid_root = repeated_hybrid.root();
    assert_eq!(
        sitewise.hash(sitewise_root).unwrap(),
        hybrid.hash(hybrid_root).unwrap()
    );
    assert_eq!(sitewise.engine_id(), hybrid.engine_id());
    assert_ne!(sitewise.engine_version(), hybrid.engine_version());
    assert_ne!(sitewise.action_set_hash(), hybrid.action_set_hash());

    let mut sitewise_candidates = Vec::new();
    sitewise
        .candidates(
            sitewise_root,
            CandidateOptions::default(),
            &mut sitewise_candidates,
        )
        .unwrap();
    let mut hybrid_candidates = Vec::new();
    hybrid
        .candidates(
            hybrid_root,
            CandidateOptions::default(),
            &mut hybrid_candidates,
        )
        .unwrap();
    let mut repeated_hybrid_candidates = Vec::new();
    repeated_hybrid
        .candidates(
            repeated_hybrid_root,
            CandidateOptions::default(),
            &mut repeated_hybrid_candidates,
        )
        .unwrap();
    assert!(hybrid_candidates.len() < sitewise_candidates.len());
    assert_eq!(hybrid_candidates.len(), repeated_hybrid_candidates.len());
    assert_eq!(
        hybrid_candidates
            .iter()
            .map(|candidate| {
                let info = hybrid.candidate_info(hybrid_root, *candidate).unwrap();
                (info.candidate_hash, info.kind, info.display_name)
            })
            .collect::<Vec<_>>(),
        repeated_hybrid_candidates
            .iter()
            .map(|candidate| {
                let info = repeated_hybrid
                    .candidate_info(repeated_hybrid_root, *candidate)
                    .unwrap();
                (info.candidate_hash, info.kind, info.display_name)
            })
            .collect::<Vec<_>>()
    );
    assert!(hybrid_candidates.iter().copied().any(|candidate| {
        hybrid
            .candidate_info(hybrid_root, candidate)
            .unwrap()
            .display_name
            .contains('×')
    }));

    let first_hash = hybrid
        .candidate_info(hybrid_root, hybrid_candidates[0])
        .unwrap()
        .candidate_hash;
    let mut limited = Vec::new();
    hybrid
        .candidates(
            hybrid_root,
            CandidateOptions {
                max_candidates: Some(1),
                deterministic_order: true,
            },
            &mut limited,
        )
        .unwrap();
    assert_eq!(limited.len(), 1);
    assert_eq!(
        hybrid
            .candidate_info(hybrid_root, limited[0])
            .unwrap()
            .candidate_hash,
        first_hash
    );

    sitewise.release(&[], &sitewise_candidates).unwrap();
    hybrid
        .release(&[], &[hybrid_candidates, limited].concat())
        .unwrap();
    repeated_hybrid
        .release(&[], &repeated_hybrid_candidates)
        .unwrap();
}

#[test]
fn end_to_end_mode_represents_every_native_candidate() {
    let (mut sitewise, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::Sitewise,
        imported_repeated_multiply(),
    );
    let (hybrid, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::HybridV1,
        imported_repeated_multiply(),
    );
    let (mut end_to_end, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::EndToEndV1,
        imported_repeated_multiply(),
    );
    let (end_to_end_nvidia, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::EndToEndNvidiaV1,
        imported_repeated_multiply(),
    );
    let (optimization_only_nvidia, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::EndToEndNvidiaV2,
        imported_repeated_multiply(),
    );
    let (optional_only_nvidia, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::EndToEndNvidiaV4,
        imported_repeated_multiply(),
    );
    let (compact_provider_nvidia, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::EndToEndNvidiaV5,
        imported_repeated_multiply(),
    );
    let sitewise_root = sitewise.root();
    let end_to_end_root = end_to_end.root();
    assert_ne!(end_to_end.engine_version(), hybrid.engine_version());
    assert_ne!(end_to_end.action_set_hash(), hybrid.action_set_hash());
    assert_ne!(
        end_to_end.engine_version(),
        end_to_end_nvidia.engine_version()
    );
    assert_ne!(
        end_to_end.action_set_hash(),
        end_to_end_nvidia.action_set_hash()
    );
    assert_ne!(
        end_to_end_nvidia.engine_version(),
        optimization_only_nvidia.engine_version()
    );
    assert_ne!(
        end_to_end_nvidia.action_set_hash(),
        optimization_only_nvidia.action_set_hash()
    );
    assert_ne!(
        optimization_only_nvidia.engine_version(),
        optional_only_nvidia.engine_version()
    );
    assert_ne!(
        optimization_only_nvidia.action_set_hash(),
        optional_only_nvidia.action_set_hash()
    );
    assert_ne!(
        optional_only_nvidia.engine_version(),
        compact_provider_nvidia.engine_version()
    );
    assert_ne!(
        optional_only_nvidia.action_set_hash(),
        compact_provider_nvidia.action_set_hash()
    );

    let mut sitewise_candidates = Vec::new();
    sitewise
        .candidates(
            sitewise_root,
            CandidateOptions::default(),
            &mut sitewise_candidates,
        )
        .unwrap();
    let mut end_to_end_candidates = Vec::new();
    end_to_end
        .candidates(
            end_to_end_root,
            CandidateOptions::default(),
            &mut end_to_end_candidates,
        )
        .unwrap();
    let represented = end_to_end_candidates
        .iter()
        .map(|candidate| {
            let info = end_to_end
                .candidate_info(end_to_end_root, *candidate)
                .unwrap();
            u64::from_le_bytes(info.metadata.bytes.try_into().unwrap())
        })
        .sum::<u64>();
    assert_eq!(represented, sitewise_candidates.len() as u64);

    sitewise.release(&[], &sitewise_candidates).unwrap();
    end_to_end.release(&[], &end_to_end_candidates).unwrap();
}

#[test]
fn target_bounded_hybrid_modes_filter_actions_and_separate_identity() {
    let (mut hybrid, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::HybridV1,
        imported_repeated_multiply(),
    );
    let (mut providers, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::HybridProviderV1,
        imported_repeated_multiply(),
    );
    let (mut nvidia, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::HybridNvidiaV1,
        imported_repeated_multiply(),
    );
    let (mut batched_nvidia, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::HybridNvidiaV2,
        imported_repeated_multiply(),
    );
    let hybrid_root = hybrid.root();
    let provider_root = providers.root();
    let nvidia_root = nvidia.root();
    let batched_nvidia_root = batched_nvidia.root();
    assert_eq!(
        hybrid.hash(hybrid_root).unwrap(),
        providers.hash(provider_root).unwrap()
    );
    assert_eq!(
        hybrid.hash(hybrid_root).unwrap(),
        nvidia.hash(nvidia_root).unwrap()
    );
    assert_eq!(
        hybrid.hash(hybrid_root).unwrap(),
        batched_nvidia.hash(batched_nvidia_root).unwrap()
    );
    assert_ne!(hybrid.engine_version(), providers.engine_version());
    assert_ne!(hybrid.engine_version(), nvidia.engine_version());
    assert_ne!(providers.engine_version(), nvidia.engine_version());
    assert_ne!(nvidia.engine_version(), batched_nvidia.engine_version());
    assert_ne!(hybrid.action_set_hash(), providers.action_set_hash());
    assert_ne!(hybrid.action_set_hash(), nvidia.action_set_hash());
    assert_ne!(providers.action_set_hash(), nvidia.action_set_hash());
    assert_ne!(nvidia.action_set_hash(), batched_nvidia.action_set_hash());

    let mut hybrid_candidates = Vec::new();
    hybrid
        .candidates(
            hybrid_root,
            CandidateOptions::default(),
            &mut hybrid_candidates,
        )
        .unwrap();
    let mut provider_candidates = Vec::new();
    providers
        .candidates(
            provider_root,
            CandidateOptions::default(),
            &mut provider_candidates,
        )
        .unwrap();
    let mut nvidia_candidates = Vec::new();
    nvidia
        .candidates(
            nvidia_root,
            CandidateOptions::default(),
            &mut nvidia_candidates,
        )
        .unwrap();
    let mut batched_nvidia_candidates = Vec::new();
    batched_nvidia
        .candidates(
            batched_nvidia_root,
            CandidateOptions::default(),
            &mut batched_nvidia_candidates,
        )
        .unwrap();

    assert!(!provider_candidates.is_empty());
    assert!(provider_candidates.len() <= hybrid_candidates.len());
    assert!(provider_candidates.len() <= nvidia_candidates.len());
    assert!(nvidia_candidates.len() <= hybrid_candidates.len());
    assert!(provider_candidates.len() <= batched_nvidia_candidates.len());
    assert!(batched_nvidia_candidates.len() <= hybrid_candidates.len());
    assert!(provider_candidates.iter().copied().all(|candidate| {
        providers
            .candidate_info(provider_root, candidate)
            .unwrap()
            .display_name
            .starts_with("nvidia.")
    }));
    assert!(nvidia_candidates.iter().copied().all(|candidate| {
        let name = nvidia
            .candidate_info(nvidia_root, candidate)
            .unwrap()
            .display_name;
        name.starts_with("nvidia.")
            || name.starts_with("graph.fuse_matmul_activation")
            || name.starts_with("algorithm.fuse_decomposed_rope")
    }));
    assert!(batched_nvidia_candidates.iter().copied().all(|candidate| {
        let name = batched_nvidia
            .candidate_info(batched_nvidia_root, candidate)
            .unwrap()
            .display_name;
        name.starts_with("nvidia.")
            || name.starts_with("graph.fuse_matmul_activation")
            || name.starts_with("algorithm.fuse_decomposed_rope")
    }));

    hybrid.release(&[], &hybrid_candidates).unwrap();
    providers.release(&[], &provider_candidates).unwrap();
    nvidia.release(&[], &nvidia_candidates).unwrap();
    batched_nvidia
        .release(&[], &batched_nvidia_candidates)
        .unwrap();
}

#[test]
fn hybrid_batch_application_is_one_search_step_with_multiple_engine_steps() {
    let (mut engine, _, _) = engine_with_mode_and_root(
        EvaluatorBehavior::Pass,
        false,
        CompilerCandidateMode::HybridV1,
        imported_repeated_multiply(),
    );
    let root = engine.root();
    let mut candidates = Vec::new();
    engine
        .candidates(root, CandidateOptions::default(), &mut candidates)
        .unwrap();
    let candidate = candidates
        .iter()
        .copied()
        .find(|candidate| {
            engine
                .candidate_info(root, *candidate)
                .unwrap()
                .display_name
                .contains('×')
        })
        .expect("fixture must expose a hybrid batch");
    let info = engine.candidate_info(root, candidate).unwrap();
    assert_eq!(info.metadata.bytes.len(), std::mem::size_of::<u64>());
    assert!(
        u64::from_le_bytes(info.metadata.bytes.try_into().unwrap()) > 1,
        "hybrid metadata must report the native member count"
    );
    let applied = engine.apply(root, candidate).unwrap();
    assert!(applied.changed);
    assert!(applied.metrics.engine_steps.is_some_and(|steps| steps > 1));
    assert!(matches!(
        engine.candidate_info(applied.after, candidate),
        Err(EngineError::StaleCandidate { .. })
    ));

    let mut extractor =
        CompilerFeatureExtractor::new(&engine, feature_config(candidates.len() + 1)).unwrap();
    let row = extractor
        .extract(&engine, root, &candidates, position())
        .unwrap();
    assert_eq!(row.actions.len(), candidates.len() + 1);

    engine.release(&[applied.after], &candidates).unwrap();
}

#[test]
#[ignore = "requires the external Qwen3-0.6B source artifact"]
fn qwen_end_to_end_nvidia_v3_candidate_closure() {
    let source = std::env::var_os("GRAPHS_QWEN3_PREFILL_SOURCE")
        .expect("GRAPHS_QWEN3_PREFILL_SOURCE must name the source artifact");

    let mut sitewise = qwen_engine(Path::new(&source), CompilerCandidateMode::Sitewise);
    let sitewise_root = sitewise.root();
    let started = std::time::Instant::now();
    let mut sitewise_candidates = Vec::new();
    sitewise
        .candidates(
            sitewise_root,
            CandidateOptions::default(),
            &mut sitewise_candidates,
        )
        .unwrap();
    eprintln!(
        "QWEN_CANDIDATES mode=sitewise count={} elapsed_ms={:.3}",
        sitewise_candidates.len(),
        started.elapsed().as_secs_f64() * 1.0e3
    );

    let mut end_to_end = qwen_engine(Path::new(&source), CompilerCandidateMode::EndToEndNvidiaV3);
    let end_to_end_root = end_to_end.root();
    let started = std::time::Instant::now();
    let mut end_to_end_candidates = Vec::new();
    end_to_end
        .candidates(
            end_to_end_root,
            CandidateOptions::default(),
            &mut end_to_end_candidates,
        )
        .unwrap();
    eprintln!(
        "QWEN_CANDIDATES mode=end-to-end-nvidia-v3 count={} elapsed_ms={:.3}",
        end_to_end_candidates.len(),
        started.elapsed().as_secs_f64() * 1.0e3
    );
    assert!(end_to_end_candidates.iter().any(|candidate| {
        end_to_end
            .candidate_info(end_to_end_root, *candidate)
            .unwrap()
            .display_name
            .starts_with("nvidia.")
    }));
    assert!(
        end_to_end_candidates.len() > 100,
        "end-to-end NVIDIA mode unexpectedly narrowed to {} root actions",
        end_to_end_candidates.len()
    );
    assert!(end_to_end_candidates.len() < sitewise_candidates.len());
    let represented = end_to_end_candidates
        .iter()
        .map(|candidate| {
            let info = end_to_end
                .candidate_info(end_to_end_root, *candidate)
                .unwrap();
            u64::from_le_bytes(info.metadata.bytes.try_into().unwrap())
        })
        .sum::<u64>();
    assert!(represented < sitewise_candidates.len() as u64);
    assert!(represented >= end_to_end_candidates.len() as u64);

    let apply_started = std::time::Instant::now();
    let mut applied_count = 0_usize;
    let mut failures = Vec::new();
    for candidate in end_to_end_candidates.iter().copied() {
        let info = end_to_end
            .candidate_info(end_to_end_root, candidate)
            .unwrap();
        match end_to_end.apply(end_to_end_root, candidate) {
            Ok(applied) => {
                applied_count += 1;
                end_to_end.release(&[applied.after], &[]).unwrap();
            }
            Err(error) => failures.push(format!("{}: {error}", info.display_name)),
        }
    }
    eprintln!(
        "QWEN_APPLY_CLOSURE actions={applied_count} elapsed_ms={:.3}",
        apply_started.elapsed().as_secs_f64() * 1.0e3
    );
    assert!(
        failures.is_empty(),
        "end-to-end NVIDIA root candidates failed to apply:\n{}",
        failures.join("\n")
    );

    sitewise.release(&[], &sitewise_candidates).unwrap();
    end_to_end.release(&[], &end_to_end_candidates).unwrap();
}

#[test]
#[ignore = "requires the external Qwen3-0.6B source artifact"]
fn qwen_candidate_identity_survives_state_artifact_round_trip() {
    const STEPS: usize = 24;

    let source = std::env::var_os("GRAPHS_QWEN3_PREFILL_SOURCE")
        .expect("GRAPHS_QWEN3_PREFILL_SOURCE must name the source artifact");
    let source = Path::new(&source);
    let mut original = qwen_engine(source, CompilerCandidateMode::EndToEndNvidiaV3);
    let mut restored = qwen_engine(source, CompilerCandidateMode::EndToEndNvidiaV3);
    let mut current = original.root();
    let mut current_is_owned = false;
    let mut random = 42_u64;

    for step in 0..STEPS {
        let mut candidates = Vec::new();
        original
            .candidates(current, CandidateOptions::default(), &mut candidates)
            .unwrap();
        if candidates.is_empty() {
            original.release(&[], &candidates).unwrap();
            break;
        }
        random = random
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let selected = candidates[(random as usize) % candidates.len()];
        let previous = current;
        current = original.apply(previous, selected).unwrap().after;
        let previous_graph = [previous];
        original
            .release(
                if current_is_owned {
                    &previous_graph
                } else {
                    &[]
                },
                &candidates,
            )
            .unwrap();
        current_is_owned = true;

        let artifact = original.export_graph(current).unwrap();
        let round_tripped = restored.restore_graph(&artifact).unwrap();
        let mut original_candidates = Vec::new();
        original
            .candidates(
                current,
                CandidateOptions::default(),
                &mut original_candidates,
            )
            .unwrap();
        let mut restored_candidates = Vec::new();
        restored
            .candidates(
                round_tripped,
                CandidateOptions::default(),
                &mut restored_candidates,
            )
            .unwrap();
        let candidate_hashes = |engine: &CompilerEngine,
                                graph: CompilerStateId,
                                candidates: &[CompilerCandidateId]| {
            candidates
                .iter()
                .map(|candidate| {
                    engine
                        .candidate_info(graph, *candidate)
                        .unwrap()
                        .candidate_hash
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            candidate_hashes(&original, current, &original_candidates),
            candidate_hashes(&restored, round_tripped, &restored_candidates),
            "candidate identity changed after artifact round-trip at step {step}"
        );
        original.release(&[], &original_candidates).unwrap();
        restored
            .release(&[round_tripped], &restored_candidates)
            .unwrap();
    }

    if current_is_owned {
        original.release(&[current], &[]).unwrap();
    }
}

#[test]
#[ignore = "requires the external Qwen3-0.6B source artifact"]
fn qwen_default_pipeline_fits_end_to_end_nvidia_v3_horizon() {
    const SEARCH_HORIZON: usize = 24;

    let source = std::env::var_os("GRAPHS_QWEN3_PREFILL_SOURCE")
        .expect("GRAPHS_QWEN3_PREFILL_SOURCE must name the source artifact");
    let mut engine = qwen_engine(Path::new(&source), CompilerCandidateMode::EndToEndNvidiaV3);
    let root = engine.root();
    let mut current = root;
    let mut search_steps = 0_usize;
    let mut native_steps = 0_u64;
    let mut applications = BTreeMap::<&str, usize>::new();

    loop {
        let sweep_start = search_steps;
        for definition in DEFAULT_OPTIMIZATION_IDS {
            loop {
                let mut candidates = Vec::new();
                engine
                    .candidates(current, CandidateOptions::default(), &mut candidates)
                    .unwrap();
                let selected = candidates.iter().copied().find(|candidate| {
                    let display_name = engine
                        .candidate_info(current, *candidate)
                        .unwrap()
                        .display_name;
                    display_name == *definition
                        || display_name
                            .strip_prefix(*definition)
                            .is_some_and(|suffix| suffix.starts_with(" ×"))
                });
                let Some(selected) = selected else {
                    engine.release(&[], &candidates).unwrap();
                    break;
                };

                let before = current;
                let applied = engine.apply(before, selected).unwrap();
                assert!(applied.changed, "{definition} produced an unchanged graph");
                native_steps += applied.metrics.engine_steps.unwrap();
                search_steps += 1;
                *applications.entry(definition).or_default() += 1;
                let previous = [before];
                engine
                    .release(if before == root { &[] } else { &previous }, &candidates)
                    .unwrap();
                current = applied.after;
            }
        }
        if search_steps == sweep_start {
            break;
        }
    }

    eprintln!(
        "QWEN_DEFAULT_PIPELINE search_steps={search_steps} native_steps={native_steps} applications={applications:?}"
    );
    assert!(
        search_steps <= SEARCH_HORIZON,
        "default pipeline needs {search_steps} search actions, exceeding horizon {SEARCH_HORIZON}"
    );
    engine
        .prepare_measurement(current, engine.measure_options(1).unwrap())
        .unwrap();
    if current != root {
        engine.release(&[current], &[]).unwrap();
    }
}

#[test]
#[ignore = "requires the external Qwen3-0.6B source artifact"]
fn qwen_end_to_end_nvidia_v3_random_prefixes_prepare() {
    const ROLLOUTS: usize = 64;
    const STEPS: usize = 24;

    let source = std::env::var_os("GRAPHS_QWEN3_PREFILL_SOURCE")
        .expect("GRAPHS_QWEN3_PREFILL_SOURCE must name the source artifact");
    let mut engine = qwen_engine(Path::new(&source), CompilerCandidateMode::EndToEndNvidiaV3);
    let root = engine.root();
    let mut random = 42_u64;

    for rollout in 0..ROLLOUTS {
        let mut current = root;
        let mut owned = false;
        let mut trace = Vec::with_capacity(STEPS);
        let rollout_steps = rollout % (STEPS + 1);
        for _ in 0..rollout_steps {
            let mut candidates = Vec::new();
            engine
                .candidates(current, CandidateOptions::default(), &mut candidates)
                .unwrap();
            if candidates.is_empty() {
                engine.release(&[], &candidates).unwrap();
                break;
            }
            random = random
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let selected = candidates[(random as usize) % candidates.len()];
            trace.push(
                engine
                    .candidate_info(current, selected)
                    .unwrap()
                    .display_name,
            );
            let previous = current;
            let applied = engine.apply(previous, selected).unwrap();
            let previous_graph = [previous];
            engine
                .release(if owned { &previous_graph } else { &[] }, &candidates)
                .unwrap();
            current = applied.after;
            owned = true;
        }

        let options = engine.measure_options(1).unwrap();
        engine
            .prepare_measurement(current, options)
            .unwrap_or_else(|error| {
                panic!("rollout {rollout} is not NVIDIA-completable: {error}\n{trace:#?}")
            });
        eprintln!(
            "QWEN_RANDOM_TERMINAL rollout={rollout} steps={} status=prepared",
            trace.len()
        );
        if owned {
            engine.release(&[current], &[]).unwrap();
        }
    }
}

#[test]
#[ignore = "requires the external pi0.5 source artifact"]
fn pi05_end_to_end_nvidia_v3_random_prefixes_prepare() {
    const ROLLOUTS: usize = 64;
    const STEPS: usize = 24;

    let source = std::env::var_os("GRAPHS_PI05_SOURCE")
        .expect("GRAPHS_PI05_SOURCE must name the source artifact");
    let mut engine = qwen_engine(Path::new(&source), CompilerCandidateMode::EndToEndNvidiaV3);
    let root = engine.root();
    let mut random = 42_u64;

    for rollout in 0..ROLLOUTS {
        let mut current = root;
        let mut owned = false;
        let mut trace = Vec::with_capacity(STEPS);
        let rollout_steps = rollout % (STEPS + 1);
        for _ in 0..rollout_steps {
            let mut candidates = Vec::new();
            engine
                .candidates(current, CandidateOptions::default(), &mut candidates)
                .unwrap();
            if candidates.is_empty() {
                engine.release(&[], &candidates).unwrap();
                break;
            }
            random = random
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let selected = candidates[(random as usize) % candidates.len()];
            let info = engine.candidate_info(current, selected).unwrap();
            trace.push((info.candidate_hash, info.display_name));
            let previous = current;
            let applied = engine.apply(previous, selected).unwrap();
            let previous_graph = [previous];
            engine
                .release(if owned { &previous_graph } else { &[] }, &candidates)
                .unwrap();
            current = applied.after;
            owned = true;
        }

        let options = engine.measure_options(1).unwrap();
        engine
            .prepare_measurement(current, options)
            .unwrap_or_else(|error| {
                panic!("pi0.5 rollout {rollout} is not NVIDIA-completable: {error}\n{trace:#?}")
            });
        eprintln!(
            "PI05_RANDOM_TERMINAL rollout={rollout} steps={} status=prepared",
            trace.len()
        );
        if owned {
            engine.release(&[current], &[]).unwrap();
        }
    }
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
    assert_eq!(
        decode_measurement_evidence(&first.metadata)
            .unwrap()
            .unwrap()
            .execution_ns,
        vec![1_000_000, 2_000_000, 4_000_000]
    );

    let compile_calls = payload.calls.load(Ordering::Relaxed);
    assert!(compile_calls > 0);
    let second = engine.measure(root, options).unwrap();
    assert!(second.valid);
    assert_eq!(payload.calls.load(Ordering::Relaxed), compile_calls);
    assert_eq!(evaluator.accuracy_calls.load(Ordering::Relaxed), 1);
    assert_eq!(evaluator.performance_calls.load(Ordering::Relaxed), 2);
}

#[test]
fn explicit_preparation_does_not_run_the_evaluator() {
    let (mut engine, payload, evaluator) = engine(EvaluatorBehavior::Pass, false);
    let root = engine.root();
    let options = engine.measure_options(1).unwrap();

    engine.prepare_measurement(root, options).unwrap();
    let compile_calls = payload.calls.load(Ordering::Relaxed);
    assert!(compile_calls > 0);
    assert_eq!(evaluator.accuracy_calls.load(Ordering::Relaxed), 0);
    assert_eq!(evaluator.performance_calls.load(Ordering::Relaxed), 0);

    let measured = engine
        .measure_with_warmup(root, options, 0)
        .unwrap()
        .validate()
        .unwrap();
    assert!(measured.valid);
    assert_eq!(payload.calls.load(Ordering::Relaxed), compile_calls);
    assert_eq!(evaluator.accuracy_calls.load(Ordering::Relaxed), 1);
    assert_eq!(evaluator.performance_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn preparation_errors_preserve_the_native_source_chain() {
    let (mut engine, _, _) = engine(EvaluatorBehavior::Pass, true);
    let error = engine
        .prepare_measurement(engine.root(), engine.measure_options(1).unwrap())
        .unwrap_err();

    assert!(matches!(
        error,
        EngineError::Internal { message, .. }
            if message.as_str().contains("fake payload compiler failure")
    ));
}

#[test]
fn performance_only_policy_skips_accuracy_evaluation() {
    let (mut engine, _, evaluator) =
        engine_with_measurement_policy(CompilerMeasurementPolicy::PerformanceOnly);
    let result = engine
        .measure(engine.root(), engine.measure_options(1).unwrap())
        .unwrap()
        .validate()
        .unwrap();

    assert!(result.valid);
    assert_eq!(result.scalar_reward, Some(-2.0));
    assert_eq!(evaluator.accuracy_calls.load(Ordering::Relaxed), 0);
    assert_eq!(evaluator.performance_calls.load(Ordering::Relaxed), 1);
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
    assert_eq!(decode_measurement_evidence(&result.metadata).unwrap(), None);
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

#[test]
fn compiler_features_are_deterministic_and_bind_complete_actions() {
    let (mut first, _, _) = engine(EvaluatorBehavior::Pass, false);
    let (mut second, _, _) = engine(EvaluatorBehavior::Pass, false);
    let first_root = first.root();
    let second_root = second.root();
    let mut first_candidates = Vec::new();
    let mut second_candidates = Vec::new();
    first
        .candidates(
            first_root,
            CandidateOptions::default(),
            &mut first_candidates,
        )
        .unwrap();
    second
        .candidates(
            second_root,
            CandidateOptions::default(),
            &mut second_candidates,
        )
        .unwrap();
    assert_eq!(first_candidates.len(), second_candidates.len());

    let config = feature_config(first_candidates.len() + 1);
    let mut first_extractor = CompilerFeatureExtractor::new(&first, config).unwrap();
    let mut second_extractor = CompilerFeatureExtractor::new(&second, config).unwrap();
    let first_row = first_extractor
        .extract(&first, first_root, &first_candidates, position())
        .unwrap();
    let second_row = second_extractor
        .extract(&second, second_root, &second_candidates, position())
        .unwrap();

    assert_eq!(first_extractor.schema(), second_extractor.schema());
    assert_eq!(first_row, second_row);
    assert_eq!(first_row.actions.len(), first_candidates.len() + 1);
    assert_eq!(
        first_row.actions.last().unwrap().kind_token,
        STOP_ACTION_KIND_TOKEN
    );
    assert!(
        first_row.actions[..first_candidates.len()]
            .iter()
            .all(|action| !action.subjects.is_empty())
    );
    assert_eq!(
        first_row.node_attrs.len(),
        first_row.node_count as usize
            * usize::from(first_extractor.schema().config().node_attr_dim)
    );

    first.release(&[], &first_candidates).unwrap();
    second.release(&[], &second_candidates).unwrap();
}

#[test]
fn compact_compiler_features_remove_port_nodes_and_preserve_actions() {
    let (mut engine, _, _) = engine(EvaluatorBehavior::Pass, false);
    let root = engine.root();
    let mut candidates = Vec::new();
    engine
        .candidates(root, CandidateOptions::default(), &mut candidates)
        .unwrap();
    let mut expanded =
        CompilerFeatureExtractor::new(&engine, feature_config(candidates.len() + 1)).unwrap();
    let mut compact = CompilerFeatureExtractor::new(
        &engine,
        CompilerFeatureExtractorConfig {
            layout: CompilerFeatureLayout::CompactV1,
            ..feature_config(candidates.len() + 1)
        },
    )
    .unwrap();

    let expanded = expanded
        .extract(&engine, root, &candidates, position())
        .unwrap();
    let compact = compact
        .extract(&engine, root, &candidates, position())
        .unwrap();
    assert!(compact.node_count < expanded.node_count);
    assert_eq!(compact.actions, expanded.actions);
    assert!(compact.edges.iter().all(|edge| edge.edge_type < 100));
    engine.release(&[], &candidates).unwrap();
}

#[test]
fn compiler_features_follow_applied_states() {
    let (mut engine, _, _) = engine(EvaluatorBehavior::Pass, false);
    let root = engine.root();
    let mut root_candidates = Vec::new();
    engine
        .candidates(root, CandidateOptions::default(), &mut root_candidates)
        .unwrap();
    let child = engine.apply(root, root_candidates[0]).unwrap().after;
    let mut child_candidates = Vec::new();
    engine
        .candidates(child, CandidateOptions::default(), &mut child_candidates)
        .unwrap();

    let max_actions = root_candidates.len().max(child_candidates.len()) + 1;
    let mut extractor =
        CompilerFeatureExtractor::new(&engine, feature_config(max_actions)).unwrap();
    let root_row = extractor
        .extract(&engine, root, &root_candidates, position())
        .unwrap();
    let child_row = extractor
        .extract(&engine, child, &child_candidates, position())
        .unwrap();
    assert_ne!(
        (root_row.node_tokens, root_row.node_attrs, root_row.edges),
        (child_row.node_tokens, child_row.node_attrs, child_row.edges)
    );

    engine
        .release(&[child], &[root_candidates, child_candidates].concat())
        .unwrap();
}

#[test]
fn compiler_feature_bounds_fail_before_row_construction() {
    let (engine, _, _) = engine(EvaluatorBehavior::Pass, false);
    let mut extractor = CompilerFeatureExtractor::new(
        &engine,
        CompilerFeatureExtractorConfig {
            layout: CompilerFeatureLayout::ExpandedPorts,
            max_nodes: 1,
            max_edges: 1,
            max_actions: 1,
            max_subjects: 1,
        },
    )
    .unwrap();

    assert!(matches!(
        extractor.extract(&engine, engine.root(), &[], position()),
        Err(FeatureError::NodeOverflow { max: 1, .. })
    ));
}

fn qwen_engine(source: &Path, candidate_mode: CompilerCandidateMode) -> CompilerEngine {
    CompilerEngine::new(CompilerEngineConfig {
        domain: CompilerEngineDomain {
            target: CompilerEngineTarget::Nvidia {
                request: NvidiaTargetRequest::sm(TargetDeviceRequest::BackendDefault, 12, 0)
                    .unwrap(),
                payload_compiler: Arc::new(FakePayloadCompiler::new(false)),
            },
            bufferization: BufferizationPolicy::default(),
            lowering: LoweringPolicy::default(),
            evaluator: Arc::new(FakeEvaluator::new(EvaluatorBehavior::Pass)),
            performance_warmup_iterations: 1,
        },
        root: import(source).unwrap(),
        candidate_mode,
        measure_config_hash: MEASURE_CONFIG_HASH,
        measurement_policy: CompilerMeasurementPolicy::VerifyAccuracy,
        workload: workload(),
    })
    .unwrap()
}

fn feature_config(max_actions: usize) -> CompilerFeatureExtractorConfig {
    CompilerFeatureExtractorConfig {
        layout: CompilerFeatureLayout::ExpandedPorts,
        max_nodes: u32::from(u16::MAX),
        max_edges: 1_000_000,
        max_actions: u32::try_from(max_actions).unwrap(),
        max_subjects: u32::from(u8::MAX),
    }
}

fn position() -> PositionFeatures {
    PositionFeatures {
        root_step: 0,
        leaf_depth: 0,
        budget_fraction: 1.0,
        budget_step: 0.0,
        opponent_reward: 0.0,
        opponent_present: false,
    }
}

fn engine(
    evaluator_behavior: EvaluatorBehavior,
    payload_fails: bool,
) -> (CompilerEngine, Arc<FakePayloadCompiler>, Arc<FakeEvaluator>) {
    engine_with_mode(
        evaluator_behavior,
        payload_fails,
        CompilerCandidateMode::Sitewise,
    )
}

fn engine_with_mode(
    evaluator_behavior: EvaluatorBehavior,
    payload_fails: bool,
    candidate_mode: CompilerCandidateMode,
) -> (CompilerEngine, Arc<FakePayloadCompiler>, Arc<FakeEvaluator>) {
    engine_with_mode_and_root(
        evaluator_behavior,
        payload_fails,
        candidate_mode,
        imported_multiply(),
    )
}

fn engine_with_mode_and_root(
    evaluator_behavior: EvaluatorBehavior,
    payload_fails: bool,
    candidate_mode: CompilerCandidateMode,
    root: graphs_compiler::importer::ImportedProgram,
) -> (CompilerEngine, Arc<FakePayloadCompiler>, Arc<FakeEvaluator>) {
    engine_with_mode_root_and_measurement_policy(
        evaluator_behavior,
        payload_fails,
        candidate_mode,
        root,
        CompilerMeasurementPolicy::VerifyAccuracy,
    )
}

fn engine_with_measurement_policy(
    measurement_policy: CompilerMeasurementPolicy,
) -> (CompilerEngine, Arc<FakePayloadCompiler>, Arc<FakeEvaluator>) {
    engine_with_mode_root_and_measurement_policy(
        EvaluatorBehavior::AccuracyFail,
        false,
        CompilerCandidateMode::Sitewise,
        imported_multiply(),
        measurement_policy,
    )
}

fn engine_with_mode_root_and_measurement_policy(
    evaluator_behavior: EvaluatorBehavior,
    payload_fails: bool,
    candidate_mode: CompilerCandidateMode,
    root: graphs_compiler::importer::ImportedProgram,
    measurement_policy: CompilerMeasurementPolicy,
) -> (CompilerEngine, Arc<FakePayloadCompiler>, Arc<FakeEvaluator>) {
    let payload = Arc::new(FakePayloadCompiler::new(payload_fails));
    let evaluator = Arc::new(FakeEvaluator::new(evaluator_behavior));
    let engine = CompilerEngine::new(CompilerEngineConfig {
        domain: CompilerEngineDomain {
            target: CompilerEngineTarget::Nvidia {
                request: NvidiaTargetRequest::sm(TargetDeviceRequest::BackendDefault, 11, 0)
                    .unwrap(),
                payload_compiler: payload.clone(),
            },
            bufferization: BufferizationPolicy::default(),
            lowering: LoweringPolicy::default(),
            evaluator: evaluator.clone(),
            performance_warmup_iterations: 1,
        },
        root,
        candidate_mode,
        measure_config_hash: MEASURE_CONFIG_HASH,
        measurement_policy,
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

fn imported_repeated_multiply() -> graphs_compiler::importer::ImportedProgram {
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
            "args": [
                {"kind": "value", "value_id": "value.1"},
                {"kind": "value", "value_id": "value.0"}
            ],
            "fx_op": "call_function",
            "id": "node.3",
            "kwargs": [],
            "outputs": ["value.3"],
            "target": {
                "kind": "aten_overload",
                "namespace": "aten",
                "operator": "mul",
                "overload": "Tensor"
            }
        }));
    graph["nodes"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({
            "fx_op": "output",
            "id": "node.4",
            "inputs": ["value.2", "value.3"]
        }));
    graph["values"].as_array_mut().unwrap().truncate(3);
    let mut second_output = graph["values"][2].clone();
    second_output["id"] = serde_json::Value::String("value.3".to_owned());
    graph["values"].as_array_mut().unwrap().push(second_output);
    graph["signature"]["outputs"] = serde_json::json!([
        {
            "id": "output.0",
            "role": "user_output",
            "value_id": "value.2"
        },
        {
            "id": "output.1",
            "role": "user_output",
            "value_id": "value.3"
        }
    ]);
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
                content: artifact::ContentBlob::from(fake_payload(
                    expectation.format,
                    &request.kernel.kernel,
                )),
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
        let overhead = LatencyDistribution::try_from(vec![
            Duration::from_millis(2),
            Duration::from_millis(3),
            Duration::from_millis(5),
        ])
        .unwrap();
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
