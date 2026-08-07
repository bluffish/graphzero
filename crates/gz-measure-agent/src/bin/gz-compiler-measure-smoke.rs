#![forbid(unsafe_code)]

use gz_engine::{
    CandidateHash, CandidateOptions, GraphEngine, GraphHash, MeasureSummary, PortableCandidateRef,
    PortableGraphId, PortableSearchActionRef, ReplayGraphContext, SearchStepRef,
};
use gz_engine_compiler::{
    CompilerEngine, CompilerFeatureExtractor, CompilerFeatureExtractorConfig,
    CompilerFeatureLayout, CompilerStateId,
};
use gz_features::{FeatureExtractor, PositionFeatures, encode_feature_row};
use gz_measure_agent::{CompilerDeployment, compiler_measure_submission_pair};
use gz_measure_service::{
    CommittedMeasurement, Coordinator, CoordinatorConfig, CoordinatorHandle, CoordinatorServer,
    CoordinatorTlsConfig, DeviceId, Enrollment, MeasureSubmission, certificate_fingerprint,
};
use gz_replay::{
    ReplayContract, ReplayDataMode, ReplayEpisodeRecord, ReplayOutcome, ReplayRow, ReplayRowKind,
    ReplayStore,
};
use gz_search::trajectory_distill_config_hash;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1).collect())?;
    let deployment = CompilerDeployment::load(&args.deployment)?;
    let client_certificate = pem::parse(read(&args.client_cert)?)
        .map_err(|error| format!("failed to parse {}: {error}", args.client_cert.display()))?;
    let coordinator = Coordinator::new(CoordinatorConfig {
        enrollment: Enrollment::one_mutual_tls(
            certificate_fingerprint(client_certificate.contents()),
            args.device_id,
        ),
        ..CoordinatorConfig::default()
    })
    .map_err(|error| error.to_string())?;
    let handle = coordinator.handle();
    let listener = tokio::net::TcpListener::bind(args.listen)
        .await
        .map_err(|error| error.to_string())?;
    let listen = listener.local_addr().map_err(|error| error.to_string())?;
    let server = CoordinatorServer::mutual_tls(
        coordinator,
        CoordinatorTlsConfig {
            certificate_pem: read(&args.server_cert)?,
            private_key_pem: read(&args.server_key)?,
            client_ca_pem: read(&args.client_ca)?,
        },
    );
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server_task = tokio::spawn(async move {
        server
            .serve_with_shutdown(listener, async move {
                let _ = shutdown_rx.await;
            })
            .await
    });
    println!(
        "coordinator_listening={listen} profile_hash={} measure_config_hash={}",
        deployment.device_profile_hash(),
        deployment.measure_config_hash()
    );

    let mut engine = deployment.build_search_engine()?;
    let (subjects, retained_graphs) = if args.root_only {
        let root = engine.root();
        ([root, root], Vec::new())
    } else if let Some(sequence) = &args.candidate_sequence {
        let subjects = candidate_sequence_subjects(&mut engine, sequence)?;
        (subjects, vec![subjects[0]])
    } else if let Some(prefix) = &args.candidate_prefix {
        let sequence = vec![prefix.clone(); args.candidate_steps];
        let subjects = candidate_sequence_subjects(&mut engine, &sequence)?;
        (subjects, vec![subjects[0]])
    } else if let Some(action_hashes) = &args.action_hashes {
        let first = replay_subject(&mut engine, action_hashes)?;
        let graph_hash = engine.hash(first).map_err(|error| error.to_string())?;
        if let Some(expected) = args.expected_graph_hash
            && graph_hash != expected
        {
            return Err(format!(
                "replayed graph hash mismatch: expected {expected}, got {graph_hash}"
            ));
        }
        println!(
            "replayed_actions={} graph_hash={graph_hash}",
            action_hashes.len()
        );
        if let Some(comparison_hashes) = &args.comparison_action_hashes {
            let second = replay_subject(&mut engine, comparison_hashes)?;
            let comparison_hash = engine.hash(second).map_err(|error| error.to_string())?;
            if let Some(expected) = args.expected_comparison_graph_hash
                && comparison_hash != expected
            {
                return Err(format!(
                    "replayed comparison graph hash mismatch: expected {expected}, got {comparison_hash}"
                ));
            }
            println!(
                "replayed_comparison_actions={} graph_hash={comparison_hash}",
                comparison_hashes.len()
            );
            ([first, second], vec![first, second])
        } else {
            ([first, first], vec![first])
        }
    } else {
        let subjects = default_subjects(&mut engine)?;
        (subjects, subjects.to_vec())
    };
    let options = deployment.measure_options(args.samples)?;
    let warmup_requests = measurement_requests(
        &engine,
        subjects,
        options,
        deployment.device_profile_hash(),
        args.warmup_jobs,
    )?;
    let warmup = measure_batch(&handle, warmup_requests, args.timeout).await?;
    validate_measurements(warmup, subjects, options)?;
    println!(
        "warmup_jobs_completed={} pipeline_slots={}",
        args.warmup_jobs,
        deployment.measurement_pipeline_slots()
    );

    let requests = measurement_requests(
        &engine,
        subjects,
        options,
        deployment.device_profile_hash(),
        args.jobs,
    )?;
    let started = Instant::now();
    let committed = measure_batch(&handle, requests, args.timeout).await?;
    let elapsed = started.elapsed();
    let report = validate_measurements(committed, subjects, options)?;
    let jobs_per_second = args.jobs as f64 / elapsed.as_secs_f64();
    let subjects_per_second = (args.jobs * 2) as f64 / elapsed.as_secs_f64();
    println!(
        "benchmark_jobs={} benchmark_subjects={} pipeline_slots={} elapsed_s={:.3} jobs_per_s={:.6} subjects_per_s={:.6} mean_job_compile_s={:.3} first_job_id={} first_schedule={} subject_0_hash={} subject_0_reward={} subject_1_hash={} subject_1_reward={} reward_pairs={:?} sample_pairs_ms={:?}",
        args.jobs,
        args.jobs * 2,
        deployment.measurement_pipeline_slots(),
        elapsed.as_secs_f64(),
        jobs_per_second,
        subjects_per_second,
        report.mean_job_compile_seconds,
        report.first_job_id,
        report.first_schedule,
        report.subject_hashes[0],
        report.subject_rewards[0],
        report.subject_hashes[1],
        report.subject_rewards[1],
        report.reward_pairs,
        report.sample_pairs_ms,
    );

    if let Some(replay_dir) = &args.teacher_replay_dir {
        let action_hashes = args
            .action_hashes
            .as_deref()
            .expect("validated teacher replay has action hashes");
        write_teacher_replay(
            &mut engine,
            action_hashes,
            report.first_subject_measure.clone(),
            options,
            replay_dir,
            &args,
        )?;
    }

    engine
        .release(&retained_graphs, &[])
        .map_err(|error| error.to_string())?;
    handle.drain_agents();
    let _ = shutdown_tx.send(());
    server_task
        .await
        .map_err(|error| error.to_string())?
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn candidate_sequence_subjects(
    engine: &mut CompilerEngine,
    prefixes: &[String],
) -> Result<[CompilerStateId; 2], String> {
    let root = engine.root();
    let mut graph = root;
    let mut graph_is_owned = false;
    let mut selected_hashes = Vec::with_capacity(prefixes.len());
    for (step, prefix) in prefixes.iter().enumerate() {
        let mut candidates = Vec::new();
        engine
            .candidates(graph, CandidateOptions::default(), &mut candidates)
            .map_err(|error| error.to_string())?;
        let matches = candidates
            .iter()
            .copied()
            .filter_map(|candidate| {
                engine
                    .candidate_info(graph, candidate)
                    .ok()
                    .filter(|info| info.display_name.starts_with(prefix))
                    .map(|info| (candidate, info))
            })
            .collect::<Vec<_>>();
        let Some((selected, info)) = matches.first() else {
            let mut available = candidates
                .iter()
                .copied()
                .filter_map(|candidate| {
                    engine
                        .candidate_info(graph, candidate)
                        .ok()
                        .map(|info| info.display_name)
                })
                .collect::<Vec<_>>();
            available.sort();
            available.dedup();
            engine
                .release(graph_is_owned.then_some(graph).as_slice(), &candidates)
                .map_err(|error| error.to_string())?;
            return Err(format!(
                "compiler graph exposed no candidate matching {prefix:?} at step {step}; available definitions: {}",
                available.join(", ")
            ));
        };
        let selected = *selected;
        let candidate_hash = info.candidate_hash;
        let display_name = info.display_name.clone();
        let matching_candidates = matches.len();
        let applied = engine
            .apply(graph, selected)
            .map_err(|error| error.to_string())?;
        engine
            .release(graph_is_owned.then_some(graph).as_slice(), &candidates)
            .map_err(|error| error.to_string())?;
        graph = applied.after;
        graph_is_owned = true;
        selected_hashes.push(candidate_hash);
        println!(
            "candidate_step={step} selected_candidate={display_name:?} candidate_hash={candidate_hash} matching_candidates={matching_candidates} graph_hash={}",
            applied.after_hash
        );
    }
    println!(
        "selected_candidate_steps={} candidate_hashes={}",
        selected_hashes.len(),
        selected_hashes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok([graph, root])
}

fn default_subjects(engine: &mut CompilerEngine) -> Result<[CompilerStateId; 2], String> {
    let root = engine.root();
    let mut root_candidates = Vec::new();
    engine
        .candidates(root, CandidateOptions::default(), &mut root_candidates)
        .map_err(|error| error.to_string())?;
    let attention = root_candidates
        .iter()
        .copied()
        .find(|candidate| {
            engine.candidate_info(root, *candidate).is_ok_and(|info| {
                info.display_name
                    .starts_with("nvidia.native_cuda.sdpa.bf16")
            })
        })
        .ok_or_else(|| "compiler root exposed no native CUDA SDPA candidate".to_owned())?;
    let baseline = engine
        .apply(root, attention)
        .map_err(|error| error.to_string())?
        .after;
    engine
        .release(&[], &root_candidates)
        .map_err(|error| error.to_string())?;
    let mut baseline_candidates = Vec::new();
    engine
        .candidates(
            baseline,
            CandidateOptions::default(),
            &mut baseline_candidates,
        )
        .map_err(|error| error.to_string())?;
    let alternate_backend = baseline_candidates
        .iter()
        .copied()
        .find(|candidate| {
            engine
                .candidate_info(baseline, *candidate)
                .is_ok_and(|info| {
                    info.display_name
                        .starts_with("nvidia.native_cuda.pointwise.elementwise")
                })
        })
        .ok_or_else(|| "compiler baseline exposed no native CUDA pointwise candidate".to_owned())?;
    let alternate = engine
        .apply(baseline, alternate_backend)
        .map_err(|error| error.to_string())?
        .after;
    engine
        .release(&[], &baseline_candidates)
        .map_err(|error| error.to_string())?;
    Ok([baseline, alternate])
}

fn replay_subject(
    engine: &mut CompilerEngine,
    action_hashes: &[CandidateHash],
) -> Result<CompilerStateId, String> {
    let mut graph = engine.root();
    let mut graph_is_owned = false;
    for (step, expected_hash) in action_hashes.iter().copied().enumerate() {
        let mut candidates = Vec::new();
        engine
            .candidates(
                graph,
                CandidateOptions {
                    max_candidates: None,
                    deterministic_order: true,
                },
                &mut candidates,
            )
            .map_err(|error| error.to_string())?;
        let selected = candidates.iter().copied().find(|candidate| {
            engine
                .candidate_info(graph, *candidate)
                .is_ok_and(|info| info.candidate_hash == expected_hash)
        });
        let Some(selected) = selected else {
            engine
                .release(graph_is_owned.then_some(graph).as_slice(), &candidates)
                .map_err(|error| error.to_string())?;
            return Err(format!(
                "trajectory candidate {expected_hash} is unavailable at step {step}"
            ));
        };
        let display_name = engine
            .candidate_info(graph, selected)
            .map_err(|error| error.to_string())?
            .display_name;
        let applied = engine
            .apply(graph, selected)
            .map_err(|error| error.to_string())?;
        engine
            .release(graph_is_owned.then_some(graph).as_slice(), &candidates)
            .map_err(|error| error.to_string())?;
        graph = applied.after;
        graph_is_owned = true;
        println!(
            "replay_step={step} candidate={display_name:?} candidate_hash={expected_hash} graph_hash={}",
            applied.after_hash
        );
    }
    Ok(graph)
}

fn write_teacher_replay(
    engine: &mut CompilerEngine,
    action_hashes: &[CandidateHash],
    final_measure: MeasureSummary,
    measure_options: gz_engine::MeasureOptions,
    replay_dir: &Path,
    args: &Args,
) -> Result<(), String> {
    if !final_measure.measured || !final_measure.valid {
        return Err("teacher trajectory requires a valid measured terminal graph".to_owned());
    }
    let reward = final_measure
        .scalar_reward
        .ok_or_else(|| "teacher terminal measurement omitted scalar reward".to_owned())?;
    let max_actions = u32::try_from(args.teacher_max_candidates + 1)
        .map_err(|_| "--teacher-max-candidates exceeds u32 action width".to_owned())?;
    let mut extractor = CompilerFeatureExtractor::new(
        engine,
        CompilerFeatureExtractorConfig {
            layout: args.teacher_feature_layout,
            max_nodes: args.teacher_max_nodes,
            max_edges: args.teacher_max_edges,
            max_actions,
            max_subjects: args.teacher_max_subjects,
        },
    )
    .map_err(|error| error.to_string())?;
    let store = ReplayStore::open(replay_dir).map_err(|error| error.to_string())?;
    if store.counters().produced_rows != 0 || store.episode_counters().0 != 0 {
        return Err("teacher replay directory is not empty".to_owned());
    }
    store
        .ensure_contract(&ReplayContract::featurized(
            ReplayDataMode::Standard,
            extractor.schema().config().clone(),
            gz_engine::EngineIdentity::from_engine(engine),
        ))
        .map_err(|error| error.to_string())?;

    let candidate_options = CandidateOptions {
        max_candidates: Some(args.teacher_max_candidates + 1),
        deterministic_order: true,
    };
    let search_config_hash = trajectory_distill_config_hash(
        args.teacher_max_steps,
        true,
        CandidateOptions {
            max_candidates: Some(args.teacher_max_candidates),
            deterministic_order: true,
        },
        measure_options,
        action_hashes,
    );
    let root_context = graph_context(
        engine,
        engine.hash(engine.root()).map_err(|e| e.to_string())?,
    );
    let final_context = graph_context(engine, final_measure.graph_hash);
    let mut graph = engine.root();
    let mut graph_is_owned = false;
    let mut action_history = Vec::with_capacity(action_hashes.len());
    let mut steps = Vec::with_capacity(action_hashes.len());
    let mut rows = Vec::with_capacity(action_hashes.len());

    for (step_index, expected_hash) in action_hashes.iter().copied().enumerate() {
        let state_hash = engine.hash(graph).map_err(|error| error.to_string())?;
        let state_context = graph_context(engine, state_hash);
        let mut candidates = Vec::new();
        engine
            .candidates(graph, candidate_options, &mut candidates)
            .map_err(|error| error.to_string())?;
        if candidates.len() > args.teacher_max_candidates {
            engine
                .release(graph_is_owned.then_some(graph).as_slice(), &candidates)
                .map_err(|error| error.to_string())?;
            return Err(format!(
                "teacher state {step_index} exposes {} candidates, exceeding {}",
                candidates.len(),
                args.teacher_max_candidates
            ));
        }
        let mut legal_actions = Vec::with_capacity(candidates.len() + 1);
        let mut selected = None;
        for (index, candidate) in candidates.iter().copied().enumerate() {
            let info = engine
                .candidate_info(graph, candidate)
                .map_err(|error| error.to_string())?;
            legal_actions.push(PortableSearchActionRef::candidate(
                PortableCandidateRef::new(state_context, info.candidate_hash),
            ));
            if info.candidate_hash == expected_hash {
                selected = Some((index, candidate));
            }
        }
        legal_actions.push(PortableSearchActionRef::stop(state_context));
        let Some((selected_index, selected_candidate)) = selected else {
            engine
                .release(graph_is_owned.then_some(graph).as_slice(), &candidates)
                .map_err(|error| error.to_string())?;
            return Err(format!(
                "teacher candidate {expected_hash} is unavailable at step {step_index}"
            ));
        };
        let selected_action = legal_actions[selected_index];
        let remaining = args.teacher_max_steps.saturating_sub(step_index);
        let position = PositionFeatures {
            root_step: u32::try_from(step_index)
                .map_err(|_| "teacher step index exceeds u32".to_owned())?,
            leaf_depth: 0,
            budget_fraction: remaining as f32 / args.teacher_max_steps as f32,
            budget_step: 1.0 / args.teacher_max_steps as f32,
            opponent_reward: 0.0,
            opponent_present: false,
        };
        let feature_row = extractor
            .extract(engine, graph, &candidates, position)
            .map_err(|error| error.to_string())?;
        let mut feature_bytes = Vec::new();
        encode_feature_row(&feature_row, extractor.schema(), &mut feature_bytes)
            .map_err(|error| error.to_string())?;
        let applied = engine
            .apply(graph, selected_candidate)
            .map_err(|error| error.to_string())?;
        let after_context = graph_context(engine, applied.after_hash);
        let mut policy_target = vec![0.0; legal_actions.len()];
        policy_target[selected_index] = 1.0;
        let step_ref = SearchStepRef::new(state_context, selected_action, after_context)
            .map_err(|error| error.to_string())?;
        steps.push(step_ref);
        rows.push(ReplayRow {
            kind: ReplayRowKind::Decision,
            step_index: u32::try_from(step_index)
                .map_err(|_| "teacher step index exceeds u32".to_owned())?,
            root: root_context,
            state: state_context,
            action_history: action_history.clone(),
            legal_actions,
            policy_target,
            selected_action: Some(selected_action),
            value_target: None,
            horizon_value_targets: None,
            reward_target: Some(reward),
            final_measure: final_measure.clone(),
            model_version: None,
            search_config_hash,
            feature_row: Some(feature_bytes),
        });
        action_history.push(selected_action);
        engine
            .release(graph_is_owned.then_some(graph).as_slice(), &candidates)
            .map_err(|error| error.to_string())?;
        graph = applied.after;
        graph_is_owned = true;
    }

    let replayed_final_hash = engine.hash(graph).map_err(|error| error.to_string())?;
    if replayed_final_hash != final_measure.graph_hash {
        engine
            .release(graph_is_owned.then_some(graph).as_slice(), &[])
            .map_err(|error| error.to_string())?;
        return Err(format!(
            "teacher terminal hash mismatch: measured {}, replayed {replayed_final_hash}",
            final_measure.graph_hash
        ));
    }
    engine
        .release(graph_is_owned.then_some(graph).as_slice(), &[])
        .map_err(|error| error.to_string())?;
    let record = ReplayEpisodeRecord {
        root: root_context,
        final_graph: final_context,
        steps,
        final_measure,
        outcome: ReplayOutcome::new(None, reward, false),
        search_config_hash,
        row_count: u32::try_from(rows.len())
            .map_err(|_| "teacher row count exceeds u32".to_owned())?,
    };
    store
        .append_episode(&record, &rows)
        .map_err(|error| error.to_string())?;
    println!(
        "teacher_replay={} rows={} terminal_reward={reward}",
        replay_dir.display(),
        rows.len()
    );
    Ok(())
}

fn graph_context(engine: &CompilerEngine, graph_hash: GraphHash) -> ReplayGraphContext {
    ReplayGraphContext::new(
        PortableGraphId::new(graph_hash, engine.engine_id(), engine.engine_version()),
        engine.action_set_hash(),
    )
}

fn measurement_requests(
    engine: &gz_engine_compiler::CompilerEngine,
    graphs: [<gz_engine_compiler::CompilerEngine as GraphEngine>::Graph; 2],
    options: gz_engine::MeasureOptions,
    profile_hash: gz_measure_service::DeviceProfileHash,
    jobs: usize,
) -> Result<Vec<MeasureSubmission>, String> {
    (0..jobs)
        .map(|_| {
            compiler_measure_submission_pair(engine, graphs, options, profile_hash)
                .map_err(|error| error.to_string())
        })
        .collect()
}

async fn measure_batch(
    handle: &CoordinatorHandle,
    requests: Vec<MeasureSubmission>,
    timeout: Duration,
) -> Result<Vec<CommittedMeasurement>, String> {
    let mut tasks = tokio::task::JoinSet::new();
    for (index, request) in requests.into_iter().enumerate() {
        let handle = handle.clone();
        tasks.spawn(async move {
            let committed = tokio::time::timeout(timeout, handle.measure(request))
                .await
                .map_err(|_| {
                    format!(
                        "compiler measurement {index} did not finish within {} seconds",
                        timeout.as_secs()
                    )
                })?
                .map_err(|error| format!("compiler measurement {index} failed: {error}"))?;
            Ok::<_, String>((index, committed))
        });
    }
    let mut committed = Vec::with_capacity(tasks.len());
    while let Some(result) = tasks.join_next().await {
        committed.push(result.map_err(|error| error.to_string())??);
    }
    committed.sort_unstable_by_key(|(index, _)| *index);
    Ok(committed
        .into_iter()
        .map(|(_, measurement)| measurement)
        .collect())
}

struct MeasurementBatchReport {
    first_job_id: String,
    first_schedule: String,
    subject_hashes: [String; 2],
    subject_rewards: [f32; 2],
    reward_pairs: Vec<[f32; 2]>,
    sample_pairs_ms: Vec<[Vec<f32>; 2]>,
    mean_job_compile_seconds: f64,
    first_subject_measure: MeasureSummary,
}

fn validate_measurements<G: Copy>(
    committed: Vec<CommittedMeasurement>,
    graphs: [G; 2],
    options: gz_engine::MeasureOptions,
) -> Result<MeasurementBatchReport, String> {
    let job_count = committed.len();
    if job_count == 0 {
        return Err("measurement batch must contain at least one job".to_owned());
    }
    let mut first_job_id = None;
    let mut first_schedule = None;
    let mut subject_hashes = None;
    let mut subject_rewards = None;
    let mut reward_pairs = Vec::with_capacity(job_count);
    let mut sample_pairs_ms = Vec::with_capacity(job_count);
    let mut job_compile_ns = 0_u128;
    let mut first_subject_measure = None;
    for committed in committed {
        if first_job_id.is_none() {
            first_job_id = Some(committed.job_id.to_string());
            first_schedule = Some(
                committed
                    .report
                    .samples
                    .iter()
                    .map(|sample| sample.logical_index.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        job_compile_ns += committed
            .report
            .subjects
            .iter()
            .filter_map(|subject| subject.compile_elapsed_ns)
            .max()
            .map(u128::from)
            .unwrap_or_default();
        let [first, second] = committed
            .into_measure_pair_results(graphs, options)
            .map_err(|error| error.to_string())?;
        first_subject_measure.get_or_insert_with(|| MeasureSummary::from(&first));
        for (logical_index, result) in [&first, &second].into_iter().enumerate() {
            if !result.valid {
                return Err(format!(
                    "subject {logical_index} measurement failed: {:?}",
                    result.failure
                ));
            }
        }
        subject_hashes
            .get_or_insert_with(|| [first.graph_hash.to_string(), second.graph_hash.to_string()]);
        let rewards = [
            first
                .scalar_reward
                .ok_or_else(|| "subject 0 omitted scalar reward".to_owned())?,
            second
                .scalar_reward
                .ok_or_else(|| "subject 1 omitted scalar reward".to_owned())?,
        ];
        let samples = [
            first
                .latency
                .as_ref()
                .map(|latency| latency.samples_ms.clone())
                .unwrap_or_default(),
            second
                .latency
                .as_ref()
                .map(|latency| latency.samples_ms.clone())
                .unwrap_or_default(),
        ];
        subject_rewards.get_or_insert(rewards);
        reward_pairs.push(rewards);
        sample_pairs_ms.push(samples);
    }
    Ok(MeasurementBatchReport {
        first_job_id: first_job_id.expect("non-empty batch has a first job"),
        first_schedule: first_schedule.expect("non-empty batch has a first schedule"),
        subject_hashes: subject_hashes.expect("non-empty batch has subject hashes"),
        subject_rewards: subject_rewards.expect("non-empty batch has subject rewards"),
        reward_pairs,
        sample_pairs_ms,
        mean_job_compile_seconds: job_compile_ns as f64 / 1_000_000_000.0 / job_count as f64,
        first_subject_measure: first_subject_measure.expect("non-empty batch has a first subject"),
    })
}

struct Args {
    listen: SocketAddr,
    server_cert: PathBuf,
    server_key: PathBuf,
    client_ca: PathBuf,
    client_cert: PathBuf,
    device_id: DeviceId,
    deployment: PathBuf,
    root_only: bool,
    candidate_sequence: Option<Vec<String>>,
    candidate_prefix: Option<String>,
    candidate_steps: usize,
    action_hashes: Option<Vec<CandidateHash>>,
    expected_graph_hash: Option<GraphHash>,
    comparison_action_hashes: Option<Vec<CandidateHash>>,
    expected_comparison_graph_hash: Option<GraphHash>,
    samples: u32,
    warmup_jobs: usize,
    jobs: usize,
    timeout: Duration,
    teacher_replay_dir: Option<PathBuf>,
    teacher_feature_layout: CompilerFeatureLayout,
    teacher_max_nodes: u32,
    teacher_max_edges: u32,
    teacher_max_candidates: usize,
    teacher_max_subjects: u32,
    teacher_max_steps: usize,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut listen = None;
        let mut server_cert = None;
        let mut server_key = None;
        let mut client_ca = None;
        let mut client_cert = None;
        let mut device_id = None;
        let mut deployment = None;
        let mut root_only = false;
        let mut candidate_sequence = None;
        let mut candidate_prefix = None;
        let mut candidate_steps = 1_usize;
        let mut action_hashes = None;
        let mut expected_graph_hash = None;
        let mut comparison_action_hashes = None;
        let mut expected_comparison_graph_hash = None;
        let mut samples = 4_u32;
        let mut warmup_jobs = 1_usize;
        let mut jobs = 1_usize;
        let mut timeout_seconds = 600_u64;
        let mut teacher_replay_dir = None;
        let mut teacher_feature_layout = CompilerFeatureLayout::CompactV1;
        let mut teacher_max_nodes = 98_304_u32;
        let mut teacher_max_edges = 196_608_u32;
        let mut teacher_max_candidates = 256_usize;
        let mut teacher_max_subjects = 24_u32;
        let mut teacher_max_steps = 15_usize;
        let mut index = 0;
        while index < args.len() {
            let flag = &args[index];
            index += 1;
            let value = args
                .get(index)
                .ok_or_else(|| format!("missing value for {flag}\n{}", usage()))?;
            index += 1;
            match flag.as_str() {
                "--listen" => {
                    listen = Some(
                        value
                            .parse()
                            .map_err(|_| "--listen expects HOST:PORT".to_owned())?,
                    )
                }
                "--server-cert" => server_cert = Some(PathBuf::from(value)),
                "--server-key" => server_key = Some(PathBuf::from(value)),
                "--client-ca" => client_ca = Some(PathBuf::from(value)),
                "--client-cert" => client_cert = Some(PathBuf::from(value)),
                "--device-id" => {
                    device_id = Some(DeviceId::from_str(value).map_err(|error| error.to_string())?)
                }
                "--deployment" => deployment = Some(PathBuf::from(value)),
                "--root-only" => {
                    root_only = value
                        .parse::<bool>()
                        .map_err(|_| "--root-only expects true or false".to_owned())?;
                }
                "--candidate-sequence" => {
                    let sequence = value
                        .split(',')
                        .map(str::trim)
                        .map(str::to_owned)
                        .collect::<Vec<_>>();
                    if sequence.is_empty() || sequence.iter().any(String::is_empty) {
                        return Err(
                            "--candidate-sequence expects comma-separated non-empty prefixes"
                                .to_owned(),
                        );
                    }
                    candidate_sequence = Some(sequence);
                }
                "--candidate-prefix" => candidate_prefix = Some(value.clone()),
                "--candidate-steps" => {
                    candidate_steps = value
                        .parse::<usize>()
                        .map_err(|_| "--candidate-steps expects an unsigned integer".to_owned())?;
                    if candidate_steps == 0 {
                        return Err("--candidate-steps must be greater than zero".to_owned());
                    }
                }
                "--action-hashes" => {
                    action_hashes = Some(parse_candidate_hashes(value, "--action-hashes")?);
                }
                "--expected-graph-hash" => {
                    expected_graph_hash = Some(
                        GraphHash::from_str(value)
                            .map_err(|error| format!("invalid graph hash: {error}"))?,
                    );
                }
                "--comparison-action-hashes" => {
                    comparison_action_hashes =
                        Some(parse_candidate_hashes(value, "--comparison-action-hashes")?);
                }
                "--expected-comparison-graph-hash" => {
                    expected_comparison_graph_hash = Some(
                        GraphHash::from_str(value)
                            .map_err(|error| format!("invalid graph hash: {error}"))?,
                    );
                }
                "--samples" => {
                    samples = value
                        .parse::<u32>()
                        .map_err(|_| "--samples expects an unsigned integer".to_owned())?;
                    if samples == 0 {
                        return Err("--samples must be greater than zero".to_owned());
                    }
                }
                "--warmup-jobs" => {
                    warmup_jobs = value
                        .parse::<usize>()
                        .map_err(|_| "--warmup-jobs expects an unsigned integer".to_owned())?;
                    if warmup_jobs == 0 {
                        return Err("--warmup-jobs must be greater than zero".to_owned());
                    }
                }
                "--jobs" => {
                    jobs = value
                        .parse::<usize>()
                        .map_err(|_| "--jobs expects an unsigned integer".to_owned())?;
                    if jobs == 0 {
                        return Err("--jobs must be greater than zero".to_owned());
                    }
                }
                "--timeout-seconds" => {
                    timeout_seconds = value
                        .parse::<u64>()
                        .map_err(|_| "--timeout-seconds expects an unsigned integer".to_owned())?;
                    if timeout_seconds == 0 {
                        return Err("--timeout-seconds must be greater than zero".to_owned());
                    }
                }
                "--teacher-replay-dir" => teacher_replay_dir = Some(PathBuf::from(value)),
                "--teacher-feature-layout" => {
                    teacher_feature_layout = CompilerFeatureLayout::from_str(value)?;
                }
                "--teacher-max-nodes" => {
                    teacher_max_nodes = parse_positive(value, "--teacher-max-nodes")?;
                }
                "--teacher-max-edges" => {
                    teacher_max_edges = parse_positive(value, "--teacher-max-edges")?;
                }
                "--teacher-max-candidates" => {
                    teacher_max_candidates = parse_positive(value, "--teacher-max-candidates")?;
                }
                "--teacher-max-subjects" => {
                    teacher_max_subjects = parse_positive(value, "--teacher-max-subjects")?;
                }
                "--teacher-max-steps" => {
                    teacher_max_steps = parse_positive(value, "--teacher-max-steps")?;
                }
                _ => return Err(format!("unknown flag: {flag}\n{}", usage())),
            }
        }
        if expected_graph_hash.is_some() && action_hashes.is_none() {
            return Err("--expected-graph-hash requires --action-hashes".to_owned());
        }
        let candidate_selector_count = usize::from(root_only)
            + usize::from(candidate_sequence.is_some())
            + usize::from(candidate_prefix.is_some())
            + usize::from(action_hashes.is_some());
        if candidate_selector_count > 1 {
            return Err(
                "--candidate-sequence, --candidate-prefix, and --action-hashes conflict".to_owned(),
            );
        }
        if candidate_steps != 1 && candidate_prefix.is_none() {
            return Err("--candidate-steps requires --candidate-prefix".to_owned());
        }
        if comparison_action_hashes.is_some() && action_hashes.is_none() {
            return Err("--comparison-action-hashes requires --action-hashes".to_owned());
        }
        if expected_comparison_graph_hash.is_some() && comparison_action_hashes.is_none() {
            return Err(
                "--expected-comparison-graph-hash requires --comparison-action-hashes".to_owned(),
            );
        }
        if teacher_replay_dir.is_some() && action_hashes.is_none() {
            return Err("--teacher-replay-dir requires --action-hashes".to_owned());
        }
        if let Some(hashes) = &action_hashes
            && teacher_replay_dir.is_some()
            && hashes.len() > teacher_max_steps
        {
            return Err("teacher trajectory exceeds --teacher-max-steps".to_owned());
        }
        Ok(Self {
            listen: required(listen, "--listen")?,
            server_cert: required(server_cert, "--server-cert")?,
            server_key: required(server_key, "--server-key")?,
            client_ca: required(client_ca, "--client-ca")?,
            client_cert: required(client_cert, "--client-cert")?,
            device_id: required(device_id, "--device-id")?,
            deployment: required(deployment, "--deployment")?,
            root_only,
            candidate_sequence,
            candidate_prefix,
            candidate_steps,
            action_hashes,
            expected_graph_hash,
            comparison_action_hashes,
            expected_comparison_graph_hash,
            samples,
            warmup_jobs,
            jobs,
            timeout: Duration::from_secs(timeout_seconds),
            teacher_replay_dir,
            teacher_feature_layout,
            teacher_max_nodes,
            teacher_max_edges,
            teacher_max_candidates,
            teacher_max_subjects,
            teacher_max_steps,
        })
    }
}

fn parse_positive<T>(value: &str, flag: &str) -> Result<T, String>
where
    T: FromStr + PartialEq + Default,
{
    let parsed = value
        .parse::<T>()
        .map_err(|_| format!("{flag} expects a positive integer"))?;
    if parsed == T::default() {
        return Err(format!("{flag} must be greater than zero"));
    }
    Ok(parsed)
}

fn parse_candidate_hashes(value: &str, flag: &str) -> Result<Vec<CandidateHash>, String> {
    let hashes = value
        .split(',')
        .map(|value| {
            CandidateHash::from_str(value)
                .map_err(|error| format!("invalid candidate hash: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if hashes.is_empty() {
        return Err(format!("{flag} must not be empty"));
    }
    Ok(hashes)
}

fn required<T>(value: Option<T>, flag: &str) -> Result<T, String> {
    value.ok_or_else(|| format!("missing required {flag}\n{}", usage()))
}

fn read(path: &PathBuf) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))
}

fn usage() -> &'static str {
    "usage: gz-compiler-measure-smoke --listen HOST:PORT --server-cert PATH --server-key PATH --client-ca PATH --client-cert PATH --device-id HEX --deployment PATH [--root-only true | --candidate-sequence PREFIX,... | --candidate-prefix PREFIX [--candidate-steps N] | --action-hashes HASH,... --expected-graph-hash HASH] [--comparison-action-hashes HASH,... --expected-comparison-graph-hash HASH] [--samples N] [--warmup-jobs N] [--jobs N] [--timeout-seconds N] [--teacher-replay-dir PATH --teacher-feature-layout expanded-ports|compact-v1 --teacher-max-nodes N --teacher-max-edges N --teacher-max-candidates N --teacher-max-subjects N --teacher-max-steps N]"
}
