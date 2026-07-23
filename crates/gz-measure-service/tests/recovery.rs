use gz_engine::{EngineIdentity, GraphEngine, MeasureOptions};
use gz_engine_whittle::WhittleEngine;
use gz_measure_service::wire::measure_fleet_client::MeasureFleetClient;
use gz_measure_service::{
    Coordinator, CoordinatorConfig, CoordinatorServer, DeviceId, EncodedMeasureConfig, Enrollment,
    InsecureTransport, MEASUREMENT_PROTOCOL_VERSION, MeasureSubmission, RequestNonce,
    device_profile_hash, engine_identity_to_wire, wire,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn persisted_report_lease_rebinds_to_reconnected_session() {
    let device_id = DeviceId::from_bytes([23; 16]);
    let profile = profile();
    let profile_hash = device_profile_hash(&profile);
    let coordinator = Coordinator::new(CoordinatorConfig {
        enrollment: Enrollment::one_insecure_test_device(device_id),
        ..CoordinatorConfig::default()
    })
    .unwrap();
    let handle = coordinator.handle();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        CoordinatorServer::insecure_for_tests(coordinator, InsecureTransport::for_tests())
            .serve_with_shutdown(listener, async move {
                let _ = shutdown_rx.await;
            })
            .await
    });

    let mut engine = WhittleEngine::default();
    let graph = engine.root();
    let options = MeasureOptions::new(engine.measure_options().config_hash, 1, None, true).unwrap();
    let expected = engine.measure(graph, options).unwrap();
    let submission = MeasureSubmission::from_engine(
        &engine,
        graph,
        options,
        EncodedMeasureConfig {
            encoding: 1,
            payload: vec![1],
        },
        profile_hash,
    )
    .unwrap();
    let capability = capability(&engine);

    let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = MeasureFleetClient::new(channel);
    let (first_events, first_rx) = mpsc::channel(8);
    first_events
        .send(hello(device_id, &profile, capability.clone(), Vec::new()))
        .await
        .unwrap();
    let mut first_commands = client
        .connect(ReceiverStream::new(first_rx))
        .await
        .unwrap()
        .into_inner();
    let first_welcome = welcome(first_commands.message().await.unwrap().unwrap());
    first_events
        .send(ready(first_welcome.session_id.clone()))
        .await
        .unwrap();

    let (job_id, committed) = handle
        .enqueue_with_nonce(submission, RequestNonce::from_bytes([31; 16]))
        .unwrap();
    let lease = match first_commands
        .message()
        .await
        .unwrap()
        .unwrap()
        .command
        .unwrap()
    {
        wire::coordinator_command::Command::Lease(lease) => lease,
        _ => panic!("expected lease"),
    };
    first_events
        .send(wire::AgentEvent {
            event: Some(wire::agent_event::Event::Accepted(wire::JobAccepted {
                session_id: first_welcome.session_id.clone(),
                job_id: lease.job_id.clone(),
                lease_id: lease.lease_id.clone(),
            })),
        })
        .await
        .unwrap();
    wait_until(|| handle.snapshot().running_leases == 1).await;
    drop(first_events);
    drop(first_commands);
    wait_until(|| handle.snapshot().connected_agents == 0).await;

    let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut recovered_client = MeasureFleetClient::new(channel);
    let (recovered_events, recovered_rx) = mpsc::channel(8);
    recovered_events
        .send(hello(
            device_id,
            &profile,
            capability,
            vec![wire::LeaseRecovery {
                job_id: lease.job_id.clone(),
                lease_id: lease.lease_id.clone(),
                phase: wire::JobPhase::SubmitReport as i32,
                report_persisted: true,
            }],
        ))
        .await
        .unwrap();
    let mut recovered_commands = recovered_client
        .connect(ReceiverStream::new(recovered_rx))
        .await
        .unwrap()
        .into_inner();
    let recovered_welcome = welcome(recovered_commands.message().await.unwrap().unwrap());
    assert_eq!(recovered_welcome.recoveries.len(), 1);
    assert_eq!(
        recovered_welcome.recoveries[0].action,
        wire::RecoveryAction::RecoverySubmitReport as i32
    );

    let report = wire::MeasureReport {
        origin_session_id: first_welcome.session_id,
        device_id: device_id.to_vec(),
        device_profile_hash: profile_hash.to_vec(),
        job_id: lease.job_id.clone(),
        measurement_key: lease.measurement_key.clone(),
        request_nonce: lease.request_nonce.clone(),
        lease_id: lease.lease_id.clone(),
        engine: lease.engine.clone(),
        measure_config_hash: lease.measure_config_hash.clone(),
        kind: lease.kind,
        outcome: wire::MeasureAttemptOutcome::MeasureOutcomeSucceeded as i32,
        subjects: vec![wire::SubjectMeasurement {
            logical_index: 0,
            graph_hash: expected.graph_hash.as_bytes().to_vec(),
            compile_elapsed_ns: None,
            capture_elapsed_ns: None,
            scalar_reward: expected.scalar_reward.map(f64::from),
            engine_metadata: expected.metadata.bytes.clone(),
        }],
        samples: Vec::new(),
        telemetry: Vec::new(),
        failure: None,
    };
    let ack = recovered_client
        .submit_result(wire::SubmitResultRequest {
            current_session_id: recovered_welcome.session_id,
            report: Some(report),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(
        ack.disposition,
        wire::ResultDisposition::ResultJobCommitted as i32
    );
    let committed = committed.await.unwrap().unwrap();
    assert_eq!(committed.job_id, job_id);
    assert_eq!(
        committed.into_measure_result(graph, options).unwrap(),
        expected
    );

    drop(recovered_events);
    drop(recovered_commands);
    let _ = shutdown_tx.send(());
    server.await.unwrap().unwrap();
}

fn hello(
    device_id: DeviceId,
    profile: &wire::DeviceProfile,
    capability: wire::EngineCapability,
    recoveries: Vec<wire::LeaseRecovery>,
) -> wire::AgentEvent {
    wire::AgentEvent {
        event: Some(wire::agent_event::Event::Hello(wire::AgentHello {
            protocol_major: 1,
            protocol_minor: 0,
            device_id: device_id.to_vec(),
            agent_build: "recovery-test".to_owned(),
            profile: Some(profile.clone()),
            capabilities: vec![capability],
            recoveries,
        })),
    }
}

fn ready(session_id: Vec<u8>) -> wire::AgentEvent {
    wire::AgentEvent {
        event: Some(wire::agent_event::Event::Ready(wire::AgentReady {
            session_id,
            free_slots: 1,
            telemetry: None,
        })),
    }
}

fn welcome(command: wire::CoordinatorCommand) -> wire::AgentWelcome {
    match command.command.unwrap() {
        wire::coordinator_command::Command::Welcome(welcome) => welcome,
        _ => panic!("expected welcome"),
    }
}

fn capability(engine: &WhittleEngine) -> wire::EngineCapability {
    wire::EngineCapability {
        engine: Some(engine_identity_to_wire(EngineIdentity::from_engine(engine))),
        artifact_formats: vec![wire::ArtifactFormatCapability {
            format_kind: wire::GraphArtifactFormatKind::GraphArtifactFormatBinary as i32,
            adapter_format_id: 0,
        }],
        measure_config_encodings: vec![1],
        measurement_protocol_versions: vec![MEASUREMENT_PROTOCOL_VERSION],
    }
}

fn profile() -> wire::DeviceProfile {
    wire::DeviceProfile {
        platform_family: "test".to_owned(),
        board_model: "recovery".to_owned(),
        soc: "test".to_owned(),
        gpu_architecture: "test".to_owned(),
        usable_memory_bytes: 1,
        operating_system_image_digest: vec![1; 32],
        platform_release: "test".to_owned(),
        cuda_version: "test".to_owned(),
        gpu_driver_version: "test".to_owned(),
        compiler_version: "test".to_owned(),
        compiler_runtime_version: "test".to_owned(),
        agent_image_digest: vec![2; 32],
        power_profile: "test".to_owned(),
        clock_policy: "test".to_owned(),
        cooling_policy: "test".to_owned(),
        measurement_protocol_version: MEASUREMENT_PROTOCOL_VERSION,
    }
}

async fn wait_until(predicate: impl Fn() -> bool) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
    while !predicate() && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(predicate());
}
