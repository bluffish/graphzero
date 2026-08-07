use gz_engine::{GraphEngine, MeasureOptions};
use gz_engine_whittle::WhittleEngine;
use gz_measure_agent::{WhittleBackend, named_test_profile, named_test_profile_hash, submission};
use gz_measure_service::{
    AgentBackend, AgentBackendError, AgentConfig, AgentMeasurement, AgentRuntime, AgentTlsConfig,
    Coordinator, CoordinatorConfig, CoordinatorServer, CoordinatorTlsConfig, DeviceId, Enrollment,
    FetchedSubject, InsecureTransport, ReceiptLedgerConfig, certificate_fingerprint, wire,
};
use rcgen::{
    BasicConstraints, Certificate, CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer,
    KeyPair, KeyUsagePurpose,
};
use std::sync::{Arc, Condvar, Mutex};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn whittle_measure_round_trips_over_mtls() {
    let certificates = certificates();
    let device_id = DeviceId::from_bytes([9; 16]);
    let receipt_dir = tempfile::tempdir().unwrap();
    let coordinator_config = CoordinatorConfig {
        enrollment: Enrollment::one_mutual_tls(
            certificate_fingerprint(certificates.client_cert.der().as_ref()),
            device_id,
        ),
        receipt_ledger: ReceiptLedgerConfig::Directory {
            path: receipt_dir.path().to_owned(),
        },
        ..CoordinatorConfig::default()
    };
    let coordinator = Coordinator::new(coordinator_config.clone()).unwrap();
    let handle = coordinator.handle();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = CoordinatorServer::mutual_tls(
        coordinator,
        CoordinatorTlsConfig {
            certificate_pem: certificates.server_cert.pem().into_bytes(),
            private_key_pem: certificates.server_key.serialize_pem().into_bytes(),
            client_ca_pem: certificates.ca_cert.pem().into_bytes(),
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

    let state_dir = tempfile::tempdir().unwrap();
    let profile_name = "network-test";
    let agent_config = AgentConfig::mutual_tls(
        format!("https://{address}"),
        device_id,
        named_test_profile(profile_name),
        "network-test-agent",
        state_dir.path().to_owned(),
        AgentTlsConfig {
            server_ca_pem: certificates.ca_cert.pem().into_bytes(),
            certificate_pem: certificates.client_cert.pem().into_bytes(),
            private_key_pem: certificates.client_key.serialize_pem().into_bytes(),
            server_name: "localhost".to_owned(),
        },
    );
    let agent = AgentRuntime::new(agent_config, WhittleBackend).unwrap();
    let agent_task = tokio::spawn(async move { agent.run_session().await });

    let mut engine = WhittleEngine::default();
    let graph = engine.root();
    let base = engine.measure_options();
    let options = MeasureOptions::new(base.config_hash, 1, Some(5_000), true).unwrap();
    let expected = engine.measure(graph, options).unwrap();
    let request = submission(
        &engine,
        graph,
        options,
        named_test_profile_hash(profile_name),
    )
    .unwrap();
    let committed = handle.measure(request).await.unwrap();
    let actual = committed.into_measure_result(graph, options).unwrap();

    assert_eq!(actual, expected);

    let paired_request = submission(
        &engine,
        graph,
        options,
        named_test_profile_hash(profile_name),
    )
    .unwrap()
    .into_symmetric_pair(&engine, graph)
    .unwrap();
    let paired = handle.measure(paired_request).await.unwrap();
    assert_eq!(
        paired.report.kind,
        gz_measure_service::wire::MeasureKind::SymmetricPair as i32
    );
    assert_eq!(paired.report.subjects.len(), 2);
    let paired_results = paired
        .into_measure_pair_results([graph, graph], options)
        .unwrap();
    assert_eq!(paired_results, [expected.clone(), expected]);
    assert_eq!(handle.snapshot().committed_receipts, 2);
    let cleanup_deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
    while state_dir.path().read_dir().unwrap().next().is_some()
        && tokio::time::Instant::now() < cleanup_deadline
    {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(state_dir.path().read_dir().unwrap().next().is_none());

    agent_task.abort();
    let _ = shutdown_tx.send(());
    server_task.await.unwrap().unwrap();

    let restarted = Coordinator::new(coordinator_config).unwrap();
    assert_eq!(restarted.snapshot().committed_receipts, 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_advertised_slots_execute_measurements_concurrently() {
    let device_id = DeviceId::from_bytes([10; 16]);
    let profile_name = "two-slot-network-test";
    let coordinator = Coordinator::new(CoordinatorConfig {
        enrollment: Enrollment::one_insecure_test_device(device_id),
        ..CoordinatorConfig::default()
    })
    .unwrap();
    let handle = coordinator.handle();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server_task = tokio::spawn(async move {
        CoordinatorServer::insecure_for_tests(coordinator, InsecureTransport::for_tests())
            .serve_with_shutdown(listener, async move {
                let _ = shutdown_rx.await;
            })
            .await
    });

    let gate = Arc::new(MeasurementGate::default());
    let state_dir = tempfile::tempdir().unwrap();
    let agent = AgentRuntime::new(
        AgentConfig::insecure_for_tests(
            format!("http://{address}"),
            device_id,
            named_test_profile(profile_name),
            "two-slot-network-test-agent",
            state_dir.path().to_owned(),
            InsecureTransport::for_tests(),
        ),
        BlockingWhittleBackend {
            gate: Arc::clone(&gate),
        },
    )
    .unwrap();
    let agent_task = tokio::spawn(async move { agent.run_session().await });

    let engine = WhittleEngine::default();
    let graph = engine.root();
    let options =
        MeasureOptions::new(engine.measure_options().config_hash, 1, Some(5_000), true).unwrap();
    let profile_hash = named_test_profile_hash(profile_name);
    let first_submission = submission(&engine, graph, options, profile_hash).unwrap();
    let second_submission = submission(&engine, graph, options, profile_hash).unwrap();
    let first_handle = handle.clone();
    let first = tokio::spawn(async move { first_handle.measure(first_submission).await });
    let second_handle = handle.clone();
    let second = tokio::spawn(async move { second_handle.measure(second_submission).await });

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
    let observed_concurrency = loop {
        if gate.active() == 2 {
            break true;
        }
        if tokio::time::Instant::now() >= deadline {
            break false;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    };
    let active_leases = handle.snapshot().active_leases;
    gate.release();

    assert!(
        observed_concurrency,
        "both advertised measurement slots did not run concurrently"
    );
    assert_eq!(active_leases, 2);
    let first = tokio::time::timeout(std::time::Duration::from_secs(2), first)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let second = tokio::time::timeout(std::time::Duration::from_secs(2), second)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(first.into_measure_result(graph, options).unwrap().valid);
    assert!(second.into_measure_result(graph, options).unwrap().valid);
    assert_eq!(gate.peak(), 2);

    agent_task.abort();
    let _ = shutdown_tx.send(());
    server_task.await.unwrap().unwrap();
}

#[derive(Default)]
struct MeasurementGate {
    state: Mutex<MeasurementGateState>,
    changed: Condvar,
}

#[derive(Default)]
struct MeasurementGateState {
    active: usize,
    peak: usize,
    released: bool,
}

impl MeasurementGate {
    fn enter(&self) {
        let mut state = self.state.lock().unwrap();
        state.active += 1;
        state.peak = state.peak.max(state.active);
        self.changed.notify_all();
        while !state.released {
            state = self.changed.wait(state).unwrap();
        }
    }

    fn leave(&self) {
        self.state.lock().unwrap().active -= 1;
    }

    fn active(&self) -> usize {
        self.state.lock().unwrap().active
    }

    fn peak(&self) -> usize {
        self.state.lock().unwrap().peak
    }

    fn release(&self) {
        self.state.lock().unwrap().released = true;
        self.changed.notify_all();
    }
}

struct BlockingWhittleBackend {
    gate: Arc<MeasurementGate>,
}

impl AgentBackend for BlockingWhittleBackend {
    fn capability(&self) -> wire::EngineCapability {
        WhittleBackend.capability()
    }

    fn concurrent_job_slots(&self) -> u32 {
        2
    }

    fn measure(
        &self,
        lease: &wire::MeasureLease,
        subjects: Vec<FetchedSubject>,
    ) -> Result<AgentMeasurement, AgentBackendError> {
        self.gate.enter();
        let result = WhittleBackend.measure(lease, subjects);
        self.gate.leave();
        result
    }
}

struct TestCertificates {
    ca_cert: Certificate,
    server_cert: Certificate,
    server_key: KeyPair,
    client_cert: Certificate,
    client_key: KeyPair,
}

fn certificates() -> TestCertificates {
    let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    ca_params.key_usages = vec![
        KeyUsagePurpose::DigitalSignature,
        KeyUsagePurpose::KeyCertSign,
        KeyUsagePurpose::CrlSign,
    ];
    let ca_key = KeyPair::generate().unwrap();
    let ca_cert = ca_params.self_signed(&ca_key).unwrap();
    let issuer = Issuer::new(ca_params, ca_key);
    let (server_cert, server_key) = leaf(&issuer, "localhost", ExtendedKeyUsagePurpose::ServerAuth);
    let (client_cert, client_key) = leaf(
        &issuer,
        "graphzero-test-agent",
        ExtendedKeyUsagePurpose::ClientAuth,
    );
    TestCertificates {
        ca_cert,
        server_cert,
        server_key,
        client_cert,
        client_key,
    }
}

fn leaf(
    issuer: &Issuer<'_, KeyPair>,
    name: &str,
    usage: ExtendedKeyUsagePurpose,
) -> (Certificate, KeyPair) {
    let mut params = CertificateParams::new(vec![name.to_owned()]).unwrap();
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![usage];
    params.use_authority_key_identifier_extension = true;
    let key = KeyPair::generate().unwrap();
    let cert = params.signed_by(&key, issuer).unwrap();
    (cert, key)
}
