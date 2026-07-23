use gz_cli::remote_measure::{RemoteAgentEnrollment, RemoteMeasureConfig};
use gz_cli::selfplay::{EvaluatorMode, ReplayInitConfig, SelfplayConfig, init_replay, run};
use gz_measure_agent::{WhittleBackend, named_test_profile};
use gz_measure_service::{AgentConfig, AgentRuntime, AgentTlsConfig, DeviceId};
use gz_replay::{ReplayDataMode, ReplayEpisodeId, ReplayStore};
use rcgen::{
    BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("gz-cli-selfplay-test-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn replay_init_persists_the_feature_schema() {
    let dir = TestDir::new();
    let summary = init_replay(ReplayInitConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        max_candidates: 255,
        mask_stop: false,
    })
    .unwrap();

    assert_eq!(summary.max_actions, 256);
    let store = ReplayStore::open(dir.path()).unwrap();
    let schema = store.feature_schema().unwrap().unwrap();
    assert_eq!(schema.max_actions, 256);
}

#[test]
fn stub_selfplay_appends_both_symmetric_perspectives() {
    let dir = TestDir::new();
    let summary = run(short_config(dir.path())).unwrap();

    assert_eq!(summary.evaluator, EvaluatorMode::Stub);
    assert_eq!(summary.episodes_appended, 2);
    assert_eq!(summary.episodes_dropped, 0);
    assert_eq!(summary.wins + summary.losses + summary.ties, 4);
    assert_eq!(summary.wins, summary.losses);
    assert_eq!(summary.rows_produced, summary.replay_rows);
    assert!(summary.rows_produced > 0);

    let store = ReplayStore::open(dir.path()).unwrap();
    assert_eq!(
        store.data_mode().unwrap(),
        ReplayDataMode::SymmetricSelfplay
    );
    assert!(store.feature_schema().unwrap().is_some());
    for pair in [[0, 1], [2, 3]] {
        let left = store
            .episode(ReplayEpisodeId::new(pair[0]))
            .unwrap()
            .unwrap();
        let right = store
            .episode(ReplayEpisodeId::new(pair[1]))
            .unwrap()
            .unwrap();
        assert_eq!(
            left.outcome.value_target,
            right.outcome.value_target.map(|v| -v)
        );
        assert!(left.outcome.value_target.is_some());
        assert!(right.outcome.value_target.is_some());
    }
}

#[test]
fn stub_selfplay_commits_terminal_measures_through_remote_agent() {
    let dir = TestDir::new();
    let replay_dir = dir.path().join("replay");
    let receipt_dir = dir.path().join("receipts");
    let state_dir = dir.path().join("agent-state");
    let certificates = certificates();
    let server_cert = dir.path().join("server.pem");
    let server_key = dir.path().join("server.key");
    let client_cert = dir.path().join("client.pem");
    let client_key = dir.path().join("client.key");
    let ca = dir.path().join("ca.pem");
    std::fs::write(&server_cert, certificates.server_cert.pem()).unwrap();
    std::fs::write(&server_key, certificates.server_key.serialize_pem()).unwrap();
    std::fs::write(&client_cert, certificates.client_cert.pem()).unwrap();
    std::fs::write(&client_key, certificates.client_key.serialize_pem()).unwrap();
    std::fs::write(&ca, certificates.ca_cert.pem()).unwrap();

    let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = reservation.local_addr().unwrap();
    drop(reservation);
    let device_id = DeviceId::from_bytes([19; 16]);
    let profile = "cli-remote-test";
    let stop = Arc::new(AtomicBool::new(false));
    let agent_stop = Arc::clone(&stop);
    let agent = std::thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let config = AgentConfig::mutual_tls(
            format!("https://{address}"),
            device_id,
            named_test_profile(profile),
            "cli-remote-test",
            state_dir,
            AgentTlsConfig {
                server_ca_pem: std::fs::read(ca).unwrap(),
                certificate_pem: std::fs::read(client_cert).unwrap(),
                private_key_pem: std::fs::read(client_key).unwrap(),
                server_name: "localhost".to_owned(),
            },
        );
        let agent = AgentRuntime::new(config, WhittleBackend).unwrap();
        runtime.block_on(async move {
            while !agent_stop.load(Ordering::Acquire) {
                let _ = agent.run_session().await;
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
    });

    let mut config = short_config(&replay_dir);
    config.remote_measure = RemoteMeasureConfig {
        listen: Some(address),
        server_certificate: Some(server_cert),
        server_private_key: Some(server_key),
        client_ca: Some(dir.path().join("ca.pem")),
        agents: vec![RemoteAgentEnrollment {
            device_id,
            certificate: dir.path().join("client.pem"),
        }],
        profile: Some(profile.to_owned()),
        receipt_dir: Some(receipt_dir.clone()),
        startup_timeout: Duration::from_secs(5),
    };
    let summary = run(config).unwrap();
    stop.store(true, Ordering::Release);
    agent.join().unwrap();

    assert_eq!(summary.episodes_appended, 2);
    assert_eq!(receipt_count(&receipt_dir.join("receipts.log")), 4);
}

#[test]
fn stop_enabled_selfplay_uses_the_stop_replay_contract() {
    let dir = TestDir::new();
    let mut config = short_config(dir.path());
    config.episodes = 1;
    config.mask_stop = false;

    run(config).unwrap();

    let store = ReplayStore::open(dir.path()).unwrap();
    assert_eq!(
        store.data_mode().unwrap(),
        ReplayDataMode::SymmetricSelfplayStop
    );
    for id in 0..2 {
        let episode = store.episode(ReplayEpisodeId::new(id)).unwrap().unwrap();
        assert!(episode.outcome.value_target.is_some());
    }
}

#[test]
fn validation_rejects_incoherent_runtime_settings() {
    let dir = TestDir::new();
    let mut config = short_config(dir.path());
    config.mask_stop = false;
    config.position_features = false;
    assert!(config.validate().unwrap_err().contains("position-features"));

    let mut config = short_config(dir.path());
    config.episodes = 0;
    assert!(config.validate().unwrap_err().contains("serve-socket"));

    let mut config = short_config(dir.path());
    config.eval_processes = 2;
    assert!(config.validate().unwrap_err().contains("cannot exceed"));

    let mut config = short_config(dir.path());
    config.checkpoint_pointer = Some("step_50000.json".to_owned());
    assert!(
        config
            .validate()
            .unwrap_err()
            .contains("checkpoint-pointer")
    );
}

#[test]
fn torch_evaluator_args_select_checkpoint_and_device() {
    let dir = TestDir::new();
    let mut config = short_config(dir.path());
    config.evaluator = EvaluatorMode::Torch;
    config.checkpoint_dir = Some(PathBuf::from("/checkpoints"));
    config.checkpoint_pointer = Some("step_50000.json".to_owned());
    config.eval_device = Some("cuda:1".to_owned());
    config.eval_poll_interval = Some(0.0);
    config.validate().unwrap();

    let args = config.evaluator_extra_args();
    assert!(args.windows(2).any(|pair| pair == ["--backend", "torch"]));
    assert!(
        args.windows(2)
            .any(|pair| pair == ["--checkpoint-pointer", "step_50000.json"])
    );
    assert!(args.windows(2).any(|pair| pair == ["--device", "cuda:1"]));
    assert!(args.windows(2).any(|pair| pair == ["--poll-interval", "0"]));
    assert!(!args.iter().any(|arg| arg.starts_with("--require-")));
}

fn short_config(path: &Path) -> SelfplayConfig {
    SelfplayConfig {
        replay_dir: Some(path.to_path_buf()),
        episodes: 2,
        lanes: 1,
        workers_per_lane: 1,
        seed: 42,
        max_steps: 2,
        simulations: 2,
        max_considered: 2,
        gumbel_scale: 0.0,
        gumbel_noise_overlap: -1.0,
        tree_reuse: false,
        max_candidates: 255,
        max_batch: 2,
        evaluator: EvaluatorMode::Stub,
        mask_stop: true,
        no_backtrack: true,
        ..SelfplayConfig::default()
    }
}

fn receipt_count(path: &Path) -> usize {
    let bytes = std::fs::read(path).unwrap();
    let mut offset = 0;
    let mut count = 0;
    while offset < bytes.len() {
        let length = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4 + length;
        count += 1;
    }
    assert_eq!(offset, bytes.len());
    count
}

struct TestCertificates {
    ca_cert: rcgen::Certificate,
    server_cert: rcgen::Certificate,
    server_key: KeyPair,
    client_cert: rcgen::Certificate,
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
) -> (rcgen::Certificate, KeyPair) {
    let mut params = CertificateParams::new(vec![name.to_owned()]).unwrap();
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![usage];
    params.use_authority_key_identifier_extension = true;
    let key = KeyPair::generate().unwrap();
    let cert = params.signed_by(&key, issuer).unwrap();
    (cert, key)
}
