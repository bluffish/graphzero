#![forbid(unsafe_code)]

use gz_engine::{GraphEngine, MeasureOptions};
use gz_engine_whittle::WhittleEngine;
use gz_measure_agent::{named_test_profile_hash, submission};
use gz_measure_service::{
    Coordinator, CoordinatorConfig, CoordinatorServer, CoordinatorTlsConfig, DeviceId, Enrollment,
    certificate_fingerprint,
};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1).collect())?;
    let client_certificate = pem::parse(read(&args.client_cert)?)
        .map_err(|error| format!("failed to parse {}: {error}", args.client_cert.display()))?;
    let config = CoordinatorConfig {
        enrollment: Enrollment::one_mutual_tls(
            certificate_fingerprint(client_certificate.contents()),
            args.device_id,
        ),
        ..CoordinatorConfig::default()
    };
    let coordinator = Coordinator::new(config).map_err(|error| error.to_string())?;
    let handle = coordinator.handle();
    let listener = tokio::net::TcpListener::bind(args.listen)
        .await
        .map_err(|error| error.to_string())?;
    let listen = listener.local_addr().map_err(|error| error.to_string())?;
    let tls = CoordinatorTlsConfig {
        certificate_pem: read(&args.server_cert)?,
        private_key_pem: read(&args.server_key)?,
        client_ca_pem: read(&args.client_ca)?,
    };
    let server = CoordinatorServer::mutual_tls(coordinator, tls);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server_task = tokio::spawn(async move {
        server
            .serve_with_shutdown(listener, async move {
                let _ = shutdown_rx.await;
            })
            .await
    });
    println!("coordinator_listening={listen}");

    let mut engine = WhittleEngine::default();
    let root = engine.root();
    let base_options = engine.measure_options();
    let options = MeasureOptions::new(
        base_options.config_hash,
        base_options.samples,
        Some(args.timeout_ms),
        base_options.deterministic,
    )
    .map_err(|error| error.to_string())?;
    let expected = engine
        .measure(root, options)
        .map_err(|error| error.to_string())?;
    let request = submission(
        &engine,
        root,
        options,
        named_test_profile_hash(&args.profile),
    )
    .map_err(|error| error.to_string())?;
    let committed = handle
        .measure(request)
        .await
        .map_err(|error| error.to_string())?;
    let job_id = committed.job_id;
    let actual = committed
        .into_measure_result(root, options)
        .map_err(|error| error.to_string())?;
    if actual != expected {
        return Err(format!(
            "remote Whittle result differs from local result: local={expected:?} remote={actual:?}"
        ));
    }
    println!(
        "job_id={job_id} graph_hash={} scalar_reward={}",
        actual.graph_hash,
        actual.scalar_reward.unwrap()
    );
    let _ = shutdown_tx.send(());
    server_task
        .await
        .map_err(|error| error.to_string())?
        .map_err(|error| error.to_string())?;
    Ok(())
}

struct Args {
    listen: SocketAddr,
    server_cert: PathBuf,
    server_key: PathBuf,
    client_ca: PathBuf,
    client_cert: PathBuf,
    device_id: DeviceId,
    profile: String,
    timeout_ms: u64,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut listen = None;
        let mut server_cert = None;
        let mut server_key = None;
        let mut client_ca = None;
        let mut client_cert = None;
        let mut device_id = None;
        let mut profile = None;
        let mut timeout_ms = 60_000;
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
                "--profile" => profile = Some(value.clone()),
                "--timeout-ms" => {
                    timeout_ms = value
                        .parse::<u64>()
                        .map_err(|_| "--timeout-ms expects an unsigned integer".to_owned())?;
                    if timeout_ms == 0 {
                        return Err("--timeout-ms must be greater than zero".to_owned());
                    }
                }
                _ => return Err(format!("unknown flag: {flag}\n{}", usage())),
            }
        }
        Ok(Self {
            listen: required(listen, "--listen")?,
            server_cert: required(server_cert, "--server-cert")?,
            server_key: required(server_key, "--server-key")?,
            client_ca: required(client_ca, "--client-ca")?,
            client_cert: required(client_cert, "--client-cert")?,
            device_id: required(device_id, "--device-id")?,
            profile: required(profile, "--profile")?,
            timeout_ms,
        })
    }
}

fn required<T>(value: Option<T>, flag: &str) -> Result<T, String> {
    value.ok_or_else(|| format!("missing required {flag}\n{}", usage()))
}

fn read(path: &PathBuf) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))
}

fn usage() -> &'static str {
    "usage: gz-measure-smoke --listen HOST:PORT --server-cert PATH --server-key PATH --client-ca PATH --client-cert PATH --device-id HEX --profile NAME [--timeout-ms MS]"
}
