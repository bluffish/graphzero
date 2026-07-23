#![forbid(unsafe_code)]

use gz_measure_agent::{WhittleBackend, named_test_profile};
use gz_measure_service::{AgentConfig, AgentRuntime, AgentTlsConfig, DeviceId};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1).collect())?;
    let tls = AgentTlsConfig {
        server_ca_pem: read(&args.ca)?,
        certificate_pem: read(&args.cert)?,
        private_key_pem: read(&args.key)?,
        server_name: args.server_name,
    };
    let config = AgentConfig::mutual_tls(
        args.coordinator,
        args.device_id,
        named_test_profile(&args.profile),
        env!("CARGO_PKG_VERSION"),
        args.state_dir,
        tls,
    );
    let runtime = AgentRuntime::new(config, WhittleBackend).map_err(|error| error.to_string())?;

    let mut failed_sessions = 0_u64;
    loop {
        tokio::select! {
            result = runtime.run_session() => {
                match result {
                    Ok(()) => return Ok(()),
                    Err(error) => {
                        failed_sessions = failed_sessions.saturating_add(1);
                        if failed_sessions == 1 || failed_sessions.is_multiple_of(60) {
                            eprintln!(
                                "measurement session unavailable failures={failed_sessions}: {error}"
                            );
                        }
                    }
                }
            }
            signal = tokio::signal::ctrl_c() => {
                signal.map_err(|error| error.to_string())?;
                return Ok(());
            }
        }
        tokio::time::sleep(args.reconnect_delay).await;
    }
}

struct Args {
    coordinator: String,
    server_name: String,
    ca: PathBuf,
    cert: PathBuf,
    key: PathBuf,
    device_id: DeviceId,
    profile: String,
    state_dir: PathBuf,
    reconnect_delay: Duration,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut coordinator = None;
        let mut server_name = None;
        let mut ca = None;
        let mut cert = None;
        let mut key = None;
        let mut device_id = None;
        let mut profile = None;
        let mut state_dir = None;
        let mut reconnect_ms = 1_000;
        let mut index = 0;
        while index < args.len() {
            let flag = &args[index];
            index += 1;
            let value = args
                .get(index)
                .ok_or_else(|| format!("missing value for {flag}\n{}", usage()))?;
            index += 1;
            match flag.as_str() {
                "--coordinator" => coordinator = Some(value.clone()),
                "--server-name" => server_name = Some(value.clone()),
                "--ca" => ca = Some(PathBuf::from(value)),
                "--cert" => cert = Some(PathBuf::from(value)),
                "--key" => key = Some(PathBuf::from(value)),
                "--device-id" => {
                    device_id = Some(DeviceId::from_str(value).map_err(|error| error.to_string())?)
                }
                "--profile" => profile = Some(value.clone()),
                "--state-dir" => state_dir = Some(PathBuf::from(value)),
                "--reconnect-ms" => {
                    reconnect_ms = value
                        .parse::<u64>()
                        .map_err(|_| "--reconnect-ms expects an unsigned integer".to_owned())?;
                    if reconnect_ms == 0 {
                        return Err("--reconnect-ms must be greater than zero".to_owned());
                    }
                }
                _ => return Err(format!("unknown flag: {flag}\n{}", usage())),
            }
        }

        Ok(Self {
            coordinator: required(coordinator, "--coordinator")?,
            server_name: required(server_name, "--server-name")?,
            ca: required(ca, "--ca")?,
            cert: required(cert, "--cert")?,
            key: required(key, "--key")?,
            device_id: required(device_id, "--device-id")?,
            profile: required(profile, "--profile")?,
            state_dir: required(state_dir, "--state-dir")?,
            reconnect_delay: Duration::from_millis(reconnect_ms),
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
    "usage: gz-measure-agent --coordinator https://HOST:PORT --server-name NAME --ca PATH --cert PATH --key PATH --device-id HEX --profile NAME --state-dir PATH [--reconnect-ms MS]"
}
