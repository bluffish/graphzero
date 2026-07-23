use gz_measure_service::{
    CertificateFingerprint, Coordinator, CoordinatorConfig, CoordinatorHandle, CoordinatorServer,
    CoordinatorTlsConfig, DeviceId, Enrollment, ReceiptLedgerConfig, certificate_fingerprint,
};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct RemoteAgentEnrollment {
    pub device_id: DeviceId,
    pub certificate: PathBuf,
}

#[derive(Clone, Debug)]
pub struct RemoteMeasureConfig {
    pub listen: Option<SocketAddr>,
    pub server_certificate: Option<PathBuf>,
    pub server_private_key: Option<PathBuf>,
    pub client_ca: Option<PathBuf>,
    pub agents: Vec<RemoteAgentEnrollment>,
    pub profile: Option<String>,
    pub receipt_dir: Option<PathBuf>,
    pub startup_timeout: Duration,
}

impl Default for RemoteMeasureConfig {
    fn default() -> Self {
        Self {
            listen: None,
            server_certificate: None,
            server_private_key: None,
            client_ca: None,
            agents: Vec::new(),
            profile: None,
            receipt_dir: None,
            startup_timeout: Duration::from_secs(60),
        }
    }
}

impl RemoteMeasureConfig {
    pub fn validate(&self) -> Result<(), String> {
        let enabled = self.listen.is_some();
        let any_field = self.server_certificate.is_some()
            || self.server_private_key.is_some()
            || self.client_ca.is_some()
            || !self.agents.is_empty()
            || self.profile.is_some()
            || self.receipt_dir.is_some();
        if !enabled && any_field {
            return Err("remote measurement options require --measure-listen".to_owned());
        }
        if !enabled {
            return Ok(());
        }
        if self.server_certificate.is_none() {
            return Err("remote measurement requires --measure-server-cert".to_owned());
        }
        if self.server_private_key.is_none() {
            return Err("remote measurement requires --measure-server-key".to_owned());
        }
        if self.client_ca.is_none() {
            return Err("remote measurement requires --measure-client-ca".to_owned());
        }
        if self.agents.is_empty() {
            return Err("remote measurement requires at least one --measure-agent".to_owned());
        }
        if self.profile.as_deref().is_none_or(str::is_empty) {
            return Err("remote measurement requires --measure-profile".to_owned());
        }
        if self.receipt_dir.is_none() {
            return Err("remote measurement requires --measure-receipt-dir".to_owned());
        }
        if self.startup_timeout.is_zero() {
            return Err("--measure-startup-timeout-ms must be greater than zero".to_owned());
        }
        Ok(())
    }
}

pub(crate) struct RemoteCoordinator {
    handle: CoordinatorHandle,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<JoinHandle<Result<(), String>>>,
}

impl RemoteCoordinator {
    pub fn start(
        config: &RemoteMeasureConfig,
        job_capacity: usize,
    ) -> Result<Option<Self>, String> {
        config.validate()?;
        let Some(listen) = config.listen else {
            return Ok(None);
        };
        let mut enrolled = BTreeMap::<CertificateFingerprint, DeviceId>::new();
        for agent in &config.agents {
            let certificate = read_certificate_der(&agent.certificate)?;
            let fingerprint = certificate_fingerprint(&certificate);
            if enrolled.insert(fingerprint, agent.device_id).is_some() {
                return Err("duplicate remote agent certificate".to_owned());
            }
        }
        let coordinator = Coordinator::new(CoordinatorConfig {
            enrollment: Enrollment::MutualTls(enrolled),
            queue_capacity: job_capacity,
            artifact_item_capacity: job_capacity,
            receipt_ledger: ReceiptLedgerConfig::Directory {
                path: config
                    .receipt_dir
                    .clone()
                    .expect("validated receipt directory"),
            },
            ..CoordinatorConfig::default()
        })
        .map_err(|error| error.to_string())?;
        let handle = coordinator.handle();
        let tls = CoordinatorTlsConfig {
            certificate_pem: read_required(
                config
                    .server_certificate
                    .as_ref()
                    .expect("validated server cert"),
            )?,
            private_key_pem: read_required(
                config
                    .server_private_key
                    .as_ref()
                    .expect("validated server key"),
            )?,
            client_ca_pem: read_required(config.client_ca.as_ref().expect("validated client CA"))?,
        };
        let listener = std::net::TcpListener::bind(listen).map_err(|error| {
            format!("failed to bind measurement coordinator at {listen}: {error}")
        })?;
        listener
            .set_nonblocking(true)
            .map_err(|error| error.to_string())?;
        let bound = listener.local_addr().map_err(|error| error.to_string())?;
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("gz-measure-coordinator".to_owned())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .map_err(|error| error.to_string())?;
                runtime.block_on(async move {
                    let listener = tokio::net::TcpListener::from_std(listener)
                        .map_err(|error| error.to_string())?;
                    CoordinatorServer::mutual_tls(coordinator, tls)
                        .serve_with_shutdown(listener, async move {
                            let _ = shutdown_rx.await;
                        })
                        .await
                        .map_err(|error| error.to_string())
                })
            })
            .map_err(|error| error.to_string())?;
        eprintln!("event=measure_coordinator listening={bound}");
        Ok(Some(Self {
            handle,
            shutdown: Some(shutdown_tx),
            thread: Some(thread),
        }))
    }

    pub fn wait_for_agent(&mut self, timeout: Duration) -> Result<(), String> {
        let deadline = Instant::now() + timeout;
        loop {
            if self.handle.snapshot().ready_agents > 0 {
                return Ok(());
            }
            if self.thread.as_ref().is_some_and(JoinHandle::is_finished) {
                let result = self
                    .thread
                    .take()
                    .expect("finished coordinator thread exists")
                    .join()
                    .map_err(|_| "measurement coordinator panicked".to_owned())?;
                return Err(result.err().unwrap_or_else(|| {
                    "measurement coordinator stopped during startup".to_owned()
                }));
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "no remote measurement agent became ready within {} ms",
                    timeout.as_millis()
                ));
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    pub fn handle(&self) -> CoordinatorHandle {
        self.handle.clone()
    }
}

impl Drop for RemoteCoordinator {
    fn drop(&mut self) {
        self.handle.drain_agents();
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn read_certificate_der(path: &PathBuf) -> Result<Vec<u8>, String> {
    let pem = read_required(path)?;
    let certificates = pem::parse_many(pem)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
    certificates
        .into_iter()
        .find(|block| block.tag() == "CERTIFICATE")
        .map(|block| block.into_contents())
        .ok_or_else(|| format!("{} contains no CERTIFICATE", path.display()))
}

fn read_required(path: &PathBuf) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))
}
