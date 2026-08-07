use crate::wire::measure_fleet_client::MeasureFleetClient;
use crate::wire::{agent_event, coordinator_command};
use crate::{
    AgentTlsConfig, ArtifactDigest, DeviceId, DeviceProfileHash, InsecureTransport, JobId, LeaseId,
    MEASUREMENT_PROTOCOL_VERSION, PROTOCOL_MAJOR, PROTOCOL_MINOR, ServiceError, ServiceResult,
    SessionId, artifact_descriptor, device_profile_hash, wire,
};
use gz_engine::{GraphArtifact, GraphArtifactFormat};
use prost::Message;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Request;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Identity};

const MAX_BOUNDED_STRING_BYTES: usize = 512;

#[derive(Clone, Debug)]
enum AgentSecurity {
    MutualTls(AgentTlsConfig),
    InsecureForTests,
}

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub endpoint: String,
    pub device_id: DeviceId,
    pub profile: wire::DeviceProfile,
    pub agent_build: String,
    pub state_dir: PathBuf,
    pub control_capacity: usize,
    pub rpc_timeout: Duration,
    security: AgentSecurity,
}

impl AgentConfig {
    #[must_use]
    pub fn mutual_tls(
        endpoint: impl Into<String>,
        device_id: DeviceId,
        profile: wire::DeviceProfile,
        agent_build: impl Into<String>,
        state_dir: PathBuf,
        tls: AgentTlsConfig,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            device_id,
            profile,
            agent_build: agent_build.into(),
            state_dir,
            control_capacity: 8,
            rpc_timeout: Duration::from_secs(30),
            security: AgentSecurity::MutualTls(tls),
        }
    }

    #[must_use]
    pub fn insecure_for_tests(
        endpoint: impl Into<String>,
        device_id: DeviceId,
        profile: wire::DeviceProfile,
        agent_build: impl Into<String>,
        state_dir: PathBuf,
        _insecure: InsecureTransport,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            device_id,
            profile,
            agent_build: agent_build.into(),
            state_dir,
            control_capacity: 8,
            rpc_timeout: Duration::from_secs(30),
            security: AgentSecurity::InsecureForTests,
        }
    }

    pub fn validate(&self) -> ServiceResult<()> {
        if self.endpoint.is_empty() {
            return Err(ServiceError::configuration("agent endpoint is empty"));
        }
        if self.agent_build.is_empty() || self.agent_build.len() > 128 {
            return Err(ServiceError::configuration(
                "agent_build must contain 1..=128 bytes",
            ));
        }
        if self.control_capacity == 0 {
            return Err(ServiceError::configuration(
                "agent control capacity must be greater than zero",
            ));
        }
        if self.rpc_timeout.is_zero() {
            return Err(ServiceError::configuration(
                "agent RPC timeout must be greater than zero",
            ));
        }
        if self.profile.measurement_protocol_version != MEASUREMENT_PROTOCOL_VERSION {
            return Err(ServiceError::configuration(
                "agent device profile has the wrong measurement protocol version",
            ));
        }
        match &self.security {
            AgentSecurity::MutualTls(tls) => {
                if !self.endpoint.starts_with("https://") {
                    return Err(ServiceError::configuration(
                        "mTLS agent endpoint must use https",
                    ));
                }
                if tls.server_name.is_empty() {
                    return Err(ServiceError::configuration("TLS server name is empty"));
                }
            }
            AgentSecurity::InsecureForTests => {
                if !self.endpoint.starts_with("http://") {
                    return Err(ServiceError::configuration(
                        "insecure test endpoint must use http",
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct FetchedSubject {
    pub logical_index: u32,
    pub graph_hash: [u8; 32],
    pub format_kind: wire::GraphArtifactFormatKind,
    pub adapter_format_id: u32,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct AgentMeasurement {
    pub subjects: Vec<wire::SubjectMeasurement>,
    pub samples: Vec<wire::MeasurementSample>,
    pub telemetry: Vec<wire::TelemetrySample>,
}

#[derive(Clone, Debug)]
pub struct AgentBackendError {
    pub outcome: wire::MeasureAttemptOutcome,
    pub code: u32,
    pub message: String,
    pub retriable: bool,
}

impl AgentBackendError {
    #[must_use]
    pub fn new(
        outcome: wire::MeasureAttemptOutcome,
        code: u32,
        message: impl Into<String>,
        retriable: bool,
    ) -> Self {
        Self {
            outcome,
            code,
            message: message.into(),
            retriable,
        }
    }
}

pub trait AgentBackend: Send + Sync + 'static {
    fn capability(&self) -> wire::EngineCapability;

    fn concurrent_job_slots(&self) -> u32 {
        1
    }

    fn measure(
        &self,
        lease: &wire::MeasureLease,
        subjects: Vec<FetchedSubject>,
    ) -> Result<AgentMeasurement, AgentBackendError>;
}

pub struct AgentRuntime<B> {
    config: AgentConfig,
    backend: Arc<B>,
    job_slots: u32,
}

impl<B> AgentRuntime<B>
where
    B: AgentBackend,
{
    pub fn new(config: AgentConfig, backend: B) -> ServiceResult<Self> {
        config.validate()?;
        let job_slots = backend.concurrent_job_slots();
        if job_slots == 0 {
            return Err(ServiceError::configuration(
                "agent backend job slots must be greater than zero",
            ));
        }
        if usize::try_from(job_slots)
            .ok()
            .is_none_or(|slots| slots > config.control_capacity)
        {
            return Err(ServiceError::configuration(
                "agent backend job slots exceed the control queue capacity",
            ));
        }
        Ok(Self {
            config,
            backend: Arc::new(backend),
            job_slots,
        })
    }

    pub async fn run_session(&self) -> ServiceResult<()> {
        let job_slots = self.job_slots;
        let persisted_reports = load_persisted_reports(&self.config, job_slots as usize).await?;
        let channel = connect_channel(&self.config).await?;
        let mut client = MeasureFleetClient::new(channel);
        let data_client = client.clone();
        let (event_tx, event_rx) = mpsc::channel(self.config.control_capacity);
        event_tx
            .try_send(wire::AgentEvent {
                event: Some(agent_event::Event::Hello(wire::AgentHello {
                    protocol_major: PROTOCOL_MAJOR,
                    protocol_minor: PROTOCOL_MINOR,
                    device_id: self.config.device_id.to_vec(),
                    agent_build: self.config.agent_build.clone(),
                    profile: Some(self.config.profile.clone()),
                    capabilities: vec![self.backend.capability()],
                    recoveries: persisted_reports
                        .iter()
                        .map(|persisted| wire::LeaseRecovery {
                            job_id: persisted.report.job_id.clone(),
                            lease_id: persisted.report.lease_id.clone(),
                            phase: wire::JobPhase::SubmitReport as i32,
                            report_persisted: true,
                        })
                        .collect(),
                    job_slots,
                })),
            })
            .map_err(|_| ServiceError::capacity("agent control queue is full"))?;
        let response = client
            .connect(ReceiverStream::new(event_rx))
            .await
            .map_err(status_error)?;
        let mut commands = response.into_inner();
        let first = commands
            .message()
            .await
            .map_err(status_error)?
            .ok_or(ServiceError::Closed)?;
        let welcome = match first.command {
            Some(coordinator_command::Command::Welcome(welcome)) => welcome,
            Some(coordinator_command::Command::Error(error)) => {
                return Err(ServiceError::protocol(error.bounded_message));
            }
            _ => return Err(ServiceError::protocol("first command was not AgentWelcome")),
        };
        let session_id = validate_welcome(&self.config, &welcome)?;
        reconcile_persisted_reports(
            &self.config,
            &mut data_client.clone(),
            session_id,
            &welcome.recoveries,
            persisted_reports,
        )
        .await?;
        send_ready(&event_tx, session_id, job_slots).await?;
        let heartbeat_interval = Duration::from_millis(welcome.heartbeat_interval_ms);
        let config = Arc::new(self.config.clone());
        let welcome = Arc::new(welcome);
        let mut heartbeat = tokio::time::interval(heartbeat_interval);
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        heartbeat.tick().await;
        let mut tasks = tokio::task::JoinSet::new();
        let mut active = HashMap::<JobId, LeaseId>::with_capacity(job_slots as usize);
        let mut draining = false;

        loop {
            tokio::select! {
                command = commands.message() => {
                    let Some(command) = command.map_err(status_error)? else {
                        return Err(ServiceError::Closed);
                    };
                    match command.command {
                        Some(coordinator_command::Command::Lease(lease)) => {
                            let available = job_slots.saturating_sub(active.len() as u32);
                            if draining || available == 0 {
                                reject_lease(
                                    &event_tx,
                                    session_id,
                                    &lease,
                                    wire::JobRejectReason::JobRejectNotReady,
                                    "agent has no free measurement slot".to_owned(),
                                )
                                .await?;
                                send_ready(
                                    &event_tx,
                                    session_id,
                                    if draining { 0 } else { available },
                                )
                                .await?;
                                continue;
                            }
                            if let Err(error) = validate_lease(
                                &config,
                                &welcome,
                                &self.backend.capability(),
                                session_id,
                                &lease,
                            ) {
                                reject_lease(
                                    &event_tx,
                                    session_id,
                                    &lease,
                                    wire::JobRejectReason::JobRejectUnsupportedConfig,
                                    error.to_string(),
                                )
                                .await?;
                                send_ready(&event_tx, session_id, available).await?;
                                continue;
                            }
                            let job_id = JobId::from_slice(&lease.job_id)?;
                            let lease_id = LeaseId::from_slice(&lease.lease_id)?;
                            if active.insert(job_id, lease_id).is_some() {
                                return Err(ServiceError::protocol(
                                    "coordinator offered the same active job twice",
                                ));
                            }
                            let task_config = Arc::clone(&config);
                            let task_backend = Arc::clone(&self.backend);
                            let task_client = data_client.clone();
                            let task_events = event_tx.clone();
                            let task_welcome = Arc::clone(&welcome);
                            tasks.spawn(async move {
                                let result = Self::process_lease(
                                    task_config,
                                    task_backend,
                                    task_client,
                                    task_events,
                                    task_welcome,
                                    session_id,
                                    lease,
                                )
                                .await;
                                (job_id, lease_id, result)
                            });
                        }
                        Some(coordinator_command::Command::Cancel(_)) => {}
                        Some(coordinator_command::Command::Drain(_)) => {
                            draining = true;
                            send_draining(&event_tx, session_id).await?;
                            if active.is_empty() {
                                return Ok(());
                            }
                        }
                        Some(coordinator_command::Command::Error(error)) => {
                            return Err(ServiceError::protocol(error.bounded_message));
                        }
                        Some(coordinator_command::Command::Welcome(_)) => {
                            return Err(ServiceError::protocol(
                                "AgentWelcome appeared more than once",
                            ));
                        }
                        None => {
                            return Err(ServiceError::protocol(
                                "empty coordinator command",
                            ));
                        }
                    }
                }
                completed = tasks.join_next(), if !active.is_empty() => {
                    let (job_id, lease_id, result) = completed
                        .ok_or_else(|| {
                            ServiceError::transport(
                                "agent measurement task set ended unexpectedly",
                            )
                        })?
                        .map_err(|error| ServiceError::transport(error.to_string()))?;
                    if active.remove(&job_id) != Some(lease_id) {
                        return Err(ServiceError::protocol(
                            "completed measurement task was not active",
                        ));
                    }
                    result?;
                    if draining && active.is_empty() {
                        return Ok(());
                    }
                    if !draining {
                        send_ready(
                            &event_tx,
                            session_id,
                            job_slots.saturating_sub(active.len() as u32),
                        )
                        .await?;
                    }
                }
                _ = heartbeat.tick() => {
                    send_ready(
                        &event_tx,
                        session_id,
                        if draining {
                            0
                        } else {
                            job_slots.saturating_sub(active.len() as u32)
                        },
                    )
                    .await?;
                }
            }
        }
    }

    async fn process_lease(
        config: Arc<AgentConfig>,
        backend: Arc<B>,
        mut client: MeasureFleetClient<Channel>,
        event_tx: mpsc::Sender<wire::AgentEvent>,
        welcome: Arc<wire::AgentWelcome>,
        session_id: SessionId,
        lease: wire::MeasureLease,
    ) -> ServiceResult<()> {
        let job_id = JobId::from_slice(&lease.job_id)?;
        let lease_id = LeaseId::from_slice(&lease.lease_id)?;
        event_tx
            .send(wire::AgentEvent {
                event: Some(agent_event::Event::Accepted(wire::JobAccepted {
                    session_id: session_id.to_vec(),
                    job_id: job_id.to_vec(),
                    lease_id: lease_id.to_vec(),
                })),
            })
            .await
            .map_err(|_| ServiceError::Closed)?;
        send_heartbeat(
            &event_tx,
            session_id,
            job_id,
            lease_id,
            wire::JobPhase::FetchArtifact,
            0,
        )
        .await?;

        let attempt_started = Instant::now();
        let mut subjects = Vec::with_capacity(lease.subjects.len());
        for subject in &lease.subjects {
            subjects.push(
                fetch_subject(
                    &mut client,
                    config.rpc_timeout,
                    welcome.max_artifact_bytes,
                    session_id,
                    job_id,
                    lease_id,
                    subject,
                )
                .await?,
            );
        }
        let fetch_elapsed_ms =
            u64::try_from(attempt_started.elapsed().as_millis()).unwrap_or(u64::MAX);
        send_heartbeat(
            &event_tx,
            session_id,
            job_id,
            lease_id,
            wire::JobPhase::VerifyArtifact,
            fetch_elapsed_ms,
        )
        .await?;

        let backend_lease = lease.clone();
        let measurement =
            tokio::task::spawn_blocking(move || backend.measure(&backend_lease, subjects));
        tokio::pin!(measurement);
        let heartbeat = Duration::from_millis(welcome.heartbeat_interval_ms);
        let mut measured = loop {
            tokio::select! {
                result = &mut measurement => {
                    break result.map_err(|error| ServiceError::transport(error.to_string()))?;
                }
                _ = tokio::time::sleep(heartbeat) => {
                    send_heartbeat(
                        &event_tx,
                        session_id,
                        job_id,
                        lease_id,
                        wire::JobPhase::Measure,
                        u64::try_from(attempt_started.elapsed().as_millis())
                            .unwrap_or(u64::MAX),
                    ).await?;
                }
            }
        };
        if let Ok(measurement) = &mut measured {
            for sample in &mut measurement.telemetry {
                sample.attempt_elapsed_ms =
                    sample.attempt_elapsed_ms.saturating_add(fetch_elapsed_ms);
            }
            measurement.telemetry.insert(
                0,
                wire::TelemetrySample {
                    attempt_elapsed_ms: fetch_elapsed_ms,
                    phase: wire::JobPhase::FetchArtifact as i32,
                    telemetry: None,
                },
            );
        }

        let report = build_report(
            &config,
            session_id,
            device_profile_hash(&config.profile),
            &lease,
            measured,
        )?;
        send_heartbeat(
            &event_tx,
            session_id,
            job_id,
            lease_id,
            wire::JobPhase::PersistReport,
            u64::try_from(attempt_started.elapsed().as_millis()).unwrap_or(u64::MAX),
        )
        .await?;
        let report_path = persist_report(&config.state_dir, &report).await?;
        send_heartbeat(
            &event_tx,
            session_id,
            job_id,
            lease_id,
            wire::JobPhase::SubmitReport,
            u64::try_from(attempt_started.elapsed().as_millis()).unwrap_or(u64::MAX),
        )
        .await?;

        submit_persisted_report(
            &mut client,
            config.rpc_timeout,
            &config.state_dir,
            session_id,
            report,
            report_path,
        )
        .await?;
        Ok(())
    }
}

struct PersistedReport {
    path: PathBuf,
    report: wire::MeasureReport,
}

async fn load_persisted_reports(
    config: &AgentConfig,
    max_reports: usize,
) -> ServiceResult<Vec<PersistedReport>> {
    let state_dir = config.state_dir.clone();
    let device_id = config.device_id;
    let profile_hash = device_profile_hash(&config.profile);
    tokio::task::spawn_blocking(move || {
        std::fs::create_dir_all(&state_dir).map_err(|error| ServiceError::io(error.to_string()))?;
        let mut persisted = Vec::new();
        for entry in
            std::fs::read_dir(&state_dir).map_err(|error| ServiceError::io(error.to_string()))?
        {
            let entry = entry.map_err(|error| ServiceError::io(error.to_string()))?;
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("report") {
                continue;
            }
            let bytes =
                std::fs::read(&path).map_err(|error| ServiceError::io(error.to_string()))?;
            let report = wire::MeasureReport::decode(bytes.as_slice())
                .map_err(|error| ServiceError::protocol(error.to_string()))?;
            if DeviceId::from_slice(&report.device_id)? != device_id
                || DeviceProfileHash::from_slice(&report.device_profile_hash)? != profile_hash
            {
                return Err(ServiceError::protocol(
                    "persisted report belongs to a different agent profile",
                ));
            }
            JobId::from_slice(&report.job_id)?;
            LeaseId::from_slice(&report.lease_id)?;
            persisted.push(PersistedReport { path, report });
        }
        if persisted.len() > max_reports {
            return Err(ServiceError::protocol(
                "agent found more persisted reports than measurement slots",
            ));
        }
        Ok(persisted)
    })
    .await
    .map_err(|error| ServiceError::io(error.to_string()))?
}

async fn reconcile_persisted_reports(
    config: &AgentConfig,
    client: &mut MeasureFleetClient<Channel>,
    session_id: SessionId,
    dispositions: &[wire::RecoveryDisposition],
    mut persisted: Vec<PersistedReport>,
) -> ServiceResult<()> {
    if dispositions.len() != persisted.len() {
        return Err(ServiceError::protocol(
            "coordinator returned the wrong recovery disposition count",
        ));
    }
    for disposition in dispositions {
        let position = persisted.iter().position(|candidate| {
            candidate.report.job_id == disposition.job_id
                && candidate.report.lease_id == disposition.lease_id
        });
        let Some(position) = position else {
            return Err(ServiceError::protocol(
                "coordinator returned a disposition for an unknown recovery",
            ));
        };
        let persisted_report = persisted.swap_remove(position);
        let action = wire::RecoveryAction::try_from(disposition.action)
            .map_err(|_| ServiceError::protocol("unknown recovery action"))?;
        match action {
            wire::RecoveryAction::RecoverySubmitReport => {
                submit_persisted_report(
                    client,
                    config.rpc_timeout,
                    &config.state_dir,
                    session_id,
                    persisted_report.report,
                    persisted_report.path,
                )
                .await?;
            }
            wire::RecoveryAction::RecoveryAlreadyCommitted
            | wire::RecoveryAction::RecoveryCancel => {
                delete_report(&config.state_dir, persisted_report.path).await?;
            }
            wire::RecoveryAction::RecoveryContinue | wire::RecoveryAction::RecoveryUnspecified => {
                return Err(ServiceError::protocol(
                    "coordinator returned an invalid action for a persisted report",
                ));
            }
        }
    }
    Ok(())
}

async fn submit_persisted_report(
    client: &mut MeasureFleetClient<Channel>,
    rpc_timeout: Duration,
    state_dir: &Path,
    current_session_id: SessionId,
    report: wire::MeasureReport,
    report_path: PathBuf,
) -> ServiceResult<()> {
    let mut request = Request::new(wire::SubmitResultRequest {
        current_session_id: current_session_id.to_vec(),
        report: Some(report),
    });
    request.set_timeout(rpc_timeout);
    let ack = client
        .submit_result(request)
        .await
        .map_err(status_error)?
        .into_inner();
    let disposition = wire::ResultDisposition::try_from(ack.disposition)
        .map_err(|_| ServiceError::protocol("unknown result disposition"))?;
    if !matches!(
        disposition,
        wire::ResultDisposition::ResultJobCommitted
            | wire::ResultDisposition::ResultAlreadyCommitted
    ) {
        return Err(ServiceError::protocol(format!(
            "coordinator rejected result: {}",
            ack.bounded_message
        )));
    }
    if ack.agent_may_delete_report {
        delete_report(state_dir, report_path).await?;
    }
    Ok(())
}

async fn delete_report(state_dir: &Path, report_path: PathBuf) -> ServiceResult<()> {
    let state_dir = state_dir.to_owned();
    tokio::task::spawn_blocking(move || {
        std::fs::remove_file(report_path).map_err(|error| ServiceError::io(error.to_string()))?;
        File::open(state_dir)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| ServiceError::io(error.to_string()))
    })
    .await
    .map_err(|error| ServiceError::io(error.to_string()))?
}

async fn connect_channel(config: &AgentConfig) -> ServiceResult<Channel> {
    let mut endpoint = Endpoint::from_shared(config.endpoint.clone())
        .map_err(|error| ServiceError::configuration(error.to_string()))?
        .connect_timeout(config.rpc_timeout);
    if let AgentSecurity::MutualTls(tls) = &config.security {
        endpoint = endpoint
            .tls_config(
                ClientTlsConfig::new()
                    .ca_certificate(Certificate::from_pem(&tls.server_ca_pem))
                    .identity(Identity::from_pem(
                        &tls.certificate_pem,
                        &tls.private_key_pem,
                    ))
                    .domain_name(tls.server_name.clone()),
            )
            .map_err(|error| ServiceError::configuration(error.to_string()))?;
    }
    endpoint
        .connect()
        .await
        .map_err(|error| ServiceError::transport(error.to_string()))
}

fn validate_welcome(
    config: &AgentConfig,
    welcome: &wire::AgentWelcome,
) -> ServiceResult<SessionId> {
    let session_id = SessionId::from_slice(&welcome.session_id)?;
    if welcome.negotiated_minor != PROTOCOL_MINOR {
        return Err(ServiceError::protocol(
            "unexpected negotiated minor version",
        ));
    }
    let expected_profile = device_profile_hash(&config.profile);
    if DeviceProfileHash::from_slice(&welcome.device_profile_hash)? != expected_profile {
        return Err(ServiceError::protocol(
            "coordinator changed device profile hash",
        ));
    }
    for (name, value) in [
        ("heartbeat_interval_ms", welcome.heartbeat_interval_ms),
        ("accept_timeout_ms", welcome.accept_timeout_ms),
        (
            "default_lease_duration_ms",
            welcome.default_lease_duration_ms,
        ),
        ("max_artifact_bytes", welcome.max_artifact_bytes),
        (
            "artifact_chunk_bytes",
            u64::from(welcome.artifact_chunk_bytes),
        ),
        (
            "max_measure_config_bytes",
            u64::from(welcome.max_measure_config_bytes),
        ),
        (
            "max_engine_metadata_bytes",
            u64::from(welcome.max_engine_metadata_bytes),
        ),
    ] {
        if value == 0 {
            return Err(ServiceError::protocol(format!(
                "welcome limit {name} is zero"
            )));
        }
    }
    if u64::from(welcome.artifact_chunk_bytes) > welcome.max_artifact_bytes {
        return Err(ServiceError::protocol(
            "artifact chunk limit exceeds artifact limit",
        ));
    }
    Ok(session_id)
}

fn validate_lease(
    config: &AgentConfig,
    welcome: &wire::AgentWelcome,
    capability: &wire::EngineCapability,
    session_id: SessionId,
    lease: &wire::MeasureLease,
) -> ServiceResult<()> {
    if SessionId::from_slice(&lease.session_id)? != session_id {
        return Err(ServiceError::protocol("lease session id mismatch"));
    }
    JobId::from_slice(&lease.job_id)?;
    crate::MeasurementKey::from_slice(&lease.measurement_key)?;
    crate::RequestNonce::from_slice(&lease.request_nonce)?;
    LeaseId::from_slice(&lease.lease_id)?;
    if DeviceProfileHash::from_slice(&lease.target_device_profile_hash)?
        != device_profile_hash(&config.profile)
    {
        return Err(ServiceError::protocol("lease device profile mismatch"));
    }
    let kind = wire::MeasureKind::try_from(lease.kind)
        .map_err(|_| ServiceError::protocol("lease measurement kind is unknown"))?;
    let expected_subjects = match kind {
        wire::MeasureKind::Single => 1,
        wire::MeasureKind::SymmetricPair => 2,
        wire::MeasureKind::Unspecified => {
            return Err(ServiceError::protocol(
                "lease measurement kind is unspecified",
            ));
        }
    };
    if lease.subjects.len() != expected_subjects {
        return Err(ServiceError::protocol(
            "lease measurement kind has the wrong subject count",
        ));
    }
    if lease.measure_config_payload.len() > welcome.max_measure_config_bytes as usize {
        return Err(ServiceError::capacity("lease measure config is too large"));
    }
    if lease.engine.as_ref() != capability.engine.as_ref()
        || !capability
            .measure_config_encodings
            .contains(&lease.measure_config_encoding)
        || !capability
            .measurement_protocol_versions
            .contains(&MEASUREMENT_PROTOCOL_VERSION)
    {
        return Err(ServiceError::protocol("lease engine capability mismatch"));
    }
    for (logical_index, subject) in lease.subjects.iter().enumerate() {
        if subject.logical_index != logical_index as u32 || subject.graph_hash.len() != 32 {
            return Err(ServiceError::protocol("lease subject is malformed"));
        }
        let artifact = subject
            .artifact
            .as_ref()
            .ok_or_else(|| ServiceError::protocol("lease artifact reference is missing"))?;
        ArtifactDigest::from_slice(&artifact.artifact_digest)?;
        if artifact.artifact_size > welcome.max_artifact_bytes {
            return Err(ServiceError::capacity("lease artifact is too large"));
        }
        let supported = capability.artifact_formats.iter().any(|supported| {
            supported.format_kind == artifact.format_kind
                && supported.adapter_format_id == artifact.adapter_format_id
        });
        if !supported {
            return Err(ServiceError::protocol(
                "lease artifact format is unsupported",
            ));
        }
    }
    Ok(())
}

async fn reject_lease(
    event_tx: &mpsc::Sender<wire::AgentEvent>,
    session_id: SessionId,
    lease: &wire::MeasureLease,
    reason: wire::JobRejectReason,
    message: String,
) -> ServiceResult<()> {
    event_tx
        .send(wire::AgentEvent {
            event: Some(agent_event::Event::Rejected(wire::JobRejected {
                session_id: session_id.to_vec(),
                job_id: lease.job_id.clone(),
                lease_id: lease.lease_id.clone(),
                reason: reason as i32,
                bounded_message: bounded_utf8(message, MAX_BOUNDED_STRING_BYTES),
            })),
        })
        .await
        .map_err(|_| ServiceError::Closed)
}

async fn fetch_subject(
    client: &mut MeasureFleetClient<Channel>,
    rpc_timeout: Duration,
    max_artifact_bytes: u64,
    session_id: SessionId,
    job_id: JobId,
    lease_id: LeaseId,
    subject: &wire::GraphSubject,
) -> ServiceResult<FetchedSubject> {
    let reference = subject
        .artifact
        .as_ref()
        .ok_or_else(|| ServiceError::protocol("artifact reference is missing"))?;
    let digest = ArtifactDigest::from_slice(&reference.artifact_digest)?;
    if reference.artifact_size > max_artifact_bytes {
        return Err(ServiceError::capacity("artifact exceeds negotiated limit"));
    }
    let expected_size = usize::try_from(reference.artifact_size)
        .map_err(|_| ServiceError::capacity("artifact size does not fit usize"))?;
    let format_kind = wire::GraphArtifactFormatKind::try_from(reference.format_kind)
        .map_err(|_| ServiceError::protocol("unknown artifact format"))?;
    let mut request = Request::new(wire::FetchArtifactRequest {
        session_id: session_id.to_vec(),
        job_id: job_id.to_vec(),
        lease_id: lease_id.to_vec(),
        artifact_digest: digest.to_vec(),
        offset: 0,
    });
    request.set_timeout(rpc_timeout);
    let mut chunks = client
        .fetch_artifact(request)
        .await
        .map_err(status_error)?
        .into_inner();
    let mut bytes = Vec::with_capacity(expected_size);
    while let Some(chunk) = chunks.message().await.map_err(status_error)? {
        if ArtifactDigest::from_slice(&chunk.artifact_digest)? != digest
            || chunk.total_size != reference.artifact_size
            || chunk.offset != bytes.len() as u64
        {
            return Err(ServiceError::protocol("artifact chunk identity mismatch"));
        }
        let next_len = bytes
            .len()
            .checked_add(chunk.data.len())
            .ok_or_else(|| ServiceError::capacity("artifact length overflow"))?;
        if next_len > expected_size {
            return Err(ServiceError::protocol(
                "artifact stream exceeds declared size",
            ));
        }
        bytes.extend_from_slice(&chunk.data);
    }
    if bytes.len() != expected_size {
        return Err(ServiceError::protocol("artifact stream ended early"));
    }
    let format = match format_kind {
        wire::GraphArtifactFormatKind::GraphArtifactFormatText => GraphArtifactFormat::Text,
        wire::GraphArtifactFormatKind::GraphArtifactFormatJson => GraphArtifactFormat::Json,
        wire::GraphArtifactFormatKind::GraphArtifactFormatDot => GraphArtifactFormat::Dot,
        wire::GraphArtifactFormatKind::GraphArtifactFormatBinary => GraphArtifactFormat::Binary,
        wire::GraphArtifactFormatKind::GraphArtifactFormatAdapterSpecific => {
            GraphArtifactFormat::AdapterSpecific(reference.adapter_format_id)
        }
        wire::GraphArtifactFormatKind::GraphArtifactFormatUnspecified => {
            return Err(ServiceError::protocol("artifact format is unspecified"));
        }
    };
    let artifact =
        GraphArtifact {
            graph_hash: gz_engine::GraphHash::from_bytes(
                subject.graph_hash.as_slice().try_into().map_err(|_| {
                    ServiceError::protocol("subject graph hash has the wrong width")
                })?,
            ),
            format,
            bytes: bytes.clone(),
        };
    if artifact_descriptor(&artifact)?.digest != digest {
        return Err(ServiceError::protocol(
            "artifact digest verification failed",
        ));
    }
    Ok(FetchedSubject {
        logical_index: subject.logical_index,
        graph_hash: subject
            .graph_hash
            .as_slice()
            .try_into()
            .map_err(|_| ServiceError::protocol("subject graph hash has the wrong width"))?,
        format_kind,
        adapter_format_id: reference.adapter_format_id,
        bytes,
    })
}

fn build_report(
    config: &AgentConfig,
    session_id: SessionId,
    profile_hash: DeviceProfileHash,
    lease: &wire::MeasureLease,
    measured: Result<AgentMeasurement, AgentBackendError>,
) -> ServiceResult<wire::MeasureReport> {
    let (outcome, measurement, failure) = match measured {
        Ok(measurement) => (
            wire::MeasureAttemptOutcome::MeasureOutcomeSucceeded,
            measurement,
            None,
        ),
        Err(error) => {
            if error.outcome == wire::MeasureAttemptOutcome::MeasureOutcomeSucceeded
                || error.outcome == wire::MeasureAttemptOutcome::MeasureOutcomeUnspecified
            {
                return Err(ServiceError::protocol(
                    "backend failure used a non-failure outcome",
                ));
            }
            let subjects = lease
                .subjects
                .iter()
                .map(|subject| wire::SubjectMeasurement {
                    logical_index: subject.logical_index,
                    graph_hash: subject.graph_hash.clone(),
                    compile_elapsed_ns: None,
                    capture_elapsed_ns: None,
                    scalar_reward: None,
                    engine_metadata: Vec::new(),
                })
                .collect();
            (
                error.outcome,
                AgentMeasurement {
                    subjects,
                    samples: Vec::new(),
                    telemetry: Vec::new(),
                },
                Some(wire::FailureDetail {
                    engine_error_code: error.code,
                    bounded_message: bounded_utf8(error.message, MAX_BOUNDED_STRING_BYTES),
                    agent_considers_retriable: error.retriable,
                }),
            )
        }
    };
    validate_backend_measurement(lease, &measurement)?;
    Ok(wire::MeasureReport {
        origin_session_id: session_id.to_vec(),
        device_id: config.device_id.to_vec(),
        device_profile_hash: profile_hash.to_vec(),
        job_id: lease.job_id.clone(),
        measurement_key: lease.measurement_key.clone(),
        request_nonce: lease.request_nonce.clone(),
        lease_id: lease.lease_id.clone(),
        engine: lease.engine.clone(),
        measure_config_hash: lease.measure_config_hash.clone(),
        kind: lease.kind,
        outcome: outcome as i32,
        subjects: measurement.subjects,
        samples: measurement.samples,
        telemetry: measurement.telemetry,
        failure,
    })
}

fn validate_backend_measurement(
    lease: &wire::MeasureLease,
    measurement: &AgentMeasurement,
) -> ServiceResult<()> {
    if measurement.subjects.len() != lease.subjects.len() {
        return Err(ServiceError::protocol(
            "backend returned the wrong subject count",
        ));
    }
    for (expected, actual) in lease.subjects.iter().zip(&measurement.subjects) {
        if expected.logical_index != actual.logical_index
            || expected.graph_hash != actual.graph_hash
        {
            return Err(ServiceError::protocol(
                "backend returned a different subject identity",
            ));
        }
        if actual
            .scalar_reward
            .is_some_and(|reward| !reward.is_finite())
        {
            return Err(ServiceError::protocol(
                "backend returned a non-finite reward",
            ));
        }
    }
    Ok(())
}

fn bounded_utf8(message: String, max_bytes: usize) -> String {
    let mut bounded = String::new();
    for character in message.chars() {
        if bounded.len() + character.len_utf8() > max_bytes {
            break;
        }
        bounded.push(character);
    }
    bounded
}

async fn persist_report(state_dir: &Path, report: &wire::MeasureReport) -> ServiceResult<PathBuf> {
    let state_dir = state_dir.to_owned();
    let report = report.clone();
    tokio::task::spawn_blocking(move || persist_report_blocking(&state_dir, &report))
        .await
        .map_err(|error| ServiceError::io(error.to_string()))?
}

fn persist_report_blocking(
    state_dir: &Path,
    report: &wire::MeasureReport,
) -> ServiceResult<PathBuf> {
    std::fs::create_dir_all(state_dir).map_err(|error| ServiceError::io(error.to_string()))?;
    let job = JobId::from_slice(&report.job_id)?;
    let lease = LeaseId::from_slice(&report.lease_id)?;
    let final_path = state_dir.join(format!("{job}-{lease}.report"));
    let temporary_path = state_dir.join(format!(".{job}-{lease}.tmp"));
    let encoded = report.encode_to_vec();

    if final_path.exists() {
        let existing =
            std::fs::read(&final_path).map_err(|error| ServiceError::io(error.to_string()))?;
        if existing == encoded {
            return Ok(final_path);
        }
        return Err(ServiceError::protocol(
            "persisted report differs from new report",
        ));
    }

    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temporary_path)
        .map_err(|error| ServiceError::io(error.to_string()))?;
    file.write_all(&encoded)
        .and_then(|()| file.sync_all())
        .map_err(|error| ServiceError::io(error.to_string()))?;
    std::fs::rename(&temporary_path, &final_path)
        .map_err(|error| ServiceError::io(error.to_string()))?;
    File::open(state_dir)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| ServiceError::io(error.to_string()))?;
    Ok(final_path)
}

async fn send_ready(
    event_tx: &mpsc::Sender<wire::AgentEvent>,
    session_id: SessionId,
    free_slots: u32,
) -> ServiceResult<()> {
    event_tx
        .send(wire::AgentEvent {
            event: Some(agent_event::Event::Ready(wire::AgentReady {
                session_id: session_id.to_vec(),
                free_slots,
                telemetry: None,
            })),
        })
        .await
        .map_err(|_| ServiceError::Closed)
}

async fn send_draining(
    event_tx: &mpsc::Sender<wire::AgentEvent>,
    session_id: SessionId,
) -> ServiceResult<()> {
    event_tx
        .send(wire::AgentEvent {
            event: Some(agent_event::Event::Draining(wire::AgentDraining {
                session_id: session_id.to_vec(),
            })),
        })
        .await
        .map_err(|_| ServiceError::Closed)
}

async fn send_heartbeat(
    event_tx: &mpsc::Sender<wire::AgentEvent>,
    session_id: SessionId,
    job_id: JobId,
    lease_id: LeaseId,
    phase: wire::JobPhase,
    phase_elapsed_ms: u64,
) -> ServiceResult<()> {
    event_tx
        .send(wire::AgentEvent {
            event: Some(agent_event::Event::Heartbeat(wire::JobHeartbeat {
                session_id: session_id.to_vec(),
                job_id: job_id.to_vec(),
                lease_id: lease_id.to_vec(),
                phase: phase as i32,
                phase_elapsed_ms,
                telemetry: None,
            })),
        })
        .await
        .map_err(|_| ServiceError::Closed)
}

fn status_error(error: tonic::Status) -> ServiceError {
    ServiceError::transport(error.to_string())
}
