use crate::protocol::fixed_bytes;
use crate::receipt::{ReceiptLedger, ReceiptState};
use crate::wire::measure_fleet_server::MeasureFleet;
use crate::wire::{agent_event, coordinator_command};
use crate::{
    ArtifactDescriptor, ArtifactDigest, CertificateFingerprint, CommittedMeasurement, DeviceId,
    DeviceProfileHash, JobId, LeaseId, MeasureSubmission, MeasurementKey, PROTOCOL_MAJOR,
    PROTOCOL_MINOR, RequestNonce, ServiceError, ServiceResult, SessionId, artifact_descriptor,
    certificate_fingerprint, device_profile_hash, engine_identity_to_wire, job_id, measurement_key,
    wire,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};
use tonic::{Request, Response, Status, Streaming};

const MAX_PROFILE_STRING_BYTES: usize = 128;
const MAX_AGENT_BUILD_BYTES: usize = 128;
type RecoveryReconciliation = (Vec<wire::RecoveryDisposition>, Option<(JobId, LeaseId)>);

#[derive(Clone, Debug)]
pub enum Enrollment {
    MutualTls(BTreeMap<CertificateFingerprint, DeviceId>),
    InsecureForTests(BTreeSet<DeviceId>),
}

impl Enrollment {
    #[must_use]
    pub fn one_mutual_tls(fingerprint: CertificateFingerprint, device_id: DeviceId) -> Self {
        Self::MutualTls(BTreeMap::from([(fingerprint, device_id)]))
    }

    #[must_use]
    pub fn one_insecure_test_device(device_id: DeviceId) -> Self {
        Self::InsecureForTests(BTreeSet::from([device_id]))
    }
}

#[derive(Clone, Debug)]
pub struct CoordinatorConfig {
    pub enrollment: Enrollment,
    pub queue_capacity: usize,
    pub artifact_item_capacity: usize,
    pub artifact_byte_capacity: usize,
    pub receipt_ledger: crate::ReceiptLedgerConfig,
    pub command_capacity: usize,
    pub max_artifact_bytes: usize,
    pub artifact_chunk_bytes: usize,
    pub max_measure_config_bytes: usize,
    pub max_engine_metadata_bytes: usize,
    pub max_samples_per_subject: usize,
    pub max_telemetry_samples: usize,
    pub heartbeat_interval: Duration,
    pub accept_timeout: Duration,
    pub lease_duration: Duration,
}

impl CoordinatorConfig {
    pub fn validate(&self) -> ServiceResult<()> {
        for (name, value) in [
            ("queue_capacity", self.queue_capacity),
            ("artifact_item_capacity", self.artifact_item_capacity),
            ("artifact_byte_capacity", self.artifact_byte_capacity),
            ("command_capacity", self.command_capacity),
            ("max_artifact_bytes", self.max_artifact_bytes),
            ("artifact_chunk_bytes", self.artifact_chunk_bytes),
            ("max_measure_config_bytes", self.max_measure_config_bytes),
            ("max_engine_metadata_bytes", self.max_engine_metadata_bytes),
            ("max_samples_per_subject", self.max_samples_per_subject),
            ("max_telemetry_samples", self.max_telemetry_samples),
        ] {
            if value == 0 {
                return Err(ServiceError::configuration(format!(
                    "{name} must be greater than zero"
                )));
            }
        }
        self.receipt_ledger.validate()?;
        if self.artifact_chunk_bytes > self.max_artifact_bytes {
            return Err(ServiceError::configuration(
                "artifact_chunk_bytes cannot exceed max_artifact_bytes",
            ));
        }
        for (name, value) in [
            ("heartbeat_interval", self.heartbeat_interval),
            ("accept_timeout", self.accept_timeout),
            ("lease_duration", self.lease_duration),
        ] {
            if value.is_zero() {
                return Err(ServiceError::configuration(format!(
                    "{name} must be greater than zero"
                )));
            }
        }
        if self.lease_duration <= self.heartbeat_interval {
            return Err(ServiceError::configuration(
                "lease_duration must exceed heartbeat_interval",
            ));
        }
        Ok(())
    }
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            enrollment: Enrollment::InsecureForTests(BTreeSet::new()),
            queue_capacity: 256,
            artifact_item_capacity: 256,
            artifact_byte_capacity: 256 * 1024 * 1024,
            receipt_ledger: crate::ReceiptLedgerConfig::default(),
            command_capacity: 8,
            max_artifact_bytes: 16 * 1024 * 1024,
            artifact_chunk_bytes: 256 * 1024,
            max_measure_config_bytes: 64 * 1024,
            max_engine_metadata_bytes: 64 * 1024,
            max_samples_per_subject: 4096,
            max_telemetry_samples: 4096,
            heartbeat_interval: Duration::from_secs(2),
            accept_timeout: Duration::from_secs(5),
            lease_duration: Duration::from_secs(15),
        }
    }
}

#[derive(Clone)]
pub struct Coordinator {
    inner: Arc<CoordinatorInner>,
}

#[derive(Clone)]
pub struct CoordinatorHandle {
    inner: Arc<CoordinatorInner>,
}

struct CoordinatorInner {
    config: CoordinatorConfig,
    state: Mutex<CoordinatorState>,
}

struct CoordinatorState {
    agents: BTreeMap<DeviceId, AgentSession>,
    jobs: HashMap<JobId, Job>,
    queue: VecDeque<JobId>,
    artifacts: HashMap<ArtifactDigest, ArtifactRecord>,
    artifact_bytes: usize,
    receipts: ReceiptLedger,
}

struct AgentSession {
    session_id: SessionId,
    profile_hash: DeviceProfileHash,
    capabilities: Vec<wire::EngineCapability>,
    command_tx: mpsc::Sender<Result<wire::CoordinatorCommand, Status>>,
    peer_fingerprint: Option<CertificateFingerprint>,
    ready: bool,
    draining: bool,
    active: Option<(JobId, LeaseId)>,
}

struct Job {
    submission: MeasureSubmission,
    descriptor: ArtifactDescriptor,
    measurement_key: MeasurementKey,
    request_nonce: RequestNonce,
    reply: Option<oneshot::Sender<ServiceResult<CommittedMeasurement>>>,
    state: JobState,
    recovery: Option<ActiveLease>,
}

enum JobState {
    Queued,
    Offered(ActiveLease),
    Running(ActiveLease),
}

#[derive(Clone, Copy)]
struct ActiveLease {
    device_id: DeviceId,
    current_session_id: SessionId,
    origin_session_id: SessionId,
    lease_id: LeaseId,
    deadline: Instant,
}

struct ArtifactRecord {
    bytes: Arc<[u8]>,
    references: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoordinatorSnapshot {
    pub connected_agents: usize,
    pub ready_agents: usize,
    pub queued_jobs: usize,
    pub active_leases: usize,
    pub running_leases: usize,
    pub artifact_items: usize,
    pub artifact_bytes: usize,
    pub committed_receipts: usize,
}

impl Coordinator {
    pub fn new(config: CoordinatorConfig) -> ServiceResult<Self> {
        config.validate()?;
        let receipts = ReceiptLedger::open(&config.receipt_ledger)?;
        Ok(Self {
            inner: Arc::new(CoordinatorInner {
                config,
                state: Mutex::new(CoordinatorState {
                    agents: BTreeMap::new(),
                    jobs: HashMap::new(),
                    queue: VecDeque::new(),
                    artifacts: HashMap::new(),
                    artifact_bytes: 0,
                    receipts,
                }),
            }),
        })
    }

    #[must_use]
    pub fn handle(&self) -> CoordinatorHandle {
        CoordinatorHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> CoordinatorSnapshot {
        snapshot(&lock_state(&self.inner))
    }

    pub fn reap_expired(&self) {
        let mut state = lock_state(&self.inner);
        expire_leases(&mut state, Instant::now());
        schedule(&self.inner.config, &mut state);
    }
}

impl CoordinatorHandle {
    pub async fn measure(
        &self,
        submission: MeasureSubmission,
    ) -> ServiceResult<CommittedMeasurement> {
        let timeout_ms = submission.options.timeout_ms;
        let (job_id, receiver) = self.enqueue(submission, RequestNonce::random())?;
        let receive = async { receiver.await.map_err(|_| ServiceError::Closed)? };

        match timeout_ms {
            Some(limit_ms) => {
                match tokio::time::timeout(Duration::from_millis(limit_ms), receive).await {
                    Ok(result) => result,
                    Err(_) => {
                        cancel_job(&self.inner, job_id);
                        Err(ServiceError::timeout(format!(
                            "caller deadline expired after {limit_ms} ms"
                        )))
                    }
                }
            }
            None => receive.await,
        }
    }

    pub fn enqueue_with_nonce(
        &self,
        submission: MeasureSubmission,
        request_nonce: RequestNonce,
    ) -> ServiceResult<(
        JobId,
        oneshot::Receiver<ServiceResult<CommittedMeasurement>>,
    )> {
        self.enqueue(submission, request_nonce)
    }

    #[must_use]
    pub fn snapshot(&self) -> CoordinatorSnapshot {
        snapshot(&lock_state(&self.inner))
    }

    pub fn drain_agents(&self) {
        let mut state = lock_state(&self.inner);
        for agent in state.agents.values_mut() {
            agent.draining = true;
            agent.ready = false;
            let _ = agent.command_tx.try_send(Ok(wire::CoordinatorCommand {
                command: Some(coordinator_command::Command::Drain(wire::DrainAgent {
                    cancel_active: false,
                })),
            }));
        }
    }

    fn enqueue(
        &self,
        submission: MeasureSubmission,
        request_nonce: RequestNonce,
    ) -> ServiceResult<(
        JobId,
        oneshot::Receiver<ServiceResult<CommittedMeasurement>>,
    )> {
        validate_submission(&self.inner.config, &submission)?;
        let descriptor = artifact_descriptor(&submission.artifact)?;
        let measurement_key = measurement_key(&submission, descriptor);
        let job_id = job_id(measurement_key, request_nonce);
        let (reply, receiver) = oneshot::channel();
        let mut state = lock_state(&self.inner);
        expire_leases(&mut state, Instant::now());

        if state.jobs.contains_key(&job_id) || state.receipts.contains(job_id) {
            return Err(ServiceError::protocol("job id is already admitted"));
        }
        if state.queue.len() >= self.inner.config.queue_capacity {
            return Err(ServiceError::capacity("coordinator job queue is full"));
        }
        if !state.receipts.can_admit(state.jobs.len()) {
            return Err(ServiceError::capacity("coordinator receipt ledger is full"));
        }

        retain_artifact(
            &self.inner.config,
            &mut state,
            descriptor.digest,
            &submission.artifact.bytes,
        )?;
        state.jobs.insert(
            job_id,
            Job {
                submission,
                descriptor,
                measurement_key,
                request_nonce,
                reply: Some(reply),
                state: JobState::Queued,
                recovery: None,
            },
        );
        state.queue.push_back(job_id);
        schedule(&self.inner.config, &mut state);
        Ok((job_id, receiver))
    }
}

fn validate_submission(
    config: &CoordinatorConfig,
    submission: &MeasureSubmission,
) -> ServiceResult<()> {
    if submission.artifact.bytes.len() > config.max_artifact_bytes {
        return Err(ServiceError::capacity(
            "artifact exceeds configured maximum",
        ));
    }
    if submission.measure_config.encoding == 0 {
        return Err(ServiceError::configuration(
            "measure config encoding must be non-zero",
        ));
    }
    if submission.measure_config.payload.len() > config.max_measure_config_bytes {
        return Err(ServiceError::capacity(
            "measure config exceeds configured maximum",
        ));
    }
    Ok(())
}

fn retain_artifact(
    config: &CoordinatorConfig,
    state: &mut CoordinatorState,
    digest: ArtifactDigest,
    bytes: &[u8],
) -> ServiceResult<()> {
    if let Some(record) = state.artifacts.get_mut(&digest) {
        if record.bytes.as_ref() != bytes {
            return Err(ServiceError::protocol(
                "artifact digest collision with different bytes",
            ));
        }
        record.references = record
            .references
            .checked_add(1)
            .ok_or_else(|| ServiceError::capacity("artifact reference count overflow"))?;
        return Ok(());
    }

    if state.artifacts.len() >= config.artifact_item_capacity {
        return Err(ServiceError::capacity("artifact item capacity is full"));
    }
    let new_total = state
        .artifact_bytes
        .checked_add(bytes.len())
        .ok_or_else(|| ServiceError::capacity("artifact byte count overflow"))?;
    if new_total > config.artifact_byte_capacity {
        return Err(ServiceError::capacity("artifact byte capacity is full"));
    }
    state.artifact_bytes = new_total;
    state.artifacts.insert(
        digest,
        ArtifactRecord {
            bytes: Arc::from(bytes),
            references: 1,
        },
    );
    Ok(())
}

fn release_artifact(state: &mut CoordinatorState, digest: ArtifactDigest) {
    let remove = match state.artifacts.get_mut(&digest) {
        Some(record) if record.references > 1 => {
            record.references -= 1;
            false
        }
        Some(_) => true,
        None => false,
    };
    if remove && let Some(record) = state.artifacts.remove(&digest) {
        state.artifact_bytes = state.artifact_bytes.saturating_sub(record.bytes.len());
    }
}

fn snapshot(state: &CoordinatorState) -> CoordinatorSnapshot {
    CoordinatorSnapshot {
        connected_agents: state.agents.len(),
        ready_agents: state
            .agents
            .values()
            .filter(|agent| agent.ready && !agent.draining)
            .count(),
        queued_jobs: state.queue.len(),
        active_leases: state
            .agents
            .values()
            .filter(|agent| agent.active.is_some())
            .count(),
        running_leases: state
            .jobs
            .values()
            .filter(|job| matches!(job.state, JobState::Running(_)))
            .count(),
        artifact_items: state.artifacts.len(),
        artifact_bytes: state.artifact_bytes,
        committed_receipts: state.receipts.len(),
    }
}

fn schedule(config: &CoordinatorConfig, state: &mut CoordinatorState) {
    loop {
        let assignment = state.queue.iter().find_map(|job_id| {
            let job = state.jobs.get(job_id)?;
            state
                .agents
                .iter()
                .find(|(_, agent)| agent_matches(agent, job))
                .map(|(device_id, _)| (*job_id, *device_id))
        });
        let Some((job_id, device_id)) = assignment else {
            return;
        };

        if let Some(index) = state.queue.iter().position(|queued| *queued == job_id) {
            state.queue.remove(index);
        } else {
            continue;
        }

        let session_id = state.agents[&device_id].session_id;
        let command_tx = state.agents[&device_id].command_tx.clone();
        let lease_id = LeaseId::random();
        let active = ActiveLease {
            device_id,
            current_session_id: session_id,
            origin_session_id: session_id,
            lease_id,
            deadline: Instant::now() + config.accept_timeout,
        };
        let command = lease_command(config, state.jobs.get(&job_id).unwrap(), job_id, active);

        let job = state.jobs.get_mut(&job_id).unwrap();
        job.state = JobState::Offered(active);
        job.recovery = None;
        let agent = state.agents.get_mut(&device_id).unwrap();
        agent.ready = false;
        agent.active = Some((job_id, lease_id));

        if command_tx.try_send(Ok(command)).is_err() {
            requeue_active(state, job_id, active, false);
            state.agents.remove(&device_id);
        }
    }
}

fn agent_matches(agent: &AgentSession, job: &Job) -> bool {
    agent.ready
        && !agent.draining
        && agent.active.is_none()
        && agent.profile_hash == job.submission.target_device_profile_hash
        && agent.capabilities.iter().any(|capability| {
            capability.engine.as_ref() == Some(&engine_identity_to_wire(job.submission.engine))
                && capability.artifact_formats.iter().any(|format| {
                    format.format_kind == job.descriptor.format_kind as i32
                        && format.adapter_format_id == job.descriptor.adapter_format_id
                })
                && capability
                    .measure_config_encodings
                    .contains(&job.submission.measure_config.encoding)
                && capability
                    .measurement_protocol_versions
                    .contains(&crate::MEASUREMENT_PROTOCOL_VERSION)
        })
}

fn lease_command(
    config: &CoordinatorConfig,
    job: &Job,
    job_id: JobId,
    active: ActiveLease,
) -> wire::CoordinatorCommand {
    wire::CoordinatorCommand {
        command: Some(coordinator_command::Command::Lease(wire::MeasureLease {
            session_id: active.current_session_id.to_vec(),
            job_id: job_id.to_vec(),
            measurement_key: job.measurement_key.to_vec(),
            request_nonce: job.request_nonce.to_vec(),
            lease_id: active.lease_id.to_vec(),
            lease_duration_ms: duration_ms(config.lease_duration),
            kind: wire::MeasureKind::Single as i32,
            engine: Some(engine_identity_to_wire(job.submission.engine)),
            measure_config_hash: job.submission.options.config_hash.as_bytes().to_vec(),
            measure_config_encoding: job.submission.measure_config.encoding,
            measure_config_payload: job.submission.measure_config.payload.clone(),
            target_device_profile_hash: job.submission.target_device_profile_hash.to_vec(),
            subjects: vec![wire::GraphSubject {
                logical_index: 0,
                graph_hash: job.submission.artifact.graph_hash.as_bytes().to_vec(),
                artifact: Some(wire::GraphArtifactRef {
                    artifact_digest: job.descriptor.digest.to_vec(),
                    artifact_size: job.descriptor.size,
                    format_kind: job.descriptor.format_kind as i32,
                    adapter_format_id: job.descriptor.adapter_format_id,
                }),
            }],
        })),
    }
}

fn expire_leases(state: &mut CoordinatorState, now: Instant) {
    let expired = state
        .jobs
        .iter()
        .filter_map(|(job_id, job)| match job.state {
            JobState::Offered(active) | JobState::Running(active) if active.deadline <= now => {
                Some((*job_id, active))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    for (job_id, active) in expired {
        requeue_active(state, job_id, active, false);
    }
}

fn requeue_active(
    state: &mut CoordinatorState,
    job_id: JobId,
    active: ActiveLease,
    preserve_for_recovery: bool,
) {
    if let Some(job) = state.jobs.get_mut(&job_id) {
        job.state = JobState::Queued;
        job.recovery = preserve_for_recovery.then_some(active);
        if !state.queue.contains(&job_id) {
            state.queue.push_back(job_id);
        }
    }
    if let Some(agent) = state.agents.get_mut(&active.device_id)
        && agent.session_id == active.current_session_id
        && agent.active == Some((job_id, active.lease_id))
    {
        agent.active = None;
        agent.ready = false;
    }
}

fn cancel_job(inner: &CoordinatorInner, job_id: JobId) {
    let mut state = lock_state(inner);
    if let Some(index) = state.queue.iter().position(|queued| *queued == job_id) {
        state.queue.remove(index);
    }
    let Some(job) = state.jobs.remove(&job_id) else {
        return;
    };
    if let JobState::Offered(active) | JobState::Running(active) = job.state
        && let Some(agent) = state.agents.get_mut(&active.device_id)
    {
        let _ = agent.command_tx.try_send(Ok(wire::CoordinatorCommand {
            command: Some(coordinator_command::Command::Cancel(wire::CancelLease {
                job_id: job_id.to_vec(),
                lease_id: active.lease_id.to_vec(),
                reason: wire::CancelReason::CancelCallerDeadline as i32,
            })),
        }));
        agent.active = None;
        agent.ready = false;
    }
    release_artifact(&mut state, job.descriptor.digest);
}

fn disconnect(inner: &CoordinatorInner, device_id: DeviceId, session_id: SessionId) {
    let mut state = lock_state(inner);
    let Some(agent) = state.agents.get(&device_id) else {
        return;
    };
    if agent.session_id != session_id {
        return;
    }
    let active = agent.active;
    state.agents.remove(&device_id);
    if let Some((job_id, lease_id)) = active {
        let active = state.jobs.get(&job_id).and_then(|job| match job.state {
            JobState::Offered(active) if active.lease_id == lease_id => Some((active, false)),
            JobState::Running(active) if active.lease_id == lease_id => Some((active, true)),
            _ => None,
        });
        if let Some((active, preserve_for_recovery)) = active {
            requeue_active(&mut state, job_id, active, preserve_for_recovery);
        }
    }
    schedule(&inner.config, &mut state);
}

fn lock_state(inner: &CoordinatorInner) -> MutexGuard<'_, CoordinatorState> {
    inner
        .state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn duration_ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

type CommandStream = ReceiverStream<Result<wire::CoordinatorCommand, Status>>;
type ArtifactStream =
    Pin<Box<dyn Stream<Item = Result<wire::ArtifactChunk, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl MeasureFleet for Coordinator {
    type ConnectStream = CommandStream;
    type FetchArtifactStream = ArtifactStream;

    async fn connect(
        &self,
        request: Request<Streaming<wire::AgentEvent>>,
    ) -> Result<Response<Self::ConnectStream>, Status> {
        let peer_fingerprint = request_peer_fingerprint(&request);
        let mut incoming = request.into_inner();
        let first = incoming
            .next()
            .await
            .ok_or_else(|| Status::invalid_argument("AgentHello is required"))??;
        let hello = match first.event {
            Some(agent_event::Event::Hello(hello)) => hello,
            _ => return Err(Status::invalid_argument("first event must be AgentHello")),
        };
        validate_hello(&hello).map_err(status_for_error)?;
        let device_id = DeviceId::from_slice(&hello.device_id).map_err(status_for_error)?;
        authenticate_enrollment(&self.inner.config.enrollment, peer_fingerprint, device_id)
            .map_err(status_for_error)?;
        let profile = hello
            .profile
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("device profile is required"))?;
        let profile_hash = device_profile_hash(profile);
        let session_id = SessionId::random();
        let (command_tx, command_rx) = mpsc::channel(self.inner.config.command_capacity);
        let recovery_dispositions = {
            let mut state = lock_state(&self.inner);
            if let Some(previous) = state.agents.remove(&device_id)
                && let Some((job_id, lease_id)) = previous.active
                && let Some((active, preserve_for_recovery)) =
                    state.jobs.get(&job_id).and_then(|job| match job.state {
                        JobState::Offered(active) if active.lease_id == lease_id => {
                            Some((active, false))
                        }
                        JobState::Running(active) if active.lease_id == lease_id => {
                            Some((active, true))
                        }
                        _ => None,
                    })
            {
                requeue_active(&mut state, job_id, active, preserve_for_recovery);
            }
            let (recoveries, active) = reconcile_recoveries(
                &self.inner.config,
                &mut state,
                device_id,
                session_id,
                &hello.recoveries,
            )
            .map_err(status_for_error)?;
            state.agents.insert(
                device_id,
                AgentSession {
                    session_id,
                    profile_hash,
                    capabilities: hello.capabilities,
                    command_tx: command_tx.clone(),
                    peer_fingerprint,
                    ready: false,
                    draining: false,
                    active,
                },
            );
            recoveries
        };
        let welcome = welcome(
            &self.inner.config,
            session_id,
            profile_hash,
            recovery_dispositions,
        );
        command_tx
            .try_send(Ok(welcome))
            .map_err(|_| Status::resource_exhausted("agent command queue is full"))?;

        let coordinator = self.clone();
        tokio::spawn(async move {
            while let Some(event) = incoming.next().await {
                match event {
                    Ok(event) => {
                        if coordinator
                            .handle_agent_event(device_id, session_id, event)
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            disconnect(&coordinator.inner, device_id, session_id);
        });

        Ok(Response::new(ReceiverStream::new(command_rx)))
    }

    async fn fetch_artifact(
        &self,
        request: Request<wire::FetchArtifactRequest>,
    ) -> Result<Response<Self::FetchArtifactStream>, Status> {
        let peer_fingerprint = request_peer_fingerprint(&request);
        let request = request.into_inner();
        let session_id = SessionId::from_slice(&request.session_id).map_err(status_for_error)?;
        let job_id = JobId::from_slice(&request.job_id).map_err(status_for_error)?;
        let lease_id = LeaseId::from_slice(&request.lease_id).map_err(status_for_error)?;
        let digest =
            ArtifactDigest::from_slice(&request.artifact_digest).map_err(status_for_error)?;

        let (bytes, chunk_bytes) = {
            let state = lock_state(&self.inner);
            let agent = authenticate_session(&state, session_id, peer_fingerprint)?;
            let job = state
                .jobs
                .get(&job_id)
                .ok_or_else(|| Status::not_found("unknown job"))?;
            validate_active_lease(job, agent, lease_id)?;
            if job.descriptor.digest != digest {
                return Err(Status::permission_denied(
                    "artifact is not authorized for this lease",
                ));
            }
            let bytes = Arc::clone(
                &state
                    .artifacts
                    .get(&digest)
                    .ok_or_else(|| Status::data_loss("leased artifact is missing"))?
                    .bytes,
            );
            (bytes, self.inner.config.artifact_chunk_bytes)
        };

        let offset = usize::try_from(request.offset)
            .map_err(|_| Status::invalid_argument("artifact offset does not fit usize"))?;
        if offset > bytes.len() {
            return Err(Status::invalid_argument("artifact offset exceeds size"));
        }
        let total_size = bytes.len() as u64;
        let stream = tokio_stream::iter((offset..bytes.len()).step_by(chunk_bytes).map({
            let bytes = Arc::clone(&bytes);
            move |offset| {
                let end = offset.saturating_add(chunk_bytes).min(bytes.len());
                Ok(wire::ArtifactChunk {
                    artifact_digest: digest.to_vec(),
                    total_size,
                    offset: offset as u64,
                    data: bytes[offset..end].to_vec(),
                })
            }
        }));
        Ok(Response::new(Box::pin(stream)))
    }

    async fn submit_result(
        &self,
        request: Request<wire::SubmitResultRequest>,
    ) -> Result<Response<wire::SubmitResultAck>, Status> {
        let peer_fingerprint = request_peer_fingerprint(&request);
        let request = request.into_inner();
        let current_session =
            SessionId::from_slice(&request.current_session_id).map_err(status_for_error)?;
        let report = request
            .report
            .ok_or_else(|| Status::invalid_argument("measure report is required"))?;
        let job_id = JobId::from_slice(&report.job_id).map_err(status_for_error)?;
        let lease_id = LeaseId::from_slice(&report.lease_id).map_err(status_for_error)?;

        let mut state = lock_state(&self.inner);
        let agent = authenticate_session(&state, current_session, peer_fingerprint)?;
        let device_id = DeviceId::from_slice(&report.device_id).map_err(status_for_error)?;
        if device_id != agent.0 {
            return Err(Status::permission_denied("report device identity mismatch"));
        }
        let receipt_state = state.receipts.state(job_id, &report);
        if receipt_state != ReceiptState::Missing {
            let identical = receipt_state == ReceiptState::Identical;
            return Ok(Response::new(wire::SubmitResultAck {
                disposition: if identical {
                    wire::ResultDisposition::ResultAlreadyCommitted as i32
                } else {
                    wire::ResultDisposition::ResultInvalidReport as i32
                },
                agent_may_delete_report: true,
                bounded_message: if identical {
                    "result already committed".to_owned()
                } else {
                    "job already has a different committed report".to_owned()
                },
            }));
        }

        let job = state
            .jobs
            .get(&job_id)
            .ok_or_else(|| Status::not_found("unknown job"))?;
        validate_report(
            &self.inner.config,
            job,
            agent,
            current_session,
            lease_id,
            &report,
        )?;

        match state
            .receipts
            .commit(job_id, &report)
            .map_err(status_for_error)?
        {
            ReceiptState::Missing => {}
            ReceiptState::Identical => {
                return Ok(Response::new(wire::SubmitResultAck {
                    disposition: wire::ResultDisposition::ResultAlreadyCommitted as i32,
                    agent_may_delete_report: true,
                    bounded_message: "result already committed".to_owned(),
                }));
            }
            ReceiptState::Conflict => {
                return Ok(Response::new(wire::SubmitResultAck {
                    disposition: wire::ResultDisposition::ResultInvalidReport as i32,
                    agent_may_delete_report: true,
                    bounded_message: "job already has a different committed report".to_owned(),
                }));
            }
        }

        let mut job = state.jobs.remove(&job_id).unwrap();
        if let Some(index) = state.queue.iter().position(|queued| *queued == job_id) {
            state.queue.remove(index);
        }
        if let Some(agent) = state.agents.get_mut(&device_id) {
            agent.active = None;
            agent.ready = false;
        }
        release_artifact(&mut state, job.descriptor.digest);
        if let Some(reply) = job.reply.take() {
            let _ = reply.send(Ok(CommittedMeasurement {
                job_id,
                report: report.clone(),
            }));
        }
        schedule(&self.inner.config, &mut state);

        Ok(Response::new(wire::SubmitResultAck {
            disposition: wire::ResultDisposition::ResultJobCommitted as i32,
            agent_may_delete_report: true,
            bounded_message: "result committed".to_owned(),
        }))
    }
}

impl Coordinator {
    fn handle_agent_event(
        &self,
        device_id: DeviceId,
        session_id: SessionId,
        event: wire::AgentEvent,
    ) -> ServiceResult<()> {
        let mut state = lock_state(&self.inner);
        expire_leases(&mut state, Instant::now());
        let agent = state
            .agents
            .get(&device_id)
            .ok_or_else(|| ServiceError::protocol("agent session is not registered"))?;
        if agent.session_id != session_id {
            return Err(ServiceError::protocol("agent session was superseded"));
        }

        match event.event {
            Some(agent_event::Event::Ready(ready)) => {
                if SessionId::from_slice(&ready.session_id)? != session_id {
                    return Err(ServiceError::protocol("ready session id mismatch"));
                }
                if ready.free_slots > 1 {
                    return Err(ServiceError::protocol("free_slots must be zero or one"));
                }
                let agent = state.agents.get_mut(&device_id).unwrap();
                if ready.free_slots == 1 && agent.active.is_some() {
                    return Err(ServiceError::protocol("busy agent advertised a free slot"));
                }
                agent.ready = ready.free_slots == 1;
            }
            Some(agent_event::Event::Accepted(accepted)) => {
                let active = validate_agent_lease_ids(
                    &state,
                    device_id,
                    session_id,
                    &accepted.session_id,
                    &accepted.job_id,
                    &accepted.lease_id,
                )?;
                let job_id = JobId::from_slice(&accepted.job_id)?;
                let job = state.jobs.get_mut(&job_id).unwrap();
                if !matches!(job.state, JobState::Offered(_)) {
                    return Err(ServiceError::protocol("job was accepted more than once"));
                }
                job.state = JobState::Running(ActiveLease {
                    deadline: Instant::now() + self.inner.config.lease_duration,
                    ..active
                });
            }
            Some(agent_event::Event::Rejected(rejected)) => {
                let active = validate_agent_lease_ids(
                    &state,
                    device_id,
                    session_id,
                    &rejected.session_id,
                    &rejected.job_id,
                    &rejected.lease_id,
                )?;
                let job_id = JobId::from_slice(&rejected.job_id)?;
                requeue_active(&mut state, job_id, active, false);
            }
            Some(agent_event::Event::Heartbeat(heartbeat)) => {
                let active = validate_agent_lease_ids(
                    &state,
                    device_id,
                    session_id,
                    &heartbeat.session_id,
                    &heartbeat.job_id,
                    &heartbeat.lease_id,
                )?;
                let job_id = JobId::from_slice(&heartbeat.job_id)?;
                let job = state.jobs.get_mut(&job_id).unwrap();
                if !matches!(job.state, JobState::Running(_)) {
                    return Err(ServiceError::protocol("heartbeat preceded acceptance"));
                }
                job.state = JobState::Running(ActiveLease {
                    deadline: Instant::now() + self.inner.config.lease_duration,
                    ..active
                });
            }
            Some(agent_event::Event::Draining(draining)) => {
                if SessionId::from_slice(&draining.session_id)? != session_id {
                    return Err(ServiceError::protocol("draining session id mismatch"));
                }
                let agent = state.agents.get_mut(&device_id).unwrap();
                agent.draining = true;
                agent.ready = false;
            }
            Some(agent_event::Event::Hello(_)) => {
                return Err(ServiceError::protocol("AgentHello appeared more than once"));
            }
            None => return Err(ServiceError::protocol("empty agent event")),
        }
        schedule(&self.inner.config, &mut state);
        Ok(())
    }
}

fn validate_hello(hello: &wire::AgentHello) -> ServiceResult<()> {
    if hello.protocol_major != PROTOCOL_MAJOR {
        return Err(ServiceError::protocol(format!(
            "unsupported protocol major {}",
            hello.protocol_major
        )));
    }
    DeviceId::from_slice(&hello.device_id)?;
    if hello.agent_build.is_empty() || hello.agent_build.len() > MAX_AGENT_BUILD_BYTES {
        return Err(ServiceError::protocol("agent_build length is invalid"));
    }
    let profile = hello
        .profile
        .as_ref()
        .ok_or_else(|| ServiceError::protocol("device profile is required"))?;
    validate_profile(profile)?;
    if hello.capabilities.is_empty() {
        return Err(ServiceError::protocol(
            "at least one engine capability is required",
        ));
    }
    for capability in &hello.capabilities {
        let identity = capability
            .engine
            .as_ref()
            .ok_or_else(|| ServiceError::protocol("capability engine identity is required"))?;
        crate::engine_identity_from_wire(identity)?;
        if capability.artifact_formats.is_empty()
            || capability.measure_config_encodings.is_empty()
            || capability.measurement_protocol_versions.is_empty()
        {
            return Err(ServiceError::protocol("capability lists must be non-empty"));
        }
    }
    if hello.recoveries.len() > 1 {
        return Err(ServiceError::protocol(
            "version 1 permits at most one lease recovery",
        ));
    }
    for recovery in &hello.recoveries {
        JobId::from_slice(&recovery.job_id)?;
        LeaseId::from_slice(&recovery.lease_id)?;
        let phase = wire::JobPhase::try_from(recovery.phase)
            .map_err(|_| ServiceError::protocol("unknown recovery phase"))?;
        if phase == wire::JobPhase::Unspecified {
            return Err(ServiceError::protocol("recovery phase is unspecified"));
        }
        if !recovery.report_persisted {
            return Err(ServiceError::protocol(
                "this agent version can recover persisted reports only",
            ));
        }
    }
    Ok(())
}

fn reconcile_recoveries(
    config: &CoordinatorConfig,
    state: &mut CoordinatorState,
    device_id: DeviceId,
    session_id: SessionId,
    recoveries: &[wire::LeaseRecovery],
) -> ServiceResult<RecoveryReconciliation> {
    let mut dispositions = Vec::with_capacity(recoveries.len());
    let mut agent_active = None;
    for recovery in recoveries {
        let job_id = JobId::from_slice(&recovery.job_id)?;
        let lease_id = LeaseId::from_slice(&recovery.lease_id)?;
        let action = if state.receipts.contains(job_id) {
            wire::RecoveryAction::RecoveryAlreadyCommitted
        } else {
            let recoverable = state.jobs.get(&job_id).and_then(|job| job.recovery);
            match recoverable {
                Some(active) if active.device_id == device_id && active.lease_id == lease_id => {
                    let rebound = ActiveLease {
                        current_session_id: session_id,
                        deadline: Instant::now() + config.lease_duration,
                        ..active
                    };
                    let job = state.jobs.get_mut(&job_id).unwrap();
                    job.state = JobState::Running(rebound);
                    job.recovery = None;
                    if let Some(index) = state.queue.iter().position(|queued| *queued == job_id) {
                        state.queue.remove(index);
                    }
                    agent_active = Some((job_id, lease_id));
                    wire::RecoveryAction::RecoverySubmitReport
                }
                _ => wire::RecoveryAction::RecoveryCancel,
            }
        };
        dispositions.push(wire::RecoveryDisposition {
            job_id: job_id.to_vec(),
            lease_id: lease_id.to_vec(),
            action: action as i32,
        });
    }
    Ok((dispositions, agent_active))
}

fn validate_profile(profile: &wire::DeviceProfile) -> ServiceResult<()> {
    for (name, value) in [
        ("platform_family", profile.platform_family.as_str()),
        ("board_model", profile.board_model.as_str()),
        ("soc", profile.soc.as_str()),
        ("gpu_architecture", profile.gpu_architecture.as_str()),
        ("platform_release", profile.platform_release.as_str()),
        ("cuda_version", profile.cuda_version.as_str()),
        ("gpu_driver_version", profile.gpu_driver_version.as_str()),
        ("compiler_version", profile.compiler_version.as_str()),
        (
            "compiler_runtime_version",
            profile.compiler_runtime_version.as_str(),
        ),
        ("power_profile", profile.power_profile.as_str()),
        ("clock_policy", profile.clock_policy.as_str()),
        ("cooling_policy", profile.cooling_policy.as_str()),
    ] {
        if value.is_empty() || value.len() > MAX_PROFILE_STRING_BYTES {
            return Err(ServiceError::protocol(format!(
                "device profile field {name} has invalid length"
            )));
        }
    }
    fixed_bytes::<32>(
        "operating_system_image_digest",
        &profile.operating_system_image_digest,
    )?;
    fixed_bytes::<32>("agent_image_digest", &profile.agent_image_digest)?;
    if profile.measurement_protocol_version != crate::MEASUREMENT_PROTOCOL_VERSION {
        return Err(ServiceError::protocol(
            "unsupported device measurement protocol version",
        ));
    }
    Ok(())
}

fn authenticate_enrollment(
    enrollment: &Enrollment,
    peer_fingerprint: Option<CertificateFingerprint>,
    claimed_device: DeviceId,
) -> ServiceResult<()> {
    match enrollment {
        Enrollment::MutualTls(devices) => {
            let fingerprint = peer_fingerprint
                .ok_or_else(|| ServiceError::authentication("client certificate is required"))?;
            let enrolled = devices.get(&fingerprint).ok_or_else(|| {
                ServiceError::authentication("client certificate is not enrolled")
            })?;
            if *enrolled != claimed_device {
                return Err(ServiceError::authentication(
                    "certificate and device id do not match",
                ));
            }
        }
        Enrollment::InsecureForTests(devices) => {
            if !devices.contains(&claimed_device) {
                return Err(ServiceError::authentication("device id is not enrolled"));
            }
        }
    }
    Ok(())
}

fn request_peer_fingerprint<T>(request: &Request<T>) -> Option<CertificateFingerprint> {
    request
        .peer_certs()
        .and_then(|certificates| certificates.first().cloned())
        .map(|certificate| certificate_fingerprint(certificate.as_ref()))
}

fn welcome(
    config: &CoordinatorConfig,
    session_id: SessionId,
    profile_hash: DeviceProfileHash,
    recoveries: Vec<wire::RecoveryDisposition>,
) -> wire::CoordinatorCommand {
    wire::CoordinatorCommand {
        command: Some(coordinator_command::Command::Welcome(wire::AgentWelcome {
            session_id: session_id.to_vec(),
            negotiated_minor: PROTOCOL_MINOR,
            device_profile_hash: profile_hash.to_vec(),
            heartbeat_interval_ms: duration_ms(config.heartbeat_interval),
            accept_timeout_ms: duration_ms(config.accept_timeout),
            default_lease_duration_ms: duration_ms(config.lease_duration),
            max_artifact_bytes: config.max_artifact_bytes as u64,
            artifact_chunk_bytes: config.artifact_chunk_bytes as u32,
            max_measure_config_bytes: config.max_measure_config_bytes as u32,
            max_engine_metadata_bytes: config.max_engine_metadata_bytes as u32,
            max_samples_per_subject: config.max_samples_per_subject as u32,
            max_telemetry_samples: config.max_telemetry_samples as u32,
            max_bounded_string_bytes: 512,
            recoveries,
        })),
    }
}

fn authenticate_session(
    state: &CoordinatorState,
    session_id: SessionId,
    peer_fingerprint: Option<CertificateFingerprint>,
) -> Result<(DeviceId, &AgentSession), Status> {
    let (device_id, agent) = state
        .agents
        .iter()
        .find(|(_, agent)| agent.session_id == session_id)
        .ok_or_else(|| Status::permission_denied("unknown agent session"))?;
    if agent.peer_fingerprint != peer_fingerprint {
        return Err(Status::permission_denied(
            "data-plane peer differs from control session",
        ));
    }
    Ok((*device_id, agent))
}

fn validate_active_lease(
    job: &Job,
    agent: (DeviceId, &AgentSession),
    lease_id: LeaseId,
) -> Result<(), Status> {
    let active = match job.state {
        JobState::Offered(active) | JobState::Running(active) => active,
        JobState::Queued => return Err(Status::failed_precondition("job is not leased")),
    };
    if active.device_id != agent.0
        || active.current_session_id != agent.1.session_id
        || active.lease_id != lease_id
    {
        return Err(Status::permission_denied("lease identity mismatch"));
    }
    Ok(())
}

fn validate_agent_lease_ids(
    state: &CoordinatorState,
    device_id: DeviceId,
    session_id: SessionId,
    event_session: &[u8],
    event_job: &[u8],
    event_lease: &[u8],
) -> ServiceResult<ActiveLease> {
    if SessionId::from_slice(event_session)? != session_id {
        return Err(ServiceError::protocol("event session id mismatch"));
    }
    let job_id = JobId::from_slice(event_job)?;
    let lease_id = LeaseId::from_slice(event_lease)?;
    let job = state
        .jobs
        .get(&job_id)
        .ok_or_else(|| ServiceError::protocol("event references unknown job"))?;
    let active = match job.state {
        JobState::Offered(active) | JobState::Running(active) => active,
        JobState::Queued => {
            return Err(ServiceError::protocol("event references an unleased job"));
        }
    };
    if active.device_id != device_id
        || active.current_session_id != session_id
        || active.lease_id != lease_id
    {
        return Err(ServiceError::protocol("event lease identity mismatch"));
    }
    Ok(active)
}

fn validate_report(
    config: &CoordinatorConfig,
    job: &Job,
    agent: (DeviceId, &AgentSession),
    current_session: SessionId,
    lease_id: LeaseId,
    report: &wire::MeasureReport,
) -> Result<(), Status> {
    let active = match job.state {
        JobState::Running(active) => active,
        JobState::Offered(_) => return Err(Status::failed_precondition("job was not accepted")),
        JobState::Queued => return Err(Status::failed_precondition("job is not leased")),
    };
    if active.device_id != agent.0
        || active.current_session_id != current_session
        || active.lease_id != lease_id
    {
        return Err(Status::failed_precondition("stale lease"));
    }
    if report.origin_session_id != active.origin_session_id.as_bytes()
        || report.device_profile_hash != agent.1.profile_hash.as_bytes()
        || report.measurement_key != job.measurement_key.as_bytes()
        || report.request_nonce != job.request_nonce.as_bytes()
        || report.measure_config_hash != job.submission.options.config_hash.as_bytes()
        || report.kind != wire::MeasureKind::Single as i32
        || report.engine.as_ref() != Some(&engine_identity_to_wire(job.submission.engine))
    {
        return Err(Status::invalid_argument("report identity mismatch"));
    }
    if report.subjects.len() != 1
        || report.subjects[0].logical_index != 0
        || report.subjects[0].graph_hash != job.submission.artifact.graph_hash.as_bytes()
    {
        return Err(Status::invalid_argument("report subject mismatch"));
    }
    if report.subjects[0].engine_metadata.len() > config.max_engine_metadata_bytes {
        return Err(Status::resource_exhausted("engine metadata is too large"));
    }
    if report.samples.len() > config.max_samples_per_subject
        || report.telemetry.len() > config.max_telemetry_samples
    {
        return Err(Status::resource_exhausted("report evidence is too large"));
    }
    let outcome = wire::MeasureAttemptOutcome::try_from(report.outcome)
        .map_err(|_| Status::invalid_argument("unknown report outcome"))?;
    match outcome {
        wire::MeasureAttemptOutcome::MeasureOutcomeUnspecified => {
            return Err(Status::invalid_argument("report outcome is unspecified"));
        }
        wire::MeasureAttemptOutcome::MeasureOutcomeSucceeded => {
            let reward = report.subjects[0]
                .scalar_reward
                .ok_or_else(|| Status::invalid_argument("successful report omitted reward"))?;
            if !reward.is_finite() || report.failure.is_some() {
                return Err(Status::invalid_argument("successful report is malformed"));
            }
        }
        _ => {
            let failure = report
                .failure
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("failed report omitted detail"))?;
            if report.subjects[0].scalar_reward.is_some() {
                return Err(Status::invalid_argument(
                    "failed report included scalar reward",
                ));
            }
            if failure.bounded_message.len() > 512 {
                return Err(Status::resource_exhausted("failure message is too large"));
            }
        }
    }
    Ok(())
}

fn status_for_error(error: ServiceError) -> Status {
    match error {
        ServiceError::Authentication(message) => Status::unauthenticated(message),
        ServiceError::Capacity(message) => Status::resource_exhausted(message),
        ServiceError::Configuration(message) | ServiceError::Protocol(message) => {
            Status::invalid_argument(message)
        }
        ServiceError::Timeout(message) => Status::deadline_exceeded(message),
        ServiceError::Transport(message) | ServiceError::Io(message) => {
            Status::unavailable(message)
        }
        ServiceError::RemoteFailure { message, .. } => Status::failed_precondition(message),
        ServiceError::Closed => Status::unavailable("measurement service closed"),
    }
}
