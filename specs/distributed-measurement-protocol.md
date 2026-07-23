# GraphZero Distributed Measurement Protocol

Status: Draft

Protocol version: 1.0

Last updated: 2026-07-22

## 1. Purpose

This document specifies how GraphZero dispatches terminal graph measurements
from the main search process to a fleet of edge measurement devices. The first
target fleet is a small set of NVIDIA Jetson devices, but the protocol is not
Jetson-specific.

The protocol covers:

- agent registration and capability negotiation;
- compatible-device selection;
- bounded work admission;
- job leasing and cancellation;
- graph artifact transfer;
- measurement result delivery;
- reconnect, retry, and duplicate handling;
- device telemetry and measurement provenance;
- authentication and transport security; and
- the boundary between the distributed service and `GraphEngine::measure`.

The protocol does not promise identical timing samples. It promises that every
accepted result was produced under a declared measurement configuration, on a
compatible device, with enough raw evidence and telemetry to validate or reject
the result.

The key words MUST, MUST NOT, REQUIRED, SHOULD, SHOULD NOT, and MAY are used in
their normative sense.

## 2. Goals

The protocol MUST:

1. Keep terminal measurement semantics owned by the concrete `GraphEngine`.
2. Keep networking and async-runtime dependencies out of `gz-engine`.
3. Allow devices to join and leave without restarting search workers.
4. Route work only to devices compatible with the requested engine and target
   device profile.
5. Keep all coordinator and orchestrator queues bounded.
6. Tolerate agent, connection, and coordinator-side RPC failures without
   silently losing or duplicating a committed result.
7. Preserve enough identity and provenance to audit every replay reward.
8. Support an atomic paired measurement for symmetric selfplay.
9. Prevent large artifact transfers from blocking control messages.
10. Require authentication and encryption between agents and the coordinator.

## 3. Non-goals

Version 1 does not provide:

- a general-purpose distributed compute scheduler;
- distributed neural evaluation;
- exactly-once physical execution;
- automatic comparability between different hardware profiles;
- durable recovery of an in-progress search process;
- arbitrary concurrent jobs on one GPU;
- a public API for untrusted third-party agents; or
- a guarantee of zero hardware timing variance.

Job delivery permits duplicate physical execution when a lease is lost and
retried. Result commitment is exactly-once per job. A job may still terminate
without GPU execution because of cancellation, incompatibility, or a permanent
compile failure.

## 4. Existing GraphZero constraints

The design follows these existing boundaries:

- [`GraphEngine::measure`](../crates/gz-engine/src/traits.rs) remains the
  terminal measurement interface.
- [`SearchWork::Measure`](../crates/gz-search/src/work.rs) is the asynchronous
  scheduling seam used by production search.
- [`GraphArtifact`](../crates/gz-engine/src/metadata.rs) is the portable graph
  representation. Process-local `E::Graph` handles MUST NOT cross the network.
- Replay admission continues to require a terminal, valid measurement result.
- `gz-measurer` remains responsible for projecting measured episodes into
  replay. It is not the remote measurement transport.

Local engines and tests MAY execute `GraphEngine::measure` directly. Production
compiler selfplay MUST dispatch `SearchWork::Measure` through the measurement
coordinator so a network wait does not block an engine lane.

The concrete compiler adapter owns both:

- importing its adapter-specific `GraphArtifact` on the agent; and
- converting the portable measurement report into a validated
  `MeasureResult<E::Graph>` on the main process.

### 4.1 Required repository boundaries

The intended implementation layout is:

- `gz-engine` retains dependency-light identities, `MeasureOptions`,
  `MeasureResult`, and `GraphArtifact`; it gains no transport or async-runtime
  dependency.
- `gz-search` continues to emit terminal work tokens and adds a paired or batch
  measurement work item for symmetric selfplay.
- `gz-orchestrator` gains bounded parked-measurement state analogous to its
  parked evaluator work, plus the coordinator submission and reply path.
- a new `gz-measure-service` crate owns protocol types, gRPC transport,
  coordinator state, and the coordinator client.
- the edge binary owns agent supervision and the compiler-specific artifact
  loader. It may be a dedicated `gz-measure-agent` crate or a `graphzero`
  subcommand, but it is not part of `gz-engine`.
- `gz-measurer` continues to enforce measured-before-replay and eventually
  projects a compact receipt ID rather than transporting remote jobs.

A blocking remote facade MAY be provided for CLI tools that directly call
`GraphEngine::measure`. Production search lanes MUST use the parked async path.

## 5. Architecture

```text
search task
    │ SearchWork::Measure or paired MeasureBatch
    ▼
orchestrator measurement gateway
    │ bounded pending work; keeps local graph handles
    ▼
measurement coordinator
    ├── bounded job queue
    ├── agent registry and compatibility scheduler
    ├── artifact store
    └── measurement receipt ledger
            │ outbound agent connections
            ├──────── Jetson agent A ─ compiler ─ CUDA
            ├──────── Jetson agent B ─ compiler ─ CUDA
            └──────── Jetson agent N ─ compiler ─ CUDA
```

### 5.1 Orchestrator measurement gateway

The gateway:

- receives terminal measurement work from search;
- exports each local graph handle to a `GraphArtifact`;
- submits a portable job to the coordinator;
- parks the search task while the job is pending;
- resumes the task only after receiving a committed terminal result; and
- releases or retains local graph handles according to existing search
  ownership rules.

The number of parked measurement tasks MUST be bounded. When the bound is
reached, the orchestrator MUST stop admitting additional episodes until
capacity becomes available.

### 5.2 Measurement coordinator

The coordinator:

- authenticates and registers agents;
- computes the effective device profile for each session;
- accepts jobs into a bounded queue;
- selects a compatible healthy agent;
- assigns a time-limited lease;
- serves immutable graph artifacts;
- records attempts and committed results;
- retries eligible failures; and
- returns one terminal result to the gateway.

The coordinator MUST NOT calculate candidate semantics, compile graphs, or
access process-local graph handles.

### 5.3 Measurement agent

One agent runs on each physical edge device. The agent:

- initiates the connection to the coordinator;
- reports its exact software and hardware profile;
- accepts at most one GPU measurement job in protocol version 1;
- downloads and verifies graph artifacts;
- invokes the concrete compiler engine locally;
- enforces the requested measurement protocol;
- records raw timing samples and device telemetry;
- persists a completed report until the coordinator acknowledges it; and
- reconnects and reconciles an interrupted lease.

Agents require no inbound network port.

### 5.4 Artifact store and result ledger

Artifacts are immutable and content-addressed. A deployment MAY keep them in
coordinator memory initially, but it MUST enforce byte and item bounds.

Committed reports MUST be written to a measurement receipt ledger before the
gateway is told that the measurement completed. The receipt ledger is the
authoritative audit record for device identity, raw samples, telemetry, and
attempt history.

## 6. Transport

The wire protocol is Protocol Buffers over gRPC and HTTP/2.

All production connections MUST use TLS with mutual certificate
authentication. The agent is the gRPC client and the coordinator is the gRPC
server.

The protocol has exactly three application RPCs in version 1:

```proto
syntax = "proto3";

package graphzero.measure.v1;

service MeasureFleet {
  rpc Connect(stream AgentEvent)
      returns (stream CoordinatorCommand);

  rpc FetchArtifact(FetchArtifactRequest)
      returns (stream ArtifactChunk);

  rpc SubmitResult(SubmitResultRequest)
      returns (SubmitResultAck);
}
```

`Connect` is the long-lived control plane. `FetchArtifact` and `SubmitResult`
are the data plane. Large artifact bytes MUST NOT be placed on the control
stream.

The standard gRPC health service SHOULD be exposed by the coordinator for
deployment monitoring. It does not replace agent measurement-health
heartbeats.

### 6.1 Runtime placement

The gRPC implementation belongs in a service crate such as
`gz-measure-service` and in the agent binary. Dependencies such as `tonic`,
`prost`, `tokio`, and `rustls` MUST NOT enter `gz-engine`.

The existing evaluator frame protocol is local and Unix-only. It MUST NOT be
extended directly to remote devices.

### 6.2 Finite RPC deadlines

`FetchArtifact` and `SubmitResult` MUST use explicit RPC deadlines. The
long-lived `Connect` stream uses transport keepalive and application
heartbeats instead of a short RPC deadline.

The gRPC deadline is not the job deadline and not the lease timeout. These
three concepts MUST remain separate.

### 6.3 Transport retry policy

Automatic or application retries MAY be enabled for:

- `FetchArtifact`, resuming from a verified byte offset; and
- `SubmitResult`, because result submission is idempotent.

Lease assignment MUST NOT be transparently replayed. A broken control stream
is recovered through the lease-reconciliation protocol in Section 14.3.

### 6.4 gRPC status usage

Transport and request-validation failures use gRPC status codes. Measurement
outcomes use the typed report enums in Section 12 and normally return an OK RPC
status with a `SubmitResultAck`.

| Condition | gRPC status |
| --- | --- |
| Missing or invalid client identity | `UNAUTHENTICATED` |
| Authenticated identity lacks access | `PERMISSION_DENIED` |
| Malformed field, identifier, enum, or offset | `INVALID_ARGUMENT` |
| Unsupported protocol or encoding | `FAILED_PRECONDITION` |
| Negotiated size or capacity exceeded | `RESOURCE_EXHAUSTED` |
| Referenced leased artifact is unexpectedly absent | `DATA_LOSS` |
| Coordinator cannot durably process a retriable request | `UNAVAILABLE` |
| Finite RPC deadline expires | `DEADLINE_EXCEEDED` |

Error messages are bounded and MUST NOT disclose artifact contents, compiler
inputs, credentials, or information about another device or lease.

## 7. Protocol and schema versioning

The first agent message includes a protocol major and minor version.

- A major version changes incompatible semantics or field interpretation.
- A minor version adds backward-compatible fields or capabilities.
- The coordinator MUST reject an unsupported major version.
- The coordinator selects the highest mutually supported minor version.
- A participant MUST ignore unknown fields allowed by the negotiated version.
- Removed Protobuf field numbers MUST be reserved and never reused.

Protocol versioning is separate from:

- engine identity and version;
- graph artifact format version;
- measure-config encoding version;
- device-profile encoding version; and
- agent build identity.

## 8. Identities and hashes

All fixed-width identifiers MUST be validated before use. A malformed length is
a protocol error.

| Identity | Width | Meaning |
| --- | ---: | --- |
| `device_id` | 16 bytes | Provisioned physical agent identity |
| `session_id` | 16 bytes | One accepted control-stream session |
| `lease_id` | 16 bytes | One execution attempt |
| `request_nonce` | 16 bytes | One logical call to terminal measurement |
| `measurement_key` | 32 bytes | Semantic graph/config/profile identity |
| `job_id` | 32 bytes | Stable identity of one logical call |
| `graph_hash` | 32 bytes | Engine-owned portable graph identity |
| `artifact_digest` | 32 bytes | BLAKE3 digest of artifact format and bytes |
| `measure_config_hash` | 32 bytes | Engine-owned measurement configuration |
| `device_profile_hash` | 32 bytes | Coordinator-normalized execution profile |

`session_id`, `lease_id`, and `request_nonce` MUST be generated from a
cryptographically secure random source or an equivalently collision-resistant
generator.

The coordinator has at most one current `session_id` for a `device_id`.
Accepting a new authenticated control stream atomically supersedes the previous
session and closes or invalidates its stream. An existing lease is recovered as
specified in Section 14.3; it is not silently assigned to both sessions.

### 8.1 Artifact digest

The artifact digest is:

```text
BLAKE3(
  "gz-graph-artifact-v1\0" ||
  format_kind_u32_le ||
  adapter_format_id_u32_le ||
  artifact_bytes
)
```

For non-adapter-specific formats, `adapter_format_id` is zero.

### 8.2 Measurement key and job identity

The coordinator computes the semantic measurement key over a canonical
encoding:

```text
BLAKE3(
  "gz-measure-key-v1\0" ||
  measure_kind_u32_le ||
  engine_id ||
  engine_version ||
  action_set_hash ||
  measure_config_hash ||
  target_device_profile_hash ||
  subject_count_u32_le ||
  for each subject in logical-index order {
    logical_index_u32_le ||
    graph_hash ||
    artifact_format_kind_u32_le ||
    adapter_format_id_u32_le ||
    artifact_digest ||
    artifact_size_u64_le
  }
)
```

The gateway generates one `request_nonce` for each logical call to terminal
measurement. It reuses that nonce when retrying submission of the same call.
The job ID is:

```text
BLAKE3(
  "gz-measure-job-v1\0" ||
  measurement_key ||
  request_nonce
)
```

The job ID excludes queue position, agent identity, session identity, lease
identity, attempt number, and wall-clock timestamps. Retrying one logical call
therefore produces the same `job_id` and a new `lease_id`. A separate call for
the same graph and config gets a new request nonce and job ID.

For symmetric selfplay, logical index 0 is player one and logical index 1 is
player two. Their order is significant.

### 8.3 Measure-config payload

The lease includes both `measure_config_hash` and an engine-owned immutable
measure-config payload. The payload has an explicit encoding version. The
concrete engine adapter MUST verify that the payload matches the requested
hash before measurement begins.

The payload MUST commit to all settings that can change the meaning of a
measurement, including:

- inputs, shapes, dtypes, and input-data digest;
- compiler and optimization flags;
- compiler, runtime, and worker-image identity;
- target device profile;
- power, clock, cooling, and thermal policy;
- compilation, capture, warm-up, and execution protocol;
- sample count and launches per sample;
- GPU-only versus end-to-end timing semantics;
- stability and invalidation rules;
- scalar reward projection; and
- whether compilation or capture latency contributes to reward.

## 9. Device profiles and capabilities

The agent reports a typed `DeviceProfile`; the coordinator normalizes it and
computes the effective `device_profile_hash`. The certificate identity and
reported `device_id` MUST match the coordinator's enrollment record.

A device profile MUST include at least:

- platform family and board model;
- SoC and GPU architecture;
- usable memory class;
- operating-system image identity;
- JetPack or equivalent platform version;
- CUDA, GPU driver, and runtime versions;
- compiler and compiler-runtime versions;
- agent build identity;
- power profile;
- clock policy;
- cooling policy; and
- measurement protocol implementation version.

Transient values such as current temperature are telemetry, not profile
identity. A change to power mode, clock policy, software image, compiler, or
measurement implementation changes the profile hash.

An engine capability declares:

- `EngineId`;
- `EngineVersion`;
- `ActionSetHash`;
- accepted graph artifact formats;
- accepted measure-config encoding versions; and
- accepted measurement protocol versions.

The version 1 identity and capability shapes are:

```proto
message EngineIdentity {
  bytes engine_id = 1;
  bytes engine_version = 2;
  bytes action_set_hash = 3;
}

enum GraphArtifactFormatKind {
  GRAPH_ARTIFACT_FORMAT_UNSPECIFIED = 0;
  GRAPH_ARTIFACT_FORMAT_TEXT = 1;
  GRAPH_ARTIFACT_FORMAT_JSON = 2;
  GRAPH_ARTIFACT_FORMAT_DOT = 3;
  GRAPH_ARTIFACT_FORMAT_BINARY = 4;
  GRAPH_ARTIFACT_FORMAT_ADAPTER_SPECIFIC = 5;
}

message ArtifactFormatCapability {
  GraphArtifactFormatKind format_kind = 1;
  uint32 adapter_format_id = 2;
}

message EngineCapability {
  EngineIdentity engine = 1;
  repeated ArtifactFormatCapability artifact_formats = 2;
  repeated uint32 measure_config_encodings = 3;
  repeated uint32 measurement_protocol_versions = 4;
}

message DeviceProfile {
  string platform_family = 1;
  string board_model = 2;
  string soc = 3;
  string gpu_architecture = 4;
  uint64 usable_memory_bytes = 5;
  bytes operating_system_image_digest = 6;
  string platform_release = 7;
  string cuda_version = 8;
  string gpu_driver_version = 9;
  string compiler_version = 10;
  string compiler_runtime_version = 11;
  bytes agent_image_digest = 12;
  string power_profile = 13;
  string clock_policy = 14;
  string cooling_policy = 15;
  uint32 measurement_protocol_version = 16;
}
```

All strings are bounded UTF-8 identifiers, not free-form descriptions. Image
digest fields are 32-byte BLAKE3 digests. The normalization and canonical hash
encoding must be fixed before the first profile is admitted for training.

The coordinator MUST require an exact target profile hash and exact engine
identity for version 1. It MUST NOT silently route a job to a merely similar
device.

## 10. Control messages

The following is the normative message shape. Field names may be adjusted when
the `.proto` file is introduced, but their semantics MUST be preserved.

```proto
message AgentEvent {
  oneof event {
    AgentHello hello = 1;
    AgentReady ready = 2;
    JobAccepted accepted = 3;
    JobRejected rejected = 4;
    JobHeartbeat heartbeat = 5;
    AgentDraining draining = 6;
  }
}

message CoordinatorCommand {
  oneof command {
    AgentWelcome welcome = 1;
    MeasureLease lease = 2;
    CancelLease cancel = 3;
    DrainAgent drain = 4;
    ProtocolError error = 5;
  }
}
```

### 10.1 `AgentHello`

`AgentHello` MUST be the first message on `Connect` and MUST appear exactly
once per control stream.

```proto
message AgentHello {
  uint32 protocol_major = 1;
  uint32 protocol_minor = 2;
  bytes device_id = 3;
  string agent_build = 4;
  DeviceProfile profile = 5;
  repeated EngineCapability capabilities = 6;
  repeated LeaseRecovery recoveries = 7;
}
```

Strings and repeated fields MUST have configured maximum lengths. Dynamic maps
are not used in the protocol.

Version 1 permits at most one recovery entry because an agent has one job slot.
If a new session supersedes a session with an active lease but does not report
that lease for recovery, the coordinator cancels and requeues the old lease
before accepting new readiness credit from the device.

### 10.2 `AgentWelcome`

```proto
message AgentWelcome {
  bytes session_id = 1;
  uint32 negotiated_minor = 2;
  bytes device_profile_hash = 3;
  uint64 heartbeat_interval_ms = 4;
  uint64 accept_timeout_ms = 5;
  uint64 default_lease_duration_ms = 6;
  uint64 max_artifact_bytes = 7;
  uint32 artifact_chunk_bytes = 8;
  uint32 max_measure_config_bytes = 9;
  uint32 max_engine_metadata_bytes = 10;
  uint32 max_samples_per_subject = 11;
  uint32 max_telemetry_samples = 12;
  uint32 max_bounded_string_bytes = 13;
  repeated RecoveryDisposition recoveries = 14;
}
```

The lease duration MUST exceed the heartbeat interval by enough missed
heartbeats to tolerate ordinary scheduler and network jitter. Exact values are
deployment configuration and must be verified with failure testing.

The agent MUST reject `AgentWelcome` if any limit is zero or exceeds its local
safety limits.

### 10.3 `AgentReady`

```proto
message AgentReady {
  bytes session_id = 1;
  uint32 free_slots = 2;
  DeviceTelemetry telemetry = 3;
}
```

Version 1 requires `free_slots` to be zero or one. The agent sends one only
when it has no accepted or recovering lease and its local health gate permits
new work.

While idle, the agent MUST resend `AgentReady` at the negotiated heartbeat
interval. This is the idle-session health heartbeat. It sends `free_slots = 0`
when connected but temporarily unable to accept work, and MUST send an
immediate update when its readiness changes.

This is application-level credit. The coordinator MUST NOT assign more jobs
than advertised credit, regardless of gRPC transport buffering.

### 10.4 `MeasureLease`

```proto
enum MeasureKind {
  MEASURE_KIND_UNSPECIFIED = 0;
  MEASURE_KIND_SINGLE = 1;
  MEASURE_KIND_SYMMETRIC_PAIR = 2;
}

message MeasureLease {
  bytes session_id = 1;
  bytes job_id = 2;
  bytes measurement_key = 3;
  bytes request_nonce = 4;
  bytes lease_id = 5;
  uint64 lease_duration_ms = 6;
  MeasureKind kind = 7;
  EngineIdentity engine = 8;
  bytes measure_config_hash = 9;
  uint32 measure_config_encoding = 10;
  bytes measure_config_payload = 11;
  bytes target_device_profile_hash = 12;
  repeated GraphSubject subjects = 13;
}

message GraphSubject {
  uint32 logical_index = 1;
  bytes graph_hash = 2;
  GraphArtifactRef artifact = 3;
}

message GraphArtifactRef {
  bytes artifact_digest = 1;
  uint64 artifact_size = 2;
  GraphArtifactFormatKind format_kind = 3;
  uint32 adapter_format_id = 4;
}
```

A single job contains exactly one subject at logical index 0. A symmetric pair
contains exactly two subjects at logical indices 0 and 1.

The complete job is one lease. A paired lease MUST NOT be split across agents
or partially committed.

The agent MUST validate all identities, sizes, capabilities, profile hashes,
and config encoding before accepting the lease.

### 10.5 `JobAccepted`

```proto
message JobAccepted {
  bytes session_id = 1;
  bytes job_id = 2;
  bytes lease_id = 3;
}
```

The agent MUST send `JobAccepted` before the coordinator's advertised accept
timeout. Acceptance consumes the advertised slot. Failure to accept in time
expires the lease and returns the job to the queue.

### 10.6 `JobRejected`

```proto
enum JobRejectReason {
  JOB_REJECT_UNSPECIFIED = 0;
  JOB_REJECT_PROFILE_CHANGED = 1;
  JOB_REJECT_UNSUPPORTED_ENGINE = 2;
  JOB_REJECT_UNSUPPORTED_ARTIFACT = 3;
  JOB_REJECT_UNSUPPORTED_CONFIG = 4;
  JOB_REJECT_ARTIFACT_TOO_LARGE = 5;
  JOB_REJECT_INSUFFICIENT_MEMORY = 6;
  JOB_REJECT_NOT_READY = 7;
  JOB_REJECT_AGENT_INTERNAL = 8;
}

message JobRejected {
  bytes session_id = 1;
  bytes job_id = 2;
  bytes lease_id = 3;
  JobRejectReason reason = 4;
  string bounded_message = 5;
}
```

A rejection occurs before measurement begins. It does not count as a graph
measurement and cannot produce replay reward.

Repeated capability-related rejections indicate a registry or scheduling bug;
the coordinator SHOULD quarantine the agent session.

### 10.7 `JobHeartbeat`

```proto
enum JobPhase {
  JOB_PHASE_UNSPECIFIED = 0;
  JOB_PHASE_FETCH_ARTIFACT = 1;
  JOB_PHASE_VERIFY_ARTIFACT = 2;
  JOB_PHASE_COMPILE = 3;
  JOB_PHASE_CAPTURE = 4;
  JOB_PHASE_WARMUP = 5;
  JOB_PHASE_MEASURE = 6;
  JOB_PHASE_VALIDATE = 7;
  JOB_PHASE_PERSIST_REPORT = 8;
  JOB_PHASE_SUBMIT_REPORT = 9;
}

message JobHeartbeat {
  bytes session_id = 1;
  bytes job_id = 2;
  bytes lease_id = 3;
  JobPhase phase = 4;
  uint64 phase_elapsed_ms = 5;
  DeviceTelemetry telemetry = 6;
}
```

Receipt of a valid heartbeat renews the lease for its lease duration. The
coordinator uses its monotonic clock; agent and coordinator wall clocks need not
agree.

A heartbeat reports liveness, not success. The coordinator MAY cancel a lease
whose telemetry violates an immediate safety condition.

### 10.8 Cancellation and draining

```proto
enum CancelReason {
  CANCEL_UNSPECIFIED = 0;
  CANCEL_CALLER_DEADLINE = 1;
  CANCEL_RUN_ABORTED = 2;
  CANCEL_DEVICE_UNHEALTHY = 3;
  CANCEL_ADMINISTRATIVE = 4;
  CANCEL_SUPERSEDED = 5;
}

message CancelLease {
  bytes job_id = 1;
  bytes lease_id = 2;
  CancelReason reason = 3;
}
```

After cancellation, the agent MUST stop at the earliest safe cancellation
point, terminate child compilation processes where safe, persist a cancelled
attempt report if no report is already persisted, and stop producing new
measurement samples. A non-preemptible GPU operation may finish, but its result
MUST NOT be presented as a new successful attempt after cancellation.

Cancellation and result commitment are serialized by the coordinator. If
commitment wins, a later cancel has no effect. If cancellation wins, a later
success report for that lease is stale. An agent MUST NOT overwrite an already
persisted report to change the outcome; it submits the immutable report and
obeys the coordinator's disposition.

`DrainAgent` prevents new leases. The current lease may complete unless the
command explicitly includes cancellation. An agent entering local shutdown
sends `AgentDraining` and advertises zero free slots.

### 10.9 Control-stream sequencing

The control stream follows these ordering rules:

1. The agent sends exactly one `AgentHello`.
2. The coordinator sends exactly one `AgentWelcome` or terminates the RPC.
3. Every recovery receives a disposition before new work is offered.
4. An idle agent sends periodic `AgentReady` messages.
5. The coordinator sends a lease only against available readiness credit.
6. The agent sends exactly one `JobAccepted` or `JobRejected` for the lease.
7. An accepted lease produces heartbeats until report persistence,
   cancellation, stream loss, or agent failure.
8. After a terminal result acknowledgement, the agent may advertise readiness
   again.

Messages that violate this ordering are protocol errors. Each side MUST use a
bounded outgoing control queue and continue reading while writes are pending so
bidirectional flow control cannot deadlock the connection.

## 11. Artifact transfer

```proto
message FetchArtifactRequest {
  bytes session_id = 1;
  bytes job_id = 2;
  bytes lease_id = 3;
  bytes artifact_digest = 4;
  uint64 offset = 5;
}

message ArtifactChunk {
  bytes artifact_digest = 1;
  uint64 total_size = 2;
  uint64 offset = 3;
  bytes data = 4;
}
```

The coordinator MUST authorize the requested digest against the active lease.
An agent cannot fetch arbitrary artifacts.

Chunks MUST:

- be no larger than the negotiated chunk size;
- be sent in strictly increasing contiguous offset order;
- report one immutable total size; and
- remain within the negotiated maximum artifact size.

The end of the gRPC stream marks the end of the artifact. The number of bytes
received MUST equal `total_size`, and the agent MUST verify the artifact digest
before importing or compiling it.

An interrupted transfer MAY resume from an already persisted contiguous
offset. The final digest still covers the complete artifact.

Agents MAY cache verified artifact bytes by digest. Cache corruption MUST be
handled as a cache miss, never as a compiler input.

## 12. Measurement report

`SubmitResult` carries both successful and failed attempts. A successful gRPC
status means the coordinator processed the report; it does not imply that the
measurement itself succeeded.

```proto
enum MeasureAttemptOutcome {
  MEASURE_OUTCOME_UNSPECIFIED = 0;
  MEASURE_OUTCOME_SUCCEEDED = 1;
  MEASURE_OUTCOME_ENVIRONMENT_INVALID = 2;
  MEASURE_OUTCOME_UNSTABLE = 3;
  MEASURE_OUTCOME_COMPILE_FAILED = 4;
  MEASURE_OUTCOME_CAPTURE_FAILED = 5;
  MEASURE_OUTCOME_EXECUTION_FAILED = 6;
  MEASURE_OUTCOME_TIMEOUT = 7;
  MEASURE_OUTCOME_OUT_OF_MEMORY = 8;
  MEASURE_OUTCOME_CANCELLED = 9;
  MEASURE_OUTCOME_UNSUPPORTED = 10;
  MEASURE_OUTCOME_AGENT_INTERNAL = 11;
}

message SubmitResultRequest {
  bytes current_session_id = 1;
  MeasureReport report = 2;
}

message MeasureReport {
  bytes origin_session_id = 1;
  bytes device_id = 2;
  bytes device_profile_hash = 3;
  bytes job_id = 4;
  bytes measurement_key = 5;
  bytes request_nonce = 6;
  bytes lease_id = 7;
  EngineIdentity engine = 8;
  bytes measure_config_hash = 9;
  MeasureKind kind = 10;
  MeasureAttemptOutcome outcome = 11;
  repeated SubjectMeasurement subjects = 12;
  repeated MeasurementSample samples = 13;
  repeated TelemetrySample telemetry = 14;
  FailureDetail failure = 15;
}

message SubjectMeasurement {
  uint32 logical_index = 1;
  bytes graph_hash = 2;
  optional uint64 compile_elapsed_ns = 3;
  optional uint64 capture_elapsed_ns = 4;
  optional double scalar_reward = 5;
  bytes engine_metadata = 6;
}

message MeasurementSample {
  uint32 schedule_index = 1;
  uint32 logical_index = 2;
  uint32 subject_sample_index = 3;
  uint64 gpu_elapsed_ns = 4;
  uint64 end_to_end_elapsed_ns = 5;
}

message TelemetrySample {
  uint64 attempt_elapsed_ms = 1;
  JobPhase phase = 2;
  DeviceTelemetry telemetry = 3;
}

message FailureDetail {
  uint32 engine_error_code = 1;
  string bounded_message = 2;
  bool agent_considers_retriable = 3;
}
```

Nanoseconds are used on the wire to avoid rounding raw measurements. Conversion
to the current `f32` millisecond representation occurs only when constructing a
validated `MeasureResult`.

The report MUST include the exact graph hash, engine identity, measurement
config hash, device profile, measurement key, request nonce, job, lease, and
logical subject order from the lease. A mismatch invalidates the report.

`origin_session_id` identifies the session that accepted the lease and remains
part of the persisted report. `current_session_id` identifies the session
submitting it. They differ after a successful recovery. The coordinator MUST
accept that difference only when Section 14.3 explicitly rebound the same
`lease_id` to the current session.

A successful report MUST contain every subject, every required phase duration,
the configured sample counts, and a finite scalar reward for every subject. A
failed report MUST still contain every subject identity, but it MAY omit phases
that never began and MAY contain partial samples. A successful report MUST omit
`FailureDetail`; every non-success outcome MUST include it.

The agent MUST retain all raw samples used to derive a reward. It MUST NOT
silently discard timing outliers. If the engine applies a documented robust
statistic, the report still contains the unfiltered samples.

`scalar_reward` remains engine-owned. The main-process compiler adapter MUST
validate or recompute all derivable statistics and verify that every reward is
finite before constructing `MeasureResult`.

Engine metadata and failure messages MUST have configured byte limits.

### 12.1 Paired reports

A symmetric pair is atomic:

- both subject records MUST be present;
- both subjects MUST have the requested number of valid samples;
- samples MUST preserve global execution order through `schedule_index`;
- both subjects MUST use the same device, origin session, lease, profile, and
  measurement configuration; and
- failure of either subject invalidates the pair.

The measurement config defines a balanced deterministic execution order, such
as alternating ABBA blocks. The coordinator MUST NOT decompose the pair into
independent jobs.

### 12.2 Device telemetry

`DeviceTelemetry` MUST include typed fields for all platform signals used by
the validity policy. For Jetson this includes, where available:

- GPU, CPU, and memory-controller clocks;
- GPU and relevant thermal-zone temperatures;
- power draw or power-rail readings;
- throttling and undervoltage flags;
- free device and host memory;
- active power mode; and
- monotonic time since boot.

A version 1 shape is:

```proto
message DeviceTelemetry {
  uint64 monotonic_uptime_ms = 1;
  optional uint64 gpu_clock_hz = 2;
  optional uint64 memory_controller_clock_hz = 3;
  optional uint64 free_device_memory_bytes = 4;
  optional uint64 free_host_memory_bytes = 5;
  optional uint64 throttle_flags = 6;
  string active_power_mode = 7;
  repeated ThermalReading temperatures = 8;
  repeated PowerRailReading power_rails = 9;
}

message ThermalReading {
  string zone = 1;
  sint32 millidegrees_celsius = 2;
}

message PowerRailReading {
  string rail = 1;
  uint64 milliwatts = 2;
}
```

Thermal-zone names, power-rail names, and repeated reading counts are bounded
by the negotiated protocol limits.

Unknown or unavailable readings are represented explicitly, not as zero.

Telemetry MUST be captured before compilation, before timed execution, during
timed execution at the configured interval, and after timed execution. The
measure config determines which violations make a result invalid.

## 13. Result acknowledgement and idempotency

```proto
enum ResultDisposition {
  RESULT_DISPOSITION_UNSPECIFIED = 0;
  RESULT_JOB_COMMITTED = 1;
  RESULT_ATTEMPT_RECORDED = 2;
  RESULT_ALREADY_COMMITTED = 3;
  RESULT_STALE_LEASE = 4;
  RESULT_UNKNOWN_JOB = 5;
  RESULT_INVALID_REPORT = 6;
}

message SubmitResultAck {
  ResultDisposition disposition = 1;
  bool agent_may_delete_report = 2;
  string bounded_message = 3;
}
```

The agent MUST atomically persist a complete `MeasureReport` locally before its
first submission. It wraps that immutable report with the current session ID on
each submission and MUST retain it until an acknowledgement explicitly sets
`agent_may_delete_report`.

The coordinator sets `agent_may_delete_report` only after the report or its
disposition is durably recorded. It is true for committed, already committed,
attempt-recorded, stale, unknown-job, and invalid-report dispositions. A
coordinator that cannot durably record the disposition returns a non-OK gRPC
status instead of an acknowledgement.

The coordinator handles submissions as follows:

- A valid success or permanent failure atomically commits the job and returns
  `RESULT_JOB_COMMITTED`.
- A valid retriable failure records the attempt, requeues the job according to
  retry policy, and returns `RESULT_ATTEMPT_RECORDED`.
- A duplicate submission after commitment returns
  `RESULT_ALREADY_COMMITTED` and never overwrites the committed report.
- A result for an expired or superseded lease returns `RESULT_STALE_LEASE`.
- A structurally or semantically invalid report returns
  `RESULT_INVALID_REPORT`; repeated invalid reports quarantine the session.
- A coordinator persistence failure returns a non-OK retriable gRPC status and
  no acknowledgement.

Only `RESULT_JOB_COMMITTED` or `RESULT_ALREADY_COMMITTED` can satisfy the
gateway's pending measurement. A retriable attempt does not produce replay
reward.

## 14. Connection and recovery state machines

### 14.1 Agent state

```text
DISCONNECTED
    │ connect + mTLS
    ▼
NEGOTIATING
    │ Hello / Welcome
    ▼
READY ◀────────────────────────────────────┐
    │ lease                                │ terminal acknowledgement
    ▼                                      │
LEASE_OFFERED ── reject ───────────────────┤
    │ accept                               │
    ▼                                      │
RUNNING ── heartbeat ── RUNNING            │
    │                                      │
    ▼                                      │
REPORT_PERSISTED ── SubmitResult ──────────┘
```

Any connected state may enter `DRAINING`. A stream failure enters
`DISCONNECTED` without changing whether a lease is locally running or a report
is locally persisted.

### 14.2 Job state

```text
QUEUED
   │ compatible ready agent
   ▼
LEASED
   │ JobAccepted
   ▼
RUNNING
   │ report
   ▼
ATTEMPT_RECORDED
   ├── valid success/permanent failure ── COMMITTED
   └── retriable failure ──────────────── QUEUED
```

An offer or active lease that expires returns to `QUEUED` unless the caller
deadline has expired or retry policy is exhausted.

### 14.3 Reconnect reconciliation

`AgentHello.recoveries` reports every locally running lease or persisted report.
Each entry contains:

```proto
message LeaseRecovery {
  bytes job_id = 1;
  bytes lease_id = 2;
  JobPhase phase = 3;
  bool report_persisted = 4;
}

enum RecoveryAction {
  RECOVERY_UNSPECIFIED = 0;
  RECOVERY_CONTINUE = 1;
  RECOVERY_SUBMIT_REPORT = 2;
  RECOVERY_CANCEL = 3;
  RECOVERY_ALREADY_COMMITTED = 4;
}

message RecoveryDisposition {
  bytes job_id = 1;
  bytes lease_id = 2;
  RecoveryAction action = 3;
}
```

The agent MUST take no new job until every recovery has a disposition.

- `CONTINUE` rebinds the lease to the new session and restarts heartbeat
  renewal.
- `SUBMIT_REPORT` directs the agent to resubmit its persisted report.
- `CANCEL` prevents a stale execution from being treated as current.
- `ALREADY_COMMITTED` allows the agent to delete its persisted report.

If the coordinator has already reassigned an expired lease, it MUST return
`CANCEL` even if the original agent is still running.

## 15. Scheduling and fleet behavior

The coordinator maintains these agent states:

```text
CONNECTING
READY
BUSY
DRAINING
QUARANTINED
OFFLINE
```

A job is eligible for an agent only when all of the following match:

- target device profile hash;
- engine ID and version;
- action-set hash;
- artifact format;
- measure-config encoding; and
- measurement protocol implementation.

The coordinator MUST apply compatibility filtering before load selection. A
simple oldest-eligible-job and least-outstanding-agent policy is sufficient for
version 1.

The transport does not use generic gRPC client-side load balancing to choose an
agent. Agents are clients; the coordinator explicitly assigns leases based on
capabilities and health.

### 15.1 Hardware pools

Measurements from different profile hashes are not interchangeable. Adding a
new device type creates a new target pool unless a higher-level experiment
explicitly defines a multi-profile objective.

The coordinator MUST NOT fall back from one requested profile to another.

### 15.2 Single-device exclusivity

Version 1 permits one accepted job per physical GPU. Compilation, capture,
warm-up, and timed execution are all part of the exclusive interval. The agent
MUST NOT compile another job concurrently because CPU, memory, power, and
thermal contention can affect measurement validity.

### 15.3 Queue bounds

The coordinator has explicit bounds for:

- queued jobs;
- queued artifact bytes;
- active leases;
- retained attempt reports;
- per-agent messages; and
- per-job retries.

Reaching a bound must apply backpressure or return a typed capacity error. It
MUST NOT allocate an unbounded buffer.

## 16. Timeouts and cancellation

The system has separate timeout layers:

| Timeout | Starts | Purpose |
| --- | --- | --- |
| Caller deadline | Job submission | Bounds total queue plus execution wait |
| Offer acceptance | Lease transmission | Detects an agent that did not accept |
| Lease timeout | Assignment or last heartbeat | Detects lost ownership |
| Phase timeout | Phase start | Bounds compile, capture, warm-up, or execution |
| RPC deadline | Finite RPC start | Bounds artifact/result network operation |

`MeasureOptions.timeout_ms`, when present, is the caller deadline from gateway
submission until terminal completion. Engine-owned measure-config fields define
phase timeouts. Lease and RPC timeouts are deployment controls and MUST NOT
change measurement identity.

All timeout decisions use monotonic clocks local to the deciding process.
Wall-clock timestamps are audit metadata only.

## 17. Attempt failure and retry policy

The coordinator, not the agent, makes the final retry decision. The agent's
`agent_considers_retriable` field is advisory.

The default classification is:

| Outcome | Default action |
| --- | --- |
| `SUCCEEDED` | Validate and commit |
| `ENVIRONMENT_INVALID` | Retry; quarantine on repetition |
| `UNSTABLE` | Retry; quarantine or fail after retry bound |
| `COMPILE_FAILED` | Commit permanent failure |
| `CAPTURE_FAILED` | Commit permanent failure unless classified transient |
| `EXECUTION_FAILED` | Retry only for known transient device errors |
| `TIMEOUT` | Retry according to phase and attempt bound |
| `OUT_OF_MEMORY` | Commit permanent failure for the exact profile/config |
| `CANCELLED` | Requeue only if the caller remains active |
| `UNSUPPORTED` | Treat as scheduling/configuration error |
| `AGENT_INTERNAL` | Retry elsewhere and quarantine on repetition |

Retry limits are configuration. Exhausting the retry limit produces a terminal
invalid `MeasureResult` and never a fabricated reward.

## 18. Measurement validity requirements

The wire protocol transports evidence; the concrete engine's measurement
protocol defines validity. At minimum, a compiler measurement MUST:

1. verify the exact device profile and active power/clock policy;
2. verify artifact and measure-config hashes;
3. compile on the target device or use an exactly keyed compiled artifact;
4. record compilation time independently;
5. capture and instantiate the GPU graph;
6. perform explicit upload or an untimed first launch when required;
7. execute the configured warm-up protocol;
8. collect the configured raw GPU and end-to-end samples;
9. collect pre-, during-, and post-measurement telemetry;
10. reject thermal, power, throttle, clock, or stability violations; and
11. return a finite engine-owned scalar reward only for a valid result.

Measurement-result caching and cross-request measurement deduplication are
disabled in version 1. Every separately admitted request has a new
`request_nonce` and requires physical measurement unless it terminates before
execution. Retries of that request keep the same job ID. Verified graph and
compiled artifacts MAY still be cached.

A compiled-artifact cache key MUST include at least:

- graph hash;
- artifact digest;
- action-set hash;
- engine version;
- measure-config hash;
- target device profile hash;
- compiler/runtime identity; and
- compiled-artifact format version.

If compilation time is part of the reward, the compiled-artifact cache MUST NOT
be used for that measurement.

## 19. Symmetric selfplay requirements

Player-one and player-two final graphs MUST be measured as one
`MEASURE_KIND_SYMMETRIC_PAIR` job on one physical device.

This is required because replay currently derives a categorical outcome by
comparing their terminal scalar rewards. Measuring the two subjects on
different boards would allow device bias to change the training label.

The paired measurement config MUST define:

- compilation and capture order;
- warm-up policy for both subjects;
- a balanced interleaved timing schedule;
- sample pairing semantics;
- a stability criterion; and
- the practical tie or indifference rule used by replay projection.

Exact floating-point equality MUST NOT be the only tie rule for hardware
measurements. The threshold or statistical decision rule is part of the
measure config and therefore part of `measure_config_hash`.

## 20. Security

### 20.1 Device enrollment

Each physical device receives:

- one provisioned `device_id`;
- one client certificate and private key;
- the coordinator CA certificate; and
- the coordinator endpoint.

The coordinator maintains an allowlist mapping certificate identity to
`device_id`. Unknown, revoked, expired, or mismatched identities are rejected
before protocol negotiation.

Certificate rotation MUST preserve `device_id` and replace the allowlist entry
atomically.

### 20.2 Authorization

An authenticated agent may:

- open its own control session;
- fetch only artifacts referenced by its active or recovering lease; and
- submit reports only for its active or recovering lease.

It may not enumerate jobs, artifacts, other devices, or other reports.

### 20.3 Input validation

Both sides MUST validate:

- identifier widths;
- protocol and encoding versions;
- enum values;
- repeated-field counts;
- string, metadata, sample, telemetry, and artifact byte limits;
- finite scalar rewards;
- monotonic and unique sample indices;
- graph and config identity matches; and
- all checked arithmetic involving sizes and offsets.

Malformed input terminates the affected RPC. Repeated malformed input
quarantines the session.

### 20.4 Compiler isolation

Graph artifacts are untrusted compiler input even when they come from an
authenticated coordinator. The agent SHOULD execute compilation in a
non-privileged child process with bounded CPU, memory, disk, time, and network
access. GPU access should be granted only to the measurement worker that needs
it.

The agent service itself SHOULD remain alive when a compiler child crashes.

## 21. Persistence guarantees

Version 1 provides these guarantees:

- immutable graph artifacts remain available while a queued or active job
  references them;
- a completed agent report survives agent process restart until acknowledged;
- a committed coordinator report survives coordinator restart; and
- commitment is atomic with the receipt-ledger write.

Artifact bytes need not survive coordinator restart in the initial in-memory
deployment because queued jobs and active search tasks are not restart-durable.
A deployment using durable queued jobs MUST also use a durable artifact store.

Version 1 does not require queued jobs or active search tasks to survive a main
process restart. Current search tasks retain process-local graph handles, so a
durable job broker alone would not provide end-to-end recovery.

If detached episode recovery is added later, it requires a separate storage
spec for portable episode state and cannot be inferred from this protocol.

## 22. Replay and receipts

The gateway converts a committed portable report to
`MeasureResult<E::Graph>` using the original local graph handle. It verifies:

- graph hash;
- engine identity;
- measure-config hash;
- target profile hash;
- finite statistics and reward;
- success and validity status; and
- receipt commitment.

Replay admission occurs only after that conversion succeeds.

The full report, telemetry, and raw samples SHOULD live once in the measurement
receipt ledger. Replay records SHOULD eventually carry a compact receipt ID and
summary rather than cloning raw samples into every replay row.

## 23. Observability

The coordinator exposes aggregate metrics for:

- connected, ready, busy, draining, quarantined, and offline agents;
- queue depth and oldest queued age by target profile;
- active leases and lease expirations;
- job throughput by outcome and profile;
- compile, capture, and execution duration distributions;
- result retries and duplicate submissions;
- artifact bytes and cache hits;
- environment-invalid and unstable attempt rates; and
- device calibration or sentinel drift.

High-cardinality values such as `job_id`, `graph_hash`, and `lease_id` belong in
structured logs and traces, not metric labels.

Every job log record includes `job_id`; every attempt record additionally
includes `lease_id`, `device_id`, and `session_id`.

## 24. Conformance tests

An implementation is not conformant until automated tests cover:

### 24.1 Negotiation and security

- valid mutual-TLS enrollment;
- unknown and mismatched device identity rejection;
- unsupported protocol-major rejection;
- minor-version negotiation;
- incompatible engine and device-profile rejection; and
- message and artifact size limits.

### 24.2 Lease lifecycle

- separate calls with the same measurement key receive distinct job IDs;
- resubmitting one logical call preserves its request nonce and job ID;
- ready credit prevents over-assignment;
- accept timeout requeues an offer;
- heartbeat renews a lease;
- missing heartbeat expires and requeues a lease;
- cancellation prevents successful commitment;
- draining prevents new assignments; and
- a paired job is never split.

### 24.3 Artifact transfer

- complete verified transfer;
- interrupted transfer resumes from an offset;
- wrong offset is rejected;
- digest mismatch prevents compilation;
- unauthorized artifact fetch is rejected; and
- cached corruption becomes a cache miss.

### 24.4 Results and recovery

- valid result commits once;
- identical resubmission returns already committed;
- stale lease cannot overwrite a committed result;
- retriable failure records the attempt and requeues;
- permanent failure commits without reward;
- coordinator persistence failure causes safe resubmission;
- network loss during compilation recovers or cancels deterministically;
- network loss after local report persistence resubmits safely; and
- replay receives no row before terminal measurement commitment.

### 24.5 Report validation

- wrong graph, engine, config, profile, job, or lease identity is rejected;
- non-finite reward is rejected;
- missing, duplicate, or out-of-order sample indices are rejected;
- incomplete symmetric pairs are rejected;
- telemetry required by the config is present; and
- engine metadata and failure messages are bounded.

### 24.6 Measurement validation

Before accepting a new device profile for training, run a fixed corpus:

- repeatedly on one device;
- across cold and steady-state conditions;
- across device reboots;
- across every nominally equivalent device; and
- with paired subject order reversed.

The resulting data establishes stability thresholds, sentinel limits, and any
practical tie band. These values MUST be measured rather than assumed.

## 25. Initial implementation sequence

1. Implement the protocol types, canonical measurement key and job ID, and
   in-memory coordinator state machine without networking.
2. Add conformance tests for leases, duplicate reports, and paired jobs.
3. Add gRPC transport and mutual-TLS enrollment.
4. Add artifact streaming and agent-side verified storage.
5. Add a single-device agent with a deterministic non-GPU fixture backend.
6. Integrate orchestrator parking and resumption.
7. Validate the complete path with Whittle before adding the compiler engine.
8. Add one real edge device and establish the measurement protocol.
9. Add the remaining devices, calibration, quarantine, and operational metrics.

No message broker, Kubernetes scheduler, multi-coordinator consensus, or remote
measurement-result cache is part of the initial implementation.

## 26. Decisions required before implementation

The transport and failure semantics are fixed by this draft. These
engine-specific decisions still require concrete values or encodings:

1. The compiler engine's canonical measure-config payload.
2. The exact target device profile fields and normalization rules.
3. Whether scalar reward represents GPU execution time, synchronized
   end-to-end execution time, or a combination.
4. Whether compilation or graph capture contributes to scalar reward.
5. The paired sampling schedule and evidence-based tie rule.
6. The device telemetry fields available on the first target image.
7. Coordinator and agent queue, size, timeout, and retry bounds.

Changing any item that changes measurement meaning requires a new
`measure_config_hash` or `device_profile_hash`, not an in-place reinterpretation
of existing results.

## 27. Non-normative references

- [gRPC flow control](https://grpc.io/docs/guides/flow-control/)
- [gRPC authentication](https://grpc.io/docs/guides/auth/)
- [gRPC keepalive](https://grpc.io/docs/guides/keepalive/)
- [gRPC deadlines](https://grpc.io/docs/guides/deadlines/)
- [gRPC retry behavior](https://grpc.io/docs/guides/retry/)
- [gRPC health checking](https://grpc.io/docs/guides/health-checking/)
- [NVIDIA Jetson tegrastats](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)
- [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
