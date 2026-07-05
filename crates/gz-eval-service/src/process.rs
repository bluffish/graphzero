use crate::{
    BackendOutputs, FRAME_ERROR, FRAME_EVAL, FRAME_EVAL_RESULT, FRAME_HELLO, FRAME_HELLO_ACK,
    FRAME_PING, FRAME_PONG, FeatureEvalBackend, Hello, HelloAck, PROTOCOL_VERSION, PendingBatch,
    ServiceError, ServiceResult, decode_error, read_frame, write_frame,
};
use gz_engine::ModelVersion;
use gz_features::{decode_outputs, validate_batch_action_counts};
use std::io::ErrorKind;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct EvaluatorProcessConfig {
    pub python: PathBuf,
    pub module: String,
    pub working_dir: PathBuf,
    pub socket_path: PathBuf,
    pub ready_timeout: Duration,
    pub io_timeout: Duration,
    pub extra_args: Vec<String>,
}

impl Default for EvaluatorProcessConfig {
    fn default() -> Self {
        Self {
            python: PathBuf::from("python3"),
            module: "gz.evaluator".to_owned(),
            working_dir: PathBuf::new(),
            socket_path: PathBuf::new(),
            ready_timeout: Duration::from_secs(10),
            io_timeout: Duration::from_secs(30),
            extra_args: Vec::new(),
        }
    }
}

pub struct EvaluatorProcess {
    child: Child,
    config: EvaluatorProcessConfig,
    connect_started: bool,
}

impl EvaluatorProcess {
    pub fn spawn(config: EvaluatorProcessConfig) -> ServiceResult<Self> {
        if config.socket_path.as_os_str().is_empty() {
            return Err(ServiceError::io("missing evaluator socket path"));
        }

        let mut command = Command::new(&config.python);
        command
            .arg("-m")
            .arg(&config.module)
            .arg("--socket")
            .arg(&config.socket_path)
            .args(&config.extra_args)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        if !config.working_dir.as_os_str().is_empty() {
            command.current_dir(&config.working_dir);
        }

        let child = command.spawn().map_err(|error| {
            ServiceError::io(format!(
                "failed to spawn {} -m {} --socket {}: {error}",
                config.python.display(),
                config.module,
                config.socket_path.display()
            ))
        })?;

        Ok(Self {
            child,
            config,
            connect_started: false,
        })
    }

    pub fn connect(&mut self, hello: &Hello) -> ServiceResult<ProcessBackend> {
        if self.connect_started {
            return Err(ServiceError::protocol(
                "evaluator process already connected",
            ));
        }
        self.connect_started = true;

        let stream = self.connect_stream()?;
        ProcessBackend::connect_stream(stream, hello, self.config.io_timeout)
    }

    #[must_use]
    pub fn id(&self) -> u32 {
        self.child.id()
    }

    pub fn try_wait(&mut self) -> ServiceResult<Option<ExitStatus>> {
        self.child
            .try_wait()
            .map_err(|error| ServiceError::io(error.to_string()))
    }

    pub fn wait(&mut self) -> ServiceResult<ExitStatus> {
        self.child
            .wait()
            .map_err(|error| ServiceError::io(error.to_string()))
    }

    fn connect_stream(&mut self) -> ServiceResult<UnixStream> {
        let deadline = Instant::now() + self.config.ready_timeout;
        loop {
            if let Some(status) = self.try_wait()? {
                return Err(ServiceError::io(format!(
                    "evaluator process exited before connect: {status}"
                )));
            }

            match UnixStream::connect(&self.config.socket_path) {
                Ok(stream) => return Ok(stream),
                Err(error)
                    if matches!(
                        error.kind(),
                        ErrorKind::NotFound | ErrorKind::ConnectionRefused
                    ) && Instant::now() < deadline =>
                {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(error) => return Err(ServiceError::io(error.to_string())),
            }
        }
    }
}

impl Drop for EvaluatorProcess {
    fn drop(&mut self) {
        match self.child.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) => {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
            Err(_) => {
                let _ = self.child.wait();
            }
        }
    }
}

pub struct ProcessBackend {
    stream: UnixStream,
    read_buf: Vec<u8>,
    write_buf: Vec<u8>,
    batch_id: u64,
    model_version: ModelVersion,
}

impl ProcessBackend {
    pub fn connect_stream(
        mut stream: UnixStream,
        hello: &Hello,
        io_timeout: Duration,
    ) -> ServiceResult<Self> {
        stream
            .set_read_timeout(Some(io_timeout))
            .map_err(|error| ServiceError::io(error.to_string()))?;
        stream
            .set_write_timeout(Some(io_timeout))
            .map_err(|error| ServiceError::io(error.to_string()))?;

        let mut read_buf = Vec::new();
        let mut write_buf = Vec::new();
        let mut encode_buf = Vec::new();
        hello.encode(&mut encode_buf);
        write_frame(&mut stream, &mut write_buf, FRAME_HELLO, &[&encode_buf])?;
        let (frame_type, payload) = read_frame(&mut stream, &mut read_buf)?;
        match frame_type {
            FRAME_HELLO_ACK => {
                let ack = HelloAck::decode(payload)?;
                if ack.protocol_version != PROTOCOL_VERSION {
                    return Err(ServiceError::handshake("protocol version mismatch"));
                }
                Ok(Self {
                    stream,
                    read_buf,
                    write_buf,
                    batch_id: 0,
                    model_version: ack.model_version,
                })
            }
            FRAME_ERROR => {
                let (code, message) = decode_error(payload)?;
                Err(ServiceError::handshake(format!(
                    "server error {code}: {message}"
                )))
            }
            _ => Err(ServiceError::protocol("expected HELLO_ACK")),
        }
    }

    pub fn ping(&mut self) -> ServiceResult<()> {
        let nonce = self.batch_id ^ 0x9e37_79b9_7f4a_7c15;
        let nonce_bytes = nonce.to_le_bytes();
        write_frame(
            &mut self.stream,
            &mut self.write_buf,
            FRAME_PING,
            &[&nonce_bytes],
        )?;
        let (frame_type, payload) = read_frame(&mut self.stream, &mut self.read_buf)?;
        match frame_type {
            FRAME_PONG => {
                if payload.len() != 8 {
                    return Err(ServiceError::protocol("bad PONG length"));
                }
                let actual = u64::from_le_bytes(payload.try_into().expect("pong length checked"));
                if actual != nonce {
                    return Err(ServiceError::protocol("PONG nonce mismatch"));
                }
                Ok(())
            }
            FRAME_ERROR => Err(error_payload(payload)),
            _ => Err(ServiceError::protocol("expected PONG")),
        }
    }

    pub fn model_version(&self) -> ModelVersion {
        self.model_version
    }
}

impl FeatureEvalBackend for ProcessBackend {
    fn eval(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<BackendOutputs> {
        let pending = self.submit(batch_bytes, action_counts)?;
        self.receive(pending)
    }

    /// Sends the batch and returns immediately; the evaluator process
    /// stages and runs it while the caller collects the next batch. The
    /// stream is FIFO, so one submitted batch may be outstanding while
    /// its predecessor's result is read.
    fn submit(&mut self, batch_bytes: &[u8], action_counts: &[u32]) -> ServiceResult<PendingBatch> {
        validate_batch_action_counts(batch_bytes, action_counts)
            .map_err(|error| ServiceError::protocol(error.to_string()))?;

        let batch_id = self.batch_id;
        self.batch_id = self.batch_id.wrapping_add(1);
        let batch_id_bytes = batch_id.to_le_bytes();
        write_frame(
            &mut self.stream,
            &mut self.write_buf,
            FRAME_EVAL,
            &[&batch_id_bytes, batch_bytes],
        )?;

        Ok(PendingBatch::InFlight {
            batch_id,
            action_counts: action_counts.to_vec(),
        })
    }

    fn receive(&mut self, pending: PendingBatch) -> ServiceResult<BackendOutputs> {
        let (batch_id, action_counts) = match pending {
            PendingBatch::Ready(outputs) => return Ok(outputs),
            PendingBatch::InFlight {
                batch_id,
                action_counts,
            } => (batch_id, action_counts),
        };

        let (frame_type, payload) = read_frame(&mut self.stream, &mut self.read_buf)?;
        match frame_type {
            FRAME_EVAL_RESULT => decode_eval_result(payload, batch_id, &action_counts),
            FRAME_ERROR => Err(error_payload(payload)),
            _ => Err(ServiceError::protocol("expected EVAL_RESULT")),
        }
    }
}

fn decode_eval_result(
    payload: &[u8],
    expected_batch_id: u64,
    action_counts: &[u32],
) -> ServiceResult<BackendOutputs> {
    if payload.len() < 24 {
        return Err(ServiceError::protocol("EVAL_RESULT frame truncated"));
    }
    let batch_id = u64::from_le_bytes(payload[0..8].try_into().expect("slice checked"));
    if batch_id != expected_batch_id {
        return Err(ServiceError::protocol("batch id mismatch"));
    }

    let model_version = ModelVersion::from_bytes(payload[8..24].try_into().expect("slice checked"));
    let rows = decode_outputs(&payload[24..], action_counts)
        .map_err(|error| ServiceError::protocol(error.to_string()))?;
    Ok(BackendOutputs {
        model_version,
        rows,
    })
}

fn error_payload(payload: &[u8]) -> ServiceError {
    match decode_error(payload) {
        Ok((code, message)) => ServiceError::backend(code, message),
        Err(error) => error,
    }
}
