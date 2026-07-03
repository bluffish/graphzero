use gz_features::{
    ENCODING_VERSION, FeatureCollator, FeatureRow, FeatureSchema, RowTargets, decode_feature_row,
    encode_feature_schema_config, encode_training_targets, validate_feature_row_header,
};
use gz_replay::{ReplayError, ReplayStore, SampleConfig};
use std::io::{ErrorKind, Read, Write};
use std::num::{NonZeroU64, NonZeroUsize};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::Duration;

pub const SAMPLE_PROTOCOL_VERSION: u32 = 2;

const MAX_FRAME: usize = 256 * 1024 * 1024;
const FRAME_HELLO: u8 = 1;
const FRAME_HELLO_ACK: u8 = 2;
const FRAME_SAMPLE: u8 = 3;
const FRAME_SAMPLE_RESULT: u8 = 4;
const FRAME_ERROR: u8 = 5;

const ERROR_PROTOCOL: u32 = 1;
const ERROR_ENCODING: u32 = 2;
const ERROR_EMPTY_STORE: u32 = 3;
const ERROR_BAD_REQUEST: u32 = 4;
const ERROR_MISSING_FEATURES: u32 = 5;

#[derive(Clone, Debug)]
pub struct ReplayServeConfig {
    pub replay_dir: PathBuf,
    pub socket: PathBuf,
    pub max_batch: usize,
}

impl ReplayServeConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.replay_dir.as_os_str().is_empty() {
            return Err("missing required --replay-dir".to_owned());
        }
        if self.socket.as_os_str().is_empty() {
            return Err("missing required --socket".to_owned());
        }
        if self.max_batch == 0 {
            return Err("--max-batch must be greater than zero".to_owned());
        }
        Ok(())
    }
}

pub fn run(config: ReplayServeConfig) -> Result<(), String> {
    let mut server = ReplaySampleServer::bind(config)?;
    loop {
        // A connection error (client vanished, socket timeout on an idle
        // trainer) ends that connection, never the service.
        if let Err(error) = server.accept_one() {
            eprintln!("replay sample connection ended: {error}");
        }
    }
}

pub fn run_one(config: ReplayServeConfig) -> Result<(), String> {
    ReplaySampleServer::bind(config)?.accept_one()
}

/// Serve loop over a store shared with a live producer (the in-process
/// sample service of `graphzero selfplay --serve-socket`). Append and
/// sample are `&self` and internally serialized, so one process owning
/// the store dissolves the RocksDB single-writer constraint.
pub fn run_shared(
    store: std::sync::Arc<ReplayStore>,
    socket: PathBuf,
    max_batch: usize,
) -> Result<(), String> {
    let mut server = ReplaySampleServer::bind_shared(store, socket, max_batch)?;
    loop {
        // Same rule as `run`: a per-connection error must never take down
        // the producing selfplay process wrapped around this loop.
        if let Err(error) = server.accept_one() {
            eprintln!("replay sample connection ended: {error}");
        }
    }
}

struct ReplaySampleServer {
    listener: UnixListener,
    store: std::sync::Arc<ReplayStore>,
    collator: FeatureCollator,
    max_batch: NonZeroUsize,
}

impl ReplaySampleServer {
    fn bind(config: ReplayServeConfig) -> Result<Self, String> {
        config.validate()?;
        let store = ReplayStore::open(&config.replay_dir).map_err(|error| error.to_string())?;
        Self::bind_shared(std::sync::Arc::new(store), config.socket, config.max_batch)
    }

    fn bind_shared(
        store: std::sync::Arc<ReplayStore>,
        socket: PathBuf,
        max_batch: usize,
    ) -> Result<Self, String> {
        let schema_config = store
            .feature_schema()
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "store was not produced by featurized selfplay".to_owned())?;
        let schema = FeatureSchema::new(schema_config).map_err(|error| error.to_string())?;
        let max_batch = NonZeroUsize::new(max_batch)
            .ok_or_else(|| "--max-batch must be greater than zero".to_owned())?;
        let collator = FeatureCollator::new(schema, max_batch);

        if socket.exists() {
            std::fs::remove_file(&socket).map_err(|error| error.to_string())?;
        }
        let listener = UnixListener::bind(&socket).map_err(|error| error.to_string())?;

        Ok(Self {
            listener,
            store,
            collator,
            max_batch,
        })
    }

    fn accept_one(&mut self) -> Result<(), String> {
        let (mut stream, _) = self.listener.accept().map_err(|error| error.to_string())?;
        stream
            .set_read_timeout(Some(Duration::from_secs(300)))
            .map_err(|error| error.to_string())?;
        stream
            .set_write_timeout(Some(Duration::from_secs(300)))
            .map_err(|error| error.to_string())?;
        self.handle_client(&mut stream)
    }

    fn handle_client(&mut self, stream: &mut UnixStream) -> Result<(), String> {
        let mut read_buf = Vec::new();
        let mut write_buf = Vec::new();
        let mut batch_buf = Vec::new();
        let mut target_buf = Vec::new();

        let Some((frame_type, payload)) =
            read_frame(stream, &mut read_buf).map_err(|error| error.to_string())?
        else {
            return Ok(());
        };
        if frame_type != FRAME_HELLO {
            send_error(stream, &mut write_buf, ERROR_PROTOCOL, "expected HELLO")?;
            return Ok(());
        }
        if let Err(error) = self.handle_hello(payload, stream, &mut write_buf) {
            send_error(stream, &mut write_buf, error.0, error.1)?;
            return Ok(());
        }

        while let Some((frame_type, payload)) =
            read_frame(stream, &mut read_buf).map_err(|error| error.to_string())?
        {
            // A repeated HELLO re-acks with fresh produced_rows so a
            // long-lived trainer connection can watch production advance.
            if frame_type == FRAME_HELLO {
                if let Err(error) = self.handle_hello(payload, stream, &mut write_buf) {
                    send_error(stream, &mut write_buf, error.0, error.1)?;
                    return Ok(());
                }
                continue;
            }
            if frame_type != FRAME_SAMPLE {
                send_error(stream, &mut write_buf, ERROR_PROTOCOL, "expected SAMPLE")?;
                return Ok(());
            }
            match self.handle_sample(payload, &mut batch_buf, &mut target_buf) {
                Ok(()) => {
                    let gzfb_len = (batch_buf.len() as u32).to_le_bytes();
                    write_frame(
                        stream,
                        &mut write_buf,
                        FRAME_SAMPLE_RESULT,
                        &[&gzfb_len, &batch_buf, &target_buf],
                    )
                    .map_err(|error| error.to_string())?;
                }
                Err(error) => {
                    send_error(stream, &mut write_buf, error.0, error.1)?;
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    fn handle_hello(
        &self,
        payload: &[u8],
        stream: &mut UnixStream,
        write_buf: &mut Vec<u8>,
    ) -> Result<(), (u32, &'static str)> {
        if payload.len() != 8 {
            return Err((ERROR_PROTOCOL, "bad HELLO length"));
        }
        let protocol_version = u32::from_le_bytes(payload[0..4].try_into().expect("len checked"));
        let encoding_version = u32::from_le_bytes(payload[4..8].try_into().expect("len checked"));
        if protocol_version != SAMPLE_PROTOCOL_VERSION {
            return Err((ERROR_PROTOCOL, "protocol version mismatch"));
        }
        if encoding_version != ENCODING_VERSION {
            return Err((ERROR_ENCODING, "encoding version mismatch"));
        }

        let mut schema_config = Vec::new();
        encode_feature_schema_config(self.collator.schema().config(), &mut schema_config)
            .map_err(|_| (ERROR_ENCODING, "failed to encode schema config"))?;
        let (episodes, episodes_stopped) = self.store.episode_counters();
        let mut payload = Vec::with_capacity(64 + schema_config.len());
        payload.extend_from_slice(&SAMPLE_PROTOCOL_VERSION.to_le_bytes());
        payload.extend_from_slice(self.collator.schema().hash().as_bytes());
        payload.extend_from_slice(&(self.max_batch.get() as u32).to_le_bytes());
        payload.extend_from_slice(&self.store.counters().produced_rows.to_le_bytes());
        payload.extend_from_slice(&episodes.to_le_bytes());
        payload.extend_from_slice(&episodes_stopped.to_le_bytes());
        payload.extend_from_slice(&schema_config);
        write_frame(stream, write_buf, FRAME_HELLO_ACK, &[&payload])
            .map_err(|_| (ERROR_PROTOCOL, "failed to write HELLO_ACK"))
    }

    fn handle_sample(
        &mut self,
        payload: &[u8],
        batch_buf: &mut Vec<u8>,
        target_buf: &mut Vec<u8>,
    ) -> Result<(), (u32, &'static str)> {
        if payload.len() != 20 {
            return Err((ERROR_PROTOCOL, "bad SAMPLE length"));
        }
        let batch = u32::from_le_bytes(payload[0..4].try_into().expect("len checked")) as usize;
        let window = u64::from_le_bytes(payload[4..12].try_into().expect("len checked"));
        let seed = u64::from_le_bytes(payload[12..20].try_into().expect("len checked"));
        if batch == 0 || batch > self.max_batch.get() || window == 0 {
            return Err((ERROR_BAD_REQUEST, "invalid SAMPLE request"));
        }

        let rows = self
            .store
            .sample_rows(SampleConfig {
                batch: NonZeroUsize::new(batch).expect("batch checked"),
                window_rows: NonZeroU64::new(window).expect("window checked"),
                seed,
            })
            .map_err(sample_error)?;
        let mut feature_rows = Vec::<FeatureRow>::with_capacity(rows.len());
        let mut targets = Vec::<RowTargets>::with_capacity(rows.len());
        let schema_hash = self.collator.schema().hash();

        for (_, row) in rows {
            let bytes = row
                .feature_row
                .ok_or((ERROR_MISSING_FEATURES, "row is missing feature payload"))?;
            validate_feature_row_header(&bytes, &schema_hash)
                .map_err(|_| (ERROR_ENCODING, "feature row schema mismatch"))?;
            let feature_row =
                decode_feature_row(&bytes).map_err(|_| (ERROR_ENCODING, "bad feature row"))?;
            let reward = row
                .reward_target
                .ok_or((ERROR_ENCODING, "missing reward target"))?;
            targets.push(RowTargets {
                policy: row.policy_target,
                value: row.value_target,
                reward,
            });
            feature_rows.push(feature_row);
        }

        self.collator
            .collate_into(&feature_rows, batch_buf)
            .map_err(|_| (ERROR_ENCODING, "feature collation failed"))?;
        encode_training_targets(
            &targets,
            self.max_batch.get(),
            self.collator.schema().config().max_actions as usize,
            target_buf,
        )
        .map_err(|_| (ERROR_ENCODING, "target encoding failed"))?;

        Ok(())
    }
}

fn sample_error(error: ReplayError) -> (u32, &'static str) {
    match error {
        ReplayError::Empty => (ERROR_EMPTY_STORE, "replay store is empty"),
        _ => (ERROR_BAD_REQUEST, "sampling failed"),
    }
}

fn read_frame<'a>(
    stream: &mut UnixStream,
    buf: &'a mut Vec<u8>,
) -> std::io::Result<Option<(u8, &'a [u8])>> {
    let mut len = [0u8; 4];
    match stream.read_exact(&mut len) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error),
    }
    let body_len = u32::from_le_bytes(len) as usize;
    if body_len == 0 || body_len > MAX_FRAME {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "bad frame size",
        ));
    }

    if buf.len() < body_len {
        buf.resize(body_len, 0);
    }
    stream.read_exact(&mut buf[..body_len])?;
    Ok(Some((buf[0], &buf[1..body_len])))
}

fn write_frame(
    stream: &mut UnixStream,
    buf: &mut Vec<u8>,
    frame_type: u8,
    parts: &[&[u8]],
) -> std::io::Result<()> {
    let body_len = parts
        .iter()
        .try_fold(1usize, |total, part| total.checked_add(part.len()))
        .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "frame length overflow"))?;
    if body_len > MAX_FRAME {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "frame exceeds maximum size",
        ));
    }
    let frame_len = 4 + body_len;
    if buf.len() < frame_len {
        buf.resize(frame_len, 0);
    }
    buf[0..4].copy_from_slice(&(body_len as u32).to_le_bytes());
    buf[4] = frame_type;
    let mut cursor = 5;
    for part in parts {
        let end = cursor + part.len();
        buf[cursor..end].copy_from_slice(part);
        cursor = end;
    }
    stream.write_all(&buf[..frame_len])
}

fn send_error(
    stream: &mut UnixStream,
    write_buf: &mut Vec<u8>,
    code: u32,
    message: &'static str,
) -> Result<(), String> {
    let message = truncate_message(message);
    let mut payload = Vec::with_capacity(6 + message.len());
    payload.extend_from_slice(&code.to_le_bytes());
    payload.extend_from_slice(&(message.len() as u16).to_le_bytes());
    payload.extend_from_slice(message.as_bytes());
    write_frame(stream, write_buf, FRAME_ERROR, &[&payload]).map_err(|error| error.to_string())
}

fn truncate_message(message: &'static str) -> &'static str {
    const MAX: usize = 512;
    if message.len() <= MAX {
        message
    } else {
        &message[..MAX]
    }
}
