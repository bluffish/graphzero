use gz_engine::EngineIdentity;
use gz_features::{ENCODING_VERSION, FeatureSchemaHash};
use std::io::{ErrorKind, Read, Write};
use std::num::{NonZeroU64, NonZeroUsize};

pub const SAMPLE_PROTOCOL_VERSION: u32 = 13;
pub const HELLO_ACK_FIXED_LEN: usize = 224;

pub(crate) const FRAME_HELLO: u8 = 1;
pub(crate) const FRAME_HELLO_ACK: u8 = 2;
pub(crate) const FRAME_SAMPLE: u8 = 3;
pub(crate) const FRAME_SAMPLE_RESULT: u8 = 4;
pub(crate) const FRAME_ERROR: u8 = 5;

pub(crate) const ERROR_PROTOCOL: u32 = 1;
pub(crate) const ERROR_ENCODING: u32 = 2;
pub(crate) const ERROR_EMPTY_STORE: u32 = 3;
pub(crate) const ERROR_BAD_REQUEST: u32 = 4;
pub(crate) const ERROR_MISSING_FEATURES: u32 = 5;

const MAX_FRAME: usize = 256 * 1024 * 1024;

pub(crate) type ProtocolResult<T> = Result<T, (u32, &'static str)>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SampleRequest {
    pub batch: NonZeroUsize,
    pub terminal_only: bool,
    pub window_rows: NonZeroU64,
    pub seed: u64,
}

impl SampleRequest {
    pub fn decode(payload: &[u8], max_batch: NonZeroUsize) -> ProtocolResult<Self> {
        if payload.len() != 24 {
            return Err((ERROR_PROTOCOL, "bad SAMPLE length"));
        }
        let batch = u32::from_le_bytes(payload[0..4].try_into().expect("len checked")) as usize;
        let terminal_only = match u32::from_le_bytes(payload[4..8].try_into().expect("len checked"))
        {
            0 => false,
            1 => true,
            _ => return Err((ERROR_BAD_REQUEST, "invalid SAMPLE row kind")),
        };
        let window_rows = u64::from_le_bytes(payload[8..16].try_into().expect("len checked"));
        let seed = u64::from_le_bytes(payload[16..24].try_into().expect("len checked"));
        let batch = NonZeroUsize::new(batch)
            .filter(|batch| *batch <= max_batch)
            .ok_or((ERROR_BAD_REQUEST, "invalid SAMPLE request"))?;
        let window_rows =
            NonZeroU64::new(window_rows).ok_or((ERROR_BAD_REQUEST, "invalid SAMPLE request"))?;
        Ok(Self {
            batch,
            terminal_only,
            window_rows,
            seed,
        })
    }
}

pub(crate) struct HelloAck<'a> {
    pub feature_schema_hash: FeatureSchemaHash,
    pub max_batch: NonZeroUsize,
    pub produced_rows: u64,
    pub episodes: u64,
    pub episodes_stopped: u64,
    pub outcome_emas: [f32; 3],
    pub learner_win_rate_ema: f32,
    pub episode_latency_ema: f32,
    pub best_cost: f32,
    pub value_sign_accuracy_emas: [f32; 2],
    pub symmetric_metrics: Option<[f32; 10]>,
    pub engine_identity: EngineIdentity,
    pub feature_schema: &'a [u8],
}

impl HelloAck<'_> {
    pub fn encode_into(&self, payload: &mut Vec<u8>) {
        payload.clear();
        payload.reserve(HELLO_ACK_FIXED_LEN + self.feature_schema.len());
        push_u32(payload, SAMPLE_PROTOCOL_VERSION);
        payload.extend_from_slice(self.feature_schema_hash.as_bytes());
        push_u32(payload, self.max_batch.get() as u32);
        push_u64(payload, self.produced_rows);
        push_u64(payload, self.episodes);
        push_u64(payload, self.episodes_stopped);
        for value in self.outcome_emas {
            push_f32(payload, value);
        }
        push_f32(payload, self.learner_win_rate_ema);
        push_f32(payload, self.episode_latency_ema);
        push_f32(payload, self.best_cost);

        // V12 reserved these five fixed-root telemetry fields. They stay zero
        // until the next explicit protocol version; changing offsets in place
        // would make old trainers silently misdecode every following field.
        payload.extend_from_slice(&[0; 20]);
        for value in self.value_sign_accuracy_emas {
            push_f32(payload, value);
        }
        push_u32(payload, u32::from(self.symmetric_metrics.is_some()));
        for value in self.symmetric_metrics.unwrap_or([0.0; 10]) {
            push_f32(payload, value);
        }
        payload.extend_from_slice(self.engine_identity.engine_id.as_bytes());
        payload.extend_from_slice(self.engine_identity.engine_version.as_bytes());
        payload.extend_from_slice(self.engine_identity.action_set_hash.as_bytes());
        debug_assert_eq!(payload.len(), HELLO_ACK_FIXED_LEN);
        payload.extend_from_slice(self.feature_schema);
    }
}

pub(crate) fn validate_hello(payload: &[u8]) -> ProtocolResult<()> {
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
    Ok(())
}

pub(crate) fn read_frame<'a>(
    stream: &mut impl Read,
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

pub(crate) fn write_frame(
    stream: &mut impl Write,
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

pub(crate) fn send_error(
    stream: &mut impl Write,
    write_buf: &mut Vec<u8>,
    code: u32,
    message: &'static str,
) -> Result<(), String> {
    let message = if message.len() <= 512 {
        message
    } else {
        &message[..512]
    };
    let mut payload = Vec::with_capacity(6 + message.len());
    push_u32(&mut payload, code);
    payload.extend_from_slice(&(message.len() as u16).to_le_bytes());
    payload.extend_from_slice(message.as_bytes());
    write_frame(stream, write_buf, FRAME_ERROR, &[&payload]).map_err(|error| error.to_string())
}

fn push_u32(payload: &mut Vec<u8>, value: u32) {
    payload.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(payload: &mut Vec<u8>, value: u64) {
    payload.extend_from_slice(&value.to_le_bytes());
}

fn push_f32(payload: &mut Vec<u8>, value: f32) {
    payload.extend_from_slice(&value.to_le_bytes());
}
