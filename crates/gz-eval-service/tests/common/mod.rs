#![allow(dead_code)]

use gz_engine::{ActionSetHash, EngineId, EngineVersion};
use gz_eval_service::{
    FRAME_ERROR, FRAME_HELLO_ACK, Hello, HelloAck, PROTOCOL_VERSION, STUB_MODEL_VERSION,
    ServiceResult, write_frame,
};
use gz_features::{
    ActionFeature, FeatureCollator, FeatureRow, FeatureSchema, FeatureSchemaConfig,
    PositionFeatures, RowOutput, STOP_ACTION_KIND_TOKEN,
};
use std::fs;
use std::io::Read;
use std::num::NonZeroUsize;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{self, JoinHandle};

static NEXT_SOCKET: AtomicU64 = AtomicU64::new(0);

pub fn schema(name: &str, max_actions: u32) -> FeatureSchema {
    FeatureSchema::new(FeatureSchemaConfig {
        name: name.to_owned(),
        node_vocab_size: 64,
        node_attr_dim: 1,
        edge_type_count: 4,
        action_kind_vocab_size: 32,
        max_nodes: 8,
        max_edges: 8,
        max_actions,
        max_subjects: 3,
        opponent_reward_scale: 256.0,
        expander_degree: 0,
        expander_seed: 0,
    })
    .unwrap()
}

pub fn row(node_count: u32, action_count: usize) -> FeatureRow {
    assert!(action_count > 0);
    let actions = (0..action_count)
        .map(|index| {
            if index + 1 == action_count {
                ActionFeature {
                    kind_token: STOP_ACTION_KIND_TOKEN,
                    static_prior: 0.0,
                    subjects: Vec::new(),
                }
            } else {
                ActionFeature {
                    kind_token: 2 + index as u32,
                    static_prior: index as f32 * 0.25,
                    subjects: vec![index as u32 % node_count],
                }
            }
        })
        .collect();

    FeatureRow {
        node_count,
        node_tokens: (0..node_count).map(|index| 2 + index as u16).collect(),
        node_attrs: (0..node_count).map(|index| index as f32 + 0.5).collect(),
        edges: Vec::new(),
        actions,
        position: PositionFeatures {
            root_step: 1,
            leaf_depth: 2,
            budget_fraction: 0.5,
            budget_step: 0.25,
            opponent_reward: 0.0,
            opponent_present: false,
        },
    }
}

pub fn collate(schema: FeatureSchema, capacity: usize, rows: &[FeatureRow]) -> (Vec<u8>, Vec<u32>) {
    let mut collator = FeatureCollator::new(schema, NonZeroUsize::new(capacity).unwrap());
    let mut bytes = Vec::new();
    collator.collate_into(rows, &mut bytes).unwrap();
    let action_counts = rows.iter().map(|row| row.actions.len() as u32).collect();
    (bytes, action_counts)
}

pub fn hello(schema: &FeatureSchema, batch_capacity: u32) -> Hello {
    Hello::new(
        schema.hash(),
        batch_capacity,
        EngineId::from_bytes([1; 16]),
        EngineVersion::from_bytes([2; 16]),
        ActionSetHash::from_bytes([3; 32]),
    )
}

pub fn temp_socket(name: &str) -> PathBuf {
    let id = NEXT_SOCKET.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "gz-eval-service-{name}-{}-{id}.sock",
        std::process::id()
    ))
}

pub struct ScriptedServer {
    pub path: PathBuf,
    handle: Option<JoinHandle<()>>,
}

impl ScriptedServer {
    pub fn new<F>(name: &str, handler: F) -> Self
    where
        F: FnOnce(UnixStream) + Send + 'static,
    {
        let path = temp_socket(name);
        let _ = fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        let thread_path = path.clone();
        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handler(stream);
            let _ = fs::remove_file(thread_path);
        });
        Self {
            path,
            handle: Some(handle),
        }
    }
}

impl Drop for ScriptedServer {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap();
        }
        let _ = fs::remove_file(&self.path);
    }
}

pub fn send_ack(stream: &mut UnixStream) -> ServiceResult<()> {
    let mut payload = Vec::new();
    HelloAck {
        protocol_version: PROTOCOL_VERSION,
        model_version: STUB_MODEL_VERSION,
    }
    .encode(&mut payload);
    let mut write_buf = Vec::new();
    write_frame(stream, &mut write_buf, FRAME_HELLO_ACK, &[&payload])
}

pub fn send_error(stream: &mut UnixStream, code: u32, message: &str) -> ServiceResult<()> {
    let message = message.as_bytes();
    let message = &message[..message.len().min(512)];
    let mut payload = Vec::with_capacity(6 + message.len());
    payload.extend_from_slice(&code.to_le_bytes());
    payload.extend_from_slice(&(message.len() as u16).to_le_bytes());
    payload.extend_from_slice(message);
    let mut write_buf = Vec::new();
    write_frame(stream, &mut write_buf, FRAME_ERROR, &[&payload])
}

pub fn read_frame_type(stream: &mut UnixStream) -> u8 {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).unwrap();
    let body_len = u32::from_le_bytes(len) as usize;
    let mut body = vec![0; body_len];
    stream.read_exact(&mut body).unwrap();
    body[0]
}

pub fn output_payload(rows: &[RowOutput], capacity: usize, max_actions: usize) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GZFO");
    out.extend_from_slice(&gz_features::BATCH_ENCODING_VERSION.to_le_bytes());
    out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    out.extend_from_slice(&(max_actions as u32).to_le_bytes());
    for index in 0..capacity {
        let value = rows.get(index).map_or(0.0, |row| row.value);
        out.extend_from_slice(&value.to_le_bytes());
    }
    for row_index in 0..capacity {
        for action_index in 0..max_actions {
            let logit = rows
                .get(row_index)
                .and_then(|row| row.policy_logits.get(action_index))
                .copied()
                .unwrap_or(0.0);
            out.extend_from_slice(&logit.to_le_bytes());
        }
    }
    out
}

pub fn assert_outputs_equal_bits(actual: &[RowOutput], expected: &[RowOutput]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.value.to_bits(), expected.value.to_bits());
        assert_eq!(actual.policy_logits.len(), expected.policy_logits.len());
        for (&actual, &expected) in actual.policy_logits.iter().zip(&expected.policy_logits) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}

pub fn model_version_bytes() -> [u8; 16] {
    *STUB_MODEL_VERSION.as_bytes()
}
