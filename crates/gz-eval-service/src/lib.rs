#![forbid(unsafe_code)]

//! Unix-only process evaluator protocol and backends.
//!
//! This crate uses `std::os::unix::net::UnixStream` and is intentionally
//! limited to Unix platforms.

mod backend;
mod error;
mod frames;
mod hello;
mod process;
mod stub;

pub use backend::{BackendOutputs, FeatureEvalBackend, PendingBatch, StubBackend};
pub use error::{ServiceError, ServiceResult};
pub use frames::{
    FRAME_ERROR, FRAME_EVAL, FRAME_EVAL_RESULT, FRAME_HELLO, FRAME_HELLO_ACK, FRAME_PING,
    FRAME_PONG, MAX_FRAME, PROTOCOL_VERSION, read_frame, write_frame,
};
pub use hello::{
    ERROR_CAPACITY, ERROR_ENCODING, ERROR_MALFORMED, ERROR_PROTOCOL, ERROR_SCHEMA, Hello, HelloAck,
    decode_error,
};
pub use process::{EvaluatorProcess, EvaluatorProcessConfig, ProcessBackend};
pub use stub::{STUB_MODEL_VERSION, stub_row_outputs};
