use graphs_compiler::engine::CompilerEngineError as NativeCompilerEngineError;
use gz_engine::{EngineError, ErrorCode, ErrorMessage};
use std::fmt;

pub(crate) const NATIVE_ENGINE_ERROR_CODE: ErrorCode = ErrorCode::new(1);
const MEASURE_CONFIG_MISMATCH_CODE: ErrorCode = ErrorCode::new(2);
const UNSUPPORTED_MEASURE_TIMEOUT_CODE: ErrorCode = ErrorCode::new(3);
const INVALID_ARTIFACT_CODE: ErrorCode = ErrorCode::new(4);

#[derive(Debug)]
pub enum CompilerEngineBuildError {
    Native(NativeCompilerEngineError),
    TooManyActionKinds { count: usize },
}

impl fmt::Display for CompilerEngineBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native(error) => write!(formatter, "{error}"),
            Self::TooManyActionKinds { count } => {
                write!(
                    formatter,
                    "compiler exposes {count} action kinds; maximum is {}",
                    u64::from(u32::MAX) + 1
                )
            }
        }
    }
}

impl std::error::Error for CompilerEngineBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Native(error) => Some(error),
            Self::TooManyActionKinds { .. } => None,
        }
    }
}

impl From<NativeCompilerEngineError> for CompilerEngineBuildError {
    fn from(error: NativeCompilerEngineError) -> Self {
        Self::Native(error)
    }
}

pub(crate) fn native_error(error: NativeCompilerEngineError) -> EngineError {
    match error {
        NativeCompilerEngineError::UnknownState(_)
        | NativeCompilerEngineError::ReleasedState(_)
        | NativeCompilerEngineError::StaleState(_) => {
            EngineError::UnknownGraph { graph_hash: None }
        }
        NativeCompilerEngineError::UnknownCandidate(_)
        | NativeCompilerEngineError::ReleasedCandidate(_)
        | NativeCompilerEngineError::StaleCandidate(_) => EngineError::UnknownCandidate {
            candidate_hash: None,
        },
        error => internal_error(NATIVE_ENGINE_ERROR_CODE, error.to_string()),
    }
}

pub(crate) fn action_kind_invariant() -> EngineError {
    internal_error(
        NATIVE_ENGINE_ERROR_CODE,
        "compiler candidate references an unregistered action definition",
    )
}

pub(crate) fn measure_config_mismatch() -> EngineError {
    internal_error(
        MEASURE_CONFIG_MISMATCH_CODE,
        "measure config hash does not match this compiler engine",
    )
}

pub(crate) fn unsupported_measure_timeout() -> EngineError {
    internal_error(
        UNSUPPORTED_MEASURE_TIMEOUT_CODE,
        "compiler engine does not support local measurement timeouts",
    )
}

pub(crate) fn invalid_artifact(message: impl Into<String>) -> EngineError {
    internal_error(INVALID_ARTIFACT_CODE, message)
}

pub(crate) fn internal_error(code: ErrorCode, message: impl Into<String>) -> EngineError {
    EngineError::Internal {
        code,
        message: bounded_message(message.into()),
    }
}

pub(crate) fn bounded_message(mut message: String) -> ErrorMessage {
    if message.len() > ErrorMessage::MAX_LEN {
        let mut boundary = ErrorMessage::MAX_LEN;
        while !message.is_char_boundary(boundary) {
            boundary -= 1;
        }
        message.truncate(boundary);
    }
    ErrorMessage::new(message).expect("message was bounded to ErrorMessage::MAX_LEN")
}
