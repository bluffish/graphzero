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
}

impl fmt::Display for CompilerEngineBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for CompilerEngineBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Native(error) => Some(error),
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
        error => internal_error(NATIVE_ENGINE_ERROR_CODE, error_chain(&error)),
    }
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

pub(crate) fn error_chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(error) = source {
        message.push_str(": ");
        message.push_str(&error.to_string());
        source = error.source();
    }
    message
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
