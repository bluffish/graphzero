use std::fmt;

pub type ServiceResult<T> = Result<T, ServiceError>;

const MAX_ERROR_BYTES: usize = 512;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServiceError {
    Configuration(String),
    Capacity(String),
    Authentication(String),
    Protocol(String),
    Transport(String),
    Timeout(String),
    RemoteFailure { code: u32, message: String },
    Io(String),
    Closed,
}

impl ServiceError {
    pub fn configuration(message: impl AsRef<str>) -> Self {
        Self::Configuration(bound_message(message.as_ref()))
    }

    pub fn capacity(message: impl AsRef<str>) -> Self {
        Self::Capacity(bound_message(message.as_ref()))
    }

    pub fn authentication(message: impl AsRef<str>) -> Self {
        Self::Authentication(bound_message(message.as_ref()))
    }

    pub fn protocol(message: impl AsRef<str>) -> Self {
        Self::Protocol(bound_message(message.as_ref()))
    }

    pub fn transport(message: impl AsRef<str>) -> Self {
        Self::Transport(bound_message(message.as_ref()))
    }

    pub fn timeout(message: impl AsRef<str>) -> Self {
        Self::Timeout(bound_message(message.as_ref()))
    }

    pub fn remote_failure(code: u32, message: impl AsRef<str>) -> Self {
        Self::RemoteFailure {
            code,
            message: bound_message(message.as_ref()),
        }
    }

    pub fn io(message: impl AsRef<str>) -> Self {
        Self::Io(bound_message(message.as_ref()))
    }
}

impl fmt::Display for ServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(message) => write!(f, "measurement configuration error: {message}"),
            Self::Capacity(message) => write!(f, "measurement capacity exhausted: {message}"),
            Self::Authentication(message) => {
                write!(f, "measurement authentication failed: {message}")
            }
            Self::Protocol(message) => write!(f, "measurement protocol error: {message}"),
            Self::Transport(message) => write!(f, "measurement transport error: {message}"),
            Self::Timeout(message) => write!(f, "measurement timed out: {message}"),
            Self::RemoteFailure { code, message } => {
                write!(f, "remote measurement failed with code {code}: {message}")
            }
            Self::Io(message) => write!(f, "measurement io error: {message}"),
            Self::Closed => f.write_str("measurement service closed"),
        }
    }
}

impl std::error::Error for ServiceError {}

pub(crate) fn bound_message(message: &str) -> String {
    if message.len() <= MAX_ERROR_BYTES {
        return message.to_owned();
    }

    let mut end = 0;
    for (index, ch) in message.char_indices() {
        let next = index + ch.len_utf8();
        if next > MAX_ERROR_BYTES {
            break;
        }
        end = next;
    }
    message[..end].to_owned()
}
