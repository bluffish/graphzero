use std::fmt;

pub type ReplayResult<T> = Result<T, ReplayError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayError {
    NotMeasured,
    InvalidRecord,
    SchemaMismatch,
    EngineIdentityMismatch,
    DataModeMismatch,
    TargetContractMismatch,
    Empty,
    Storage(String),
}

impl ReplayError {
    pub(crate) fn storage(error: impl ToString) -> Self {
        let mut message = error.to_string();
        const MAX_LEN: usize = 512;

        if message.len() > MAX_LEN {
            message.truncate(MAX_LEN);
        }

        Self::Storage(message)
    }
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotMeasured => write!(f, "episode final graph was not measured"),
            Self::InvalidRecord => write!(f, "invalid replay record"),
            Self::SchemaMismatch => write!(f, "replay schema version mismatch"),
            Self::EngineIdentityMismatch => write!(f, "replay engine identity mismatch"),
            Self::DataModeMismatch => write!(f, "replay data mode mismatch"),
            Self::TargetContractMismatch => write!(f, "replay target contract mismatch"),
            Self::Empty => write!(f, "replay store is empty"),
            Self::Storage(message) => write!(f, "replay storage error: {message}"),
        }
    }
}

impl std::error::Error for ReplayError {}

impl From<rocksdb::Error> for ReplayError {
    fn from(error: rocksdb::Error) -> Self {
        Self::storage(error)
    }
}

impl From<postcard::Error> for ReplayError {
    fn from(error: postcard::Error) -> Self {
        Self::storage(error)
    }
}
