use gz_engine::EngineError;
use std::fmt;

pub type FeatureResult<T> = Result<T, FeatureError>;

#[derive(Clone, Debug, PartialEq)]
pub enum FeatureError {
    InvalidSchema(&'static str),
    NodeOverflow { max: u32, actual: u32 },
    EdgeOverflow { max: u32, actual: usize },
    ActionOverflow { max: u32, actual: usize },
    SubjectOverflow { max: u32, actual: usize },
    InvalidRow(&'static str),
    BatchOverflow { capacity: usize, actual: usize },
    EmptyBatch,
    InvalidEncoding(&'static str),
    Engine(EngineError),
}

impl fmt::Display for FeatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchema(message) => write!(f, "invalid feature schema: {message}"),
            Self::NodeOverflow { max, actual } => {
                write!(f, "node count {actual} exceeds schema max_nodes {max}")
            }
            Self::EdgeOverflow { max, actual } => {
                write!(f, "edge count {actual} exceeds schema max_edges {max}")
            }
            Self::ActionOverflow { max, actual } => {
                write!(f, "action count {actual} exceeds schema max_actions {max}")
            }
            Self::SubjectOverflow { max, actual } => {
                write!(
                    f,
                    "subject count {actual} exceeds schema max_subjects {max}"
                )
            }
            Self::InvalidRow(message) => write!(f, "invalid feature row: {message}"),
            Self::BatchOverflow { capacity, actual } => {
                write!(f, "batch row count {actual} exceeds capacity {capacity}")
            }
            Self::EmptyBatch => f.write_str("feature batch is empty"),
            Self::InvalidEncoding(message) => write!(f, "invalid feature encoding: {message}"),
            Self::Engine(error) => write!(f, "engine feature extraction failed: {error}"),
        }
    }
}

impl std::error::Error for FeatureError {}

impl From<EngineError> for FeatureError {
    fn from(value: EngineError) -> Self {
        Self::Engine(value)
    }
}
