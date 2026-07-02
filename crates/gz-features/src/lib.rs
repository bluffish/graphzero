#![forbid(unsafe_code)]

//! Feature schema, row validation, and fixed-layout batch encoding.

mod collator;
mod error;
mod row;
mod schema;

pub use collator::{FeatureBatchView, FeatureCollator, RowOutput};
pub use error::{FeatureError, FeatureResult};
pub use row::{ActionFeature, FeatureEdge, FeatureRow, PositionFeatures};
pub use schema::{
    ENCODING_VERSION, FeatureSchema, FeatureSchemaConfig, FeatureSchemaHash, STOP_ACTION_KIND_TOKEN,
};

use gz_engine::GraphEngine;

pub trait FeatureExtractor<E: GraphEngine>: Send {
    fn schema(&self) -> &FeatureSchema;

    fn extract(
        &mut self,
        engine: &E,
        graph: E::Graph,
        candidates: &[E::Candidate],
        position: PositionFeatures,
    ) -> FeatureResult<FeatureRow>;
}
