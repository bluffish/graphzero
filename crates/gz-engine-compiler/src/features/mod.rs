mod compact;
mod state;
mod value;

use crate::{CompilerEngine, native_error};
use gz_engine::{GraphEngine, GraphHash};
use gz_features::{
    ActionFeature, FeatureError, FeatureExtractor, FeatureResult, FeatureRow, FeatureSchema,
    FeatureSchemaConfig, PositionFeatures, STOP_ACTION_KIND_TOKEN,
};
use state::{
    COMPILER_EDGE_TYPES, COMPILER_NODE_ATTR_DIM, OP_TOKEN_BASE, StateFeatures, state_features,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CompilerFeatureLayout {
    #[default]
    ExpandedPorts,
    CompactV1,
}

impl std::str::FromStr for CompilerFeatureLayout {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "expanded-ports" => Ok(Self::ExpandedPorts),
            "compact-v1" => Ok(Self::CompactV1),
            _ => Err(format!("unknown compiler feature layout: {value}")),
        }
    }
}

/// Run-level schema bounds for compiler feature rows.
///
/// Compiler rewrites can grow the IR, so these are explicit rather than
/// inferred from the root.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompilerFeatureExtractorConfig {
    pub layout: CompilerFeatureLayout,
    /// Total feature nodes after adding explicit operand/result endpoint nodes.
    pub max_nodes: u32,
    pub max_edges: u32,
    /// Compiler candidates plus the search-owned STOP action.
    pub max_actions: u32,
    pub max_subjects: u32,
}

/// Canonical compiler IR and transform-action projection for GraphZero models.
#[derive(Clone, Debug)]
pub struct CompilerFeatureExtractor {
    schema: FeatureSchema,
    layout: CompilerFeatureLayout,
}

impl CompilerFeatureExtractor {
    pub fn new(
        engine: &CompilerEngine,
        config: CompilerFeatureExtractorConfig,
    ) -> FeatureResult<Self> {
        let state = engine
            .native
            .state_view(engine.root)
            .map_err(native_error)
            .map_err(FeatureError::from)?;
        let node_vocab_size = OP_TOKEN_BASE
            .checked_add(state.module.view().registry.len())
            .and_then(|size| u16::try_from(size).ok())
            .ok_or(FeatureError::InvalidSchema(
                "compiler operation vocabulary exceeds u16",
            ))?;
        let action_kind_vocab_size =
            engine
                .native
                .action_kind_count()
                .checked_add(2)
                .ok_or(FeatureError::InvalidSchema(
                    "compiler action-kind vocabulary overflows u32",
                ))?;
        let (name, edge_type_count) = match config.layout {
            CompilerFeatureLayout::ExpandedPorts => ("compiler-v1", COMPILER_EDGE_TYPES),
            CompilerFeatureLayout::CompactV1 => {
                ("compiler-compact-v1", compact::COMPACT_EDGE_TYPES)
            }
        };
        let schema = FeatureSchema::new(FeatureSchemaConfig {
            name: name.to_string(),
            node_vocab_size,
            node_attr_dim: COMPILER_NODE_ATTR_DIM,
            edge_type_count,
            action_kind_vocab_size,
            max_nodes: config.max_nodes,
            max_edges: config.max_edges,
            max_actions: config.max_actions,
            max_subjects: config.max_subjects,
            opponent_reward_scale: 1.0,
            expander_degree: 0,
            expander_seed: 0,
        })?;
        Ok(Self {
            schema,
            layout: config.layout,
        })
    }

    #[must_use]
    pub const fn schema(&self) -> &FeatureSchema {
        &self.schema
    }

    fn actions(
        &self,
        engine: &CompilerEngine,
        graph_hash: GraphHash,
        candidates: &[<CompilerEngine as GraphEngine>::Candidate],
        state: &StateFeatures,
    ) -> FeatureResult<Vec<ActionFeature>> {
        if candidates.len().saturating_add(1) > self.schema.config().max_actions as usize {
            return Err(FeatureError::ActionOverflow {
                max: self.schema.config().max_actions,
                actual: candidates.len().saturating_add(1),
            });
        }
        let mut actions = Vec::with_capacity(candidates.len() + 1);
        for candidate in candidates.iter().copied() {
            let (kind, candidate_subjects) = engine.candidate_feature(graph_hash, candidate)?;
            let subjects = candidate_subjects
                .iter()
                .copied()
                .map(|subject| {
                    state
                        .indexes
                        .subject(subject)
                        .ok_or(FeatureError::InvalidRow(
                            "compiler action subject is absent from the feature graph",
                        ))
                })
                .collect::<FeatureResult<Vec<_>>>()?;
            actions.push(ActionFeature {
                kind_token: kind + 2,
                static_prior: 0.0,
                subjects,
            });
        }
        actions.push(ActionFeature {
            kind_token: STOP_ACTION_KIND_TOKEN,
            static_prior: 0.0,
            subjects: Vec::new(),
        });
        Ok(actions)
    }
}

impl FeatureExtractor<CompilerEngine> for CompilerFeatureExtractor {
    fn schema(&self) -> &FeatureSchema {
        &self.schema
    }

    fn extract(
        &mut self,
        engine: &CompilerEngine,
        graph: <CompilerEngine as GraphEngine>::Graph,
        candidates: &[<CompilerEngine as GraphEngine>::Candidate],
        position: PositionFeatures,
    ) -> FeatureResult<FeatureRow> {
        let graph_hash = engine.hash(graph)?;
        let view = engine
            .native
            .state_view(graph)
            .map_err(native_error)
            .map_err(FeatureError::from)?;
        let state = match self.layout {
            CompilerFeatureLayout::ExpandedPorts => state_features(view, self.schema.config())?,
            CompilerFeatureLayout::CompactV1 => {
                compact::state_features(view, self.schema.config())?
            }
        };
        let actions = self.actions(engine, graph_hash, candidates, &state)?;
        let row = FeatureRow {
            node_count: state.node_count,
            node_tokens: state.node_tokens,
            node_attrs: state.node_attrs,
            edges: state.edges,
            actions,
            position,
            opponent: None,
        };
        row.validate(&self.schema)?;
        Ok(row)
    }
}
