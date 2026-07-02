use crate::{NO_NODE, OpCode, RULE_COUNT, WhittleEngine, WhittleGraphId};
use gz_engine::{GraphEngine, PortableGraphId};
use gz_features::{
    ActionFeature, FeatureEdge, FeatureError, FeatureExtractor, FeatureResult, FeatureRow,
    FeatureSchema, FeatureSchemaConfig, PositionFeatures, STOP_ACTION_KIND_TOKEN,
};
use std::collections::HashMap;

const WHITTLE_MAX_ACTIONS: u32 = 256;
const WHITTLE_MAX_SUBJECTS: u32 = 8;

#[derive(Clone, Debug)]
pub struct WhittleFeatureExtractor {
    schema: FeatureSchema,
    state_cache: HashMap<PortableGraphId, CachedStateFeatures>,
}

impl WhittleFeatureExtractor {
    #[must_use]
    pub fn new(engine: &WhittleEngine) -> Self {
        let root = engine.root();
        let root_graph = engine.graph(root).expect("Whittle root graph exists");
        let schema = FeatureSchema::new(FeatureSchemaConfig {
            name: "whittle-v1".to_string(),
            node_vocab_size: 7,
            node_attr_dim: 0,
            edge_type_count: 2,
            action_kind_vocab_size: RULE_COUNT + 2,
            max_nodes: u32::from(root_graph.capacity),
            max_edges: u32::from(root_graph.capacity) * 2,
            max_actions: WHITTLE_MAX_ACTIONS,
            max_subjects: WHITTLE_MAX_SUBJECTS,
        })
        .expect("Whittle feature schema is valid");

        Self {
            schema,
            state_cache: HashMap::new(),
        }
    }

    #[must_use]
    pub const fn schema(&self) -> &FeatureSchema {
        &self.schema
    }
}

impl FeatureExtractor<WhittleEngine> for WhittleFeatureExtractor {
    fn schema(&self) -> &FeatureSchema {
        &self.schema
    }

    fn extract(
        &mut self,
        engine: &WhittleEngine,
        graph: WhittleGraphId,
        candidates: &[<WhittleEngine as GraphEngine>::Candidate],
        position: PositionFeatures,
    ) -> FeatureResult<FeatureRow> {
        let graph_hash = engine.hash(graph)?;
        let graph_id =
            PortableGraphId::new(graph_hash, engine.engine_id(), engine.engine_version());
        let state = if let Some(cached) = self.state_cache.get(&graph_id) {
            cached.clone()
        } else {
            let graph_body = engine.graph(graph)?;
            let state = CachedStateFeatures {
                node_count: graph_body.op.len() as u32,
                node_tokens: graph_body.op.iter().copied().map(node_token).collect(),
                edges: graph_edges(
                    graph_body.op.as_ref(),
                    graph_body.arg0.as_ref(),
                    graph_body.arg1.as_ref(),
                ),
            };
            self.state_cache.insert(graph_id, state.clone());
            state
        };

        let mut actions = Vec::with_capacity(candidates.len() + 1);
        for candidate in candidates.iter().copied() {
            let info = engine.candidate_info(graph, candidate)?;
            let mut subjects = Vec::with_capacity(info.subjects.len());
            for subject in info.subjects {
                let subject = u32::try_from(subject.get())
                    .map_err(|_| FeatureError::InvalidRow("Whittle subject id overflow"))?;
                subjects.push(subject);
            }
            actions.push(ActionFeature {
                kind_token: info.kind.get() + 2,
                static_prior: info.static_prior,
                subjects,
            });
        }
        actions.push(ActionFeature {
            kind_token: STOP_ACTION_KIND_TOKEN,
            static_prior: 0.0,
            subjects: Vec::new(),
        });

        let row = FeatureRow {
            node_count: state.node_count,
            node_tokens: state.node_tokens,
            node_attrs: Vec::new(),
            edges: state.edges,
            actions,
            position,
        };
        row.validate(&self.schema)?;
        Ok(row)
    }
}

#[derive(Clone, Debug)]
struct CachedStateFeatures {
    node_count: u32,
    node_tokens: Vec<u16>,
    edges: Vec<FeatureEdge>,
}

const fn node_token(op: OpCode) -> u16 {
    op as u16 + 1
}

fn graph_edges(op: &[OpCode], arg0: &[u32], arg1: &[u32]) -> Vec<FeatureEdge> {
    let mut edges = Vec::with_capacity(op.len().saturating_mul(2));
    for (dst, code) in op.iter().copied().enumerate() {
        match code {
            OpCode::Input | OpCode::Const => {}
            OpCode::Not | OpCode::Output => push_edge(&mut edges, arg0[dst], dst as u32, 0),
            OpCode::And | OpCode::Or => {
                push_edge(&mut edges, arg0[dst], dst as u32, 0);
                push_edge(&mut edges, arg1[dst], dst as u32, 1);
            }
        }
    }
    edges
}

fn push_edge(edges: &mut Vec<FeatureEdge>, src: u32, dst: u32, edge_type: u8) {
    if src != NO_NODE {
        edges.push(FeatureEdge {
            src,
            dst,
            edge_type,
        });
    }
}
