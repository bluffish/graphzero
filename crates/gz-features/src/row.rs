use crate::{FeatureError, FeatureResult, FeatureSchema, STOP_ACTION_KIND_TOKEN};

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureRow {
    pub node_count: u32,
    pub node_tokens: Vec<u16>,
    pub node_attrs: Vec<f32>,
    pub edges: Vec<FeatureEdge>,
    pub actions: Vec<ActionFeature>,
    pub position: PositionFeatures,
}

impl FeatureRow {
    pub fn validate(&self, schema: &FeatureSchema) -> FeatureResult<()> {
        let config = schema.config();
        if self.node_count == 0 {
            return Err(FeatureError::InvalidRow("node_count must be positive"));
        }
        if self.node_count > config.max_nodes {
            return Err(FeatureError::NodeOverflow {
                max: config.max_nodes,
                actual: self.node_count,
            });
        }
        if self.node_tokens.len() != self.node_count as usize {
            return Err(FeatureError::InvalidRow("node token length mismatch"));
        }
        for token in self.node_tokens.iter().copied() {
            if token == 0 || token >= config.node_vocab_size {
                return Err(FeatureError::InvalidRow("node token out of range"));
            }
        }
        let expected_attrs = self.node_count as usize * usize::from(config.node_attr_dim);
        if self.node_attrs.len() != expected_attrs {
            return Err(FeatureError::InvalidRow("node attr length mismatch"));
        }
        if self.node_attrs.iter().any(|value| !value.is_finite()) {
            return Err(FeatureError::InvalidRow("non-finite node attr"));
        }
        if self.edges.len() > config.max_edges as usize {
            return Err(FeatureError::EdgeOverflow {
                max: config.max_edges,
                actual: self.edges.len(),
            });
        }
        for edge in &self.edges {
            if edge.src >= self.node_count || edge.dst >= self.node_count {
                return Err(FeatureError::InvalidRow("edge endpoint out of range"));
            }
            if edge.edge_type >= config.edge_type_count {
                return Err(FeatureError::InvalidRow("edge type out of range"));
            }
        }
        if self.actions.is_empty() {
            return Err(FeatureError::InvalidRow("actions must include STOP"));
        }
        if self.actions.len() > config.max_actions as usize {
            return Err(FeatureError::ActionOverflow {
                max: config.max_actions,
                actual: self.actions.len(),
            });
        }
        for (index, action) in self.actions.iter().enumerate() {
            if action.kind_token == 0 || action.kind_token >= config.action_kind_vocab_size {
                return Err(FeatureError::InvalidRow("action kind token out of range"));
            }
            if !action.static_prior.is_finite() {
                return Err(FeatureError::InvalidRow("non-finite action static prior"));
            }
            if action.subjects.len() > config.max_subjects as usize {
                return Err(FeatureError::SubjectOverflow {
                    max: config.max_subjects,
                    actual: action.subjects.len(),
                });
            }
            for subject in action.subjects.iter().copied() {
                if subject >= self.node_count {
                    return Err(FeatureError::InvalidRow("action subject out of range"));
                }
            }
            let is_last = index + 1 == self.actions.len();
            if action.kind_token == STOP_ACTION_KIND_TOKEN {
                if !is_last {
                    return Err(FeatureError::InvalidRow("STOP action must be last"));
                }
                if !action.subjects.is_empty() || action.static_prior != 0.0 {
                    return Err(FeatureError::InvalidRow(
                        "STOP action payload must be empty",
                    ));
                }
            } else if is_last {
                return Err(FeatureError::InvalidRow("last action must be STOP"));
            }
        }
        if !self.position.budget_fraction.is_finite()
            || !self.position.budget_step.is_finite()
            || !self.position.opponent_reward.is_finite()
        {
            return Err(FeatureError::InvalidRow("non-finite position feature"));
        }
        if !self.position.opponent_present && self.position.opponent_reward != 0.0 {
            return Err(FeatureError::InvalidRow(
                "absent opponent must have zero reward",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureEdge {
    pub src: u32,
    pub dst: u32,
    pub edge_type: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActionFeature {
    pub kind_token: u32,
    pub static_prior: f32,
    pub subjects: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionFeatures {
    pub root_step: u32,
    pub leaf_depth: u32,
    pub budget_fraction: f32,
    pub budget_step: f32,
    pub opponent_reward: f32,
    pub opponent_present: bool,
}
