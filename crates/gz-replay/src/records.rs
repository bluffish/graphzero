use crate::{ReplayError, ReplayResult};
use gz_engine::{
    CandidateHash, MeasureSummary, ModelVersion, PortableCandidateRef, PortableSearchActionRef,
    ReplayGraphContext, SearchConfigHash, SearchStepRef,
};
use gz_features::{
    FeatureSchemaHash, bf16_bits_to_f32, f32_to_bf16_bits, validate_feature_row_header,
};

#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, serde::Deserialize, serde::Serialize,
)]
#[serde(transparent)]
pub struct ReplayEpisodeId(u64);

impl ReplayEpisodeId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ReplayEpisodeRecord {
    pub root: ReplayGraphContext,
    pub final_graph: ReplayGraphContext,
    pub steps: Vec<SearchStepRef>,
    pub final_measure: MeasureSummary,
    pub outcome: ReplayOutcome,
    pub search_config_hash: SearchConfigHash,
    pub row_count: u32,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ReplayOutcome {
    pub value_target: Option<f32>,
    pub learner_reward: f32,
    pub reference: Option<ReplayReference>,
    /// True when the search selected STOP; false when the episode hit the
    /// move budget.
    pub stopped: bool,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ReplayReference {
    pub kind: ReplayReferenceKind,
    pub reward: f32,
    pub final_graph: Option<ReplayGraphContext>,
    pub trajectory_id: Option<u64>,
    pub search_config_hash: Option<SearchConfigHash>,
    pub model_version: Option<ModelVersion>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum ReplayReferenceKind {
    RootBaseline,
    Greedy,
    Beam,
    Random,
    Gumbel,
    // Appended last: postcard encodes enum variant indexes, so adding at
    // the end keeps every existing store's bytes decoding unchanged. Any
    // future variant must also be appended, never inserted.
    SelfAverage,
    /// Historical-best policy rollout: the bar is the best greedy rollout
    /// any published checkpoint achieved this run; model_version is the
    /// incumbent that set it.
    GatedPolicy,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ReplayRow {
    pub step_index: u32,
    pub root: ReplayGraphContext,
    pub state: ReplayGraphContext,
    pub action_history: Vec<PortableSearchActionRef>,
    pub legal_actions: Vec<PortableSearchActionRef>,
    pub policy_target: Vec<f32>,
    pub selected_action: PortableSearchActionRef,
    pub value_target: Option<f32>,
    pub reward_target: Option<f32>,
    pub final_measure: MeasureSummary,
    pub model_version: Option<ModelVersion>,
    pub search_config_hash: SearchConfigHash,
    pub feature_row: Option<Vec<u8>>,
}

/// Storage twin of [`ReplayRow`]. Every legal action of a row references
/// the row's state context by construction (candidates are enumerated at
/// the state graph and STOP references it), so storage keeps one context
/// and a bare hash per action instead of repeating a 96-byte context per
/// legal action -- ~80% of row bytes at the 1024-action shape. Policy
/// targets store as bf16 bits, the precision the trainer already sees on
/// the batch wire. Reconstruction is exact apart from that bf16 rounding;
/// the context invariant is validated at append.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(crate) struct StoredReplayRow {
    step_index: u32,
    root: ReplayGraphContext,
    state: ReplayGraphContext,
    action_history: Vec<PortableSearchActionRef>,
    legal_actions: Vec<StoredLegalAction>,
    policy_target_bf16: Vec<u16>,
    selected_action: PortableSearchActionRef,
    value_target: Option<f32>,
    reward_target: Option<f32>,
    final_measure: MeasureSummary,
    model_version: Option<ModelVersion>,
    search_config_hash: SearchConfigHash,
    feature_row: Option<Vec<u8>>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(crate) enum StoredLegalAction {
    Candidate(CandidateHash),
    Stop,
}

impl StoredReplayRow {
    pub(crate) fn from_row(row: &ReplayRow) -> ReplayResult<Self> {
        let mut legal_actions = Vec::with_capacity(row.legal_actions.len());
        for action in &row.legal_actions {
            legal_actions.push(match action {
                PortableSearchActionRef::Candidate(candidate) => {
                    if candidate.context != row.state {
                        return Err(ReplayError::InvalidRecord);
                    }
                    StoredLegalAction::Candidate(candidate.candidate_hash)
                }
                PortableSearchActionRef::Stop { context } => {
                    if *context != row.state {
                        return Err(ReplayError::InvalidRecord);
                    }
                    StoredLegalAction::Stop
                }
            });
        }

        Ok(Self {
            step_index: row.step_index,
            root: row.root,
            state: row.state,
            action_history: row.action_history.clone(),
            legal_actions,
            policy_target_bf16: row
                .policy_target
                .iter()
                .copied()
                .map(f32_to_bf16_bits)
                .collect(),
            selected_action: row.selected_action,
            value_target: row.value_target,
            reward_target: row.reward_target,
            final_measure: row.final_measure.clone(),
            model_version: row.model_version,
            search_config_hash: row.search_config_hash,
            feature_row: row.feature_row.clone(),
        })
    }

    pub(crate) fn into_row(self) -> ReplayRow {
        let state = self.state;
        let legal_actions = self
            .legal_actions
            .into_iter()
            .map(|action| match action {
                StoredLegalAction::Candidate(hash) => {
                    PortableSearchActionRef::candidate(PortableCandidateRef::new(state, hash))
                }
                StoredLegalAction::Stop => PortableSearchActionRef::stop(state),
            })
            .collect();

        ReplayRow {
            step_index: self.step_index,
            root: self.root,
            state,
            action_history: self.action_history,
            legal_actions,
            policy_target: self
                .policy_target_bf16
                .into_iter()
                .map(bf16_bits_to_f32)
                .collect(),
            selected_action: self.selected_action,
            value_target: self.value_target,
            reward_target: self.reward_target,
            final_measure: self.final_measure,
            model_version: self.model_version,
            search_config_hash: self.search_config_hash,
            feature_row: self.feature_row,
        }
    }
}

pub(crate) fn validate_episode(
    record: &ReplayEpisodeRecord,
    rows: &[ReplayRow],
    feature_schema_hash: Option<FeatureSchemaHash>,
) -> ReplayResult<()> {
    validate_admission(record)?;
    validate_outcome(record)?;

    if rows.len() != record.row_count as usize || rows.len() != record.steps.len() {
        return Err(ReplayError::InvalidRecord);
    }

    let has_feature_rows = rows
        .first()
        .map(|row| row.feature_row.is_some())
        .unwrap_or(false);
    if has_feature_rows && feature_schema_hash.is_none() {
        return Err(ReplayError::InvalidRecord);
    }

    let mut expected_history = Vec::new();

    for (index, row) in rows.iter().enumerate() {
        let step_index = u32::try_from(index).map_err(|_| ReplayError::InvalidRecord)?;

        if row.step_index != step_index {
            return Err(ReplayError::InvalidRecord);
        }
        if row.feature_row.is_some() != has_feature_rows {
            return Err(ReplayError::InvalidRecord);
        }
        if let (Some(bytes), Some(hash)) = (&row.feature_row, feature_schema_hash) {
            validate_feature_row_header(bytes, &hash).map_err(|_| ReplayError::InvalidRecord)?;
        }

        validate_row(record, row, &expected_history)?;
        expected_history.push(record.steps[index].action);
    }

    Ok(())
}

fn validate_admission(record: &ReplayEpisodeRecord) -> ReplayResult<()> {
    let Some(reward) = record.final_measure.scalar_reward else {
        return Err(ReplayError::NotMeasured);
    };

    if !record.final_measure.measured || !record.final_measure.valid || !reward.is_finite() {
        return Err(ReplayError::NotMeasured);
    }

    Ok(())
}

fn validate_outcome(record: &ReplayEpisodeRecord) -> ReplayResult<()> {
    if !record.outcome.learner_reward.is_finite() {
        return Err(ReplayError::InvalidRecord);
    }

    if Some(record.outcome.learner_reward) != record.final_measure.scalar_reward {
        return Err(ReplayError::InvalidRecord);
    }

    validate_value_target(record.outcome.value_target)?;

    match &record.outcome.reference {
        Some(reference) => {
            if !reference.reward.is_finite() {
                return Err(ReplayError::InvalidRecord);
            }

            let valid = match sign_target(record.outcome.learner_reward, reference.reward) {
                // Exact ties are coin-flipped to a hard +/-1 at projection
                // (random tie-break); either sign is a valid tie label.
                0.0 => matches!(record.outcome.value_target, Some(-1.0 | 1.0)),
                expected => record.outcome.value_target == Some(expected),
            };
            if !valid {
                return Err(ReplayError::InvalidRecord);
            }
        }
        None => {
            if record.outcome.value_target.is_some() {
                return Err(ReplayError::InvalidRecord);
            }
        }
    }

    Ok(())
}

fn validate_row(
    record: &ReplayEpisodeRecord,
    row: &ReplayRow,
    expected_history: &[PortableSearchActionRef],
) -> ReplayResult<()> {
    let step = &record.steps[row.step_index as usize];

    if row.root != record.root
        || row.state != step.before
        || row.selected_action != step.action
        || row.action_history != expected_history
        || row.final_measure != record.final_measure
        || row.search_config_hash != record.search_config_hash
        || row.value_target != record.outcome.value_target
        || row.reward_target != Some(record.outcome.learner_reward)
    {
        return Err(ReplayError::InvalidRecord);
    }

    if row.legal_actions.len() != row.policy_target.len()
        || !matches!(
            row.legal_actions.last(),
            Some(PortableSearchActionRef::Stop { .. })
        )
    {
        return Err(ReplayError::InvalidRecord);
    }

    if !row.legal_actions.contains(&row.selected_action) {
        return Err(ReplayError::InvalidRecord);
    }

    for value in &row.policy_target {
        if !value.is_finite() || *value < 0.0 {
            return Err(ReplayError::InvalidRecord);
        }
    }

    validate_value_target(row.value_target)?;

    Ok(())
}

fn validate_value_target(value: Option<f32>) -> ReplayResult<()> {
    match value {
        // Hard signs only: ties are coin-flipped at projection, so a
        // stored zero target is a producer bug.
        Some(value) if value == -1.0 || value == 1.0 => Ok(()),
        Some(_) => Err(ReplayError::InvalidRecord),
        None => Ok(()),
    }
}

fn sign_target(learner: f32, reference: f32) -> f32 {
    if learner > reference {
        1.0
    } else if learner < reference {
        -1.0
    } else {
        0.0
    }
}

/// Static facts about the fixed root graph of a single-graph run,
/// probed once at selfplay startup. Telemetry: consumed by the trainer
/// through the sample-service handshake.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ReplayRootInfo {
    pub cost: f32,
    pub node_count: u32,
    pub edge_count: u32,
    pub candidate_count: u32,
}
