use crate::ReplayDataMode;
use crate::{ReplayError, ReplayResult};
use gz_engine::{
    CandidateHash, EngineIdentity, MeasureSummary, ModelVersion, PortableCandidateRef,
    PortableSearchActionRef, ReplayGraphContext, SearchConfigHash, SearchStepRef,
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
    pub reward: f32,
    // Retained in the postcard layout so schema-v8 symmetric replay stores
    // remain readable. New records reject non-empty legacy references.
    legacy_reference: Option<StoredReplayReference>,
    /// True when the search selected STOP; false when the episode hit the
    /// move budget.
    pub stopped: bool,
}

impl ReplayOutcome {
    #[must_use]
    pub const fn new(value_target: Option<f32>, reward: f32, stopped: bool) -> Self {
        Self {
            value_target,
            reward,
            legacy_reference: None,
            stopped,
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
struct StoredReplayReference {
    kind: StoredReplayReferenceKind,
    reward: f32,
    final_graph: Option<ReplayGraphContext>,
    trajectory_id: Option<u64>,
    search_config_hash: Option<SearchConfigHash>,
    model_version: Option<ModelVersion>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
enum StoredReplayReferenceKind {
    RootBaseline,
    Greedy,
    Beam,
    Random,
    Gumbel,
    // Appended last: postcard encodes enum variant indexes, so adding at
    // the end keeps every existing store's bytes decoding unchanged. Any
    // future variant must also be appended, never inserted.
    SelfAverage,
    GatedPolicy,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct ReplayRow {
    pub kind: ReplayRowKind,
    pub step_index: u32,
    pub root: ReplayGraphContext,
    pub state: ReplayGraphContext,
    pub action_history: Vec<PortableSearchActionRef>,
    pub legal_actions: Vec<PortableSearchActionRef>,
    pub policy_target: Vec<f32>,
    pub selected_action: Option<PortableSearchActionRef>,
    pub value_target: Option<f32>,
    pub horizon_value_targets: Option<[f32; 2]>,
    pub reward_target: Option<f32>,
    pub final_measure: MeasureSummary,
    pub model_version: Option<ModelVersion>,
    pub search_config_hash: SearchConfigHash,
    pub feature_row: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum ReplayRowKind {
    Decision,
    Terminal,
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
    kind: ReplayRowKind,
    step_index: u32,
    root: ReplayGraphContext,
    state: ReplayGraphContext,
    action_history: Vec<PortableSearchActionRef>,
    legal_actions: Vec<StoredLegalAction>,
    policy_target_bf16: Vec<u16>,
    selected_action: Option<PortableSearchActionRef>,
    value_target: Option<f32>,
    horizon_value_targets: Option<[f32; 2]>,
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
            kind: row.kind,
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
            horizon_value_targets: row.horizon_value_targets,
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
            kind: self.kind,
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
            horizon_value_targets: self.horizon_value_targets,
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
    data_mode: ReplayDataMode,
) -> ReplayResult<()> {
    validate_episode_engine_identity(record, EngineIdentity::from_context(record.root))?;
    validate_admission(record)?;
    validate_outcome(record, data_mode)?;

    if rows.len() != record.row_count as usize {
        return Err(ReplayError::InvalidRecord);
    }

    let terminal_rows = rows
        .iter()
        .filter(|row| row.kind == ReplayRowKind::Terminal)
        .count();
    if rows.len() != record.steps.len() + terminal_rows || terminal_rows > 1 {
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

        match row.kind {
            ReplayRowKind::Decision => {
                validate_decision_row(record, row, &expected_history, data_mode)?;
                expected_history.push(record.steps[index].action);
            }
            ReplayRowKind::Terminal => {
                if index != record.steps.len() || terminal_rows != 1 {
                    return Err(ReplayError::InvalidRecord);
                }
                validate_terminal_row(record, row, &expected_history, data_mode)?;
            }
        }
    }

    Ok(())
}

pub(crate) fn validate_episode_engine_identity(
    record: &ReplayEpisodeRecord,
    identity: EngineIdentity,
) -> ReplayResult<()> {
    let matches = |context: ReplayGraphContext| EngineIdentity::from_context(context) == identity;
    if !matches(record.root)
        || !matches(record.final_graph)
        || record.steps.iter().any(|step| {
            !matches(step.before) || !matches(step.action.context()) || !matches(step.after)
        })
    {
        return Err(ReplayError::EngineIdentityMismatch);
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

fn validate_outcome(record: &ReplayEpisodeRecord, data_mode: ReplayDataMode) -> ReplayResult<()> {
    if !record.outcome.reward.is_finite() {
        return Err(ReplayError::InvalidRecord);
    }

    if Some(record.outcome.reward) != record.final_measure.scalar_reward {
        return Err(ReplayError::InvalidRecord);
    }

    validate_value_target(record.outcome.value_target, data_mode)?;

    if record.outcome.legacy_reference.is_some() {
        return Err(ReplayError::InvalidRecord);
    }
    let valid = if data_mode.is_symmetric_selfplay() {
        (!record.outcome.stopped || data_mode.symmetric_stop_enabled())
            && record
                .outcome
                .value_target
                .is_some_and(|value| value.is_finite() && (-1.0..=1.0).contains(&value))
            && (!data_mode.symmetric_stop_enabled()
                || record.outcome.stopped
                    == record.steps.last().is_some_and(|step| {
                        matches!(step.action, PortableSearchActionRef::Stop { .. })
                    }))
    } else {
        record.outcome.value_target.is_none()
    };
    if !valid {
        return Err(ReplayError::InvalidRecord);
    }

    Ok(())
}

fn validate_decision_row(
    record: &ReplayEpisodeRecord,
    row: &ReplayRow,
    expected_history: &[PortableSearchActionRef],
    data_mode: ReplayDataMode,
) -> ReplayResult<()> {
    let step = &record.steps[row.step_index as usize];

    if row.root != record.root
        || row.state != step.before
        || row.selected_action != Some(step.action)
        || row.action_history != expected_history
        || row.final_measure != record.final_measure
        || row.search_config_hash != record.search_config_hash
        || row.value_target != record.outcome.value_target
        || row.reward_target != Some(record.outcome.reward)
    {
        return Err(ReplayError::InvalidRecord);
    }

    let legal_action_shape_valid = if data_mode.symmetric_stop_enabled() {
        matches!(
            row.legal_actions.last(),
            Some(PortableSearchActionRef::Stop { .. })
        )
    } else if data_mode.is_symmetric_selfplay() {
        !row.legal_actions.is_empty()
            && row
                .legal_actions
                .iter()
                .all(|action| !matches!(action, PortableSearchActionRef::Stop { .. }))
    } else {
        matches!(
            row.legal_actions.last(),
            Some(PortableSearchActionRef::Stop { .. })
        )
    };
    if row.legal_actions.len() != row.policy_target.len() || !legal_action_shape_valid {
        return Err(ReplayError::InvalidRecord);
    }

    if !row
        .selected_action
        .is_some_and(|selected| row.legal_actions.contains(&selected))
    {
        return Err(ReplayError::InvalidRecord);
    }

    for value in &row.policy_target {
        if !value.is_finite() || *value < 0.0 {
            return Err(ReplayError::InvalidRecord);
        }
    }

    validate_value_target(row.value_target, data_mode)?;
    validate_horizon_value_targets(row.horizon_value_targets, data_mode)?;

    Ok(())
}

fn validate_terminal_row(
    record: &ReplayEpisodeRecord,
    row: &ReplayRow,
    expected_history: &[PortableSearchActionRef],
    data_mode: ReplayDataMode,
) -> ReplayResult<()> {
    if !data_mode.is_symmetric_selfplay()
        || row.root != record.root
        || row.state != record.final_graph
        || row.action_history != expected_history
        || !row.legal_actions.is_empty()
        || !row.policy_target.is_empty()
        || row.selected_action.is_some()
        || row.final_measure != record.final_measure
        || row.search_config_hash != record.search_config_hash
        || row.value_target != record.outcome.value_target
        || row.reward_target != Some(record.outcome.reward)
    {
        return Err(ReplayError::InvalidRecord);
    }
    validate_value_target(row.value_target, data_mode)?;
    validate_horizon_value_targets(row.horizon_value_targets, data_mode)
}

fn validate_horizon_value_targets(
    targets: Option<[f32; 2]>,
    data_mode: ReplayDataMode,
) -> ReplayResult<()> {
    match (targets, data_mode.is_symmetric_selfplay()) {
        (Some(targets), true)
            if targets
                .iter()
                .all(|target| target.is_finite() && (-1.0..=1.0).contains(target)) =>
        {
            Ok(())
        }
        (None, false) => Ok(()),
        _ => Err(ReplayError::InvalidRecord),
    }
}

fn validate_value_target(value: Option<f32>, data_mode: ReplayDataMode) -> ReplayResult<()> {
    match (value, data_mode.is_symmetric_selfplay()) {
        (Some(value), true) if value.is_finite() && (-1.0..=1.0).contains(&value) => Ok(()),
        (Some(value), false) if value == -1.0 || value == 1.0 => Ok(()),
        (Some(_), _) => Err(ReplayError::InvalidRecord),
        (None, _) => Ok(()),
    }
}
