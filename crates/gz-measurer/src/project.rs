use gz_engine::{MeasureSummary, ModelVersion, PortableSearchActionRef, SearchConfigHash};
use gz_replay::{ReplayEpisodeRecord, ReplayOutcome, ReplayRow, ReplayRowKind};

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedEpisodeArtifact {
    pub root: gz_engine::ReplayGraphContext,
    pub final_graph: gz_engine::ReplayGraphContext,
    pub final_measure: MeasureSummary,
    pub stop_selected: bool,
    pub search_config_hash: SearchConfigHash,
    pub steps: Vec<CompletedEpisodeStep>,
    pub feature_rows: Option<Vec<Vec<u8>>>,
    pub terminal_feature_row: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedEpisodeStep {
    pub before: gz_engine::ReplayGraphContext,
    pub after: gz_engine::ReplayGraphContext,
    pub selected_action: PortableSearchActionRef,
    pub legal_actions: Vec<PortableSearchActionRef>,
    pub policy_target: Vec<f32>,
    pub root_value: Option<f32>,
    pub root_search_value: Option<f32>,
    pub model_version: Option<ModelVersion>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeasurerError {
    Unmeasured,
    FeatureRowCountMismatch,
    StepCountOverflow,
    InvalidRootSearchValue,
}

pub fn project_episode(
    artifact: &CompletedEpisodeArtifact,
) -> Result<(ReplayEpisodeRecord, Vec<ReplayRow>), MeasurerError> {
    let reward = episode_reward(artifact).ok_or(MeasurerError::Unmeasured)?;
    if let Some(feature_rows) = &artifact.feature_rows
        && feature_rows.len() != artifact.steps.len()
    {
        return Err(MeasurerError::FeatureRowCountMismatch);
    }

    let mut action_history = Vec::<PortableSearchActionRef>::new();
    let mut search_steps = Vec::with_capacity(artifact.steps.len());
    let mut rows = Vec::with_capacity(artifact.steps.len());
    for (index, step) in artifact.steps.iter().enumerate() {
        let step_index = u32::try_from(index).map_err(|_| MeasurerError::StepCountOverflow)?;
        let step_ref = gz_engine::SearchStepRef::new(step.before, step.selected_action, step.after)
            .map_err(|_| MeasurerError::StepCountOverflow)?;
        search_steps.push(step_ref);
        rows.push(ReplayRow {
            kind: ReplayRowKind::Decision,
            step_index,
            root: artifact.root,
            state: step.before,
            action_history: action_history.clone(),
            legal_actions: step.legal_actions.clone(),
            policy_target: step.policy_target.clone(),
            selected_action: Some(step.selected_action),
            value_target: None,
            horizon_value_targets: None,
            reward_target: Some(reward),
            final_measure: artifact.final_measure.clone(),
            model_version: step.model_version,
            search_config_hash: artifact.search_config_hash,
            feature_row: artifact
                .feature_rows
                .as_ref()
                .map(|feature_rows| feature_rows[index].clone()),
        });
        action_history.push(step.selected_action);
    }

    if let Some(feature_row) = &artifact.terminal_feature_row {
        let step_index =
            u32::try_from(artifact.steps.len()).map_err(|_| MeasurerError::StepCountOverflow)?;
        rows.push(ReplayRow {
            kind: ReplayRowKind::Terminal,
            step_index,
            root: artifact.root,
            state: artifact.final_graph,
            action_history,
            legal_actions: Vec::new(),
            policy_target: Vec::new(),
            selected_action: None,
            value_target: None,
            horizon_value_targets: None,
            reward_target: Some(reward),
            final_measure: artifact.final_measure.clone(),
            model_version: artifact.steps.last().and_then(|step| step.model_version),
            search_config_hash: artifact.search_config_hash,
            feature_row: Some(feature_row.clone()),
        });
    }

    let record = ReplayEpisodeRecord {
        root: artifact.root,
        final_graph: artifact.final_graph,
        steps: search_steps,
        final_measure: artifact.final_measure.clone(),
        outcome: ReplayOutcome::new(None, reward, artifact.stop_selected),
        search_config_hash: artifact.search_config_hash,
        row_count: rows.len() as u32,
    };

    Ok((record, rows))
}

pub fn episode_reward(artifact: &CompletedEpisodeArtifact) -> Option<f32> {
    if !artifact.final_measure.measured || !artifact.final_measure.valid {
        return None;
    }

    match artifact.final_measure.scalar_reward {
        Some(reward) if reward.is_finite() => Some(reward),
        _ => None,
    }
}

pub fn horizon_value_targets(
    artifact: &CompletedEpisodeArtifact,
    terminal_target: f32,
) -> Result<Vec<[f32; 2]>, MeasurerError> {
    const LAMBDAS: [f32; 2] = [8.0 / 9.0, 32.0 / 33.0];

    if !terminal_target.is_finite() || !(-1.0..=1.0).contains(&terminal_target) {
        return Err(MeasurerError::InvalidRootSearchValue);
    }

    let mut targets = vec![[terminal_target; 2]; artifact.steps.len()];
    let mut next = [terminal_target; 2];
    for (step, target) in artifact.steps.iter().zip(&mut targets).rev() {
        let search_value = step
            .root_search_value
            .filter(|value| value.is_finite() && (-1.0..=1.0).contains(value))
            .ok_or(MeasurerError::InvalidRootSearchValue)?;
        for (value, lambda) in next.iter_mut().zip(LAMBDAS) {
            *value = (1.0 - lambda) * search_value + lambda * *value;
        }
        *target = next;
    }
    Ok(targets)
}
