use crate::reference::Reference;
use gz_engine::{MeasureSummary, PortableSearchActionRef};
use gz_replay::{ReplayEpisodeRecord, ReplayOutcome, ReplayReference, ReplayRow};
use gz_search::{GumbelEpisode, GumbelStopReason};

pub fn project_episode<G, C>(
    episode: &GumbelEpisode<G, C>,
    reference: Option<&Reference<G>>,
    feature_rows: Option<&[Vec<u8>]>,
) -> Option<(ReplayEpisodeRecord, Vec<ReplayRow>)> {
    let learner_reward = score(
        episode.final_measure.measured,
        episode.final_measure.valid,
        episode.final_measure.scalar_reward,
    )?;
    if let Some(feature_rows) = feature_rows
        && feature_rows.len() != episode.steps.len()
    {
        return None;
    }

    let final_measure = MeasureSummary::from(&episode.final_measure);
    let value_target =
        reference.map(|reference| sign_target(learner_reward, reference.final_reward));
    let replay_reference = reference.map(|reference| ReplayReference {
        kind: reference.kind,
        reward: reference.final_reward,
        final_graph: reference.final_graph,
        trajectory_id: None,
        search_config_hash: reference.search_config_hash,
        model_version: reference.model_version,
    });
    let mut action_history = Vec::<PortableSearchActionRef>::new();
    let mut rows = Vec::with_capacity(episode.steps.len());

    for (index, step) in episode.steps.iter().enumerate() {
        rows.push(ReplayRow {
            step_index: index as u32,
            root: episode.root_context,
            state: step.step_ref.before,
            action_history: action_history.clone(),
            legal_actions: step.legal_actions.clone(),
            policy_target: step.policy_target.clone(),
            selected_action: step.selected_action,
            value_target,
            reward_target: Some(learner_reward),
            final_measure: final_measure.clone(),
            model_version: Some(step.model_version),
            search_config_hash: episode.search_config_hash,
            feature_row: feature_rows.map(|rows| rows[index].clone()),
        });
        action_history.push(step.selected_action);
    }

    let record = ReplayEpisodeRecord {
        root: episode.root_context,
        final_graph: episode.final_context,
        steps: episode.steps.iter().map(|step| step.step_ref).collect(),
        final_measure,
        outcome: ReplayOutcome {
            value_target,
            learner_reward,
            reference: replay_reference,
            stopped: matches!(episode.stop_reason, GumbelStopReason::SelectedStop),
        },
        search_config_hash: episode.search_config_hash,
        row_count: rows.len() as u32,
    };

    Some((record, rows))
}

fn score(measured: bool, valid: bool, scalar_reward: Option<f32>) -> Option<f32> {
    if !measured || !valid {
        return None;
    }

    match scalar_reward {
        Some(reward) if reward.is_finite() => Some(reward),
        _ => None,
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
