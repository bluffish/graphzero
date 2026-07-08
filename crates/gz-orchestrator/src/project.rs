use crate::reference::Reference;
use gz_engine::{MeasureSummary, PortableSearchActionRef};
use gz_replay::{ReplayEpisodeRecord, ReplayOutcome, ReplayReference, ReplayRow};
use gz_search::{GumbelEpisode, GumbelStopReason};

pub fn project_episode<G, C>(
    episode: &GumbelEpisode<G, C>,
    reference: Option<&Reference>,
    feature_rows: Option<&[Vec<u8>]>,
    length_tiebreak: bool,
    episode_id: u64,
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
    let value_target = reference.map(|reference| {
        let reference_len =
            (length_tiebreak && !reference.steps.is_empty()).then_some(reference.steps.len());
        sign_target(
            learner_reward,
            reference.final_reward,
            episode.steps.len(),
            reference_len,
            episode_id,
        )
    });
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

/// The learner reward an episode would project with, if eligible.
/// Lets callers that drop an episode from the store still feed the
/// reference provider's reward statistics.
pub fn episode_reward<G, C>(episode: &GumbelEpisode<G, C>) -> Option<f32> {
    score(
        episode.final_measure.measured,
        episode.final_measure.valid,
        episode.final_measure.scalar_reward,
    )
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

fn sign_target(
    learner: f32,
    reference: f32,
    learner_len: usize,
    reference_len: Option<usize>,
    episode_id: u64,
) -> f32 {
    if learner > reference {
        return 1.0;
    }
    if learner < reference {
        return -1.0;
    }
    // Length tie-break (whittlezero's ptp_duration_tiebreak, discrete
    // form): at equal reward the shorter episode wins, so reward ties
    // carry an efficiency signal instead of noise. Both lengths count
    // moves; references without per-step states opt out upstream
    // (reference_len None).
    if let Some(reference_len) = reference_len {
        if learner_len < reference_len {
            return 1.0;
        }
        if learner_len > reference_len {
            return -1.0;
        }
    }
    // Random tie-break (whittlezero's ptp_sign_tie_break: random). A
    // zero target is a safe haven the search can lock onto: stopping
    // at the root guarantees the tie, so visits, policy targets, and
    // finally the argmax reference all converge on stop-at-root. A
    // fair deterministic coin keeps every label hard +/-1. Salted so
    // the coin is independent of the episode's Gumbel noise stream.
    const TIE_SALT: u64 = 0x7469_655f_6272_6561; // "tie_brea"
    if crate::root::episode_noise_seed(episode_id ^ TIE_SALT) & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}
