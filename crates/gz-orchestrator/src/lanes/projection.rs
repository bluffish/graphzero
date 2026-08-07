use super::ReplayJob;
use crate::internal;
use gz_engine::{EngineResult, GraphEngine};
use gz_features::{
    FeatureExtractor, FeatureSchema, OpponentStateFeatures, PositionFeatures, encode_feature_row,
};
use gz_measurer::{
    CompletedEpisodeArtifact, CompletedEpisodeStep, MeasuredSymmetricGame, MeasurerAdmission,
};
use gz_search::{GumbelMcts, SearchAction, SymmetricActorTrace, SymmetricEpisode};
use std::sync::mpsc::{SyncSender, sync_channel};

pub(super) struct SymmetricFeatureRows<C> {
    pub(super) p1: Vec<Vec<u8>>,
    pub(super) p2: Vec<Vec<u8>>,
    pub(super) p1_terminal: Vec<u8>,
    pub(super) p2_terminal: Vec<u8>,
    pub(super) candidates: Vec<C>,
}

pub(super) fn feature_rows_for_symmetric_episode<E, X>(
    engine: &mut E,
    extractor: &mut X,
    search: &GumbelMcts,
    episode: &SymmetricEpisode<E::Graph, E::Candidate>,
) -> EngineResult<SymmetricFeatureRows<E::Candidate>>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let mut candidates = Vec::new();
    let rows = (|| {
        let p1 = feature_rows_for_symmetric_actor(
            engine,
            extractor,
            search,
            &episode.p1,
            &episode.p2,
            false,
            &mut candidates,
        )?;
        let p2 = feature_rows_for_symmetric_actor(
            engine,
            extractor,
            search,
            &episode.p2,
            &episode.p1,
            true,
            &mut candidates,
        )?;
        let (p1_terminal, p2_terminal) =
            terminal_feature_rows(engine, extractor, search, &episode.p1, &episode.p2)?;
        Ok(SymmetricFeatureRows {
            p1,
            p2,
            p1_terminal,
            p2_terminal,
            candidates: std::mem::take(&mut candidates),
        })
    })();
    match rows {
        Ok(rows) => Ok(rows),
        Err(error) => {
            engine.release(&[], &candidates)?;
            Err(error)
        }
    }
}

fn terminal_feature_rows<E, X>(
    engine: &E,
    extractor: &mut X,
    search: &GumbelMcts,
    p1: &SymmetricActorTrace<E::Graph, E::Candidate>,
    p2: &SymmetricActorTrace<E::Graph, E::Candidate>,
) -> EngineResult<(Vec<u8>, Vec<u8>)>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let terminal_row = |extractor: &mut X,
                        actor: &SymmetricActorTrace<E::Graph, E::Candidate>,
                        opponent: &SymmetricActorTrace<E::Graph, E::Candidate>|
     -> EngineResult<Vec<u8>> {
        let (actor_step, actor_inactive) =
            symmetric_position_state(actor, actor.steps.len(), false, search.config());
        let mut position = replay_position_features(
            episode_position_config(search),
            extractor.schema(),
            actor_step,
        )?;
        if actor_inactive {
            position.budget_step = -position.budget_step.abs();
        }
        position.opponent_present = true;
        let mut row = extractor
            .extract(engine, actor.final_graph, &[], position)
            .map_err(|_| internal("terminal feature extraction failed"))?;
        row.actions.clear();

        let (opponent_step, opponent_inactive) =
            symmetric_position_state(opponent, opponent.steps.len(), false, search.config());
        let mut opponent_position = replay_position_features(
            episode_position_config(search),
            extractor.schema(),
            opponent_step,
        )?;
        if opponent_inactive {
            opponent_position.budget_step = -opponent_position.budget_step.abs();
        }
        let opponent_row = extractor
            .extract(engine, opponent.final_graph, &[], opponent_position)
            .map_err(|_| internal("terminal opponent feature extraction failed"))?;
        row.opponent = Some(OpponentStateFeatures {
            node_count: opponent_row.node_count,
            node_tokens: opponent_row.node_tokens,
            node_attrs: opponent_row.node_attrs,
            edges: opponent_row.edges,
            position: opponent_row.position,
        });
        let mut bytes = Vec::new();
        encode_feature_row(&row, extractor.schema(), &mut bytes)
            .map_err(|_| internal("terminal feature row encoding failed"))?;
        Ok(bytes)
    };

    Ok((
        terminal_row(extractor, p1, p2)?,
        terminal_row(extractor, p2, p1)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn feature_rows_for_symmetric_actor<E, X>(
    engine: &mut E,
    extractor: &mut X,
    search: &GumbelMcts,
    actor: &SymmetricActorTrace<E::Graph, E::Candidate>,
    opponent: &SymmetricActorTrace<E::Graph, E::Candidate>,
    opponent_after_turn: bool,
    created_candidates: &mut Vec<E::Candidate>,
) -> EngineResult<Vec<Vec<u8>>>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let schema = extractor.schema().clone();
    let mut rows = Vec::with_capacity(actor.steps.len());
    let mut candidates = Vec::new();
    for (index, step) in actor.steps.iter().enumerate() {
        candidates.clear();
        engine.candidates(
            step.before,
            search.config().candidate_options,
            &mut candidates,
        )?;
        created_candidates.extend(candidates.iter().copied());
        validate_reenumerated_candidates(
            engine,
            step.before,
            step.step_ref.before,
            &candidates,
            &step.legal_actions,
            search.config().mask_stop,
        )?;
        let (actor_step, actor_inactive) =
            symmetric_position_state(actor, index, false, search.config());
        let mut position = replay_position_features(
            episode_position_config(search),
            extractor.schema(),
            actor_step,
        )?;
        if actor_inactive {
            position.budget_step = -position.budget_step.abs();
        }
        position.opponent_present = true;
        let mut row = extractor
            .extract(engine, step.before, &candidates, position)
            .map_err(|error| {
                eprintln!("event=feature_error role=actor step={index} error={error}");
                internal("feature extraction failed")
            })?;

        let requested_opponent_index = index + usize::from(opponent_after_turn);
        let opponent_index = requested_opponent_index.min(opponent.steps.len());
        let opponent_graph = symmetric_actor_state(opponent, opponent_index);
        let (opponent_step, opponent_inactive) = symmetric_position_state(
            opponent,
            opponent_index,
            requested_opponent_index > opponent.steps.len(),
            search.config(),
        );
        let mut opponent_position = replay_position_features(
            episode_position_config(search),
            extractor.schema(),
            opponent_step,
        )?;
        if opponent_inactive {
            opponent_position.budget_step = -opponent_position.budget_step.abs();
        }
        let opponent_row = extractor
            .extract(engine, opponent_graph, &[], opponent_position)
            .map_err(|error| {
                eprintln!(
                    "event=feature_error role=opponent step={index} opponent_step={opponent_index} error={error}"
                );
                internal("opponent feature extraction failed")
            })?;
        row.opponent = Some(OpponentStateFeatures {
            node_count: opponent_row.node_count,
            node_tokens: opponent_row.node_tokens,
            node_attrs: opponent_row.node_attrs,
            edges: opponent_row.edges,
            position: opponent_row.position,
        });
        let expected_actions = if search.config().mask_stop {
            step.legal_actions.len().saturating_add(1)
        } else {
            step.legal_actions.len()
        };
        if row.actions.len() != expected_actions {
            return Err(internal("symmetric feature row action count mismatch"));
        }

        let mut bytes = Vec::new();
        encode_feature_row(&row, &schema, &mut bytes)
            .map_err(|_| internal("feature row encoding failed"))?;
        rows.push(bytes);
    }
    Ok(rows)
}

fn validate_reenumerated_candidates<E: GraphEngine>(
    engine: &E,
    graph: E::Graph,
    context: gz_engine::ReplayGraphContext,
    candidates: &[E::Candidate],
    expected_actions: &[gz_engine::PortableSearchActionRef],
    mask_stop: bool,
) -> EngineResult<()> {
    let mut expected = expected_actions.iter();
    for candidate in candidates.iter().copied() {
        let Some(gz_engine::PortableSearchActionRef::Candidate(expected_candidate)) =
            expected.next()
        else {
            return Err(internal("symmetric candidate projection shape mismatch"));
        };
        let info = engine.candidate_info(graph, candidate)?;
        if expected_candidate.context != context
            || info.candidate_hash != expected_candidate.candidate_hash
            || info.graph_hash != context.graph.graph_hash
            || info.action_set_hash != context.action_set_hash
        {
            return Err(internal("symmetric candidate projection identity mismatch"));
        }
    }

    match (mask_stop, expected.next(), expected.next()) {
        (true, None, None) => Ok(()),
        (false, Some(gz_engine::PortableSearchActionRef::Stop { context: stop }), None)
            if *stop == context =>
        {
            Ok(())
        }
        _ => Err(internal("symmetric candidate projection shape mismatch")),
    }
}

fn symmetric_actor_state<G: Copy, C>(actor: &SymmetricActorTrace<G, C>, index: usize) -> G {
    if index == 0 {
        actor.root
    } else {
        actor
            .steps
            .get(index - 1)
            .map_or(actor.final_graph, |step| step.after)
    }
}

fn symmetric_position_state<G, C>(
    actor: &SymmetricActorTrace<G, C>,
    decision_count: usize,
    observed_after_trace: bool,
    config: gz_search::GumbelMctsConfig,
) -> (usize, bool) {
    let rewrites = actor.steps[..decision_count]
        .iter()
        .filter(|step| matches!(step.action, SearchAction::Candidate(_)))
        .count();
    let at_trace_end = decision_count == actor.steps.len();
    let inactive = at_trace_end
        && (actor.stopped || rewrites >= config.max_steps || actor.blocked && observed_after_trace);
    (rewrites, inactive)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn measured_symmetric_game<G: Copy, C: Copy>(
    lane: usize,
    episode: &SymmetricEpisode<G, C>,
    rows: &SymmetricFeatureRows<C>,
) -> MeasuredSymmetricGame {
    MeasuredSymmetricGame {
        lane,
        p1_artifact: symmetric_artifact(
            &episode.p1,
            &rows.p1,
            &rows.p1_terminal,
            episode.search_config_hash,
        ),
        p2_artifact: symmetric_artifact(
            &episode.p2,
            &rows.p2,
            &rows.p2_terminal,
            episode.search_config_hash,
        ),
    }
}

fn symmetric_artifact<G: Copy, C>(
    actor: &SymmetricActorTrace<G, C>,
    feature_rows: &[Vec<u8>],
    terminal_feature_row: &[u8],
    search_config_hash: gz_engine::SearchConfigHash,
) -> CompletedEpisodeArtifact {
    CompletedEpisodeArtifact {
        root: actor.root_context,
        final_graph: actor.final_context,
        final_measure: gz_engine::MeasureSummary::from(&actor.final_measure),
        stop_selected: actor.stopped,
        search_config_hash,
        steps: actor
            .steps
            .iter()
            .map(|step| CompletedEpisodeStep {
                before: step.step_ref.before,
                after: step.step_ref.after,
                selected_action: step.selected_action,
                legal_actions: step.legal_actions.clone(),
                policy_target: step.policy_target.clone(),
                root_value: Some(step.root_value),
                root_search_value: Some(step.root_search_value),
                model_version: Some(step.model_version),
            })
            .collect(),
        feature_rows: Some(feature_rows.to_vec()),
        terminal_feature_row: Some(terminal_feature_row.to_vec()),
    }
}

struct EpisodePositionConfig {
    max_steps: usize,
    export_position: bool,
}

fn episode_position_config(search: &GumbelMcts) -> EpisodePositionConfig {
    let config = search.config();
    EpisodePositionConfig {
        max_steps: config.max_steps,
        export_position: config.export_position,
    }
}

fn replay_position_features(
    config: EpisodePositionConfig,
    _schema: &FeatureSchema,
    index: usize,
) -> EngineResult<PositionFeatures> {
    let (root_step, budget_fraction, budget_step) = if config.export_position {
        let budget_step = if config.max_steps == 0 {
            0.0
        } else {
            1.0 / config.max_steps as f32
        };
        let budget_fraction = if config.max_steps == 0 {
            1.0
        } else {
            config.max_steps.saturating_sub(index) as f32 / config.max_steps as f32
        };
        (
            u32::try_from(index).map_err(|_| internal("root step overflow"))?,
            budget_fraction,
            budget_step,
        )
    } else {
        (0, 0.0, 0.0)
    };
    Ok(PositionFeatures {
        root_step,
        leaf_depth: 0,
        budget_fraction,
        budget_step,
        opponent_reward: 0.0,
        opponent_present: false,
    })
}

pub(super) fn release_symmetric_episode_handles<E>(
    engine: &mut E,
    episode: &SymmetricEpisode<E::Graph, E::Candidate>,
    extra_candidates: &[E::Candidate],
) -> EngineResult<()>
where
    E: GraphEngine,
{
    release_created_handles(
        engine,
        &episode.created_graphs,
        &episode.created_candidates,
        extra_candidates,
    )
}

fn release_created_handles<E>(
    engine: &mut E,
    created_graphs: &[E::Graph],
    created_candidates: &[E::Candidate],
    extra_candidates: &[E::Candidate],
) -> EngineResult<()>
where
    E: GraphEngine,
{
    if extra_candidates.is_empty() {
        return engine.release(created_graphs, created_candidates);
    }

    let mut candidates = Vec::with_capacity(created_candidates.len() + extra_candidates.len());
    candidates.extend_from_slice(created_candidates);
    candidates.extend_from_slice(extra_candidates);
    engine.release(created_graphs, &candidates)
}

pub(super) fn append_symmetric_replay_job(
    replay_tx: &SyncSender<ReplayJob>,
    game: MeasuredSymmetricGame,
) -> EngineResult<MeasurerAdmission> {
    let (ack, done) = sync_channel(1);
    replay_tx
        .send(ReplayJob::Symmetric {
            game: Box::new(game),
            ack,
        })
        .map_err(|_| internal("replay sink failed"))?;
    done.recv().map_err(|_| internal("replay sink failed"))?
}
