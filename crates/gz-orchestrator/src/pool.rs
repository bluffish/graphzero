use crate::root::RootSource;
use crate::serial::OrchestratedEpisode;
use crate::service::{internal, service_engine_work};
use crate::{EpisodeId, WorkerId};
use gz_engine::{EngineResult, GraphEngine};
use gz_eval::{EvalOutput, EvalRequest};
use gz_features::{FeatureExtractor, FeatureRow, PositionFeatures};
use gz_search::{
    EngineIdentity, GumbelEpisodeTask, GumbelHandleBatch, GumbelMcts, SearchPoll, SearchWork,
    SearchWorkResult, WorkToken,
};
use std::hash::Hash;
use std::num::NonZeroUsize;

pub(crate) struct WorkerPool<G, C> {
    slots: Vec<Slot<G, C>>,
}

#[derive(Clone, Debug)]
pub(crate) struct ParkedEval {
    pub slot: usize,
    pub token: WorkToken,
    pub request: EvalRequest,
    pub row: Option<FeatureRow>,
    pub action_count: u32,
}

pub(crate) struct Admission<'a> {
    pub search: &'a GumbelMcts,
    pub identity: EngineIdentity,
    pub context: gz_search::GumbelEpisodeContext,
    pub next_episode_id: &'a mut u64,
}

struct Slot<G, C> {
    worker_id: WorkerId,
    state: SlotState<G, C>,
}

struct ActiveEpisode<G, C> {
    task: GumbelEpisodeTask<G, C>,
    episode_id: EpisodeId,
}

#[allow(clippy::large_enum_variant)]
enum SlotState<G, C> {
    Idle,
    Running(ActiveEpisode<G, C>),
    Parked {
        episode: ActiveEpisode<G, C>,
        token: WorkToken,
        request: EvalRequest,
        row: Option<FeatureRow>,
        action_count: u32,
        sent: bool,
    },
}

impl<G, C> SlotState<G, C> {
    fn take(&mut self) -> Self {
        std::mem::replace(self, Self::Idle)
    }

    fn take_running(&mut self) -> Option<ActiveEpisode<G, C>> {
        match self.take() {
            Self::Running(episode) => Some(episode),
            other => {
                *self = other;
                None
            }
        }
    }

    fn take_parked(&mut self) -> Option<ActiveEpisode<G, C>> {
        match self.take() {
            Self::Parked { episode, .. } => Some(episode),
            other => {
                *self = other;
                None
            }
        }
    }

    fn parked_token(&self) -> Option<WorkToken> {
        match self {
            Self::Parked { token, .. } => Some(*token),
            _ => None,
        }
    }
}

impl<G, C> WorkerPool<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    pub(crate) fn new(workers: NonZeroUsize, worker_id_base: u64) -> Self {
        let slots = (0..workers.get())
            .map(|index| Slot {
                worker_id: WorkerId::new(worker_id_base + index as u64),
                state: SlotState::Idle,
            })
            .collect();
        Self { slots }
    }

    pub(crate) fn admit<E, R, F>(
        &mut self,
        engine: &mut E,
        roots: &mut R,
        admission: &mut Admission<'_>,
        mut episode_context: F,
    ) -> EngineResult<bool>
    where
        E: GraphEngine<Graph = G, Candidate = C>,
        R: RootSource<E>,
        F: FnMut(
            &mut E,
            EpisodeId,
            G,
            gz_search::GumbelEpisodeContext,
        ) -> EngineResult<gz_search::GumbelEpisodeContext>,
    {
        for slot in &mut self.slots {
            if !matches!(slot.state, SlotState::Idle) {
                continue;
            }

            let Some(root) = roots.next_root(engine)? else {
                return Ok(true);
            };

            let episode_id = EpisodeId::new(*admission.next_episode_id);
            *admission.next_episode_id += 1;
            let context = episode_context(
                engine,
                episode_id,
                root,
                gz_search::GumbelEpisodeContext {
                    noise_seed: crate::root::episode_noise_seed(episode_id.value()),
                    ..admission.context
                },
            )?;
            slot.state = SlotState::Running(ActiveEpisode {
                task: GumbelEpisodeTask::new(admission.search, admission.identity, root, context),
                episode_id,
            });
        }

        Ok(false)
    }

    /// Admits one episode outside the root source -- the opponent rollout
    /// path. The caller supplies the root, search config, context, and
    /// episode id. Returns false when no worker slot is idle.
    pub(crate) fn admit_direct(
        &mut self,
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: gz_search::GumbelEpisodeContext,
        episode_id: EpisodeId,
    ) -> bool {
        for slot in &mut self.slots {
            if !matches!(slot.state, SlotState::Idle) {
                continue;
            }

            slot.state = SlotState::Running(ActiveEpisode {
                task: GumbelEpisodeTask::new(search, identity, root, context),
                episode_id,
            });
            return true;
        }

        false
    }

    pub(crate) fn drive<E, F>(
        &mut self,
        engine: &mut E,
        blocked_message: &'static str,
        mut extractor: Option<&mut dyn FeatureExtractor<E>>,
        mut decorate_row: F,
    ) -> EngineResult<Vec<OrchestratedEpisode<G, C>>>
    where
        E: GraphEngine<Graph = G, Candidate = C>,
        F: FnMut(EpisodeId, u32, &mut FeatureRow),
    {
        let mut completed = Vec::new();

        for slot in &mut self.slots {
            while let Some(mut episode) = slot.state.take_running() {
                let poll = match episode.task.poll() {
                    Ok(poll) => poll,
                    Err(error) => {
                        release_task_all(engine, &mut episode.task)?;
                        return Err(error);
                    }
                };
                release_task_releasable(engine, &mut episode.task)?;

                match poll {
                    SearchPoll::Work(work) => {
                        let token = work.token();
                        let result = match service_engine_work(engine, &work) {
                            Ok(result) => result,
                            Err(error) => {
                                release_task_all(engine, &mut episode.task)?;
                                return Err(error);
                            }
                        };
                        if let Some(result) = result {
                            if let Err(error) = episode.task.resume(token, result) {
                                release_task_all(engine, &mut episode.task)?;
                                return Err(error);
                            }
                            release_task_releasable(engine, &mut episode.task)?;
                            slot.state = SlotState::Running(episode);
                            continue;
                        }

                        let SearchWork::Eval(work) = work else {
                            release_task_all(engine, &mut episode.task)?;
                            return Err(internal("unsupported search work"));
                        };
                        let action_count = match u32::try_from(work.request.actions.len()) {
                            Ok(action_count) => action_count,
                            Err(_) => {
                                release_task_all(engine, &mut episode.task)?;
                                return Err(internal("action count overflow"));
                            }
                        };
                        let row = match extractor.as_deref_mut() {
                            Some(extractor) => {
                                let scale = extractor.schema().config().opponent_reward_scale;
                                let position = position_features(work.request.position, scale);
                                match extractor.extract(
                                    engine,
                                    work.graph,
                                    &work.candidates,
                                    position,
                                ) {
                                    Ok(mut row) => {
                                        // Pair evals attach the opponent state at
                                        // the row the search aligned to (real root
                                        // step + leaf depth, advanced to the
                                        // opponent's horizon for STOP re-evals) --
                                        // never the request's exported root_step,
                                        // which export_position zeroes. The task's
                                        // real step is the fallback for references
                                        // without per-step states.
                                        let root_step =
                                            match u32::try_from(episode.task.step_index()) {
                                                Ok(root_step) => root_step,
                                                Err(_) => {
                                                    release_task_all(engine, &mut episode.task)?;
                                                    return Err(internal("root step overflow"));
                                                }
                                            };
                                        let opponent_row = work
                                            .request
                                            .position
                                            .opponent_row()
                                            .unwrap_or(root_step);
                                        decorate_row(episode.episode_id, opponent_row, &mut row);
                                        Some(row)
                                    }
                                    Err(_) => {
                                        release_task_all(engine, &mut episode.task)?;
                                        return Err(internal("feature extraction failed"));
                                    }
                                }
                            }
                            None => None,
                        };
                        slot.state = SlotState::Parked {
                            episode,
                            token,
                            request: work.request,
                            row,
                            action_count,
                            sent: false,
                        };
                    }
                    SearchPoll::Blocked => {
                        release_task_all(engine, &mut episode.task)?;
                        return Err(internal(blocked_message));
                    }
                    SearchPoll::Done(result) => {
                        completed.push(OrchestratedEpisode {
                            worker_id: slot.worker_id,
                            episode_id: episode.episode_id,
                            episode: result,
                        });
                    }
                }
            }
        }

        Ok(completed)
    }

    pub(crate) fn parked(&self) -> Vec<ParkedEval> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| match &slot.state {
                SlotState::Parked {
                    token,
                    request,
                    row,
                    action_count,
                    ..
                } => Some(ParkedEval {
                    slot: index,
                    token: *token,
                    request: request.clone(),
                    row: row.clone(),
                    action_count: *action_count,
                }),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn take_unsent_parked(&mut self) -> Vec<ParkedEval> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(index, slot)| match &mut slot.state {
                SlotState::Parked {
                    token,
                    request,
                    row,
                    action_count,
                    sent,
                    ..
                } if !*sent => {
                    *sent = true;
                    Some(ParkedEval {
                        slot: index,
                        token: *token,
                        request: request.clone(),
                        row: row.clone(),
                        action_count: *action_count,
                    })
                }
                _ => None,
            })
            .collect()
    }

    pub(crate) fn resume<E>(
        &mut self,
        engine: &mut E,
        slot_index: usize,
        token: WorkToken,
        output: EvalOutput,
    ) -> EngineResult<()>
    where
        E: GraphEngine<Graph = G, Candidate = C>,
    {
        let slot = self
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| internal("unknown work token"))?;

        let Some(expected) = slot.state.parked_token() else {
            return Err(internal("resume without pending work"));
        };
        if expected != token {
            return Err(internal("unknown work token"));
        }

        let mut episode = slot
            .state
            .take_parked()
            .expect("token check ensures the slot is parked");
        if let Err(error) = episode.task.resume(token, SearchWorkResult::Eval(output)) {
            release_task_all(engine, &mut episode.task)?;
            return Err(error);
        }
        release_task_releasable(engine, &mut episode.task)?;
        slot.state = SlotState::Running(episode);
        Ok(())
    }

    pub(crate) fn has_running(&self) -> bool {
        self.slots
            .iter()
            .any(|slot| matches!(slot.state, SlotState::Running(_)))
    }

    pub(crate) fn has_parked(&self) -> bool {
        self.slots
            .iter()
            .any(|slot| matches!(slot.state, SlotState::Parked { .. }))
    }

    pub(crate) fn active(&self) -> bool {
        self.has_running() || self.has_parked()
    }
}

fn release_task_releasable<E>(
    engine: &mut E,
    task: &mut GumbelEpisodeTask<E::Graph, E::Candidate>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    release_handles(engine, task.take_releasable())
}

fn release_task_all<E>(
    engine: &mut E,
    task: &mut GumbelEpisodeTask<E::Graph, E::Candidate>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    release_handles(engine, task.take_all_handles())
}

fn release_handles<E>(
    engine: &mut E,
    handles: GumbelHandleBatch<E::Graph, E::Candidate>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    if handles.is_empty() {
        return Ok(());
    }
    engine.release(&handles.graphs, &handles.candidates)
}

fn position_features(
    position: gz_eval::EvalPositionContext,
    opponent_reward_scale: f32,
) -> PositionFeatures {
    let opponent = position.opponent;
    PositionFeatures {
        root_step: position.root_step,
        leaf_depth: position.leaf_depth,
        budget_fraction: position.budget_fraction,
        budget_step: position.budget_step,
        opponent_reward: opponent.map_or(0.0, |opponent| {
            opponent.final_reward / opponent_reward_scale
        }),
        opponent_present: opponent.is_some(),
    }
}
