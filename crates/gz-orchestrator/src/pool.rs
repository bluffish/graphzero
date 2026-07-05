use crate::root::RootSource;
use crate::serial::OrchestratedEpisode;
use crate::service::{internal, service_engine_work};
use crate::{EpisodeId, WorkerId};
use gz_engine::{EngineResult, GraphEngine};
use gz_eval::{EvalOutput, EvalRequest};
use gz_features::{FeatureExtractor, FeatureRow, PositionFeatures};
use gz_search::{
    EngineIdentity, GumbelEpisodeTask, GumbelMcts, SearchPoll, SearchWork, SearchWorkResult,
    WorkToken,
};
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
    G: Copy + Eq,
    C: Copy,
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

    pub(crate) fn admit<E, R>(
        &mut self,
        engine: &mut E,
        roots: &mut R,
        admission: &mut Admission<'_>,
    ) -> EngineResult<(Vec<(EpisodeId, G)>, bool)>
    where
        E: GraphEngine<Graph = G, Candidate = C>,
        R: RootSource<E>,
    {
        let mut admitted = Vec::new();

        for slot in &mut self.slots {
            if !matches!(slot.state, SlotState::Idle) {
                continue;
            }

            let Some(root) = roots.next_root(engine)? else {
                return Ok((admitted, true));
            };

            let episode_id = EpisodeId::new(*admission.next_episode_id);
            *admission.next_episode_id += 1;
            admitted.push((episode_id, root));
            slot.state = SlotState::Running(ActiveEpisode {
                task: GumbelEpisodeTask::new(
                    admission.search,
                    admission.identity,
                    root,
                    gz_search::GumbelEpisodeContext {
                        noise_seed: crate::root::episode_noise_seed(episode_id.value()),
                        ..admission.context
                    },
                ),
                episode_id,
            });
        }

        Ok((admitted, false))
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

    pub(crate) fn drive<E>(
        &mut self,
        engine: &mut E,
        blocked_message: &'static str,
        mut extractor: Option<&mut dyn FeatureExtractor<E>>,
    ) -> EngineResult<Vec<OrchestratedEpisode<G, C>>>
    where
        E: GraphEngine<Graph = G, Candidate = C>,
    {
        let mut completed = Vec::new();

        for slot in &mut self.slots {
            while let Some(mut episode) = slot.state.take_running() {
                match episode.task.poll()? {
                    SearchPoll::Work(work) => {
                        let token = work.token();
                        if let Some(result) = service_engine_work(engine, &work)? {
                            episode.task.resume(token, result)?;
                            slot.state = SlotState::Running(episode);
                            continue;
                        }

                        let SearchWork::Eval(work) = work else {
                            return Err(internal("unsupported search work"));
                        };
                        let action_count = u32::try_from(work.request.actions.len())
                            .map_err(|_| internal("action count overflow"))?;
                        let row = match extractor.as_deref_mut() {
                            Some(extractor) => {
                                let position = position_features(work.request.position);
                                Some(
                                    extractor
                                        .extract(engine, work.graph, &work.candidates, position)
                                        .map_err(|_| internal("feature extraction failed"))?,
                                )
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
                        slot.state = SlotState::Running(episode);
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

    pub(crate) fn resume(
        &mut self,
        slot_index: usize,
        token: WorkToken,
        output: EvalOutput,
    ) -> EngineResult<()> {
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
        episode.task.resume(token, SearchWorkResult::Eval(output))?;
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

const fn position_features(position: gz_eval::EvalPositionContext) -> PositionFeatures {
    PositionFeatures {
        root_step: position.root_step,
        leaf_depth: position.leaf_depth,
        budget_fraction: position.budget_fraction,
        budget_step: position.budget_step,
    }
}
