use crate::root::RootSource;
use crate::{EpisodeId, internal};
use gz_engine::{EngineResult, GraphEngine, MeasureOptions, MeasureResult};
use gz_eval::EvalOutput;
use gz_features::{FeatureExtractor, FeatureRow, PositionFeatures};
use gz_search::{
    EngineIdentity, ExpandResult, ExpandedCandidate, GumbelMcts, SearchHandleBatch, SearchPoll,
    SearchWork, SearchWorkResult, SymmetricEpisode, SymmetricSelfplayEpisodeTask, WorkToken,
};
use std::hash::Hash;
use std::num::NonZeroUsize;

pub(crate) struct WorkerPool<G, C> {
    slots: Vec<Slot<G, C>>,
}

pub(crate) struct CompletedTask<G, C> {
    pub episode_id: EpisodeId,
    pub evaluations: u64,
    pub episode: SymmetricEpisode<G, C>,
}

#[derive(Clone, Debug)]
pub(crate) struct ParkedEval {
    pub episode_id: EpisodeId,
    pub slot: usize,
    pub token: WorkToken,
    pub row: FeatureRow,
    pub action_count: u32,
    pub pressure_reserved: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ParkedMeasure<G> {
    pub slot: usize,
    pub token: WorkToken,
    pub graph: G,
    pub options: MeasureOptions,
}

pub(crate) struct Admission<'a> {
    pub search: &'a GumbelMcts,
    pub identity: EngineIdentity,
    pub pressure_reserved: bool,
    pub next_episode_id: &'a mut u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionResult {
    pub roots_exhausted: bool,
    pub admitted: usize,
}

struct Slot<G, C> {
    state: SlotState<G, C>,
}

struct ActiveEpisode<G, C> {
    task: SymmetricSelfplayEpisodeTask<G, C>,
    episode_id: EpisodeId,
    evaluations: u64,
    pressure_reserved: bool,
}

#[allow(clippy::large_enum_variant)]
enum SlotState<G, C> {
    Idle,
    Running(ActiveEpisode<G, C>),
    Parked {
        episode: ActiveEpisode<G, C>,
        evals: Vec<ParkedEvalState>,
        measure: Option<ParkedMeasureState<G>>,
    },
}

struct ParkedEvalState {
    token: WorkToken,
    row: Option<FeatureRow>,
    action_count: u32,
    sent: bool,
}

struct ParkedMeasureState<G> {
    token: WorkToken,
    graph: G,
    options: MeasureOptions,
    sent: bool,
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

    fn has_parked_token(&self, token: WorkToken) -> bool {
        match self {
            Self::Parked { evals, measure, .. } => {
                evals.iter().any(|eval| eval.token == token)
                    || measure
                        .as_ref()
                        .is_some_and(|measure| measure.token == token)
            }
            _ => false,
        }
    }
}

impl<G, C> WorkerPool<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    pub(crate) fn new(workers: NonZeroUsize) -> Self {
        let slots = (0..workers.get())
            .map(|_| Slot {
                state: SlotState::Idle,
            })
            .collect();
        Self { slots }
    }

    pub(crate) fn admit_limited<E, R, F>(
        &mut self,
        engine: &mut E,
        roots: &mut R,
        admission: &mut Admission<'_>,
        limit: usize,
        mut episode_context: F,
    ) -> EngineResult<AdmissionResult>
    where
        E: GraphEngine<Graph = G, Candidate = C>,
        R: RootSource<E>,
        F: FnMut(&mut E, EpisodeId, G) -> EngineResult<()>,
    {
        if limit == 0 {
            return Ok(AdmissionResult {
                roots_exhausted: false,
                admitted: 0,
            });
        }

        let mut admitted = 0;
        for slot in &mut self.slots {
            if admitted >= limit {
                break;
            }
            if !matches!(slot.state, SlotState::Idle) {
                continue;
            }

            let Some(root) = roots.next_root(engine)? else {
                return Ok(AdmissionResult {
                    roots_exhausted: true,
                    admitted,
                });
            };

            let episode_id = EpisodeId::new(*admission.next_episode_id);
            *admission.next_episode_id += 1;
            let root_is_owned = roots.episode_roots_are_owned();
            if let Err(error) = episode_context(engine, episode_id, root) {
                if root_is_owned {
                    engine.release(&[root], &[])?;
                }
                return Err(error);
            }
            let context = gz_search::GumbelEpisodeContext {
                noise_seed: crate::root::episode_noise_seed(episode_id.value()),
            };
            let mut task = SymmetricSelfplayEpisodeTask::new(
                admission.search,
                admission.identity,
                root,
                context,
            );
            if root_is_owned {
                task.track_owned_root();
            }
            slot.state = SlotState::Running(ActiveEpisode {
                task,
                episode_id,
                evaluations: 0,
                pressure_reserved: admission.pressure_reserved,
            });
            admitted += 1;
        }

        Ok(AdmissionResult {
            roots_exhausted: false,
            admitted,
        })
    }

    pub(crate) fn drive<E, X>(
        &mut self,
        engine: &mut E,
        extractor: &mut X,
        remote_measure: bool,
    ) -> EngineResult<Vec<CompletedTask<G, C>>>
    where
        E: GraphEngine<Graph = G, Candidate = C>,
        X: FeatureExtractor<E>,
    {
        let mut completed = Vec::new();

        for slot in &mut self.slots {
            let Some(mut episode) = slot.state.take_running() else {
                continue;
            };
            let mut parked_evals = Vec::new();
            let mut parked_measure = None;
            loop {
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
                        let result = match service_engine_work(engine, &work, remote_measure) {
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
                            continue;
                        }

                        match work {
                            SearchWork::Eval(work) => {
                                episode.evaluations = episode.evaluations.saturating_add(1);
                                let action_count = match u32::try_from(work.request.actions.len()) {
                                    Ok(action_count) => action_count,
                                    Err(_) => {
                                        release_task_all(engine, &mut episode.task)?;
                                        return Err(internal("action count overflow"));
                                    }
                                };
                                let position = position_features(
                                    work.request.position,
                                    work.opponent.is_some(),
                                );
                                let mut row = match extractor.extract(
                                    engine,
                                    work.graph,
                                    &work.candidates,
                                    position,
                                ) {
                                    Ok(row) => row,
                                    Err(_) => {
                                        release_task_all(engine, &mut episode.task)?;
                                        return Err(internal("feature extraction failed"));
                                    }
                                };
                                if let Some(opponent) = work.opponent.as_deref() {
                                    let opponent_position =
                                        position_features(opponent.position, false);
                                    let opponent_row = match extractor.extract(
                                        engine,
                                        opponent.graph,
                                        &[],
                                        opponent_position,
                                    ) {
                                        Ok(row) => row,
                                        Err(_) => {
                                            release_task_all(engine, &mut episode.task)?;
                                            return Err(internal(
                                                "opponent feature extraction failed",
                                            ));
                                        }
                                    };
                                    row.opponent = Some(opponent_state(opponent_row));
                                }
                                parked_evals.push(ParkedEvalState {
                                    token,
                                    row: Some(row),
                                    action_count,
                                    sent: false,
                                });
                            }
                            SearchWork::Measure(work) if remote_measure => {
                                if parked_measure.is_some() || !parked_evals.is_empty() {
                                    release_task_all(engine, &mut episode.task)?;
                                    return Err(internal("worker produced overlapping work"));
                                }
                                parked_measure = Some(ParkedMeasureState {
                                    token,
                                    graph: work.graph,
                                    options: work.options,
                                    sent: false,
                                });
                            }
                            _ => {
                                release_task_all(engine, &mut episode.task)?;
                                return Err(internal("unsupported search work"));
                            }
                        }
                    }
                    SearchPoll::Blocked => {
                        if parked_evals.is_empty() && parked_measure.is_none() {
                            release_task_all(engine, &mut episode.task)?;
                            return Err(internal("worker blocked"));
                        }
                        slot.state = SlotState::Parked {
                            episode,
                            evals: parked_evals,
                            measure: parked_measure,
                        };
                        break;
                    }
                    SearchPoll::Done(result) => {
                        if !parked_evals.is_empty() || parked_measure.is_some() {
                            release_task_all(engine, &mut episode.task)?;
                            return Err(internal("search completed with pending evaluations"));
                        }
                        completed.push(CompletedTask {
                            episode_id: episode.episode_id,
                            evaluations: episode.evaluations,
                            episode: result,
                        });
                        break;
                    }
                }
            }
        }

        Ok(completed)
    }

    pub(crate) fn take_unsent_parked(&mut self) -> Vec<ParkedEval> {
        let mut parked = Vec::new();
        for (index, slot) in self.slots.iter_mut().enumerate() {
            let SlotState::Parked { episode, evals, .. } = &mut slot.state else {
                continue;
            };
            let mut pressure_reserved = episode.pressure_reserved;
            for eval in evals.iter_mut().filter(|eval| !eval.sent) {
                eval.sent = true;
                parked.push(ParkedEval {
                    episode_id: episode.episode_id,
                    slot: index,
                    token: eval.token,
                    row: eval
                        .row
                        .take()
                        .expect("unsent eval retains its feature row"),
                    action_count: eval.action_count,
                    pressure_reserved,
                });
                pressure_reserved = false;
            }
        }
        parked
    }

    pub(crate) fn take_unsent_measurements(&mut self) -> Vec<ParkedMeasure<G>> {
        let mut parked = Vec::new();
        for (slot, state) in self.slots.iter_mut().enumerate() {
            let SlotState::Parked { measure, .. } = &mut state.state else {
                continue;
            };
            let Some(measure) = measure.as_mut().filter(|measure| !measure.sent) else {
                continue;
            };
            measure.sent = true;
            parked.push(ParkedMeasure {
                slot,
                token: measure.token,
                graph: measure.graph,
                options: measure.options,
            });
        }
        parked
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

        if !slot.state.has_parked_token(token) {
            return Err(internal("resume without pending work"));
        }
        let SlotState::Parked {
            episode,
            evals,
            measure,
        } = &mut slot.state
        else {
            unreachable!("token check ensures the slot is parked");
        };
        if measure.is_some() {
            return Err(internal("eval reply for pending measurement"));
        }
        let index = evals
            .iter()
            .position(|eval| eval.token == token)
            .expect("token check ensures the eval exists");
        evals.swap_remove(index);
        if let Err(error) = episode.task.resume(token, SearchWorkResult::Eval(output)) {
            release_task_all(engine, &mut episode.task)?;
            slot.state = SlotState::Idle;
            return Err(error);
        }
        release_task_releasable(engine, &mut episode.task)?;
        if evals.is_empty() && measure.is_none() {
            let SlotState::Parked { episode, .. } = slot.state.take() else {
                unreachable!("slot remains parked until its final eval reply");
            };
            slot.state = SlotState::Running(episode);
        }
        Ok(())
    }

    pub(crate) fn resume_measure<E>(
        &mut self,
        engine: &mut E,
        slot_index: usize,
        token: WorkToken,
        result: EngineResult<MeasureResult<G>>,
    ) -> EngineResult<()>
    where
        E: GraphEngine<Graph = G, Candidate = C>,
    {
        let slot = self
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| internal("unknown work token"))?;
        if !slot.state.has_parked_token(token) {
            return Err(internal("resume without pending work"));
        }
        let SlotState::Parked {
            episode,
            evals,
            measure,
        } = &mut slot.state
        else {
            unreachable!("token check ensures the slot is parked");
        };
        if !evals.is_empty()
            || measure
                .as_ref()
                .is_none_or(|measure| measure.token != token)
        {
            return Err(internal("measure reply for different pending work"));
        }
        let output = match result {
            Ok(output) => output,
            Err(error) => {
                release_task_all(engine, &mut episode.task)?;
                slot.state = SlotState::Idle;
                return Err(error);
            }
        };
        *measure = None;
        if let Err(error) = episode
            .task
            .resume(token, SearchWorkResult::Measure(output))
        {
            release_task_all(engine, &mut episode.task)?;
            slot.state = SlotState::Idle;
            return Err(error);
        }
        release_task_releasable(engine, &mut episode.task)?;
        let SlotState::Parked { episode, .. } = slot.state.take() else {
            unreachable!("slot remains parked until its measurement reply");
        };
        slot.state = SlotState::Running(episode);
        Ok(())
    }

    pub(crate) fn consume_pressure_reservation(
        &mut self,
        slot_index: usize,
        token: WorkToken,
    ) -> EngineResult<()> {
        let slot = self
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| internal("unknown pressure reservation slot"))?;
        let SlotState::Parked { episode, evals, .. } = &mut slot.state else {
            return Err(internal("pressure reservation without pending work"));
        };
        if !evals.iter().any(|eval| eval.token == token) {
            return Err(internal("pressure reservation token mismatch"));
        }
        episode.pressure_reserved = false;
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

    pub(crate) fn active_count(&self) -> usize {
        self.slots.len() - self.idle_count()
    }

    pub(crate) fn idle_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| matches!(slot.state, SlotState::Idle))
            .count()
    }
}

fn release_task_releasable<E>(
    engine: &mut E,
    task: &mut SymmetricSelfplayEpisodeTask<E::Graph, E::Candidate>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    release_handles(engine, task.take_releasable())
}

fn release_task_all<E>(
    engine: &mut E,
    task: &mut SymmetricSelfplayEpisodeTask<E::Graph, E::Candidate>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    release_handles(engine, task.take_all_handles())
}

fn release_handles<E>(
    engine: &mut E,
    handles: SearchHandleBatch<E::Graph, E::Candidate>,
) -> EngineResult<()>
where
    E: GraphEngine,
{
    if handles.is_empty() {
        return Ok(());
    }
    engine.release(&handles.graphs, &handles.candidates)
}

fn service_engine_work<E>(
    engine: &mut E,
    work: &SearchWork<E::Graph, E::Candidate>,
    remote_measure: bool,
) -> EngineResult<Option<SearchWorkResult<E::Graph, E::Candidate>>>
where
    E: GraphEngine,
{
    match work {
        SearchWork::Expand(work) => {
            service_expand_work(engine, *work).map(|result| Some(SearchWorkResult::Expand(result)))
        }
        SearchWork::Apply(work) => engine
            .apply(work.graph, work.candidate)
            .map(|result| Some(SearchWorkResult::Apply(result))),
        SearchWork::Measure(_) if remote_measure => Ok(None),
        SearchWork::Measure(work) => engine
            .measure(work.graph, work.options)
            .map(|result| Some(SearchWorkResult::Measure(result))),
        SearchWork::Eval(_) => Ok(None),
        _ => Err(internal("unsupported search work")),
    }
}

fn service_expand_work<E>(
    engine: &mut E,
    work: gz_search::ExpandWork<E::Graph>,
) -> EngineResult<ExpandResult<E::Candidate>>
where
    E: GraphEngine,
{
    let mut candidates = Vec::new();
    engine.candidates(work.graph, work.options, &mut candidates)?;
    let graph_hash = engine.hash(work.graph)?;
    let candidates = candidates
        .into_iter()
        .map(|candidate| {
            engine
                .candidate_info(work.graph, candidate)?
                .validate()
                .map_err(|_| internal("invalid candidate info"))
                .map(|info| ExpandedCandidate {
                    candidate,
                    candidate_hash: info.candidate_hash,
                    kind: info.kind,
                    tags: info.tags,
                    static_prior: info.static_prior,
                })
        })
        .collect::<EngineResult<Vec<_>>>()?;

    Ok(ExpandResult {
        graph_hash,
        candidates,
    })
}

fn position_features(
    position: gz_eval::EvalPositionContext,
    dynamic_opponent: bool,
) -> PositionFeatures {
    PositionFeatures {
        root_step: position.root_step,
        leaf_depth: position.leaf_depth,
        budget_fraction: position.budget_fraction,
        budget_step: position.budget_step,
        opponent_reward: 0.0,
        opponent_present: dynamic_opponent,
    }
}

fn opponent_state(row: FeatureRow) -> gz_features::OpponentStateFeatures {
    gz_features::OpponentStateFeatures {
        node_count: row.node_count,
        node_tokens: row.node_tokens,
        node_attrs: row.node_attrs,
        edges: row.edges,
        position: row.position,
    }
}
