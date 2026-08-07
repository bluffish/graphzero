use super::super::{
    GumbelEpisodeContext, GumbelHandleBatch, GumbelMcts, GumbelMctsConfig, GumbelPlayer,
    GumbelRootStats, GumbelStep, GumbelValueMode,
};
use super::root::{SymmetricRootResult, SymmetricSelfplayRootTask};
use crate::support::internal;
use crate::work::{
    EngineIdentity, MeasurePairWork, SearchPoll, SearchWork, SearchWorkResult, WorkToken,
};
use gz_engine::{EngineResult, MeasureResult, ReplayGraphContext, SearchConfigHash};
use std::collections::HashSet;
use std::hash::Hash;

pub struct SymmetricActorTrace<G, C> {
    pub root: G,
    pub final_graph: G,
    pub root_context: ReplayGraphContext,
    pub final_context: ReplayGraphContext,
    pub steps: Vec<GumbelStep<G, C>>,
    pub root_stats: Vec<GumbelRootStats>,
    pub blocked: bool,
    pub stopped: bool,
    pub final_measure: MeasureResult<G>,
}

pub struct SymmetricEpisode<G, C> {
    pub p1: SymmetricActorTrace<G, C>,
    pub p2: SymmetricActorTrace<G, C>,
    pub search_config_hash: SearchConfigHash,
    pub created_graphs: Vec<G>,
    pub created_candidates: Vec<C>,
}

pub struct SymmetricSelfplayEpisodeTask<G, C> {
    config: GumbelMctsConfig,
    search_config_hash: SearchConfigHash,
    identity: EngineIdentity,
    context: GumbelEpisodeContext,
    actors: [Actor<G, C>; 2],
    player: GumbelPlayer,
    visited: [HashSet<ReplayGraphContext>; 2],
    path_graphs: Vec<G>,
    releasable: GumbelHandleBatch<G, C>,
    next_token: u64,
    root_pending: Vec<RootPending>,
    pending: Option<Pending>,
    state: State<G, C>,
}

impl<G, C> SymmetricSelfplayEpisodeTask<G, C>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
{
    pub fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: GumbelEpisodeContext,
    ) -> Self {
        assert_eq!(
            search.config().value_mode,
            GumbelValueMode::SymmetricSelfplay,
            "symmetric task requires symmetric value mode"
        );
        Self {
            config: search.config(),
            search_config_hash: crate::symmetric_selfplay_search_config_hash(
                search.search_config_hash(),
            ),
            identity,
            context,
            actors: [Actor::new(root), Actor::new(root)],
            player: GumbelPlayer::One,
            visited: [HashSet::new(), HashSet::new()],
            path_graphs: Vec::new(),
            releasable: GumbelHandleBatch::default(),
            next_token: 0,
            root_pending: Vec::new(),
            pending: None,
            state: State::Turn,
        }
    }

    /// Resumes symmetric self-play from an explicit pair position. The input
    /// graph handles remain caller-owned; only graphs created after resumption
    /// are returned by the task.
    #[allow(clippy::too_many_arguments)]
    pub fn from_position(
        search: &GumbelMcts,
        identity: EngineIdentity,
        graphs: [G; 2],
        contexts: [Option<ReplayGraphContext>; 2],
        rewrites: [usize; 2],
        blocked: [bool; 2],
        stopped: [bool; 2],
        player: GumbelPlayer,
        visited: [HashSet<ReplayGraphContext>; 2],
        context: GumbelEpisodeContext,
    ) -> Self {
        assert_eq!(
            search.config().value_mode,
            GumbelValueMode::SymmetricSelfplay,
            "symmetric task requires symmetric value mode"
        );
        Self {
            config: search.config(),
            search_config_hash: crate::symmetric_selfplay_search_config_hash(
                search.search_config_hash(),
            ),
            identity,
            context,
            actors: [
                Actor::from_position(graphs[0], contexts[0], rewrites[0], blocked[0], stopped[0]),
                Actor::from_position(graphs[1], contexts[1], rewrites[1], blocked[1], stopped[1]),
            ],
            player,
            visited,
            path_graphs: Vec::new(),
            releasable: GumbelHandleBatch::default(),
            next_token: 0,
            root_pending: Vec::new(),
            pending: None,
            state: State::Turn,
        }
    }

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, SymmetricEpisode<G, C>>> {
        if self.pending.is_some() {
            return Ok(SearchPoll::Blocked);
        }
        loop {
            let state = std::mem::replace(&mut self.state, State::Done);
            match state {
                State::Turn => self.start_turn()?,
                State::Root(mut task) => {
                    let poll = match task.poll() {
                        Ok(poll) => poll,
                        Err(error) => {
                            self.append_releasable(task.take_all_handles());
                            return Err(error);
                        }
                    };
                    self.append_releasable(task.take_releasable());
                    match poll {
                        SearchPoll::Work(work) => {
                            let outer = self.next_token();
                            let inner = work.token();
                            self.root_pending.push(RootPending { outer, inner });
                            self.state = State::Root(task);
                            return Ok(SearchPoll::Work(retokenize(work, outer)));
                        }
                        SearchPoll::Blocked => {
                            self.state = State::Root(task);
                            return Ok(SearchPoll::Blocked);
                        }
                        SearchPoll::Done(result) => {
                            if !self.root_pending.is_empty() {
                                self.append_releasable(task.take_all_handles());
                                return Err(internal("symmetric root completed with pending work"));
                            }
                            if let Err(error) = self.finish_turn(result) {
                                self.append_releasable(task.take_all_handles());
                                return Err(error);
                            }
                            let mut reused = match task.take_reused_task(self.visited.clone()) {
                                Ok(reused) => reused,
                                Err(error) => {
                                    self.append_releasable(task.take_all_handles());
                                    return Err(error);
                                }
                            };
                            self.append_releasable(task.take_all_handles());
                            if let Some(mut reused_task) = reused.take() {
                                while !self.actor_active(self.player)
                                    && (self.actor_active(GumbelPlayer::One)
                                        || self.actor_active(GumbelPlayer::Two))
                                {
                                    self.player = self.player.opponent();
                                }
                                let contexts = match reused_task.align_root_board(
                                    [self.actors[0].current, self.actors[1].current],
                                    [self.actors[0].context, self.actors[1].context],
                                    [self.actors[0].rewrites, self.actors[1].rewrites],
                                    [
                                        self.actors[0].blocked || self.actors[0].stopped,
                                        self.actors[1].blocked || self.actors[1].stopped,
                                    ],
                                    [self.actors[0].stopped, self.actors[1].stopped],
                                    self.player,
                                ) {
                                    Ok(contexts) => contexts,
                                    Err(error) => {
                                        self.append_releasable(reused_task.take_all_handles());
                                        return Err(error);
                                    }
                                };
                                if !self.actor_active(self.player) {
                                    self.append_releasable(reused_task.take_all_handles());
                                    return Err(internal("symmetric reused root is not active"));
                                }
                                self.actors[0].context = contexts[0];
                                self.actors[1].context = contexts[1];
                                self.state = State::Root(reused_task);
                            }
                        }
                    }
                }
                State::MeasurePair => {
                    let token = self.next_token();
                    self.pending = Some(Pending::MeasurePair { token });
                    return Ok(SearchPoll::Work(SearchWork::MeasurePair(MeasurePairWork {
                        token,
                        graphs: [self.actors[0].current, self.actors[1].current],
                        options: self.config.measure_options,
                    })));
                }
                State::DoneResult(result) => {
                    self.state = State::Done;
                    return Ok(SearchPoll::Done(*result));
                }
                State::Done => return Err(internal("poll after symmetric episode completion")),
            }
        }
    }

    pub fn resume(&mut self, token: WorkToken, result: SearchWorkResult<G, C>) -> EngineResult<()> {
        if let Some(index) = self
            .root_pending
            .iter()
            .position(|pending| pending.outer == token)
        {
            let pending = self.root_pending.swap_remove(index);
            let state = std::mem::replace(&mut self.state, State::Done);
            let State::Root(mut task) = state else {
                self.state = state;
                self.root_pending.push(pending);
                return Err(internal("symmetric root result outside root state"));
            };
            if let Err(error) = task.resume(pending.inner, result) {
                self.append_releasable(task.take_all_handles());
                return Err(error);
            }
            self.append_releasable(task.take_releasable());
            self.state = State::Root(task);
            return Ok(());
        }

        let pending = self
            .pending
            .take()
            .ok_or_else(|| internal("resume without pending symmetric episode work"))?;
        if pending.token() != token {
            self.pending = Some(pending);
            return Err(internal("unknown symmetric episode work token"));
        }
        match pending {
            Pending::MeasurePair { token } => match result {
                SearchWorkResult::MeasurePair([p1_measure, p2_measure]) => {
                    self.finish_episode(p1_measure, p2_measure)
                }
                _ => {
                    self.pending = Some(Pending::MeasurePair { token });
                    Err(internal("mismatched symmetric measure pair"))
                }
            },
        }
    }

    pub fn step_index(&self) -> usize {
        self.actors[self.player.index()].rewrites
    }

    pub fn take_releasable(&mut self) -> GumbelHandleBatch<G, C> {
        self.drain_root(false);
        std::mem::take(&mut self.releasable)
    }

    pub fn track_owned_root(&mut self) {
        self.path_graphs.push(self.actors[0].root);
    }

    pub fn take_all_handles(&mut self) -> GumbelHandleBatch<G, C> {
        self.drain_root(true);
        let mut handles = self.take_releasable();
        handles.graphs.append(&mut self.path_graphs);
        handles
    }

    fn start_turn(&mut self) -> EngineResult<()> {
        if !self.actor_active(GumbelPlayer::One) && !self.actor_active(GumbelPlayer::Two) {
            self.state = State::MeasurePair;
            return Ok(());
        }
        if !self.actor_active(self.player) {
            self.player = self.player.opponent();
            self.state = State::Turn;
            return Ok(());
        }
        let task = SymmetricSelfplayRootTask::new(
            &GumbelMcts::new(self.config),
            self.identity,
            [self.actors[0].current, self.actors[1].current],
            [self.actors[0].context, self.actors[1].context],
            [self.actors[0].rewrites, self.actors[1].rewrites],
            [
                self.actors[0].blocked || self.actors[0].stopped,
                self.actors[1].blocked || self.actors[1].stopped,
            ],
            [self.actors[0].stopped, self.actors[1].stopped],
            self.player,
            self.context.noise_seed,
            self.visited.clone(),
        );
        self.state = State::Root(task);
        Ok(())
    }

    fn finish_turn(&mut self, result: SymmetricRootResult<G, C>) -> EngineResult<()> {
        match result {
            SymmetricRootResult::Pass { player, context } => {
                if player != self.player {
                    return Err(internal("symmetric pass player mismatch"));
                }
                let actor = &mut self.actors[player.index()];
                actor.context.get_or_insert(context);
                actor.root_context.get_or_insert(context);
                actor.blocked = true;
            }
            SymmetricRootResult::Action(result) => {
                let selected_stop = matches!(result.step.action, crate::SearchAction::Stop);
                if result.player != self.player {
                    if !selected_stop {
                        self.releasable.graphs.push(result.selected_after);
                    }
                    return Err(internal("symmetric action player mismatch"));
                }
                let player = result.player;
                let before_context = result.step.step_ref.before;
                let actor = &mut self.actors[player.index()];
                actor.root_context.get_or_insert(before_context);
                actor.current = result.selected_after;
                actor.context = Some(result.selected_after_context);
                actor.steps.push(result.step);
                actor.root_stats.push(result.stats);
                if selected_stop {
                    actor.stopped = true;
                } else {
                    actor.rewrites += 1;
                    if self.config.no_backtrack {
                        self.visited[player.index()].insert(before_context);
                    }
                    self.path_graphs.push(result.selected_after);
                }
            }
        }
        self.player = self.player.opponent();
        self.state = State::Turn;
        Ok(())
    }

    fn finish_episode(
        &mut self,
        p1_measure: MeasureResult<G>,
        p2_measure: MeasureResult<G>,
    ) -> EngineResult<()> {
        if !valid_measure(&p1_measure) || !valid_measure(&p2_measure) {
            return Err(internal("symmetric episode final measure is invalid"));
        }
        let p1 = self.finish_actor(GumbelPlayer::One, p1_measure)?;
        let p2 = self.finish_actor(GumbelPlayer::Two, p2_measure)?;
        self.state = State::DoneResult(Box::new(SymmetricEpisode {
            p1,
            p2,
            search_config_hash: self.search_config_hash,
            created_graphs: std::mem::take(&mut self.path_graphs),
            created_candidates: Vec::new(),
        }));
        Ok(())
    }

    fn finish_actor(
        &mut self,
        player: GumbelPlayer,
        final_measure: MeasureResult<G>,
    ) -> EngineResult<SymmetricActorTrace<G, C>> {
        let actor = &mut self.actors[player.index()];
        let final_context = actor
            .context
            .unwrap_or_else(|| self.identity.context(final_measure.graph_hash));
        let root_context = actor.root_context.unwrap_or(final_context);
        Ok(SymmetricActorTrace {
            root: actor.root,
            final_graph: actor.current,
            root_context,
            final_context,
            steps: std::mem::take(&mut actor.steps),
            root_stats: std::mem::take(&mut actor.root_stats),
            blocked: actor.blocked,
            stopped: actor.stopped,
            final_measure,
        })
    }

    fn actor_active(&self, player: GumbelPlayer) -> bool {
        let actor = &self.actors[player.index()];
        !actor.blocked && !actor.stopped && actor.rewrites < self.config.max_steps
    }

    fn append_releasable(&mut self, mut handles: GumbelHandleBatch<G, C>) {
        self.releasable.graphs.append(&mut handles.graphs);
        self.releasable.candidates.append(&mut handles.candidates);
    }

    fn drain_root(&mut self, all: bool) {
        let handles = match &mut self.state {
            State::Root(task) => Some(if all {
                task.take_all_handles()
            } else {
                task.take_releasable()
            }),
            _ => None,
        };
        if let Some(handles) = handles {
            self.append_releasable(handles);
        }
    }

    fn next_token(&mut self) -> WorkToken {
        let token = WorkToken::new(self.next_token);
        self.next_token += 1;
        token
    }
}

fn valid_measure<G>(measure: &MeasureResult<G>) -> bool {
    measure.measured
        && measure.valid
        && measure
            .scalar_reward
            .is_some_and(|reward| reward.is_finite())
}

fn retokenize<G, C>(work: SearchWork<G, C>, token: WorkToken) -> SearchWork<G, C> {
    match work {
        SearchWork::Expand(mut work) => {
            work.token = token;
            SearchWork::Expand(work)
        }
        SearchWork::Apply(mut work) => {
            work.token = token;
            SearchWork::Apply(work)
        }
        SearchWork::Measure(mut work) => {
            work.token = token;
            SearchWork::Measure(work)
        }
        SearchWork::MeasurePair(mut work) => {
            work.token = token;
            SearchWork::MeasurePair(work)
        }
        SearchWork::Eval(mut work) => {
            work.token = token;
            SearchWork::Eval(work)
        }
    }
}

struct Actor<G, C> {
    root: G,
    current: G,
    context: Option<ReplayGraphContext>,
    root_context: Option<ReplayGraphContext>,
    blocked: bool,
    stopped: bool,
    rewrites: usize,
    steps: Vec<GumbelStep<G, C>>,
    root_stats: Vec<GumbelRootStats>,
}

impl<G: Copy, C> Actor<G, C> {
    fn new(root: G) -> Self {
        Self {
            root,
            current: root,
            context: None,
            root_context: None,
            blocked: false,
            stopped: false,
            rewrites: 0,
            steps: Vec::new(),
            root_stats: Vec::new(),
        }
    }

    fn from_position(
        graph: G,
        context: Option<ReplayGraphContext>,
        rewrites: usize,
        blocked: bool,
        stopped: bool,
    ) -> Self {
        Self {
            root: graph,
            current: graph,
            context,
            root_context: context,
            blocked,
            stopped,
            rewrites,
            steps: Vec::new(),
            root_stats: Vec::new(),
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum State<G, C> {
    Turn,
    Root(SymmetricSelfplayRootTask<G, C>),
    MeasurePair,
    DoneResult(Box<SymmetricEpisode<G, C>>),
    Done,
}

enum Pending {
    MeasurePair { token: WorkToken },
}

impl Pending {
    fn token(&self) -> WorkToken {
        match self {
            Self::MeasurePair { token } => *token,
        }
    }
}

#[derive(Clone, Copy)]
struct RootPending {
    outer: WorkToken,
    inner: WorkToken,
}
