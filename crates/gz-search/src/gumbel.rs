use crate::support::{candidate_info, internal, step_ref};
use crate::work::{
    ApplyWork, EngineIdentity, EvalWork, ExpandResult, ExpandWork, ExpandedCandidate, MeasureWork,
    SearchPoll, SearchWork, SearchWorkResult, WorkToken,
};
use crate::{SearchAction, SearchCandidateSummary, gumbel_search_config_hash};
use gz_engine::{
    ApplyResult, CandidateOptions, EngineResult, GraphEngine, MeasureOptions, MeasureResult,
    ModelVersion, PortableCandidateRef, PortableSearchActionRef, ReplayGraphContext,
    SearchConfigHash, SearchStepRef,
};
use gz_eval::{
    EngineEvalRequest, EngineEvaluator, EvalAction, EvalOpponentContext, EvalPositionContext,
    EvalRequest, eval_error_to_engine_error,
};
use std::collections::HashSet;
use std::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelMctsConfig {
    pub max_steps: usize,
    pub simulations: NonZeroUsize,
    pub max_considered_actions: NonZeroUsize,
    pub seed: u64,
    pub gumbel_scale: f32,
    pub c_visit: f32,
    pub c_scale: f32,
    pub temperature_moves: usize,
    pub tree_reuse: bool,
    pub candidate_options: CandidateOptions,
    pub measure_options: MeasureOptions,
}

pub struct GumbelMcts {
    config: GumbelMctsConfig,
    search_config_hash: SearchConfigHash,
}

impl GumbelMcts {
    #[must_use]
    pub fn new(config: GumbelMctsConfig) -> Self {
        assert!(config.gumbel_scale.is_finite() && config.gumbel_scale >= 0.0);
        assert!(config.c_visit.is_finite() && config.c_visit >= 0.0);
        assert!(config.c_scale.is_finite() && config.c_scale >= 0.0);

        let search_config_hash = gumbel_search_config_hash(
            config.max_steps,
            config.simulations.get(),
            config.max_considered_actions.get(),
            config.seed,
            config.gumbel_scale,
            config.c_visit,
            config.c_scale,
            config.temperature_moves,
            config.tree_reuse,
            config.candidate_options,
            config.measure_options,
        );

        Self {
            config,
            search_config_hash,
        }
    }

    #[must_use]
    pub const fn config(&self) -> GumbelMctsConfig {
        self.config
    }

    #[must_use]
    pub const fn search_config_hash(&self) -> SearchConfigHash {
        self.search_config_hash
    }

    #[must_use]
    pub fn root_budget(&self, step: usize) -> (f32, f32) {
        let budget_step = if self.config.max_steps == 0 {
            0.0
        } else {
            1.0 / self.config.max_steps as f32
        };

        (budget_fraction(self.config.max_steps, step), budget_step)
    }

    pub fn run_from_root<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
    ) -> EngineResult<GumbelEpisode<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>,
    {
        let root = engine.root();
        self.run(engine, evaluator, root, GumbelEpisodeContext::default())
    }

    pub fn run<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
        root: E::Graph,
        context: GumbelEpisodeContext,
    ) -> EngineResult<GumbelEpisode<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>,
    {
        let identity = EngineIdentity::from_engine(engine);
        let mut task = GumbelEpisodeTask::new(self, identity, root, context);

        loop {
            match task.poll()? {
                SearchPoll::Work(work) => {
                    let token = work.token();
                    let result = service_search_work(engine, evaluator, work)?;
                    task.resume(token, result)?;
                }
                SearchPoll::Blocked => return Err(internal("serial driver blocked")),
                SearchPoll::Done(result) => return Ok(result),
            }
        }
    }

    pub fn search_root<E, V>(
        &self,
        engine: &mut E,
        evaluator: &mut V,
        root: E::Graph,
        context: GumbelSearchContext,
    ) -> EngineResult<GumbelRootResult<E::Graph, E::Candidate>>
    where
        E: GraphEngine,
        V: EngineEvaluator<E>,
    {
        let identity = EngineIdentity::from_engine(engine);
        let mut task = GumbelRootTask::new(self, identity, root, context);

        loop {
            match task.poll()? {
                SearchPoll::Work(work) => {
                    let token = work.token();
                    let result = service_search_work(engine, evaluator, work)?;
                    task.resume(token, result)?;
                }
                SearchPoll::Blocked => return Err(internal("serial driver blocked")),
                SearchPoll::Done(result) => return Ok(result),
            }
        }
    }
}

fn service_search_work<E, V>(
    engine: &mut E,
    evaluator: &mut V,
    work: SearchWork<E::Graph, E::Candidate>,
) -> EngineResult<SearchWorkResult<E::Graph, E::Candidate>>
where
    E: GraphEngine,
    V: EngineEvaluator<E>,
{
    match work {
        SearchWork::Expand(work) => service_expand_work(engine, work).map(SearchWorkResult::Expand),
        SearchWork::Apply(work) => engine
            .apply(work.graph, work.candidate)
            .map(SearchWorkResult::Apply),
        SearchWork::Measure(work) => engine
            .measure(work.graph, work.options)
            .map(SearchWorkResult::Measure),
        SearchWork::Eval(work) => {
            let output = evaluator.evaluate(
                engine,
                EngineEvalRequest {
                    graph: work.graph,
                    candidates: &work.candidates,
                    request: &work.request,
                    measure_options: work.measure_options,
                },
            )?;
            Ok(SearchWorkResult::Eval(output))
        }
    }
}

pub(crate) fn service_expand_work<E>(
    engine: &mut E,
    work: ExpandWork<E::Graph>,
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
            candidate_info(engine, work.graph, candidate).map(|info| ExpandedCandidate {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub struct GumbelEpisodeContext {
    pub opponent: Option<GumbelOpponentContext>,
    /// Mixed into the root Gumbel RNG so episodes sharing a root explore
    /// differently. Zero (the default) preserves the historical seeding;
    /// drivers derive it from the episode id.
    pub noise_seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelSearchContext {
    pub root_step: u32,
    pub budget_fraction: f32,
    pub budget_step: f32,
    pub selection_temperature: f32,
    pub opponent: Option<GumbelOpponentContext>,
    pub noise_seed: u64,
}

impl Default for GumbelSearchContext {
    fn default() -> Self {
        Self {
            root_step: 0,
            budget_fraction: 1.0,
            budget_step: 0.0,
            selection_temperature: 0.0,
            opponent: None,
            noise_seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GumbelOpponentContext {
    pub trajectory_id: u64,
    pub row_count: u32,
}

impl From<GumbelOpponentContext> for EvalOpponentContext {
    fn from(context: GumbelOpponentContext) -> Self {
        Self {
            trajectory_id: context.trajectory_id,
            row_count: context.row_count,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GumbelRootResult<G, C> {
    pub root: G,
    pub root_context: ReplayGraphContext,
    pub selected_after: G,
    pub selected_after_context: ReplayGraphContext,
    pub selected_action: SearchAction<C>,
    pub selected_action_ref: PortableSearchActionRef,
    pub selected_candidate: Option<SearchCandidateSummary>,
    pub selected_action_index: usize,
    pub engine_candidate_count: usize,
    pub action_count: usize,
    pub legal_actions: Vec<PortableSearchActionRef>,
    pub considered_action_indices: Vec<usize>,
    pub policy_target: Vec<f32>,
    pub root_value: f32,
    pub root_search_value: f32,
    pub root_q_max: f32,
    pub model_version: ModelVersion,
    pub stats: GumbelRootStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GumbelRootStats {
    pub simulations: usize,
    pub expanded_nodes: usize,
    pub eval_count: usize,
    pub carried_nodes: usize,
    pub carried_root_visits: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GumbelEpisode<G, C> {
    pub root: G,
    pub final_graph: G,
    pub root_context: ReplayGraphContext,
    pub final_context: ReplayGraphContext,
    pub steps: Vec<GumbelStep<G, C>>,
    pub root_stats: Vec<GumbelRootStats>,
    pub created_graphs: Vec<G>,
    pub created_candidates: Vec<C>,
    pub final_measure: MeasureResult<G>,
    pub stop_reason: GumbelStopReason,
    pub search_config_hash: SearchConfigHash,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GumbelStep<G, C> {
    pub before: G,
    pub after: G,
    pub action: SearchAction<C>,
    pub step_ref: SearchStepRef,
    pub selected_action: PortableSearchActionRef,
    pub selected_candidate: Option<SearchCandidateSummary>,
    pub engine_candidate_count: usize,
    pub action_count: usize,
    pub selected_rank: usize,
    pub legal_actions: Vec<PortableSearchActionRef>,
    pub policy_target: Vec<f32>,
    pub considered_action_indices: Vec<usize>,
    pub root_value: f32,
    pub root_search_value: f32,
    pub root_q_max: f32,
    pub model_version: ModelVersion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GumbelStopReason {
    MaxSteps,
    SelectedStop,
}

pub struct GumbelEpisodeTask<G, C> {
    config: GumbelMctsConfig,
    search_config_hash: SearchConfigHash,
    identity: EngineIdentity,
    root: G,
    context: GumbelEpisodeContext,
    current: G,
    current_context: Option<ReplayGraphContext>,
    root_context: Option<ReplayGraphContext>,
    steps: Vec<GumbelStep<G, C>>,
    root_stats: Vec<GumbelRootStats>,
    created_graphs: Vec<G>,
    created_candidates: Vec<C>,
    step_index: usize,
    budget_step: f32,
    next_token: u64,
    pending: Option<PendingEpisodeWork<G, C>>,
    state: EpisodeTaskState<G, C>,
}

impl<G, C> GumbelEpisodeTask<G, C>
where
    G: Copy,
    C: Copy,
{
    pub fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: GumbelEpisodeContext,
    ) -> Self {
        let budget_step = if search.config.max_steps == 0 {
            0.0
        } else {
            1.0 / search.config.max_steps as f32
        };

        Self {
            config: search.config,
            search_config_hash: search.search_config_hash,
            identity,
            root,
            context,
            current: root,
            current_context: None,
            root_context: None,
            steps: Vec::new(),
            root_stats: Vec::new(),
            created_graphs: Vec::new(),
            created_candidates: Vec::new(),
            step_index: 0,
            budget_step,
            next_token: 0,
            pending: None,
            state: EpisodeTaskState::Start,
        }
    }

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, GumbelEpisode<G, C>>> {
        if self.pending.is_some() {
            return Ok(SearchPoll::Blocked);
        }

        loop {
            let state = std::mem::replace(&mut self.state, EpisodeTaskState::Done);
            match state {
                EpisodeTaskState::Start => {
                    if self.config.max_steps == 0 {
                        self.state = EpisodeTaskState::Measure {
                            stop_reason: GumbelStopReason::MaxSteps,
                        };
                    } else {
                        self.state = EpisodeTaskState::Root(self.new_root_task());
                    }
                }
                EpisodeTaskState::Root(mut root_task) => match root_task.poll()? {
                    SearchPoll::Work(work) => {
                        let outer = self.next_token();
                        let inner = work.token();
                        let work = retokenize_work(work, outer);
                        self.pending = Some(PendingEpisodeWork::Root {
                            token: outer,
                            inner,
                            root_task,
                        });
                        return Ok(SearchPoll::Work(work));
                    }
                    SearchPoll::Blocked => {
                        self.state = EpisodeTaskState::Root(root_task);
                        return Ok(SearchPoll::Blocked);
                    }
                    SearchPoll::Done(result) => {
                        let root_context = root_task
                            .root_context()
                            .ok_or_else(|| internal("missing root context"))?;
                        if root_context != result.root_context {
                            return Err(internal("root context mismatch"));
                        }
                        let reused_root = self.reused_root_task(&root_task, &result)?;
                        self.finish_root_result(result, reused_root)?;
                    }
                },
                EpisodeTaskState::Measure { stop_reason } => {
                    let token = self.next_token();
                    self.pending = Some(PendingEpisodeWork::Measure { token, stop_reason });
                    return Ok(SearchPoll::Work(SearchWork::Measure(MeasureWork {
                        token,
                        graph: self.current,
                        options: self.config.measure_options,
                    })));
                }
                EpisodeTaskState::DoneResult(episode) => {
                    self.state = EpisodeTaskState::Done;
                    return Ok(SearchPoll::Done(episode));
                }
                EpisodeTaskState::Done => return Err(internal("poll after done")),
            }
        }
    }

    pub fn resume(&mut self, token: WorkToken, result: SearchWorkResult<G, C>) -> EngineResult<()> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| internal("resume without pending work"))?;

        if pending.token() != token {
            self.pending = Some(pending);
            return Err(internal("unknown work token"));
        }

        match (pending, result) {
            (
                PendingEpisodeWork::Root {
                    inner,
                    mut root_task,
                    ..
                },
                result @ (SearchWorkResult::Expand(_)
                | SearchWorkResult::Apply(_)
                | SearchWorkResult::Eval(_)),
            ) => {
                self.track_created_handles(&result);
                root_task.resume(inner, result)?;
                self.state = EpisodeTaskState::Root(root_task);
                Ok(())
            }
            (
                PendingEpisodeWork::Measure { stop_reason, .. },
                SearchWorkResult::Measure(measure),
            ) => {
                let root_context = self
                    .root_context
                    .unwrap_or_else(|| self.identity.context(measure.graph_hash));
                let final_context = self
                    .current_context
                    .unwrap_or_else(|| self.identity.context(measure.graph_hash));
                self.current_context = Some(final_context);
                self.root_context = Some(root_context);
                self.state = EpisodeTaskState::DoneResult(GumbelEpisode {
                    root: self.root,
                    final_graph: self.current,
                    root_context,
                    final_context,
                    steps: std::mem::take(&mut self.steps),
                    root_stats: std::mem::take(&mut self.root_stats),
                    created_graphs: std::mem::take(&mut self.created_graphs),
                    created_candidates: std::mem::take(&mut self.created_candidates),
                    final_measure: measure,
                    stop_reason,
                    search_config_hash: self.search_config_hash,
                });
                Ok(())
            }
            (pending, _) => {
                self.pending = Some(pending);
                Err(internal("mismatched work result"))
            }
        }
    }

    fn next_token(&mut self) -> WorkToken {
        let token = WorkToken::new(self.next_token);
        self.next_token += 1;
        token
    }

    fn track_created_handles(&mut self, result: &SearchWorkResult<G, C>) {
        match result {
            SearchWorkResult::Expand(expanded) => {
                self.created_candidates.extend(
                    expanded
                        .candidates
                        .iter()
                        .map(|candidate| candidate.candidate),
                );
            }
            SearchWorkResult::Apply(applied) => self.created_graphs.push(applied.after),
            SearchWorkResult::Measure(_) | SearchWorkResult::Eval(_) => {}
        }
    }

    fn new_root_task(&self) -> GumbelRootTask<G, C> {
        self.root_task_at(self.step_index, self.current)
    }

    fn root_task_at(&self, step_index: usize, root: G) -> GumbelRootTask<G, C> {
        let root_step = step_index as u32;
        GumbelRootTask::new(
            &GumbelMcts {
                config: self.config,
                search_config_hash: self.search_config_hash,
            },
            self.identity,
            root,
            GumbelSearchContext {
                root_step,
                budget_fraction: budget_fraction(self.config.max_steps, step_index),
                budget_step: self.budget_step,
                selection_temperature: if step_index < self.config.temperature_moves {
                    1.0
                } else {
                    0.0
                },
                noise_seed: self.context.noise_seed,
                opponent: self.context.opponent,
            },
        )
    }

    fn reused_root_task(
        &self,
        root_task: &GumbelRootTask<G, C>,
        result: &GumbelRootResult<G, C>,
    ) -> EngineResult<Option<GumbelRootTask<G, C>>> {
        let next_step = self.step_index + 1;
        if !self.config.tree_reuse
            || matches!(result.selected_action, SearchAction::Stop)
            || next_step >= self.config.max_steps
        {
            return Ok(None);
        }

        let fresh = self.root_task_at(next_step, result.selected_after);
        root_task.reused_child_task(
            result.selected_action_index,
            result.selected_after,
            result.selected_after_context,
            fresh.context,
        )
    }

    fn finish_root_result(
        &mut self,
        result: GumbelRootResult<G, C>,
        reused_root: Option<GumbelRootTask<G, C>>,
    ) -> EngineResult<()> {
        let before_context = self.current_context.unwrap_or(result.root_context);
        let step_ref = step_ref(
            before_context,
            result.selected_action_ref,
            result.selected_after_context,
        )?;
        let selected_stop = matches!(result.selected_action, SearchAction::Stop);
        let step = GumbelStep {
            before: self.current,
            after: result.selected_after,
            action: result.selected_action,
            step_ref,
            selected_action: result.selected_action_ref,
            selected_candidate: result.selected_candidate,
            engine_candidate_count: result.engine_candidate_count,
            action_count: result.action_count,
            selected_rank: result.selected_action_index,
            legal_actions: result.legal_actions,
            policy_target: result.policy_target,
            considered_action_indices: result.considered_action_indices,
            root_value: result.root_value,
            root_search_value: result.root_search_value,
            root_q_max: result.root_q_max,
            model_version: result.model_version,
        };

        if self.root_context.is_none() {
            self.root_context = Some(result.root_context);
        }
        self.current = result.selected_after;
        self.current_context = Some(result.selected_after_context);
        self.steps.push(step);
        self.root_stats.push(result.stats);
        self.step_index += 1;

        if selected_stop {
            self.state = EpisodeTaskState::Measure {
                stop_reason: GumbelStopReason::SelectedStop,
            };
        } else if self.step_index >= self.config.max_steps {
            self.state = EpisodeTaskState::Measure {
                stop_reason: GumbelStopReason::MaxSteps,
            };
        } else if let Some(root_task) = reused_root {
            self.state = EpisodeTaskState::Root(root_task);
        } else {
            self.state = EpisodeTaskState::Root(self.new_root_task());
        }

        Ok(())
    }
}

#[allow(clippy::large_enum_variant)]
enum EpisodeTaskState<G, C> {
    Start,
    Root(GumbelRootTask<G, C>),
    Measure { stop_reason: GumbelStopReason },
    Done,
    DoneResult(GumbelEpisode<G, C>),
}

#[allow(clippy::large_enum_variant)]
enum PendingEpisodeWork<G, C> {
    Root {
        token: WorkToken,
        inner: WorkToken,
        root_task: GumbelRootTask<G, C>,
    },
    Measure {
        token: WorkToken,
        stop_reason: GumbelStopReason,
    },
}

impl<G, C> PendingEpisodeWork<G, C> {
    const fn token(&self) -> WorkToken {
        match self {
            Self::Root { token, .. } | Self::Measure { token, .. } => *token,
        }
    }
}

fn retokenize_work<G, C>(work: SearchWork<G, C>, token: WorkToken) -> SearchWork<G, C> {
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
        SearchWork::Eval(mut work) => {
            work.token = token;
            SearchWork::Eval(work)
        }
    }
}

pub struct GumbelRootTask<G, C> {
    config: GumbelMctsConfig,
    identity: EngineIdentity,
    root: G,
    context: GumbelSearchContext,
    root_context: Option<ReplayGraphContext>,
    tree: Tree<G, C>,
    next_token: u64,
    pending: Option<PendingRootWork<G, C>>,
    state: RootTaskState<G, C>,
}

impl<G, C> GumbelRootTask<G, C>
where
    G: Copy,
    C: Copy,
{
    pub fn new(
        search: &GumbelMcts,
        identity: EngineIdentity,
        root: G,
        context: GumbelSearchContext,
    ) -> Self {
        Self {
            config: search.config,
            identity,
            root,
            context,
            root_context: None,
            tree: Tree::new(search, context),
            next_token: 0,
            pending: None,
            state: RootTaskState::EmitNodeExpand {
                graph: root,
                expected_context: None,
                depth: 0,
                run: None,
            },
        }
    }

    fn reused_child_task(
        &self,
        action: usize,
        root: G,
        expected_context: ReplayGraphContext,
        context: GumbelSearchContext,
    ) -> EngineResult<Option<Self>> {
        let Some(child_index) = self.tree.nodes[0].children[action] else {
            return Ok(None);
        };
        let (tree, root_context) = self.tree.compact_subtree(child_index, context)?;
        if root_context != expected_context {
            return Err(internal("reused root context mismatch"));
        }

        let mut task = Self {
            config: self.config,
            identity: self.identity,
            root,
            context,
            root_context: Some(root_context),
            tree,
            next_token: 0,
            pending: None,
            state: RootTaskState::Done,
        };
        let run = task.start_run_state();
        task.state = RootTaskState::Running(run);
        Ok(Some(task))
    }

    pub fn poll(&mut self) -> EngineResult<SearchPoll<G, C, GumbelRootResult<G, C>>> {
        if self.pending.is_some() {
            return Ok(SearchPoll::Blocked);
        }

        let state = std::mem::replace(&mut self.state, RootTaskState::Done);
        match state {
            RootTaskState::EmitNodeExpand {
                graph,
                expected_context,
                depth,
                run,
            } => {
                let token = self.next_token();
                self.pending = Some(PendingRootWork::ExpandNode {
                    token,
                    graph,
                    expected_context,
                    depth,
                    run,
                });
                Ok(SearchPoll::Work(SearchWork::Expand(ExpandWork {
                    token,
                    graph,
                    options: self.config.candidate_options,
                })))
            }
            RootTaskState::EmitNodeEval { expansion, run } => self.poll_node_eval(expansion, run),
            RootTaskState::Running(run) => self.poll_running(run),
            RootTaskState::Done => Err(internal("poll after done")),
        }
    }

    pub fn resume(&mut self, token: WorkToken, result: SearchWorkResult<G, C>) -> EngineResult<()> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| internal("resume without pending work"))?;

        if pending.token() != token {
            self.pending = Some(pending);
            return Err(internal("unknown work token"));
        }

        match (pending, result) {
            (
                PendingRootWork::ExpandNode {
                    graph,
                    expected_context,
                    depth,
                    run,
                    ..
                },
                SearchWorkResult::Expand(result),
            ) => self.resume_expand(graph, expected_context, depth, run, result),
            (
                PendingRootWork::EvalNode {
                    expansion,
                    run,
                    request,
                    ..
                },
                SearchWorkResult::Eval(output),
            ) => self.resume_node_eval(expansion, run, &request, output),
            (PendingRootWork::Apply { run, action, .. }, SearchWorkResult::Apply(result)) => {
                self.resume_apply(run, action, result)
            }
            (
                PendingRootWork::StopEval {
                    mut run, request, ..
                },
                SearchWorkResult::Eval(output),
            ) => {
                output
                    .validate_for(&request)
                    .map_err(eval_error_to_engine_error)?;
                self.tree.eval_count += 1;
                let path = run
                    .descent
                    .as_ref()
                    .ok_or_else(|| internal("missing descent"))?
                    .path
                    .clone();
                self.tree.backup(&path, output.value);
                run.complete_simulation();
                self.state = RootTaskState::Running(run);
                Ok(())
            }
            (pending, _) => {
                self.pending = Some(pending);
                Err(internal("mismatched work result"))
            }
        }
    }

    #[must_use]
    pub const fn root_context(&self) -> Option<ReplayGraphContext> {
        self.root_context
    }

    fn next_token(&mut self) -> WorkToken {
        let token = WorkToken::new(self.next_token);
        self.next_token += 1;
        token
    }

    fn poll_node_eval(
        &mut self,
        expansion: NodeExpansion<G, C>,
        run: Option<RunState<G, C>>,
    ) -> EngineResult<SearchPoll<G, C, GumbelRootResult<G, C>>> {
        let request = EvalRequest::with_position(
            expansion.context,
            expansion.eval_actions.clone(),
            self.tree.position(expansion.depth),
        )
        .map_err(|_| internal("invalid gumbel eval request"))?;
        let token = self.next_token();
        let work = EvalWork {
            token,
            graph: expansion.graph,
            candidates: expansion.candidates.clone(),
            request: request.clone(),
            measure_options: self.config.measure_options,
        };

        self.pending = Some(PendingRootWork::EvalNode {
            token,
            expansion,
            run,
            request,
        });
        Ok(SearchPoll::Work(SearchWork::Eval(work)))
    }

    fn poll_running(
        &mut self,
        mut run: RunState<G, C>,
    ) -> EngineResult<SearchPoll<G, C, GumbelRootResult<G, C>>> {
        loop {
            if run.descent.is_none() && !self.start_descent(&mut run) {
                return self.finish_root(run);
            }

            match self.poll_descent(run)? {
                DescentPoll::Continue(next) => run = next,
                DescentPoll::Work(work, pending) => {
                    self.pending = Some(pending);
                    return Ok(SearchPoll::Work(work));
                }
            }
        }
    }

    fn start_descent(&self, run: &mut RunState<G, C>) -> bool {
        if run.schedule_index >= run.schedule.len() {
            return false;
        }

        let root_scores = self.tree.root_scores(0, &run.base_scores);
        let action = loop {
            if run.schedule_index >= run.schedule.len() {
                return false;
            }

            let target_visits = run.schedule[run.schedule_index];
            if let Some(action) = best_eligible(
                &self.tree.nodes[0],
                &run.considered,
                target_visits,
                &root_scores,
                self.config.tree_reuse,
            ) {
                break action;
            }

            if !self.config.tree_reuse {
                return false;
            }

            run.schedule_index += 1;
        };

        run.descent = Some(DescentState {
            node_index: 0,
            depth: 0,
            path: Vec::new(),
            seen: HashSet::from([self.tree.nodes[0].context]),
            forced: Some(action),
        });
        true
    }

    fn poll_descent(&mut self, mut run: RunState<G, C>) -> EngineResult<DescentPoll<G, C>> {
        let mut descent = run
            .descent
            .take()
            .ok_or_else(|| internal("missing descent"))?;

        let action = match descent.forced.take() {
            Some(action) => action,
            None => self.tree.select_nonroot(descent.node_index),
        };

        if self.tree.nodes[descent.node_index].is_stop(action) {
            descent.path.push(Edge {
                node_index: descent.node_index,
                action,
            });

            if let Some(request) = self.stop_eval_request(descent.node_index, descent.depth)? {
                run.descent = Some(descent);
                let token = self.next_token();
                let graph = self.tree.nodes[run.descent.as_ref().unwrap().node_index].graph;
                let candidates = self.tree.nodes[run.descent.as_ref().unwrap().node_index]
                    .candidates
                    .clone();
                let work = EvalWork {
                    token,
                    graph,
                    candidates,
                    request: request.clone(),
                    measure_options: self.config.measure_options,
                };
                let pending = PendingRootWork::StopEval {
                    token,
                    run,
                    request,
                };
                return Ok(DescentPoll::Work(SearchWork::Eval(work), pending));
            }

            let value = self.tree.nodes[descent.node_index].value;
            self.tree.backup(&descent.path, value);
            run.descent = Some(descent);
            run.complete_simulation();
            return Ok(DescentPoll::Continue(run));
        }

        let graph = self.tree.nodes[descent.node_index].graph;
        let candidate = self.tree.nodes[descent.node_index].candidates[action];
        run.descent = Some(descent);
        let token = self.next_token();
        let work = ApplyWork {
            token,
            graph,
            candidate,
        };
        let pending = PendingRootWork::Apply { token, run, action };
        Ok(DescentPoll::Work(SearchWork::Apply(work), pending))
    }

    fn resume_expand(
        &mut self,
        graph: G,
        expected_context: Option<ReplayGraphContext>,
        depth: usize,
        run: Option<RunState<G, C>>,
        result: ExpandResult<C>,
    ) -> EngineResult<()> {
        let context = self.identity.context(result.graph_hash);
        if let Some(expected) = expected_context
            && expected != context
        {
            return Err(internal("expand graph hash mismatch"));
        }

        if run.is_none() {
            self.root_context = Some(context);
        }

        let mut candidates = Vec::with_capacity(result.candidates.len());
        let mut eval_actions = Vec::with_capacity(result.candidates.len() + 1);
        let mut action_refs = Vec::with_capacity(result.candidates.len() + 1);
        let mut summaries = Vec::with_capacity(result.candidates.len() + 1);

        for expanded in result.candidates {
            candidates.push(expanded.candidate);
            let candidate_ref = PortableCandidateRef::new(context, expanded.candidate_hash);
            let action_ref = PortableSearchActionRef::candidate(candidate_ref);
            eval_actions.push(EvalAction::candidate(
                candidate_ref,
                expanded.kind,
                expanded.tags,
                expanded.static_prior,
            ));
            action_refs.push(action_ref);
            summaries.push(Some(SearchCandidateSummary {
                kind: expanded.kind,
                tags: expanded.tags,
                static_prior: expanded.static_prior,
            }));
        }

        eval_actions.push(EvalAction::stop(context));
        action_refs.push(PortableSearchActionRef::stop(context));
        summaries.push(None);

        self.state = RootTaskState::EmitNodeEval {
            expansion: NodeExpansion {
                graph,
                context,
                depth,
                candidates,
                eval_actions,
                action_refs,
                summaries,
            },
            run,
        };
        Ok(())
    }

    fn resume_node_eval(
        &mut self,
        expansion: NodeExpansion<G, C>,
        mut run: Option<RunState<G, C>>,
        request: &EvalRequest,
        output: gz_eval::EvalOutput,
    ) -> EngineResult<()> {
        output
            .validate_for(request)
            .map_err(eval_error_to_engine_error)?;
        let node_index = self.finalize_node(expansion, output);

        if let Some(mut run) = run.take() {
            let value = self.tree.nodes[node_index].value;
            let descent = run
                .descent
                .as_ref()
                .ok_or_else(|| internal("missing descent"))?;
            let edge = *descent
                .path
                .last()
                .ok_or_else(|| internal("missing edge"))?;
            self.tree.nodes[edge.node_index].children[edge.action] = Some(node_index);
            self.tree.backup(&descent.path, value);
            run.complete_simulation();
            self.state = RootTaskState::Running(run);
        } else {
            self.state = RootTaskState::Running(self.start_run_state());
        }

        Ok(())
    }

    fn resume_apply(
        &mut self,
        mut run: RunState<G, C>,
        action: usize,
        applied: ApplyResult<G, C>,
    ) -> EngineResult<()> {
        let mut descent = run
            .descent
            .take()
            .ok_or_else(|| internal("missing descent"))?;

        if applied.rejected.is_some() {
            self.tree.mask_action(descent.node_index, action);
            run.descent = Some(descent);
            self.state = RootTaskState::Running(run);
            return Ok(());
        }

        descent.path.push(Edge {
            node_index: descent.node_index,
            action,
        });

        let child_context = self.identity.context(applied.after_hash);

        if let Some(child) = self.tree.nodes[descent.node_index].children[action] {
            if !descent.seen.insert(self.tree.nodes[child].context) {
                let value = self.tree.nodes[child].value;
                self.tree.backup(&descent.path, value);
                run.descent = Some(descent);
                run.complete_simulation();
                self.state = RootTaskState::Running(run);
                return Ok(());
            }

            descent.node_index = child;
            descent.depth += 1;
            run.descent = Some(descent);
            self.state = RootTaskState::Running(run);
            return Ok(());
        }

        let depth = descent.depth + 1;
        run.descent = Some(descent);
        self.state = RootTaskState::EmitNodeExpand {
            graph: applied.after,
            expected_context: Some(child_context),
            depth,
            run: Some(run),
        };
        Ok(())
    }

    fn stop_eval_request(
        &self,
        node_index: usize,
        depth: usize,
    ) -> EngineResult<Option<EvalRequest>> {
        let Some(opponent) = self.context.opponent else {
            return Ok(None);
        };
        let Some(last) = opponent.row_count.checked_sub(1) else {
            return Ok(None);
        };
        let needed_depth = last.saturating_sub(self.context.root_step) as usize;
        let effective_depth = depth.max(needed_depth);

        if effective_depth == depth {
            return Ok(None);
        }

        let node = &self.tree.nodes[node_index];
        EvalRequest::with_position(
            node.context,
            node.eval_actions.clone(),
            self.tree.position(effective_depth),
        )
        .map(Some)
        .map_err(|_| internal("invalid gumbel stop eval request"))
    }

    fn finalize_node(
        &mut self,
        expansion: NodeExpansion<G, C>,
        output: gz_eval::EvalOutput,
    ) -> usize {
        let priors = softmax(&output.policy_logits);
        let action_count = output.policy_logits.len();
        let node = Node {
            graph: expansion.graph,
            context: expansion.context,
            candidates: expansion.candidates,
            eval_actions: expansion.eval_actions,
            action_refs: expansion.action_refs,
            summaries: expansion.summaries,
            logits: output.policy_logits,
            priors,
            value: output.value,
            model_version: output.model_version,
            children: vec![None; action_count],
            visits: vec![0; action_count],
            value_sum: vec![0.0; action_count],
            q: vec![0.0; action_count],
        };

        self.tree.eval_count += 1;
        self.tree.nodes.push(node);
        self.tree.nodes.len() - 1
    }

    /// Carried visits are information the previous step's search already
    /// paid for, so a reused root spends its budget only on what is new.
    /// A floor of a quarter budget keeps fresh exploration under the new
    /// Gumbel draw even when the carried subtree exceeds the budget.
    fn fresh_simulation_budget(&self) -> usize {
        let simulations = self.config.simulations.get();
        let carried = self.tree.carried_root_visits as usize;
        if carried == 0 {
            return simulations;
        }

        simulations
            .saturating_sub(carried)
            .max((simulations / 4).max(1))
    }

    fn start_run_state(&self) -> RunState<G, C> {
        let root_index = 0;
        let action_count = self.tree.nodes[root_index].action_count();
        let mut rng = GumbelRng::new(root_seed(
            self.config.seed ^ self.context.noise_seed,
            self.context.root_step,
        ));
        let root_gumbels = sample_root_gumbels(action_count, self.config.gumbel_scale, &mut rng);
        let mut base_scores = Vec::with_capacity(action_count);

        for (index, logit) in self.tree.nodes[root_index]
            .logits
            .iter()
            .copied()
            .enumerate()
        {
            base_scores.push(logit + root_gumbels[index]);
        }

        let considered = considered_actions(&base_scores, self.config.max_considered_actions.get());
        let schedule = considered_visit_sequence(considered.len(), self.fresh_simulation_budget());

        RunState {
            base_scores,
            considered,
            schedule,
            schedule_index: 0,
            simulations: 0,
            rng,
            descent: None,
            _marker: std::marker::PhantomData,
        }
    }

    fn finish_root(
        &mut self,
        mut run: RunState<G, C>,
    ) -> EngineResult<SearchPoll<G, C, GumbelRootResult<G, C>>> {
        let root_index = 0;
        let root_node = &self.tree.nodes[root_index];
        let root_scores = self.tree.root_scores(root_index, &run.base_scores);
        let policy_target = self.tree.improved_policy(root_index);
        let selectable = selectable_root_actions(root_node, &run.considered);
        let fallback = best_count_action(root_node, &selectable, &root_scores);
        let selected = if self.context.selection_temperature > 0.0 {
            sample_count_action(
                &mut run.rng,
                &root_node.visits,
                self.context.selection_temperature,
                fallback,
            )
        } else {
            fallback
        };
        let selected_after = self.tree.selected_after(root_index, selected)?;
        let selected_after_context = self.tree.selected_after_context(root_index, selected)?;
        let selected_action = root_node.search_action(selected)?;
        let root_context = self
            .root_context
            .ok_or_else(|| internal("missing root context"))?;

        self.state = RootTaskState::Done;
        Ok(SearchPoll::Done(GumbelRootResult {
            root: self.root,
            root_context,
            selected_after,
            selected_after_context,
            selected_action,
            selected_action_ref: root_node.action_refs[selected],
            selected_candidate: root_node.summaries[selected],
            selected_action_index: selected,
            engine_candidate_count: root_node.candidates.len(),
            action_count: root_node.action_count(),
            legal_actions: root_node.action_refs.clone(),
            considered_action_indices: run.considered,
            policy_target,
            root_value: root_node.value,
            root_search_value: search_value(root_node),
            root_q_max: root_q_max(root_node),
            model_version: root_node.model_version,
            stats: GumbelRootStats {
                simulations: run.simulations,
                expanded_nodes: self.tree.nodes.len(),
                eval_count: self.tree.eval_count,
                carried_nodes: self.tree.carried_nodes,
                carried_root_visits: self.tree.carried_root_visits,
            },
        }))
    }
}

enum RootTaskState<G, C> {
    EmitNodeExpand {
        graph: G,
        expected_context: Option<ReplayGraphContext>,
        depth: usize,
        run: Option<RunState<G, C>>,
    },
    EmitNodeEval {
        expansion: NodeExpansion<G, C>,
        run: Option<RunState<G, C>>,
    },
    Running(RunState<G, C>),
    Done,
}

enum PendingRootWork<G, C> {
    ExpandNode {
        token: WorkToken,
        graph: G,
        expected_context: Option<ReplayGraphContext>,
        depth: usize,
        run: Option<RunState<G, C>>,
    },
    EvalNode {
        token: WorkToken,
        expansion: NodeExpansion<G, C>,
        run: Option<RunState<G, C>>,
        request: EvalRequest,
    },
    Apply {
        token: WorkToken,
        run: RunState<G, C>,
        action: usize,
    },
    StopEval {
        token: WorkToken,
        run: RunState<G, C>,
        request: EvalRequest,
    },
}

impl<G, C> PendingRootWork<G, C> {
    const fn token(&self) -> WorkToken {
        match self {
            Self::ExpandNode { token, .. }
            | Self::EvalNode { token, .. }
            | Self::Apply { token, .. }
            | Self::StopEval { token, .. } => *token,
        }
    }
}

struct NodeExpansion<G, C> {
    graph: G,
    context: ReplayGraphContext,
    depth: usize,
    candidates: Vec<C>,
    eval_actions: Vec<EvalAction>,
    action_refs: Vec<PortableSearchActionRef>,
    summaries: Vec<Option<SearchCandidateSummary>>,
}

struct RunState<G, C> {
    base_scores: Vec<f32>,
    considered: Vec<usize>,
    schedule: Vec<u32>,
    schedule_index: usize,
    simulations: usize,
    rng: GumbelRng,
    descent: Option<DescentState>,
    _marker: std::marker::PhantomData<(G, C)>,
}

impl<G, C> RunState<G, C> {
    fn complete_simulation(&mut self) {
        self.simulations += 1;
        self.schedule_index += 1;
        self.descent = None;
    }
}

struct DescentState {
    node_index: usize,
    depth: usize,
    path: Vec<Edge>,
    seen: HashSet<ReplayGraphContext>,
    forced: Option<usize>,
}

#[allow(clippy::large_enum_variant)]
enum DescentPoll<G, C> {
    Continue(RunState<G, C>),
    Work(SearchWork<G, C>, PendingRootWork<G, C>),
}

struct Tree<G, C> {
    config: GumbelMctsConfig,
    context: GumbelSearchContext,
    nodes: Vec<Node<G, C>>,
    eval_count: usize,
    carried_nodes: usize,
    carried_root_visits: u32,
}

impl<G, C> Tree<G, C>
where
    G: Copy,
    C: Copy,
{
    fn new(search: &GumbelMcts, context: GumbelSearchContext) -> Self {
        Self {
            config: search.config,
            context,
            nodes: Vec::new(),
            eval_count: 0,
            carried_nodes: 0,
            carried_root_visits: 0,
        }
    }

    fn compact_subtree(
        &self,
        root_index: usize,
        context: GumbelSearchContext,
    ) -> EngineResult<(Self, ReplayGraphContext)> {
        let mut remap = vec![None; self.nodes.len()];
        let mut old_indices = Vec::new();
        let mut stack = vec![root_index];

        while let Some(index) = stack.pop() {
            if remap[index].is_some() {
                continue;
            }

            remap[index] = Some(old_indices.len());
            old_indices.push(index);

            for child in self.nodes[index].children.iter().rev().flatten() {
                stack.push(*child);
            }
        }

        let mut nodes = Vec::with_capacity(old_indices.len());
        for &old_index in &old_indices {
            let mut node = self.nodes[old_index].clone();
            for child in &mut node.children {
                if let Some(old_child) = *child {
                    *child = remap[old_child];
                }
            }
            nodes.push(node);
        }

        let root_context = nodes
            .first()
            .map(|node| node.context)
            .ok_or_else(|| internal("empty reused subtree"))?;
        let carried_root_visits = nodes[0].visits.iter().sum();
        let carried_nodes = nodes.len();

        Ok((
            Self {
                config: self.config,
                context,
                nodes,
                eval_count: 0,
                carried_nodes,
                carried_root_visits,
            },
            root_context,
        ))
    }

    fn backup(&mut self, path: &[Edge], value: f32) {
        for edge in path {
            let node = &mut self.nodes[edge.node_index];
            node.visits[edge.action] += 1;
            node.value_sum[edge.action] += value;
            node.q[edge.action] = node.value_sum[edge.action] / node.visits[edge.action] as f32;
        }
    }

    fn selected_after(&self, node_index: usize, action: usize) -> EngineResult<G> {
        let node = &self.nodes[node_index];
        if node.is_stop(action) {
            return Ok(node.graph);
        }

        let child = node.children[action].ok_or_else(|| internal("missing selected child"))?;
        Ok(self.nodes[child].graph)
    }

    fn selected_after_context(
        &self,
        node_index: usize,
        action: usize,
    ) -> EngineResult<ReplayGraphContext> {
        let node = &self.nodes[node_index];
        if node.is_stop(action) {
            return Ok(node.context);
        }

        let child = node.children[action].ok_or_else(|| internal("missing selected child"))?;
        Ok(self.nodes[child].context)
    }

    fn root_scores(&self, node_index: usize, base_scores: &[f32]) -> Vec<f32> {
        let node = &self.nodes[node_index];
        let completed_q = completed_q(node);
        let max_visits = node.visits.iter().copied().max().unwrap_or(0) as f32;
        let scale = (self.config.c_visit + max_visits) * self.config.c_scale;

        base_scores
            .iter()
            .zip(&node.logits)
            .zip(completed_q)
            .map(|((score, logit), q)| {
                if logit.is_finite() {
                    score + scale * q
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect()
    }

    fn improved_policy(&self, node_index: usize) -> Vec<f32> {
        let node = &self.nodes[node_index];
        let completed_q = completed_q(node);
        let max_visits = node.visits.iter().copied().max().unwrap_or(0) as f32;
        let scale = (self.config.c_visit + max_visits) * self.config.c_scale;
        let scores = node
            .logits
            .iter()
            .zip(completed_q)
            .map(|(logit, q)| logit + scale * q)
            .collect::<Vec<_>>();
        softmax(&scores)
    }

    fn select_nonroot(&self, node_index: usize) -> usize {
        let node = &self.nodes[node_index];
        let policy = self.improved_policy(node_index);
        let total_visits = node.visits.iter().copied().sum::<u32>() as f32;
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (index, policy) in policy.iter().copied().enumerate() {
            let score = policy - node.visits[index] as f32 / (1.0 + total_visits);
            if score > best_score {
                best = index;
                best_score = score;
            }
        }

        best
    }

    fn mask_action(&mut self, node_index: usize, action: usize) {
        let node = &mut self.nodes[node_index];
        node.logits[action] = f32::NEG_INFINITY;
        node.priors[action] = 0.0;
    }

    fn position(&self, leaf_depth: usize) -> EvalPositionContext {
        EvalPositionContext {
            root_step: self.context.root_step,
            leaf_depth: leaf_depth as u32,
            budget_fraction: self.context.budget_fraction,
            budget_step: self.context.budget_step,
            opponent: self.context.opponent.map(Into::into),
        }
    }
}

#[derive(Clone, Debug)]
struct Node<G, C> {
    graph: G,
    context: ReplayGraphContext,
    candidates: Vec<C>,
    eval_actions: Vec<EvalAction>,
    action_refs: Vec<PortableSearchActionRef>,
    summaries: Vec<Option<SearchCandidateSummary>>,
    logits: Vec<f32>,
    priors: Vec<f32>,
    value: f32,
    model_version: ModelVersion,
    children: Vec<Option<usize>>,
    visits: Vec<u32>,
    value_sum: Vec<f32>,
    q: Vec<f32>,
}

impl<G, C> Node<G, C>
where
    C: Copy,
{
    fn action_count(&self) -> usize {
        self.logits.len()
    }

    fn is_stop(&self, action: usize) -> bool {
        action == self.candidates.len()
    }

    fn search_action(&self, action: usize) -> EngineResult<SearchAction<C>> {
        if self.is_stop(action) {
            Ok(SearchAction::Stop)
        } else {
            self.candidates
                .get(action)
                .copied()
                .map(SearchAction::Candidate)
                .ok_or_else(|| internal("invalid selected action"))
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    node_index: usize,
    action: usize,
}

pub fn considered_visit_sequence(max_considered: usize, simulations: usize) -> Vec<u32> {
    if max_considered <= 1 {
        return (0..simulations as u32).collect();
    }

    let log2max = (max_considered as f64).log2().ceil() as usize;
    let mut sequence = Vec::with_capacity(simulations);
    let mut visits = vec![0_u32; max_considered];
    let mut considered = max_considered;

    while sequence.len() < simulations {
        let extra = (simulations / (log2max * considered)).max(1);
        for _ in 0..extra {
            sequence.extend_from_slice(&visits[..considered]);
            for visit in &mut visits[..considered] {
                *visit += 1;
            }
        }
        considered = (considered / 2).max(2);
    }

    sequence.truncate(simulations);
    sequence
}

fn considered_actions(base_scores: &[f32], max_considered: usize) -> Vec<usize> {
    let mut actions = (0..base_scores.len()).collect::<Vec<_>>();
    actions.sort_by(|&left, &right| {
        base_scores[right]
            .total_cmp(&base_scores[left])
            .then_with(|| left.cmp(&right))
    });
    actions.truncate(max_considered.min(actions.len()));
    actions
}

fn best_eligible<G, C>(
    node: &Node<G, C>,
    considered: &[usize],
    target_visits: u32,
    scores: &[f32],
    tree_reuse: bool,
) -> Option<usize> {
    considered
        .iter()
        .copied()
        .filter(|&action| {
            node.logits[action].is_finite()
                && if tree_reuse {
                    node.visits[action] <= target_visits
                } else {
                    node.visits[action] == target_visits
                }
        })
        .max_by(|&left, &right| {
            scores[left]
                .total_cmp(&scores[right])
                .then_with(|| right.cmp(&left))
        })
}

fn selectable_root_actions<G, C>(node: &Node<G, C>, considered: &[usize]) -> Vec<usize> {
    let mut actions = considered
        .iter()
        .copied()
        .filter(|&action| node.logits[action].is_finite())
        .collect::<Vec<_>>();

    if actions.is_empty() {
        actions.extend(
            node.logits
                .iter()
                .enumerate()
                .filter_map(|(action, logit)| logit.is_finite().then_some(action)),
        );
    }

    actions
}

fn best_count_action<G, C>(node: &Node<G, C>, considered: &[usize], scores: &[f32]) -> usize {
    considered
        .iter()
        .copied()
        .max_by(|&left, &right| {
            node.visits[left]
                .cmp(&node.visits[right])
                .then_with(|| scores[left].total_cmp(&scores[right]))
                .then_with(|| right.cmp(&left))
        })
        .expect("considered actions is non-empty")
}

fn completed_q<G, C>(node: &Node<G, C>) -> Vec<f32> {
    let mixed = mixed_value(node);
    node.visits
        .iter()
        .zip(&node.q)
        .map(|(visits, q)| if *visits > 0 { *q } else { mixed })
        .collect()
}

fn mixed_value<G, C>(node: &Node<G, C>) -> f32 {
    let visits = node.visits.iter().copied().sum::<u32>();
    if visits == 0 {
        return node.value;
    }

    let mut prior_mass = 0.0;
    let mut weighted = 0.0;

    for ((visits, prior), q) in node.visits.iter().zip(&node.priors).zip(&node.q) {
        if *visits == 0 {
            continue;
        }
        prior_mass += prior;
        weighted += prior * q;
    }

    if prior_mass <= 0.0 {
        return node.value;
    }

    (node.value + visits as f32 * weighted / prior_mass) / (1.0 + visits as f32)
}

fn search_value<G, C>(node: &Node<G, C>) -> f32 {
    let mut visits = 0;
    let mut value = 0.0;

    for (count, q) in node.visits.iter().zip(&node.q) {
        if *count == 0 {
            continue;
        }
        visits += *count;
        value += *count as f32 * *q;
    }

    if visits == 0 {
        node.value
    } else {
        value / visits as f32
    }
}

fn root_q_max<G, C>(node: &Node<G, C>) -> f32 {
    node.visits
        .iter()
        .zip(&node.q)
        .filter_map(|(visits, q)| (*visits > 0).then_some(*q))
        .reduce(f32::max)
        .unwrap_or(node.value)
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .reduce(f32::max);
    let Some(max) = max else {
        return vec![1.0 / values.len() as f32; values.len()];
    };

    let mut out = Vec::with_capacity(values.len());
    let mut total = 0.0;

    for value in values {
        let next = if value.is_finite() {
            (*value - max).exp()
        } else {
            0.0
        };
        total += next;
        out.push(next);
    }

    if total <= 0.0 || !total.is_finite() {
        let legal = values.iter().filter(|value| value.is_finite()).count();
        let uniform = 1.0 / legal.max(1) as f32;
        for (out, value) in out.iter_mut().zip(values) {
            *out = if value.is_finite() { uniform } else { 0.0 };
        }
        return out;
    }

    for value in &mut out {
        *value /= total;
    }

    out
}

fn sample_root_gumbels(count: usize, scale: f32, rng: &mut GumbelRng) -> Vec<f32> {
    if scale == 0.0 {
        return vec![0.0; count];
    }

    (0..count)
        .map(|_| scale * -(-rng.unit().ln()).ln())
        .collect()
}

fn sample_count_action(
    rng: &mut GumbelRng,
    visits: &[u32],
    temperature: f32,
    fallback: usize,
) -> usize {
    if temperature <= 0.0 {
        return fallback;
    }

    let inv_temp = 1.0 / temperature;
    let mut total = 0.0;
    let mut weights = Vec::with_capacity(visits.len());

    for visits in visits {
        let weight = if *visits == 0 {
            0.0
        } else {
            (*visits as f32).powf(inv_temp)
        };
        total += weight;
        weights.push(weight);
    }

    if total <= 0.0 || !total.is_finite() {
        return fallback;
    }

    let mut threshold = rng.unit() * total;
    for (index, weight) in weights.into_iter().enumerate() {
        if threshold <= weight {
            return index;
        }
        threshold -= weight;
    }

    fallback
}

fn budget_fraction(max_steps: usize, step: usize) -> f32 {
    if max_steps == 0 {
        1.0
    } else {
        max_steps.saturating_sub(step) as f32 / max_steps as f32
    }
}

fn root_seed(seed: u64, root_step: u32) -> u64 {
    seed ^ 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(u64::from(root_step) + 1)
}

struct GumbelRng {
    state: u64,
}

impl GumbelRng {
    const STEP: u64 = 0x9e37_79b9_7f4a_7c15;

    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn unit(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        let unit = (value as f32 + 0.5) / (1_u32 << 24) as f32;
        unit.clamp(1.0e-7, 1.0 - 1.0e-7)
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::STEP);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}
