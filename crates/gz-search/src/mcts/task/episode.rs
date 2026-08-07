use super::super::math::budget_fraction;
use super::super::strategy::{MctsStrategy, MctsStrategyState};
use super::super::types::{
    MctsConfig, MctsEpisode, MctsEpisodeContext, MctsHandleBatch, MctsRootResult, MctsRootStats,
    MctsSearchContext, MctsStep, MctsStopReason,
};
use super::root::{MctsRootTask, ReusedMctsRootTask};
use crate::SearchAction;
use crate::support::{internal, step_ref};
use crate::work::{
    EngineIdentity, MeasureWork, SearchPoll, SearchWork, SearchWorkResult, WorkToken,
};
use gz_engine::{EngineResult, ReplayGraphContext, SearchConfigHash};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

pub(crate) struct MctsEpisodeTask<G, C, S>
where
    S: MctsStrategyState,
{
    config: MctsConfig,
    strategy: S,
    search_config_hash: SearchConfigHash,
    identity: EngineIdentity,
    root: G,
    context: MctsEpisodeContext,
    current: G,
    current_context: Option<ReplayGraphContext>,
    root_context: Option<ReplayGraphContext>,
    steps: Vec<MctsStep<G, C>>,
    root_stats: Vec<MctsRootStats>,
    visited: HashSet<ReplayGraphContext>,
    path_graphs: Vec<G>,
    move_graphs: Vec<G>,
    move_candidates: Vec<C>,
    releasable: MctsHandleBatch<G, C>,
    step_index: usize,
    budget_step: f32,
    next_token: u64,
    pending: Option<PendingEpisodeWork<G, C, S>>,
    state: EpisodeTaskState<G, C, S>,
}

impl<G, C, S> MctsEpisodeTask<G, C, S>
where
    G: Copy + Eq + Hash,
    C: Copy + Eq + Hash,
    S: MctsStrategy<G, C>,
{
    pub(crate) fn new(
        config: MctsConfig,
        strategy: S,
        search_config_hash: SearchConfigHash,
        identity: EngineIdentity,
        root: G,
        context: MctsEpisodeContext,
    ) -> Self {
        let budget_step = if config.max_steps == 0 {
            0.0
        } else {
            1.0 / config.max_steps as f32
        };
        Self {
            config,
            strategy,
            search_config_hash,
            identity,
            root,
            context,
            current: root,
            current_context: None,
            root_context: None,
            steps: Vec::new(),
            root_stats: Vec::new(),
            visited: HashSet::new(),
            path_graphs: Vec::new(),
            move_graphs: Vec::new(),
            move_candidates: Vec::new(),
            releasable: MctsHandleBatch::default(),
            step_index: 0,
            budget_step,
            next_token: 0,
            pending: None,
            state: EpisodeTaskState::Start,
        }
    }

    pub(crate) fn poll(&mut self) -> EngineResult<SearchPoll<G, C, MctsEpisode<G, C>>> {
        if self.pending.is_some() {
            return Ok(SearchPoll::Blocked);
        }
        loop {
            let state = std::mem::replace(&mut self.state, EpisodeTaskState::Done);
            match state {
                EpisodeTaskState::Start => {
                    if self.config.max_steps == 0 {
                        self.state = EpisodeTaskState::Measure {
                            stop_reason: MctsStopReason::MaxSteps,
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

    pub(crate) fn resume(
        &mut self,
        token: WorkToken,
        result: SearchWorkResult<G, C>,
    ) -> EngineResult<()> {
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
                self.state = EpisodeTaskState::DoneResult(MctsEpisode {
                    root: self.root,
                    final_graph: self.current,
                    root_context,
                    final_context,
                    steps: std::mem::take(&mut self.steps),
                    root_stats: std::mem::take(&mut self.root_stats),
                    created_graphs: self.take_final_graphs(),
                    created_candidates: self.take_final_candidates(),
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

    pub(crate) fn step_index(&self) -> usize {
        self.step_index
    }

    pub(crate) fn take_releasable(&mut self) -> MctsHandleBatch<G, C> {
        std::mem::take(&mut self.releasable)
    }

    pub(crate) fn track_owned_root(&mut self) {
        self.path_graphs.push(self.root);
    }

    pub(crate) fn take_all_handles(&mut self) -> MctsHandleBatch<G, C> {
        let mut handles = self.take_releasable();
        handles.graphs.append(&mut self.path_graphs);
        handles.graphs.append(&mut self.move_graphs);
        handles.candidates.append(&mut self.move_candidates);
        handles
    }

    fn next_token(&mut self) -> WorkToken {
        let token = WorkToken::new(self.next_token);
        self.next_token += 1;
        token
    }

    fn track_created_handles(&mut self, result: &SearchWorkResult<G, C>) {
        match result {
            SearchWorkResult::Expand(expanded) => self.move_candidates.extend(
                expanded
                    .candidates
                    .iter()
                    .map(|candidate| candidate.candidate),
            ),
            SearchWorkResult::Apply(applied) => self.move_graphs.push(applied.after),
            SearchWorkResult::Measure(_)
            | SearchWorkResult::MeasurePair(_)
            | SearchWorkResult::Eval(_) => {}
        }
    }

    fn new_root_task(&self) -> MctsRootTask<G, C, S> {
        self.root_task_at(self.step_index, self.current)
    }

    fn root_task_at(&self, step_index: usize, root: G) -> MctsRootTask<G, C, S> {
        let mut task = MctsRootTask::new(
            self.config,
            self.strategy,
            self.identity,
            root,
            MctsSearchContext {
                root_step: step_index as u32,
                budget_fraction: budget_fraction(self.config.max_steps, step_index),
                budget_step: self.budget_step,
                selection_temperature: if step_index < self.config.temperature_moves {
                    1.0
                } else {
                    0.0
                },
                noise_seed: self.context.noise_seed,
                export_position: self.config.export_position,
            },
        );
        if self.config.no_backtrack {
            task.set_visited(self.visited.clone());
        }
        task
    }

    fn reused_root_task(
        &self,
        root_task: &MctsRootTask<G, C, S>,
        result: &MctsRootResult<G, C>,
    ) -> EngineResult<Option<ReusedMctsRootTask<G, C, S>>> {
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
        result: MctsRootResult<G, C>,
        reused_root: Option<ReusedMctsRootTask<G, C, S>>,
    ) -> EngineResult<()> {
        let before_context = self.current_context.unwrap_or(result.root_context);
        let step_ref = step_ref(
            before_context,
            result.selected_action_ref,
            result.selected_after_context,
        )?;
        let selected_stop = matches!(result.selected_action, SearchAction::Stop);
        let selected_graph = (!selected_stop).then_some(result.selected_after);
        let step = MctsStep {
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
        if self.config.no_backtrack {
            self.visited.insert(result.root_context);
        }

        let (reused_root, carried) = match reused_root {
            Some(reused) => (Some(reused.task), reused.handles),
            None => (None, MctsHandleBatch::default()),
        };
        let reused_root = reused_root.map(|mut task| {
            if self.config.no_backtrack {
                task.set_visited(self.visited.clone());
            }
            task
        });
        self.partition_move_handles(&carried, selected_graph);

        if selected_stop {
            self.state = EpisodeTaskState::Measure {
                stop_reason: MctsStopReason::SelectedStop,
            };
        } else if self.step_index >= self.config.max_steps {
            self.state = EpisodeTaskState::Measure {
                stop_reason: MctsStopReason::MaxSteps,
            };
        } else if let Some(root_task) = reused_root {
            self.state = EpisodeTaskState::Root(root_task);
        } else {
            self.state = EpisodeTaskState::Root(self.new_root_task());
        }
        Ok(())
    }

    fn partition_move_handles(
        &mut self,
        carried: &MctsHandleBatch<G, C>,
        selected_graph: Option<G>,
    ) {
        let mut selected_graphs = HashMap::new();
        if let Some(graph) = selected_graph {
            selected_graphs.insert(graph, 1);
        }
        let mut carried_graphs = handle_counts(&carried.graphs);
        let mut next_graphs = Vec::new();
        for graph in self.move_graphs.drain(..) {
            if decrement_count(&mut selected_graphs, graph) {
                self.path_graphs.push(graph);
            } else if decrement_count(&mut carried_graphs, graph) {
                next_graphs.push(graph);
            } else {
                self.releasable.graphs.push(graph);
            }
        }
        self.move_graphs = next_graphs;

        let mut carried_candidates = handle_counts(&carried.candidates);
        let mut next_candidates = Vec::new();
        for candidate in self.move_candidates.drain(..) {
            if decrement_count(&mut carried_candidates, candidate) {
                next_candidates.push(candidate);
            } else {
                self.releasable.candidates.push(candidate);
            }
        }
        self.move_candidates = next_candidates;
    }

    fn take_final_graphs(&mut self) -> Vec<G> {
        let mut graphs = std::mem::take(&mut self.releasable.graphs);
        graphs.append(&mut self.path_graphs);
        graphs.append(&mut self.move_graphs);
        graphs
    }

    fn take_final_candidates(&mut self) -> Vec<C> {
        let mut candidates = std::mem::take(&mut self.releasable.candidates);
        candidates.append(&mut self.move_candidates);
        candidates
    }
}

fn handle_counts<T>(handles: &[T]) -> HashMap<T, usize>
where
    T: Copy + Eq + Hash,
{
    let mut counts = HashMap::with_capacity(handles.len());
    for handle in handles {
        *counts.entry(*handle).or_insert(0) += 1;
    }
    counts
}

fn decrement_count<T>(counts: &mut HashMap<T, usize>, handle: T) -> bool
where
    T: Copy + Eq + Hash,
{
    let Some(count) = counts.get_mut(&handle) else {
        return false;
    };
    *count -= 1;
    if *count == 0 {
        counts.remove(&handle);
    }
    true
}

#[allow(clippy::large_enum_variant)]
enum EpisodeTaskState<G, C, S>
where
    S: MctsStrategyState,
{
    Start,
    Root(MctsRootTask<G, C, S>),
    Measure { stop_reason: MctsStopReason },
    Done,
    DoneResult(MctsEpisode<G, C>),
}

#[allow(clippy::large_enum_variant)]
enum PendingEpisodeWork<G, C, S>
where
    S: MctsStrategyState,
{
    Root {
        token: WorkToken,
        inner: WorkToken,
        root_task: MctsRootTask<G, C, S>,
    },
    Measure {
        token: WorkToken,
        stop_reason: MctsStopReason,
    },
}

impl<G, C, S> PendingEpisodeWork<G, C, S>
where
    S: MctsStrategyState,
{
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
