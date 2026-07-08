use super::super::schedule::{
    best_count_action, best_eligible, root_q_max, sample_count_action, search_value,
    selectable_root_actions, softmax,
};
use super::super::tree::{Edge, Node, Tree};
use super::super::{
    GumbelHandleBatch, GumbelMcts, GumbelMctsConfig, GumbelRootResult, GumbelRootStats,
    GumbelSearchContext,
};
use super::state::{
    DescentPoll, DescentState, NodeExpansion, PendingRootWork, RootTaskState, RunState,
};
use crate::SearchCandidateSummary;
use crate::support::internal;
use crate::work::{
    ApplyWork, EngineIdentity, EvalWork, ExpandResult, ExpandWork, SearchPoll, SearchWork,
    SearchWorkResult, WorkToken,
};
use gz_engine::{ApplyResult, EngineResult, PortableCandidateRef, ReplayGraphContext};
use gz_eval::{EvalAction, EvalRequest, eval_error_to_engine_error};
use std::collections::HashSet;

pub struct GumbelRootTask<G, C> {
    pub(super) config: GumbelMctsConfig,
    identity: EngineIdentity,
    root: G,
    pub(super) context: GumbelSearchContext,
    pub(super) root_context: Option<ReplayGraphContext>,
    /// Prior episode roots (no_backtrack): applied children matching one
    /// of these are masked. The current root is checked separately via
    /// root_context. Empty unless the episode task installs it.
    visited: HashSet<ReplayGraphContext>,
    pub(super) tree: Tree<G, C>,
    pub(super) next_token: u64,
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
            visited: HashSet::new(),
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

    pub(super) fn set_visited(&mut self, visited: HashSet<ReplayGraphContext>) {
        self.visited = visited;
    }

    pub(super) fn reused_child_task(
        &self,
        action: usize,
        root: G,
        expected_context: ReplayGraphContext,
        context: GumbelSearchContext,
    ) -> EngineResult<Option<ReusedRootTask<G, C>>> {
        let Some(child_index) = self.tree.nodes[0].children[action] else {
            return Ok(None);
        };
        let (tree, root_context, handles) = self.tree.compact_subtree(child_index, context)?;
        if root_context != expected_context {
            return Err(internal("reused root context mismatch"));
        }

        let mut task = Self {
            config: self.config,
            identity: self.identity,
            root,
            context,
            root_context: Some(root_context),
            visited: HashSet::new(),
            tree,
            next_token: 0,
            pending: None,
            state: RootTaskState::Done,
        };
        let run = task.start_run_state();
        task.state = RootTaskState::Running(run);
        Ok(Some(ReusedRootTask { task, handles }))
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
        self.tree.portable_contexts += 1;
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
        let mut candidate_hashes = Vec::with_capacity(result.candidates.len());
        let mut summaries = Vec::with_capacity(result.candidates.len() + 1);

        for expanded in result.candidates {
            candidates.push(expanded.candidate);
            let candidate_ref = PortableCandidateRef::new(context, expanded.candidate_hash);
            candidate_hashes.push(expanded.candidate_hash);
            eval_actions.push(EvalAction::candidate(
                candidate_ref,
                expanded.kind,
                expanded.tags,
                expanded.static_prior,
            ));
            summaries.push(Some(SearchCandidateSummary {
                kind: expanded.kind,
                tags: expanded.tags,
                static_prior: expanded.static_prior,
            }));
        }

        eval_actions.push(EvalAction::stop(context));
        summaries.push(None);

        self.state = RootTaskState::EmitNodeEval {
            expansion: NodeExpansion {
                graph,
                context,
                depth,
                candidates,
                eval_actions,
                candidate_hashes,
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

        let child_context = self.identity.context(applied.after_hash);
        self.tree.portable_contexts += 1;

        if self.config.no_backtrack
            && (self.root_context == Some(child_context) || self.visited.contains(&child_context))
        {
            self.tree.mask_action(descent.node_index, action);
            run.descent = Some(descent);
            self.state = RootTaskState::Running(run);
            return Ok(());
        }

        descent.path.push(Edge {
            node_index: descent.node_index,
            action,
        });

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
        let mut node = Node {
            graph: expansion.graph,
            context: expansion.context,
            candidates: expansion.candidates,
            eval_actions: if self.context.opponent.is_some() {
                expansion.eval_actions
            } else {
                Vec::new()
            },
            candidate_hashes: expansion.candidate_hashes,
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
        if self.config.mask_stop && !node.candidates.is_empty() {
            let stop = node.candidates.len();
            node.logits[stop] = f32::NEG_INFINITY;
            node.priors[stop] = 0.0;
        }

        self.tree.eval_count += 1;
        self.tree.nodes.push(node);
        self.tree.nodes.len() - 1
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
        // whittlezero's rule: the most-visited considered action, score
        // tie-break -- winning the halving is an implicit consolidation
        // filter (a raw score argmax plays 1-visit actions with lucky Q
        // samples, which measurably sends weak-net episodes wandering).
        // Correct only on fresh trees: carried visits freeze the previous
        // move's preference, so whittlezero-faithful selection requires
        // tree_reuse off.
        let fallback = best_count_action(&root_node.visits, &selectable, &root_scores);
        let selected = if self.context.selection_temperature > 0.0 {
            sample_count_action(
                &mut run.rng,
                &root_node.visits,
                &selectable,
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
            selected_action_ref: root_node.action_ref(selected)?,
            selected_candidate: root_node.summaries[selected],
            selected_action_index: selected,
            engine_candidate_count: root_node.candidates.len(),
            action_count: root_node.action_count(),
            legal_actions: root_node.action_refs(),
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
                portable_contexts: self.tree.portable_contexts,
                carried_nodes: self.tree.carried_nodes,
                carried_root_visits: self.tree.carried_root_visits,
            },
        }))
    }
}

pub(super) struct ReusedRootTask<G, C> {
    pub(super) task: GumbelRootTask<G, C>,
    pub(super) handles: GumbelHandleBatch<G, C>,
}
