use super::schedule::{completed_q, softmax};
use super::{GumbelHandleBatch, GumbelMcts, GumbelMctsConfig, GumbelSearchContext};
use crate::support::internal;
use crate::{SearchAction, SearchCandidateSummary};
use gz_engine::{
    CandidateHash, EngineResult, ModelVersion, PortableCandidateRef, PortableSearchActionRef,
    ReplayGraphContext,
};
use gz_eval::{EvalAction, EvalPositionContext};

pub(super) struct Tree<G, C> {
    pub(super) config: GumbelMctsConfig,
    pub(super) context: GumbelSearchContext,
    pub(super) nodes: Vec<Node<G, C>>,
    pub(super) eval_count: usize,
    pub(super) portable_contexts: usize,
    pub(super) carried_nodes: usize,
    pub(super) carried_root_visits: u32,
}

impl<G, C> Tree<G, C>
where
    G: Copy,
    C: Copy,
{
    pub(super) fn new(search: &GumbelMcts, context: GumbelSearchContext) -> Self {
        Self {
            config: search.config,
            context,
            nodes: Vec::new(),
            eval_count: 0,
            portable_contexts: 0,
            carried_nodes: 0,
            carried_root_visits: 0,
        }
    }

    pub(super) fn compact_subtree(
        &self,
        root_index: usize,
        context: GumbelSearchContext,
    ) -> EngineResult<(Self, ReplayGraphContext, GumbelHandleBatch<G, C>)> {
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
        let mut handles = GumbelHandleBatch::default();
        for &old_index in &old_indices {
            let mut node = self.nodes[old_index].clone();
            handles.graphs.push(node.graph);
            handles.candidates.extend(node.candidates.iter().copied());
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
                portable_contexts: 0,
                carried_nodes,
                carried_root_visits,
            },
            root_context,
            handles,
        ))
    }

    pub(super) fn backup(&mut self, path: &[Edge], value: f32) {
        for edge in path {
            let node = &mut self.nodes[edge.node_index];
            node.visits[edge.action] += 1;
            node.value_sum[edge.action] += value;
            node.q[edge.action] = node.value_sum[edge.action] / node.visits[edge.action] as f32;
        }
    }

    pub(super) fn selected_after(&self, node_index: usize, action: usize) -> EngineResult<G> {
        let node = &self.nodes[node_index];
        if node.is_stop(action) {
            return Ok(node.graph);
        }

        let child = node.children[action].ok_or_else(|| internal("missing selected child"))?;
        Ok(self.nodes[child].graph)
    }

    pub(super) fn selected_after_context(
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

    pub(super) fn root_scores(&self, node_index: usize, base_scores: &[f32]) -> Vec<f32> {
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

    pub(super) fn improved_policy(&self, node_index: usize) -> Vec<f32> {
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

    pub(super) fn select_nonroot(&self, node_index: usize) -> usize {
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

    pub(super) fn mask_action(&mut self, node_index: usize, action: usize) {
        let node = &mut self.nodes[node_index];
        node.logits[action] = f32::NEG_INFINITY;
        node.priors[action] = 0.0;
    }

    pub(super) fn position(&self, leaf_depth: usize) -> EvalPositionContext {
        // Opponent alignment always uses the real step and depth --
        // export_position zeroing applies to the exported scalars only,
        // never to which opponent state a pair eval sees.
        let opponent = self.context.opponent.map(|opponent| {
            opponent.aligned_to(u64::from(self.context.root_step) + leaf_depth as u64)
        });
        if !self.context.export_position {
            return EvalPositionContext {
                root_step: 0,
                leaf_depth: 0,
                budget_fraction: 0.0,
                budget_step: 0.0,
                opponent,
            };
        }
        EvalPositionContext {
            root_step: self.context.root_step,
            leaf_depth: leaf_depth as u32,
            budget_fraction: self.context.budget_fraction,
            budget_step: self.context.budget_step,
            opponent,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Node<G, C> {
    pub(super) graph: G,
    pub(super) context: ReplayGraphContext,
    pub(super) candidates: Vec<C>,
    pub(super) eval_actions: Vec<EvalAction>,
    pub(super) candidate_hashes: Vec<CandidateHash>,
    pub(super) summaries: Vec<Option<SearchCandidateSummary>>,
    pub(super) logits: Vec<f32>,
    pub(super) priors: Vec<f32>,
    pub(super) value: f32,
    pub(super) model_version: ModelVersion,
    pub(super) children: Vec<Option<usize>>,
    pub(super) visits: Vec<u32>,
    pub(super) value_sum: Vec<f32>,
    pub(super) q: Vec<f32>,
}

impl<G, C> Node<G, C>
where
    C: Copy,
{
    pub(super) fn action_count(&self) -> usize {
        self.logits.len()
    }

    pub(super) fn is_stop(&self, action: usize) -> bool {
        action == self.candidates.len()
    }

    pub(super) fn search_action(&self, action: usize) -> EngineResult<SearchAction<C>> {
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

    pub(super) fn action_ref(&self, action: usize) -> EngineResult<PortableSearchActionRef> {
        if self.is_stop(action) {
            Ok(PortableSearchActionRef::stop(self.context))
        } else {
            self.candidate_hashes
                .get(action)
                .copied()
                .map(|hash| {
                    PortableSearchActionRef::candidate(PortableCandidateRef::new(
                        self.context,
                        hash,
                    ))
                })
                .ok_or_else(|| internal("invalid selected action"))
        }
    }

    pub(super) fn action_refs(&self) -> Vec<PortableSearchActionRef> {
        let mut refs = Vec::with_capacity(self.candidate_hashes.len() + 1);
        refs.extend(self.candidate_hashes.iter().copied().map(|hash| {
            PortableSearchActionRef::candidate(PortableCandidateRef::new(self.context, hash))
        }));
        refs.push(PortableSearchActionRef::stop(self.context));
        refs
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Edge {
    pub(super) node_index: usize,
    pub(super) action: usize,
}
