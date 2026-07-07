use crate::{SearchAction, SearchCandidateSummary};
use gz_engine::{
    CandidateOptions, MeasureOptions, MeasureResult, ModelVersion, PortableSearchActionRef,
    ReplayGraphContext, SearchConfigHash, SearchStepRef,
};
use gz_eval::EvalOpponentContext;
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
    /// Auto-temper the root Gumbel noise (whittlezero's overlap): when
    /// non-negative, per-root bisection replaces gumbel_scale with the
    /// scale at which a noisy argmax lands in the prior's top-m actions
    /// with probability overlap + 0.05 (the noisy argmax distributes as
    /// softmax(logits/scale)). Sharp policies get more noise, flat ones
    /// less; negative disables. Part of the search config hash.
    pub gumbel_noise_overlap: f32,
    pub tree_reuse: bool,
    /// Export real position features (root_step, leaf_depth, budget) to
    /// evals and feature rows. Off zeroes the exported values so the
    /// model conditions on graph + opponent alone (and eval-cache keys
    /// collide across steps/depths). The search itself always uses the
    /// real values internally (noise seeding, budgets); deliberately not
    /// part of the search config hash.
    pub export_position: bool,
    /// Mask STOP out of node priors/logits wherever a rewrite exists
    /// (STOP-only nodes keep it). Set by policy_rollout(): an argmax
    /// reference that can stop converges to stop-at-root, freezing the
    /// bar at root cost (whittlezero's rollouts exclude STOP the same
    /// way). Part of the search config hash.
    pub mask_stop: bool,
    /// Mask any action whose applied child is the current root or a
    /// prior root of this episode (whittlezero's no_backtrack): the
    /// search must find genuinely new states, and a root where every
    /// rewrite revisits history collapses the policy target onto STOP.
    /// Within-simulation cycles are already handled by the descent seen
    /// set. Part of the search config hash.
    pub no_backtrack: bool,
    pub candidate_options: CandidateOptions,
    pub measure_options: MeasureOptions,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
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
    /// See [`GumbelMctsConfig::export_position`]; consulted only when
    /// exporting eval position contexts, never for search internals.
    pub export_position: bool,
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
            export_position: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GumbelOpponentContext {
    pub trajectory_id: u64,
    pub row_count: u32,
    pub final_reward: f32,
}

impl From<GumbelOpponentContext> for EvalOpponentContext {
    fn from(context: GumbelOpponentContext) -> Self {
        Self {
            trajectory_id: context.trajectory_id,
            row_count: context.row_count,
            final_reward: context.final_reward,
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
    pub portable_contexts: usize,
    pub carried_nodes: usize,
    pub carried_root_visits: u32,
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GumbelHandleBatch<G, C> {
    pub graphs: Vec<G>,
    pub candidates: Vec<C>,
}

impl<G, C> Default for GumbelHandleBatch<G, C> {
    fn default() -> Self {
        Self {
            graphs: Vec::new(),
            candidates: Vec::new(),
        }
    }
}

impl<G, C> GumbelHandleBatch<G, C> {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty() && self.candidates.is_empty()
    }
}

impl<G, C> PartialEq for GumbelEpisode<G, C> {
    fn eq(&self, other: &Self) -> bool {
        self.root_context == other.root_context
            && self.final_context == other.final_context
            && self.steps == other.steps
            && self.root_stats == other.root_stats
            && measure_result_eq(&self.final_measure, &other.final_measure)
            && self.stop_reason == other.stop_reason
            && self.search_config_hash == other.search_config_hash
    }
}

fn measure_result_eq<G>(left: &MeasureResult<G>, right: &MeasureResult<G>) -> bool {
    left.graph_hash == right.graph_hash
        && left.config_hash == right.config_hash
        && left.measured == right.measured
        && left.valid == right.valid
        && left.latency == right.latency
        && left.scalar_reward == right.scalar_reward
        && left.failure == right.failure
        && left.metadata == right.metadata
}

#[derive(Clone, Debug)]
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

impl<G, C> PartialEq for GumbelStep<G, C> {
    fn eq(&self, other: &Self) -> bool {
        self.step_ref == other.step_ref
            && self.selected_action == other.selected_action
            && self.selected_candidate == other.selected_candidate
            && self.engine_candidate_count == other.engine_candidate_count
            && self.action_count == other.action_count
            && self.selected_rank == other.selected_rank
            && self.legal_actions == other.legal_actions
            && self.policy_target == other.policy_target
            && self.considered_action_indices == other.considered_action_indices
            && self.root_value == other.root_value
            && self.root_search_value == other.root_search_value
            && self.root_q_max == other.root_q_max
            && self.model_version == other.model_version
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GumbelStopReason {
    MaxSteps,
    SelectedStop,
}
