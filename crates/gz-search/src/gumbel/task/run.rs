use super::super::schedule::{
    GumbelRng, considered_actions, considered_visit_sequence, overlap_noise_scale, root_seed,
    sample_root_gumbels,
};
use super::root::GumbelRootTask;
use super::state::RunState;
use crate::work::WorkToken;
use gz_engine::ReplayGraphContext;

impl<G, C> GumbelRootTask<G, C>
where
    G: Copy,
    C: Copy,
{
    /// Carried visits are information the previous step's search already
    /// paid for, so a reused root spends its budget only on what is new.
    /// A floor of a quarter budget keeps fresh exploration under the new
    /// Gumbel draw even when the carried subtree exceeds the budget.
    pub(super) fn fresh_simulation_budget(&self) -> usize {
        let simulations = self.config.simulations.get();
        let carried = self.tree.carried_root_visits as usize;
        if carried == 0 {
            return simulations;
        }

        simulations
            .saturating_sub(carried)
            .max((simulations / 4).max(1))
    }

    pub(super) fn next_token(&mut self) -> WorkToken {
        let token = WorkToken::new(self.next_token);
        self.next_token += 1;
        token
    }

    #[must_use]
    pub const fn root_context(&self) -> Option<ReplayGraphContext> {
        self.root_context
    }

    pub(super) fn start_run_state(&self) -> RunState<G, C> {
        let root_index = 0;
        let action_count = self.tree.nodes[root_index].action_count();
        let mut rng = GumbelRng::new(root_seed(
            self.config.seed ^ self.context.noise_seed,
            self.context.root_step,
        ));
        let scale = if self.config.gumbel_noise_overlap >= 0.0 {
            overlap_noise_scale(
                &self.tree.nodes[root_index].logits,
                self.config.max_considered_actions.get(),
                self.config.gumbel_noise_overlap,
                self.config.gumbel_scale,
            )
        } else {
            self.config.gumbel_scale
        };
        let root_gumbels = sample_root_gumbels(action_count, scale, &mut rng);
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
}
