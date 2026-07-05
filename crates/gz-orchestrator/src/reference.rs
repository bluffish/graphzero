use gz_engine::{
    EngineResult, GraphEngine, MeasureOptions, ModelVersion, PortableGraphId, ReplayGraphContext,
    SearchConfigHash,
};
use gz_replay::ReplayReferenceKind;
use gz_search::{BeamSearch, GreedySearch, RandomSearch, SearchStep};

pub trait ReferenceProvider<E: GraphEngine> {
    fn reference(
        &mut self,
        engine: &mut E,
        root: E::Graph,
    ) -> EngineResult<Option<Reference<E::Graph>>>;

    /// Called by the replay drivers for every replay-eligible completed
    /// episode with the learner's final measured reward. Default: no-op.
    fn observe(&mut self, learner_reward: f32) {
        let _ = learner_reward;
    }

    /// Rollout-driven providers (the policy opponent) answer true when
    /// the lane should play one opponent episode from the fixed root:
    /// `latest` is the newest model version seen on this lane's eval
    /// replies, and a rollout is due whenever it differs from the version
    /// the current reference was played under. Default: never.
    fn rollout_due(&self, latest: Option<ModelVersion>) -> bool {
        let _ = latest;
        false
    }

    /// The lane admitted the requested rollout episode under `version`.
    fn begin_rollout(&mut self, version: ModelVersion) {
        let _ = version;
    }

    /// The rollout episode finished. None means it went unmeasured or
    /// invalid; the provider keeps its previous reference and the lane
    /// will retry while the version still differs.
    fn finish_rollout(&mut self, outcome: Option<RolloutOutcome>) {
        let _ = outcome;
    }
}

/// The measured result of an opponent rollout episode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RolloutOutcome {
    pub final_reward: f32,
    pub final_graph: ReplayGraphContext,
    pub search_config_hash: SearchConfigHash,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reference<G> {
    pub kind: ReplayReferenceKind,
    pub final_reward: f32,
    pub final_graph: Option<ReplayGraphContext>,
    pub steps: Vec<ReferenceStep<G>>,
    pub search_config_hash: Option<SearchConfigHash>,
    pub model_version: Option<ModelVersion>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceStep<G> {
    pub graph: G,
    pub context: ReplayGraphContext,
}

pub struct RootBaselineProvider {
    measure_options: MeasureOptions,
}

impl RootBaselineProvider {
    #[must_use]
    pub const fn new(measure_options: MeasureOptions) -> Self {
        Self { measure_options }
    }
}

impl<E> ReferenceProvider<E> for RootBaselineProvider
where
    E: GraphEngine,
{
    fn reference(
        &mut self,
        engine: &mut E,
        root: E::Graph,
    ) -> EngineResult<Option<Reference<E::Graph>>> {
        let measure = engine.measure(root, self.measure_options)?;
        let Some(final_reward) = score(measure.measured, measure.valid, measure.scalar_reward)
        else {
            return Ok(None);
        };
        let final_graph = context(engine, measure.graph_hash);

        Ok(Some(Reference {
            kind: ReplayReferenceKind::RootBaseline,
            final_reward,
            final_graph: Some(final_graph),
            steps: vec![ReferenceStep {
                graph: root,
                context: final_graph,
            }],
            search_config_hash: None,
            model_version: None,
        }))
    }
}

pub struct GreedyReferenceProvider {
    search: GreedySearch,
}

impl GreedyReferenceProvider {
    #[must_use]
    pub fn new(search: GreedySearch) -> Self {
        Self { search }
    }
}

impl<E> ReferenceProvider<E> for GreedyReferenceProvider
where
    E: GraphEngine,
{
    fn reference(
        &mut self,
        engine: &mut E,
        root: E::Graph,
    ) -> EngineResult<Option<Reference<E::Graph>>> {
        let episode = self.search.run(engine, root)?;
        Ok(project_search_episode(
            ReplayReferenceKind::Greedy,
            episode.root,
            episode.final_context,
            &episode.steps,
            score(
                episode.final_measure.measured,
                episode.final_measure.valid,
                episode.final_measure.scalar_reward,
            ),
            Some(episode.search_config_hash),
        ))
    }
}

pub struct BeamReferenceProvider {
    search: BeamSearch,
}

impl BeamReferenceProvider {
    #[must_use]
    pub fn new(search: BeamSearch) -> Self {
        Self { search }
    }
}

impl<E> ReferenceProvider<E> for BeamReferenceProvider
where
    E: GraphEngine,
{
    fn reference(
        &mut self,
        engine: &mut E,
        root: E::Graph,
    ) -> EngineResult<Option<Reference<E::Graph>>> {
        let episode = self.search.run(engine, root)?;
        Ok(project_search_episode(
            ReplayReferenceKind::Beam,
            episode.root,
            episode.final_context,
            &episode.steps,
            score(
                episode.final_measure.measured,
                episode.final_measure.valid,
                episode.final_measure.scalar_reward,
            ),
            Some(episode.search_config_hash),
        ))
    }
}

pub struct RandomReferenceProvider {
    search: RandomSearch,
}

impl RandomReferenceProvider {
    #[must_use]
    pub fn new(search: RandomSearch) -> Self {
        Self { search }
    }
}

impl<E> ReferenceProvider<E> for RandomReferenceProvider
where
    E: GraphEngine,
{
    fn reference(
        &mut self,
        engine: &mut E,
        root: E::Graph,
    ) -> EngineResult<Option<Reference<E::Graph>>> {
        let episode = self.search.run(engine, root)?;
        Ok(project_search_episode(
            ReplayReferenceKind::Random,
            episode.root,
            episode.final_context,
            &episode.steps,
            score(
                episode.final_measure.measured,
                episode.final_measure.valid,
                episode.final_measure.scalar_reward,
            ),
            Some(episode.search_config_hash),
        ))
    }
}

fn project_search_episode<G, C>(
    kind: ReplayReferenceKind,
    root: G,
    final_graph: ReplayGraphContext,
    steps: &[SearchStep<G, C>],
    final_reward: Option<f32>,
    search_config_hash: Option<SearchConfigHash>,
) -> Option<Reference<G>>
where
    G: Copy,
{
    let final_reward = final_reward?;
    let mut reference_steps = Vec::with_capacity(steps.len() + 1);

    match steps.first() {
        Some(step) => reference_steps.push(ReferenceStep {
            graph: root,
            context: step.step_ref.before,
        }),
        None => reference_steps.push(ReferenceStep {
            graph: root,
            context: final_graph,
        }),
    }

    reference_steps.extend(steps.iter().map(|step| ReferenceStep {
        graph: step.after,
        context: step.step_ref.after,
    }));

    Some(Reference {
        kind,
        final_reward,
        final_graph: Some(final_graph),
        steps: reference_steps,
        search_config_hash,
        model_version: None,
    })
}

/// Adaptive reference: a reward EMA of the learner's own recent episodes
/// on this provider's lane. Unlabeled until the first observed episode
/// seeds the EMA. Never touches the engine.
pub struct SelfAverageProvider {
    decay: f64,
    ema: Option<f64>,
}

impl SelfAverageProvider {
    #[must_use]
    pub fn new(decay: f32) -> Self {
        assert!(
            decay.is_finite() && decay > 0.0 && decay < 1.0,
            "self-average decay must be in (0, 1)"
        );
        Self {
            decay: f64::from(decay),
            ema: None,
        }
    }
}

impl<E> ReferenceProvider<E> for SelfAverageProvider
where
    E: GraphEngine,
{
    fn reference(
        &mut self,
        _engine: &mut E,
        _root: E::Graph,
    ) -> EngineResult<Option<Reference<E::Graph>>> {
        let Some(ema) = self.ema else {
            return Ok(None);
        };

        Ok(Some(Reference {
            kind: ReplayReferenceKind::SelfAverage,
            final_reward: ema as f32,
            final_graph: None,
            steps: Vec::new(),
            search_config_hash: None,
            model_version: None,
        }))
    }

    fn observe(&mut self, learner_reward: f32) {
        let reward = f64::from(learner_reward);
        self.ema = Some(match self.ema {
            None => reward,
            Some(ema) => self.decay * ema + (1.0 - self.decay) * reward,
        });
    }
}

/// The network itself as the opponent: the terminal reward of a greedy
/// (temperature-0, one-simulation) policy rollout from the fixed root,
/// played once per published checkpoint. The lane drives the rollout
/// through the normal episode machinery and reports back through the
/// rollout hooks; this provider only holds the resulting scalar.
/// Episodes are unlabeled until the first rollout completes.
pub struct PolicyReferenceProvider {
    current: Option<PolicyReference>,
    pending_version: Option<ModelVersion>,
}

struct PolicyReference {
    reward: f32,
    version: ModelVersion,
    final_graph: ReplayGraphContext,
    search_config_hash: SearchConfigHash,
}

impl PolicyReferenceProvider {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            current: None,
            pending_version: None,
        }
    }
}

impl Default for PolicyReferenceProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl<E> ReferenceProvider<E> for PolicyReferenceProvider
where
    E: GraphEngine,
{
    fn reference(
        &mut self,
        _engine: &mut E,
        _root: E::Graph,
    ) -> EngineResult<Option<Reference<E::Graph>>> {
        let Some(current) = &self.current else {
            return Ok(None);
        };

        Ok(Some(Reference {
            kind: ReplayReferenceKind::Gumbel,
            final_reward: current.reward,
            final_graph: Some(current.final_graph),
            steps: Vec::new(),
            search_config_hash: Some(current.search_config_hash),
            model_version: Some(current.version),
        }))
    }

    fn rollout_due(&self, latest: Option<ModelVersion>) -> bool {
        if self.pending_version.is_some() {
            return false;
        }
        match latest {
            Some(latest) => self
                .current
                .as_ref()
                .is_none_or(|current| current.version != latest),
            None => false,
        }
    }

    fn begin_rollout(&mut self, version: ModelVersion) {
        self.pending_version = Some(version);
    }

    fn finish_rollout(&mut self, outcome: Option<RolloutOutcome>) {
        let Some(version) = self.pending_version.take() else {
            return;
        };
        if let Some(outcome) = outcome {
            self.current = Some(PolicyReference {
                reward: outcome.final_reward,
                version,
                final_graph: outcome.final_graph,
                search_config_hash: outcome.search_config_hash,
            });
        }
    }
}

fn score(measured: bool, valid: bool, scalar_reward: Option<f32>) -> Option<f32> {
    if !measured || !valid {
        return None;
    }

    match scalar_reward {
        Some(reward) if reward.is_finite() => Some(reward),
        _ => None,
    }
}

fn context<E: GraphEngine>(engine: &E, graph_hash: gz_engine::GraphHash) -> ReplayGraphContext {
    ReplayGraphContext::new(
        PortableGraphId::new(graph_hash, engine.engine_id(), engine.engine_version()),
        engine.action_set_hash(),
    )
}
