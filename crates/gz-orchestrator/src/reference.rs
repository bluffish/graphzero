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
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reference<G> {
    pub kind: ReplayReferenceKind,
    pub final_reward: f32,
    pub final_graph: ReplayGraphContext,
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
            final_graph,
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
        final_graph,
        steps: reference_steps,
        search_config_hash,
        model_version: None,
    })
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
