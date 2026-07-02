use gz_engine::{
    ActionSetHash, ApplyResult, CandidateHash, CandidateInfo, CandidateKindId, CandidateMetadata,
    CandidateOptions, CandidateTags, EngineId, EngineResult, EngineVersion, GraphArtifact,
    GraphArtifactFormat, GraphEngine, GraphHash, MeasureConfigHash, MeasureMetadata,
    MeasureOptions, MeasureResult, SubjectId,
};
use gz_engine_whittle::{WhittleEngine, WhittleEngineConfig};
use gz_orchestrator::reference::{
    BeamReferenceProvider, GreedyReferenceProvider, RandomReferenceProvider, ReferenceProvider,
    RootBaselineProvider,
};
use gz_search::{
    BeamSearch, BeamSearchConfig, GreedySearch, GreedySearchConfig, RandomSearch,
    RandomSearchConfig, SearchStep,
};
use std::num::NonZeroUsize;

fn whittle() -> WhittleEngine {
    WhittleEngine::new(WhittleEngineConfig::default()).unwrap()
}

fn measure_options(engine: &WhittleEngine) -> MeasureOptions {
    engine.measure_options()
}

fn greedy(engine: &WhittleEngine) -> GreedySearch {
    GreedySearch::new(GreedySearchConfig {
        max_steps: 3,
        candidate_options: CandidateOptions::default(),
        measure_options: measure_options(engine),
    })
}

fn beam(engine: &WhittleEngine) -> BeamSearch {
    BeamSearch::new(BeamSearchConfig {
        max_depth: 3,
        beam_width: NonZeroUsize::new(4).unwrap(),
        candidate_options: CandidateOptions::default(),
        measure_options: measure_options(engine),
    })
}

fn random(engine: &WhittleEngine) -> RandomSearch {
    RandomSearch::new(RandomSearchConfig {
        max_steps: 3,
        seed: 11,
        candidate_options: CandidateOptions::default(),
        measure_options: measure_options(engine),
    })
}

#[test]
fn root_baseline_reward_matches_engine_measure() {
    let mut engine = whittle();
    let root = engine.root();
    let expected = engine
        .measure(root, measure_options(&engine))
        .unwrap()
        .scalar_reward
        .unwrap();
    let mut provider = RootBaselineProvider::new(measure_options(&engine));
    let reference = provider.reference(&mut engine, root).unwrap().unwrap();

    assert_eq!(reference.kind, gz_replay::ReplayReferenceKind::RootBaseline);
    assert_eq!(reference.final_reward, expected);
    assert_eq!(reference.steps.len(), 1);
    assert_eq!(reference.steps[0].graph, root);
    assert_eq!(reference.final_graph, reference.steps[0].context);
    assert_eq!(reference.search_config_hash, None);
}

#[test]
fn greedy_provider_matches_direct_run() {
    let mut engine = whittle();
    let root = engine.root();
    let mut provider = GreedyReferenceProvider::new(greedy(&engine));
    let reference = provider.reference(&mut engine, root).unwrap().unwrap();

    let mut direct_engine = whittle();
    let direct_search = greedy(&direct_engine);
    let direct_root = direct_engine.root();
    let direct = direct_search.run(&mut direct_engine, direct_root).unwrap();

    assert_eq!(reference.kind, gz_replay::ReplayReferenceKind::Greedy);
    assert_eq!(
        reference.final_reward,
        direct.final_measure.scalar_reward.unwrap()
    );
    assert_eq!(reference.final_graph, direct.final_context);
    assert_eq!(
        reference.search_config_hash,
        Some(direct.search_config_hash)
    );
    assert_eq!(contexts(&reference.steps), direct_contexts(&direct.steps));
}

#[test]
fn beam_provider_matches_direct_run() {
    let mut engine = whittle();
    let root = engine.root();
    let mut provider = BeamReferenceProvider::new(beam(&engine));
    let reference = provider.reference(&mut engine, root).unwrap().unwrap();

    let mut direct_engine = whittle();
    let direct_search = beam(&direct_engine);
    let direct_root = direct_engine.root();
    let direct = direct_search.run(&mut direct_engine, direct_root).unwrap();

    assert_eq!(reference.kind, gz_replay::ReplayReferenceKind::Beam);
    assert_eq!(
        reference.final_reward,
        direct.final_measure.scalar_reward.unwrap()
    );
    assert_eq!(reference.final_graph, direct.final_context);
    assert_eq!(
        reference.search_config_hash,
        Some(direct.search_config_hash)
    );
    assert_eq!(contexts(&reference.steps), direct_contexts(&direct.steps));
}

#[test]
fn random_provider_matches_direct_run() {
    let mut engine = whittle();
    let root = engine.root();
    let mut provider = RandomReferenceProvider::new(random(&engine));
    let reference = provider.reference(&mut engine, root).unwrap().unwrap();

    let mut direct_engine = whittle();
    let direct_search = random(&direct_engine);
    let direct_root = direct_engine.root();
    let direct = direct_search.run(&mut direct_engine, direct_root).unwrap();

    assert_eq!(reference.kind, gz_replay::ReplayReferenceKind::Random);
    assert_eq!(
        reference.final_reward,
        direct.final_measure.scalar_reward.unwrap()
    );
    assert_eq!(reference.final_graph, direct.final_context);
    assert_eq!(
        reference.search_config_hash,
        Some(direct.search_config_hash)
    );
    assert_eq!(contexts(&reference.steps), direct_contexts(&direct.steps));
}

#[test]
fn provider_is_deterministic_across_fresh_engines() {
    let mut left_engine = whittle();
    let mut right_engine = whittle();
    let mut left = RandomReferenceProvider::new(random(&left_engine));
    let mut right = RandomReferenceProvider::new(random(&right_engine));

    let left_root = left_engine.root();
    let right_root = right_engine.root();
    let left = left
        .reference(&mut left_engine, left_root)
        .unwrap()
        .unwrap();
    let right = right
        .reference(&mut right_engine, right_root)
        .unwrap()
        .unwrap();

    assert_eq!(left.final_reward, right.final_reward);
    assert_eq!(left.final_graph, right.final_graph);
    assert_eq!(contexts(&left.steps), contexts(&right.steps));
}

#[test]
fn unscoreable_final_measure_returns_none() {
    let mut engine = UnscoreableEngine;
    let root = engine.root();
    let mut provider =
        RootBaselineProvider::new(MeasureOptions::new(config_hash(), 1, None, true).unwrap());

    assert!(provider.reference(&mut engine, root).unwrap().is_none());
}

fn contexts<G>(
    steps: &[gz_orchestrator::reference::ReferenceStep<G>],
) -> Vec<gz_engine::ReplayGraphContext> {
    steps.iter().map(|step| step.context).collect()
}

fn direct_contexts<G, C>(steps: &[SearchStep<G, C>]) -> Vec<gz_engine::ReplayGraphContext> {
    let mut contexts = Vec::new();
    if let Some(first) = steps.first() {
        contexts.push(first.step_ref.before);
    }
    contexts.extend(steps.iter().map(|step| step.step_ref.after));
    contexts
}

fn graph_hash(value: u8) -> GraphHash {
    GraphHash::from_bytes([value; 32])
}

fn candidate_hash(value: u8) -> CandidateHash {
    CandidateHash::from_bytes([value; 32])
}

fn config_hash() -> MeasureConfigHash {
    MeasureConfigHash::from_bytes([3; 32])
}

struct UnscoreableEngine;

impl GraphEngine for UnscoreableEngine {
    type Graph = u8;
    type Candidate = u8;

    fn engine_id(&self) -> EngineId {
        EngineId::from_bytes([1; 16])
    }

    fn engine_version(&self) -> EngineVersion {
        EngineVersion::from_bytes([2; 16])
    }

    fn action_set_hash(&self) -> ActionSetHash {
        ActionSetHash::from_bytes([4; 32])
    }

    fn root(&self) -> Self::Graph {
        0
    }

    fn hash(&self, graph: Self::Graph) -> EngineResult<GraphHash> {
        Ok(graph_hash(graph))
    }

    fn candidates(
        &mut self,
        _graph: Self::Graph,
        _options: CandidateOptions,
        out: &mut Vec<Self::Candidate>,
    ) -> EngineResult<()> {
        out.clear();
        Ok(())
    }

    fn candidate_info(
        &self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<CandidateInfo> {
        Ok(CandidateInfo {
            candidate_hash: candidate_hash(candidate),
            graph_hash: graph_hash(graph),
            action_set_hash: self.action_set_hash(),
            kind: CandidateKindId::new(0),
            display_name: "candidate".to_owned(),
            static_prior: 0.0,
            tags: CandidateTags::EMPTY,
            subjects: Vec::<SubjectId>::new(),
            metadata: CandidateMetadata::default(),
        })
    }

    fn apply(
        &mut self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<ApplyResult<Self::Graph, Self::Candidate>> {
        Ok(ApplyResult {
            before: graph,
            after: graph,
            before_hash: graph_hash(graph),
            after_hash: graph_hash(graph),
            candidate,
            candidate_hash: candidate_hash(candidate),
            changed: false,
            rejected: None,
            metrics: gz_engine::ApplyMetrics::default(),
        })
    }

    fn measure(
        &mut self,
        graph: Self::Graph,
        options: MeasureOptions,
    ) -> EngineResult<MeasureResult<Self::Graph>> {
        Ok(MeasureResult {
            graph,
            graph_hash: graph_hash(graph),
            config_hash: options.config_hash,
            measured: true,
            valid: false,
            latency: None,
            scalar_reward: None,
            failure: None,
            metadata: MeasureMetadata::default(),
        })
    }

    fn export_graph(&self, graph: Self::Graph) -> EngineResult<GraphArtifact> {
        Ok(GraphArtifact {
            graph_hash: graph_hash(graph),
            format: GraphArtifactFormat::Binary,
            bytes: vec![graph],
        })
    }
}
