use gz_engine::{
    ActionSetHash, ApplyMetrics, ApplyResult, CandidateHash, CandidateInfo, CandidateKindId,
    CandidateMetadata, CandidateOptions, CandidateTags, EngineId, EngineResult, EngineVersion,
    GraphArtifact, GraphArtifactFormat, GraphEngine, GraphHash, MeasureConfigHash, MeasureMetadata,
    MeasureOptions, MeasureResult, SubjectId,
};
use gz_eval::{EvalOutput, EvalRequest, EvalResult, Evaluator};
use gz_orchestrator::{
    CountedRoots, SerialGumbelOrchestrator, ThreadedGumbelOrchestrator, ThreadedOrchestratorConfig,
    WorkerId,
};
use gz_search::{GumbelEpisodeContext, GumbelMcts, GumbelMctsConfig};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Default)]
struct ReleaseLog {
    entries: Arc<Mutex<Vec<ReleaseEntry>>>,
}

type ReleaseEntry = (Vec<u8>, Vec<u8>);

impl ReleaseLog {
    fn entries(&self) -> Vec<ReleaseEntry> {
        self.entries.lock().unwrap().clone()
    }
}

struct ReleaseEngine {
    log: ReleaseLog,
}

impl ReleaseEngine {
    fn new(log: ReleaseLog) -> Self {
        Self { log }
    }
}

impl GraphEngine for ReleaseEngine {
    type Graph = u8;
    type Candidate = u8;

    fn engine_id(&self) -> EngineId {
        EngineId::from_bytes([3; 16])
    }

    fn engine_version(&self) -> EngineVersion {
        EngineVersion::from_bytes([4; 16])
    }

    fn action_set_hash(&self) -> ActionSetHash {
        ActionSetHash::from_bytes([5; 32])
    }

    fn root(&self) -> Self::Graph {
        0
    }

    fn hash(&self, graph: Self::Graph) -> EngineResult<GraphHash> {
        Ok(graph_hash(graph))
    }

    fn candidates(
        &mut self,
        graph: Self::Graph,
        _options: CandidateOptions,
        out: &mut Vec<Self::Candidate>,
    ) -> EngineResult<()> {
        out.clear();
        if graph == 0 {
            out.push(1);
        }
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
            kind: CandidateKindId::new(candidate.into()),
            display_name: format!("candidate-{candidate}"),
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
            after: 1,
            before_hash: graph_hash(graph),
            after_hash: graph_hash(1),
            candidate,
            candidate_hash: candidate_hash(candidate),
            changed: true,
            rejected: None,
            metrics: ApplyMetrics::default(),
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
            valid: true,
            latency: None,
            scalar_reward: Some(f32::from(graph)),
            failure: None,
            metadata: MeasureMetadata::default(),
        })
    }

    fn release(
        &mut self,
        graphs: &[Self::Graph],
        candidates: &[Self::Candidate],
    ) -> EngineResult<()> {
        self.log
            .entries
            .lock()
            .unwrap()
            .push((graphs.to_vec(), candidates.to_vec()));
        Ok(())
    }

    fn export_graph(&self, graph: Self::Graph) -> EngineResult<GraphArtifact> {
        Ok(GraphArtifact {
            graph_hash: graph_hash(graph),
            format: GraphArtifactFormat::Binary,
            bytes: vec![graph],
        })
    }
}

#[derive(Clone, Copy)]
struct PickCandidate;

impl Evaluator for PickCandidate {
    fn evaluate_batch(
        &mut self,
        requests: &[EvalRequest],
        out: &mut Vec<EvalOutput>,
    ) -> EvalResult<()> {
        out.clear();
        for request in requests {
            out.push(EvalOutput {
                model_version: gz_engine::ModelVersion::from_bytes([7; 16]),
                policy_logits: if request.action_count() == 1 {
                    vec![0.0]
                } else {
                    vec![10.0, -10.0]
                },
                value: 0.0,
            });
        }
        Ok(())
    }
}

#[test]
fn serial_orchestrator_releases_completed_episode_handles() {
    let log = ReleaseLog::default();
    let engine = ReleaseEngine::new(log.clone());
    let mut orchestrator =
        SerialGumbelOrchestrator::new(WorkerId::new(0), engine, PickCandidate, search());

    let episode = orchestrator
        .run_from_root(GumbelEpisodeContext::default())
        .unwrap();

    assert_eq!(episode.episode.created_graphs, vec![1]);
    assert_eq!(episode.episode.created_candidates, Vec::<u8>::new());
    assert_eq!(log.entries(), vec![(vec![], vec![1]), (vec![1], vec![])]);
}

#[test]
fn threaded_orchestrator_releases_each_completed_episode() {
    let log = ReleaseLog::default();
    let engine = ReleaseEngine::new(log.clone());
    let orchestrator = ThreadedGumbelOrchestrator::new(
        vec![engine],
        PickCandidate,
        search(),
        ThreadedOrchestratorConfig {
            workers_per_lane: NonZeroUsize::new(1).unwrap(),
            max_batch: NonZeroUsize::new(1).unwrap(),
            flush_after: Duration::from_millis(1),
        },
    );

    let run = orchestrator
        .run(
            vec![CountedRoots::new(2, |engine: &mut ReleaseEngine| {
                Ok(engine.root())
            })],
            GumbelEpisodeContext::default(),
        )
        .unwrap();

    assert_eq!(run.lanes[0].episodes.len(), 2);
    assert_eq!(
        log.entries(),
        vec![
            (vec![], vec![1]),
            (vec![1], vec![]),
            (vec![], vec![1]),
            (vec![1], vec![])
        ]
    );
}

fn search() -> GumbelMcts {
    GumbelMcts::new(GumbelMctsConfig {
        max_steps: 1,
        simulations: NonZeroUsize::new(1).unwrap(),
        max_considered_actions: NonZeroUsize::new(2).unwrap(),
        seed: 0,
        gumbel_scale: 0.0,
        gumbel_noise_overlap: -1.0,
        c_visit: 50.0,
        c_scale: 1.0,
        temperature_moves: 0,
        tree_reuse: false,
        export_position: true,
        mask_stop: false,
        no_backtrack: false,
        candidate_options: CandidateOptions::default(),
        measure_options: measure_options(),
    })
}

fn measure_options() -> MeasureOptions {
    MeasureOptions::new(MeasureConfigHash::from_bytes([6; 32]), 1, None, true).unwrap()
}

fn graph_hash(value: u8) -> GraphHash {
    GraphHash::from_bytes([value; 32])
}

fn candidate_hash(value: u8) -> CandidateHash {
    CandidateHash::from_bytes([value; 32])
}
