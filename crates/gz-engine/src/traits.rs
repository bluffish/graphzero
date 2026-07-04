//! Engine trait definitions.

use crate::{
    ActionSetHash, ApplyJob, ApplyResult, CandidateInfo, CandidateOptions, EngineId, EngineResult,
    EngineVersion, GraphArtifact, GraphHash, MeasureOptions, MeasureResult, PortableCandidateRef,
    PortableGraphId,
};
use std::hash::Hash;

pub trait GraphEngine {
    type Graph: Copy + Eq + Hash + Send + Sync + 'static;
    type Candidate: Copy + Eq + Hash + Send + Sync + 'static;

    fn engine_id(&self) -> EngineId;
    fn engine_version(&self) -> EngineVersion;
    fn action_set_hash(&self) -> ActionSetHash;

    fn root(&self) -> Self::Graph;

    fn hash(&self, graph: Self::Graph) -> EngineResult<GraphHash>;

    fn candidates(
        &mut self,
        graph: Self::Graph,
        options: CandidateOptions,
        out: &mut Vec<Self::Candidate>,
    ) -> EngineResult<()>;

    fn candidate_info(
        &self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<CandidateInfo>;

    fn apply(
        &mut self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<ApplyResult<Self::Graph, Self::Candidate>>;

    fn measure(
        &mut self,
        graph: Self::Graph,
        options: MeasureOptions,
    ) -> EngineResult<MeasureResult<Self::Graph>>;

    fn release(
        &mut self,
        graphs: &[Self::Graph],
        candidates: &[Self::Candidate],
    ) -> EngineResult<()> {
        let _ = (graphs, candidates);
        Ok(())
    }

    fn export_graph(&self, graph: Self::Graph) -> EngineResult<GraphArtifact>;
}

pub trait BatchGraphEngine: GraphEngine {
    fn candidates_batch(
        &mut self,
        graphs: &[Self::Graph],
        options: CandidateOptions,
    ) -> Vec<EngineResult<Vec<Self::Candidate>>> {
        let mut results = Vec::with_capacity(graphs.len());

        for graph in graphs.iter().copied() {
            let mut candidates = Vec::new();
            results.push(
                self.candidates(graph, options, &mut candidates)
                    .map(|()| candidates),
            );
        }

        results
    }

    fn apply_batch(
        &mut self,
        jobs: &[ApplyJob<Self::Graph, Self::Candidate>],
    ) -> Vec<EngineResult<ApplyResult<Self::Graph, Self::Candidate>>> {
        jobs.iter()
            .copied()
            .map(|job| self.apply(job.graph, job.candidate))
            .collect()
    }

    fn measure_batch(
        &mut self,
        graphs: &[Self::Graph],
        options: MeasureOptions,
    ) -> Vec<EngineResult<MeasureResult<Self::Graph>>> {
        graphs
            .iter()
            .copied()
            .map(|graph| self.measure(graph, options))
            .collect()
    }
}

pub trait EngineReplayResolver<E: GraphEngine> {
    fn resolve_graph(&mut self, graph: PortableGraphId) -> EngineResult<E::Graph>;

    fn resolve_candidate(&mut self, candidate: PortableCandidateRef) -> EngineResult<E::Candidate>;
}
