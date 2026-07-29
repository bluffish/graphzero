#![forbid(unsafe_code)]

//! GraphZero adapter for the native compiler search engine.

mod error;
mod identity;
mod measurement;

use graphs_compiler::{
    actions::CompilerStateArtifact, engine::CompilerEngine as NativeCompilerEngine,
    transforms::TransformDefinitionId,
};
use gz_engine::{
    ActionSetHash, ApplyMetrics, ApplyResult, BatchGraphEngine, CandidateHash, CandidateInfo,
    CandidateKindId, CandidateMetadata, CandidateOptions, CandidateTags, EngineError, EngineId,
    EngineResult, EngineVersion, GraphArtifact, GraphArtifactFormat, GraphEngine, GraphHash,
    MeasureConfigHash, MeasureOptions, MeasureOptionsError, MeasureResult,
};
use std::collections::HashMap;

pub use graphs_compiler as native;
pub use graphs_compiler::{
    engine::{
        CompilerCandidateId, CompilerEngineDomain, CompilerEvaluationWorkload, CompilerStateId,
    },
    importer::ImportedProgram,
};

pub use error::CompilerEngineBuildError;
use error::{
    action_kind_invariant, invalid_artifact, measure_config_mismatch, native_error,
    unsupported_measure_timeout,
};
use identity::{
    action_set_hash, candidate_hash, compiler_engine_id, compiler_engine_version, graph_hash,
};
use measurement::project_measurement;
pub use measurement::{
    ACCURACY_EVALUATION_FAILURE_CODE, ACCURACY_FAILURE_REWARD, COMPILE_FAILURE_CODE,
    COMPILER_MEASUREMENT_FAILURE_CODE, PERFORMANCE_EVALUATION_FAILURE_CODE,
};

pub const COMPILER_STATE_ARTIFACT_FORMAT: u32 = 1;

pub struct CompilerEngineConfig {
    pub domain: CompilerEngineDomain,
    pub root: ImportedProgram,
    /// Identity of the exact workload and reward projection used by measurement.
    pub measure_config_hash: MeasureConfigHash,
    pub workload: CompilerEvaluationWorkload,
}

pub struct CompilerEngine {
    native: NativeCompilerEngine,
    root: CompilerStateId,
    engine_id: EngineId,
    engine_version: EngineVersion,
    action_set_hash: ActionSetHash,
    measure_config_hash: MeasureConfigHash,
    action_kinds: HashMap<TransformDefinitionId, CandidateKindId>,
}

impl CompilerEngine {
    pub fn new(config: CompilerEngineConfig) -> Result<Self, CompilerEngineBuildError> {
        let mut native = NativeCompilerEngine::new(config.domain)?;
        let action_kinds = action_kind_ids(native.action_definition_ids())?;
        let engine_id = compiler_engine_id();
        let engine_version = compiler_engine_version(native.domain_hash());
        let action_set_hash = action_set_hash(native.action_set_hash());
        let root = native.insert_root(config.root, config.workload)?;

        Ok(Self {
            native,
            root,
            engine_id,
            engine_version,
            action_set_hash,
            measure_config_hash: config.measure_config_hash,
            action_kinds,
        })
    }

    #[must_use]
    pub const fn measure_config_hash(&self) -> MeasureConfigHash {
        self.measure_config_hash
    }

    pub fn measure_options(&self, samples: u32) -> Result<MeasureOptions, MeasureOptionsError> {
        MeasureOptions::new(self.measure_config_hash, samples, None, true)
    }

    /// Restores one owned graph reference after validating both artifact layers.
    pub fn restore_graph(&mut self, artifact: &GraphArtifact) -> EngineResult<CompilerStateId> {
        if artifact.format != GraphArtifactFormat::AdapterSpecific(COMPILER_STATE_ARTIFACT_FORMAT) {
            return Err(invalid_artifact(
                "graph artifact format is not a compiler state artifact",
            ));
        }

        let native_artifact = CompilerStateArtifact::from(artifact.bytes.clone());
        let graph = self
            .native
            .restore_state(&native_artifact)
            .map_err(native_error)?;
        let graph_hash = self.hash(graph)?;
        if graph_hash != artifact.graph_hash {
            self.native.release(&[graph], &[]).map_err(native_error)?;
            return Err(invalid_artifact(
                "graph artifact envelope hash does not match the compiler state",
            ));
        }

        Ok(graph)
    }

    fn validate_candidate_parent(
        &self,
        actual_graph_hash: GraphHash,
        candidate: CompilerCandidateId,
    ) -> EngineResult<CandidateHash> {
        let candidate_hash = self
            .native
            .candidate_hash(candidate)
            .map(candidate_hash)
            .map_err(native_error)?;
        let parent_hash = self
            .native
            .candidate_parent_hash(candidate)
            .map(graph_hash)
            .map_err(native_error)?;

        if parent_hash != actual_graph_hash {
            return Err(EngineError::StaleCandidate {
                expected_graph_hash: parent_hash,
                actual_graph_hash,
                candidate_hash,
            });
        }

        Ok(candidate_hash)
    }
}

impl GraphEngine for CompilerEngine {
    type Graph = CompilerStateId;
    type Candidate = CompilerCandidateId;

    fn engine_id(&self) -> EngineId {
        self.engine_id
    }

    fn engine_version(&self) -> EngineVersion {
        self.engine_version
    }

    fn action_set_hash(&self) -> ActionSetHash {
        self.action_set_hash
    }

    fn root(&self) -> Self::Graph {
        self.root
    }

    fn hash(&self, graph: Self::Graph) -> EngineResult<GraphHash> {
        self.native
            .state_hash(graph)
            .map(graph_hash)
            .map_err(native_error)
    }

    fn candidates(
        &mut self,
        graph: Self::Graph,
        options: CandidateOptions,
        out: &mut Vec<Self::Candidate>,
    ) -> EngineResult<()> {
        self.native
            .candidates(graph, options.max_candidates, out)
            .map_err(native_error)
    }

    fn candidate_info(
        &self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<CandidateInfo> {
        let graph_hash = self.hash(graph)?;
        let candidate_hash = self.validate_candidate_parent(graph_hash, candidate)?;
        let candidate = self
            .native
            .candidate_view(candidate)
            .map_err(native_error)?;
        let kind = self
            .action_kinds
            .get(candidate.definition)
            .copied()
            .ok_or_else(action_kind_invariant)?;

        Ok(CandidateInfo {
            candidate_hash,
            graph_hash,
            action_set_hash: self.action_set_hash,
            kind,
            display_name: candidate.definition.to_string(),
            static_prior: 0.0,
            tags: CandidateTags::EMPTY,
            subjects: Vec::new(),
            metadata: CandidateMetadata::default(),
        })
    }

    fn apply(
        &mut self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<ApplyResult<Self::Graph, Self::Candidate>> {
        let before_hash = self.hash(graph)?;
        let candidate_hash = self.validate_candidate_parent(before_hash, candidate)?;
        let after = self.native.apply(graph, candidate).map_err(native_error)?;
        let after_hash = self.hash(after)?;

        Ok(ApplyResult {
            before: graph,
            after,
            before_hash,
            after_hash,
            candidate,
            candidate_hash,
            changed: before_hash != after_hash,
            rejected: None,
            metrics: ApplyMetrics::default(),
        })
    }

    fn measure(
        &mut self,
        graph: Self::Graph,
        options: MeasureOptions,
    ) -> EngineResult<MeasureResult<Self::Graph>> {
        let graph_hash = self.hash(graph)?;
        if options.config_hash != self.measure_config_hash {
            return Err(measure_config_mismatch());
        }
        if options.timeout_ms.is_some() {
            return Err(unsupported_measure_timeout());
        }

        project_measurement(
            graph,
            graph_hash,
            options.config_hash,
            self.native.measure(graph, options.samples),
        )
    }

    fn release(
        &mut self,
        graphs: &[Self::Graph],
        candidates: &[Self::Candidate],
    ) -> EngineResult<()> {
        self.native
            .release(graphs, candidates)
            .map_err(native_error)
    }

    fn export_graph(&self, graph: Self::Graph) -> EngineResult<GraphArtifact> {
        let graph_hash = self.hash(graph)?;
        let artifact = self.native.export_state(graph).map_err(native_error)?;

        Ok(GraphArtifact {
            graph_hash,
            format: GraphArtifactFormat::AdapterSpecific(COMPILER_STATE_ARTIFACT_FORMAT),
            bytes: artifact.as_ref().to_vec(),
        })
    }
}

impl BatchGraphEngine for CompilerEngine {}

fn action_kind_ids(
    definitions: &[TransformDefinitionId],
) -> Result<HashMap<TransformDefinitionId, CandidateKindId>, CompilerEngineBuildError> {
    definitions
        .iter()
        .enumerate()
        .map(|(index, definition)| {
            let index =
                u32::try_from(index).map_err(|_| CompilerEngineBuildError::TooManyActionKinds {
                    count: definitions.len(),
                })?;
            Ok((definition.clone(), CandidateKindId::new(index)))
        })
        .collect()
}
