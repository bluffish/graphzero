#![forbid(unsafe_code)]

//! GraphZero adapter for the native compiler search engine.

mod candidates;
mod error;
mod features;
mod identity;
mod measurement;

use graphs_compiler::{
    actions::CompilerStateArtifact, engine::CompilerEngine as NativeCompilerEngine,
    matching::MatchScope,
};
use gz_engine::{
    ActionSetHash, ApplyMetrics, ApplyResult, BatchGraphEngine, CandidateHash, CandidateInfo,
    CandidateKindId, CandidateMetadata, CandidateOptions, CandidateTags, EngineError, EngineId,
    EngineResult, EngineVersion, GraphArtifact, GraphArtifactFormat, GraphEngine, GraphHash,
    MeasureConfigHash, MeasureOptions, MeasureOptionsError, MeasureResult,
};
use std::collections::{HashMap, HashSet};

pub use graphs_compiler as native;
pub use graphs_compiler::{
    engine::{
        CompilerCandidateId, CompilerEngineDomain, CompilerEngineTarget,
        CompilerEvaluationWorkload, CompilerStateId,
    },
    importer::ImportedProgram,
};

pub use candidates::CompilerCandidateMode;
use candidates::{CompilerCandidateGroup, group_hybrid_candidates};
pub use error::CompilerEngineBuildError;
use error::{invalid_artifact, measure_config_mismatch, native_error, unsupported_measure_timeout};
pub use features::{
    CompilerFeatureExtractor, CompilerFeatureExtractorConfig, CompilerFeatureLayout,
};
use identity::{
    action_set_hash, candidate_hash, compiler_engine_id, compiler_engine_version, graph_hash,
};
pub use measurement::{
    ACCURACY_EVALUATION_FAILURE_CODE, ACCURACY_FAILURE_REWARD, COMPILE_FAILURE_CODE,
    COMPILER_MEASUREMENT_FAILURE_CODE, CompilerMeasurementEvidence,
    PERFORMANCE_EVALUATION_FAILURE_CODE, accuracy_failure_evidence, decode_measurement_evidence,
};
use measurement::{project_measurement, project_performance_only_measurement};

pub const COMPILER_STATE_ARTIFACT_FORMAT: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompilerMeasurementPolicy {
    VerifyAccuracy,
    PerformanceOnly,
}

pub struct CompilerEngineConfig {
    pub domain: CompilerEngineDomain,
    pub root: ImportedProgram,
    /// Search-action granularity. This participates in engine and action-set
    /// identity.
    pub candidate_mode: CompilerCandidateMode,
    /// Identity of the exact workload and reward projection used by measurement.
    pub measure_config_hash: MeasureConfigHash,
    pub measurement_policy: CompilerMeasurementPolicy,
    pub workload: CompilerEvaluationWorkload,
}

pub struct CompilerEngine {
    native: NativeCompilerEngine,
    root: CompilerStateId,
    engine_id: EngineId,
    engine_version: EngineVersion,
    action_set_hash: ActionSetHash,
    measure_config_hash: MeasureConfigHash,
    measurement_policy: CompilerMeasurementPolicy,
    candidate_mode: CompilerCandidateMode,
    candidate_groups: HashMap<CompilerCandidateId, CompilerCandidateGroup>,
}

impl CompilerEngine {
    pub fn new(config: CompilerEngineConfig) -> Result<Self, CompilerEngineBuildError> {
        let candidate_mode = config.candidate_mode;
        let mut native = NativeCompilerEngine::new(config.domain)?;
        if candidate_mode.uses_candidate_groups() {
            let definitions = native
                .action_definition_ids()
                .iter()
                .filter(|definition| candidate_mode.allows_definition(definition.as_ref()))
                .cloned()
                .collect::<Vec<_>>();
            let matching = if candidate_mode.components_only() {
                MatchScope::Components
            } else {
                MatchScope::WholeModule
            };
            native.configure_candidate_enumeration(definitions, matching)?;
        }
        let engine_id = compiler_engine_id();
        let engine_version = compiler_engine_version(native.domain_hash(), candidate_mode);
        let action_set_hash = action_set_hash(native.action_set_hash(), candidate_mode);
        let root = native.insert_root(config.root, config.workload)?;

        Ok(Self {
            native,
            root,
            engine_id,
            engine_version,
            action_set_hash,
            measure_config_hash: config.measure_config_hash,
            measurement_policy: config.measurement_policy,
            candidate_mode,
            candidate_groups: HashMap::new(),
        })
    }

    #[must_use]
    pub const fn candidate_mode(&self) -> CompilerCandidateMode {
        self.candidate_mode
    }

    #[must_use]
    pub const fn measure_config_hash(&self) -> MeasureConfigHash {
        self.measure_config_hash
    }

    pub fn measure_options(&self, samples: u32) -> Result<MeasureOptions, MeasureOptionsError> {
        MeasureOptions::new(self.measure_config_hash, samples, None, true)
    }

    #[must_use]
    pub const fn performance_warmup_iterations(&self) -> u32 {
        self.native.performance_warmup_iterations()
    }

    /// Completes and compiles a retained graph without running the evaluator.
    pub fn prepare_measurement(
        &mut self,
        graph: CompilerStateId,
        options: MeasureOptions,
    ) -> EngineResult<()> {
        self.validate_measure_options(options)?;
        self.native.prepare_measurement(graph).map_err(native_error)
    }

    /// Measures a retained graph with an explicit warmup count.
    pub fn measure_with_warmup(
        &mut self,
        graph: CompilerStateId,
        options: MeasureOptions,
        warmup_iterations: u32,
    ) -> EngineResult<MeasureResult<CompilerStateId>> {
        self.validate_measure_options(options)?;
        let graph_hash = self.hash(graph)?;
        match self.measurement_policy {
            CompilerMeasurementPolicy::VerifyAccuracy => project_measurement(
                graph,
                graph_hash,
                options.config_hash,
                self.native
                    .measure_with_warmup(graph, warmup_iterations, options.samples),
            ),
            CompilerMeasurementPolicy::PerformanceOnly => project_performance_only_measurement(
                graph,
                graph_hash,
                options.config_hash,
                self.native.measure_performance_only_with_warmup(
                    graph,
                    warmup_iterations,
                    options.samples,
                ),
            ),
        }
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
        if self.candidate_mode.uses_candidate_groups() {
            let group =
                self.candidate_groups
                    .get(&candidate)
                    .ok_or(EngineError::UnknownCandidate {
                        candidate_hash: None,
                    })?;
            if group.parent_hash != actual_graph_hash {
                return Err(EngineError::StaleCandidate {
                    expected_graph_hash: group.parent_hash,
                    actual_graph_hash,
                    candidate_hash: group.candidate_hash,
                });
            }
            return Ok(group.candidate_hash);
        }

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

    fn validate_measure_options(&self, options: MeasureOptions) -> EngineResult<()> {
        if options.config_hash != self.measure_config_hash {
            return Err(measure_config_mismatch());
        }
        if options.timeout_ms.is_some() {
            return Err(unsupported_measure_timeout());
        }
        Ok(())
    }

    pub(crate) fn candidate_feature(
        &self,
        graph_hash: GraphHash,
        candidate: CompilerCandidateId,
    ) -> EngineResult<(u32, &[native::matching::BoundObject])> {
        self.validate_candidate_parent(graph_hash, candidate)?;
        match self.candidate_mode {
            CompilerCandidateMode::Sitewise => {
                let candidate = self
                    .native
                    .candidate_view(candidate)
                    .map_err(native_error)?;
                Ok((candidate.kind.index(), &candidate.site.bindings.values))
            }
            CompilerCandidateMode::HybridV1
            | CompilerCandidateMode::HybridProviderV1
            | CompilerCandidateMode::HybridNvidiaV1
            | CompilerCandidateMode::HybridNvidiaV2
            | CompilerCandidateMode::EndToEndV1
            | CompilerCandidateMode::EndToEndNvidiaV1
            | CompilerCandidateMode::EndToEndNvidiaV2
            | CompilerCandidateMode::EndToEndNvidiaV3
            | CompilerCandidateMode::EndToEndNvidiaV4
            | CompilerCandidateMode::EndToEndNvidiaV5 => {
                let group =
                    self.candidate_groups
                        .get(&candidate)
                        .ok_or(EngineError::UnknownCandidate {
                            candidate_hash: None,
                        })?;
                Ok((group.kind.index(), &group.subjects))
            }
        }
    }

    fn hybrid_candidates(
        &mut self,
        graph: CompilerStateId,
        options: CandidateOptions,
        out: &mut Vec<CompilerCandidateId>,
    ) -> EngineResult<()> {
        out.clear();
        let parent_hash = self.hash(graph)?;
        let mut native_candidates = Vec::new();
        self.native
            .candidates(graph, None, &mut native_candidates)
            .map_err(native_error)?;
        let groups = match group_hybrid_candidates(
            &self.native,
            graph,
            parent_hash,
            self.action_set_hash,
            &native_candidates,
            self.candidate_mode,
        ) {
            Ok(groups) => groups,
            Err(error) => {
                self.native
                    .release(&[], &native_candidates)
                    .map_err(native_error)?;
                return Err(error);
            }
        };
        let keep = options
            .max_candidates
            .unwrap_or(groups.len())
            .min(groups.len());
        let mut retained = groups;
        let omitted = retained.split_off(keep);
        let omitted_members = omitted
            .iter()
            .flat_map(|group| group.members.iter().copied())
            .collect::<Vec<_>>();
        self.native
            .release(&[], &omitted_members)
            .map_err(native_error)?;

        out.reserve(retained.len());
        for group in retained {
            let representative = group.representative();
            let previous = self.candidate_groups.insert(representative, group);
            debug_assert!(previous.is_none(), "native candidate handles are unique");
            out.push(representative);
        }
        Ok(())
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
        match self.candidate_mode {
            CompilerCandidateMode::Sitewise => self
                .native
                .candidates(graph, options.max_candidates, out)
                .map_err(native_error),
            CompilerCandidateMode::HybridV1
            | CompilerCandidateMode::HybridProviderV1
            | CompilerCandidateMode::HybridNvidiaV1
            | CompilerCandidateMode::HybridNvidiaV2
            | CompilerCandidateMode::EndToEndV1
            | CompilerCandidateMode::EndToEndNvidiaV1
            | CompilerCandidateMode::EndToEndNvidiaV2
            | CompilerCandidateMode::EndToEndNvidiaV3
            | CompilerCandidateMode::EndToEndNvidiaV4
            | CompilerCandidateMode::EndToEndNvidiaV5 => {
                self.hybrid_candidates(graph, options, out)
            }
        }
    }

    fn candidate_info(
        &self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<CandidateInfo> {
        let graph_hash = self.hash(graph)?;
        let candidate_hash = self.validate_candidate_parent(graph_hash, candidate)?;
        let (kind, display_name, metadata) = match self.candidate_mode {
            CompilerCandidateMode::Sitewise => {
                let candidate = self
                    .native
                    .candidate_view(candidate)
                    .map_err(native_error)?;
                (
                    CandidateKindId::new(candidate.kind.index()),
                    candidate.definition.to_string(),
                    CandidateMetadata::default(),
                )
            }
            CompilerCandidateMode::HybridV1
            | CompilerCandidateMode::HybridProviderV1
            | CompilerCandidateMode::HybridNvidiaV1
            | CompilerCandidateMode::HybridNvidiaV2
            | CompilerCandidateMode::EndToEndV1
            | CompilerCandidateMode::EndToEndNvidiaV1
            | CompilerCandidateMode::EndToEndNvidiaV2
            | CompilerCandidateMode::EndToEndNvidiaV3
            | CompilerCandidateMode::EndToEndNvidiaV4
            | CompilerCandidateMode::EndToEndNvidiaV5 => {
                let group =
                    self.candidate_groups
                        .get(&candidate)
                        .ok_or(EngineError::UnknownCandidate {
                            candidate_hash: None,
                        })?;
                (
                    CandidateKindId::new(group.kind.index()),
                    group.display_name.to_string(),
                    CandidateMetadata {
                        bytes: (group.members.len() as u64).to_le_bytes().to_vec(),
                    },
                )
            }
        };

        Ok(CandidateInfo {
            candidate_hash,
            graph_hash,
            action_set_hash: self.action_set_hash,
            kind,
            display_name,
            static_prior: 0.0,
            tags: CandidateTags::EMPTY,
            subjects: Vec::new(),
            metadata,
        })
    }

    fn apply(
        &mut self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<ApplyResult<Self::Graph, Self::Candidate>> {
        let before_hash = self.hash(graph)?;
        let candidate_hash = self.validate_candidate_parent(before_hash, candidate)?;
        let (after, engine_steps) = match self.candidate_mode {
            CompilerCandidateMode::Sitewise => (
                self.native.apply(graph, candidate).map_err(native_error)?,
                None,
            ),
            CompilerCandidateMode::HybridV1
            | CompilerCandidateMode::HybridProviderV1
            | CompilerCandidateMode::HybridNvidiaV1
            | CompilerCandidateMode::HybridNvidiaV2
            | CompilerCandidateMode::EndToEndV1
            | CompilerCandidateMode::EndToEndNvidiaV1
            | CompilerCandidateMode::EndToEndNvidiaV2
            | CompilerCandidateMode::EndToEndNvidiaV3
            | CompilerCandidateMode::EndToEndNvidiaV4
            | CompilerCandidateMode::EndToEndNvidiaV5 => {
                let members = self
                    .candidate_groups
                    .get(&candidate)
                    .ok_or(EngineError::UnknownCandidate {
                        candidate_hash: None,
                    })?
                    .members
                    .clone();
                (
                    self.native
                        .apply_group(graph, &members)
                        .map_err(native_error)?,
                    Some(members.len() as u64),
                )
            }
        };
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
            metrics: ApplyMetrics {
                elapsed_ms: None,
                engine_steps,
            },
        })
    }

    fn measure(
        &mut self,
        graph: Self::Graph,
        options: MeasureOptions,
    ) -> EngineResult<MeasureResult<Self::Graph>> {
        self.validate_measure_options(options)?;
        let graph_hash = self.hash(graph)?;
        match self.measurement_policy {
            CompilerMeasurementPolicy::VerifyAccuracy => project_measurement(
                graph,
                graph_hash,
                options.config_hash,
                self.native.measure(graph, options.samples),
            ),
            CompilerMeasurementPolicy::PerformanceOnly => project_performance_only_measurement(
                graph,
                graph_hash,
                options.config_hash,
                self.native.measure_performance_only(graph, options.samples),
            ),
        }
    }

    fn release(
        &mut self,
        graphs: &[Self::Graph],
        candidates: &[Self::Candidate],
    ) -> EngineResult<()> {
        if self.candidate_mode == CompilerCandidateMode::Sitewise {
            return self
                .native
                .release(graphs, candidates)
                .map_err(native_error);
        }

        let mut seen = HashSet::with_capacity(candidates.len());
        let mut members = Vec::new();
        for candidate in candidates.iter().copied() {
            if !seen.insert(candidate) {
                return Err(EngineError::UnknownCandidate {
                    candidate_hash: None,
                });
            }
            let group =
                self.candidate_groups
                    .get(&candidate)
                    .ok_or(EngineError::UnknownCandidate {
                        candidate_hash: None,
                    })?;
            members.extend(group.members.iter().copied());
        }
        self.native
            .release(graphs, &members)
            .map_err(native_error)?;
        for candidate in candidates {
            self.candidate_groups.remove(candidate);
        }
        Ok(())
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
