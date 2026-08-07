use crate::{
    CompilerCandidateId, CompilerStateId, NativeCompilerEngine, candidate_hash,
    error::{NATIVE_ENGINE_ERROR_CODE, internal_error, native_error},
    identity::batch_candidate_hash,
};
use graphs_compiler::{
    actions::{CompilerActionKindId, CompilerCandidateBatchPolicy, CompilerStateView},
    ir::{NodeId, OperandSlot, ResultSlot, ScopeId, ValueId},
    matching::{BoundObject, MatchSite},
};
use gz_engine::{ActionSetHash, CandidateHash, EngineResult, GraphHash};
use std::collections::{HashMap, HashSet};

/// Candidate granularity exposed by the GraphZero compiler adapter.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum CompilerCandidateMode {
    /// Expose every native compiler action as a separate search action.
    #[default]
    Sitewise,
    /// Batch audited canonicalization and backend actions globally, and batch
    /// equivalent attention choices by operation signature.
    ///
    /// The version suffix is part of the semantic contract: changing this
    /// policy requires a new variant so replay identities cannot be mixed.
    HybridV1,
    /// Expose only NVIDIA provider-completion actions, using the HybridV1
    /// batching policy.
    ///
    /// This keeps every reachable state within the provider completion
    /// closure while target-independent optimization actions are not yet
    /// guaranteed composable in arbitrary search order.
    HybridProviderV1,
    /// Expose NVIDIA provider-completion actions plus a frozen set of
    /// target-independent optimizations audited for terminal NVIDIA
    /// compilation under arbitrary search ordering.
    ///
    /// Unlike [`Self::HybridV1`], this excludes catalog transforms that can
    /// produce IR outside NVIDIA's provider-completion closure.
    HybridNvidiaV1,
    /// Use the [`Self::HybridNvidiaV1`] action set while batching every
    /// decomposed-RoPE canonicalization site into one deterministic action.
    HybridNvidiaV2,
    /// Expose every optional optimization and NVIDIA provider-selection action.
    ///
    /// This is the end-to-end compiler-search contract. Equivalent optional
    /// fusions share a decision, while deterministic cleanup and decomposed
    /// RoPE canonicalization run as whole-graph batches. Changing either
    /// eligibility or grouping requires a new version.
    EndToEndV1,
    /// Expose NVIDIA provider actions plus the frozen NVIDIA default
    /// optimization surface that remains within provider-completion closure.
    /// Uses the same batching policy as [`Self::EndToEndV1`].
    EndToEndNvidiaV1,
    /// Expose the frozen NVIDIA optimization surface while leaving mandatory
    /// provider selection to terminal compiler completion.
    ///
    /// Native provider rewrites are order-sensitive completion steps, not
    /// freely composable search actions. They require a separate candidate
    /// contract once the native engine can expose conflict-safe batches.
    EndToEndNvidiaV2,
    /// Expose the full NVIDIA provider and optimization surface while using
    /// native batching policies to keep every grouped action conflict-safe.
    /// Terminal compilation cleans arbitrary search prefixes before provider
    /// completion.
    EndToEndNvidiaV3,
    /// Expose only optional NVIDIA-compatible optimizations. Mandatory terminal
    /// cleanup and provider selection remain compiler-owned completion phases.
    EndToEndNvidiaV4,
    /// Expose optional NVIDIA-compatible optimizations and compact provider
    /// assignments, while excluding the combinatorial pointwise-fusion family.
    /// Equivalent provider sites are batched by semantic signature.
    EndToEndNvidiaV5,
}

impl CompilerCandidateMode {
    pub(crate) const fn identity_tag(self) -> u8 {
        match self {
            Self::Sitewise => 0,
            Self::HybridV1 => 1,
            Self::HybridProviderV1 => 2,
            Self::HybridNvidiaV1 => 3,
            Self::HybridNvidiaV2 => 4,
            Self::EndToEndV1 => 5,
            Self::EndToEndNvidiaV1 => 6,
            Self::EndToEndNvidiaV2 => 7,
            Self::EndToEndNvidiaV3 => 8,
            Self::EndToEndNvidiaV4 => 9,
            Self::EndToEndNvidiaV5 => 10,
        }
    }

    pub(crate) const fn uses_candidate_groups(self) -> bool {
        !matches!(self, Self::Sitewise)
    }

    pub(crate) fn allows_definition(self, definition: &str) -> bool {
        match self {
            Self::Sitewise | Self::HybridV1 | Self::EndToEndV1 => true,
            Self::HybridProviderV1 => definition.starts_with("nvidia."),
            Self::HybridNvidiaV1 | Self::HybridNvidiaV2 => {
                definition.starts_with("nvidia.")
                    || NVIDIA_COMPATIBLE_OPTIMIZATION_IDS_V1.contains(&definition)
            }
            Self::EndToEndNvidiaV1 => {
                definition.starts_with("nvidia.")
                    || NVIDIA_END_TO_END_OPTIMIZATION_IDS_V1.contains(&definition)
            }
            Self::EndToEndNvidiaV2 => NVIDIA_END_TO_END_OPTIMIZATION_IDS_V1.contains(&definition),
            Self::EndToEndNvidiaV3 => {
                definition.starts_with("nvidia.")
                    || NVIDIA_END_TO_END_OPTIMIZATION_IDS_V1.contains(&definition)
            }
            Self::EndToEndNvidiaV4 => {
                NVIDIA_END_TO_END_OPTIMIZATION_IDS_V1.contains(&definition)
                    && !END_TO_END_GLOBAL_CLEANUP_IDS_V1.contains(&definition)
            }
            Self::EndToEndNvidiaV5 => {
                (definition.starts_with("nvidia.")
                    && definition != "nvidia.native_cuda.pointwise.fuse_elementwise_pair")
                    || (NVIDIA_END_TO_END_OPTIMIZATION_IDS_V1.contains(&definition)
                        && !END_TO_END_GLOBAL_CLEANUP_IDS_V1.contains(&definition))
            }
        }
    }

    pub(crate) const fn components_only(self) -> bool {
        matches!(self, Self::EndToEndNvidiaV3 | Self::EndToEndNvidiaV5)
    }
}

// Frozen as part of HybridNvidiaV1's replay identity. Adding another audited
// optimization requires a new candidate-mode version so replay identities do
// not silently change.
const NVIDIA_COMPATIBLE_OPTIMIZATION_IDS_V1: &[&str] = &[
    "graph.fuse_matmul_activation",
    "algorithm.fuse_decomposed_rope",
];

// Frozen as part of EndToEndNvidiaV1's replay identity. Standalone SwiGLU is
// omitted because NVIDIA provider completion cannot lower execution.swiglu.
const NVIDIA_END_TO_END_OPTIMIZATION_IDS_V1: &[&str] = &[
    "graph.fuse_matmul_activation",
    "graph.canonicalize_linear_matmul_affine",
    "graph.fuse_matmul_bias_affine",
    "algorithm.fuse_rmsnorm.self_multiply",
    "algorithm.fuse_rmsnorm.self_multiply_leading_cast",
    "algorithm.fuse_rmsnorm.self_multiply_trailing_cast",
    "algorithm.fuse_rmsnorm.self_multiply_leading_trailing_cast",
    "algorithm.fuse_rmsnorm.power_two",
    "algorithm.fuse_rmsnorm.power_two_leading_cast",
    "algorithm.fuse_rmsnorm.power_two_trailing_cast",
    "algorithm.fuse_rmsnorm.power_two_leading_trailing_cast",
    "algorithm.fuse_gemma_rmsnorm.self_multiply",
    "algorithm.fuse_gemma_rmsnorm.self_multiply_leading_cast",
    "algorithm.fuse_gemma_rmsnorm.self_multiply_trailing_cast",
    "algorithm.fuse_gemma_rmsnorm.self_multiply_leading_trailing_cast",
    "graph.fuse_residual_add_rmsnorm",
    "graph.fuse_gemma_weight_rmsnorm",
    "algorithm.fuse_decomposed_rope",
    "graph.drop_empty_concat_inputs",
    "graph.alias_single_input_concat",
    "view.bypass_noop_materialize_copy",
    "view.prove_materialized_view",
    "view.bypass_identity_alias",
    "graph.remove_dead_pure_leaf",
];

// Frozen as part of EndToEndV1's batching contract. These are optional,
// potentially competing fusions, so only sites with the same semantic
// signature share one search decision.
const END_TO_END_EQUIVALENT_FUSION_IDS_V1: &[&str] = &[
    "graph.fuse_matmul_activation",
    "graph.canonicalize_linear_matmul_affine",
    "graph.fuse_matmul_bias_affine",
    "algorithm.fuse_rmsnorm.self_multiply",
    "algorithm.fuse_rmsnorm.self_multiply_leading_cast",
    "algorithm.fuse_rmsnorm.self_multiply_trailing_cast",
    "algorithm.fuse_rmsnorm.self_multiply_leading_trailing_cast",
    "algorithm.fuse_rmsnorm.power_two",
    "algorithm.fuse_rmsnorm.power_two_leading_cast",
    "algorithm.fuse_rmsnorm.power_two_trailing_cast",
    "algorithm.fuse_rmsnorm.power_two_leading_trailing_cast",
    "algorithm.fuse_gemma_rmsnorm.self_multiply",
    "algorithm.fuse_gemma_rmsnorm.self_multiply_leading_cast",
    "algorithm.fuse_gemma_rmsnorm.self_multiply_trailing_cast",
    "algorithm.fuse_gemma_rmsnorm.self_multiply_leading_trailing_cast",
    "graph.fuse_residual_add_rmsnorm",
    "graph.fuse_gemma_weight_rmsnorm",
    "algorithm.fuse_silu_mul_swiglu",
];

// Frozen as part of EndToEndV1's batching contract. Every currently matched
// site for these canonicalizations is one deterministic cleanup pass.
const END_TO_END_GLOBAL_CLEANUP_IDS_V1: &[&str] = &[
    "graph.drop_empty_concat_inputs",
    "graph.alias_single_input_concat",
    "view.bypass_noop_materialize_copy",
    "view.prove_materialized_view",
    "view.bypass_identity_alias",
    "graph.remove_dead_pure_leaf",
];

pub(crate) struct CompilerCandidateGroup {
    pub(crate) parent_hash: GraphHash,
    pub(crate) candidate_hash: CandidateHash,
    pub(crate) kind: CompilerActionKindId,
    pub(crate) display_name: Box<str>,
    pub(crate) subjects: Box<[BoundObject]>,
    pub(crate) members: Box<[CompilerCandidateId]>,
}

impl CompilerCandidateGroup {
    pub(crate) fn representative(&self) -> CompilerCandidateId {
        self.members[0]
    }
}

pub(crate) fn group_hybrid_candidates(
    native: &NativeCompilerEngine,
    graph: CompilerStateId,
    parent_hash: GraphHash,
    action_set_hash: ActionSetHash,
    candidates: &[CompilerCandidateId],
    candidate_mode: CompilerCandidateMode,
) -> EngineResult<Vec<CompilerCandidateGroup>> {
    let state = native.state_view(graph).map_err(native_error)?;
    let signatures = SiteSignatures::new(state)?;
    let mut groups = Vec::<PendingGroup>::new();
    let mut grouped = HashMap::<GroupKey, Vec<usize>>::new();

    for candidate in candidates.iter().copied() {
        let view = native.candidate_view(candidate).map_err(native_error)?;
        let grouping = grouping(candidate_mode, view.definition.as_ref());
        let key = match (grouping, view.batch_policy) {
            (_, CompilerCandidateBatchPolicy::Singleton) | (Grouping::Sitewise, _) => None,
            (Grouping::AllSites, _) => Some(GroupKey::AllSites(view.kind)),
            (Grouping::EquivalentSites, _) => Some(GroupKey::EquivalentSites(
                view.kind,
                signatures.for_site(view.site)?,
            )),
        };

        if let Some(key) = key {
            if let Some(index) = grouped.get(&key).and_then(|indexes| {
                indexes.iter().copied().find(|index| {
                    groups[*index].can_add(view.batch_policy, &view.site.bindings.values)
                })
            }) {
                groups[index].add(candidate, &view.site.bindings.values);
                continue;
            }
            grouped.entry(key).or_default().push(groups.len());
        }
        groups.push(PendingGroup {
            kind: view.kind,
            definition: view.definition.to_string().into_boxed_str(),
            subjects: view.site.bindings.values.clone(),
            members: vec![candidate],
            claimed_bindings: view.site.bindings.values.iter().copied().collect(),
        });
    }

    groups
        .into_iter()
        .map(|group| {
            let member_hashes = group
                .members
                .iter()
                .copied()
                .map(|candidate| {
                    native
                        .candidate_hash(candidate)
                        .map(candidate_hash)
                        .map_err(native_error)
                })
                .collect::<EngineResult<Vec<_>>>()?;
            let candidate_hash = if member_hashes.len() == 1 {
                member_hashes[0]
            } else {
                batch_candidate_hash(parent_hash, action_set_hash, &member_hashes)
            };
            let display_name = if group.members.len() == 1 {
                group.definition
            } else {
                format!("{} ×{}", group.definition, group.members.len()).into_boxed_str()
            };
            Ok(CompilerCandidateGroup {
                parent_hash,
                candidate_hash,
                kind: group.kind,
                display_name,
                subjects: group.subjects,
                members: group.members.into_boxed_slice(),
            })
        })
        .collect()
}

struct PendingGroup {
    kind: CompilerActionKindId,
    definition: Box<str>,
    subjects: Box<[BoundObject]>,
    members: Vec<CompilerCandidateId>,
    claimed_bindings: HashSet<BoundObject>,
}

impl PendingGroup {
    fn can_add(
        &self,
        batch_policy: CompilerCandidateBatchPolicy,
        bindings: &[BoundObject],
    ) -> bool {
        match batch_policy {
            CompilerCandidateBatchPolicy::Independent => true,
            CompilerCandidateBatchPolicy::DisjointBindings => bindings
                .iter()
                .all(|binding| !self.claimed_bindings.contains(binding)),
            CompilerCandidateBatchPolicy::Singleton => false,
        }
    }

    fn add(&mut self, candidate: CompilerCandidateId, bindings: &[BoundObject]) {
        self.members.push(candidate);
        self.claimed_bindings.extend(bindings.iter().copied());
    }
}

#[derive(Clone, Copy)]
enum Grouping {
    Sitewise,
    AllSites,
    EquivalentSites,
}

fn grouping(candidate_mode: CompilerCandidateMode, definition: &str) -> Grouping {
    let end_to_end = matches!(
        candidate_mode,
        CompilerCandidateMode::EndToEndV1
            | CompilerCandidateMode::EndToEndNvidiaV1
            | CompilerCandidateMode::EndToEndNvidiaV2
            | CompilerCandidateMode::EndToEndNvidiaV3
            | CompilerCandidateMode::EndToEndNvidiaV4
            | CompilerCandidateMode::EndToEndNvidiaV5
    );
    if end_to_end {
        if END_TO_END_EQUIVALENT_FUSION_IDS_V1.contains(&definition) {
            return Grouping::EquivalentSites;
        }
        if END_TO_END_GLOBAL_CLEANUP_IDS_V1.contains(&definition) {
            return Grouping::AllSites;
        }
    }
    if matches!(candidate_mode, CompilerCandidateMode::EndToEndNvidiaV5)
        && definition.starts_with("nvidia.")
    {
        return Grouping::EquivalentSites;
    }
    if matches!(
        candidate_mode,
        CompilerCandidateMode::HybridNvidiaV2
            | CompilerCandidateMode::EndToEndV1
            | CompilerCandidateMode::EndToEndNvidiaV1
            | CompilerCandidateMode::EndToEndNvidiaV2
            | CompilerCandidateMode::EndToEndNvidiaV3
            | CompilerCandidateMode::EndToEndNvidiaV4
            | CompilerCandidateMode::EndToEndNvidiaV5
    ) && definition == "algorithm.fuse_decomposed_rope"
    {
        return Grouping::AllSites;
    }
    match definition {
        // These transforms bind one operation and make a deterministic,
        // independently applicable rewrite or backend assignment.
        "view.bypass_noop_materialize_copy"
        | "view.canonicalize_identity_view"
        | "nvidia.cublas.matmul.nn"
        | "nvidia.cublas.matmul.nt"
        | "nvidia.cublas.matmul.tn"
        | "nvidia.cublas.matmul.tt"
        | "nvidia.cublaslt.matmul_affine.identity.sm110.heuristic"
        | "nvidia.native_cuda.arange"
        | "nvidia.native_cuda.concat"
        | "nvidia.native_cuda.embedding"
        | "nvidia.native_cuda.materialize_copy"
        | "nvidia.native_cuda.pointwise.cast"
        | "nvidia.native_cuda.pointwise.elementwise"
        | "nvidia.native_cuda.reduce"
        | "nvidia.native_cuda.sdpa.bf16" => Grouping::AllSites,
        // Attention profile selection is a choice, so only equivalent
        // operations selecting the same profile share one search action.
        "algorithm.attention_form" => Grouping::EquivalentSites,
        _ => Grouping::Sitewise,
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum GroupKey {
    AllSites(CompilerActionKindId),
    EquivalentSites(CompilerActionKindId, SiteSignature),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SiteSignature([u8; 32]);

struct SiteSignatures<'a> {
    state: CompilerStateView<'a>,
    nodes: HashMap<NodeId, usize>,
    values: HashMap<ValueId, usize>,
}

impl<'a> SiteSignatures<'a> {
    fn new(state: CompilerStateView<'a>) -> EngineResult<Self> {
        if state.node_attribute_fingerprints.len() != state.canonical_node_order.len()
            || state.value_type_fingerprints.len() != state.canonical_value_order.len()
        {
            return Err(internal_error(
                NATIVE_ENGINE_ERROR_CODE,
                "compiler semantic fingerprint order is inconsistent",
            ));
        }
        Ok(Self {
            nodes: indexes(state.canonical_node_order),
            values: indexes(state.canonical_value_order),
            state,
        })
    }

    fn for_site(&self, site: &MatchSite) -> EngineResult<SiteSignature> {
        let mut hash = blake3::Hasher::new();
        hash.update(b"gz-engine-compiler-equivalent-site-v1\0");
        hash.update(&[match site.scope {
            ScopeId::Component(_) => 0,
            ScopeId::Body(_) => 1,
        }]);
        update_len(&mut hash, site.bindings.values.len());
        for subject in site.bindings.values.iter().copied() {
            match subject {
                BoundObject::Op(node) => self.hash_node(&mut hash, node)?,
                BoundObject::Value(value) => {
                    hash.update(&[1]);
                    hash.update(self.value_fingerprint(value)?);
                }
                BoundObject::Component(_) | BoundObject::Body(_) => {
                    return Err(internal_error(
                        NATIVE_ENGINE_ERROR_CODE,
                        "equivalent-site batching does not support scope bindings",
                    ));
                }
            }
        }
        Ok(SiteSignature(*hash.finalize().as_bytes()))
    }

    fn hash_node(&self, hash: &mut blake3::Hasher, node: NodeId) -> EngineResult<()> {
        let index = self.nodes.get(&node).copied().ok_or_else(|| {
            internal_error(
                NATIVE_ENGINE_ERROR_CODE,
                "compiler action operation is absent from canonical state order",
            )
        })?;
        let graph = self.state.module.view().graph;
        let node = &graph[node];
        hash.update(&[0]);
        update_len(hash, node.op.index());
        hash.update(self.state.node_attribute_fingerprints[index].as_ref());
        update_len(hash, node.operand_slots());
        for slot in 0..node.operand_slots() {
            let values = &node[OperandSlot::new(slot)];
            update_len(hash, values.len());
            for value in values {
                hash.update(self.value_fingerprint(*value)?);
            }
        }
        update_len(hash, node.result_slots());
        for slot in 0..node.result_slots() {
            let values = &node[ResultSlot::new(slot)];
            update_len(hash, values.len());
            for value in values {
                hash.update(self.value_fingerprint(*value)?);
            }
        }
        update_len(hash, node.bodies.len());
        hash.update(&[match node.parent {
            ScopeId::Component(_) => 0,
            ScopeId::Body(_) => 1,
        }]);
        Ok(())
    }

    fn value_fingerprint(&self, value: ValueId) -> EngineResult<&[u8; 32]> {
        self.values
            .get(&value)
            .copied()
            .map(|index| self.state.value_type_fingerprints[index].as_ref())
            .ok_or_else(|| {
                internal_error(
                    NATIVE_ENGINE_ERROR_CODE,
                    "compiler action value is absent from canonical state order",
                )
            })
    }
}

fn indexes<T: Copy + Eq + std::hash::Hash>(values: &[T]) -> HashMap<T, usize> {
    values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (value, index))
        .collect()
}

fn update_len(hash: &mut blake3::Hasher, value: usize) {
    hash.update(&(value as u64).to_le_bytes());
}
