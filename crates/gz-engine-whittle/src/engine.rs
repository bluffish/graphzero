use crate::graph::{
    GraphBody, GraphError, NO_NODE, OpCode, WhittleCandidateId, WhittleGraph, WhittleGraphId,
    compact_graph, deserialize_wav1, serialize_wav1,
};
use crate::rules::{
    RawCandidate, RuleError, apply_graph, category_weight, enumerate_graph, inverse_rule_id,
    rule_name,
};
use gz_engine::{
    ActionSetHash, ApplyMetrics, ApplyResult, BatchGraphEngine, CandidateHash, CandidateInfo,
    CandidateKindId, CandidateMetadata, CandidateOptions, CandidateTags, EngineContractFixture,
    EngineError, EngineId, EngineResult, EngineVersion, ErrorCode, ErrorMessage, GraphArtifact,
    GraphArtifactFormat, GraphEngine, GraphHash, MeasureConfigHash, MeasureMetadata,
    MeasureOptions, MeasureResult, SubjectId,
};
use std::collections::{HashMap, HashSet};
use std::fmt;

pub type WhittleRng = rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct WhittleEngineConfig {
    pub root: WhittleRoot,
    pub include_reverse_constant_folding: bool,
    pub measure_mode: WhittleMeasureMode,
    pub cache_candidates: bool,
    pub cache_transitions: bool,
}

impl Default for WhittleEngineConfig {
    fn default() -> Self {
        Self {
            root: WhittleRoot::Input {
                arity: 1,
                capacity: 16,
                input_index: 0,
            },
            include_reverse_constant_folding: false,
            measure_mode: WhittleMeasureMode::NegativeCost,
            cache_candidates: true,
            cache_transitions: true,
        }
    }
}

#[derive(Clone, Debug)]
pub enum WhittleRoot {
    Input {
        arity: u16,
        capacity: u16,
        input_index: u16,
    },
    Artifact(Vec<u8>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhittleMeasureMode {
    NegativeCost,
}

#[derive(Clone, Debug)]
pub struct WhittleGraphGeneratorConfig {
    pub arity: u16,
    pub capacity: u16,
    pub exception_terms_min: u16,
    pub exception_terms_max: u16,
    pub prewalk_steps_min: u16,
    pub prewalk_steps_max: u16,
}

impl Default for WhittleGraphGeneratorConfig {
    fn default() -> Self {
        Self {
            arity: 6,
            capacity: 256,
            exception_terms_min: 5,
            exception_terms_max: 7,
            prewalk_steps_min: 4,
            prewalk_steps_max: 64,
        }
    }
}

impl WhittleGraphGeneratorConfig {
    pub fn validate(self) -> Result<Self, WhittleGeneratorConfigError> {
        if self.arity == 0 {
            return Err(WhittleGeneratorConfigError::ZeroArity);
        }
        if self.arity > 16 {
            return Err(WhittleGeneratorConfigError::ArityTooLarge {
                max: 16,
                actual: self.arity,
            });
        }
        if self.capacity < self.arity + 1 {
            return Err(WhittleGeneratorConfigError::CapacityTooSmall);
        }
        if self.exception_terms_min > self.exception_terms_max {
            return Err(WhittleGeneratorConfigError::InvalidExceptionTermRange);
        }
        if self.exception_terms_max == 0 {
            return Err(WhittleGeneratorConfigError::ZeroExceptionTerms);
        }
        if self.prewalk_steps_min > self.prewalk_steps_max {
            return Err(WhittleGeneratorConfigError::InvalidPrewalkRange);
        }

        Ok(self)
    }
}

pub struct WhittleGraphGenerator {
    config: WhittleGraphGeneratorConfig,
    rng: WhittleRng,
}

impl WhittleGraphGenerator {
    pub fn from_seed(config: WhittleGraphGeneratorConfig, seed: u64) -> Self {
        use rand::SeedableRng;

        Self {
            config,
            rng: WhittleRng::seed_from_u64(seed),
        }
    }

    pub fn sample_into(
        &mut self,
        engine: &mut WhittleEngine,
    ) -> EngineResult<GeneratedWhittleGraph> {
        let config = self.config.clone().validate().map_err(generator_error)?;
        let seed_body = truth_table_seed(&config, &mut self.rng).map_err(generator_error)?;
        let seed_graph = engine.graphs.insert(seed_body).map_err(internal_graph)?;
        let seed = engine.graph(seed_graph)?;
        let start_cost = seed.cost();
        let mut graph = seed_graph;
        let mut body = seed.body();
        let steps = random_u16_inclusive(
            &mut self.rng,
            config.prewalk_steps_min,
            config.prewalk_steps_max,
        );
        let mut applied = 0;
        let mut last_rule = None;

        for _ in 0..steps {
            let blocked_rule = last_rule.and_then(inverse_rule_id);
            let candidates = enumerate_graph(&body, true).map_err(internal_rule)?;
            let Some(candidate) =
                choose_prewalk_candidate(&candidates, blocked_rule, &mut self.rng)
            else {
                break;
            };

            body = apply_graph(&body, candidate).map_err(internal_rule)?;
            graph = engine.graphs.insert(body.clone()).map_err(internal_graph)?;
            last_rule = Some(candidate.rule_id);
            applied += 1;
        }

        Ok(GeneratedWhittleGraph {
            graph,
            seed_graph,
            prewalk_steps_requested: steps,
            prewalk_steps_applied: applied,
            start_cost,
            final_cost: engine.graph(graph)?.cost(),
        })
    }
}

pub struct GeneratedWhittleGraph {
    pub graph: WhittleGraphId,
    pub seed_graph: WhittleGraphId,
    pub prewalk_steps_requested: u16,
    pub prewalk_steps_applied: u16,
    pub start_cost: u32,
    pub final_cost: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhittleGeneratorConfigError {
    ZeroArity,
    ArityTooLarge { max: u16, actual: u16 },
    CapacityTooSmall,
    InvalidExceptionTermRange,
    ZeroExceptionTerms,
    InvalidPrewalkRange,
    SeedGraphExceedsCapacity,
}

impl fmt::Display for WhittleGeneratorConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroArity => f.write_str("arity must be greater than zero"),
            Self::ArityTooLarge { max, actual } => {
                write!(f, "arity must be <= {max}, got {actual}")
            }
            Self::CapacityTooSmall => f.write_str("capacity must fit inputs and output"),
            Self::InvalidExceptionTermRange => {
                f.write_str("exception term minimum must be <= maximum")
            }
            Self::ZeroExceptionTerms => f.write_str("exception term maximum must be positive"),
            Self::InvalidPrewalkRange => f.write_str("prewalk minimum must be <= maximum"),
            Self::SeedGraphExceedsCapacity => f.write_str("seed graph exceeds capacity"),
        }
    }
}

impl std::error::Error for WhittleGeneratorConfigError {}

pub struct WhittleEngine {
    config: WhittleEngineConfig,
    root: WhittleGraphId,
    graphs: GraphArena,
    candidates: CandidateArena,
    caches: WhittleCaches,
    engine_id: EngineId,
    engine_version: EngineVersion,
    action_set_hash: ActionSetHash,
    measure_config_hash: MeasureConfigHash,
}

impl WhittleEngine {
    pub fn new(config: WhittleEngineConfig) -> EngineResult<Self> {
        let engine_id = engine_id();
        let engine_version = engine_version();
        let action_set_hash =
            action_set_hash(engine_version, config.include_reverse_constant_folding);
        let measure_config_hash = measure_config_hash(config.measure_mode);
        let mut graphs = GraphArena::new(engine_id, engine_version);
        let root_body = match &config.root {
            WhittleRoot::Input {
                arity,
                capacity,
                input_index,
            } => GraphBody::input(*arity, *capacity, *input_index).map_err(internal_graph)?,
            WhittleRoot::Artifact(bytes) => deserialize_wav1(bytes)
                .and_then(|body| compact_graph(&body))
                .map_err(internal_graph)?,
        };
        let root = graphs.insert(root_body).map_err(internal_graph)?;

        Ok(Self {
            config,
            root,
            graphs,
            candidates: CandidateArena::default(),
            caches: WhittleCaches::default(),
            engine_id,
            engine_version,
            action_set_hash,
            measure_config_hash,
        })
    }

    #[must_use]
    pub fn measure_options(&self) -> MeasureOptions {
        MeasureOptions::new(self.measure_config_hash, 1, None, true)
            .expect("static Whittle measure options are valid")
    }

    #[must_use]
    pub const fn measure_config_hash(&self) -> MeasureConfigHash {
        self.measure_config_hash
    }

    pub(crate) fn graph(&self, graph: WhittleGraphId) -> EngineResult<&WhittleGraph> {
        self.graphs
            .get(graph)
            .ok_or(EngineError::UnknownGraph { graph_hash: None })
    }

    fn candidate(&self, candidate: WhittleCandidateId) -> EngineResult<&WhittleCandidate> {
        self.candidates
            .get(candidate)
            .ok_or(EngineError::UnknownCandidate {
                candidate_hash: None,
            })
    }

    fn enumerate_candidate_ids(
        &mut self,
        graph: WhittleGraphId,
    ) -> EngineResult<Vec<WhittleCandidateId>> {
        let graph_hash = self.hash(graph)?;
        let cache_key = (graph_hash, self.action_set_hash);

        if self.config.cache_candidates
            && let Some(cached) = self.caches.candidates.get(&cache_key)
        {
            return Ok(cached.clone());
        }

        let graph_body = self.graph(graph)?.body();
        let raw = enumerate_graph(&graph_body, self.config.include_reverse_constant_folding)
            .map_err(internal_rule)?;
        let ids: Vec<_> = raw
            .into_iter()
            .map(|raw| {
                self.candidates.insert(WhittleCandidate::new(
                    graph,
                    graph_hash,
                    self.action_set_hash,
                    raw,
                ))
            })
            .collect();

        if self.config.cache_candidates {
            self.caches.candidates.insert(cache_key, ids.clone());
        }

        Ok(ids)
    }
}

impl Default for WhittleEngine {
    fn default() -> Self {
        Self::new(WhittleEngineConfig::default()).expect("default Whittle config is valid")
    }
}

impl GraphEngine for WhittleEngine {
    type Graph = WhittleGraphId;
    type Candidate = WhittleCandidateId;

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
        Ok(self.graph(graph)?.hash)
    }

    fn candidates(
        &mut self,
        graph: Self::Graph,
        options: CandidateOptions,
        out: &mut Vec<Self::Candidate>,
    ) -> EngineResult<()> {
        out.clear();
        out.extend(self.enumerate_candidate_ids(graph)?);

        if let Some(max_candidates) = options.max_candidates {
            out.truncate(max_candidates);
        }

        Ok(())
    }

    fn candidate_info(
        &self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<CandidateInfo> {
        let graph_hash = self.hash(graph)?;
        let candidate = self.candidate(candidate)?;

        if candidate.graph != graph || candidate.graph_hash != graph_hash {
            return Err(EngineError::StaleCandidate {
                expected_graph_hash: candidate.graph_hash,
                actual_graph_hash: graph_hash,
                candidate_hash: candidate.candidate_hash,
            });
        }

        Ok(CandidateInfo {
            candidate_hash: candidate.candidate_hash,
            graph_hash,
            action_set_hash: self.action_set_hash,
            kind: CandidateKindId::new(candidate.rule_id.into()),
            display_name: format!("{}@{}", rule_name(candidate.rule_id), candidate.root),
            static_prior: 0.0,
            tags: CandidateTags::EMPTY,
            subjects: candidate
                .matched_slice()
                .iter()
                .copied()
                .map(u64::from)
                .map(SubjectId::new)
                .collect(),
            metadata: CandidateMetadata {
                bytes: candidate_metadata(candidate),
            },
        })
    }

    fn apply(
        &mut self,
        graph: Self::Graph,
        candidate: Self::Candidate,
    ) -> EngineResult<ApplyResult<Self::Graph, Self::Candidate>> {
        let before_hash = self.hash(graph)?;
        let candidate_body = *self.candidate(candidate)?;

        if candidate_body.graph_hash != before_hash {
            return Err(EngineError::StaleCandidate {
                expected_graph_hash: candidate_body.graph_hash,
                actual_graph_hash: before_hash,
                candidate_hash: candidate_body.candidate_hash,
            });
        }

        let transition_key = (
            before_hash,
            self.action_set_hash,
            candidate_body.candidate_hash,
        );

        let after = if self.config.cache_transitions {
            self.caches.transitions.get(&transition_key).copied()
        } else {
            None
        };

        let after = match after {
            Some(after) => after,
            None => {
                let before = self.graph(graph)?.body();
                let after_body = apply_graph(&before, candidate_body.raw).map_err(internal_rule)?;
                let after = self.graphs.insert(after_body).map_err(internal_graph)?;
                if self.config.cache_transitions {
                    self.caches.transitions.insert(transition_key, after);
                }
                after
            }
        };
        let after_hash = self.hash(after)?;

        Ok(ApplyResult {
            before: graph,
            after,
            before_hash,
            after_hash,
            candidate,
            candidate_hash: candidate_body.candidate_hash,
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
        let graph_body = self.graph(graph)?;
        let cost = graph_body.cost();

        Ok(MeasureResult {
            graph,
            graph_hash: graph_body.hash,
            config_hash: options.config_hash,
            measured: true,
            valid: true,
            latency: None,
            scalar_reward: Some(-(cost as f32)),
            failure: None,
            metadata: MeasureMetadata {
                bytes: measure_metadata(cost, graph_body.arity, graph_body.capacity),
            },
        })
    }

    fn export_graph(&self, graph: Self::Graph) -> EngineResult<GraphArtifact> {
        let graph = self.graph(graph)?;

        Ok(GraphArtifact {
            graph_hash: graph.hash,
            format: GraphArtifactFormat::Binary,
            bytes: graph.canonical.to_vec(),
        })
    }
}

impl BatchGraphEngine for WhittleEngine {}

struct GraphArena {
    items: Vec<WhittleGraph>,
    by_hash: HashMap<GraphHash, WhittleGraphId>,
    engine_id: EngineId,
    engine_version: EngineVersion,
}

impl GraphArena {
    fn new(engine_id: EngineId, engine_version: EngineVersion) -> Self {
        Self {
            items: Vec::new(),
            by_hash: HashMap::new(),
            engine_id,
            engine_version,
        }
    }

    fn insert(&mut self, body: GraphBody) -> Result<WhittleGraphId, GraphError> {
        let compact = compact_graph(&body)?;
        let canonical = serialize_wav1(&compact);
        let hash = graph_hash(self.engine_id, self.engine_version, &canonical);

        if let Some(id) = self.by_hash.get(&hash).copied() {
            return Ok(id);
        }

        let id = WhittleGraphId::from_raw(self.items.len() as u32);
        self.items.push(WhittleGraph {
            arity: compact.arity,
            capacity: compact.capacity,
            output_node: compact.output_node,
            op: compact.op.into_boxed_slice(),
            arg0: compact.arg0.into_boxed_slice(),
            arg1: compact.arg1.into_boxed_slice(),
            canonical: canonical.into_boxed_slice(),
            hash,
        });
        self.by_hash.insert(hash, id);
        Ok(id)
    }

    fn get(&self, id: WhittleGraphId) -> Option<&WhittleGraph> {
        self.items.get(id.raw() as usize)
    }
}

#[derive(Clone, Copy, Debug)]
struct WhittleCandidate {
    graph: WhittleGraphId,
    graph_hash: GraphHash,
    candidate_hash: CandidateHash,
    rule_id: u16,
    root: u32,
    match_len: u8,
    matched: [u32; 8],
    raw: RawCandidate,
}

impl WhittleCandidate {
    fn new(
        graph: WhittleGraphId,
        graph_hash: GraphHash,
        action_set_hash: ActionSetHash,
        raw: RawCandidate,
    ) -> Self {
        Self {
            graph,
            graph_hash,
            candidate_hash: candidate_hash(graph_hash, action_set_hash, raw),
            rule_id: raw.rule_id,
            root: raw.root,
            match_len: raw.match_len,
            matched: raw.matched,
            raw,
        }
    }

    fn matched_slice(&self) -> &[u32] {
        &self.matched[..usize::from(self.match_len)]
    }
}

#[derive(Default)]
struct CandidateArena {
    items: Vec<WhittleCandidate>,
}

impl CandidateArena {
    fn insert(&mut self, candidate: WhittleCandidate) -> WhittleCandidateId {
        let id = WhittleCandidateId::from_raw(self.items.len() as u32);
        self.items.push(candidate);
        id
    }

    fn get(&self, id: WhittleCandidateId) -> Option<&WhittleCandidate> {
        self.items.get(id.raw() as usize)
    }
}

#[derive(Default)]
struct WhittleCaches {
    candidates: HashMap<(GraphHash, ActionSetHash), Vec<WhittleCandidateId>>,
    transitions: HashMap<(GraphHash, ActionSetHash, CandidateHash), WhittleGraphId>,
}

pub struct WhittleContractFixture;

impl EngineContractFixture for WhittleContractFixture {
    type Engine = WhittleEngine;

    fn make_engine(&self) -> Self::Engine {
        WhittleEngine::new(WhittleEngineConfig {
            root: WhittleRoot::Artifact(and_idempotent_artifact()),
            ..WhittleEngineConfig::default()
        })
        .expect("Whittle contract fixture config is valid")
    }

    fn measure_options(&self) -> MeasureOptions {
        self.make_engine().measure_options()
    }

    fn known_path(&self) -> Vec<<Self::Engine as GraphEngine>::Candidate> {
        vec![WhittleCandidateId::from_raw(0)]
    }

    fn unknown_graph(&self) -> Option<<Self::Engine as GraphEngine>::Graph> {
        Some(WhittleGraphId::from_raw(u32::MAX))
    }

    fn unknown_candidate(&self) -> Option<<Self::Engine as GraphEngine>::Candidate> {
        Some(WhittleCandidateId::from_raw(u32::MAX))
    }
}

fn and_idempotent_artifact() -> Vec<u8> {
    serialize_wav1(&GraphBody {
        arity: 1,
        capacity: 16,
        output_node: 2,
        op: vec![OpCode::Input, OpCode::And, OpCode::Output],
        arg0: vec![0, 0, 1],
        arg1: vec![u32::MAX, 0, u32::MAX],
    })
}

fn truth_table_seed(
    config: &WhittleGraphGeneratorConfig,
    rng: &mut WhittleRng,
) -> Result<GraphBody, WhittleGeneratorConfigError> {
    let bit_count = 1u32 << config.arity;
    let max_terms = u32::from(config.exception_terms_max).min(bit_count / 2);
    let min_terms = u32::from(config.exception_terms_min).min(max_terms);
    let term_count = random_u32_inclusive(rng, min_terms, max_terms);
    let selected = sample_assignments(rng, bit_count, term_count);
    let exceptions_are_true = random_bool(rng);

    let mut op = vec![OpCode::Input; config.arity as usize];
    let mut arg0: Vec<_> = (0..u32::from(config.arity)).collect();
    let mut arg1 = vec![NO_NODE; config.arity as usize];
    let mut neg_ref = vec![None; config.arity as usize];
    let mut dnf_root = None;

    for assignment in selected {
        let mut term_root = None;

        for var in 0..u32::from(config.arity) {
            let literal = if ((assignment >> var) & 1) == 1 {
                var
            } else {
                match neg_ref[var as usize] {
                    Some(node) => node,
                    None => {
                        let node =
                            append_node(&mut op, &mut arg0, &mut arg1, OpCode::Not, var, NO_NODE);
                        neg_ref[var as usize] = Some(node);
                        node
                    }
                }
            };

            term_root = Some(match term_root {
                Some(term) => {
                    append_node(&mut op, &mut arg0, &mut arg1, OpCode::And, term, literal)
                }
                None => literal,
            });
        }

        let term = term_root.expect("arity validation guarantees non-empty terms");
        dnf_root = Some(match dnf_root {
            Some(dnf) => append_node(&mut op, &mut arg0, &mut arg1, OpCode::Or, dnf, term),
            None => term,
        });
    }

    let mut root = dnf_root.expect("exception term validation guarantees non-empty DNF");
    if !exceptions_are_true {
        root = append_node(&mut op, &mut arg0, &mut arg1, OpCode::Not, root, NO_NODE);
    }
    let output = append_node(&mut op, &mut arg0, &mut arg1, OpCode::Output, root, NO_NODE);

    if op.len() > usize::from(config.capacity) {
        return Err(WhittleGeneratorConfigError::SeedGraphExceedsCapacity);
    }

    let body = GraphBody::new(config.arity, config.capacity, output, op, arg0, arg1)
        .map_err(|_| WhittleGeneratorConfigError::SeedGraphExceedsCapacity)?;
    compact_graph(&body).map_err(|_| WhittleGeneratorConfigError::SeedGraphExceedsCapacity)
}

fn sample_assignments(rng: &mut WhittleRng, bit_count: u32, term_count: u32) -> Vec<u32> {
    let mut selected = Vec::with_capacity(term_count as usize);
    let mut seen = HashSet::with_capacity(term_count as usize);

    while selected.len() < term_count as usize {
        let assignment = random_u32_exclusive(rng, bit_count);
        if seen.insert(assignment) {
            selected.push(assignment);
        }
    }

    selected
}

fn choose_prewalk_candidate(
    candidates: &[RawCandidate],
    blocked_rule: Option<u16>,
    rng: &mut WhittleRng,
) -> Option<RawCandidate> {
    let total: f64 = candidates
        .iter()
        .filter(|candidate| Some(candidate.rule_id) != blocked_rule)
        .map(|candidate| category_weight(candidate.rule_id))
        .sum();

    if total <= 0.0 {
        return None;
    }

    let mut target = random_f64(rng) * total;

    for candidate in candidates
        .iter()
        .copied()
        .filter(|candidate| Some(candidate.rule_id) != blocked_rule)
    {
        let weight = category_weight(candidate.rule_id);
        if target <= weight {
            return Some(candidate);
        }
        target -= weight;
    }

    candidates
        .iter()
        .copied()
        .rev()
        .find(|candidate| Some(candidate.rule_id) != blocked_rule)
}

fn append_node(
    op: &mut Vec<OpCode>,
    arg0: &mut Vec<u32>,
    arg1: &mut Vec<u32>,
    code: OpCode,
    a: u32,
    b: u32,
) -> u32 {
    op.push(code);
    arg0.push(a);
    arg1.push(b);
    (op.len() - 1) as u32
}

fn random_u16_inclusive(rng: &mut WhittleRng, low: u16, high: u16) -> u16 {
    use rand::RngExt;

    rng.random_range(low..=high)
}

fn random_u32_inclusive(rng: &mut WhittleRng, low: u32, high: u32) -> u32 {
    use rand::RngExt;

    rng.random_range(low..=high)
}

fn random_u32_exclusive(rng: &mut WhittleRng, high: u32) -> u32 {
    use rand::RngExt;

    rng.random_range(0..high)
}

fn random_bool(rng: &mut WhittleRng) -> bool {
    use rand::RngExt;

    rng.random_bool(0.5)
}

fn random_f64(rng: &mut WhittleRng) -> f64 {
    use rand::RngExt;

    rng.random()
}

fn candidate_metadata(candidate: &WhittleCandidate) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(1 + 2 + 4 + 1 + usize::from(candidate.match_len) * 4);
    bytes.push(1);
    bytes.extend_from_slice(&candidate.rule_id.to_le_bytes());
    bytes.extend_from_slice(&candidate.root.to_le_bytes());
    bytes.push(candidate.match_len);
    for node in candidate.matched_slice() {
        bytes.extend_from_slice(&node.to_le_bytes());
    }
    bytes
}

fn measure_metadata(cost: u32, arity: u16, capacity: u16) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(9);
    bytes.push(1);
    bytes.extend_from_slice(&cost.to_le_bytes());
    bytes.extend_from_slice(&arity.to_le_bytes());
    bytes.extend_from_slice(&capacity.to_le_bytes());
    bytes
}

fn engine_id() -> EngineId {
    let bytes = hash32(b"gz-engine-whittle", &[]);
    let mut id = [0; 16];
    id.copy_from_slice(&bytes[..16]);
    EngineId::from_bytes(id)
}

fn engine_version() -> EngineVersion {
    let bytes = hash32(b"whittle-rules-v1", &[&44u16.to_le_bytes(), b"WAV1", &[1]]);
    let mut id = [0; 16];
    id.copy_from_slice(&bytes[..16]);
    EngineVersion::from_bytes(id)
}

fn action_set_hash(
    engine_version: EngineVersion,
    include_reverse_constant_folding: bool,
) -> ActionSetHash {
    ActionSetHash::from_bytes(hash32(
        b"whittle-action-set-v1",
        &[
            engine_version.as_bytes().as_slice(),
            &[u8::from(include_reverse_constant_folding)],
        ],
    ))
}

fn graph_hash(engine_id: EngineId, engine_version: EngineVersion, canonical: &[u8]) -> GraphHash {
    GraphHash::from_bytes(hash32(
        b"whittle-graph-v1",
        &[
            engine_id.as_bytes().as_slice(),
            engine_version.as_bytes().as_slice(),
            canonical,
        ],
    ))
}

fn candidate_hash(
    graph_hash: GraphHash,
    action_set_hash: ActionSetHash,
    candidate: RawCandidate,
) -> CandidateHash {
    let mut fixed = Vec::with_capacity(2 + 4 + 1 + usize::from(candidate.match_len) * 4);
    fixed.extend_from_slice(&candidate.rule_id.to_le_bytes());
    fixed.extend_from_slice(&candidate.root.to_le_bytes());
    fixed.push(candidate.match_len);
    for node in candidate.matched_slice() {
        fixed.extend_from_slice(&node.to_le_bytes());
    }

    CandidateHash::from_bytes(hash32(
        b"whittle-candidate-v1",
        &[
            graph_hash.as_bytes().as_slice(),
            action_set_hash.as_bytes().as_slice(),
            &fixed,
        ],
    ))
}

fn measure_config_hash(mode: WhittleMeasureMode) -> MeasureConfigHash {
    let mode = match mode {
        WhittleMeasureMode::NegativeCost => [0],
    };
    MeasureConfigHash::from_bytes(hash32(b"whittle-measure-config-v1", &[&mode]))
}

fn hash32(domain: &[u8], chunks: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(domain.len() as u64).to_le_bytes());
    hasher.update(domain);

    for chunk in chunks {
        hasher.update(&(chunk.len() as u64).to_le_bytes());
        hasher.update(chunk);
    }

    *hasher.finalize().as_bytes()
}

fn internal_graph(error: GraphError) -> EngineError {
    internal_error(1, error.to_string())
}

fn internal_rule(error: RuleError) -> EngineError {
    internal_error(2, error.to_string())
}

fn generator_error(error: WhittleGeneratorConfigError) -> EngineError {
    internal_error(3, error.to_string())
}

fn internal_error(code: u32, message: impl Into<String>) -> EngineError {
    let message = ErrorMessage::new(message).unwrap_or_else(|_| {
        ErrorMessage::new("whittle internal error").expect("fallback message is valid")
    });
    EngineError::Internal {
        code: ErrorCode::new(code),
        message,
    }
}
