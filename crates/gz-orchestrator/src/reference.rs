use gz_engine::{
    CandidateOptions, EngineError, EngineResult, ErrorCode, ErrorMessage, GraphEngine,
    MeasureOptions, ModelVersion, PortableGraphId, ReplayGraphContext, SearchConfigHash,
};
use gz_features::{FeatureExtractor, FeatureRow, OpponentStateFeatures, PositionFeatures};
use gz_replay::ReplayReferenceKind;
use gz_search::{BeamSearch, GreedySearch, RandomSearch, SearchStep};

pub trait ReferenceProvider<E: GraphEngine> {
    fn reference(&mut self, engine: &mut E, root: E::Graph) -> EngineResult<Option<Reference>>;

    fn reference_with_features<X>(
        &mut self,
        engine: &mut E,
        root: E::Graph,
        extractor: &mut X,
        candidate_options: CandidateOptions,
        export_position: bool,
    ) -> EngineResult<Option<Reference>>
    where
        Self: Sized,
        X: FeatureExtractor<E>,
    {
        let _ = (extractor, candidate_options, export_position);
        self.reference(engine, root)
    }

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

    /// Whether episodes are expected to carry a reference once this
    /// provider is warmed up. When true, episodes that completed before
    /// the first reference existed (the pre-rollout admission wave) are
    /// dropped instead of stored as unlabeled rows: the store then only
    /// ever contains labeled, on-distribution training data.
    fn expects_reference(&self) -> bool {
        true
    }
}

/// The measured result of an opponent rollout episode.
#[derive(Clone, Debug, PartialEq)]
pub struct RolloutOutcome {
    pub final_reward: f32,
    pub final_graph: ReplayGraphContext,
    pub steps: Vec<ReferenceStep>,
    pub search_config_hash: SearchConfigHash,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reference {
    pub kind: ReplayReferenceKind,
    pub final_reward: f32,
    pub final_graph: Option<ReplayGraphContext>,
    pub steps: Vec<ReferenceStep>,
    pub search_config_hash: Option<SearchConfigHash>,
    pub model_version: Option<ModelVersion>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceStep {
    pub context: ReplayGraphContext,
    pub features: Option<OpponentStateFeatures>,
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
    fn reference(&mut self, engine: &mut E, root: E::Graph) -> EngineResult<Option<Reference>> {
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
                context: final_graph,
                features: None,
            }],
            search_config_hash: None,
            model_version: None,
        }))
    }

    fn reference_with_features<X>(
        &mut self,
        engine: &mut E,
        root: E::Graph,
        extractor: &mut X,
        candidate_options: CandidateOptions,
        export_position: bool,
    ) -> EngineResult<Option<Reference>>
    where
        X: FeatureExtractor<E>,
    {
        let measure = engine.measure(root, self.measure_options)?;
        let Some(final_reward) = score(measure.measured, measure.valid, measure.scalar_reward)
        else {
            return Ok(None);
        };
        let final_graph = context(engine, measure.graph_hash);
        let mut created_candidates = Vec::new();
        let step = feature_reference_step(
            engine,
            extractor,
            root,
            final_graph,
            candidate_options,
            ReferenceFeatureContext {
                index: 0,
                final_reward,
                export_position,
            },
            &mut created_candidates,
        );
        let release = engine.release(&[], &created_candidates);
        let step = step?;
        release?;

        Ok(Some(Reference {
            kind: ReplayReferenceKind::RootBaseline,
            final_reward,
            final_graph: Some(final_graph),
            steps: vec![step],
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
    fn reference(&mut self, engine: &mut E, root: E::Graph) -> EngineResult<Option<Reference>> {
        let episode = self.search.run(engine, root)?;
        let reference = project_search_episode(
            ReplayReferenceKind::Greedy,
            episode.final_context,
            &episode.steps,
            score(
                episode.final_measure.measured,
                episode.final_measure.valid,
                episode.final_measure.scalar_reward,
            ),
            Some(episode.search_config_hash),
        );
        engine.release(&episode.created_graphs, &episode.created_candidates)?;
        Ok(reference)
    }

    fn reference_with_features<X>(
        &mut self,
        engine: &mut E,
        root: E::Graph,
        extractor: &mut X,
        _candidate_options: CandidateOptions,
        export_position: bool,
    ) -> EngineResult<Option<Reference>>
    where
        X: FeatureExtractor<E>,
    {
        let episode = self.search.run(engine, root)?;
        let reference = project_search_episode_with_features(
            engine,
            extractor,
            SearchReferenceProjection {
                kind: ReplayReferenceKind::Greedy,
                final_graph: episode.final_graph,
                final_context: episode.final_context,
                steps: &episode.steps,
                final_reward: score(
                    episode.final_measure.measured,
                    episode.final_measure.valid,
                    episode.final_measure.scalar_reward,
                ),
                search_config_hash: Some(episode.search_config_hash),
                candidate_options: self.search.config().candidate_options,
                export_position,
            },
        );
        engine.release(&episode.created_graphs, &episode.created_candidates)?;
        reference
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
    fn reference(&mut self, engine: &mut E, root: E::Graph) -> EngineResult<Option<Reference>> {
        let episode = self.search.run(engine, root)?;
        let reference = project_search_episode(
            ReplayReferenceKind::Beam,
            episode.final_context,
            &episode.steps,
            score(
                episode.final_measure.measured,
                episode.final_measure.valid,
                episode.final_measure.scalar_reward,
            ),
            Some(episode.search_config_hash),
        );
        engine.release(&episode.created_graphs, &episode.created_candidates)?;
        Ok(reference)
    }

    fn reference_with_features<X>(
        &mut self,
        engine: &mut E,
        root: E::Graph,
        extractor: &mut X,
        _candidate_options: CandidateOptions,
        export_position: bool,
    ) -> EngineResult<Option<Reference>>
    where
        X: FeatureExtractor<E>,
    {
        let episode = self.search.run(engine, root)?;
        let reference = project_search_episode_with_features(
            engine,
            extractor,
            SearchReferenceProjection {
                kind: ReplayReferenceKind::Beam,
                final_graph: episode.final_graph,
                final_context: episode.final_context,
                steps: &episode.steps,
                final_reward: score(
                    episode.final_measure.measured,
                    episode.final_measure.valid,
                    episode.final_measure.scalar_reward,
                ),
                search_config_hash: Some(episode.search_config_hash),
                candidate_options: self.search.config().candidate_options,
                export_position,
            },
        );
        engine.release(&episode.created_graphs, &episode.created_candidates)?;
        reference
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
    fn reference(&mut self, engine: &mut E, root: E::Graph) -> EngineResult<Option<Reference>> {
        let episode = self.search.run(engine, root)?;
        let reference = project_search_episode(
            ReplayReferenceKind::Random,
            episode.final_context,
            &episode.steps,
            score(
                episode.final_measure.measured,
                episode.final_measure.valid,
                episode.final_measure.scalar_reward,
            ),
            Some(episode.search_config_hash),
        );
        engine.release(&episode.created_graphs, &episode.created_candidates)?;
        Ok(reference)
    }

    fn reference_with_features<X>(
        &mut self,
        engine: &mut E,
        root: E::Graph,
        extractor: &mut X,
        _candidate_options: CandidateOptions,
        export_position: bool,
    ) -> EngineResult<Option<Reference>>
    where
        X: FeatureExtractor<E>,
    {
        let episode = self.search.run(engine, root)?;
        let reference = project_search_episode_with_features(
            engine,
            extractor,
            SearchReferenceProjection {
                kind: ReplayReferenceKind::Random,
                final_graph: episode.final_graph,
                final_context: episode.final_context,
                steps: &episode.steps,
                final_reward: score(
                    episode.final_measure.measured,
                    episode.final_measure.valid,
                    episode.final_measure.scalar_reward,
                ),
                search_config_hash: Some(episode.search_config_hash),
                candidate_options: self.search.config().candidate_options,
                export_position,
            },
        );
        engine.release(&episode.created_graphs, &episode.created_candidates)?;
        reference
    }
}

fn project_search_episode<G, C>(
    kind: ReplayReferenceKind,
    final_graph: ReplayGraphContext,
    steps: &[SearchStep<G, C>],
    final_reward: Option<f32>,
    search_config_hash: Option<SearchConfigHash>,
) -> Option<Reference> {
    let final_reward = final_reward?;
    let mut reference_steps = Vec::with_capacity(steps.len() + 1);

    match steps.first() {
        Some(step) => reference_steps.push(ReferenceStep {
            context: step.step_ref.before,
            features: None,
        }),
        None => reference_steps.push(ReferenceStep {
            context: final_graph,
            features: None,
        }),
    }

    reference_steps.extend(steps.iter().map(|step| ReferenceStep {
        context: step.step_ref.after,
        features: None,
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

struct SearchReferenceProjection<'a, E: GraphEngine> {
    kind: ReplayReferenceKind,
    final_graph: E::Graph,
    final_context: ReplayGraphContext,
    steps: &'a [SearchStep<E::Graph, E::Candidate>],
    final_reward: Option<f32>,
    search_config_hash: Option<SearchConfigHash>,
    candidate_options: CandidateOptions,
    export_position: bool,
}

fn project_search_episode_with_features<E, X>(
    engine: &mut E,
    extractor: &mut X,
    projection: SearchReferenceProjection<'_, E>,
) -> EngineResult<Option<Reference>>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let Some(final_reward) = projection.final_reward else {
        return Ok(None);
    };
    let mut feature_candidates = Vec::new();
    let mut reference_steps = Vec::with_capacity(projection.steps.len() + 1);

    match projection.steps.first() {
        Some(step) => reference_steps.push(feature_reference_step(
            engine,
            extractor,
            step.before,
            step.step_ref.before,
            projection.candidate_options,
            ReferenceFeatureContext {
                index: 0,
                final_reward,
                export_position: projection.export_position,
            },
            &mut feature_candidates,
        )),
        None => reference_steps.push(feature_reference_step(
            engine,
            extractor,
            projection.final_graph,
            projection.final_context,
            projection.candidate_options,
            ReferenceFeatureContext {
                index: 0,
                final_reward,
                export_position: projection.export_position,
            },
            &mut feature_candidates,
        )),
    }

    for (index, step) in projection.steps.iter().enumerate() {
        reference_steps.push(feature_reference_step(
            engine,
            extractor,
            step.after,
            step.step_ref.after,
            projection.candidate_options,
            ReferenceFeatureContext {
                index: index + 1,
                final_reward,
                export_position: projection.export_position,
            },
            &mut feature_candidates,
        ));
    }

    let release = engine.release(&[], &feature_candidates);
    let mut steps = Vec::with_capacity(reference_steps.len());
    for step in reference_steps {
        steps.push(step?);
    }
    release?;

    Ok(Some(Reference {
        kind: projection.kind,
        final_reward,
        final_graph: Some(projection.final_context),
        steps,
        search_config_hash: projection.search_config_hash,
        model_version: None,
    }))
}

#[derive(Clone, Copy)]
struct ReferenceFeatureContext {
    index: usize,
    final_reward: f32,
    export_position: bool,
}

fn feature_reference_step<E, X>(
    engine: &mut E,
    extractor: &mut X,
    graph: E::Graph,
    context: ReplayGraphContext,
    candidate_options: CandidateOptions,
    feature_context: ReferenceFeatureContext,
    created_candidates: &mut Vec<E::Candidate>,
) -> EngineResult<ReferenceStep>
where
    E: GraphEngine,
    X: FeatureExtractor<E>,
{
    let mut candidates = Vec::new();
    engine.candidates(graph, candidate_options, &mut candidates)?;
    created_candidates.extend(candidates.iter().copied());
    let scale = extractor.schema().config().opponent_reward_scale;
    let (root_step, budget_fraction, budget_step) = if feature_context.export_position {
        (
            u32::try_from(feature_context.index).map_err(|_| internal("root step overflow"))?,
            0.0,
            0.0,
        )
    } else {
        (0, 0.0, 0.0)
    };
    let row = extractor
        .extract(
            engine,
            graph,
            &candidates,
            PositionFeatures {
                root_step,
                leaf_depth: 0,
                budget_fraction,
                budget_step,
                opponent_reward: feature_context.final_reward / scale,
                opponent_present: true,
            },
        )
        .map_err(|_| internal("reference feature extraction failed"))?;

    Ok(ReferenceStep {
        context,
        features: Some(opponent_state(row)),
    })
}

fn opponent_state(row: FeatureRow) -> OpponentStateFeatures {
    OpponentStateFeatures {
        node_count: row.node_count,
        node_tokens: row.node_tokens,
        node_attrs: row.node_attrs,
        edges: row.edges,
        position: row.position,
    }
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
    fn reference(&mut self, _engine: &mut E, _root: E::Graph) -> EngineResult<Option<Reference>> {
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
    gate: PolicyGate,
    current: Option<PolicyReference>,
    /// The newest measured rollout regardless of gate verdict, for the
    /// gamma mix: whittlezero plays 1-gamma of its episodes against the
    /// gated best and gamma against the current checkpoint, so the label
    /// channel stays winnable when the bar outruns the noisy player.
    latest: Option<PolicyReference>,
    gamma: f32,
    mix_seed: u64,
    draws: u64,
    last_challenged: Option<ModelVersion>,
    pending_version: Option<ModelVersion>,
}

/// How a finished challenger rollout updates the reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyGate {
    /// Every measured rollout replaces the reference: the bar tracks the
    /// newest checkpoint, up or down.
    Latest,
    /// whittlezero's arena gate on the fixed root: a challenger is
    /// accepted only if it strictly beats the incumbent, so the bar is
    /// monotone and rows attribute the incumbent's version. Every
    /// checkpoint is still challenged exactly once.
    Best,
}

#[derive(Clone)]
struct PolicyReference {
    reward: f32,
    version: ModelVersion,
    final_graph: ReplayGraphContext,
    steps: Vec<ReferenceStep>,
    search_config_hash: SearchConfigHash,
}

impl PolicyReferenceProvider {
    #[must_use]
    pub const fn new() -> Self {
        Self::with_gate(PolicyGate::Latest)
    }

    #[must_use]
    pub const fn gated() -> Self {
        Self::with_gate(PolicyGate::Best)
    }

    /// The gated provider with whittlezero's gamma mix: each episode's
    /// reference is the latest measured rollout with probability `gamma`
    /// (seeded, per-provider draw sequence) and the gated incumbent
    /// otherwise.
    #[must_use]
    pub const fn gated_with_gamma(gamma: f32, mix_seed: u64) -> Self {
        let mut provider = Self::with_gate(PolicyGate::Best);
        provider.gamma = gamma;
        provider.mix_seed = mix_seed;
        provider
    }

    const fn with_gate(gate: PolicyGate) -> Self {
        Self {
            gate,
            current: None,
            latest: None,
            gamma: 0.0,
            mix_seed: 0,
            draws: 0,
            last_challenged: None,
            pending_version: None,
        }
    }

    fn mix_unit(&mut self) -> f32 {
        self.draws += 1;
        let mut value = self.mix_seed ^ self.draws.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        (value >> 40) as f32 / (1u64 << 24) as f32
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
    fn reference(&mut self, _engine: &mut E, _root: E::Graph) -> EngineResult<Option<Reference>> {
        let use_latest = self.gamma > 0.0 && self.latest.is_some() && self.mix_unit() < self.gamma;
        let chosen = if use_latest {
            self.latest.as_ref()
        } else {
            self.current.as_ref()
        };
        let Some(current) = chosen else {
            return Ok(None);
        };

        Ok(Some(Reference {
            kind: match self.gate {
                PolicyGate::Latest => ReplayReferenceKind::Gumbel,
                PolicyGate::Best => ReplayReferenceKind::GatedPolicy,
            },
            final_reward: current.reward,
            final_graph: Some(current.final_graph),
            steps: current.steps.clone(),
            search_config_hash: Some(current.search_config_hash),
            model_version: Some(current.version),
        }))
    }

    fn rollout_due(&self, latest: Option<ModelVersion>) -> bool {
        if self.pending_version.is_some() {
            return false;
        }
        // Dueness anchors on the last MEASURED challenge, not the
        // incumbent: gated rejections must not retry, unmeasured rollouts
        // must. Latest mode is unchanged by this anchor (it accepts every
        // measured challenge, so last_challenged tracks current.version).
        match latest {
            Some(latest) => self.last_challenged != Some(latest),
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
        // Unmeasured challengers retry: last_challenged stays put.
        let Some(outcome) = outcome else {
            return;
        };
        self.last_challenged = Some(version);
        let accepted = match self.gate {
            PolicyGate::Latest => true,
            PolicyGate::Best => self
                .current
                .as_ref()
                .is_none_or(|incumbent| outcome.final_reward > incumbent.reward),
        };
        if self.gate == PolicyGate::Best {
            eprintln!(
                "event=policy_gate accepted={accepted} challenger={} best={} version={version}",
                outcome.final_reward,
                self.current
                    .as_ref()
                    .map_or(outcome.final_reward, |incumbent| incumbent.reward),
            );
        }
        let challenger = PolicyReference {
            reward: outcome.final_reward,
            version,
            final_graph: outcome.final_graph,
            steps: outcome.steps,
            search_config_hash: outcome.search_config_hash,
        };
        self.latest = Some(challenger.clone());
        if accepted {
            self.current = Some(challenger);
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

fn internal(message: &'static str) -> EngineError {
    EngineError::Internal {
        code: ErrorCode::new(9_001),
        message: ErrorMessage::new(message).expect("static error message fits"),
    }
}
