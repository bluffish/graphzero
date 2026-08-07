use crate::GumbelValueMode;
use gz_engine::{CandidateHash, CandidateOptions, MeasureOptions, SearchConfigHash};

pub fn greedy_search_config_hash(
    max_steps: usize,
    candidate_options: CandidateOptions,
    measure_options: MeasureOptions,
) -> SearchConfigHash {
    let mut hasher = blake3::Hasher::new();
    update_chunk(&mut hasher, b"gz-search-greedy-v1");
    update_u64(&mut hasher, max_steps as u64);
    update_candidate_options(&mut hasher, candidate_options);
    update_measure_options(&mut hasher, measure_options);
    SearchConfigHash::from_bytes(*hasher.finalize().as_bytes())
}

pub fn beam_search_config_hash(
    max_depth: usize,
    beam_width: usize,
    candidate_options: CandidateOptions,
    measure_options: MeasureOptions,
) -> SearchConfigHash {
    let mut hasher = blake3::Hasher::new();
    update_chunk(&mut hasher, b"gz-search-beam-v1");
    update_u64(&mut hasher, max_depth as u64);
    update_u64(&mut hasher, beam_width as u64);
    update_candidate_options(&mut hasher, candidate_options);
    update_measure_options(&mut hasher, measure_options);
    SearchConfigHash::from_bytes(*hasher.finalize().as_bytes())
}

pub fn symmetric_selfplay_search_config_hash(base: SearchConfigHash) -> SearchConfigHash {
    let mut hasher = blake3::Hasher::new();
    // v2: retired players preserve their true rewrite count, and corrected
    // reused no-backtrack masks/root considered-set replenishment affect
    // selected actions and policy targets.
    update_chunk(&mut hasher, b"gz-search-gumbel-symmetric-selfplay-v2");
    update_chunk(&mut hasher, base.as_bytes());
    SearchConfigHash::from_bytes(*hasher.finalize().as_bytes())
}

pub fn reducing_uniform_distill_config_hash(
    max_steps: usize,
    export_position: bool,
    candidate_options: CandidateOptions,
    measure_options: MeasureOptions,
) -> SearchConfigHash {
    let mut hasher = blake3::Hasher::new();
    update_chunk(&mut hasher, b"gz-distill-reducing-uniform-v1");
    update_u64(&mut hasher, max_steps as u64);
    update_bool(&mut hasher, export_position);
    update_candidate_options(&mut hasher, candidate_options);
    update_measure_options(&mut hasher, measure_options);
    SearchConfigHash::from_bytes(*hasher.finalize().as_bytes())
}

pub fn trajectory_distill_config_hash(
    max_steps: usize,
    export_position: bool,
    candidate_options: CandidateOptions,
    measure_options: MeasureOptions,
    trajectory: &[CandidateHash],
) -> SearchConfigHash {
    let mut hasher = blake3::Hasher::new();
    update_chunk(&mut hasher, b"gz-distill-measured-trajectory-v1");
    update_u64(&mut hasher, max_steps as u64);
    update_bool(&mut hasher, export_position);
    update_candidate_options(&mut hasher, candidate_options);
    update_measure_options(&mut hasher, measure_options);
    update_u64(&mut hasher, trajectory.len() as u64);
    for candidate in trajectory {
        update_chunk(&mut hasher, candidate.as_bytes());
    }
    SearchConfigHash::from_bytes(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
pub fn gumbel_search_config_hash(
    max_steps: usize,
    simulations: usize,
    max_considered_actions: usize,
    seed: u64,
    gumbel_scale: f32,
    gumbel_noise_overlap: f32,
    c_visit: f32,
    c_scale: f32,
    temperature_moves: usize,
    tree_reuse: bool,
    mask_stop: bool,
    no_backtrack: bool,
    candidate_options: CandidateOptions,
    measure_options: MeasureOptions,
    value_mode: GumbelValueMode,
) -> SearchConfigHash {
    let mut hasher = blake3::Hasher::new();
    // v3: reused roots credit carried visits against the simulation
    // budget (semantics change without a config-shape change).
    // v4: mask_stop joins the config shape.
    // v5: no_backtrack joins the config shape.
    // v6: gumbel_noise_overlap joins the config shape.
    // v7: tree reuse carries cached evals only -- decision ledgers reset
    //     and every move runs the full simulation budget (semantics
    //     change without a config-shape change).
    // v8: tree_reuse=true carries shifted subtree visit/Q ledgers like
    //     policy-based-self-competition; fresh simulations are counted
    //     relative to the carried visit baseline. tree_reuse=false keeps
    //     the v7 namespace because fresh-tree semantics did not change.
    match value_mode {
        GumbelValueMode::SymmetricSelfplay => {
            update_chunk(&mut hasher, b"gz-search-gumbel-mcts-symmetric-selfplay-v2");
        }
        GumbelValueMode::SingleAgent if tree_reuse => {
            update_chunk(&mut hasher, b"gz-search-gumbel-mcts-v8");
        }
        GumbelValueMode::SingleAgent => {
            update_chunk(&mut hasher, b"gz-search-gumbel-mcts-v7");
        }
    }
    update_u64(&mut hasher, max_steps as u64);
    update_u64(&mut hasher, simulations as u64);
    update_u64(&mut hasher, max_considered_actions as u64);
    update_u64(&mut hasher, seed);
    update_u32(&mut hasher, gumbel_scale.to_bits());
    update_u32(&mut hasher, gumbel_noise_overlap.to_bits());
    update_u32(&mut hasher, c_visit.to_bits());
    update_u32(&mut hasher, c_scale.to_bits());
    update_u64(&mut hasher, temperature_moves as u64);
    update_bool(&mut hasher, tree_reuse);
    update_bool(&mut hasher, mask_stop);
    update_bool(&mut hasher, no_backtrack);
    update_candidate_options(&mut hasher, candidate_options);
    update_measure_options(&mut hasher, measure_options);
    SearchConfigHash::from_bytes(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
pub fn puct_search_config_hash(
    max_steps: usize,
    simulations: usize,
    c_puct: f32,
    seed: u64,
    temperature_moves: usize,
    tree_reuse: bool,
    mask_stop: bool,
    no_backtrack: bool,
    candidate_options: CandidateOptions,
    measure_options: MeasureOptions,
) -> SearchConfigHash {
    let mut hasher = blake3::Hasher::new();
    update_chunk(&mut hasher, b"gz-search-puct-mcts-v1");
    update_u64(&mut hasher, max_steps as u64);
    update_u64(&mut hasher, simulations as u64);
    update_u32(&mut hasher, c_puct.to_bits());
    update_u64(&mut hasher, seed);
    update_u64(&mut hasher, temperature_moves as u64);
    update_bool(&mut hasher, tree_reuse);
    update_bool(&mut hasher, mask_stop);
    update_bool(&mut hasher, no_backtrack);
    update_candidate_options(&mut hasher, candidate_options);
    update_measure_options(&mut hasher, measure_options);
    SearchConfigHash::from_bytes(*hasher.finalize().as_bytes())
}

fn update_candidate_options(hasher: &mut blake3::Hasher, options: CandidateOptions) {
    update_option_usize(hasher, options.max_candidates);
    update_bool(hasher, options.deterministic_order);
}

fn update_measure_options(hasher: &mut blake3::Hasher, options: MeasureOptions) {
    update_chunk(hasher, options.config_hash.as_bytes());
    update_u64(hasher, options.samples.into());
    update_option_u64(hasher, options.timeout_ms);
    update_bool(hasher, options.deterministic);
}

fn update_chunk(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    update_u64(hasher, bytes.len() as u64);
    hasher.update(bytes);
}

fn update_bool(hasher: &mut blake3::Hasher, value: bool) {
    hasher.update(&[u8::from(value)]);
}

fn update_option_usize(hasher: &mut blake3::Hasher, value: Option<usize>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            update_u64(hasher, value as u64);
        }
        None => {
            hasher.update(&[0]);
        }
    };
}

fn update_option_u64(hasher: &mut blake3::Hasher, value: Option<u64>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            update_u64(hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    };
}

fn update_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn update_u32(hasher: &mut blake3::Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}
