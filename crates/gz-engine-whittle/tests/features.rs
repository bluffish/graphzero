use gz_engine::{CandidateOptions, GraphEngine};
use gz_engine_whittle::{
    WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor, WhittleFeatureExtractorConfig,
    WhittleGraphId, WhittleRoot,
};
use gz_features::{
    FeatureCollator, FeatureExtractor, FeatureRow, PositionFeatures, STOP_ACTION_KIND_TOKEN,
};
use std::num::NonZeroUsize;

const NO_NODE: u32 = u32::MAX;
const AND_IDEMPOTENT_ROW_FINGERPRINT: &str =
    "fecd61be5733ec3b9b401e8c456b4f7d6b6d301373f3ae760f53685021c823ff";
const AND_IDEMPOTENT_BATCH_FINGERPRINT: &str =
    "7979120dbc22bc9e88eb68ec077f0d119250dd1acf2c69351ae925cbc434a40b";

fn and_idempotent_artifact() -> Vec<u8> {
    wav1(1, 16, 2, &[(0, 0, NO_NODE), (2, 0, 0), (5, 1, NO_NODE)])
}

fn wav1(arity: u16, capacity: u16, output_node: u32, nodes: &[(i8, u32, u32)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"WAV1");
    bytes.extend_from_slice(&arity.to_le_bytes());
    bytes.extend_from_slice(&capacity.to_le_bytes());
    bytes.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&output_node.to_le_bytes());

    for (op, arg0, arg1) in nodes {
        bytes.push(*op as u8);
        bytes.extend_from_slice(&arg0.to_le_bytes());
        bytes.extend_from_slice(&arg1.to_le_bytes());
    }

    bytes
}

fn and_engine() -> WhittleEngine {
    WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Artifact(and_idempotent_artifact()),
        ..WhittleEngineConfig::default()
    })
    .unwrap()
}

fn four_node_engine() -> WhittleEngine {
    WhittleEngine::new(WhittleEngineConfig {
        root: WhittleRoot::Artifact(wav1(
            2,
            16,
            3,
            &[(0, 0, NO_NODE), (0, 1, NO_NODE), (2, 0, 1), (5, 2, NO_NODE)],
        )),
        ..WhittleEngineConfig::default()
    })
    .unwrap()
}

#[test]
fn whittle_extractor_maps_graph_and_actions() {
    let mut engine = and_engine();
    let root = engine.root();
    let mut candidates = Vec::new();
    engine
        .candidates(root, CandidateOptions::default(), &mut candidates)
        .unwrap();
    let mut extractor = WhittleFeatureExtractor::new(&engine);

    let row = extractor
        .extract(
            &engine,
            root,
            &candidates,
            PositionFeatures {
                root_step: 4,
                leaf_depth: 2,
                budget_fraction: 0.5,
                budget_step: 0.01,
            },
        )
        .unwrap();

    assert_eq!(extractor.schema().config().name, "whittle-v1");
    assert_eq!(extractor.schema().config().max_nodes, 16);
    assert_eq!(extractor.schema().config().edge_type_count, 3);
    assert_eq!(extractor.schema().config().expander_degree, 5);
    assert_eq!(row.node_count, 3);
    assert_eq!(row.node_tokens, vec![1, 3, 6]);
    assert!(row.edges.len() >= 3);
    assert_eq!(row.actions.len(), candidates.len() + 1);
    assert_eq!(
        row.actions.last().unwrap().kind_token,
        STOP_ACTION_KIND_TOKEN
    );
    assert!(row.actions.last().unwrap().subjects.is_empty());
    assert_eq!(row.position.root_step, 4);
    assert_fingerprint(
        "and-idempotent row",
        &row_fingerprint(&row),
        AND_IDEMPOTENT_ROW_FINGERPRINT,
    );

    let mut collator =
        FeatureCollator::new(extractor.schema().clone(), NonZeroUsize::new(2).unwrap());
    let mut batch = Vec::new();
    collator
        .collate_into(std::slice::from_ref(&row), &mut batch)
        .unwrap();
    assert_fingerprint(
        "and-idempotent batch",
        &fingerprint(&batch),
        AND_IDEMPOTENT_BATCH_FINGERPRINT,
    );

    let again = extractor
        .extract(&engine, root, &candidates, row.position)
        .unwrap();
    assert_eq!(again, row);
}

#[test]
fn degree_zero_extractor_emits_arg_edges_only() {
    let mut engine = and_engine();
    let root = engine.root();
    let mut candidates = Vec::new();
    engine
        .candidates(root, CandidateOptions::default(), &mut candidates)
        .unwrap();
    let mut extractor = WhittleFeatureExtractor::with_config(
        &engine,
        WhittleFeatureExtractorConfig {
            expander_degree: 0,
            expander_seed: 0,
            ..WhittleFeatureExtractorConfig::default()
        },
    );
    let row = feature_row(&mut extractor, &engine, root, &candidates);

    assert_eq!(extractor.schema().config().edge_type_count, 2);
    assert_eq!(extractor.schema().config().max_edges, 32);
    assert_eq!(row.edges.len(), 3);
    assert!(row.edges.iter().all(|edge| edge.edge_type < 2));
}

#[test]
fn expander_edges_are_deterministic_bounded_and_seeded() {
    let mut engine = and_engine();
    let root = engine.root();
    let mut candidates = Vec::new();
    engine
        .candidates(root, CandidateOptions::default(), &mut candidates)
        .unwrap();
    let config = WhittleFeatureExtractorConfig {
        expander_degree: 3,
        expander_seed: 11,
        ..WhittleFeatureExtractorConfig::default()
    };
    let mut left = WhittleFeatureExtractor::with_config(&engine, config);
    let mut right = WhittleFeatureExtractor::with_config(&engine, config);
    let left_row = feature_row(&mut left, &engine, root, &candidates);
    let left_again = feature_row(&mut left, &engine, root, &candidates);
    let right_row = feature_row(&mut right, &engine, root, &candidates);

    assert_eq!(left_row, left_again);
    assert_eq!(expander_edges(&left_row), expander_edges(&right_row));
    assert_expander_bounds(&left_row, 3);

    let mut seeded = WhittleFeatureExtractor::with_config(
        &engine,
        WhittleFeatureExtractorConfig {
            expander_degree: 3,
            expander_seed: 12,
            ..WhittleFeatureExtractorConfig::default()
        },
    );
    let seeded_row = feature_row(&mut seeded, &engine, root, &candidates);
    assert_ne!(expander_edges(&left_row), expander_edges(&seeded_row));

    let mut larger = four_node_engine();
    let larger_root = larger.root();
    let mut larger_candidates = Vec::new();
    larger
        .candidates(
            larger_root,
            CandidateOptions::default(),
            &mut larger_candidates,
        )
        .unwrap();
    let mut larger_extractor = WhittleFeatureExtractor::with_config(&larger, config);
    let larger_row = feature_row(
        &mut larger_extractor,
        &larger,
        larger_root,
        &larger_candidates,
    );
    assert_ne!(expander_edges(&left_row), expander_edges(&larger_row));
    assert_expander_bounds(&larger_row, 3);
}

fn feature_row(
    extractor: &mut WhittleFeatureExtractor,
    engine: &WhittleEngine,
    graph: WhittleGraphId,
    candidates: &[gz_engine_whittle::WhittleCandidateId],
) -> FeatureRow {
    extractor
        .extract(
            engine,
            graph,
            candidates,
            PositionFeatures {
                root_step: 0,
                leaf_depth: 0,
                budget_fraction: 1.0,
                budget_step: 1.0,
            },
        )
        .unwrap()
}

fn expander_edges(row: &FeatureRow) -> Vec<(u32, u32, u8)> {
    row.edges
        .iter()
        .filter(|edge| edge.edge_type == 2)
        .map(|edge| (edge.src, edge.dst, edge.edge_type))
        .collect()
}

fn assert_expander_bounds(row: &FeatureRow, degree: u8) {
    let edges = expander_edges(row);
    assert!(edges.len() <= usize::from(degree) * row.node_count as usize);
    assert!(!edges.is_empty());
    for (src, dst, edge_type) in edges {
        assert_eq!(edge_type, 2);
        assert!(src < row.node_count);
        assert!(dst < row.node_count);
        assert_ne!(src, dst);
    }
}

fn row_fingerprint(row: &FeatureRow) -> String {
    let mut hasher = blake3::Hasher::new();
    update_u32(&mut hasher, row.node_count);
    update_u32(&mut hasher, row.node_tokens.len() as u32);
    for &token in &row.node_tokens {
        hasher.update(&token.to_le_bytes());
    }
    update_u32(&mut hasher, row.node_attrs.len() as u32);
    for &attr in &row.node_attrs {
        update_f32(&mut hasher, attr);
    }
    update_u32(&mut hasher, row.edges.len() as u32);
    for edge in &row.edges {
        update_u32(&mut hasher, edge.src);
        update_u32(&mut hasher, edge.dst);
        hasher.update(&[edge.edge_type]);
    }
    update_u32(&mut hasher, row.actions.len() as u32);
    for action in &row.actions {
        update_u32(&mut hasher, action.kind_token);
        update_f32(&mut hasher, action.static_prior);
        update_u32(&mut hasher, action.subjects.len() as u32);
        for &subject in &action.subjects {
            update_u32(&mut hasher, subject);
        }
    }
    update_u32(&mut hasher, row.position.root_step);
    update_u32(&mut hasher, row.position.leaf_depth);
    update_f32(&mut hasher, row.position.budget_fraction);
    update_f32(&mut hasher, row.position.budget_step);
    hasher.finalize().to_hex().to_string()
}

fn fingerprint(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn update_u32(hasher: &mut blake3::Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}

fn update_f32(hasher: &mut blake3::Hasher, value: f32) {
    hasher.update(&value.to_bits().to_le_bytes());
}

fn assert_fingerprint(name: &str, actual: &str, expected: &str) {
    assert_eq!(actual, expected, "{name} fingerprint: {actual}");
}
