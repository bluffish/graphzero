use gz_engine::{CandidateOptions, GraphEngine};
use gz_engine_whittle::{WhittleEngine, WhittleEngineConfig, WhittleFeatureExtractor, WhittleRoot};
use gz_features::{
    FeatureCollator, FeatureExtractor, FeatureRow, PositionFeatures, STOP_ACTION_KIND_TOKEN,
};
use std::num::NonZeroUsize;

const NO_NODE: u32 = u32::MAX;
const AND_IDEMPOTENT_ROW_FINGERPRINT: &str =
    "87b92adcef7d69b167801095a60ee2528c6cb4394b3d2e17dc5cbfacc00a3ff8";
const AND_IDEMPOTENT_BATCH_FINGERPRINT: &str =
    "51fe84f0d66c98ab3c7b5a5f007417270c94586b9dbfde08f5bdecdae062a318";

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
    assert_eq!(row.node_count, 3);
    assert_eq!(row.node_tokens, vec![1, 3, 6]);
    assert_eq!(row.edges.len(), 3);
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
