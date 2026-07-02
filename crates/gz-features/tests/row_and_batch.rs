use gz_features::{
    ActionFeature, FeatureBatchView, FeatureCollator, FeatureEdge, FeatureError, FeatureRow,
    FeatureSchema, FeatureSchemaConfig, PositionFeatures,
};
use std::num::NonZeroUsize;

const HAND_BUILT_BATCH_FINGERPRINT: &str =
    "665f9353ee10a9d6bb9a32cd57dd20200ca98ff36115cf80416b973ea6cfbfc9";

fn schema() -> FeatureSchema {
    FeatureSchema::new(FeatureSchemaConfig {
        name: "test-v1".to_string(),
        node_vocab_size: 8,
        node_attr_dim: 1,
        edge_type_count: 2,
        action_kind_vocab_size: 16,
        max_nodes: 4,
        max_edges: 4,
        max_actions: 4,
        max_subjects: 3,
    })
    .unwrap()
}

fn row() -> FeatureRow {
    FeatureRow {
        node_count: 3,
        node_tokens: vec![1, 3, 6],
        node_attrs: vec![0.5, 1.5, 2.5],
        edges: vec![
            FeatureEdge {
                src: 0,
                dst: 1,
                edge_type: 0,
            },
            FeatureEdge {
                src: 1,
                dst: 2,
                edge_type: 1,
            },
        ],
        actions: vec![
            ActionFeature {
                kind_token: 4,
                static_prior: 0.25,
                subjects: vec![1, 2],
            },
            ActionFeature {
                kind_token: 1,
                static_prior: 0.0,
                subjects: Vec::new(),
            },
        ],
        position: PositionFeatures {
            root_step: 2,
            leaf_depth: 3,
            budget_fraction: 0.75,
            budget_step: 0.125,
        },
    }
}

#[test]
fn row_validation_catches_stop_order() {
    let schema = schema();
    let mut row = row();
    row.actions.swap(0, 1);

    assert!(matches!(
        row.validate(&schema),
        Err(FeatureError::InvalidRow(_))
    ));
}

#[test]
fn collate_parse_roundtrips_sections_and_padding() {
    let schema = schema();
    let mut collator = FeatureCollator::new(schema.clone(), NonZeroUsize::new(2).unwrap());
    let row = row();
    let mut bytes = Vec::new();

    collator
        .collate_into(std::slice::from_ref(&row), &mut bytes)
        .unwrap();
    let again = {
        let mut out = Vec::new();
        collator
            .collate_into(std::slice::from_ref(&row), &mut out)
            .unwrap();
        out
    };
    assert_eq!(bytes, again);
    assert_fingerprint(
        "hand-built batch",
        &fingerprint(&bytes),
        HAND_BUILT_BATCH_FINGERPRINT,
    );

    let view = FeatureBatchView::parse(&bytes).unwrap();
    assert_eq!(view.schema_hash, schema.hash());
    assert_eq!(view.batch_capacity, 2);
    assert_eq!(view.row_count, 1);
    assert_eq!(view.node_count, vec![3, 0]);
    assert_eq!(view.node_tokens[0..4], [1, 3, 6, 0]);
    assert_eq!(view.edge_count, vec![2, 0]);
    assert_eq!(view.edge_src[0..4], [0, 1, 0, 0]);
    assert_eq!(view.edge_dst[0..4], [1, 2, 0, 0]);
    assert_eq!(view.edge_type[0..4], [0, 1, 0, 0]);
    assert_eq!(view.action_count, vec![2, 0]);
    assert_eq!(view.action_kind[0..4], [4, 1, 0, 0]);
    assert_eq!(view.subject_count[0..4], [2, 0, 0, 0]);
    assert_eq!(view.action_subjects[0..3], [1, 2, u32::MAX]);
    assert_eq!(view.position[0], [2.0, 3.0, 0.75, 0.125]);
}

#[test]
fn collate_rejects_empty_and_overflow() {
    let schema = schema();
    let mut collator = FeatureCollator::new(schema, NonZeroUsize::new(1).unwrap());
    let mut bytes = Vec::new();

    assert!(matches!(
        collator.collate_into(&[], &mut bytes),
        Err(FeatureError::EmptyBatch)
    ));
    assert!(matches!(
        collator.collate_into(&[row(), row()], &mut bytes),
        Err(FeatureError::BatchOverflow { .. })
    ));
}

#[test]
fn decode_outputs_truncates_policy_by_action_count() {
    let schema = schema();
    let collator = FeatureCollator::new(schema, NonZeroUsize::new(2).unwrap());
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GZFO");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&4u32.to_le_bytes());
    for value in [0.5f32, -0.25] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in [1.0f32, 2.0, 99.0, 99.0, -1.0, -2.0, -3.0, 99.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    let rows = collator.decode_outputs(&bytes, &[2, 3]).unwrap();
    assert_eq!(rows[0].value, 0.5);
    assert_eq!(rows[0].policy_logits, vec![1.0, 2.0]);
    assert_eq!(rows[1].value, -0.25);
    assert_eq!(rows[1].policy_logits, vec![-1.0, -2.0, -3.0]);
}

fn fingerprint(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn assert_fingerprint(name: &str, actual: &str, expected: &str) {
    assert_eq!(actual, expected, "{name} fingerprint: {actual}");
}
