use gz_features::{
    ActionFeature, ENCODING_VERSION, FeatureBatchView, FeatureCollator, FeatureEdge, FeatureError,
    FeatureRow, FeatureSchema, FeatureSchemaConfig, PositionFeatures, RowTargets,
    TrainingTargetsView, decode_feature_row, encode_feature_row, encode_training_targets,
    validate_batch_action_counts, validate_feature_row_header,
};
use std::num::NonZeroUsize;

const HAND_BUILT_BATCH_FINGERPRINT: &str =
    "9ccea8576a120381ba540bb8f5b250fb758997c0807007d9912ddef870617de0";

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
        opponent_reward_scale: 256.0,
        expander_degree: 0,
        expander_seed: 0,
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
            opponent_reward: 0.5,
            opponent_present: true,
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
    assert_eq!(view.action_subjects[0..3], [1, 2, u32::from(u16::MAX)]);
    assert_eq!(view.position[0], [2.0, 3.0, 0.75, 0.125]);
    assert_eq!(view.opponent_reward, vec![0.5, 0.0]);
    assert_eq!(view.opponent_present, vec![1, 0]);
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
fn feature_row_codec_roundtrips_and_checks_header() {
    let schema = schema();
    let row = row();
    let mut bytes = Vec::new();

    encode_feature_row(&row, &schema, &mut bytes).unwrap();
    validate_feature_row_header(&bytes, &schema.hash()).unwrap();
    assert_eq!(decode_feature_row(&bytes).unwrap(), row);

    let mut bad_magic = bytes.clone();
    bad_magic[0] = b'X';
    assert!(matches!(
        validate_feature_row_header(&bad_magic, &schema.hash()),
        Err(FeatureError::InvalidEncoding(_))
    ));

    let mut bad_version = bytes.clone();
    bad_version[4..8].copy_from_slice(&(ENCODING_VERSION + 1).to_le_bytes());
    assert!(matches!(
        validate_feature_row_header(&bad_version, &schema.hash()),
        Err(FeatureError::InvalidEncoding(_))
    ));

    let mut other_config = schema.config().clone();
    other_config.max_nodes += 1;
    let other_schema = FeatureSchema::new(other_config).unwrap();
    assert!(matches!(
        validate_feature_row_header(&bytes, &other_schema.hash()),
        Err(FeatureError::InvalidEncoding(_))
    ));
}

#[test]
fn feature_row_codec_has_stable_layout() {
    let schema = schema();
    let row = row();
    let mut bytes = Vec::new();
    encode_feature_row(&row, &schema, &mut bytes).unwrap();

    // v2 layout: u16 node indexes and kind tokens, bf16 floats, u8
    // subject counts. All float fixtures are dyadic, so their bf16 bits
    // are the top half of the f32 bits.
    let bf16 = |value: f32| (value.to_bits() >> 16) as u16;
    let mut expected = Vec::new();
    expected.extend_from_slice(b"GZFR");
    expected.extend_from_slice(&ENCODING_VERSION.to_le_bytes());
    expected.extend_from_slice(schema.hash().as_bytes());
    expected.extend_from_slice(&3u32.to_le_bytes());
    expected.extend_from_slice(&3u32.to_le_bytes());
    for token in [1u16, 3, 6] {
        expected.extend_from_slice(&token.to_le_bytes());
    }
    expected.extend_from_slice(&3u32.to_le_bytes());
    for value in [0.5f32, 1.5, 2.5] {
        expected.extend_from_slice(&bf16(value).to_le_bytes());
    }
    expected.extend_from_slice(&2u32.to_le_bytes());
    for (src, dst, edge_type) in [(0u16, 1u16, 0u8), (1, 2, 1)] {
        expected.extend_from_slice(&src.to_le_bytes());
        expected.extend_from_slice(&dst.to_le_bytes());
        expected.push(edge_type);
    }
    expected.extend_from_slice(&2u32.to_le_bytes());
    expected.extend_from_slice(&4u16.to_le_bytes());
    expected.extend_from_slice(&bf16(0.25).to_le_bytes());
    expected.push(2);
    expected.extend_from_slice(&1u16.to_le_bytes());
    expected.extend_from_slice(&2u16.to_le_bytes());
    expected.extend_from_slice(&1u16.to_le_bytes());
    expected.extend_from_slice(&bf16(0.0).to_le_bytes());
    expected.push(0);
    expected.extend_from_slice(&2u32.to_le_bytes());
    expected.extend_from_slice(&3u32.to_le_bytes());
    expected.extend_from_slice(&bf16(0.75).to_le_bytes());
    expected.extend_from_slice(&bf16(0.125).to_le_bytes());
    expected.extend_from_slice(&bf16(0.5).to_le_bytes());
    expected.push(1);

    assert_eq!(bytes, expected);
}

#[test]
fn training_targets_codec_writes_padded_sections() {
    let targets = [
        RowTargets {
            policy: vec![1.0, 2.0, 3.0],
            value: Some(1.0),
            reward: 0.25,
        },
        RowTargets {
            policy: vec![-1.0],
            value: None,
            reward: -0.5,
        },
    ];
    let mut bytes = Vec::new();

    encode_training_targets(&targets, 2, 3, &mut bytes).unwrap();
    let view = TrainingTargetsView::parse(&bytes).unwrap();

    assert_eq!(view.capacity, 2);
    assert_eq!(view.row_count, 2);
    assert_eq!(view.max_actions, 3);
    assert_eq!(view.policy, vec![1.0, 2.0, 3.0, -1.0, 0.0, 0.0]);
    assert_eq!(view.value, vec![1.0, 0.0]);
    assert_eq!(view.value_valid, vec![1, 0]);
    assert_eq!(view.reward, vec![0.25, -0.5]);
    assert_eq!(
        fingerprint(&bytes),
        "618b58f8d534fe00f1b36c449d800dba7e83b6adf45c243f4eafc02ac030e77c"
    );
}

#[test]
fn training_targets_codec_rejects_bad_policy_width() {
    let mut bytes = Vec::new();

    assert!(matches!(
        encode_training_targets(
            &[RowTargets {
                policy: vec![1.0, 2.0],
                value: Some(1.0),
                reward: 0.0,
            }],
            1,
            1,
            &mut bytes,
        ),
        Err(FeatureError::ActionOverflow { .. })
    ));
}

#[test]
fn decode_outputs_truncates_policy_by_action_count() {
    let schema = schema();
    let collator = FeatureCollator::new(schema, NonZeroUsize::new(2).unwrap());
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GZFO");
    bytes.extend_from_slice(&gz_features::BATCH_ENCODING_VERSION.to_le_bytes());
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

#[test]
fn validate_batch_action_counts_checks_lengths_mismatches_and_overflow() {
    let schema = schema();
    let mut collator = FeatureCollator::new(schema, NonZeroUsize::new(2).unwrap());
    let rows = [row(), row()];
    let mut bytes = Vec::new();

    collator.collate_into(&rows, &mut bytes).unwrap();

    validate_batch_action_counts(&bytes, &[2, 2]).unwrap();
    assert!(matches!(
        validate_batch_action_counts(&bytes, &[2]),
        Err(FeatureError::InvalidEncoding(_))
    ));
    assert!(matches!(
        validate_batch_action_counts(&bytes, &[2, 2, 2]),
        Err(FeatureError::InvalidEncoding(_))
    ));
    assert!(matches!(
        validate_batch_action_counts(&bytes, &[1, 2]),
        Err(FeatureError::InvalidEncoding(_))
    ));
    assert!(matches!(
        validate_batch_action_counts(&bytes, &[2, 1]),
        Err(FeatureError::InvalidEncoding(_))
    ));
    assert!(matches!(
        validate_batch_action_counts(&bytes, &[5, 2]),
        Err(FeatureError::ActionOverflow { .. })
    ));
}

fn fingerprint(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn assert_fingerprint(name: &str, actual: &str, expected: &str) {
    assert_eq!(actual, expected, "{name} fingerprint: {actual}");
}
