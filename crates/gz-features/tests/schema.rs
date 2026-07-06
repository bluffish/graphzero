use gz_features::{
    FeatureError, FeatureSchema, FeatureSchemaConfig, FeatureSchemaHash, STOP_ACTION_KIND_TOKEN,
    decode_feature_schema_config, encode_feature_schema_config,
};
use std::str::FromStr;

fn schema_config() -> FeatureSchemaConfig {
    FeatureSchemaConfig {
        name: "test-v1".to_string(),
        node_vocab_size: 8,
        node_attr_dim: 2,
        edge_type_count: 3,
        action_kind_vocab_size: 12,
        max_nodes: 4,
        max_edges: 6,
        max_actions: 5,
        max_subjects: 3,
        opponent_reward_scale: 256.0,
        expander_degree: 0,
        expander_seed: 0,
    }
}

#[test]
fn schema_validates_and_hash_roundtrips() {
    let schema = FeatureSchema::new(schema_config()).unwrap();
    let hash = schema.hash();
    let hex = hash.to_string();

    assert_eq!(hex.len(), FeatureSchemaHash::HEX_LEN);
    assert_eq!(FeatureSchemaHash::try_from_hex(&hex).unwrap(), hash);
    assert_eq!(FeatureSchemaHash::from_str(&hex).unwrap(), hash);
    assert_eq!(format!("{hash:?}"), format!("FeatureSchemaHash({hex})"));
    assert_eq!(STOP_ACTION_KIND_TOKEN, 1);
}

#[test]
fn schema_hash_changes_when_config_changes() {
    let first = FeatureSchema::new(schema_config()).unwrap();
    let mut changed = schema_config();
    changed.max_nodes += 1;
    let second = FeatureSchema::new(changed).unwrap();

    assert_ne!(first.hash(), second.hash());

    let first = FeatureSchema::new(schema_config()).unwrap();
    let mut changed = schema_config();
    changed.expander_degree = 1;
    changed.max_edges = changed.max_nodes + 1;
    let second = FeatureSchema::new(changed).unwrap();

    assert_ne!(first.hash(), second.hash());

    let first = FeatureSchema::new(schema_config()).unwrap();
    let mut changed = schema_config();
    changed.expander_seed = 1;
    let second = FeatureSchema::new(changed).unwrap();

    assert_ne!(first.hash(), second.hash());
}

#[test]
fn schema_rejects_invalid_config() {
    let mut config = schema_config();
    config.name.clear();

    assert!(matches!(
        FeatureSchema::new(config),
        Err(FeatureError::InvalidSchema(_))
    ));

    let mut config = schema_config();
    config.expander_degree = 2;
    config.max_edges = config.max_nodes * 2;

    assert!(matches!(
        FeatureSchema::new(config),
        Err(FeatureError::InvalidSchema(_))
    ));
}

#[test]
fn schema_config_codec_roundtrips_and_validates() {
    let mut bytes = Vec::new();
    let config = schema_config();

    encode_feature_schema_config(&config, &mut bytes).unwrap();
    assert_eq!(decode_feature_schema_config(&bytes).unwrap(), config);

    bytes.push(0);
    assert!(matches!(
        decode_feature_schema_config(&bytes),
        Err(FeatureError::InvalidEncoding(_))
    ));

    let mut invalid = schema_config();
    invalid.expander_degree = 2;
    invalid.max_edges = invalid.max_nodes * 2;
    let mut invalid_bytes = Vec::new();
    invalid_bytes.extend_from_slice(&(invalid.name.len() as u16).to_le_bytes());
    invalid_bytes.extend_from_slice(invalid.name.as_bytes());
    invalid_bytes.extend_from_slice(&invalid.node_vocab_size.to_le_bytes());
    invalid_bytes.extend_from_slice(&invalid.node_attr_dim.to_le_bytes());
    invalid_bytes.push(invalid.edge_type_count);
    invalid_bytes.extend_from_slice(&invalid.action_kind_vocab_size.to_le_bytes());
    invalid_bytes.extend_from_slice(&invalid.max_nodes.to_le_bytes());
    invalid_bytes.extend_from_slice(&invalid.max_edges.to_le_bytes());
    invalid_bytes.extend_from_slice(&invalid.max_actions.to_le_bytes());
    invalid_bytes.extend_from_slice(&invalid.max_subjects.to_le_bytes());
    invalid_bytes.extend_from_slice(&invalid.opponent_reward_scale.to_le_bytes());
    invalid_bytes.push(invalid.expander_degree);
    invalid_bytes.extend_from_slice(&invalid.expander_seed.to_le_bytes());

    assert!(matches!(
        decode_feature_schema_config(&invalid_bytes),
        Err(FeatureError::InvalidSchema(_))
    ));
}
