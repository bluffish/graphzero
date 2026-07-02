use gz_features::{
    FeatureError, FeatureSchema, FeatureSchemaConfig, FeatureSchemaHash, STOP_ACTION_KIND_TOKEN,
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
}

#[test]
fn schema_rejects_invalid_config() {
    let mut config = schema_config();
    config.name.clear();

    assert!(matches!(
        FeatureSchema::new(config),
        Err(FeatureError::InvalidSchema(_))
    ));
}
