use crate::{FeatureError, FeatureResult};
use gz_engine::HexParseError;
use std::fmt;
use std::str::FromStr;

/// Row/targets encoding version: rows persist in replay stores, so this
/// only moves with a store schema bump. v3 = v2 plus opponent scalar
/// feature sections.
pub const ENCODING_VERSION: u32 = 3;
/// Eval-wire batch/output encoding version. v3 = v2 plus opponent
/// scalar feature sections.
pub const BATCH_ENCODING_VERSION: u32 = 3;
pub const STOP_ACTION_KIND_TOKEN: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureSchemaConfig {
    pub name: String,
    pub node_vocab_size: u16,
    pub node_attr_dim: u16,
    pub edge_type_count: u8,
    pub action_kind_vocab_size: u32,
    pub max_nodes: u32,
    pub max_edges: u32,
    pub max_actions: u32,
    pub max_subjects: u32,
    pub opponent_reward_scale: f32,
    pub expander_degree: u8,
    pub expander_seed: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeatureSchema {
    config: FeatureSchemaConfig,
    hash: FeatureSchemaHash,
}

impl FeatureSchema {
    pub fn new(config: FeatureSchemaConfig) -> FeatureResult<Self> {
        validate_config(&config)?;
        let hash = FeatureSchemaHash::derive(&config);
        Ok(Self { config, hash })
    }

    #[must_use]
    pub const fn config(&self) -> &FeatureSchemaConfig {
        &self.config
    }

    #[must_use]
    pub const fn hash(&self) -> FeatureSchemaHash {
        self.hash
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FeatureSchemaHash([u8; 32]);

impl FeatureSchemaHash {
    pub const BYTE_LEN: usize = 32;
    pub const HEX_LEN: usize = 64;

    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn try_from_hex(hex: &str) -> Result<Self, HexParseError> {
        parse_hex_array(hex).map(Self)
    }

    fn derive(config: &FeatureSchemaConfig) -> Self {
        let mut hasher = blake3::Hasher::new();
        update_chunk(&mut hasher, b"gz-features-schema-v1");
        update_u32(&mut hasher, ENCODING_VERSION);
        update_string(&mut hasher, &config.name);
        update_u16(&mut hasher, config.node_vocab_size);
        update_u16(&mut hasher, config.node_attr_dim);
        update_u8(&mut hasher, config.edge_type_count);
        update_u32(&mut hasher, config.action_kind_vocab_size);
        update_u32(&mut hasher, config.max_nodes);
        update_u32(&mut hasher, config.max_edges);
        update_u32(&mut hasher, config.max_actions);
        update_u32(&mut hasher, config.max_subjects);
        update_u32(&mut hasher, config.opponent_reward_scale.to_bits());
        update_u8(&mut hasher, config.expander_degree);
        update_u64(&mut hasher, config.expander_seed);
        Self(*hasher.finalize().as_bytes())
    }
}

impl fmt::Display for FeatureSchemaHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for FeatureSchemaHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FeatureSchemaHash({self})")
    }
}

impl FromStr for FeatureSchemaHash {
    type Err = HexParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_from_hex(s)
    }
}

fn validate_config(config: &FeatureSchemaConfig) -> FeatureResult<()> {
    if config.name.is_empty() {
        return Err(FeatureError::InvalidSchema("name must be non-empty"));
    }
    if config.node_vocab_size < 2 {
        return Err(FeatureError::InvalidSchema("node_vocab_size must be >= 2"));
    }
    if config.action_kind_vocab_size < 3 {
        return Err(FeatureError::InvalidSchema(
            "action_kind_vocab_size must be >= 3",
        ));
    }
    if config.max_nodes == 0 {
        return Err(FeatureError::InvalidSchema("max_nodes must be positive"));
    }
    if config.max_edges == 0 {
        return Err(FeatureError::InvalidSchema("max_edges must be positive"));
    }
    if config.max_actions == 0 {
        return Err(FeatureError::InvalidSchema("max_actions must be positive"));
    }
    if config.max_subjects == 0 {
        return Err(FeatureError::InvalidSchema("max_subjects must be positive"));
    }
    if !config.opponent_reward_scale.is_finite() || config.opponent_reward_scale <= 0.0 {
        return Err(FeatureError::InvalidSchema(
            "opponent_reward_scale must be finite and positive",
        ));
    }
    if config.expander_degree > 0 {
        if config.edge_type_count == 0 {
            return Err(FeatureError::InvalidSchema(
                "edge_type_count must include expander type",
            ));
        }
        let required = u32::from(config.expander_degree)
            .checked_mul(config.max_nodes)
            .and_then(|value| value.checked_add(1))
            .ok_or(FeatureError::InvalidSchema("expander edge budget overflow"))?;
        if config.max_edges < required {
            return Err(FeatureError::InvalidSchema(
                "max_edges too small for expander_degree",
            ));
        }
    }
    Ok(())
}

fn update_string(hasher: &mut blake3::Hasher, value: &str) {
    update_chunk(hasher, value.as_bytes());
}

fn update_chunk(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    update_u64(hasher, bytes.len() as u64);
    hasher.update(bytes);
}

fn update_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn update_u32(hasher: &mut blake3::Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}

fn update_u16(hasher: &mut blake3::Hasher, value: u16) {
    hasher.update(&value.to_le_bytes());
}

fn update_u8(hasher: &mut blake3::Hasher, value: u8) {
    hasher.update(&[value]);
}

fn parse_hex_array(hex: &str) -> Result<[u8; 32], HexParseError> {
    let expected = FeatureSchemaHash::HEX_LEN;
    let actual = hex.len();

    if actual != expected {
        return Err(HexParseError::InvalidLength { expected, actual });
    }

    let mut bytes = [0u8; 32];
    let hex = hex.as_bytes();
    for (index, byte) in bytes.iter_mut().enumerate() {
        let hi_index = index * 2;
        let lo_index = hi_index + 1;
        let hi = hex_value(hex[hi_index]).ok_or(HexParseError::InvalidCharacter {
            index: hi_index,
            byte: hex[hi_index],
        })?;
        let lo = hex_value(hex[lo_index]).ok_or(HexParseError::InvalidCharacter {
            index: lo_index,
            byte: hex[lo_index],
        })?;
        *byte = (hi << 4) | lo;
    }

    Ok(bytes)
}

const fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
