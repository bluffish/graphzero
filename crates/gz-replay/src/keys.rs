pub(crate) const SCHEMA_VERSION: u32 = 5;

pub(crate) const CF_META: &str = "meta";
pub(crate) const CF_EPISODES: &str = "episodes";
pub(crate) const CF_ROWS: &str = "rows";
pub(crate) const CF_ROW_INDEX: &str = "row_index";

pub(crate) const META_SCHEMA_VERSION: &[u8] = b"schema_version";
pub(crate) const META_EPISODES_STOPPED: &[u8] = b"episodes_stopped";
pub(crate) const META_NEXT_EPISODE_SEQ: &[u8] = b"next_episode_seq";
pub(crate) const META_PRODUCED_ROWS: &[u8] = b"produced_rows";
pub(crate) const META_CONSUMED_ROWS: &[u8] = b"consumed_rows";
pub(crate) const META_FEATURE_SCHEMA: &[u8] = b"feature_schema";

pub(crate) const EPISODE_KEY_LEN: usize = 8;
pub(crate) const ROW_KEY_LEN: usize = 12;

pub(crate) fn episode_key(seq: u64) -> [u8; EPISODE_KEY_LEN] {
    seq.to_be_bytes()
}

pub(crate) fn row_key(episode_seq: u64, step_index: u32) -> [u8; ROW_KEY_LEN] {
    let mut key = [0; ROW_KEY_LEN];
    key[..8].copy_from_slice(&episode_seq.to_be_bytes());
    key[8..].copy_from_slice(&step_index.to_be_bytes());
    key
}

pub(crate) fn row_index_key(seq: u64) -> [u8; 8] {
    seq.to_be_bytes()
}

pub(crate) fn decode_u64_key(key: &[u8]) -> Option<u64> {
    let bytes: [u8; 8] = key.try_into().ok()?;
    Some(u64::from_be_bytes(bytes))
}

pub(crate) fn decode_episode_from_row_key(key: &[u8]) -> Option<u64> {
    if key.len() != ROW_KEY_LEN {
        return None;
    }

    let bytes: [u8; 8] = key[..8].try_into().ok()?;
    Some(u64::from_be_bytes(bytes))
}

pub(crate) fn encode_u32(value: u32) -> [u8; 4] {
    value.to_be_bytes()
}

pub(crate) fn decode_u32(value: &[u8]) -> Option<u32> {
    let bytes: [u8; 4] = value.try_into().ok()?;
    Some(u32::from_be_bytes(bytes))
}

pub(crate) fn encode_u64(value: u64) -> [u8; 8] {
    value.to_be_bytes()
}

pub(crate) fn decode_u64(value: &[u8]) -> Option<u64> {
    let bytes: [u8; 8] = value.try_into().ok()?;
    Some(u64::from_be_bytes(bytes))
}
