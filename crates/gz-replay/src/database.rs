use crate::error::{ReplayError, ReplayResult};
use crate::keys::{
    CF_EPISODES, CF_LEGACY_POLICY_ROW_INDEX, CF_LEGACY_VALUE_ROW_INDEX, CF_META, CF_ROW_INDEX,
    CF_ROWS, META_COMPLETED_GAMES, META_CONSUMED_ROWS, META_DATA_MODE, META_ENGINE_IDENTITY,
    META_FEATURE_SCHEMA, META_NEXT_EPISODE_SEQ, META_PRODUCED_ROWS, META_SCHEMA_VERSION,
    META_SYMMETRIC_REWARD_TARGET_SCALE, META_SYMMETRIC_REWARD_TIE_EPSILON, SCHEMA_VERSION,
    decode_u32, decode_u64, decode_u64_key, encode_u32, encode_u64,
};
use crate::store::{ReplayDataMode, SymmetricRewardTargetScale, SymmetricRewardTieEpsilon};
use gz_engine::{ActionSetHash, EngineId, EngineIdentity, EngineVersion};
use gz_features::FeatureSchemaConfig;
use rocksdb::{
    BlockBasedOptions, Cache, ColumnFamilyDescriptor, DB, DBCompressionType, IteratorMode, Options,
    WriteBatch,
};
use std::path::Path;

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
struct StoredFeatureSchemaConfig {
    name: String,
    node_vocab_size: u16,
    node_attr_dim: u16,
    edge_type_count: u8,
    action_kind_vocab_size: u32,
    max_nodes: u32,
    max_edges: u32,
    max_actions: u32,
    max_subjects: u32,
    opponent_reward_scale: f32,
    expander_degree: u8,
    expander_seed: u64,
}

impl From<&FeatureSchemaConfig> for StoredFeatureSchemaConfig {
    fn from(config: &FeatureSchemaConfig) -> Self {
        Self {
            name: config.name.clone(),
            node_vocab_size: config.node_vocab_size,
            node_attr_dim: config.node_attr_dim,
            edge_type_count: config.edge_type_count,
            action_kind_vocab_size: config.action_kind_vocab_size,
            max_nodes: config.max_nodes,
            max_edges: config.max_edges,
            max_actions: config.max_actions,
            max_subjects: config.max_subjects,
            opponent_reward_scale: config.opponent_reward_scale,
            expander_degree: config.expander_degree,
            expander_seed: config.expander_seed,
        }
    }
}

impl From<StoredFeatureSchemaConfig> for FeatureSchemaConfig {
    fn from(config: StoredFeatureSchemaConfig) -> Self {
        Self {
            name: config.name,
            node_vocab_size: config.node_vocab_size,
            node_attr_dim: config.node_attr_dim,
            edge_type_count: config.edge_type_count,
            action_kind_vocab_size: config.action_kind_vocab_size,
            max_nodes: config.max_nodes,
            max_edges: config.max_edges,
            max_actions: config.max_actions,
            max_subjects: config.max_subjects,
            opponent_reward_scale: config.opponent_reward_scale,
            expander_degree: config.expander_degree,
            expander_seed: config.expander_seed,
        }
    }
}

pub(crate) fn open_db(path: &Path) -> ReplayResult<DB> {
    let mut options = Options::default();
    options.create_if_missing(true);
    options.create_missing_column_families(true);
    options.increase_parallelism(8);
    options.set_max_background_jobs(8);

    // Sized for the full sample window. Cache misses under write load
    // materially stall the trainer's batched reads.
    let cache = Cache::new_lru_cache(16 * 1024 * 1024 * 1024);
    let mut block = BlockBasedOptions::default();
    block.set_block_cache(&cache);

    let mut value_cf = Options::default();
    value_cf.set_write_buffer_size(256 * 1024 * 1024);
    value_cf.set_target_file_size_base(128 * 1024 * 1024);
    value_cf.set_level_compaction_dynamic_level_bytes(true);
    value_cf.set_compression_type(DBCompressionType::Lz4);
    value_cf.set_block_based_table_factory(&block);

    let mut index_cf = Options::default();
    index_cf.set_write_buffer_size(64 * 1024 * 1024);
    index_cf.set_level_compaction_dynamic_level_bytes(true);
    index_cf.set_compression_type(DBCompressionType::Lz4);
    index_cf.set_block_based_table_factory(&block);

    let descriptors = [
        ColumnFamilyDescriptor::new(CF_META, Options::default()),
        ColumnFamilyDescriptor::new(CF_EPISODES, index_cf.clone()),
        ColumnFamilyDescriptor::new(CF_ROWS, value_cf),
        ColumnFamilyDescriptor::new(CF_ROW_INDEX, index_cf.clone()),
        // Schema v8 stores contain these retired indexes. RocksDB requires
        // every existing column family to be named when opening the database.
        ColumnFamilyDescriptor::new(CF_LEGACY_POLICY_ROW_INDEX, index_cf.clone()),
        ColumnFamilyDescriptor::new(CF_LEGACY_VALUE_ROW_INDEX, index_cf),
    ];
    DB::open_cf_descriptors(&options, path, descriptors).map_err(ReplayError::from)
}

pub(crate) fn ensure_schema(db: &DB) -> ReplayResult<()> {
    let meta = cf(db, CF_META)?;
    match db.get_cf(&meta, META_SCHEMA_VERSION)? {
        Some(bytes) if decode_u32(&bytes) == Some(SCHEMA_VERSION) => Ok(()),
        Some(_) => Err(ReplayError::SchemaMismatch),
        None => {
            let mut batch = WriteBatch::default();
            batch.put_cf(&meta, META_SCHEMA_VERSION, encode_u32(SCHEMA_VERSION));
            batch.put_cf(&meta, META_NEXT_EPISODE_SEQ, encode_u64(0));
            batch.put_cf(&meta, META_COMPLETED_GAMES, encode_u64(0));
            batch.put_cf(&meta, META_PRODUCED_ROWS, encode_u64(0));
            batch.put_cf(&meta, META_CONSUMED_ROWS, encode_u64(0));
            db.write(batch).map_err(ReplayError::from)
        }
    }
}

pub(crate) fn recover_next_episode_seq(db: &DB) -> ReplayResult<u64> {
    let episodes = cf(db, CF_EPISODES)?;
    let mut iter = db.iterator_cf(&episodes, IteratorMode::End);
    match iter.next().transpose()? {
        Some((key, _)) => decode_u64_key(&key)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| ReplayError::storage("corrupt episode key")),
        None => Ok(0),
    }
}

pub(crate) fn recover_next_row_seq(db: &DB) -> ReplayResult<u64> {
    recover_next_index_seq(db, CF_ROW_INDEX)
}

fn recover_next_index_seq(db: &DB, index_name: &'static str) -> ReplayResult<u64> {
    let row_index = cf(db, index_name)?;
    let mut iter = db.iterator_cf(&row_index, IteratorMode::End);
    match iter.next().transpose()? {
        Some((key, _)) => decode_u64_key(&key)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| ReplayError::storage("corrupt row index key")),
        None => Ok(0),
    }
}

pub(crate) fn read_meta_u64(db: &DB, key: &[u8]) -> ReplayResult<Option<u64>> {
    let meta = cf(db, CF_META)?;
    db.get_cf(&meta, key)?
        .map(|bytes| decode_u64(&bytes).ok_or_else(|| ReplayError::storage("corrupt meta u64")))
        .transpose()
}

pub(crate) fn write_meta_u64(db: &DB, key: &[u8], value: u64) -> ReplayResult<()> {
    let meta = cf(db, CF_META)?;
    db.put_cf(&meta, key, encode_u64(value))
        .map_err(ReplayError::from)
}

pub(crate) fn read_feature_schema(db: &DB) -> ReplayResult<Option<FeatureSchemaConfig>> {
    let meta = cf(db, CF_META)?;
    db.get_cf(&meta, META_FEATURE_SCHEMA)?
        .map(|bytes| {
            postcard::from_bytes::<StoredFeatureSchemaConfig>(&bytes)
                .map(FeatureSchemaConfig::from)
                .map_err(ReplayError::from)
        })
        .transpose()
}

pub(crate) fn stored_feature_schema(config: &FeatureSchemaConfig) -> ReplayResult<Vec<u8>> {
    postcard::to_allocvec(&StoredFeatureSchemaConfig::from(config)).map_err(ReplayError::from)
}

pub(crate) fn read_engine_identity(db: &DB) -> ReplayResult<Option<EngineIdentity>> {
    let meta = cf(db, CF_META)?;
    db.get_cf(&meta, META_ENGINE_IDENTITY)?
        .map(|bytes| {
            if bytes.len() != 64 {
                return Err(ReplayError::storage("corrupt engine identity"));
            }
            Ok(EngineIdentity {
                engine_id: EngineId::from_bytes(
                    bytes[0..16]
                        .try_into()
                        .map_err(|_| ReplayError::storage("corrupt engine id"))?,
                ),
                engine_version: EngineVersion::from_bytes(
                    bytes[16..32]
                        .try_into()
                        .map_err(|_| ReplayError::storage("corrupt engine version"))?,
                ),
                action_set_hash: ActionSetHash::from_bytes(
                    bytes[32..64]
                        .try_into()
                        .map_err(|_| ReplayError::storage("corrupt action-set hash"))?,
                ),
            })
        })
        .transpose()
}

pub(crate) fn stored_engine_identity(identity: EngineIdentity) -> [u8; 64] {
    let mut bytes = [0; 64];
    bytes[0..16].copy_from_slice(identity.engine_id.as_bytes());
    bytes[16..32].copy_from_slice(identity.engine_version.as_bytes());
    bytes[32..64].copy_from_slice(identity.action_set_hash.as_bytes());
    bytes
}

pub(crate) fn read_data_mode(db: &DB) -> ReplayResult<Option<ReplayDataMode>> {
    let meta = cf(db, CF_META)?;
    db.get_cf(&meta, META_DATA_MODE)?
        .map(|bytes| ReplayDataMode::from_bytes(&bytes).ok_or(ReplayError::DataModeMismatch))
        .transpose()
}

pub(crate) fn read_symmetric_reward_tie_epsilon(
    db: &DB,
) -> ReplayResult<Option<SymmetricRewardTieEpsilon>> {
    let meta = cf(db, CF_META)?;
    db.get_cf(&meta, META_SYMMETRIC_REWARD_TIE_EPSILON)?
        .map(|bytes| {
            let bytes: [u8; 4] = bytes
                .as_slice()
                .try_into()
                .map_err(|_| ReplayError::storage("corrupt symmetric reward tie epsilon"))?;
            SymmetricRewardTieEpsilon::new(f32::from_bits(u32::from_be_bytes(bytes)))
                .ok_or_else(|| ReplayError::storage("corrupt symmetric reward tie epsilon"))
        })
        .transpose()
}

pub(crate) fn stored_symmetric_reward_tie_epsilon(epsilon: SymmetricRewardTieEpsilon) -> [u8; 4] {
    epsilon.get().to_bits().to_be_bytes()
}

pub(crate) fn read_symmetric_reward_target_scale(
    db: &DB,
) -> ReplayResult<Option<SymmetricRewardTargetScale>> {
    let meta = cf(db, CF_META)?;
    db.get_cf(&meta, META_SYMMETRIC_REWARD_TARGET_SCALE)?
        .map(|bytes| {
            let bytes: [u8; 4] = bytes
                .as_slice()
                .try_into()
                .map_err(|_| ReplayError::storage("corrupt symmetric reward target scale"))?;
            SymmetricRewardTargetScale::new(f32::from_bits(u32::from_be_bytes(bytes)))
                .ok_or_else(|| ReplayError::storage("corrupt symmetric reward target scale"))
        })
        .transpose()
}

pub(crate) fn stored_symmetric_reward_target_scale(scale: SymmetricRewardTargetScale) -> [u8; 4] {
    scale.get().to_bits().to_be_bytes()
}

fn cf<'a>(db: &'a DB, name: &'static str) -> ReplayResult<&'a rocksdb::ColumnFamily> {
    db.cf_handle(name)
        .ok_or_else(|| ReplayError::storage(format!("missing column family {name}")))
}
