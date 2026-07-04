use crate::error::{ReplayError, ReplayResult};
use crate::keys::{
    CF_EPISODES, CF_META, CF_ROW_INDEX, CF_ROWS, META_CONSUMED_ROWS, META_DELETED_FLOOR,
    META_EPISODES_STOPPED, META_FEATURE_SCHEMA, META_NEXT_EPISODE_SEQ, META_PRODUCED_ROWS,
    META_RETAINED_FLOOR, META_ROOT_INFO, META_SCHEMA_VERSION, SCHEMA_VERSION,
    decode_episode_from_row_key, decode_step_from_row_key, decode_u32, decode_u64, decode_u64_key,
    encode_u32, encode_u64, episode_key, row_index_key, row_key,
};
use crate::records::{
    ReplayEpisodeId, ReplayEpisodeRecord, ReplayRootInfo, ReplayRow, validate_episode,
};
use crate::sample::{ReplayRng, SampleConfig};
use gz_features::{FeatureSchema, FeatureSchemaConfig};
use rocksdb::{
    BlockBasedOptions, Cache, ColumnFamilyDescriptor, DB, DBCompressionType, IteratorMode, Options,
    WriteBatch,
};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub struct ReplayStore {
    db: Arc<DB>,
    write_lock: Mutex<()>,
    next_episode_seq: AtomicU64,
    episodes_stopped: AtomicU64,
    produced_rows: AtomicU64,
    consumed_rows: AtomicU64,
    /// Rows below this sequence may be gone; sampling clamps to it.
    retained_floor: AtomicU64,
    retain_rows: Option<u64>,
    /// Episode-weighted EMAs over recent appends (decay 0.99), stored as
    /// f64 bits; zero bits = unseeded. Telemetry only: not persisted.
    cost_ema_bits: AtomicU64,
    len_ema_bits: AtomicU64,
    stop_ema_bits: AtomicU64,
    best_cost_bits: AtomicU64,
}

const OUTCOME_EMA_DECAY: f64 = 0.99;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayCounters {
    pub produced_rows: u64,
    pub consumed_rows: u64,
}

impl ReplayStore {
    pub fn open(path: &Path) -> ReplayResult<Self> {
        Self::open_with_retention(path, None)
    }

    /// `retain_rows` bounds the store: once produced rows exceed the bound
    /// by 25%, whole episodes below `produced - retain_rows` are
    /// range-deleted and the sampling window clamps to the new floor.
    pub fn open_with_retention(path: &Path, retain_rows: Option<u64>) -> ReplayResult<Self> {
        let db = Arc::new(open_db(path)?);
        ensure_schema(&db)?;

        let next_episode_seq = recover_next_episode_seq(&db)?;
        let produced_rows = recover_next_row_seq(&db)?;
        let consumed_rows = read_meta_u64(&db, META_CONSUMED_ROWS)?.unwrap_or(0);
        let episodes_stopped = read_meta_u64(&db, META_EPISODES_STOPPED)?.unwrap_or(0);
        let retained_floor = read_meta_u64(&db, META_RETAINED_FLOOR)?.unwrap_or(0);
        write_meta_u64(&db, META_NEXT_EPISODE_SEQ, next_episode_seq)?;
        write_meta_u64(&db, META_PRODUCED_ROWS, produced_rows)?;

        Ok(Self {
            db,
            write_lock: Mutex::new(()),
            next_episode_seq: AtomicU64::new(next_episode_seq),
            episodes_stopped: AtomicU64::new(episodes_stopped),
            produced_rows: AtomicU64::new(produced_rows),
            consumed_rows: AtomicU64::new(consumed_rows),
            retained_floor: AtomicU64::new(retained_floor),
            retain_rows,
            cost_ema_bits: AtomicU64::new(0),
            len_ema_bits: AtomicU64::new(0),
            stop_ema_bits: AtomicU64::new(0),
            best_cost_bits: AtomicU64::new(0),
        })
    }

    pub fn append_episode(
        &self,
        record: &ReplayEpisodeRecord,
        rows: &[ReplayRow],
    ) -> ReplayResult<ReplayEpisodeId> {
        let _guard = self.write_lock.lock().map_err(ReplayError::storage)?;
        let feature_schema = read_feature_schema(&self.db)?;
        let feature_schema_hash = feature_schema
            .as_ref()
            .map(|config| FeatureSchema::new(config.clone()).map(|schema| schema.hash()))
            .transpose()
            .map_err(|_| ReplayError::InvalidRecord)?;
        validate_episode(record, rows, feature_schema_hash)?;

        let episode_seq = self.next_episode_seq.load(Ordering::Acquire);
        let row_seq = self.produced_rows.load(Ordering::Acquire);
        let next_episode_seq = episode_seq
            .checked_add(1)
            .ok_or_else(|| ReplayError::storage("episode id overflow"))?;
        let produced_rows = row_seq
            .checked_add(rows.len() as u64)
            .ok_or(ReplayError::InvalidRecord)?;
        let id = ReplayEpisodeId::new(episode_seq);

        let episodes = self.cf(CF_EPISODES)?;
        let row_cf = self.cf(CF_ROWS)?;
        let row_index = self.cf(CF_ROW_INDEX)?;
        let meta = self.cf(CF_META)?;
        let mut batch = WriteBatch::default();

        batch.put_cf(
            &episodes,
            episode_key(episode_seq),
            postcard::to_allocvec(record)?,
        );

        for (offset, row) in rows.iter().enumerate() {
            let key = row_key(episode_seq, row.step_index);
            batch.put_cf(&row_cf, key, postcard::to_allocvec(row)?);
            batch.put_cf(
                &row_index,
                row_index_key(row_seq + offset as u64),
                key.as_slice(),
            );
        }

        batch.put_cf(&meta, META_NEXT_EPISODE_SEQ, encode_u64(next_episode_seq));
        batch.put_cf(&meta, META_PRODUCED_ROWS, encode_u64(produced_rows));
        let episodes_stopped =
            self.episodes_stopped.load(Ordering::Acquire) + u64::from(record.outcome.stopped);
        batch.put_cf(&meta, META_EPISODES_STOPPED, encode_u64(episodes_stopped));
        self.db.write(batch)?;
        self.episodes_stopped
            .store(episodes_stopped, Ordering::Release);
        self.next_episode_seq
            .store(next_episode_seq, Ordering::Release);
        self.produced_rows.store(produced_rows, Ordering::Release);
        self.enforce_retention(produced_rows)?;
        let cost = f64::from(-record.outcome.learner_reward);
        self.update_outcome_emas(
            cost,
            rows.len() as f64,
            f64::from(u8::from(record.outcome.stopped)),
        );
        let best = self.best_cost_bits.load(Ordering::Acquire);
        if best == 0 || cost < f64::from_bits(best) {
            self.best_cost_bits.store(cost.to_bits(), Ordering::Release);
        }

        Ok(id)
    }

    /// Runs under the append write lock. Two floors make this safe against
    /// lock-free samplers: keys are only deleted below the floor published
    /// on the PREVIOUS cycle, and any in-flight sampler already clamped to
    /// at least that floor before picking row sequences.
    fn enforce_retention(&self, produced_rows: u64) -> ReplayResult<()> {
        let Some(retain) = self.retain_rows else {
            return Ok(());
        };
        let floor = self.retained_floor.load(Ordering::Acquire);
        if produced_rows.saturating_sub(floor) <= retain + retain / 4 {
            return Ok(());
        }

        let row_index = self.cf(CF_ROW_INDEX)?;
        let target = produced_rows - retain;
        let target_key = self
            .db
            .get_cf(&row_index, row_index_key(target))?
            .ok_or_else(|| ReplayError::storage("missing row index entry at retention target"))?;
        let step = decode_step_from_row_key(&target_key)
            .ok_or_else(|| ReplayError::storage("corrupt row key at retention target"))?;
        // Align the floor to the cutoff episode's first row so episodes are
        // deleted whole.
        let new_floor = target - u64::from(step);
        if new_floor <= floor {
            return Ok(());
        }

        let deleted = read_meta_u64(&self.db, META_DELETED_FLOOR)?.unwrap_or(0);
        let deleted_episode = if deleted == 0 {
            0
        } else {
            let key = self
                .db
                .get_cf(&row_index, row_index_key(deleted))?
                .ok_or_else(|| ReplayError::storage("missing row index entry at deleted floor"))?;
            decode_episode_from_row_key(&key)
                .ok_or_else(|| ReplayError::storage("corrupt row key at deleted floor"))?
        };
        let floor_episode = if floor == 0 {
            0
        } else {
            let key = self
                .db
                .get_cf(&row_index, row_index_key(floor))?
                .ok_or_else(|| ReplayError::storage("missing row index entry at retained floor"))?;
            decode_episode_from_row_key(&key)
                .ok_or_else(|| ReplayError::storage("corrupt row key at retained floor"))?
        };

        let rows = self.cf(CF_ROWS)?;
        let episodes = self.cf(CF_EPISODES)?;
        let mut batch = WriteBatch::default();
        batch.delete_range_cf(&row_index, row_index_key(deleted), row_index_key(floor));
        batch.delete_range_cf(
            &rows,
            row_key(deleted_episode, 0),
            row_key(floor_episode, 0),
        );
        batch.delete_range_cf(
            &episodes,
            episode_key(deleted_episode),
            episode_key(floor_episode),
        );
        let meta = self.cf(CF_META)?;
        batch.put_cf(&meta, META_DELETED_FLOOR, encode_u64(floor));
        batch.put_cf(&meta, META_RETAINED_FLOOR, encode_u64(new_floor));
        self.db.write(batch)?;
        self.retained_floor.store(new_floor, Ordering::Release);

        Ok(())
    }

    pub fn ensure_feature_schema(&self, config: &FeatureSchemaConfig) -> ReplayResult<()> {
        FeatureSchema::new(config.clone()).map_err(|_| ReplayError::InvalidRecord)?;

        let _guard = self.write_lock.lock().map_err(ReplayError::storage)?;
        let Some(stored) = read_feature_schema(&self.db)? else {
            let meta = self.cf(CF_META)?;
            self.db
                .put_cf(
                    &meta,
                    META_FEATURE_SCHEMA,
                    postcard::to_allocvec(&StoredFeatureSchemaConfig::from(config))?,
                )
                .map_err(ReplayError::from)?;
            return Ok(());
        };

        if &stored == config {
            Ok(())
        } else {
            Err(ReplayError::InvalidRecord)
        }
    }

    pub fn feature_schema(&self) -> ReplayResult<Option<FeatureSchemaConfig>> {
        read_feature_schema(&self.db)
    }

    pub fn episode(&self, id: ReplayEpisodeId) -> ReplayResult<Option<ReplayEpisodeRecord>> {
        let episodes = self.cf(CF_EPISODES)?;
        self.db
            .get_cf(&episodes, episode_key(id.get()))?
            .map(|bytes| postcard::from_bytes(&bytes).map_err(ReplayError::from))
            .transpose()
    }

    pub fn sample_rows(
        &self,
        config: SampleConfig,
    ) -> ReplayResult<Vec<(ReplayEpisodeId, ReplayRow)>> {
        // Lock-free against producers: appends commit their WriteBatch before
        // publishing produced_rows, so every sampled row_seq is fully visible.
        let produced = self.produced_rows.load(Ordering::Acquire);
        let floor = self.retained_floor.load(Ordering::Acquire);

        if produced == floor || produced == 0 {
            return Err(ReplayError::Empty);
        }

        let window = config.window_rows.get().min(produced - floor);
        let start = produced - window;
        let mut rng = ReplayRng::new(config.seed);
        let row_index = self.cf(CF_ROW_INDEX)?;
        let rows = self.cf(CF_ROWS)?;
        let mut out = Vec::with_capacity(config.batch.get());

        for _ in 0..config.batch.get() {
            let row_seq = start + rng.next_bounded(window);
            let row_key = self
                .db
                .get_cf(&row_index, row_index_key(row_seq))?
                .ok_or_else(|| ReplayError::storage("missing row index entry"))?;
            let episode_seq = decode_episode_from_row_key(&row_key)
                .ok_or_else(|| ReplayError::storage("corrupt row key"))?;
            let row = self
                .db
                .get_cf(&rows, &row_key)?
                .ok_or_else(|| ReplayError::storage("missing replay row"))?;

            out.push((
                ReplayEpisodeId::new(episode_seq),
                postcard::from_bytes(&row)?,
            ));
        }

        let consumed_rows = self
            .consumed_rows
            .fetch_add(config.batch.get() as u64, Ordering::AcqRel)
            .checked_add(config.batch.get() as u64)
            .ok_or_else(|| ReplayError::storage("consumed row counter overflow"))?;
        write_meta_u64(&self.db, META_CONSUMED_ROWS, consumed_rows)?;

        Ok(out)
    }

    fn update_outcome_emas(&self, cost: f64, len: f64, stopped: f64) {
        for (bits, value) in [
            (&self.cost_ema_bits, cost),
            (&self.len_ema_bits, len),
            (&self.stop_ema_bits, stopped),
        ] {
            let previous = bits.load(Ordering::Acquire);
            let next = if previous == 0 {
                value
            } else {
                OUTCOME_EMA_DECAY * f64::from_bits(previous) + (1.0 - OUTCOME_EMA_DECAY) * value
            };
            bits.store(next.to_bits(), Ordering::Release);
        }
    }

    /// Episode-weighted EMAs over recent appends:
    /// (terminal cost, episode length, stop rate). None until seeded.
    #[must_use]
    pub fn outcome_emas(&self) -> Option<(f64, f64, f64)> {
        let cost = self.cost_ema_bits.load(Ordering::Acquire);
        if cost == 0 {
            return None;
        }
        Some((
            f64::from_bits(cost),
            f64::from_bits(self.len_ema_bits.load(Ordering::Acquire)),
            f64::from_bits(self.stop_ema_bits.load(Ordering::Acquire)),
        ))
    }

    /// Lowest terminal cost of any appended episode. None until seeded.
    #[must_use]
    pub fn best_cost(&self) -> Option<f64> {
        let bits = self.best_cost_bits.load(Ordering::Acquire);
        (bits != 0).then(|| f64::from_bits(bits))
    }

    /// Static root facts for single-graph runs; survives reopen.
    pub fn set_root_info(&self, info: &ReplayRootInfo) -> ReplayResult<()> {
        let meta = self.cf(CF_META)?;
        self.db
            .put_cf(&meta, META_ROOT_INFO, postcard::to_allocvec(info)?)
            .map_err(ReplayError::from)
    }

    pub fn root_info(&self) -> ReplayResult<Option<ReplayRootInfo>> {
        let meta = self.cf(CF_META)?;
        self.db
            .get_cf(&meta, META_ROOT_INFO)?
            .map(|bytes| postcard::from_bytes(&bytes).map_err(ReplayError::from))
            .transpose()
    }

    /// (episodes appended, episodes that ended by selecting STOP).
    #[must_use]
    pub fn episode_counters(&self) -> (u64, u64) {
        (
            self.next_episode_seq.load(Ordering::Acquire),
            self.episodes_stopped.load(Ordering::Acquire),
        )
    }

    #[must_use]
    pub fn counters(&self) -> ReplayCounters {
        ReplayCounters {
            produced_rows: self.produced_rows.load(Ordering::Acquire),
            consumed_rows: self.consumed_rows.load(Ordering::Acquire),
        }
    }

    fn cf(&self, name: &'static str) -> ReplayResult<&rocksdb::ColumnFamily> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| ReplayError::storage(format!("missing column family {name}")))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
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
            expander_degree: config.expander_degree,
            expander_seed: config.expander_seed,
        }
    }
}

fn open_db(path: &Path) -> ReplayResult<DB> {
    let mut options = Options::default();
    options.create_if_missing(true);
    options.create_missing_column_families(true);
    // Selfplay writes tens of MB/s of large rows continuously; defaults
    // (64 MB memtables, 2 background jobs, 8 MB cache) accumulate
    // compaction debt until reads and appends stall mid-run.
    options.increase_parallelism(8);
    options.set_max_background_jobs(8);

    let cache = Cache::new_lru_cache(2 * 1024 * 1024 * 1024);
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
        ColumnFamilyDescriptor::new(CF_ROW_INDEX, index_cf),
    ];

    DB::open_cf_descriptors(&options, path, descriptors).map_err(ReplayError::from)
}

fn ensure_schema(db: &DB) -> ReplayResult<()> {
    let meta = db
        .cf_handle(CF_META)
        .ok_or_else(|| ReplayError::storage("missing meta column family"))?;

    match db.get_cf(&meta, META_SCHEMA_VERSION)? {
        Some(bytes) => {
            if decode_u32(&bytes) == Some(SCHEMA_VERSION) {
                Ok(())
            } else {
                Err(ReplayError::SchemaMismatch)
            }
        }
        None => {
            let mut batch = WriteBatch::default();
            batch.put_cf(&meta, META_SCHEMA_VERSION, encode_u32(SCHEMA_VERSION));
            batch.put_cf(&meta, META_NEXT_EPISODE_SEQ, encode_u64(0));
            batch.put_cf(&meta, META_PRODUCED_ROWS, encode_u64(0));
            batch.put_cf(&meta, META_CONSUMED_ROWS, encode_u64(0));
            db.write(batch).map_err(ReplayError::from)
        }
    }
}

fn recover_next_episode_seq(db: &DB) -> ReplayResult<u64> {
    let episodes = db
        .cf_handle(CF_EPISODES)
        .ok_or_else(|| ReplayError::storage("missing episodes column family"))?;
    let mut iter = db.iterator_cf(&episodes, IteratorMode::End);

    match iter.next().transpose()? {
        Some((key, _)) => decode_u64_key(&key)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| ReplayError::storage("corrupt episode key")),
        None => Ok(0),
    }
}

fn recover_next_row_seq(db: &DB) -> ReplayResult<u64> {
    let row_index = db
        .cf_handle(CF_ROW_INDEX)
        .ok_or_else(|| ReplayError::storage("missing row_index column family"))?;
    let mut iter = db.iterator_cf(&row_index, IteratorMode::End);

    match iter.next().transpose()? {
        Some((key, _)) => decode_u64_key(&key)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| ReplayError::storage("corrupt row index key")),
        None => Ok(0),
    }
}

fn read_meta_u64(db: &DB, key: &[u8]) -> ReplayResult<Option<u64>> {
    let meta = db
        .cf_handle(CF_META)
        .ok_or_else(|| ReplayError::storage("missing meta column family"))?;

    db.get_cf(&meta, key)?
        .map(|bytes| decode_u64(&bytes).ok_or_else(|| ReplayError::storage("corrupt meta u64")))
        .transpose()
}

fn write_meta_u64(db: &DB, key: &[u8], value: u64) -> ReplayResult<()> {
    let meta = db
        .cf_handle(CF_META)
        .ok_or_else(|| ReplayError::storage("missing meta column family"))?;

    db.put_cf(&meta, key, encode_u64(value))
        .map_err(ReplayError::from)
}

fn read_feature_schema(db: &DB) -> ReplayResult<Option<FeatureSchemaConfig>> {
    let meta = db
        .cf_handle(CF_META)
        .ok_or_else(|| ReplayError::storage("missing meta column family"))?;

    db.get_cf(&meta, META_FEATURE_SCHEMA)?
        .map(|bytes| {
            postcard::from_bytes::<StoredFeatureSchemaConfig>(&bytes)
                .map(FeatureSchemaConfig::from)
                .map_err(ReplayError::from)
        })
        .transpose()
}
