use crate::append::{AppendSequences, EpisodeAppend, stage_episodes};
use crate::database::{
    ensure_schema, open_db, read_data_mode, read_engine_identity, read_feature_schema,
    read_meta_u64, recover_next_episode_seq, recover_next_row_seq, write_meta_u64,
};
use crate::error::{ReplayError, ReplayResult};
use crate::keys::{
    CF_EPISODES, CF_META, CF_ROW_INDEX, CF_ROWS, META_COMPLETED_GAMES, META_CONSUMED_ROWS,
    META_DELETED_FLOOR, META_EPISODES_STOPPED, META_NEXT_EPISODE_SEQ, META_PRODUCED_ROWS,
    META_RETAINED_FLOOR, META_SYMMETRIC_BEST_COST, META_SYMMETRIC_COST_MARGIN_EMA,
    META_SYMMETRIC_DRAW_EMA, META_SYMMETRIC_GAMES, META_SYMMETRIC_LEN_MARGIN_EMA,
    META_SYMMETRIC_P1_COST_EMA, META_SYMMETRIC_P1_LEN_EMA, META_SYMMETRIC_P1_WIN_EMA,
    META_SYMMETRIC_P2_COST_EMA, META_SYMMETRIC_P2_LEN_EMA, META_SYMMETRIC_P2_WIN_EMA,
    META_TERMINAL_COST_BEST, META_TERMINAL_COST_EMA, decode_episode_from_row_key,
    decode_step_from_row_key, encode_u64, episode_key, row_index_key, row_key,
};
use crate::records::{
    ReplayEpisodeId, ReplayEpisodeRecord, ReplayRow, ReplayRowKind, StoredReplayRow,
    validate_episode, validate_episode_engine_identity,
};
use crate::sample::{ReplayRng, SampleConfig};
use gz_engine::EngineIdentity;
use gz_features::{FeatureSchema, FeatureSchemaConfig};
use rocksdb::{DB, WriteBatch};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

mod contract;

pub use contract::{ReplayContract, SymmetricRewardTargetScale, SymmetricRewardTieEpsilon};

pub struct ReplayStore {
    db: Arc<DB>,
    write_lock: Mutex<()>,
    consumed_lock: Mutex<()>,
    data_mode: Mutex<Option<ReplayDataMode>>,
    engine_identity: Mutex<Option<EngineIdentity>>,
    next_episode_seq: AtomicU64,
    completed_games: AtomicU64,
    episodes_stopped: AtomicU64,
    produced_rows: AtomicU64,
    consumed_rows: AtomicU64,
    /// Rows below this sequence may be gone; sampling clamps to it.
    retained_floor: AtomicU64,
    retain_rows: Option<u64>,
    /// Episode-weighted EMAs over recent appends (decay 0.99), stored as
    /// f64 bits; zero bits = unseeded. Terminal cost persists across reopen.
    cost_ema_bits: AtomicU64,
    len_ema_bits: AtomicU64,
    stop_ema_bits: AtomicU64,
    /// EMA of value_target > 0 over labeled appends only: the
    /// episode-weighted learner win rate.
    win_ema_bits: AtomicU64,
    /// Episode-weighted accuracy of raw learner value predictions against the
    /// subsequently measured target, split at learner step 40.
    value_sign_early_ema_bits: AtomicU64,
    value_sign_late_ema_bits: AtomicU64,
    /// EMA of admission-to-completion wall seconds, fed by lanes at
    /// episode completion: the async lag's queueing term.
    latency_ema_bits: AtomicU64,
    best_cost_bits: AtomicU64,
    symmetric_metrics: SymmetricMetricAtoms,
}

const OUTCOME_EMA_DECAY: f64 = 0.99;

fn next_ema_bits(previous: u64, value: f64) -> u64 {
    let next = if previous == 0 {
        value
    } else {
        OUTCOME_EMA_DECAY * f64::from_bits(previous) + (1.0 - OUTCOME_EMA_DECAY) * value
    };
    next.to_bits()
}

fn next_best_cost_bits(previous: u64, cost: f64) -> u64 {
    if previous == 0 || cost < f64::from_bits(previous) {
        cost.to_bits()
    } else {
        previous
    }
}

struct SymmetricMetricAtoms {
    games: AtomicU64,
    p1_win_ema_bits: AtomicU64,
    p2_win_ema_bits: AtomicU64,
    draw_ema_bits: AtomicU64,
    p1_cost_ema_bits: AtomicU64,
    p2_cost_ema_bits: AtomicU64,
    cost_margin_ema_bits: AtomicU64,
    p1_len_ema_bits: AtomicU64,
    p2_len_ema_bits: AtomicU64,
    len_margin_ema_bits: AtomicU64,
    best_cost_bits: AtomicU64,
}

#[derive(Clone, Copy)]
struct SymmetricMetricState {
    games: u64,
    p1_win_ema_bits: u64,
    p2_win_ema_bits: u64,
    draw_ema_bits: u64,
    p1_cost_ema_bits: u64,
    p2_cost_ema_bits: u64,
    cost_margin_ema_bits: u64,
    p1_len_ema_bits: u64,
    p2_len_ema_bits: u64,
    len_margin_ema_bits: u64,
    best_cost_bits: u64,
}

impl SymmetricMetricAtoms {
    fn load(db: &DB) -> ReplayResult<Self> {
        Ok(Self {
            games: AtomicU64::new(read_meta_u64(db, META_SYMMETRIC_GAMES)?.unwrap_or(0)),
            p1_win_ema_bits: AtomicU64::new(
                read_meta_u64(db, META_SYMMETRIC_P1_WIN_EMA)?.unwrap_or(0),
            ),
            p2_win_ema_bits: AtomicU64::new(
                read_meta_u64(db, META_SYMMETRIC_P2_WIN_EMA)?.unwrap_or(0),
            ),
            draw_ema_bits: AtomicU64::new(read_meta_u64(db, META_SYMMETRIC_DRAW_EMA)?.unwrap_or(0)),
            p1_cost_ema_bits: AtomicU64::new(
                read_meta_u64(db, META_SYMMETRIC_P1_COST_EMA)?.unwrap_or(0),
            ),
            p2_cost_ema_bits: AtomicU64::new(
                read_meta_u64(db, META_SYMMETRIC_P2_COST_EMA)?.unwrap_or(0),
            ),
            cost_margin_ema_bits: AtomicU64::new(
                read_meta_u64(db, META_SYMMETRIC_COST_MARGIN_EMA)?.unwrap_or(0),
            ),
            p1_len_ema_bits: AtomicU64::new(
                read_meta_u64(db, META_SYMMETRIC_P1_LEN_EMA)?.unwrap_or(0),
            ),
            p2_len_ema_bits: AtomicU64::new(
                read_meta_u64(db, META_SYMMETRIC_P2_LEN_EMA)?.unwrap_or(0),
            ),
            len_margin_ema_bits: AtomicU64::new(
                read_meta_u64(db, META_SYMMETRIC_LEN_MARGIN_EMA)?.unwrap_or(0),
            ),
            best_cost_bits: AtomicU64::new(
                read_meta_u64(db, META_SYMMETRIC_BEST_COST)?.unwrap_or(0),
            ),
        })
    }

    fn next(
        &self,
        primary: (&ReplayEpisodeRecord, &[ReplayRow]),
        secondary: (&ReplayEpisodeRecord, &[ReplayRow]),
    ) -> ReplayResult<SymmetricMetricState> {
        let p1_target = primary
            .0
            .outcome
            .value_target
            .ok_or(ReplayError::InvalidRecord)?;
        let p2_target = secondary
            .0
            .outcome
            .value_target
            .ok_or(ReplayError::InvalidRecord)?;
        if p2_target != -p1_target {
            return Err(ReplayError::InvalidRecord);
        }

        let games = self.games.load(Ordering::Acquire);
        let initialized = games != 0;
        let p1_cost = -f64::from(primary.0.outcome.reward);
        let p2_cost = -f64::from(secondary.0.outcome.reward);
        let p1_len = symmetric_rewrite_count(primary.0) as f64;
        let p2_len = symmetric_rewrite_count(secondary.0) as f64;
        Ok(SymmetricMetricState {
            games: games
                .checked_add(1)
                .ok_or_else(|| ReplayError::storage("symmetric game counter overflow"))?,
            p1_win_ema_bits: next_symmetric_ema_bits(
                self.p1_win_ema_bits.load(Ordering::Acquire),
                initialized,
                f64::from(u8::from(p1_target > 0.0)),
            ),
            p2_win_ema_bits: next_symmetric_ema_bits(
                self.p2_win_ema_bits.load(Ordering::Acquire),
                initialized,
                f64::from(u8::from(p2_target > 0.0)),
            ),
            draw_ema_bits: next_symmetric_ema_bits(
                self.draw_ema_bits.load(Ordering::Acquire),
                initialized,
                f64::from(u8::from(p1_target == 0.0)),
            ),
            p1_cost_ema_bits: next_symmetric_ema_bits(
                self.p1_cost_ema_bits.load(Ordering::Acquire),
                initialized,
                p1_cost,
            ),
            p2_cost_ema_bits: next_symmetric_ema_bits(
                self.p2_cost_ema_bits.load(Ordering::Acquire),
                initialized,
                p2_cost,
            ),
            cost_margin_ema_bits: next_symmetric_ema_bits(
                self.cost_margin_ema_bits.load(Ordering::Acquire),
                initialized,
                (p1_cost - p2_cost).abs(),
            ),
            p1_len_ema_bits: next_symmetric_ema_bits(
                self.p1_len_ema_bits.load(Ordering::Acquire),
                initialized,
                p1_len,
            ),
            p2_len_ema_bits: next_symmetric_ema_bits(
                self.p2_len_ema_bits.load(Ordering::Acquire),
                initialized,
                p2_len,
            ),
            len_margin_ema_bits: next_symmetric_ema_bits(
                self.len_margin_ema_bits.load(Ordering::Acquire),
                initialized,
                (p1_len - p2_len).abs(),
            ),
            best_cost_bits: if initialized {
                let previous = f64::from_bits(self.best_cost_bits.load(Ordering::Acquire));
                previous.min(p1_cost).min(p2_cost).to_bits()
            } else {
                p1_cost.min(p2_cost).to_bits()
            },
        })
    }

    fn stage(meta: &rocksdb::ColumnFamily, batch: &mut WriteBatch, state: SymmetricMetricState) {
        for (key, value) in [
            (META_SYMMETRIC_GAMES, state.games),
            (META_SYMMETRIC_P1_WIN_EMA, state.p1_win_ema_bits),
            (META_SYMMETRIC_P2_WIN_EMA, state.p2_win_ema_bits),
            (META_SYMMETRIC_DRAW_EMA, state.draw_ema_bits),
            (META_SYMMETRIC_P1_COST_EMA, state.p1_cost_ema_bits),
            (META_SYMMETRIC_P2_COST_EMA, state.p2_cost_ema_bits),
            (META_SYMMETRIC_COST_MARGIN_EMA, state.cost_margin_ema_bits),
            (META_SYMMETRIC_P1_LEN_EMA, state.p1_len_ema_bits),
            (META_SYMMETRIC_P2_LEN_EMA, state.p2_len_ema_bits),
            (META_SYMMETRIC_LEN_MARGIN_EMA, state.len_margin_ema_bits),
            (META_SYMMETRIC_BEST_COST, state.best_cost_bits),
        ] {
            batch.put_cf(meta, key, encode_u64(value));
        }
    }

    fn publish(&self, state: SymmetricMetricState) {
        self.p1_win_ema_bits
            .store(state.p1_win_ema_bits, Ordering::Release);
        self.p2_win_ema_bits
            .store(state.p2_win_ema_bits, Ordering::Release);
        self.draw_ema_bits
            .store(state.draw_ema_bits, Ordering::Release);
        self.p1_cost_ema_bits
            .store(state.p1_cost_ema_bits, Ordering::Release);
        self.p2_cost_ema_bits
            .store(state.p2_cost_ema_bits, Ordering::Release);
        self.cost_margin_ema_bits
            .store(state.cost_margin_ema_bits, Ordering::Release);
        self.p1_len_ema_bits
            .store(state.p1_len_ema_bits, Ordering::Release);
        self.p2_len_ema_bits
            .store(state.p2_len_ema_bits, Ordering::Release);
        self.len_margin_ema_bits
            .store(state.len_margin_ema_bits, Ordering::Release);
        self.best_cost_bits
            .store(state.best_cost_bits, Ordering::Release);
        self.games.store(state.games, Ordering::Release);
    }

    fn snapshot(&self) -> Option<SymmetricSelfplayMetrics> {
        (self.games.load(Ordering::Acquire) != 0).then(|| {
            let p1_win_rate_ema = f64::from_bits(self.p1_win_ema_bits.load(Ordering::Acquire));
            let p2_win_rate_ema = f64::from_bits(self.p2_win_ema_bits.load(Ordering::Acquire));
            let p1_terminal_cost_ema =
                f64::from_bits(self.p1_cost_ema_bits.load(Ordering::Acquire));
            let p2_terminal_cost_ema =
                f64::from_bits(self.p2_cost_ema_bits.load(Ordering::Acquire));
            let p1_episode_len_ema = f64::from_bits(self.p1_len_ema_bits.load(Ordering::Acquire));
            let p2_episode_len_ema = f64::from_bits(self.p2_len_ema_bits.load(Ordering::Acquire));
            SymmetricSelfplayMetrics {
                p1_win_rate_ema,
                p2_win_rate_ema,
                draw_rate_ema: f64::from_bits(self.draw_ema_bits.load(Ordering::Acquire)),
                seat_advantage_ema: p1_win_rate_ema - p2_win_rate_ema,
                p1_terminal_cost_ema,
                p2_terminal_cost_ema,
                mean_terminal_cost_ema: 0.5 * (p1_terminal_cost_ema + p2_terminal_cost_ema),
                terminal_cost_margin_ema: f64::from_bits(
                    self.cost_margin_ema_bits.load(Ordering::Acquire),
                ),
                terminal_cost_best: f64::from_bits(self.best_cost_bits.load(Ordering::Acquire)),
                p1_episode_len_ema,
                p2_episode_len_ema,
                game_len_ema: p1_episode_len_ema + p2_episode_len_ema,
                episode_len_margin_ema: f64::from_bits(
                    self.len_margin_ema_bits.load(Ordering::Acquire),
                ),
            }
        })
    }
}

fn symmetric_rewrite_count(record: &ReplayEpisodeRecord) -> usize {
    record
        .steps
        .iter()
        .filter(|step| {
            matches!(
                step.action,
                gz_engine::PortableSearchActionRef::Candidate(_)
            )
        })
        .count()
}

fn next_symmetric_ema_bits(previous: u64, initialized: bool, value: f64) -> u64 {
    if initialized {
        (OUTCOME_EMA_DECAY * f64::from_bits(previous) + (1.0 - OUTCOME_EMA_DECAY) * value).to_bits()
    } else {
        value.to_bits()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayCounters {
    pub produced_rows: u64,
    pub consumed_rows: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymmetricSelfplayMetrics {
    pub p1_win_rate_ema: f64,
    pub p2_win_rate_ema: f64,
    pub draw_rate_ema: f64,
    pub seat_advantage_ema: f64,
    pub p1_terminal_cost_ema: f64,
    pub p2_terminal_cost_ema: f64,
    pub mean_terminal_cost_ema: f64,
    pub terminal_cost_margin_ema: f64,
    pub terminal_cost_best: f64,
    pub p1_episode_len_ema: f64,
    pub p2_episode_len_ema: f64,
    pub game_len_ema: f64,
    pub episode_len_margin_ema: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayDataMode {
    Standard,
    SymmetricSelfplay,
    SymmetricSelfplayStop,
}

impl ReplayDataMode {
    #[must_use]
    pub const fn is_symmetric_selfplay(self) -> bool {
        matches!(self, Self::SymmetricSelfplay | Self::SymmetricSelfplayStop)
    }

    #[must_use]
    pub const fn symmetric_stop_enabled(self) -> bool {
        matches!(self, Self::SymmetricSelfplayStop)
    }

    fn bytes(self) -> Vec<u8> {
        match self {
            Self::Standard => b"standard-v1".to_vec(),
            Self::SymmetricSelfplay => b"symmetric-selfplay-v1".to_vec(),
            Self::SymmetricSelfplayStop => b"symmetric-selfplay-v2".to_vec(),
        }
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Option<Self> {
        match bytes {
            b"standard-v1" => Some(Self::Standard),
            b"symmetric-selfplay-v1" => Some(Self::SymmetricSelfplay),
            b"symmetric-selfplay-v2" => Some(Self::SymmetricSelfplayStop),
            _ => None,
        }
    }
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
        let completed_games = read_meta_u64(&db, META_COMPLETED_GAMES)?.unwrap_or(next_episode_seq);
        let episodes_stopped = read_meta_u64(&db, META_EPISODES_STOPPED)?.unwrap_or(0);
        let retained_floor = read_meta_u64(&db, META_RETAINED_FLOOR)?.unwrap_or(0);
        let cost_ema_bits = read_meta_u64(&db, META_TERMINAL_COST_EMA)?.unwrap_or(0);
        let best_cost_bits = read_meta_u64(&db, META_TERMINAL_COST_BEST)?.unwrap_or(0);
        let symmetric_metrics = SymmetricMetricAtoms::load(&db)?;
        let data_mode = read_data_mode(&db)?;
        let engine_identity = read_engine_identity(&db)?;
        write_meta_u64(&db, META_NEXT_EPISODE_SEQ, next_episode_seq)?;
        write_meta_u64(&db, META_PRODUCED_ROWS, produced_rows)?;
        write_meta_u64(&db, META_COMPLETED_GAMES, completed_games)?;

        Ok(Self {
            db,
            write_lock: Mutex::new(()),
            consumed_lock: Mutex::new(()),
            data_mode: Mutex::new(data_mode),
            engine_identity: Mutex::new(engine_identity),
            next_episode_seq: AtomicU64::new(next_episode_seq),
            completed_games: AtomicU64::new(completed_games),
            episodes_stopped: AtomicU64::new(episodes_stopped),
            produced_rows: AtomicU64::new(produced_rows),
            consumed_rows: AtomicU64::new(consumed_rows),
            retained_floor: AtomicU64::new(retained_floor),
            retain_rows,
            cost_ema_bits: AtomicU64::new(cost_ema_bits),
            win_ema_bits: AtomicU64::new(0),
            value_sign_early_ema_bits: AtomicU64::new(0),
            value_sign_late_ema_bits: AtomicU64::new(0),
            latency_ema_bits: AtomicU64::new(0),
            len_ema_bits: AtomicU64::new(0),
            stop_ema_bits: AtomicU64::new(0),
            best_cost_bits: AtomicU64::new(best_cost_bits),
            symmetric_metrics,
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
        let data_mode = self.data_mode()?;
        if data_mode.is_symmetric_selfplay() {
            return Err(ReplayError::InvalidRecord);
        }
        if let Some(identity) = *self.engine_identity.lock().map_err(ReplayError::storage)? {
            validate_episode_engine_identity(record, identity)?;
        }
        validate_episode(record, rows, feature_schema_hash, data_mode)?;

        let sequences = AppendSequences {
            next_episode: self.next_episode_seq.load(Ordering::Acquire),
            next_row: self.produced_rows.load(Ordering::Acquire),
        };
        let id = ReplayEpisodeId::new(sequences.next_episode);
        let completed_games = self
            .completed_games
            .load(Ordering::Acquire)
            .checked_add(1)
            .ok_or_else(|| ReplayError::storage("completed game counter overflow"))?;
        let cost = f64::from(-record.outcome.reward);
        let cost_ema_bits = next_ema_bits(self.cost_ema_bits.load(Ordering::Acquire), cost);
        let best_cost_bits = next_best_cost_bits(self.best_cost_bits.load(Ordering::Acquire), cost);

        let meta = self.cf(CF_META)?;
        let mut batch = WriteBatch::default();
        let sequences = stage_episodes(
            &self.db,
            &mut batch,
            &[EpisodeAppend { record, rows }],
            sequences,
        )?;

        batch.put_cf(
            &meta,
            META_NEXT_EPISODE_SEQ,
            encode_u64(sequences.next_episode),
        );
        batch.put_cf(&meta, META_PRODUCED_ROWS, encode_u64(sequences.next_row));
        batch.put_cf(&meta, META_COMPLETED_GAMES, encode_u64(completed_games));
        batch.put_cf(&meta, META_TERMINAL_COST_EMA, encode_u64(cost_ema_bits));
        batch.put_cf(&meta, META_TERMINAL_COST_BEST, encode_u64(best_cost_bits));
        let episodes_stopped =
            self.episodes_stopped.load(Ordering::Acquire) + u64::from(record.outcome.stopped);
        batch.put_cf(&meta, META_EPISODES_STOPPED, encode_u64(episodes_stopped));
        self.db.write(batch)?;
        self.episodes_stopped
            .store(episodes_stopped, Ordering::Release);
        self.next_episode_seq
            .store(sequences.next_episode, Ordering::Release);
        self.completed_games
            .store(completed_games, Ordering::Release);
        self.produced_rows
            .store(sequences.next_row, Ordering::Release);
        self.cost_ema_bits.store(cost_ema_bits, Ordering::Release);
        self.best_cost_bits.store(best_cost_bits, Ordering::Release);
        self.enforce_retention(sequences.next_row)?;
        self.update_ema(&self.len_ema_bits, rows.len() as f64);
        self.update_ema(
            &self.stop_ema_bits,
            f64::from(u8::from(record.outcome.stopped)),
        );
        if let Some(value_target) = record.outcome.value_target {
            // Stored biased by +1.0: an honest all-loss EMA of 0.0 would
            // collide with the zero-bits unseeded sentinel.
            self.update_ema(
                &self.win_ema_bits,
                f64::from(u8::from(value_target > 0.0)) + 1.0,
            );
        }
        Ok(id)
    }

    /// Atomically appends both perspectives of one paired game. The primary
    /// record supplies episode-level telemetry; row and
    /// storage counters include both records, while game counters advance once.
    pub fn append_episode_pair(
        &self,
        primary: (&ReplayEpisodeRecord, &[ReplayRow]),
        secondary: (&ReplayEpisodeRecord, &[ReplayRow]),
    ) -> ReplayResult<(ReplayEpisodeId, ReplayEpisodeId)> {
        let _guard = self.write_lock.lock().map_err(ReplayError::storage)?;
        let feature_schema = read_feature_schema(&self.db)?;
        let feature_schema_hash = feature_schema
            .as_ref()
            .map(|config| FeatureSchema::new(config.clone()).map(|schema| schema.hash()))
            .transpose()
            .map_err(|_| ReplayError::InvalidRecord)?;
        let data_mode = self.data_mode()?;
        if let Some(identity) = *self.engine_identity.lock().map_err(ReplayError::storage)? {
            validate_episode_engine_identity(primary.0, identity)?;
            validate_episode_engine_identity(secondary.0, identity)?;
        }
        validate_episode(primary.0, primary.1, feature_schema_hash, data_mode)?;
        validate_episode(secondary.0, secondary.1, feature_schema_hash, data_mode)?;
        let symmetric_metrics = data_mode
            .is_symmetric_selfplay()
            .then(|| self.symmetric_metrics.next(primary, secondary))
            .transpose()?;

        let sequences = AppendSequences {
            next_episode: self.next_episode_seq.load(Ordering::Acquire),
            next_row: self.produced_rows.load(Ordering::Acquire),
        };
        let first_seq = sequences.next_episode;
        let second_seq = first_seq
            .checked_add(1)
            .ok_or_else(|| ReplayError::storage("episode id overflow"))?;
        let completed_games = self
            .completed_games
            .load(Ordering::Acquire)
            .checked_add(1)
            .ok_or_else(|| ReplayError::storage("completed game counter overflow"))?;
        let cost = f64::from(-primary.0.outcome.reward);
        let cost_ema_bits = next_ema_bits(self.cost_ema_bits.load(Ordering::Acquire), cost);
        let best_cost_bits = next_best_cost_bits(self.best_cost_bits.load(Ordering::Acquire), cost);

        let meta = self.cf(CF_META)?;
        let mut batch = WriteBatch::default();
        let sequences = stage_episodes(
            &self.db,
            &mut batch,
            &[
                EpisodeAppend {
                    record: primary.0,
                    rows: primary.1,
                },
                EpisodeAppend {
                    record: secondary.0,
                    rows: secondary.1,
                },
            ],
            sequences,
        )?;

        let episodes_stopped =
            self.episodes_stopped.load(Ordering::Acquire) + u64::from(primary.0.outcome.stopped);
        batch.put_cf(
            &meta,
            META_NEXT_EPISODE_SEQ,
            encode_u64(sequences.next_episode),
        );
        batch.put_cf(&meta, META_PRODUCED_ROWS, encode_u64(sequences.next_row));
        batch.put_cf(&meta, META_COMPLETED_GAMES, encode_u64(completed_games));
        batch.put_cf(&meta, META_EPISODES_STOPPED, encode_u64(episodes_stopped));
        batch.put_cf(&meta, META_TERMINAL_COST_EMA, encode_u64(cost_ema_bits));
        batch.put_cf(&meta, META_TERMINAL_COST_BEST, encode_u64(best_cost_bits));
        if let Some(metrics) = symmetric_metrics {
            SymmetricMetricAtoms::stage(meta, &mut batch, metrics);
        }
        self.db.write(batch)?;

        self.next_episode_seq
            .store(sequences.next_episode, Ordering::Release);
        self.completed_games
            .store(completed_games, Ordering::Release);
        self.episodes_stopped
            .store(episodes_stopped, Ordering::Release);
        self.produced_rows
            .store(sequences.next_row, Ordering::Release);
        self.cost_ema_bits.store(cost_ema_bits, Ordering::Release);
        self.best_cost_bits.store(best_cost_bits, Ordering::Release);
        if let Some(metrics) = symmetric_metrics {
            self.symmetric_metrics.publish(metrics);
        }
        self.enforce_retention(sequences.next_row)?;

        self.update_ema(&self.len_ema_bits, primary.1.len() as f64);
        self.update_ema(
            &self.stop_ema_bits,
            f64::from(u8::from(primary.0.outcome.stopped)),
        );
        if let Some(value_target) = primary.0.outcome.value_target {
            self.update_ema(
                &self.win_ema_bits,
                f64::from(u8::from(value_target > 0.0)) + 1.0,
            );
        }
        Ok((
            ReplayEpisodeId::new(first_seq),
            ReplayEpisodeId::new(second_seq),
        ))
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

    pub fn feature_schema(&self) -> ReplayResult<Option<FeatureSchemaConfig>> {
        read_feature_schema(&self.db)
    }

    pub fn engine_identity(&self) -> ReplayResult<Option<EngineIdentity>> {
        self.engine_identity
            .lock()
            .map(|identity| *identity)
            .map_err(ReplayError::storage)
    }

    pub fn episode(&self, id: ReplayEpisodeId) -> ReplayResult<Option<ReplayEpisodeRecord>> {
        let episodes = self.cf(CF_EPISODES)?;
        self.db
            .get_cf(&episodes, episode_key(id.get()))?
            .map(|bytes| postcard::from_bytes(&bytes).map_err(ReplayError::from))
            .transpose()
    }

    /// Exclusive upper bound of assigned replay episode IDs. Retention may
    /// remove records below this bound.
    #[must_use]
    pub fn episode_sequence_end(&self) -> u64 {
        self.next_episode_seq.load(Ordering::Acquire)
    }

    pub fn sample_rows(
        &self,
        config: SampleConfig,
    ) -> ReplayResult<Vec<(ReplayEpisodeId, ReplayRow)>> {
        let out = self.sample_rows_inner(config)?;
        self.record_consumed(config.batch.get() as u64)?;
        Ok(out)
    }

    /// Samples replay rows without advancing learner-consumption accounting.
    /// Intended for offline diagnostics, not trainer input.
    pub fn inspect_rows(
        &self,
        config: SampleConfig,
    ) -> ReplayResult<Vec<(ReplayEpisodeId, ReplayRow)>> {
        self.sample_rows_inner(config)
    }

    /// Samples one terminal row from uniformly sampled retained episodes.
    /// The row window determines the oldest eligible episode; sampling itself
    /// is episode-uniform so long trajectories do not dominate value training.
    pub fn sample_terminal_rows(
        &self,
        config: SampleConfig,
    ) -> ReplayResult<Vec<(ReplayEpisodeId, ReplayRow)>> {
        let out = self.sample_terminal_rows_inner(config)?;
        self.record_consumed(config.batch.get() as u64)?;
        Ok(out)
    }

    fn sample_rows_inner(
        &self,
        config: SampleConfig,
    ) -> ReplayResult<Vec<(ReplayEpisodeId, ReplayRow)>> {
        // Lock-free against producers: appends commit all indexes and rows in
        // one WriteBatch before publishing the corresponding sequence count.
        // Load the floor before its producer count. Retention publishes the
        // floor after the append's producer count, so this ordering cannot
        // combine a newer floor with an older count.
        let floor = self.retained_floor.load(Ordering::Acquire);
        let produced = self.produced_rows.load(Ordering::Acquire);

        if produced <= floor || produced == 0 {
            return Err(ReplayError::Empty);
        }

        let window = config.window_rows.get().min(produced - floor);
        let start = produced - window;
        let mut rng = ReplayRng::new(config.seed);
        let batch = config.batch.get();

        // Two batched MultiGet phases instead of 2 x batch sequential
        // point gets: index keys resolve to row keys, row keys resolve to
        // rows. Results return in request order, so sampling stays
        // bit-identical to the sequential loop for a given seed.
        let sequences = (0..batch)
            .map(|_| start + rng.next_bounded(window))
            .collect::<Vec<_>>();
        let out = self.read_row_sequences(&sequences)?;
        Ok(out)
    }

    fn sample_terminal_rows_inner(
        &self,
        config: SampleConfig,
    ) -> ReplayResult<Vec<(ReplayEpisodeId, ReplayRow)>> {
        let floor = self.retained_floor.load(Ordering::Acquire);
        let produced = self.produced_rows.load(Ordering::Acquire);
        if produced <= floor || produced == 0 {
            return Err(ReplayError::Empty);
        }

        let window = config.window_rows.get().min(produced - floor);
        let first_row = produced - window;
        let first_episode = self.episode_for_row_sequence(first_row)?;
        let episode_end = self.next_episode_seq.load(Ordering::Acquire);
        if first_episode >= episode_end {
            return Err(ReplayError::Empty);
        }

        let mut rng = ReplayRng::new(config.seed);
        let episode_count = episode_end - first_episode;
        let sequences = (0..config.batch.get())
            .map(|_| first_episode + rng.next_bounded(episode_count))
            .collect::<Vec<_>>();
        self.read_terminal_episode_sequences(&sequences)
    }

    fn episode_for_row_sequence(&self, sequence: u64) -> ReplayResult<u64> {
        let row_index = self.cf(CF_ROW_INDEX)?;
        let key = self
            .db
            .get_cf(&row_index, row_index_key(sequence))?
            .ok_or_else(|| ReplayError::storage("missing row index entry"))?;
        decode_episode_from_row_key(&key).ok_or_else(|| ReplayError::storage("corrupt row key"))
    }

    fn read_terminal_episode_sequences(
        &self,
        sequences: &[u64],
    ) -> ReplayResult<Vec<(ReplayEpisodeId, ReplayRow)>> {
        let episodes = self.cf(CF_EPISODES)?;
        let rows = self.cf(CF_ROWS)?;
        let episode_keys = sequences
            .iter()
            .copied()
            .map(episode_key)
            .collect::<Vec<_>>();
        let mut row_keys = Vec::with_capacity(sequences.len());
        for (sequence, result) in sequences.iter().copied().zip(self.db.batched_multi_get_cf(
            &episodes,
            &episode_keys,
            false,
        )) {
            let bytes = result?.ok_or_else(|| ReplayError::storage("missing episode record"))?;
            let record = postcard::from_bytes::<ReplayEpisodeRecord>(&bytes)?;
            if record.row_count as usize != record.steps.len() + 1 {
                return Err(ReplayError::storage("episode has no terminal row"));
            }
            let terminal_step = u32::try_from(record.steps.len())
                .map_err(|_| ReplayError::storage("terminal step overflow"))?;
            row_keys.push(row_key(sequence, terminal_step));
        }

        let mut out = Vec::with_capacity(sequences.len());
        for (sequence, result) in sequences.iter().copied().zip(self.db.batched_multi_get_cf(
            &rows,
            row_keys.iter(),
            false,
        )) {
            let bytes = result?.ok_or_else(|| ReplayError::storage("missing terminal row"))?;
            let row = postcard::from_bytes::<StoredReplayRow>(&bytes)?.into_row();
            if row.kind != ReplayRowKind::Terminal {
                return Err(ReplayError::storage("terminal row kind mismatch"));
            }
            out.push((ReplayEpisodeId::new(sequence), row));
        }
        Ok(out)
    }

    fn read_row_sequences(
        &self,
        sequences: &[u64],
    ) -> ReplayResult<Vec<(ReplayEpisodeId, ReplayRow)>> {
        let row_index = self.cf(CF_ROW_INDEX)?;
        let rows = self.cf(CF_ROWS)?;
        let index_keys = sequences
            .iter()
            .copied()
            .map(row_index_key)
            .collect::<Vec<_>>();
        let mut row_keys = Vec::with_capacity(sequences.len());
        let mut episode_seqs = Vec::with_capacity(sequences.len());
        for result in self.db.batched_multi_get_cf(&row_index, &index_keys, false) {
            let row_key = result?.ok_or_else(|| ReplayError::storage("missing row index entry"))?;
            let episode_seq = decode_episode_from_row_key(&row_key)
                .ok_or_else(|| ReplayError::storage("corrupt row key"))?;
            episode_seqs.push(episode_seq);
            row_keys.push(row_key);
        }

        let mut out = Vec::with_capacity(sequences.len());
        for (episode_seq, result) in episode_seqs.into_iter().zip(self.db.batched_multi_get_cf(
            &rows,
            row_keys.iter(),
            false,
        )) {
            let row = result?.ok_or_else(|| ReplayError::storage("missing replay row"))?;
            out.push((
                ReplayEpisodeId::new(episode_seq),
                postcard::from_bytes::<StoredReplayRow>(&row)?.into_row(),
            ));
        }
        Ok(out)
    }

    fn record_consumed(&self, count: u64) -> ReplayResult<()> {
        // Concurrent sample sessions may finish out of order. Serialize the
        // metadata update so a late write cannot persist an older counter.
        let _guard = self.consumed_lock.lock().map_err(ReplayError::storage)?;
        let consumed_rows = self
            .consumed_rows
            .load(Ordering::Acquire)
            .checked_add(count)
            .ok_or_else(|| ReplayError::storage("consumed row counter overflow"))?;
        write_meta_u64(&self.db, META_CONSUMED_ROWS, consumed_rows)?;
        self.consumed_rows.store(consumed_rows, Ordering::Release);
        Ok(())
    }

    fn update_ema(&self, bits: &AtomicU64, value: f64) {
        let previous = bits.load(Ordering::Acquire);
        bits.store(next_ema_bits(previous, value), Ordering::Release);
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

    /// Observe one episode's admission-to-completion wall time. Called
    /// by lanes at completion (dropped episodes included: their latency
    /// is real even when their rows are not stored).
    pub fn observe_episode_latency(&self, seconds: f64) {
        if seconds > 0.0 && seconds.is_finite() {
            self.update_ema(&self.latency_ema_bits, seconds);
        }
    }

    /// EMA of episode wall-clock latency in seconds. None until seeded.
    #[must_use]
    pub fn episode_latency_ema(&self) -> Option<f64> {
        let bits = self.latency_ema_bits.load(Ordering::Acquire);
        (bits != 0).then(|| f64::from_bits(bits))
    }

    /// Episode-weighted EMA of "learner beat its reference" over labeled
    /// appends. None until a labeled episode lands.
    #[must_use]
    pub fn win_rate_ema(&self) -> Option<f64> {
        let bits = self.win_ema_bits.load(Ordering::Acquire);
        (bits != 0).then(|| f64::from_bits(bits) - 1.0)
    }

    pub fn observe_value_sign_accuracy(&self, early: Option<f64>, late: Option<f64>) {
        for (bits, value) in [
            (&self.value_sign_early_ema_bits, early),
            (&self.value_sign_late_ema_bits, late),
        ] {
            if let Some(value) =
                value.filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
            {
                // Offset by one so an honest 0% accuracy remains distinct from
                // the zero-bits unseeded sentinel.
                self.update_ema(bits, value + 1.0);
            }
        }
    }

    #[must_use]
    pub fn value_sign_accuracy_emas(&self) -> (Option<f64>, Option<f64>) {
        let decode = |bits: &AtomicU64| {
            let bits = bits.load(Ordering::Acquire);
            (bits != 0).then(|| f64::from_bits(bits) - 1.0)
        };
        (
            decode(&self.value_sign_early_ema_bits),
            decode(&self.value_sign_late_ema_bits),
        )
    }

    /// Lowest terminal cost of any appended episode. None until seeded.
    #[must_use]
    pub fn best_cost(&self) -> Option<f64> {
        let bits = self.best_cost_bits.load(Ordering::Acquire);
        (bits != 0).then(|| f64::from_bits(bits))
    }

    #[must_use]
    pub fn symmetric_selfplay_metrics(&self) -> Option<SymmetricSelfplayMetrics> {
        self.symmetric_metrics.snapshot()
    }

    /// (completed games, primary learner episodes that selected STOP).
    #[must_use]
    pub fn episode_counters(&self) -> (u64, u64) {
        (
            self.completed_games.load(Ordering::Acquire),
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

    pub fn data_mode(&self) -> ReplayResult<ReplayDataMode> {
        Ok(self
            .data_mode
            .lock()
            .map_err(ReplayError::storage)?
            .unwrap_or(ReplayDataMode::Standard))
    }
}
