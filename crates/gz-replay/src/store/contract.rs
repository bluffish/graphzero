use super::{ReplayDataMode, ReplayStore};
use crate::database::{
    read_feature_schema, read_symmetric_reward_target_scale, read_symmetric_reward_tie_epsilon,
    stored_engine_identity, stored_feature_schema, stored_symmetric_reward_target_scale,
    stored_symmetric_reward_tie_epsilon,
};
use crate::error::{ReplayError, ReplayResult};
use crate::keys::{
    CF_EPISODES, CF_META, META_DATA_MODE, META_ENGINE_IDENTITY, META_FEATURE_SCHEMA,
    META_SYMMETRIC_REWARD_TARGET_SCALE, META_SYMMETRIC_REWARD_TIE_EPSILON,
};
use crate::records::{ReplayEpisodeRecord, validate_episode_engine_identity};
use gz_engine::EngineIdentity;
use gz_features::{FeatureSchema, FeatureSchemaConfig};
use rocksdb::{IteratorMode, WriteBatch};
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymmetricRewardTieEpsilon(f32);

impl SymmetricRewardTieEpsilon {
    pub const ZERO: Self = Self(0.0);

    #[must_use]
    pub fn new(value: f32) -> Option<Self> {
        (value.is_finite() && value >= 0.0).then_some(Self(if value == 0.0 { 0.0 } else { value }))
    }

    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymmetricRewardTargetScale(f32);

impl SymmetricRewardTargetScale {
    pub const HARD: Self = Self(0.0);

    #[must_use]
    pub fn new(value: f32) -> Option<Self> {
        (value.is_finite() && value >= 0.0).then_some(Self(if value == 0.0 { 0.0 } else { value }))
    }

    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }

    #[must_use]
    pub const fn is_graded(self) -> bool {
        self.0 > 0.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayContract {
    pub data_mode: ReplayDataMode,
    pub feature_schema: Option<FeatureSchemaConfig>,
    pub engine_identity: EngineIdentity,
    pub symmetric_reward_tie_epsilon: Option<SymmetricRewardTieEpsilon>,
    pub symmetric_reward_target_scale: Option<SymmetricRewardTargetScale>,
}

impl ReplayContract {
    #[must_use]
    pub fn featurized(
        data_mode: ReplayDataMode,
        feature_schema: FeatureSchemaConfig,
        engine_identity: EngineIdentity,
    ) -> Self {
        Self {
            data_mode,
            feature_schema: Some(feature_schema),
            engine_identity,
            symmetric_reward_tie_epsilon: data_mode
                .is_symmetric_selfplay()
                .then_some(SymmetricRewardTieEpsilon::ZERO),
            symmetric_reward_target_scale: data_mode
                .is_symmetric_selfplay()
                .then_some(SymmetricRewardTargetScale::HARD),
        }
    }

    #[must_use]
    pub fn unfeaturized(data_mode: ReplayDataMode, engine_identity: EngineIdentity) -> Self {
        Self {
            data_mode,
            feature_schema: None,
            engine_identity,
            symmetric_reward_tie_epsilon: data_mode
                .is_symmetric_selfplay()
                .then_some(SymmetricRewardTieEpsilon::ZERO),
            symmetric_reward_target_scale: data_mode
                .is_symmetric_selfplay()
                .then_some(SymmetricRewardTargetScale::HARD),
        }
    }

    #[must_use]
    pub fn with_symmetric_reward_tie_epsilon(mut self, epsilon: SymmetricRewardTieEpsilon) -> Self {
        self.symmetric_reward_tie_epsilon = Some(epsilon);
        self
    }

    #[must_use]
    pub fn with_symmetric_reward_target_scale(mut self, scale: SymmetricRewardTargetScale) -> Self {
        self.symmetric_reward_target_scale = Some(scale);
        self
    }
}

impl ReplayStore {
    /// Validate and bind all replay semantics in one RocksDB write.
    /// Existing unbound stores are migrated only after retained episodes pass
    /// identity checks; a feature schema cannot be added after rows exist.
    pub fn ensure_contract(&self, contract: &ReplayContract) -> ReplayResult<()> {
        if contract.data_mode.is_symmetric_selfplay()
            != contract.symmetric_reward_tie_epsilon.is_some()
            || contract.data_mode.is_symmetric_selfplay()
                != contract.symmetric_reward_target_scale.is_some()
            || contract
                .symmetric_reward_target_scale
                .is_some_and(|scale| scale.is_graded())
                && contract
                    .symmetric_reward_tie_epsilon
                    .is_some_and(|epsilon| epsilon != SymmetricRewardTieEpsilon::ZERO)
        {
            return Err(ReplayError::InvalidRecord);
        }
        if let Some(config) = &contract.feature_schema {
            FeatureSchema::new(config.clone()).map_err(|_| ReplayError::InvalidRecord)?;
        }

        let _write = self.write_lock.lock().map_err(ReplayError::storage)?;
        let mut data_mode = self.data_mode.lock().map_err(ReplayError::storage)?;
        let mut engine_identity = self.engine_identity.lock().map_err(ReplayError::storage)?;
        let stored_schema = read_feature_schema(&self.db)?;
        let stored_reward_tie_epsilon = read_symmetric_reward_tie_epsilon(&self.db)?;
        let stored_reward_target_scale = read_symmetric_reward_target_scale(&self.db)?;

        match (&stored_schema, &contract.feature_schema) {
            (Some(stored), Some(expected)) if stored == expected => {}
            (None, Some(_)) if self.produced_rows.load(Ordering::Acquire) == 0 => {}
            (None, None) => {}
            _ => return Err(ReplayError::InvalidRecord),
        }

        if let Some(stored) = *data_mode {
            if stored != contract.data_mode {
                return Err(ReplayError::DataModeMismatch);
            }
        } else if self.next_episode_seq.load(Ordering::Acquire) > 0
            && contract.data_mode != ReplayDataMode::Standard
        {
            return Err(ReplayError::DataModeMismatch);
        }

        if let Some(stored) = *engine_identity {
            if stored != contract.engine_identity {
                return Err(ReplayError::EngineIdentityMismatch);
            }
        } else {
            let episodes = self.cf(CF_EPISODES)?;
            for item in self.db.iterator_cf(&episodes, IteratorMode::Start) {
                let (_, bytes) = item?;
                let record: ReplayEpisodeRecord = postcard::from_bytes(&bytes)?;
                validate_episode_engine_identity(&record, contract.engine_identity)?;
            }
        }

        match (
            stored_reward_tie_epsilon,
            contract.symmetric_reward_tie_epsilon,
        ) {
            (Some(stored), Some(expected)) if stored == expected => {}
            (None, Some(SymmetricRewardTieEpsilon::ZERO)) => {
                // Symmetric stores created before this contract key used exact
                // reward comparison, which is epsilon zero.
            }
            (None, Some(_)) if self.produced_rows.load(Ordering::Acquire) == 0 => {}
            (None, None) => {}
            _ => return Err(ReplayError::TargetContractMismatch),
        }

        match (
            stored_reward_target_scale,
            contract.symmetric_reward_target_scale,
        ) {
            (Some(stored), Some(expected)) if stored == expected => {}
            (None, Some(SymmetricRewardTargetScale::HARD)) => {
                // Stores created before graded targets used hard outcomes.
            }
            (None, Some(_)) if self.produced_rows.load(Ordering::Acquire) == 0 => {}
            (None, None) => {}
            _ => return Err(ReplayError::TargetContractMismatch),
        }

        let meta = self.cf(CF_META)?;
        let mut batch = WriteBatch::default();
        if data_mode.is_none() {
            batch.put_cf(&meta, META_DATA_MODE, contract.data_mode.bytes());
        }
        if engine_identity.is_none() {
            batch.put_cf(
                &meta,
                META_ENGINE_IDENTITY,
                stored_engine_identity(contract.engine_identity),
            );
        }
        if stored_schema.is_none()
            && let Some(config) = &contract.feature_schema
        {
            batch.put_cf(&meta, META_FEATURE_SCHEMA, stored_feature_schema(config)?);
        }
        if stored_reward_tie_epsilon.is_none()
            && let Some(epsilon) = contract.symmetric_reward_tie_epsilon
        {
            batch.put_cf(
                &meta,
                META_SYMMETRIC_REWARD_TIE_EPSILON,
                stored_symmetric_reward_tie_epsilon(epsilon),
            );
        }
        if stored_reward_target_scale.is_none()
            && let Some(scale) = contract.symmetric_reward_target_scale
        {
            batch.put_cf(
                &meta,
                META_SYMMETRIC_REWARD_TARGET_SCALE,
                stored_symmetric_reward_target_scale(scale),
            );
        }
        self.db.write(batch)?;

        *data_mode = Some(contract.data_mode);
        *engine_identity = Some(contract.engine_identity);
        Ok(())
    }
}
