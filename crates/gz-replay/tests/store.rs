mod common;

use common::{
    engine_identity, episode_with_feature_rows, episode_with_rows, feature_schema_config, measure,
};
use gz_engine::{ActionSetHash, EngineIdentity};
use gz_replay::{
    ReplayContract, ReplayDataMode, ReplayEpisodeId, ReplayError, ReplayStore, SampleConfig,
};
use rocksdb::{ColumnFamilyDescriptor, DB, Options};

#[test]
fn engine_identity_persists_and_rejects_mismatches() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let identity = engine_identity();

    assert_eq!(store.engine_identity().unwrap(), None);
    store
        .ensure_contract(&ReplayContract::unfeaturized(
            ReplayDataMode::Standard,
            identity,
        ))
        .unwrap();
    assert_eq!(store.engine_identity().unwrap(), Some(identity));

    let mismatched = EngineIdentity {
        action_set_hash: ActionSetHash::from_bytes([99; 32]),
        ..identity
    };
    assert_eq!(
        store
            .ensure_contract(&ReplayContract::unfeaturized(
                ReplayDataMode::Standard,
                mismatched,
            ))
            .unwrap_err(),
        ReplayError::EngineIdentityMismatch
    );

    drop(store);
    assert_eq!(
        ReplayStore::open(dir.path())
            .unwrap()
            .engine_identity()
            .unwrap(),
        Some(identity)
    );
}

#[test]
fn engine_identity_migration_checks_existing_episodes() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let (record, rows) = episode_with_rows(1);
    store.append_episode(&record, &rows).unwrap();

    let mismatched = EngineIdentity {
        action_set_hash: ActionSetHash::from_bytes([99; 32]),
        ..engine_identity()
    };
    assert_eq!(
        store
            .ensure_contract(&ReplayContract::unfeaturized(
                ReplayDataMode::Standard,
                mismatched,
            ))
            .unwrap_err(),
        ReplayError::EngineIdentityMismatch
    );
    assert_eq!(store.engine_identity().unwrap(), None);

    bind_unfeaturized(&store, ReplayDataMode::Standard);
    assert_eq!(store.engine_identity().unwrap(), Some(engine_identity()));
}

#[test]
fn replay_contract_failure_does_not_partially_bind_metadata() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    bind_unfeaturized(&store, ReplayDataMode::Standard);
    let mismatched = EngineIdentity {
        action_set_hash: ActionSetHash::from_bytes([99; 32]),
        ..engine_identity()
    };

    assert_eq!(
        store
            .ensure_contract(&ReplayContract::featurized(
                ReplayDataMode::Standard,
                feature_schema_config(),
                mismatched,
            ))
            .unwrap_err(),
        ReplayError::EngineIdentityMismatch
    );
    assert_eq!(store.feature_schema().unwrap(), None);
    assert_eq!(store.engine_identity().unwrap(), Some(engine_identity()));
}

#[test]
fn append_then_read_back_episode() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let (record, rows) = episode_with_rows(2);

    let id = store.append_episode(&record, &rows).unwrap();

    assert_eq!(id, ReplayEpisodeId::new(0));
    assert_eq!(store.episode(id).unwrap(), Some(record));
}

#[test]
fn terminal_sampling_is_episode_uniform_and_returns_only_terminal_rows() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    bind_unfeaturized(&store, ReplayDataMode::SymmetricSelfplay);
    let first = symmetric_episode_with_terminal(2, -5.0, 1.0);
    let second = symmetric_episode_with_terminal(5, 5.0, -1.0);
    store
        .append_episode_pair((&first.0, &first.1), (&second.0, &second.1))
        .unwrap();
    let before = store.counters().consumed_rows;
    let sampled = store
        .sample_terminal_rows(SampleConfig {
            batch: std::num::NonZeroUsize::new(100).unwrap(),
            window_rows: std::num::NonZeroU64::new(store.counters().produced_rows).unwrap(),
            seed: 17,
        })
        .unwrap();

    assert!(
        sampled
            .iter()
            .all(|(_, row)| row.kind == gz_replay::ReplayRowKind::Terminal)
    );
    let episode_ids = sampled
        .iter()
        .map(|(episode, _)| episode.get())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(episode_ids, std::collections::BTreeSet::from([0, 1]));
    assert_eq!(store.counters().consumed_rows, before + 100);
}

#[test]
fn episode_pair_is_atomic_and_counts_as_one_game() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let (primary_record, primary_rows) = episode_with_rows(2);
    let (secondary_record, secondary_rows) = episode_with_rows(3);

    let ids = store
        .append_episode_pair(
            (&primary_record, &primary_rows),
            (&secondary_record, &secondary_rows),
        )
        .unwrap();

    assert_eq!(ids, (ReplayEpisodeId::new(0), ReplayEpisodeId::new(1)));
    assert_eq!(store.episode_counters(), (1, 0));
    assert_eq!(store.counters().produced_rows, 5);
    assert_eq!(store.episode(ids.0).unwrap(), Some(primary_record));
    assert_eq!(store.episode(ids.1).unwrap(), Some(secondary_record));

    drop(store);
    let reopened = ReplayStore::open(dir.path()).unwrap();
    assert_eq!(reopened.episode_counters(), (1, 0));
    assert_eq!(reopened.outcome_emas().unwrap().0, -5.0);
    assert_eq!(reopened.best_cost(), Some(-5.0));
}

#[test]
fn symmetric_metrics_track_both_seats_and_survive_reopen() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    bind_unfeaturized(&store, ReplayDataMode::SymmetricSelfplay);
    assert_eq!(store.symmetric_selfplay_metrics(), None);

    let p1_win = symmetric_episode(2, -4.0, 1.0);
    let p2_loss = symmetric_episode(3, -6.0, -1.0);
    assert_eq!(
        store.append_episode(&p1_win.0, &p1_win.1).unwrap_err(),
        ReplayError::InvalidRecord
    );
    let same_sign = symmetric_episode(3, -6.0, 1.0);
    assert_eq!(
        store
            .append_episode_pair((&p1_win.0, &p1_win.1), (&same_sign.0, &same_sign.1))
            .unwrap_err(),
        ReplayError::InvalidRecord
    );
    assert_eq!(store.symmetric_selfplay_metrics(), None);
    store
        .append_episode_pair((&p1_win.0, &p1_win.1), (&p2_loss.0, &p2_loss.1))
        .unwrap();

    let first = store.symmetric_selfplay_metrics().unwrap();
    assert_eq!(first.p1_win_rate_ema, 1.0);
    assert_eq!(first.p2_win_rate_ema, 0.0);
    assert_eq!(first.draw_rate_ema, 0.0);
    assert_eq!(first.seat_advantage_ema, 1.0);
    assert_eq!(first.p1_terminal_cost_ema, 4.0);
    assert_eq!(first.p2_terminal_cost_ema, 6.0);
    assert_eq!(first.mean_terminal_cost_ema, 5.0);
    assert_eq!(first.terminal_cost_margin_ema, 2.0);
    assert_eq!(first.terminal_cost_best, 4.0);
    assert_eq!(first.p1_episode_len_ema, 2.0);
    assert_eq!(first.p2_episode_len_ema, 3.0);
    assert_eq!(first.game_len_ema, 5.0);
    assert_eq!(first.episode_len_margin_ema, 1.0);

    let p1_loss = symmetric_episode(4, -8.0, -1.0);
    let p2_win = symmetric_episode(1, -5.0, 1.0);
    store
        .append_episode_pair((&p1_loss.0, &p1_loss.1), (&p2_win.0, &p2_win.1))
        .unwrap();
    let second = store.symmetric_selfplay_metrics().unwrap();
    assert!((second.p1_win_rate_ema - 0.99).abs() < 1.0e-12);
    assert!((second.p2_win_rate_ema - 0.01).abs() < 1.0e-12);
    assert_eq!(second.draw_rate_ema, 0.0);
    assert!((second.p1_terminal_cost_ema - 4.04).abs() < 1.0e-12);
    assert!((second.p2_terminal_cost_ema - 5.99).abs() < 1.0e-12);
    assert!((second.terminal_cost_margin_ema - 2.01).abs() < 1.0e-12);
    assert_eq!(second.terminal_cost_best, 4.0);
    assert!((second.p1_episode_len_ema - 2.02).abs() < 1.0e-12);
    assert!((second.p2_episode_len_ema - 2.98).abs() < 1.0e-12);
    assert!((second.episode_len_margin_ema - 1.02).abs() < 1.0e-12);

    let p1_draw = symmetric_episode(2, -7.0, 0.0);
    let p2_draw = symmetric_episode(2, -7.0, 0.0);
    store
        .append_episode_pair((&p1_draw.0, &p1_draw.1), (&p2_draw.0, &p2_draw.1))
        .unwrap();
    let expected = store.symmetric_selfplay_metrics().unwrap();
    assert!((expected.p1_win_rate_ema - 0.9801).abs() < 1.0e-12);
    assert!((expected.p2_win_rate_ema - 0.0099).abs() < 1.0e-12);
    assert!((expected.draw_rate_ema - 0.01).abs() < 1.0e-12);

    drop(store);
    let reopened = ReplayStore::open(dir.path()).unwrap();
    assert_eq!(reopened.symmetric_selfplay_metrics(), Some(expected));
}

#[test]
fn rejected_episode_pair_writes_neither_record() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let (primary_record, primary_rows) = episode_with_rows(1);
    let (mut secondary_record, secondary_rows) = episode_with_rows(1);
    secondary_record.row_count = 2;

    assert_eq!(
        store
            .append_episode_pair(
                (&primary_record, &primary_rows),
                (&secondary_record, &secondary_rows),
            )
            .unwrap_err(),
        ReplayError::InvalidRecord
    );
    assert_eq!(store.episode_counters(), (0, 0));
    assert_eq!(store.counters().produced_rows, 0);
    assert_eq!(store.episode(ReplayEpisodeId::new(0)).unwrap(), None);
}

#[test]
fn replay_data_mode_prevents_standard_and_symmetric_mixing() {
    let legacy_dir = common::temp_dir();
    let legacy = ReplayStore::open(legacy_dir.path()).unwrap();
    let (record, rows) = episode_with_rows(1);
    legacy.append_episode(&record, &rows).unwrap();
    assert_eq!(
        legacy
            .ensure_contract(&ReplayContract::unfeaturized(
                ReplayDataMode::SymmetricSelfplay,
                engine_identity(),
            ))
            .unwrap_err(),
        ReplayError::DataModeMismatch
    );
    bind_unfeaturized(&legacy, ReplayDataMode::Standard);

    let symmetric_dir = common::temp_dir();
    let symmetric = ReplayStore::open(symmetric_dir.path()).unwrap();
    bind_unfeaturized(&symmetric, ReplayDataMode::SymmetricSelfplay);
    assert_eq!(
        symmetric
            .ensure_contract(&ReplayContract::unfeaturized(
                ReplayDataMode::SymmetricSelfplayStop,
                engine_identity(),
            ))
            .unwrap_err(),
        ReplayError::DataModeMismatch
    );
    drop(symmetric);
    let reopened = ReplayStore::open(symmetric_dir.path()).unwrap();
    assert_eq!(
        reopened.data_mode().unwrap(),
        ReplayDataMode::SymmetricSelfplay
    );
}

#[test]
fn replay_contract_prevents_symmetric_reward_epsilon_mixing() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let contract = |epsilon| {
        ReplayContract::unfeaturized(ReplayDataMode::SymmetricSelfplay, engine_identity())
            .with_symmetric_reward_tie_epsilon(
                gz_replay::SymmetricRewardTieEpsilon::new(epsilon).unwrap(),
            )
    };

    store.ensure_contract(&contract(0.01)).unwrap();
    store.ensure_contract(&contract(0.01)).unwrap();
    assert_eq!(
        store.ensure_contract(&contract(0.02)).unwrap_err(),
        ReplayError::TargetContractMismatch
    );

    drop(store);
    let reopened = ReplayStore::open(dir.path()).unwrap();
    reopened.ensure_contract(&contract(0.01)).unwrap();
}

#[test]
fn replay_contract_prevents_symmetric_reward_target_scale_mixing() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let contract = |scale| {
        ReplayContract::unfeaturized(ReplayDataMode::SymmetricSelfplay, engine_identity())
            .with_symmetric_reward_target_scale(
                gz_replay::SymmetricRewardTargetScale::new(scale).unwrap(),
            )
    };

    store.ensure_contract(&contract(5.0)).unwrap();
    store.ensure_contract(&contract(5.0)).unwrap();
    assert_eq!(
        store.ensure_contract(&contract(2.0)).unwrap_err(),
        ReplayError::TargetContractMismatch
    );
}

#[test]
fn admission_rejects_unmeasured_invalid_and_non_finite_final_measure() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();

    for final_measure in [
        measure(None, false, false),
        measure(Some(1.0), true, false),
        measure(Some(f32::NAN), true, true),
    ] {
        let (mut record, rows) = episode_with_rows(1);
        record.final_measure = final_measure;

        assert_eq!(
            store.append_episode(&record, &rows).unwrap_err(),
            ReplayError::NotMeasured
        );
    }
}

#[test]
fn admission_rejects_invalid_row_shapes() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let (mut record, mut rows) = episode_with_rows(2);

    record.row_count = 1;
    assert_eq!(
        store.append_episode(&record, &rows).unwrap_err(),
        ReplayError::InvalidRecord
    );

    let (record, mut rows_with_bad_step) = episode_with_rows(2);
    rows_with_bad_step[1].step_index = 3;
    assert_eq!(
        store
            .append_episode(&record, &rows_with_bad_step)
            .unwrap_err(),
        ReplayError::InvalidRecord
    );

    rows[0].policy_target.pop();
    assert_eq!(
        store
            .append_episode(&episode_with_rows(2).0, &rows)
            .unwrap_err(),
        ReplayError::InvalidRecord
    );

    let (record, mut rows) = episode_with_rows(1);
    rows[0].legal_actions.reverse();
    assert_eq!(
        store.append_episode(&record, &rows).unwrap_err(),
        ReplayError::InvalidRecord
    );

    let (record, mut rows) = episode_with_rows(1);
    rows[0].policy_target[0] = f32::INFINITY;
    assert_eq!(
        store.append_episode(&record, &rows).unwrap_err(),
        ReplayError::InvalidRecord
    );
}

#[test]
fn symmetric_value_target_validation_accepts_continuous_unit_interval() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    bind_unfeaturized(&store, ReplayDataMode::SymmetricSelfplay);
    let draw = symmetric_episode(1, -5.0, 0.0);
    store
        .append_episode_pair((&draw.0, &draw.1), (&draw.0, &draw.1))
        .unwrap();

    let (mut positive_record, mut positive_rows) = symmetric_episode(1, -5.0, 1.0);
    positive_record.outcome.value_target = Some(0.5);
    positive_rows[0].value_target = Some(0.5);
    let (mut negative_record, mut negative_rows) = symmetric_episode(2, -5.0, -1.0);
    negative_record.outcome.value_target = Some(-0.5);
    for row in &mut negative_rows {
        row.value_target = Some(-0.5);
    }
    store
        .append_episode_pair(
            (&positive_record, &positive_rows),
            (&negative_record, &negative_rows),
        )
        .unwrap();

    positive_record.outcome.value_target = Some(1.1);
    positive_rows[0].value_target = Some(1.1);
    assert_eq!(
        store
            .append_episode_pair(
                (&positive_record, &positive_rows),
                (&negative_record, &negative_rows),
            )
            .unwrap_err(),
        ReplayError::InvalidRecord
    );
}

#[test]
fn rejected_append_is_atomic() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let (mut record, rows) = episode_with_rows(1);
    record.row_count = 2;

    assert_eq!(
        store.append_episode(&record, &rows).unwrap_err(),
        ReplayError::InvalidRecord
    );
    assert_eq!(store.counters().produced_rows, 0);
    assert_eq!(store.episode(ReplayEpisodeId::new(0)).unwrap(), None);
}

#[test]
fn episode_ids_are_monotonic_and_survive_reopen() {
    let dir = common::temp_dir();
    {
        let store = ReplayStore::open(dir.path()).unwrap();
        let (record, rows) = episode_with_rows(1);
        assert_eq!(
            store.append_episode(&record, &rows).unwrap(),
            ReplayEpisodeId::new(0)
        );
        assert_eq!(
            store.append_episode(&record, &rows).unwrap(),
            ReplayEpisodeId::new(1)
        );
    }

    let store = ReplayStore::open(dir.path()).unwrap();
    let (record, rows) = episode_with_rows(1);
    assert_eq!(
        store.append_episode(&record, &rows).unwrap(),
        ReplayEpisodeId::new(2)
    );
}

#[test]
fn counters_survive_reopen() {
    let dir = common::temp_dir();
    {
        let store = ReplayStore::open(dir.path()).unwrap();
        let (record, rows) = episode_with_rows(3);
        store.append_episode(&record, &rows).unwrap();
        store
            .sample_rows(gz_replay::SampleConfig {
                batch: std::num::NonZeroUsize::new(2).unwrap(),
                window_rows: std::num::NonZeroU64::new(3).unwrap(),
                seed: 1,
            })
            .unwrap();
    }

    let store = ReplayStore::open(dir.path()).unwrap();
    assert_eq!(store.counters().produced_rows, 3);
    assert_eq!(store.counters().consumed_rows, 2);
}

#[test]
fn feature_schema_is_idempotent_and_survives_reopen() {
    let dir = common::temp_dir();
    let config = feature_schema_config();
    {
        let store = ReplayStore::open(dir.path()).unwrap();
        assert_eq!(store.feature_schema().unwrap(), None);
        bind_featurized(&store, &config);
        bind_featurized(&store, &config);
        assert_eq!(store.feature_schema().unwrap(), Some(config.clone()));
    }

    let store = ReplayStore::open(dir.path()).unwrap();
    assert_eq!(store.feature_schema().unwrap(), Some(config));
}

#[test]
fn feature_schema_mismatch_is_rejected() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let config = feature_schema_config();
    bind_featurized(&store, &config);

    let mut other = config;
    other.max_nodes += 1;
    assert_eq!(
        store
            .ensure_contract(&ReplayContract::featurized(
                ReplayDataMode::Standard,
                other,
                engine_identity(),
            ))
            .unwrap_err(),
        ReplayError::InvalidRecord
    );

    let mut other = feature_schema_config();
    other.expander_seed = 9;
    assert_eq!(
        store
            .ensure_contract(&ReplayContract::featurized(
                ReplayDataMode::Standard,
                other,
                engine_identity(),
            ))
            .unwrap_err(),
        ReplayError::InvalidRecord
    );
}

#[test]
fn append_roundtrips_rows_with_feature_payloads() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    bind_featurized(&store, &feature_schema_config());
    let (record, rows) = episode_with_feature_rows(2);

    let id = store.append_episode(&record, &rows).unwrap();
    let sample = store
        .sample_rows(gz_replay::SampleConfig {
            batch: std::num::NonZeroUsize::new(2).unwrap(),
            window_rows: std::num::NonZeroU64::new(2).unwrap(),
            seed: 3,
        })
        .unwrap();

    assert_eq!(id, ReplayEpisodeId::new(0));
    assert!(sample.iter().all(|(_, row)| row.feature_row.is_some()));
}

#[test]
fn inspection_sampling_does_not_advance_consumed_rows() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    bind_featurized(&store, &feature_schema_config());
    let (record, rows) = episode_with_feature_rows(2);
    store.append_episode(&record, &rows).unwrap();

    let before = store.counters();
    let sampled = store
        .inspect_rows(gz_replay::SampleConfig {
            batch: std::num::NonZeroUsize::new(2).unwrap(),
            window_rows: std::num::NonZeroU64::new(2).unwrap(),
            seed: 3,
        })
        .unwrap();

    assert_eq!(sampled.len(), 2);
    assert_eq!(store.counters(), before);
}

#[test]
fn featured_rows_require_configured_schema_and_matching_header() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let (record, rows) = episode_with_feature_rows(1);

    assert_eq!(
        store.append_episode(&record, &rows).unwrap_err(),
        ReplayError::InvalidRecord
    );

    bind_featurized(&store, &feature_schema_config());
    let (record, mut rows) = episode_with_feature_rows(1);
    rows[0].feature_row.as_mut().unwrap()[8] ^= 0xff;

    assert_eq!(
        store.append_episode(&record, &rows).unwrap_err(),
        ReplayError::InvalidRecord
    );
}

#[test]
fn mixed_feature_and_featureless_rows_are_rejected() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    bind_featurized(&store, &feature_schema_config());
    let (record, mut rows) = episode_with_feature_rows(2);
    rows[1].feature_row = None;

    assert_eq!(
        store.append_episode(&record, &rows).unwrap_err(),
        ReplayError::InvalidRecord
    );
}

#[test]
fn schema_version_mismatch_fails_open() {
    let dir = common::temp_dir();
    drop(ReplayStore::open(dir.path()).unwrap());

    let db = raw_db(dir.path());
    let meta = db.cf_handle("meta").unwrap();
    db.put_cf(&meta, b"schema_version", 999u32.to_be_bytes())
        .unwrap();
    drop(db);

    assert!(matches!(
        ReplayStore::open(dir.path()),
        Err(ReplayError::SchemaMismatch)
    ));
}

#[test]
fn old_schema_stores_fail_open() {
    let dir = common::temp_dir();
    drop(ReplayStore::open(dir.path()).unwrap());

    for version in [2u32, 3] {
        let db = raw_db(dir.path());
        let meta = db.cf_handle("meta").unwrap();
        db.put_cf(&meta, b"schema_version", version.to_be_bytes())
            .unwrap();
        drop(db);

        assert!(matches!(
            ReplayStore::open(dir.path()),
            Err(ReplayError::SchemaMismatch)
        ));
    }
}

fn raw_db(path: &std::path::Path) -> DB {
    let mut options = Options::default();
    options.create_if_missing(false);
    DB::open_cf_descriptors(
        &options,
        path,
        [
            ColumnFamilyDescriptor::new("meta", Options::default()),
            ColumnFamilyDescriptor::new("episodes", Options::default()),
            ColumnFamilyDescriptor::new("rows", Options::default()),
            ColumnFamilyDescriptor::new("row_index", Options::default()),
            ColumnFamilyDescriptor::new("policy_row_index", Options::default()),
            ColumnFamilyDescriptor::new("value_row_index", Options::default()),
        ],
    )
    .unwrap()
}

#[test]
fn retention_deletes_old_episodes_and_clamps_sampling() {
    let dir = common::temp_dir();
    let store = ReplayStore::open_with_retention(dir.path(), Some(20)).unwrap();
    bind_featurized(&store, &feature_schema_config());

    // 4-row episodes; retention 20 rows triggers past 25 produced.
    for episode in 0..20 {
        let (record, mut rows) = episode_with_feature_rows(4);
        if episode % 2 == 0 {
            for row in &mut rows {
                row.policy_target.fill(0.0);
            }
        }
        store.append_episode(&record, &rows).unwrap();
    }

    assert_eq!(store.counters().produced_rows, 80);
    // Old episodes are gone; recent ones remain.
    assert!(store.episode(ReplayEpisodeId::new(0)).unwrap().is_none());
    assert!(store.episode(ReplayEpisodeId::new(19)).unwrap().is_some());

    // Sampling a huge window never touches deleted rows.
    for seed in 0..50 {
        let sampled = store
            .sample_rows(SampleConfig {
                batch: std::num::NonZeroUsize::new(8).unwrap(),
                window_rows: std::num::NonZeroU64::new(1_000_000).unwrap(),
                seed,
            })
            .unwrap();
        assert_eq!(sampled.len(), 8);
    }

    // Floors survive reopen.
    drop(store);
    let reopened = ReplayStore::open_with_retention(dir.path(), Some(20)).unwrap();
    assert!(reopened.episode(ReplayEpisodeId::new(0)).unwrap().is_none());
    assert_eq!(reopened.counters().produced_rows, 80);
    let sampled = reopened
        .sample_rows(SampleConfig {
            batch: std::num::NonZeroUsize::new(8).unwrap(),
            window_rows: std::num::NonZeroU64::new(1_000_000).unwrap(),
            seed: 7,
        })
        .unwrap();
    assert_eq!(sampled.len(), 8);
}

#[test]
fn outcome_emas_track_recent_episodes() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    bind_featurized(&store, &feature_schema_config());
    assert!(store.outcome_emas().is_none());

    let (record, rows) = episode_with_feature_rows(2);
    store.append_episode(&record, &rows).unwrap();

    let (cost, len, stop) = store.outcome_emas().unwrap();
    assert!((cost - f64::from(-record.outcome.reward)).abs() < 1e-9);
    assert!((len - 2.0).abs() < 1e-9);
    assert!((stop - f64::from(u8::from(record.outcome.stopped))).abs() < 1e-9);

    store.append_episode(&record, &rows).unwrap();
    let (cost2, _, _) = store.outcome_emas().unwrap();
    assert!(
        (cost2 - cost).abs() < 1e-9,
        "same outcome keeps the EMA fixed"
    );
}

#[test]
fn terminal_cost_telemetry_survives_reopen() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let (record, rows) = episode_with_rows(2);
    store.append_episode(&record, &rows).unwrap();
    let expected_cost = f64::from(-record.outcome.reward);
    assert_eq!(store.outcome_emas().unwrap().0, expected_cost);
    assert_eq!(store.best_cost(), Some(expected_cost));

    drop(store);
    let reopened = ReplayStore::open(dir.path()).unwrap();
    assert_eq!(reopened.outcome_emas().unwrap().0, expected_cost);
    assert_eq!(reopened.best_cost(), Some(expected_cost));

    let (mut next_record, mut next_rows) = episode_with_rows(2);
    let next_measure = measure(Some(7.0), true, true);
    next_record.final_measure = next_measure.clone();
    next_record.outcome.reward = 7.0;
    for row in &mut next_rows {
        row.final_measure = next_measure.clone();
        row.reward_target = Some(7.0);
    }
    reopened.append_episode(&next_record, &next_rows).unwrap();
    assert!((reopened.outcome_emas().unwrap().0 - -5.02).abs() < 1e-9);
    assert_eq!(reopened.best_cost(), Some(-7.0));

    drop(reopened);
    let reopened = ReplayStore::open(dir.path()).unwrap();
    assert!((reopened.outcome_emas().unwrap().0 - -5.02).abs() < 1e-9);
    assert_eq!(reopened.best_cost(), Some(-7.0));
}

#[test]
fn episode_latency_ema_seeds_and_updates() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    assert!(store.episode_latency_ema().is_none());

    // Non-positive and non-finite observations are ignored.
    store.observe_episode_latency(0.0);
    store.observe_episode_latency(f64::NAN);
    assert!(store.episode_latency_ema().is_none());

    store.observe_episode_latency(10.0);
    assert!((store.episode_latency_ema().unwrap() - 10.0).abs() < 1e-9);
    store.observe_episode_latency(20.0);
    // 0.99 * 10 + 0.01 * 20
    assert!((store.episode_latency_ema().unwrap() - 10.1).abs() < 1e-9);
}

#[test]
fn win_rate_ema_distinguishes_all_loss_from_unseeded() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    assert!(store.win_rate_ema().is_none());
    bind_unfeaturized(&store, ReplayDataMode::SymmetricSelfplay);

    // A labeled loss seeds an honest 0.0 rate, distinct from unseeded.
    let loss = symmetric_episode(1, -6.0, -1.0);
    let win = symmetric_episode(1, -5.0, 1.0);
    store
        .append_episode_pair((&loss.0, &loss.1), (&win.0, &win.1))
        .unwrap();
    assert!((store.win_rate_ema().unwrap() - 0.0).abs() < 1e-9);

    // A win moves it by the EMA weight.
    store
        .append_episode_pair((&win.0, &win.1), (&loss.0, &loss.1))
        .unwrap();
    assert!((store.win_rate_ema().unwrap() - 0.01).abs() < 1e-9);
}

#[test]
fn value_sign_accuracy_ema_distinguishes_zero_from_unseeded() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    assert_eq!(store.value_sign_accuracy_emas(), (None, None));

    store.observe_value_sign_accuracy(Some(0.0), None);
    assert_eq!(store.value_sign_accuracy_emas(), (Some(0.0), None));

    store.observe_value_sign_accuracy(Some(1.0), Some(0.5));
    let (early, late) = store.value_sign_accuracy_emas();
    assert!((early.unwrap() - 0.01).abs() < 1e-9);
    assert!((late.unwrap() - 0.5).abs() < 1e-9);
}

fn bind_unfeaturized(store: &ReplayStore, mode: ReplayDataMode) {
    store
        .ensure_contract(&ReplayContract::unfeaturized(mode, engine_identity()))
        .unwrap();
}

fn bind_featurized(store: &ReplayStore, schema: &gz_features::FeatureSchemaConfig) {
    store
        .ensure_contract(&ReplayContract::featurized(
            ReplayDataMode::Standard,
            schema.clone(),
            engine_identity(),
        ))
        .unwrap();
}

fn symmetric_episode(
    row_count: usize,
    reward: f32,
    value_target: f32,
) -> (gz_replay::ReplayEpisodeRecord, Vec<gz_replay::ReplayRow>) {
    let (mut record, mut rows) = episode_with_rows(row_count);
    let final_measure = measure(Some(reward), true, true);
    record.final_measure = final_measure.clone();
    record.outcome.reward = reward;
    record.outcome.value_target = Some(value_target);
    record.outcome.stopped = false;
    for row in &mut rows {
        row.legal_actions.truncate(1);
        row.policy_target.truncate(1);
        row.value_target = Some(value_target);
        row.horizon_value_targets = Some([value_target; 2]);
        row.reward_target = Some(reward);
        row.final_measure = final_measure.clone();
    }
    (record, rows)
}

fn symmetric_episode_with_terminal(
    row_count: usize,
    reward: f32,
    value_target: f32,
) -> (gz_replay::ReplayEpisodeRecord, Vec<gz_replay::ReplayRow>) {
    let (mut record, mut rows) = symmetric_episode(row_count, reward, value_target);
    let action_history = record.steps.iter().map(|step| step.action).collect();
    rows.push(gz_replay::ReplayRow {
        kind: gz_replay::ReplayRowKind::Terminal,
        step_index: row_count as u32,
        root: record.root,
        state: record.final_graph,
        action_history,
        legal_actions: Vec::new(),
        policy_target: Vec::new(),
        selected_action: None,
        value_target: Some(value_target),
        horizon_value_targets: Some([value_target; 2]),
        reward_target: Some(reward),
        final_measure: record.final_measure.clone(),
        model_version: None,
        search_config_hash: record.search_config_hash,
        feature_row: None,
    });
    record.row_count += 1;
    (record, rows)
}
