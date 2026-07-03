mod common;

use common::{episode_with_feature_rows, episode_with_rows, feature_schema_config, measure};
use gz_replay::{ReplayEpisodeId, ReplayError, ReplayStore, SampleConfig};
use rocksdb::{ColumnFamilyDescriptor, DB, Options};

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
fn value_target_validation_accepts_only_loss_tie_or_win() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();

    let (mut tie_record, mut tie_rows) = episode_with_rows(1);
    tie_record.outcome.value_target = Some(0.0);
    tie_record.outcome.reference.as_mut().unwrap().reward = 5.0;
    tie_rows[0].value_target = Some(0.0);
    assert_eq!(
        store.append_episode(&tie_record, &tie_rows).unwrap(),
        ReplayEpisodeId::new(0)
    );

    let (mut loss_record, mut loss_rows) = episode_with_rows(1);
    loss_record.outcome.value_target = Some(-1.0);
    loss_record.outcome.reference.as_mut().unwrap().reward = 6.0;
    loss_rows[0].value_target = Some(-1.0);
    assert_eq!(
        store.append_episode(&loss_record, &loss_rows).unwrap(),
        ReplayEpisodeId::new(1)
    );

    let (mut invalid_record, mut invalid_rows) = episode_with_rows(1);
    invalid_record.outcome.value_target = Some(0.5);
    invalid_rows[0].value_target = Some(0.5);
    assert_eq!(
        store
            .append_episode(&invalid_record, &invalid_rows)
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
        store.ensure_feature_schema(&config).unwrap();
        store.ensure_feature_schema(&config).unwrap();
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
    store.ensure_feature_schema(&config).unwrap();

    let mut other = config;
    other.max_nodes += 1;
    assert_eq!(
        store.ensure_feature_schema(&other).unwrap_err(),
        ReplayError::InvalidRecord
    );

    let mut other = feature_schema_config();
    other.expander_seed = 9;
    assert_eq!(
        store.ensure_feature_schema(&other).unwrap_err(),
        ReplayError::InvalidRecord
    );
}

#[test]
fn append_roundtrips_rows_with_feature_payloads() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    store
        .ensure_feature_schema(&feature_schema_config())
        .unwrap();
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
fn featured_rows_require_configured_schema_and_matching_header() {
    let dir = common::temp_dir();
    let store = ReplayStore::open(dir.path()).unwrap();
    let (record, rows) = episode_with_feature_rows(1);

    assert_eq!(
        store.append_episode(&record, &rows).unwrap_err(),
        ReplayError::InvalidRecord
    );

    store
        .ensure_feature_schema(&feature_schema_config())
        .unwrap();
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
    store
        .ensure_feature_schema(&feature_schema_config())
        .unwrap();
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
        ],
    )
    .unwrap()
}

#[test]
fn retention_deletes_old_episodes_and_clamps_sampling() {
    let dir = common::temp_dir();
    let store = ReplayStore::open_with_retention(dir.path(), Some(20)).unwrap();
    store
        .ensure_feature_schema(&feature_schema_config())
        .unwrap();

    // 4-row episodes; retention 20 rows triggers past 25 produced.
    for _ in 0..20 {
        let (record, rows) = episode_with_feature_rows(4);
        store.append_episode(&record, &rows).unwrap();
    }

    let counters = store.counters();
    assert_eq!(counters.produced_rows, 80);
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
    let sampled = reopened
        .sample_rows(SampleConfig {
            batch: std::num::NonZeroUsize::new(8).unwrap(),
            window_rows: std::num::NonZeroU64::new(1_000_000).unwrap(),
            seed: 7,
        })
        .unwrap();
    assert_eq!(sampled.len(), 8);
}
