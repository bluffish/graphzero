use gz_cli::distill::{DistillGenerateConfig, PolicyTeacher, generate, reducing_uniform_label};
use gz_replay::{ReplayStore, SampleConfig};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("gz-cli-distill-test-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn reducing_uniform_teacher_labels_all_and_only_improvements() {
    let label = reducing_uniform_label(-10.0, &[-9.0, -11.0, -8.0, -10.0]);

    assert_eq!(label.policy_target, vec![0.5, 0.0, 0.5, 0.0, 0.0]);
    assert_eq!(label.selected_candidate, Some(0));
    assert_eq!(label.improving_actions, 2);
}

#[test]
fn reducing_uniform_teacher_selects_stop_without_an_improvement() {
    let label = reducing_uniform_label(-10.0, &[-10.0, -11.0]);

    assert_eq!(label.policy_target, vec![0.0, 0.0, 1.0]);
    assert_eq!(label.selected_candidate, None);
    assert_eq!(label.improving_actions, 0);
}

#[test]
fn generated_dataset_contains_unique_measured_policy_only_rows() {
    let dir = TestDir::new();
    let summary = generate(DistillGenerateConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        states: 3,
        workers: 2,
        max_attempts: 30,
        seed: 42,
        max_candidates: 1023,
        max_steps: 64,
        position_features: true,
        teacher: PolicyTeacher::ReducingUniform,
    })
    .unwrap();

    assert_eq!(summary.states, 3);
    let store = ReplayStore::open(dir.path()).unwrap();
    assert_eq!(store.counters().produced_rows, 3);
    assert_eq!(store.episode_counters().0, 3);
    let sampled = store
        .sample_rows(SampleConfig {
            batch: NonZeroUsize::new(3).unwrap(),
            window_rows: NonZeroU64::new(3).unwrap(),
            seed: 7,
        })
        .unwrap();
    for (_, row) in sampled {
        assert!(row.value_target.is_none());
        assert!(row.final_measure.measured);
        assert!(row.final_measure.valid);
        assert!(row.final_measure.scalar_reward.is_some());
        assert!(row.feature_row.is_some());
        assert!((row.policy_target.iter().sum::<f32>() - 1.0).abs() < 0.02);
        let selected = row
            .legal_actions
            .iter()
            .position(|action| Some(*action) == row.selected_action)
            .unwrap();
        assert!(row.policy_target[selected] > 0.0);
    }
}

#[test]
fn candidate_overflow_is_dropped_instead_of_truncated() {
    let dir = TestDir::new();
    let error = generate(DistillGenerateConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        states: 1,
        workers: 1,
        max_attempts: 1,
        seed: 42,
        max_candidates: 1,
        max_steps: 64,
        position_features: true,
        teacher: PolicyTeacher::ReducingUniform,
    })
    .unwrap_err();

    assert!(error.contains("candidate_overflows=1"), "{error}");
    let store = ReplayStore::open(dir.path()).unwrap();
    assert_eq!(store.counters().produced_rows, 0);
}
