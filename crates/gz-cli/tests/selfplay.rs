use gz_cli::selfplay::{EvaluatorMode, ReferenceMode, SelfplayConfig, run};
use gz_replay::ReplayStore;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("gz-cli-selfplay-test-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();

        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn selfplay_config_defaults_tree_reuse_on() {
    assert!(SelfplayConfig::default().tree_reuse);
}

#[test]
fn selfplay_cli_accepts_tree_reuse_flag() {
    let dir = TestDir::new();
    let output = Command::new(env!("CARGO_BIN_EXE_graphzero"))
        .args([
            "selfplay",
            "--replay-dir",
            dir.path().to_str().unwrap(),
            "--episodes",
            "1",
            "--lanes",
            "1",
            "--workers-per-lane",
            "1",
            "--max-steps",
            "1",
            "--simulations",
            "1",
            "--tree-reuse",
            "false",
        ])
        .output()
        .unwrap();

    assert!(output.status.success(), "{output:?}");
}

#[test]
fn selfplay_run_writes_replay_rows() {
    let dir = TestDir::new();
    let summary = run(SelfplayConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        episodes: 4,
        lanes: 2,
        workers_per_lane: 2,
        reference: ReferenceMode::Root,
        reference_ema_decay: 0.99,
        seed: 3,
        max_steps: 2,
        simulations: 2,
        tree_reuse: false,
        max_candidates: 255,
        max_batch: 4,
        evaluator: EvaluatorMode::Random,
        python_dir: None,
        checkpoint_dir: None,
        eval_device: None,
        eval_poll_interval: None,
        serve_socket: None,
        serve_max_batch: 512,
        replay_backlog: None,
    })
    .unwrap();
    let store = ReplayStore::open(dir.path()).unwrap();
    let counters = store.counters();

    assert_eq!(summary.counters, counters);
    assert_eq!(summary.rows_produced, counters.produced_rows);
    assert_eq!(summary.episodes_appended + summary.episodes_dropped, 4);
    assert!(summary.rows_produced > 0);
}

#[test]
fn selfplay_run_supports_stub_evaluator() {
    let dir = TestDir::new();
    let summary = run(SelfplayConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        episodes: 2,
        lanes: 1,
        workers_per_lane: 2,
        reference: ReferenceMode::Root,
        reference_ema_decay: 0.99,
        seed: 4,
        max_steps: 2,
        simulations: 2,
        tree_reuse: false,
        max_candidates: 255,
        max_batch: 2,
        evaluator: EvaluatorMode::Stub,
        python_dir: None,
        checkpoint_dir: None,
        eval_device: None,
        eval_poll_interval: None,
        serve_socket: None,
        serve_max_batch: 512,
        replay_backlog: None,
    })
    .unwrap();

    assert_eq!(summary.evaluator, EvaluatorMode::Stub);
    assert!(summary.model_version.is_some());
    assert_eq!(summary.episodes_appended + summary.episodes_dropped, 2);
}

#[test]
fn selfplay_run_supports_self_average_reference() {
    let dir = TestDir::new();
    let summary = run(SelfplayConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        episodes: 4,
        lanes: 1,
        workers_per_lane: 1,
        reference: ReferenceMode::SelfAverage,
        reference_ema_decay: 0.9,
        seed: 11,
        max_steps: 2,
        simulations: 2,
        tree_reuse: false,
        max_candidates: 255,
        max_batch: 1,
        evaluator: EvaluatorMode::Random,
        python_dir: None,
        checkpoint_dir: None,
        eval_device: None,
        eval_poll_interval: None,
        serve_socket: None,
        serve_max_batch: 512,
        replay_backlog: None,
    })
    .unwrap();

    assert_eq!(summary.episodes_appended, 4);
    let labeled = summary.wins + summary.losses + summary.ties;
    // The first admission per lane has no EMA yet and stays unlabeled.
    assert_eq!(labeled, 3);
}

fn serving_config(dir: &TestDir) -> SelfplayConfig {
    SelfplayConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        episodes: 0,
        lanes: 1,
        workers_per_lane: 1,
        reference: ReferenceMode::Root,
        reference_ema_decay: 0.99,
        seed: 3,
        max_steps: 2,
        simulations: 2,
        tree_reuse: false,
        max_candidates: 255,
        max_batch: 1,
        evaluator: EvaluatorMode::Stub,
        python_dir: None,
        checkpoint_dir: None,
        eval_device: None,
        eval_poll_interval: None,
        serve_socket: Some(dir.path().join("live.sock")),
        serve_max_batch: 512,
        replay_backlog: None,
    }
}

#[test]
fn selfplay_config_rejects_serve_socket_with_bounded_episodes() {
    let dir = TestDir::new();
    let mut config = serving_config(&dir);
    config.episodes = 4;

    let error = config.validate().unwrap_err();
    assert!(
        error.contains("--serve-socket requires --episodes 0"),
        "{error}"
    );
}

#[test]
fn selfplay_config_rejects_unbounded_episodes_without_serve_socket() {
    let dir = TestDir::new();
    let mut config = serving_config(&dir);
    config.serve_socket = None;

    let error = config.validate().unwrap_err();
    assert!(error.contains("requires --serve-socket"), "{error}");
}

#[test]
fn selfplay_config_rejects_serve_socket_with_random_evaluator() {
    let dir = TestDir::new();
    let mut config = serving_config(&dir);
    config.evaluator = EvaluatorMode::Random;

    let error = config.validate().unwrap_err();
    assert!(error.contains("featurized evaluator"), "{error}");
}

#[test]
fn selfplay_config_rejects_zero_replay_backlog() {
    let dir = TestDir::new();
    let mut config = serving_config(&dir);
    config.replay_backlog = Some(0);

    let error = config.validate().unwrap_err();
    assert!(error.contains("--replay-backlog"), "{error}");
}

#[test]
fn torch_evaluator_builds_the_child_command_line() {
    let dir = TestDir::new();
    let mut config = serving_config(&dir);
    config.evaluator = EvaluatorMode::Torch;
    config.checkpoint_dir = Some(PathBuf::from("/ckpt"));
    config.validate().unwrap();

    assert_eq!(
        config.evaluator_extra_args(),
        [
            "--backend",
            "torch",
            "--checkpoint-dir",
            "/ckpt",
            "--device",
            "cuda:0"
        ]
    );

    config.eval_device = Some("cuda:1".to_owned());
    assert_eq!(config.evaluator_extra_args()[5], "cuda:1");
}

#[test]
fn stub_evaluators_pass_no_extra_args() {
    let dir = TestDir::new();
    let config = serving_config(&dir);

    assert!(config.evaluator_extra_args().is_empty());
}

#[test]
fn selfplay_config_rejects_torch_without_checkpoint_dir() {
    let dir = TestDir::new();
    let mut config = serving_config(&dir);
    config.evaluator = EvaluatorMode::Torch;

    let error = config.validate().unwrap_err();
    assert!(error.contains("requires --checkpoint-dir"), "{error}");
}

#[test]
fn selfplay_config_rejects_checkpoint_dir_without_torch() {
    let dir = TestDir::new();
    let mut config = serving_config(&dir);
    config.checkpoint_dir = Some(PathBuf::from("/ckpt"));

    let error = config.validate().unwrap_err();
    assert!(error.contains("--checkpoint-dir requires"), "{error}");
}

#[test]
fn selfplay_config_rejects_eval_device_without_torch() {
    let dir = TestDir::new();
    let mut config = serving_config(&dir);
    config.eval_device = Some("cuda:1".to_owned());

    let error = config.validate().unwrap_err();
    assert!(error.contains("--eval-device requires"), "{error}");
}

#[test]
fn torch_evaluator_forwards_the_poll_interval() {
    let dir = TestDir::new();
    let mut config = serving_config(&dir);
    config.evaluator = EvaluatorMode::Torch;
    config.checkpoint_dir = Some(PathBuf::from("/ckpt"));
    config.eval_poll_interval = Some(0.5);
    config.validate().unwrap();

    let args = config.evaluator_extra_args();
    assert_eq!(&args[6..], ["--poll-interval", "0.5"]);
}

#[test]
fn selfplay_config_rejects_poll_interval_without_torch() {
    let dir = TestDir::new();
    let mut config = serving_config(&dir);
    config.eval_poll_interval = Some(0.5);

    let error = config.validate().unwrap_err();
    assert!(error.contains("--eval-poll-interval requires"), "{error}");
}

#[test]
fn selfplay_config_rejects_negative_poll_interval() {
    let dir = TestDir::new();
    let mut config = serving_config(&dir);
    config.evaluator = EvaluatorMode::Torch;
    config.checkpoint_dir = Some(PathBuf::from("/ckpt"));
    config.eval_poll_interval = Some(-1.0);

    let error = config.validate().unwrap_err();
    assert!(error.contains("--eval-poll-interval must be"), "{error}");
}

#[test]
fn selfplay_config_rejects_zero_max_candidates() {
    let dir = TestDir::new();
    let mut config = serving_config(&dir);
    config.max_candidates = 0;

    let error = config.validate().unwrap_err();
    assert!(error.contains("--max-candidates"), "{error}");
}
