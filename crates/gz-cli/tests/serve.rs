use gz_cli::selfplay::{EvaluatorMode, ReplayInitConfig, SelfplayConfig, init_replay, run};
use gz_cli::serve::{HELLO_ACK_FIXED_LEN, ReplayServeConfig, SAMPLE_PROTOCOL_VERSION, run_one};
use gz_features::{
    ENCODING_VERSION, FeatureBatchView, TrainingTargetsView, decode_feature_schema_config,
};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("gz-cli-serve-test-{}-{id}", std::process::id()));
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
fn replay_serve_returns_symmetric_features_targets_and_metrics() {
    let dir = TestDir::new();
    let summary = run(SelfplayConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        episodes: 1,
        lanes: 1,
        workers_per_lane: 1,
        seed: 5,
        max_steps: 2,
        simulations: 2,
        max_considered: 2,
        gumbel_scale: 0.0,
        gumbel_noise_overlap: -1.0,
        tree_reuse: false,
        max_candidates: 255,
        max_batch: 2,
        evaluator: EvaluatorMode::Stub,
        mask_stop: true,
        ..SelfplayConfig::default()
    })
    .unwrap();
    let socket = dir.path().join("sample.sock");
    let server = spawn_one(dir.path(), &socket, 2);
    let mut stream = connect_hello(&socket);

    let (frame_type, ack) = read_frame(&mut stream);
    assert_eq!(frame_type, 2);
    assert_eq!(
        u32::from_le_bytes(ack[0..4].try_into().unwrap()),
        SAMPLE_PROTOCOL_VERSION
    );
    assert_eq!(u32::from_le_bytes(ack[36..40].try_into().unwrap()), 2);
    assert_eq!(
        u64::from_le_bytes(ack[40..48].try_into().unwrap()),
        summary.rows_produced
    );
    assert_eq!(u32::from_le_bytes(ack[116..120].try_into().unwrap()), 1);
    assert!(ack[160..224].iter().any(|byte| *byte != 0));
    let schema = decode_feature_schema_config(&ack[HELLO_ACK_FIXED_LEN..]).unwrap();
    assert_eq!(schema.max_actions, 256);

    let mut sample = Vec::new();
    sample.extend_from_slice(&1u32.to_le_bytes());
    sample.extend_from_slice(&0u32.to_le_bytes());
    sample.extend_from_slice(&summary.rows_produced.to_le_bytes());
    sample.extend_from_slice(&123u64.to_le_bytes());
    write_frame(&mut stream, 3, &[&sample]);
    let (frame_type, result) = read_frame(&mut stream);
    assert_eq!(frame_type, 4);
    let batch_len = u32::from_le_bytes(result[0..4].try_into().unwrap()) as usize;
    let batch = FeatureBatchView::parse(&result[4..4 + batch_len]).unwrap();
    let targets = TrainingTargetsView::parse(&result[4 + batch_len..]).unwrap();
    assert_eq!(batch.row_count, 1);
    assert_eq!(targets.row_count, 1);
    assert_eq!(targets.value_valid[0], 1);
    assert_eq!(targets.horizon_value_valid[0], 1);
    assert_eq!(targets.max_actions, batch.max_actions);

    drop(stream);
    server.join().unwrap().unwrap();
}

#[test]
fn replay_serve_acks_an_initialized_empty_store() {
    let dir = TestDir::new();
    init_replay(ReplayInitConfig {
        replay_dir: Some(dir.path().to_path_buf()),
        max_candidates: 15,
        mask_stop: false,
        ..ReplayInitConfig::default()
    })
    .unwrap();
    let socket = dir.path().join("empty.sock");
    let server = spawn_one(dir.path(), &socket, 4);
    let mut stream = connect_hello(&socket);

    let (frame_type, ack) = read_frame(&mut stream);
    assert_eq!(frame_type, 2);
    assert_eq!(u64::from_le_bytes(ack[40..48].try_into().unwrap()), 0);
    assert_eq!(u32::from_le_bytes(ack[116..120].try_into().unwrap()), 0);
    assert!(ack[120..160].iter().all(|byte| *byte == 0));
    assert!(ack[160..224].iter().any(|byte| *byte != 0));
    assert_eq!(
        decode_feature_schema_config(&ack[HELLO_ACK_FIXED_LEN..])
            .unwrap()
            .max_actions,
        16
    );

    drop(stream);
    server.join().unwrap().unwrap();
}

#[test]
fn replay_serve_validates_required_paths_and_batch_size() {
    let config = ReplayServeConfig {
        replay_dir: PathBuf::new(),
        socket: PathBuf::new(),
        max_batch: 0,
    };
    assert!(config.validate().unwrap_err().contains("replay-dir"));

    let config = ReplayServeConfig {
        replay_dir: PathBuf::from("replay"),
        socket: PathBuf::from("sample.sock"),
        max_batch: 0,
    };
    assert!(config.validate().unwrap_err().contains("max-batch"));
}

fn spawn_one(
    replay_dir: &Path,
    socket: &Path,
    max_batch: usize,
) -> std::thread::JoinHandle<Result<(), String>> {
    let config = ReplayServeConfig {
        replay_dir: replay_dir.to_path_buf(),
        socket: socket.to_path_buf(),
        max_batch,
    };
    std::thread::spawn(move || run_one(config))
}

fn connect_hello(path: &Path) -> UnixStream {
    let mut stream = connect_retry(path);
    let mut hello = Vec::new();
    hello.extend_from_slice(&SAMPLE_PROTOCOL_VERSION.to_le_bytes());
    hello.extend_from_slice(&ENCODING_VERSION.to_le_bytes());
    write_frame(&mut stream, 1, &[&hello]);
    stream
}

fn connect_retry(path: &Path) -> UnixStream {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match UnixStream::connect(path) {
            Ok(stream) => return stream,
            Err(error) if Instant::now() < deadline => {
                assert!(matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
                ));
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("{error}"),
        }
    }
}

fn write_frame(stream: &mut UnixStream, frame_type: u8, parts: &[&[u8]]) {
    let body_len = 1 + parts.iter().map(|part| part.len()).sum::<usize>();
    stream.write_all(&(body_len as u32).to_le_bytes()).unwrap();
    stream.write_all(&[frame_type]).unwrap();
    for part in parts {
        stream.write_all(part).unwrap();
    }
}

fn read_frame(stream: &mut UnixStream) -> (u8, Vec<u8>) {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).unwrap();
    let body_len = u32::from_le_bytes(len) as usize;
    let mut body = vec![0; body_len];
    stream.read_exact(&mut body).unwrap();
    (body[0], body[1..].to_vec())
}
