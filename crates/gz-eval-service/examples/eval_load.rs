use gz_engine::{ActionSetHash, EngineId, EngineVersion};
use gz_eval_service::{
    EvaluatorProcess, EvaluatorProcessConfig, FeatureEvalBackend, Hello, StubBackend,
};
use gz_features::{
    ActionFeature, FeatureCollator, FeatureRow, FeatureSchema, FeatureSchemaConfig,
    PositionFeatures, STOP_ACTION_KIND_TOKEN,
};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1).collect())?;
    let schema = schema();

    match args.backend {
        BackendKind::Stub => {
            let report = run_backend(StubBackend, &schema, args.batches, args.batch_size)?;
            print_report("stub", &report);
        }
        BackendKind::Process => {
            let mut process = EvaluatorProcess::spawn(EvaluatorProcessConfig {
                working_dir: args.python_dir.unwrap_or_else(|| PathBuf::from("python")),
                socket_path: temp_socket(),
                ready_timeout: Duration::from_secs(10),
                io_timeout: Duration::from_secs(30),
                ..EvaluatorProcessConfig::default()
            })
            .map_err(|error| error.to_string())?;
            let hello = Hello::new(
                schema.hash(),
                args.batch_size as u32,
                EngineId::from_bytes([1; 16]),
                EngineVersion::from_bytes([2; 16]),
                ActionSetHash::from_bytes([3; 32]),
            );
            let backend = process.connect(&hello).map_err(|error| error.to_string())?;
            let report = run_backend(backend, &schema, args.batches, args.batch_size)?;
            print_report("process", &report);
            wait_for_process_exit(&mut process)?;
        }
    }

    Ok(())
}

fn run_backend<B>(
    mut backend: B,
    schema: &FeatureSchema,
    batches: usize,
    batch_size: usize,
) -> Result<Report, String>
where
    B: FeatureEvalBackend,
{
    let mut collator = FeatureCollator::new(
        schema.clone(),
        NonZeroUsize::new(batch_size).ok_or("batch size must be positive")?,
    );
    let mut rows = Vec::with_capacity(batch_size);
    let mut action_counts = Vec::with_capacity(batch_size);
    let mut bytes = Vec::new();
    let mut latencies = Vec::with_capacity(batches);
    let total_start = Instant::now();

    for batch in 0..batches {
        rows.clear();
        action_counts.clear();
        for row in 0..batch_size {
            let feature_row = synthetic_row(batch as u64, row as u64);
            action_counts.push(feature_row.actions.len() as u32);
            rows.push(feature_row);
        }
        collator
            .collate_into(&rows, &mut bytes)
            .map_err(|error| error.to_string())?;
        let start = Instant::now();
        let outputs = backend
            .eval(&bytes, &action_counts)
            .map_err(|error| error.to_string())?;
        if outputs.rows.len() != batch_size {
            return Err("backend returned wrong row count".to_owned());
        }
        latencies.push(start.elapsed().as_micros() as u64);
    }

    latencies.sort_unstable();
    let elapsed = total_start.elapsed();
    let rows_total = batches * batch_size;
    Ok(Report {
        batches,
        rows: rows_total,
        rows_per_second: rows_total as f64 / elapsed.as_secs_f64(),
        p50_us: percentile(&latencies, 50),
        p95_us: percentile(&latencies, 95),
        max_us: latencies.last().copied().unwrap_or(0),
    })
}

fn schema() -> FeatureSchema {
    FeatureSchema::new(FeatureSchemaConfig {
        name: "eval-load-v1".to_owned(),
        node_vocab_size: 64,
        node_attr_dim: 1,
        edge_type_count: 2,
        action_kind_vocab_size: 32,
        max_nodes: 8,
        max_edges: 8,
        max_actions: 8,
        max_subjects: 2,
        opponent_reward_scale: 256.0,
        expander_degree: 0,
        expander_seed: 0,
    })
    .unwrap()
}

fn synthetic_row(batch: u64, row: u64) -> FeatureRow {
    let node_count = 1 + ((batch.wrapping_mul(17) + row.wrapping_mul(5)) % 8) as u32;
    let action_count = 1 + ((batch.wrapping_mul(3) + row.wrapping_mul(7)) % 8) as usize;
    let mut actions = Vec::with_capacity(action_count);
    for index in 0..action_count {
        if index + 1 == action_count {
            actions.push(ActionFeature {
                kind_token: STOP_ACTION_KIND_TOKEN,
                static_prior: 0.0,
                subjects: Vec::new(),
            });
        } else {
            actions.push(ActionFeature {
                kind_token: 2 + (index as u32 % 16),
                static_prior: (index as f32) * 0.01,
                subjects: vec![index as u32 % node_count],
            });
        }
    }

    FeatureRow {
        node_count,
        node_tokens: (0..node_count)
            .map(|index| 2 + ((batch + row + u64::from(index)) % 32) as u16)
            .collect(),
        node_attrs: (0..node_count)
            .map(|index| (index as f32 + row as f32) * 0.125)
            .collect(),
        edges: Vec::new(),
        actions,
        position: PositionFeatures {
            root_step: batch as u32,
            leaf_depth: row as u32,
            budget_fraction: 1.0,
            budget_step: 0.0,
            opponent_reward: 0.0,
            opponent_present: false,
        },
    }
}

fn percentile(values: &[u64], percentile: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let index = (values.len() - 1) * percentile / 100;
    values[index]
}

fn print_report(backend: &str, report: &Report) {
    println!(
        "backend={} batches={} rows={} rows_per_s={:.1} p50_us={} p95_us={} max_us={}",
        backend,
        report.batches,
        report.rows,
        report.rows_per_second,
        report.p50_us,
        report.p95_us,
        report.max_us
    );
}

fn wait_for_process_exit(process: &mut EvaluatorProcess) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match process.try_wait().map_err(|error| error.to_string())? {
            Some(status) if status.success() => return Ok(()),
            Some(status) => return Err(format!("Python evaluator exited with {status}")),
            None if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            None => return Err("Python evaluator did not exit".to_owned()),
        }
    }
}

fn temp_socket() -> PathBuf {
    std::env::temp_dir().join(format!("gz-eval-load-{}.sock", std::process::id()))
}

struct Report {
    batches: usize,
    rows: usize,
    rows_per_second: f64,
    p50_us: u64,
    p95_us: u64,
    max_us: u64,
}

struct Args {
    backend: BackendKind,
    python_dir: Option<PathBuf>,
    batches: usize,
    batch_size: usize,
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        let mut backend = None;
        let mut python_dir = None;
        let mut batches = None;
        let mut batch_size = None;
        let mut index = 0;

        while index < args.len() {
            let flag = &args[index];
            index += 1;
            let Some(value) = args.get(index) else {
                return Err(format!("missing value for {flag}\n{}", usage()));
            };
            index += 1;

            match flag.as_str() {
                "--backend" => backend = Some(value.parse()?),
                "--python-dir" => python_dir = Some(PathBuf::from(value)),
                "--batches" => batches = Some(parse_usize(flag, value)?),
                "--batch-size" => batch_size = Some(parse_usize(flag, value)?),
                _ => return Err(format!("unknown flag: {flag}\n{}", usage())),
            }
        }

        Ok(Self {
            backend: backend.ok_or_else(|| format!("missing --backend\n{}", usage()))?,
            python_dir,
            batches: batches.ok_or_else(|| format!("missing --batches\n{}", usage()))?,
            batch_size: batch_size.ok_or_else(|| format!("missing --batch-size\n{}", usage()))?,
        })
    }
}

#[derive(Clone, Copy)]
enum BackendKind {
    Stub,
    Process,
}

impl std::str::FromStr for BackendKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "stub" => Ok(Self::Stub),
            "process" => Ok(Self::Process),
            _ => Err(format!("unknown backend: {value}")),
        }
    }
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    let value = value
        .parse()
        .map_err(|_| format!("{flag} expects a positive integer"))?;
    if value == 0 {
        return Err(format!("{flag} must be greater than zero"));
    }
    Ok(value)
}

fn usage() -> &'static str {
    "usage: eval_load --backend stub|process [--python-dir PATH] --batches N --batch-size B"
}
