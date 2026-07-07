#![forbid(unsafe_code)]

use gz_cli::selfplay::{SelfplayConfig, run};

// glibc malloc is pathological for this binary either way: default
// per-thread arenas retain ~17 MB/s of fragmentation across ~300 threads,
// and capping arenas serializes allocation (4.6x wall-clock at cap 2).
// jemalloc gives per-thread caches AND purges retained pages.
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
use gz_cli::serve::{ReplayServeConfig, run as run_replay_serve};
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        eprintln!("{}", usage());
        std::process::exit(2);
    };

    let result = match command.as_str() {
        "selfplay" => parse_selfplay(args.collect()).and_then(run).map(|summary| {
            println!(
                "episodes appended={} dropped={} rows={} labels win/loss/tie={}/{}/{} eval_batches={} mean_batch={:.3} evaluator={} model_version={} counters produced={} consumed={}",
                summary.episodes_appended,
                summary.episodes_dropped,
                summary.rows_produced,
                summary.wins,
                summary.losses,
                summary.ties,
                summary.eval_batch_count,
                summary.mean_eval_batch_size,
                summary.evaluator.as_str(),
                summary
                    .model_version
                    .map_or_else(|| "-".to_owned(), |version| version.to_string()),
                summary.counters.produced_rows,
                summary.counters.consumed_rows,
            );
            if std::env::var_os("GZ_HASH_VOLUME_STATS").is_some() {
                println!(
                    "hash_volume_contexts search={} replay_rows={} reference_steps={} total={}",
                    summary.search_contexts,
                    summary.replay_rows,
                    summary.reference_steps,
                    summary.search_contexts + summary.replay_rows + summary.reference_steps,
                );
            }
        }),
        "replay-serve" => parse_replay_serve(args.collect()).and_then(run_replay_serve),
        _ => Err(format!("unknown command: {command}\n{}", usage())),
    };

    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn parse_replay_serve(args: Vec<String>) -> Result<ReplayServeConfig, String> {
    let mut replay_dir = None;
    let mut socket = None;
    let mut max_batch = None;
    let mut index = 0;

    while index < args.len() {
        let flag = &args[index];
        index += 1;

        let Some(value) = args.get(index) else {
            return Err(format!("missing value for {flag}\n{}", usage()));
        };
        index += 1;

        match flag.as_str() {
            "--replay-dir" => replay_dir = Some(PathBuf::from(value)),
            "--socket" => socket = Some(PathBuf::from(value)),
            "--max-batch" => max_batch = Some(parse_usize(flag, value)?),
            _ => return Err(format!("unknown flag: {flag}\n{}", usage())),
        }
    }

    let config = ReplayServeConfig {
        replay_dir: replay_dir
            .ok_or_else(|| format!("missing required --replay-dir\n{}", usage()))?,
        socket: socket.ok_or_else(|| format!("missing required --socket\n{}", usage()))?,
        max_batch: max_batch.ok_or_else(|| format!("missing required --max-batch\n{}", usage()))?,
    };
    config.validate()?;
    Ok(config)
}

fn parse_selfplay(args: Vec<String>) -> Result<SelfplayConfig, String> {
    let mut config = SelfplayConfig::default();
    let mut max_batch = None;
    let mut index = 0;

    while index < args.len() {
        let flag = &args[index];
        index += 1;

        let Some(value) = args.get(index) else {
            return Err(format!("missing value for {flag}\n{}", usage()));
        };
        index += 1;

        match flag.as_str() {
            "--replay-dir" => config.replay_dir = Some(PathBuf::from(value)),
            "--episodes" => config.episodes = parse_u64(flag, value)?,
            "--lanes" => config.lanes = parse_usize(flag, value)?,
            "--workers-per-lane" => config.workers_per_lane = parse_usize(flag, value)?,
            "--reference" => config.reference = value.parse()?,
            "--root-mode" => config.root_mode = value.parse()?,
            "--reference-ema-decay" => config.reference_ema_decay = parse_f32(flag, value)?,
            "--evaluator" => config.evaluator = value.parse()?,
            "--python-dir" => config.python_dir = Some(PathBuf::from(value)),
            "--checkpoint-dir" => config.checkpoint_dir = Some(PathBuf::from(value)),
            "--eval-device" => config.eval_device = Some(value.clone()),
            "--eval-poll-interval" => {
                config.eval_poll_interval = Some(parse_f32(flag, value)?);
            }
            "--seed" => config.seed = parse_u64(flag, value)?,
            "--max-steps" => config.max_steps = parse_usize(flag, value)?,
            "--simulations" => config.simulations = parse_usize(flag, value)?,
            "--max-considered" => config.max_considered = parse_usize(flag, value)?,
            "--gumbel-scale" => config.gumbel_scale = parse_f32(flag, value)?,
            "--tree-reuse" => config.tree_reuse = parse_bool(flag, value)?,
            "--max-candidates" => config.max_candidates = parse_usize(flag, value)?,
            "--max-batch" => max_batch = Some(parse_usize(flag, value)?),
            "--serve-socket" => config.serve_socket = Some(PathBuf::from(value)),
            "--serve-max-batch" => config.serve_max_batch = parse_usize(flag, value)?,
            "--replay-backlog" => config.replay_backlog = Some(parse_u64(flag, value)?),
            "--replay-retain" => config.replay_retain = Some(parse_u64(flag, value)?),
            "--position-features" => config.position_features = parse_bool(flag, value)?,
            "--eval-processes" => config.eval_processes = parse_usize(flag, value)?,
            _ => return Err(format!("unknown flag: {flag}\n{}", usage())),
        }
    }

    config.max_batch = max_batch.unwrap_or(config.lanes * config.workers_per_lane);
    config.validate()?;
    Ok(config)
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects an unsigned integer"))
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects a positive integer"))
}

fn parse_f32(flag: &str, value: &str) -> Result<f32, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects a number"))
}

fn parse_bool(flag: &str, value: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("{flag} expects true or false")),
    }
}

fn usage() -> &'static str {
    "usage: graphzero selfplay --replay-dir PATH [--episodes N; 0 = unbounded] [--lanes L] [--workers-per-lane W] [--reference root|greedy|beam|random|self-average|policy|gated-policy|none] [--root-mode generated|fixed] [--reference-ema-decay D] [--evaluator random|stub|process-stub|torch] [--python-dir PATH] [--checkpoint-dir DIR] [--eval-device DEV] [--eval-poll-interval SECS] [--seed S] [--max-steps M] [--simulations K] [--max-considered M] [--gumbel-scale G] [--tree-reuse true|false] [--max-candidates N] [--max-batch B] [--serve-socket PATH] [--serve-max-batch B] [--replay-backlog ROWS] [--replay-retain ROWS] [--position-features true|false] [--eval-processes N]\n       graphzero replay-serve --replay-dir PATH --socket PATH --max-batch B"
}
