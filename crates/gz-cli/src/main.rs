#![forbid(unsafe_code)]

use gz_cli::distill::{DistillGenerateConfig, generate as generate_distill};
use gz_cli::remote_measure::RemoteAgentEnrollment;
use gz_cli::selfplay::{ReplayInitConfig, SelfplayConfig, init_replay, run};
use gz_measure_service::DeviceId;

// glibc malloc is pathological for this binary either way: default
// per-thread arenas retain ~17 MB/s of fragmentation across ~300 threads,
// and capping arenas serializes allocation (4.6x wall-clock at cap 2).
// jemalloc gives per-thread caches AND purges retained pages.
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
use gz_cli::serve::{ReplayServeConfig, run as run_replay_serve};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

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
        }),
        "replay-init" => parse_replay_init(args.collect())
            .and_then(init_replay)
            .map(|summary| {
                println!(
                    "replay initialized feature_schema_hash={} max_actions={}",
                    summary.feature_schema_hash, summary.max_actions,
                );
            }),
        "distill-generate" => parse_distill_generate(args.collect())
            .and_then(generate_distill)
            .map(|summary| {
                println!(
                    "distillation states={} attempts={} duplicates={} candidate_overflows={} stop_targets={} improving_actions={} elapsed_s={:.3} rows_per_s={:.3}",
                    summary.states,
                    summary.attempts,
                    summary.duplicate_states,
                    summary.candidate_overflows,
                    summary.stop_targets,
                    summary.improving_actions,
                    summary.elapsed.as_secs_f64(),
                    summary.states as f64 / summary.elapsed.as_secs_f64().max(f64::EPSILON),
                );
            }),
        "replay-serve" => parse_replay_serve(args.collect()).and_then(run_replay_serve),
        _ => Err(format!("unknown command: {command}\n{}", usage())),
    };

    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn parse_distill_generate(args: Vec<String>) -> Result<DistillGenerateConfig, String> {
    let mut config = DistillGenerateConfig::default();
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
            "--states" => config.states = parse_u64(flag, value)?,
            "--workers" => config.workers = parse_usize(flag, value)?,
            "--max-attempts" => config.max_attempts = parse_u64(flag, value)?,
            "--seed" => config.seed = parse_u64(flag, value)?,
            "--max-candidates" => config.max_candidates = parse_usize(flag, value)?,
            "--max-steps" => config.max_steps = parse_usize(flag, value)?,
            "--position-features" => config.position_features = parse_bool(flag, value)?,
            "--teacher" => config.teacher = value.parse()?,
            _ => return Err(format!("unknown flag: {flag}\n{}", usage())),
        }
    }
    config.validate()?;
    Ok(config)
}

fn parse_replay_init(args: Vec<String>) -> Result<ReplayInitConfig, String> {
    let mut config = ReplayInitConfig::default();
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
            "--max-candidates" => config.max_candidates = parse_usize(flag, value)?,
            "--mask-stop" => config.mask_stop = parse_bool(flag, value)?,
            _ => return Err(format!("unknown flag: {flag}\n{}", usage())),
        }
    }

    config.validate()?;
    Ok(config)
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
            "--evaluator" => config.evaluator = value.parse()?,
            "--python-dir" => config.python_dir = Some(PathBuf::from(value)),
            "--checkpoint-dir" => config.checkpoint_dir = Some(PathBuf::from(value)),
            "--checkpoint-pointer" => config.checkpoint_pointer = Some(value.clone()),
            "--eval-device" => config.eval_device = Some(value.clone()),
            "--eval-poll-interval" => {
                config.eval_poll_interval = Some(parse_f32(flag, value)?);
            }
            "--seed" => config.seed = parse_u64(flag, value)?,
            "--max-steps" => config.max_steps = parse_usize(flag, value)?,
            "--simulations" => config.simulations = parse_usize(flag, value)?,
            "--max-considered" => config.max_considered = parse_usize(flag, value)?,
            "--gumbel-scale" => config.gumbel_scale = parse_f32(flag, value)?,
            "--c-visit" => config.c_visit = parse_f32(flag, value)?,
            "--c-scale" => config.c_scale = parse_f32(flag, value)?,
            "--gumbel-noise-overlap" => config.gumbel_noise_overlap = parse_f32(flag, value)?,
            "--tree-reuse" => config.tree_reuse = parse_bool(flag, value)?,
            "--max-candidates" => config.max_candidates = parse_usize(flag, value)?,
            "--max-batch" => max_batch = Some(parse_usize(flag, value)?),
            "--serve-socket" => config.serve_socket = Some(PathBuf::from(value)),
            "--serve-max-batch" => config.serve_max_batch = parse_usize(flag, value)?,
            "--replay-backlog" => config.replay_backlog = Some(parse_u64(flag, value)?),
            "--replay-retain" => config.replay_retain = Some(parse_u64(flag, value)?),
            "--position-features" => config.position_features = parse_bool(flag, value)?,
            "--no-backtrack" => config.no_backtrack = parse_bool(flag, value)?,
            "--mask-stop" => config.mask_stop = parse_bool(flag, value)?,
            "--eval-processes" => config.eval_processes = parse_usize(flag, value)?,
            "--admission-stagger-ms" => config.admission_stagger_ms = parse_u64(flag, value)?,
            "--admission-smoothing" => config.admission_smoothing = parse_bool(flag, value)?,
            "--measure-listen" => {
                config.remote_measure.listen = Some(
                    value
                        .parse()
                        .map_err(|_| "--measure-listen expects HOST:PORT".to_owned())?,
                );
            }
            "--measure-server-cert" => {
                config.remote_measure.server_certificate = Some(PathBuf::from(value));
            }
            "--measure-server-key" => {
                config.remote_measure.server_private_key = Some(PathBuf::from(value));
            }
            "--measure-client-ca" => {
                config.remote_measure.client_ca = Some(PathBuf::from(value));
            }
            "--measure-agent" => {
                let (device_id, certificate) = value.split_once('=').ok_or_else(|| {
                    "--measure-agent expects DEVICE_ID=CERTIFICATE_PATH".to_owned()
                })?;
                config.remote_measure.agents.push(RemoteAgentEnrollment {
                    device_id: DeviceId::from_str(device_id).map_err(|error| error.to_string())?,
                    certificate: PathBuf::from(certificate),
                });
            }
            "--measure-profile" => config.remote_measure.profile = Some(value.clone()),
            "--measure-receipt-dir" => {
                config.remote_measure.receipt_dir = Some(PathBuf::from(value));
            }
            "--measure-startup-timeout-ms" => {
                config.remote_measure.startup_timeout =
                    Duration::from_millis(parse_u64(flag, value)?);
            }
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
    "usage: graphzero selfplay --replay-dir PATH [--episodes N; 0 = unbounded] [--lanes L] [--workers-per-lane W] [--evaluator stub|process-stub|torch] [--python-dir PATH] [--checkpoint-dir DIR] [--checkpoint-pointer FILE] [--eval-device DEV] [--eval-poll-interval SECS] [--seed S] [--max-steps M] [--simulations K] [--max-considered M] [--gumbel-scale G] [--c-visit C] [--c-scale C] [--gumbel-noise-overlap V; negative disables] [--tree-reuse true|false] [--max-candidates N] [--max-batch B] [--serve-socket PATH] [--serve-max-batch B] [--replay-backlog ROWS] [--replay-retain ROWS] [--position-features true|false] [--no-backtrack true|false] [--mask-stop true|false] [--eval-processes N] [--admission-stagger-ms MS] [--admission-smoothing true|false] [--measure-listen HOST:PORT --measure-server-cert PATH --measure-server-key PATH --measure-client-ca PATH --measure-agent DEVICE_ID=CERT_PATH ... --measure-profile NAME --measure-receipt-dir PATH --measure-startup-timeout-ms MS]\n       graphzero replay-init --replay-dir PATH [--max-candidates N]\n       graphzero distill-generate --replay-dir PATH [--states N] [--workers N] [--max-attempts N; 0 = 10x states] [--teacher reducing-uniform] [--seed S] [--max-candidates N] [--max-steps N] [--position-features true|false]\n       graphzero replay-serve --replay-dir PATH --socket PATH --max-batch B"
}
