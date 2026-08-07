//! Per-episode outcome dump for a replay store. One line per episode,
//! tab-separated, for offline distribution analysis and trajectory inspection.
//!
//! Usage: cargo run --release -p gz-cli --example store_stats -- <replay-dir> [start] [end]

use gz_engine::PortableSearchActionRef;
use gz_replay::{ReplayEpisodeId, ReplayStore};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: store_stats <replay-dir> [start] [end]");
    let store = ReplayStore::open(std::path::Path::new(&dir)).expect("open store");
    let start: u64 = args.next().map_or(0, |s| s.parse().expect("start"));
    // Episode ids are per-perspective and dense; episode_counters() counts
    // completed games (one per pair), so it must not cap this loop.
    let end: u64 = args.next().map_or(u64::MAX, |s| s.parse().expect("end"));

    println!("episode\tcost_ms\tlen\tstopped\tlabel\tgraph_hash\tsamples_ms\tactions");
    for id in start..end {
        let Ok(Some(record)) = store.episode(ReplayEpisodeId::new(id)) else {
            break;
        };
        let samples = record
            .final_measure
            .latency
            .as_ref()
            .map(|latency| {
                latency
                    .samples_ms
                    .iter()
                    .map(|sample| format!("{sample:.9}"))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default();
        let actions = record
            .steps
            .iter()
            .map(|step| match step.action {
                PortableSearchActionRef::Candidate(candidate) => {
                    candidate.candidate_hash.to_string()
                }
                PortableSearchActionRef::Stop { .. } => "STOP".to_owned(),
            })
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{id}\t{:.9}\t{}\t{}\t{}\t{}\t{samples}\t{actions}",
            -record.outcome.reward,
            record.row_count,
            u8::from(record.outcome.stopped),
            record
                .outcome
                .value_target
                .map_or_else(|| "-".to_owned(), |v| format!("{v:+.6}")),
            record.final_graph.graph.graph_hash,
        );
    }
}
