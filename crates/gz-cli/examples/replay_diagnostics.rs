//! Statistical diagnostic for replay targets and action features.
//!
//! Usage: replay_diagnostics <replay-dir> [sample-count] [window-rows] [seed]

use gz_engine::PortableSearchActionRef;
use gz_features::{STOP_ACTION_KIND_TOKEN, decode_feature_row};
use gz_replay::{ReplayRowKind, ReplayStore, SampleConfig};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::num::{NonZeroU64, NonZeroUsize};

#[derive(Default)]
struct Bucket {
    rows: u64,
    actions: u64,
    entropy: f64,
    uniform_advantage: f64,
    max_mass: f64,
    stop_mass: f64,
    selected_mass: f64,
    abs_value: f64,
    value: f64,
    value_sq: f64,
}

impl Bucket {
    fn print(&self, name: &str) {
        if self.rows == 0 {
            return;
        }
        let n = self.rows as f64;
        let mean = self.value / n;
        println!(
            "{name}\trows={}\tactions={:.2}\tentropy={:.5}\tuniform_adv={:.5}\tmax_mass={:.5}\tstop_mass={:.5}\tselected_mass={:.5}\tvalue_mean={:.5}\tvalue_abs={:.5}\tzero_mse={:.5}\tmean_mse={:.5}",
            self.rows,
            self.actions as f64 / n,
            self.entropy / n,
            self.uniform_advantage / n,
            self.max_mass / n,
            self.stop_mass / n,
            self.selected_mass / n,
            mean,
            self.abs_value / n,
            self.value_sq / n,
            self.value_sq / n - mean * mean,
        );
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: replay_diagnostics <replay-dir> [sample-count] [window-rows] [seed]");
    let store = ReplayStore::open(std::path::Path::new(&dir)).expect("open store");
    let counters = store.counters();
    let sample_count = args
        .next()
        .map_or(100_000, |value| value.parse().expect("sample-count"));
    let window_rows = args.next().map_or(counters.produced_rows, |value| {
        value.parse().expect("window-rows")
    });
    let seed = args.next().map_or(42, |value| value.parse().expect("seed"));
    let rows = store
        .inspect_rows(SampleConfig {
            batch: NonZeroUsize::new(sample_count).expect("positive sample-count"),
            window_rows: NonZeroU64::new(window_rows).expect("positive window-rows"),
            seed,
        })
        .expect("sample rows");

    let mut all = Bucket::default();
    let mut terminal = Bucket::default();
    let mut by_step = BTreeMap::<u32, Bucket>::new();
    let mut kind_rows = BTreeMap::<u32, u64>::new();
    let mut kind_mass = BTreeMap::<u32, f64>::new();
    let mut episodes = BTreeSet::new();
    let mut model_versions = BTreeSet::new();
    let mut by_model_version = BTreeMap::<String, Bucket>::new();
    let mut feature_targets = BTreeMap::<u64, (u64, f64, f64)>::new();
    let mut state_hashes = BTreeSet::new();
    let mut state_step_targets = BTreeMap::new();
    let mut root_selected_kinds = BTreeMap::<u32, (u64, f64, f64, u64)>::new();
    let mut invalid = 0_u64;

    for (episode, row) in rows {
        episodes.insert(episode);
        state_hashes.insert(row.state.graph.graph_hash);
        if let Some(value) = row.value_target {
            let entry = state_step_targets
                .entry((row.state.graph.graph_hash, row.step_index))
                .or_insert((0_u64, 0.0_f64, 0.0_f64));
            entry.0 += 1;
            entry.1 += f64::from(value);
            entry.2 += f64::from(value).powi(2);
        }
        let model_version = row.model_version.map(|version| version.to_string());
        if let Some(version) = row.model_version {
            model_versions.insert(version);
        }
        let bucket = if row.kind == ReplayRowKind::Terminal {
            &mut terminal
        } else {
            by_step.entry(row.step_index).or_default()
        };
        for destination in [&mut all, bucket] {
            destination.rows += 1;
            destination.actions += row.legal_actions.len() as u64;
            if let Some(value) = row.value_target {
                let value = f64::from(value);
                destination.value += value;
                destination.abs_value += value.abs();
                destination.value_sq += value * value;
            }
        }
        if let Some(version) = model_version.as_ref() {
            let destination = by_model_version.entry(version.clone()).or_default();
            destination.rows += 1;
            destination.actions += row.legal_actions.len() as u64;
            if let Some(value) = row.value_target {
                let value = f64::from(value);
                destination.value += value;
                destination.abs_value += value.abs();
                destination.value_sq += value * value;
            }
        }

        if row.kind == ReplayRowKind::Terminal {
            continue;
        }
        let sum: f64 = row.policy_target.iter().copied().map(f64::from).sum();
        if row.policy_target.len() != row.legal_actions.len() || (sum - 1.0).abs() > 0.01 {
            invalid += 1;
            continue;
        }
        let entropy = -row
            .policy_target
            .iter()
            .copied()
            .filter(|mass| *mass > 0.0)
            .map(|mass| f64::from(mass) * f64::from(mass).ln())
            .sum::<f64>();
        let max_mass = row.policy_target.iter().copied().fold(0.0_f32, f32::max) as f64;
        let stop_index = row
            .legal_actions
            .iter()
            .position(|action| matches!(action, PortableSearchActionRef::Stop { .. }));
        let selected_mass = row.selected_action.as_ref().and_then(|selected| {
            row.legal_actions
                .iter()
                .position(|action| action == selected)
                .map(|index| f64::from(row.policy_target[index]))
        });
        for destination in [&mut all, bucket] {
            destination.entropy += entropy;
            destination.uniform_advantage += (row.legal_actions.len() as f64).ln() - entropy;
            destination.max_mass += max_mass;
            destination.stop_mass +=
                stop_index.map_or(0.0, |index| f64::from(row.policy_target[index]));
            destination.selected_mass += selected_mass.unwrap_or(0.0);
        }
        if let Some(version) = model_version.as_ref() {
            let destination = by_model_version.entry(version.clone()).or_default();
            destination.entropy += entropy;
            destination.uniform_advantage += (row.legal_actions.len() as f64).ln() - entropy;
            destination.max_mass += max_mass;
            destination.stop_mass +=
                stop_index.map_or(0.0, |index| f64::from(row.policy_target[index]));
            destination.selected_mass += selected_mass.unwrap_or(0.0);
        }

        if let Some(encoded) = row.feature_row.as_deref() {
            let mut hasher = DefaultHasher::new();
            encoded.hash(&mut hasher);
            let entry = feature_targets.entry(hasher.finish()).or_default();
            entry.0 += 1;
            if let Some(value) = row.value_target {
                entry.1 += f64::from(value);
                entry.2 += f64::from(value).powi(2);
            }
            let features = decode_feature_row(encoded).expect("decode feature row");
            if features.actions.len() != row.policy_target.len() {
                invalid += 1;
                continue;
            }
            for (action, mass) in features.actions.iter().zip(&row.policy_target) {
                *kind_rows.entry(action.kind_token).or_default() += 1;
                *kind_mass.entry(action.kind_token).or_default() += f64::from(*mass);
            }
            if row.step_index == 0
                && let Some(selected) = row.selected_action.as_ref()
                && let Some(index) = row
                    .legal_actions
                    .iter()
                    .position(|action| action == selected)
                && let (Some(action), Some(value)) = (features.actions.get(index), row.value_target)
            {
                let entry = root_selected_kinds.entry(action.kind_token).or_default();
                entry.0 += 1;
                entry.1 += f64::from(value);
                entry.2 += f64::from(action.static_prior);
                entry.3 += action.subjects.len() as u64;
            }
        }
    }

    println!(
        "store\tproduced_rows={}\tconsumed_rows={}\tsampled_episodes={}\tmodel_versions={}\tunique_states={}\tunique_features={}\tduplicate_feature_rows={}\tinvalid_rows={invalid}",
        counters.produced_rows,
        counters.consumed_rows,
        episodes.len(),
        model_versions.len(),
        state_hashes.len(),
        feature_targets.len(),
        feature_targets
            .values()
            .map(|entry| entry.0 - 1)
            .sum::<u64>(),
    );
    let duplicate_irreducible_mse = feature_targets
        .values()
        .filter(|entry| entry.0 > 1)
        .map(|entry| entry.2 - entry.1 * entry.1 / entry.0 as f64)
        .sum::<f64>();
    let duplicate_rows = feature_targets
        .values()
        .filter(|entry| entry.0 > 1)
        .map(|entry| entry.0)
        .sum::<u64>();
    println!(
        "duplicate_features\trows={duplicate_rows}\tirreducible_mse={:.6}",
        duplicate_irreducible_mse / duplicate_rows.max(1) as f64,
    );
    let aliased_state_rows = state_step_targets
        .values()
        .filter(|entry| entry.0 > 1)
        .map(|entry| entry.0)
        .sum::<u64>();
    let aliased_state_mse = state_step_targets
        .values()
        .filter(|entry| entry.0 > 1)
        .map(|entry| entry.2 - entry.1 * entry.1 / entry.0 as f64)
        .sum::<f64>()
        / aliased_state_rows.max(1) as f64;
    println!(
        "duplicate_state_steps\trows={aliased_state_rows}\tirreducible_mse={aliased_state_mse:.6}",
    );
    all.print("all");
    terminal.print("terminal");
    for (step, bucket) in &by_step {
        bucket.print(&format!("step_{step}"));
    }
    for (version, bucket) in &by_model_version {
        bucket.print(&format!("model_{version}"));
    }
    println!("kind\taction_occurrences\tmean_target_mass_when_present");
    for (kind, count) in kind_rows {
        let label = if kind == STOP_ACTION_KIND_TOKEN {
            "STOP"
        } else {
            "candidate"
        };
        println!(
            "{kind}:{label}\t{count}\t{:.7}",
            kind_mass[&kind] / count as f64
        );
    }
    println!("root_selected_kind\trows\tvalue_mean\tstatic_prior_mean\tsubjects_mean");
    for (kind, (rows, value, prior, subjects)) in root_selected_kinds {
        println!(
            "{kind}\t{rows}\t{:.6}\t{:.6}\t{:.3}",
            value / rows as f64,
            prior / rows as f64,
            subjects as f64 / rows as f64,
        );
    }
}
