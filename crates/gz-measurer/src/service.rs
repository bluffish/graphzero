use crate::{CompletedEpisodeArtifact, MeasurerError, episode_reward, project_episode};
use gz_engine::{GraphHash, ModelVersion, PortableSearchActionRef};
use gz_replay::{ReplayError, ReplayStore, SymmetricRewardTargetScale, SymmetricRewardTieEpsilon};
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredSymmetricGame {
    pub lane: usize,
    pub p1_artifact: CompletedEpisodeArtifact,
    pub p2_artifact: CompletedEpisodeArtifact,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasurerAdmission {
    pub status: MeasurerAdmissionStatus,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeasurerAdmissionStatus {
    Appended { row_count: u64 },
    Dropped { reason: MeasurerError },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeasurerRunSummary {
    pub lanes: Vec<MeasurerLaneSummary>,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub replay_rows: u64,
    pub measure_ledger: MeasureLedgerSnapshot,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeasurerLaneSummary {
    pub lane: usize,
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub replay_rows: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MeasureLedgerSnapshot {
    pub finals: u64,
    pub distinct_finals: u64,
    pub repeat_finals: u64,
    pub distinct_by_version: Vec<(ModelVersion, u64)>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MeasurerStats {
    pub episodes_appended: u64,
    pub episodes_dropped: u64,
    pub finals: u64,
    pub distinct_finals: u64,
}

pub struct ReplayMeasurer<'a> {
    store: &'a ReplayStore,
    reward_tie_epsilon: SymmetricRewardTieEpsilon,
    reward_target_scale: SymmetricRewardTargetScale,
    summary: MeasurerRunSummary,
    ledger: MeasureLedger,
}

impl<'a> ReplayMeasurer<'a> {
    #[must_use]
    pub fn new(
        store: &'a ReplayStore,
        reward_tie_epsilon: SymmetricRewardTieEpsilon,
        reward_target_scale: SymmetricRewardTargetScale,
    ) -> Self {
        Self {
            store,
            reward_tie_epsilon,
            reward_target_scale,
            summary: MeasurerRunSummary::default(),
            ledger: MeasureLedger::default(),
        }
    }

    pub fn admit_symmetric(
        &mut self,
        game: MeasuredSymmetricGame,
    ) -> Result<MeasurerAdmission, ReplayError> {
        if episode_reward(&game.p1_artifact).is_some() {
            self.ledger.observe(&game.p1_artifact);
        }
        if episode_reward(&game.p2_artifact).is_some() {
            self.ledger.observe(&game.p2_artifact);
        }

        let projected =
            project_symmetric_game(&game, self.reward_tie_epsilon, self.reward_target_scale);
        let (status, replay_rows) = match projected {
            Ok((p1, p2, p1_target)) => {
                let value_sign_accuracy = symmetric_value_sign_accuracies(&game, p1_target);
                self.store
                    .append_episode_pair((&p1.0, &p1.1), (&p2.0, &p2.1))?;
                let row_count = (p1.1.len() + p2.1.len()) as u64;
                self.store
                    .observe_value_sign_accuracy(value_sign_accuracy.0, value_sign_accuracy.1);
                (MeasurerAdmissionStatus::Appended { row_count }, row_count)
            }
            Err(reason) => (MeasurerAdmissionStatus::Dropped { reason }, 0),
        };

        match status {
            MeasurerAdmissionStatus::Appended { row_count } => {
                self.summary.episodes_appended += 1;
                self.summary.replay_rows += row_count;
                let lane = lane_summary(&mut self.summary.lanes, game.lane);
                lane.episodes_appended += 1;
                lane.replay_rows += replay_rows;
            }
            MeasurerAdmissionStatus::Dropped { .. } => {
                self.summary.episodes_dropped += 1;
                lane_summary(&mut self.summary.lanes, game.lane).episodes_dropped += 1;
            }
        }

        Ok(MeasurerAdmission { status })
    }

    #[must_use]
    pub fn stats(&self) -> MeasurerStats {
        MeasurerStats {
            episodes_appended: self.summary.episodes_appended,
            episodes_dropped: self.summary.episodes_dropped,
            finals: self.ledger.finals,
            distinct_finals: self.ledger.seen.len() as u64,
        }
    }

    #[must_use]
    pub fn finish(mut self) -> MeasurerRunSummary {
        self.summary.measure_ledger = self.ledger.snapshot();
        self.summary
    }
}

type ProjectedEpisode = (gz_replay::ReplayEpisodeRecord, Vec<gz_replay::ReplayRow>);
type ProjectedSymmetricGame = (ProjectedEpisode, ProjectedEpisode, f32);

const VALUE_SIGN_LATE_STEP: usize = 40;

fn project_symmetric_game(
    game: &MeasuredSymmetricGame,
    reward_tie_epsilon: SymmetricRewardTieEpsilon,
    reward_target_scale: SymmetricRewardTargetScale,
) -> Result<ProjectedSymmetricGame, MeasurerError> {
    let p1_reward = episode_reward(&game.p1_artifact).ok_or(MeasurerError::Unmeasured)?;
    let p2_reward = episode_reward(&game.p2_artifact).ok_or(MeasurerError::Unmeasured)?;
    let p1_target = symmetric_outcome_target(
        p1_reward,
        p2_reward,
        symmetric_rewrite_count(&game.p1_artifact),
        symmetric_rewrite_count(&game.p2_artifact),
        reward_tie_epsilon,
        reward_target_scale,
    );
    let mut p1 = project_episode(&game.p1_artifact)?;
    let mut p2 = project_episode(&game.p2_artifact)?;
    set_value_target(&mut p1, p1_target);
    set_value_target(&mut p2, -p1_target);
    set_horizon_value_targets(&mut p1, &game.p1_artifact, p1_target)?;
    set_horizon_value_targets(&mut p2, &game.p2_artifact, -p1_target)?;
    Ok((p1, p2, p1_target))
}

fn symmetric_rewrite_count(artifact: &CompletedEpisodeArtifact) -> usize {
    artifact
        .steps
        .iter()
        .filter(|step| matches!(step.selected_action, PortableSearchActionRef::Candidate(_)))
        .count()
}

fn symmetric_outcome_target(
    p1_reward: f32,
    p2_reward: f32,
    p1_len: usize,
    p2_len: usize,
    reward_tie_epsilon: SymmetricRewardTieEpsilon,
    reward_target_scale: SymmetricRewardTargetScale,
) -> f32 {
    let reward_delta = p1_reward - p2_reward;
    if reward_target_scale.is_graded() {
        return (reward_delta / reward_target_scale.get()).tanh();
    }
    if reward_delta > reward_tie_epsilon.get() {
        1.0
    } else if reward_delta < -reward_tie_epsilon.get() {
        -1.0
    } else if p1_len < p2_len {
        1.0
    } else if p1_len > p2_len {
        -1.0
    } else {
        0.0
    }
}

fn symmetric_value_sign_accuracies(
    game: &MeasuredSymmetricGame,
    p1_target: f32,
) -> (Option<f64>, Option<f64>) {
    let mut correct = [0_u64; 2];
    let mut total = [0_u64; 2];
    for (artifact, target) in [
        (&game.p1_artifact, p1_target),
        (&game.p2_artifact, -p1_target),
    ] {
        if target == 0.0 {
            continue;
        }
        for (step, prediction) in artifact
            .steps
            .iter()
            .enumerate()
            .filter_map(|(step, artifact)| artifact.root_value.map(|value| (step, value)))
            .filter(|(_, value)| value.is_finite())
        {
            let phase = usize::from(step >= VALUE_SIGN_LATE_STEP);
            total[phase] += 1;
            correct[phase] += u64::from((prediction >= 0.0) == (target > 0.0));
        }
    }
    let accuracy =
        |phase: usize| (total[phase] != 0).then(|| correct[phase] as f64 / total[phase] as f64);
    (accuracy(0), accuracy(1))
}

fn set_value_target(projected: &mut ProjectedEpisode, target: f32) {
    projected.0.outcome.value_target = Some(target);
    for row in &mut projected.1 {
        row.value_target = Some(target);
    }
}

fn set_horizon_value_targets(
    projected: &mut ProjectedEpisode,
    artifact: &CompletedEpisodeArtifact,
    terminal_target: f32,
) -> Result<(), MeasurerError> {
    let targets = crate::horizon_value_targets(artifact, terminal_target)?;
    for (row, target) in projected.1.iter_mut().zip(targets) {
        row.horizon_value_targets = Some(target);
    }
    if let Some(row) = projected
        .1
        .last_mut()
        .filter(|row| row.kind == gz_replay::ReplayRowKind::Terminal)
    {
        row.horizon_value_targets = Some([terminal_target; 2]);
    }
    Ok(())
}

fn lane_summary(lanes: &mut Vec<MeasurerLaneSummary>, lane: usize) -> &mut MeasurerLaneSummary {
    while lanes.len() <= lane {
        let next = lanes.len();
        lanes.push(MeasurerLaneSummary {
            lane: next,
            ..MeasurerLaneSummary::default()
        });
    }
    &mut lanes[lane]
}

#[derive(Debug, Default)]
struct MeasureLedger {
    finals: u64,
    seen: HashSet<GraphHash>,
    distinct_by_version: BTreeMap<ModelVersion, u64>,
}

impl MeasureLedger {
    fn observe(&mut self, artifact: &CompletedEpisodeArtifact) {
        self.finals += 1;
        if !self.seen.insert(artifact.final_graph.graph.graph_hash) {
            return;
        }

        if let Some(version) = artifact.steps.last().and_then(|step| step.model_version) {
            *self.distinct_by_version.entry(version).or_insert(0) += 1;
        }
    }

    fn snapshot(self) -> MeasureLedgerSnapshot {
        let distinct_finals = self.seen.len() as u64;
        MeasureLedgerSnapshot {
            finals: self.finals,
            distinct_finals,
            repeat_finals: self.finals.saturating_sub(distinct_finals),
            distinct_by_version: self.distinct_by_version.into_iter().collect(),
        }
    }
}
