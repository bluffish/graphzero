#![forbid(unsafe_code)]

//! Search kernels and search result records for GraphZero.

pub use gz_engine::EngineIdentity;

mod beam;
mod episode;
mod greedy;
mod gumbel;
mod hash;
mod mcts;
mod puct;
mod sampling;
mod scratch;
mod support;
mod work;

pub use beam::{
    BeamEntrySummary, BeamEpisode, BeamLayer, BeamSearch, BeamSearchConfig, BeamStopReason,
};
pub use episode::{
    SearchAction, SearchCandidateSummary, SearchEpisode, SearchHandleBatch, SearchStep,
};
pub use greedy::{GreedyEpisode, GreedySearch, GreedySearchConfig, GreedyStopReason};
pub use gumbel::{
    GumbelEpisode, GumbelEpisodeContext, GumbelEpisodeTask, GumbelHandleBatch, GumbelMcts,
    GumbelMctsConfig, GumbelPlayer, GumbelRootResult, GumbelRootStats, GumbelRootTask,
    GumbelSearchContext, GumbelStep, GumbelStopReason, GumbelValueMode, SymmetricActorTrace,
    SymmetricEpisode, SymmetricRootAction, SymmetricRootResult, SymmetricSelfplayEpisodeTask,
    SymmetricSelfplayRootTask, considered_visit_sequence,
};
pub use hash::{
    beam_search_config_hash, greedy_search_config_hash, gumbel_search_config_hash,
    puct_search_config_hash, reducing_uniform_distill_config_hash,
    symmetric_selfplay_search_config_hash, trajectory_distill_config_hash,
};
pub use mcts::MctsHandleBatch;
pub use puct::{
    PuctEpisode, PuctEpisodeContext, PuctEpisodeTask, PuctHandleBatch, PuctMcts, PuctMctsConfig,
    PuctRootResult, PuctRootStats, PuctRootTask, PuctSearchContext, PuctStep, PuctStopReason,
};
pub use work::{
    ApplyWork, EvalOpponentWork, EvalWork, ExpandResult, ExpandWork, ExpandedCandidate,
    MeasurePairWork, MeasureWork, SearchPoll, SearchWork, SearchWorkResult, WorkToken,
};
