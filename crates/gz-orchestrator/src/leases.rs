use crate::{EpisodeId, internal};
use gz_engine::EngineResult;
use gz_eval_service::ModelGeneration;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

pub(super) struct ModelLeaseRegistry {
    state: Mutex<ModelLeaseState>,
    max_resident_generations: usize,
}

struct ModelLeaseState {
    current: ModelGeneration,
    generations: Vec<ModelGenerationState>,
    releasable: VecDeque<ModelGeneration>,
}

struct ModelGenerationState {
    model: ModelGeneration,
    users: usize,
}

struct ModelLease {
    registry: Arc<ModelLeaseRegistry>,
    model: ModelGeneration,
}

impl ModelLeaseRegistry {
    pub(super) fn new(
        current: ModelGeneration,
        max_resident_generations: usize,
    ) -> EngineResult<Self> {
        if current.id == 0 {
            return Err(internal("zero model generation"));
        }
        if max_resident_generations < 2 {
            return Err(internal("resident model generation capacity below two"));
        }
        Ok(Self {
            max_resident_generations,
            state: Mutex::new(ModelLeaseState {
                current,
                generations: vec![ModelGenerationState {
                    model: current,
                    users: 0,
                }],
                releasable: VecDeque::new(),
            }),
        })
    }

    fn acquire_current(self: &Arc<Self>) -> ModelLease {
        let mut state = self.state.lock().expect("model lease registry poisoned");
        let current = state.current;
        let generation = state
            .generations
            .iter_mut()
            .find(|generation| generation.model == current)
            .expect("current model generation is registered");
        generation.users = generation
            .users
            .checked_add(1)
            .expect("model lease count overflowed");
        ModelLease {
            registry: Arc::clone(self),
            model: current,
        }
    }

    pub(super) fn publish(&self, model: ModelGeneration) -> EngineResult<()> {
        if model.id == 0 {
            return Err(internal("zero model generation"));
        }
        let mut state = self.state.lock().expect("model lease registry poisoned");
        if state.current == model {
            return Ok(());
        }
        if state.generations.iter().any(|generation| {
            generation.model.id == model.id && generation.model.version != model.version
        }) {
            return Err(internal("model generation id changed version"));
        }
        if state.generations.iter().any(|generation| {
            generation.model.version == model.version && generation.model.id != model.id
        }) {
            return Err(internal("model version has multiple resident generations"));
        }
        if state
            .generations
            .iter()
            .all(|generation| generation.model != model)
        {
            if state.generations.len() >= self.max_resident_generations {
                return Err(internal("too many resident model generations"));
            }
            state
                .generations
                .push(ModelGenerationState { model, users: 0 });
        }
        let previous = state.current;
        state.current = model;
        if state
            .generations
            .iter()
            .any(|generation| generation.model == previous && generation.users == 0)
        {
            queue_model_release(&mut state, previous);
        }
        Ok(())
    }

    pub(super) fn take_releasable(&self) -> Vec<ModelGeneration> {
        let mut state = self.state.lock().expect("model lease registry poisoned");
        let models = state.releasable.drain(..).collect::<Vec<_>>();
        state
            .generations
            .retain(|generation| !models.contains(&generation.model));
        models
    }
}

impl Drop for ModelLease {
    fn drop(&mut self) {
        let mut state = self
            .registry
            .state
            .lock()
            .expect("model lease registry poisoned");
        let Some(generation) = state
            .generations
            .iter_mut()
            .find(|generation| generation.model == self.model)
        else {
            return;
        };
        assert!(generation.users > 0, "model lease count underflowed");
        generation.users -= 1;
        let became_unused = generation.users == 0;
        if became_unused && state.current != self.model {
            queue_model_release(&mut state, self.model);
        }
    }
}

fn queue_model_release(state: &mut ModelLeaseState, model: ModelGeneration) {
    if !state.releasable.contains(&model) {
        state.releasable.push_back(model);
    }
}

pub(super) struct EpisodeModelLeases {
    registry: Arc<ModelLeaseRegistry>,
    episodes: HashMap<EpisodeId, ModelLease>,
}

impl EpisodeModelLeases {
    pub(super) fn new(registry: Arc<ModelLeaseRegistry>) -> Self {
        Self {
            registry,
            episodes: HashMap::new(),
        }
    }

    pub(super) fn ensure(&mut self, episode_id: EpisodeId) -> EngineResult<ModelGeneration> {
        if let Some(lease) = self.episodes.get(&episode_id) {
            return Ok(lease.model);
        }
        let acquired = self.registry.acquire_current();
        let model = acquired.model;
        self.episodes.insert(episode_id, acquired);
        Ok(model)
    }

    pub(super) fn release(&mut self, episode_id: EpisodeId) {
        self.episodes.remove(&episode_id);
    }
}
